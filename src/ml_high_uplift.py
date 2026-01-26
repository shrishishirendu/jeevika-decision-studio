from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse

from src.encoding import encode_likert_5
from src.engagement_indices import (
    compute_affective_index,
    compute_behavioral_index,
    compute_cognitive_index,
    compute_total_engagement,
    minmax_norm,
)
from src.features import (
    compute_access_ease_avg,
    compute_awareness_flags_count,
    compute_barriers_count,
    compute_empowerment_score,
    compute_participation_count,
)


def make_target(df: pd.DataFrame, threshold: int = 20) -> pd.Series:
    income = pd.to_numeric(df.get("household_income_increase_percent"), errors="coerce")
    return (income >= threshold).astype("Int64")


def _clarity_score(df: pd.DataFrame) -> pd.Series:
    components = []
    if "scheme_clarity_level" in df.columns:
        components.append(minmax_norm(encode_likert_5(df["scheme_clarity_level"])))
    if "application_process_knowledge" in df.columns:
        components.append(minmax_norm(encode_likert_5(df["application_process_knowledge"])))
    if not components:
        return pd.Series([np.nan] * len(df), index=df.index)
    return pd.concat(components, axis=1).mean(axis=1, skipna=True)


def build_features(
    df: pd.DataFrame, return_meta: bool = False
) -> Tuple[pd.DataFrame, List[str] | None, List[str] | None]:
    # Returns X; optionally returns numeric_cols and categorical_cols for preprocessing.
    data = df.copy()

    data["behavioral_index"] = compute_behavioral_index(data)
    data["cognitive_index"] = compute_cognitive_index(data)
    data["affective_index"] = compute_affective_index(data)
    data["total_engagement_index"] = compute_total_engagement(data)

    data["participation_count"] = compute_participation_count(data)
    data["awareness_flags_count"] = compute_awareness_flags_count(data)
    data["access_ease_avg"] = compute_access_ease_avg(data)
    data["barriers_count"] = compute_barriers_count(data)
    data["empowerment_score"] = compute_empowerment_score(data)
    data["clarity_score"] = _clarity_score(data)

    awareness_col = "scheme_awareness_count"
    if awareness_col in data.columns:
        data[awareness_col] = pd.to_numeric(data[awareness_col], errors="coerce")
    else:
        data[awareness_col] = data["awareness_flags_count"]

    if "membership_duration_months" in data.columns:
        data["membership_duration_months"] = pd.to_numeric(
            data["membership_duration_months"], errors="coerce"
        )

    numeric_cols = [
        "behavioral_index",
        "cognitive_index",
        "affective_index",
        "total_engagement_index",
        "participation_count",
        awareness_col,
        "clarity_score",
        "access_ease_avg",
        "barriers_count",
        "empowerment_score",
        "membership_duration_months",
    ]
    numeric_cols = [col for col in numeric_cols if col in data.columns]

    categorical_cols = ["district"] if "district" in data.columns else []

    X = data[numeric_cols + categorical_cols].copy()
    if return_meta:
        return X, numeric_cols, categorical_cols
    return X, None, None


def make_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )


@dataclass
class ModelResults:
    models: Dict[str, object]
    metrics: Dict[str, Dict[str, float]]
    confusion: Dict[str, List[List[int]]]
    feature_importance: Dict[str, pd.DataFrame]
    used_rows: int


def _get_feature_names(model) -> List[str]:
    try:
        return model.named_steps["preprocess"].get_feature_names_out().tolist()
    except Exception:
        return []


def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> ModelResults:
    y_clean = pd.to_numeric(y, errors="coerce")
    mask = y_clean.notna()
    X = X.loc[mask]
    y_clean = y_clean.loc[mask].astype(int)

    preprocessor = make_preprocessor(numeric_cols, categorical_cols)

    models = {
        "LogisticRegression": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300, random_state=42, class_weight="balanced"
                    ),
                ),
            ]
        ),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )

    metrics = {}
    confusion = {}
    feature_importance = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        metrics[name] = {
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
        }
        confusion[name] = confusion_matrix(y_test, preds).tolist()

        if name == "LogisticRegression":
            coef_matrix = model.named_steps["model"].coef_
            if coef_matrix.ndim == 2:
                coef_display = np.mean(np.abs(coef_matrix), axis=0)
            else:
                coef_display = np.ravel(coef_matrix)
            feature_names = _get_feature_names(model)
            name_len = len(feature_names)
            coef_len = len(coef_display)
            if not feature_names:
                feature_names = [f"feature_{i}" for i in range(coef_len)]
            if name_len != coef_len:
                warnings.warn(
                    f"Feature name length ({name_len}) does not match coef length ({coef_len}); "
                    "truncating to min length."
                )
                min_len = min(name_len if name_len else coef_len, coef_len)
                feature_names = feature_names[:min_len]
                coef_display = coef_display[:min_len]
            fi = pd.DataFrame(
                {"feature": feature_names, "importance": coef_display}
            ).sort_values(by="importance", ascending=False)
            feature_importance[name] = fi
        else:
            perm = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
            fi = pd.DataFrame(
                {"feature": X_test.columns, "importance": perm.importances_mean}
            ).sort_values(by="importance", ascending=False)
            feature_importance[name] = fi

    return ModelResults(
        models=models,
        metrics=metrics,
        confusion=confusion,
        feature_importance=feature_importance,
        used_rows=int(X.shape[0]),
    )


def _top_drivers_logistic(model, X: pd.DataFrame, top_k: int = 3) -> List[List[str]]:
    feature_names = _get_feature_names(model)
    if not feature_names:
        return [[] for _ in range(len(X))]

    X_t = model.named_steps["preprocess"].transform(X)
    coef_matrix = model.named_steps["model"].coef_
    if coef_matrix.ndim == 2:
        coef_display = np.mean(np.abs(coef_matrix), axis=0)
    else:
        coef_display = np.ravel(coef_matrix)

    if len(feature_names) != len(coef_display):
        min_len = min(len(feature_names), len(coef_display))
        feature_names = feature_names[:min_len]
        coef_display = coef_display[:min_len]

    if sparse.issparse(X_t):
        X_t = X_t.tocsr()
    drivers = []
    for i in range(X_t.shape[0]):
        row = X_t[i]
        if sparse.issparse(row):
            row = row.toarray().ravel()
        contrib = row * coef_display
        top_idx = np.argsort(np.abs(contrib))[-top_k:][::-1]
        drivers.append([feature_names[idx] for idx in top_idx])
    return drivers


def _top_drivers_rf(model, X: pd.DataFrame, top_k: int = 3) -> List[List[str]]:
    feature_names = _get_feature_names(model)
    if not feature_names:
        return [[] for _ in range(len(X))]
    importances = model.named_steps["model"].feature_importances_
    min_len = min(len(feature_names), len(importances))
    feature_names = feature_names[:min_len]
    importances = importances[:min_len]
    top_idx = np.argsort(importances)[-top_k:][::-1]
    top_features = [feature_names[idx] for idx in top_idx]
    return [top_features for _ in range(len(X))]


def get_predictions_for_df(
    df: pd.DataFrame,
    model,
    threshold: float = 0.5,
    id_col: str = "respondent_id",
) -> pd.DataFrame:
    probs, preds = predict_with_model(df, model, threshold=threshold)

    if "model" in model.named_steps and isinstance(model.named_steps["model"], LogisticRegression):
        drivers = _top_drivers_logistic(model, X)
    else:
        drivers = _top_drivers_rf(model, X)

    if id_col in df.columns:
        ids = df[id_col]
    else:
        ids = df.index.astype(str)

    return pd.DataFrame(
        {
            "id": ids.values,
            "predicted_class": preds,
            "probability": probs,
            "top_drivers": drivers,
        },
        index=df.index,
    )


def predict_proba(df: pd.DataFrame, model) -> pd.Series:
    X, numeric_cols, categorical_cols = build_features(df, return_meta=True)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        preprocessor = make_preprocessor(numeric_cols, categorical_cols)
        X_transformed = preprocessor.fit_transform(X)
        proba = model.named_steps["model"].predict_proba(X_transformed)[:, 1]
    return pd.Series(proba, index=df.index, name="predicted_prob")


def predict_with_model(
    df: pd.DataFrame, model, threshold: float = 0.5
) -> Tuple[pd.Series, pd.Series]:
    X, _, _ = build_features(df, return_meta=True)
    proba_vec = model.predict_proba(X)[:, 1]
    if len(proba_vec) != len(df):
        raise ValueError("Prediction length mismatch with input data.")
    pred_vec = (proba_vec >= threshold).astype(int)
    return (
        pd.Series(proba_vec, index=df.index, name="predicted_prob"),
        pd.Series(pred_vec, index=df.index, name="predicted_class"),
    )
