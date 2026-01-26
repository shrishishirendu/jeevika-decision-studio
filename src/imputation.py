from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def infer_column_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "binary"

    if pd.api.types.is_numeric_dtype(series):
        unique_values = series.dropna().unique()
        if len(unique_values) <= 2:
            return "binary"
        return "numeric"

    values = series.dropna().astype(str).str.strip().str.lower()
    unique_values = values.unique()
    if len(unique_values) <= 2:
        return "binary"

    if _looks_ordinal(values):
        return "ordinal"

    return "categorical"


def compute_missing_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        series = df[col]
        missing_pct = float(series.isna().mean() * 100)
        rows.append(
            {
                "column_name": col,
                "inferred_type": infer_column_type(series),
                "missing_pct": round(missing_pct, 2),
                "non_null_count": int(series.notna().sum()),
                "unique_count": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values(by="missing_pct", ascending=False)


def impute_dataframe(df: pd.DataFrame, config: Dict | None = None) -> Tuple[pd.DataFrame, Dict]:
    data = df.copy()
    report = {
        "total_cells": int(data.shape[0] * data.shape[1]),
        "total_imputed_cells": 0,
        "per_column_imputed_count": {},
        "per_column_strategy": {},
    }

    profile = compute_missing_profile(data)

    for _, row in profile.iterrows():
        col = row["column_name"]
        missing_pct = row["missing_pct"]
        inferred_type = row["inferred_type"]
        missing_mask = data[col].isna()
        missing_count = int(missing_mask.sum())

        if missing_count == 0:
            continue

        strategy = "none"
        if inferred_type == "numeric":
            if missing_pct <= 20:
                fill_value = pd.to_numeric(data[col], errors="coerce").mean()
                data[col] = data[col].fillna(fill_value)
                strategy = "mean"
        elif inferred_type == "categorical":
            if missing_pct <= 20:
                mode_value = _safe_mode(data[col])
                data[col] = data[col].fillna(mode_value)
                strategy = "mode"
            else:
                data[col] = data[col].fillna("Unknown")
                strategy = "unknown"
        elif inferred_type == "binary":
            if missing_pct <= 20:
                mode_value = _safe_mode(data[col])
                data[col] = data[col].fillna(mode_value)
                strategy = "mode"
        elif inferred_type == "ordinal":
            if missing_pct <= 20:
                mode_value = _safe_mode(data[col])
                data[col] = data[col].fillna(mode_value)
                strategy = "mode"
            else:
                data[col] = data[col].fillna("Unknown")
                strategy = "unknown"

        if strategy != "none":
            imputed_mask = missing_mask & data[col].notna()
            data[f"{col}__is_imputed"] = imputed_mask
            imputed_count = int(imputed_mask.sum())
            report["total_imputed_cells"] += imputed_count
            report["per_column_imputed_count"][col] = imputed_count
            report["per_column_strategy"][col] = strategy

    return data, report


def _safe_mode(series: pd.Series):
    mode_series = series.dropna().mode()
    if not mode_series.empty:
        return mode_series.iloc[0]
    return np.nan


def _looks_ordinal(values: pd.Series) -> bool:
    keywords = ["low", "medium", "high", "very", "poor", "fair", "good", "excellent"]
    return any(any(word in val for word in keywords) for val in values.tolist())