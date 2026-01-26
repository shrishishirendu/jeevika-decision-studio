from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src import features as features


@dataclass
class ClusteringResult:
    cluster_id: pd.Series
    cluster_profile: pd.DataFrame
    chosen_k: int
    silhouette_scores: pd.DataFrame


def run_clustering(df: pd.DataFrame, k_min: int = 4, k_max: int = 6, random_state: int = 42) -> ClusteringResult:
    data = df.copy()

    compute_engagement_index = getattr(features, "compute_engagement_index", None)
    compute_participation_count = getattr(features, "compute_participation_count", None)
    compute_barriers_count = getattr(features, "compute_barriers_count", None)
    compute_access_ease_avg = getattr(features, "compute_access_ease_avg", None)
    compute_empowerment_score = getattr(features, "compute_empowerment_score", None)

    if compute_engagement_index and "engagement_index" not in data.columns:
        data["engagement_index"] = compute_engagement_index(data)
    if compute_participation_count and "participation_count" not in data.columns:
        data["participation_count"] = compute_participation_count(data)
    if compute_barriers_count and "barriers_count" not in data.columns:
        data["barriers_count"] = compute_barriers_count(data)
    if compute_access_ease_avg and "access_ease_avg" not in data.columns:
        data["access_ease_avg"] = compute_access_ease_avg(data)
    if compute_empowerment_score and "empowerment_score" not in data.columns:
        data["empowerment_score"] = compute_empowerment_score(data)

    numeric_cols = [
        "engagement_index",
        "participation_count",
        "barriers_count",
        "scheme_awareness_count",
        "access_ease_avg",
        "empowerment_score",
        "household_income_increase_percent",
    ]
    numeric_cols = [col for col in numeric_cols if col in data.columns]

    if not numeric_cols:
        raise ValueError("No numeric features available for clustering.")

    numeric_df = data[numeric_cols].apply(pd.to_numeric, errors="coerce")

    cat_cols = []
    if "engagement_level" in data.columns:
        cat_cols.append("engagement_level")

    if cat_cols:
        cat_df = pd.get_dummies(data[cat_cols].astype("string"), dummy_na=True)
        feature_df = pd.concat([numeric_df, cat_df], axis=1)
    else:
        feature_df = numeric_df

    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))

    scaler = StandardScaler()
    scaled_numeric = scaler.fit_transform(feature_df[numeric_df.columns])

    if cat_cols:
        cat_values = feature_df.drop(columns=numeric_df.columns).to_numpy()
        X = np.hstack([scaled_numeric, cat_values])
    else:
        X = scaled_numeric

    silhouette_rows = []
    best_k = None
    best_score = -1.0

    max_k = min(k_max, len(feature_df) - 1)
    min_k = min(k_min, max_k)
    if max_k < 2:
        raise ValueError("Not enough rows for clustering.")

    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels) if k > 1 else np.nan
        silhouette_rows.append({"k": k, "silhouette": round(float(score), 4)})
        if score > best_score:
            best_score = score
            best_k = k

    if best_k is None:
        best_k = min_k

    final_model = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    final_labels = final_model.fit_predict(X)

    cluster_id = pd.Series(final_labels, index=data.index, name="cluster_id")
    cluster_profile = (
        data.assign(cluster_id=cluster_id)
        .groupby("cluster_id")[numeric_cols]
        .mean()
        .round(2)
        .reset_index()
    )

    silhouette_df = pd.DataFrame(silhouette_rows)

    return ClusteringResult(
        cluster_id=cluster_id,
        cluster_profile=cluster_profile,
        chosen_k=int(best_k),
        silhouette_scores=silhouette_df,
    )