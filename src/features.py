from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

_ATTENDANCE_MAP = {
    "none": 0.0,
    "rarely": 1.0,
    "some": 2.0,
    "most": 3.0,
    "all": 4.0,
    "yes": 1.0,
    "no": 0.0,
    "low": 1.0,
    "medium": 2.0,
    "high": 3.0,
    "very high": 4.0,
}

_INVOLVEMENT_MAP = {
    "not_involved": 0.0,
    "passive": 1.0,
    "moderate": 3.0,
    "active": 4.0,
    "very_active": 4.0,
    "yes": 1.0,
    "no": 0.0,
    "low": 1.0,
    "medium": 2.0,
    "high": 3.0,
    "very high": 4.0,
}


def compute_engagement_index(df: pd.DataFrame) -> pd.Series:
    """Compute engagement index using meeting attendance and involvement level."""
    if "meeting_attendance" not in df.columns or "involvement_level" not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index, name="engagement_index")

    att = pd.to_numeric(df["meeting_attendance"], errors="coerce")
    attendance_score = att.clip(lower=0, upper=4).fillna(0)

    inv = df["involvement_level"].astype(str).str.strip().str.lower()
    involvement_map = {"passive": 1.0, "moderate": 3.0, "active": 4.0}
    involvement_score = inv.map(involvement_map).fillna(0)

    index_series = attendance_score + involvement_score
    index_series.name = "engagement_index"
    return index_series


def to_binary_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)

    if pd.api.types.is_numeric_dtype(series):
        coerced = pd.to_numeric(series, errors="coerce")
        return coerced.where(coerced.isin([0, 1]))

    lower = series.astype("string").str.strip().str.lower()
    mapping = {
        "yes": 1.0,
        "y": 1.0,
        "true": 1.0,
        "1": 1.0,
        "no": 0.0,
        "n": 0.0,
        "false": 0.0,
        "0": 0.0,
    }
    mapped = lower.map(mapping)
    return pd.to_numeric(mapped, errors="coerce")


def compute_kpis(df: pd.DataFrame) -> Dict[str, float | int]:
    kpis: Dict[str, float | int] = {"rows": int(len(df))}

    if "membership_duration" in df.columns and pd.api.types.is_numeric_dtype(
        df["membership_duration"]
    ):
        kpis["avg_membership_duration"] = float(df["membership_duration"].mean())

    if "household_income_increase_percent" in df.columns and pd.api.types.is_numeric_dtype(
        df["household_income_increase_percent"]
    ):
        kpis["avg_income_increase"] = float(
            df["household_income_increase_percent"].mean()
        )

    engagement_index = compute_engagement_index(df)
    if engagement_index.notna().any():
        kpis["avg_engagement_index"] = float(engagement_index.mean())

    return kpis


def to_binary_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    data = {}
    for col in cols:
        if col not in df.columns:
            continue
        data[col] = to_binary_series(df[col])
    if not data:
        return pd.DataFrame(index=df.index)
    return pd.DataFrame(data, index=df.index)


def compute_participation_count(df: pd.DataFrame) -> pd.Series:
    cols = [
        "savings_participation",
        "credit_participation",
        "agricultural_intervention",
        "livestock_activity",
        "nonfarm_enterprise",
        "didi_ki_rasoi",
        "producer_groups",
        "vo_participation",
    ]
    existing = [col for col in cols if col in df.columns]
    if not existing:
        return pd.Series([0] * len(df), index=df.index, name="participation_count")
    bin_df = to_binary_df(df, existing)
    return bin_df.fillna(0).sum(axis=1).rename("participation_count")


def compute_barriers_count(df: pd.DataFrame) -> pd.Series:
    cols = [col for col in df.columns if col.startswith("barrier_")]
    if not cols:
        return pd.Series([0] * len(df), index=df.index, name="barriers_count")
    bin_df = to_binary_df(df, cols)
    return bin_df.fillna(0).sum(axis=1).rename("barriers_count")


def compute_awareness_flags_count(df: pd.DataFrame) -> pd.Series:
    cols = [col for col in df.columns if col.endswith("_awareness")]
    if not cols:
        return pd.Series([0] * len(df), index=df.index, name="awareness_flags_count")
    bin_df = to_binary_df(df, cols)
    return bin_df.fillna(0).sum(axis=1).rename("awareness_flags_count")


def compute_access_ease_avg(df: pd.DataFrame) -> pd.Series:
    access_cols = [col for col in df.columns if col.endswith("_access_ease")]
    if not access_cols:
        return pd.Series([np.nan] * len(df), index=df.index, name="access_ease_avg")
    values = df[access_cols].apply(pd.to_numeric, errors="coerce")
    return values.mean(axis=1).rename("access_ease_avg")


def compute_empowerment_score(df: pd.DataFrame) -> pd.Series:
    cols = [col for col in df.columns if col.startswith("non_financial_")]
    if not cols:
        return pd.Series([np.nan] * len(df), index=df.index, name="empowerment_score")
    values = df[cols].apply(pd.to_numeric, errors="coerce")
    return values.mean(axis=1).rename("empowerment_score")


def _coerce_series(series: pd.Series, mapping: Dict[str, float]) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    lowered = series.astype("string").str.strip().str.lower()
    mapped = lowered.map(mapping)
    return pd.to_numeric(mapped, errors="coerce")
