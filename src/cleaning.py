from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.schema import SurveySchema

_NULL_MARKERS = ["", "NA", "N/A", "null", "None", "-", " "]


def clean_survey_df(df: pd.DataFrame, schema: SurveySchema) -> Tuple[pd.DataFrame, Dict]:
    cleaned = df.copy()

    if schema.rename_map:
        rename_map = {k: v for k, v in schema.rename_map.items() if k in cleaned.columns}
        if rename_map:
            cleaned = cleaned.rename(columns=rename_map)

    object_cols = [
        col
        for col in cleaned.columns
        if pd.api.types.is_object_dtype(cleaned[col])
        or pd.api.types.is_string_dtype(cleaned[col])
    ]
    for col in object_cols:
        cleaned[col] = cleaned[col].astype("string").str.strip()
        cleaned[col] = cleaned[col].replace(_NULL_MARKERS, np.nan)

    if "meeting_attendance" in cleaned.columns:
        cleaned["meeting_attendance"] = _normalize_meeting_attendance(
            cleaned["meeting_attendance"]
        )

    if "involvement_level" in cleaned.columns:
        cleaned["involvement_level"] = _normalize_involvement_level(
            cleaned["involvement_level"]
        )

    for col in schema.numerics:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    flag_counts = {}
    if schema.geography_col and schema.geography_col in cleaned.columns:
        flag_col = "flag_missing_geography"
        cleaned[flag_col] = cleaned[schema.geography_col].isna()
        flag_counts[flag_col] = int(cleaned[flag_col].sum())

    income_col = "household_income_increase_percent"
    if income_col in cleaned.columns:
        flag_col = "flag_missing_income_change"
        cleaned[flag_col] = cleaned[income_col].isna()
        flag_counts[flag_col] = int(cleaned[flag_col].sum())

    row_count = len(cleaned)
    col_count = len(cleaned.columns)
    missing_counts = cleaned.isna().sum()
    missing_pct = (missing_counts / max(row_count, 1) * 100).round(2)
    missingness = [
        {
            "col": col,
            "missing_count": int(missing_counts[col]),
            "missing_pct": float(missing_pct[col]),
        }
        for col in cleaned.columns
    ]

    dtypes = [{"col": col, "dtype": str(cleaned[col].dtype)} for col in cleaned.columns]

    profile = {
        "row_count": row_count,
        "col_count": col_count,
        "missingness": missingness,
        "dtypes": dtypes,
        "flag_counts": flag_counts,
    }

    return cleaned, profile


def _normalize_meeting_attendance(series: pd.Series) -> pd.Series:
    lowered = series.astype("string").str.strip().str.lower()
    normalized = lowered.copy()

    numeric = pd.to_numeric(lowered, errors="coerce")
    if numeric.notna().sum() > 0 and numeric.notna().sum() >= len(series) * 0.3:
        return numeric

    normalized = normalized.mask(lowered.str.contains("don't attend|do not", na=False), "none")
    normalized = normalized.mask(lowered.str.contains("rarely", na=False), "rarely")
    normalized = normalized.mask(lowered.str.contains("some|1", na=False), "some")
    normalized = normalized.mask(lowered.str.contains("most|2", na=False), "most")
    normalized = normalized.mask(lowered.str.contains("all|3-4", na=False), "all")

    return normalized


def _normalize_involvement_level(series: pd.Series) -> pd.Series:
    lowered = series.astype("string").str.strip().str.lower()
    normalized = lowered.copy()

    normalized = normalized.mask(lowered.str.contains("very actively", na=False), "very_active")
    normalized = normalized.mask(lowered.str.contains("actively involved|active", na=False), "active")
    normalized = normalized.mask(lowered.str.contains("moderately|moderate", na=False), "moderate")
    normalized = normalized.mask(lowered.str.contains("passively|passive", na=False), "passive")
    normalized = normalized.mask(lowered.str.contains("not involved|not_involved", na=False), "not_involved")

    return normalized
