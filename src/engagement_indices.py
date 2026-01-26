from __future__ import annotations

import numpy as np
import pandas as pd

from src.encoding import encode_likert_5, encode_referral
from src.features import compute_awareness_flags_count, compute_participation_count


def minmax_norm(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series([np.nan] * len(series), index=series.index)
    min_val = values.min()
    max_val = values.max()
    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        return pd.Series([np.nan] * len(series), index=series.index)
    return (values - min_val) / (max_val - min_val)


def compute_behavioral_index(df: pd.DataFrame) -> pd.Series:
    attendance_col = "meeting_attendance"
    if attendance_col not in df.columns and "shg_meetings_attended_monthly" in df.columns:
        attendance_col = "shg_meetings_attended_monthly"

    if attendance_col in df.columns:
        attendance = pd.to_numeric(df[attendance_col], errors="coerce")
        attendance_norm = attendance.clip(lower=0, upper=4) / 4
    else:
        attendance_norm = pd.Series([np.nan] * len(df), index=df.index)

    participation = compute_participation_count(df)
    participation_norm = (participation / 8.0).clip(upper=1.0)

    bei = 0.5 * attendance_norm + 0.5 * participation_norm
    bei.name = "behavioral_index"
    return bei


def compute_cognitive_index(df: pd.DataFrame) -> pd.Series:
    components = []

    if "scheme_awareness_count" in df.columns:
        components.append(minmax_norm(df["scheme_awareness_count"]))

    awareness_flags = compute_awareness_flags_count(df)
    if awareness_flags.notna().any():
        components.append(minmax_norm(awareness_flags))

    if "scheme_clarity_level" in df.columns:
        clarity = encode_likert_5(df["scheme_clarity_level"])
        if clarity.notna().any():
            components.append(minmax_norm(clarity))

    if "application_process_knowledge" in df.columns:
        knowledge = encode_likert_5(df["application_process_knowledge"])
        if knowledge.notna().any():
            components.append(minmax_norm(knowledge))

    if not components:
        return pd.Series([np.nan] * len(df), index=df.index, name="cognitive_index")

    stacked = pd.concat(components, axis=1)
    cei = stacked.mean(axis=1, skipna=True)
    cei.name = "cognitive_index"
    return cei


def compute_affective_index(df: pd.DataFrame) -> pd.Series:
    components = []

    if "satisfaction_level" in df.columns:
        satisfaction = encode_likert_5(df["satisfaction_level"])
        components.append((satisfaction - 1) / 4)

    if "future_program_confidence" in df.columns:
        confidence = encode_likert_5(df["future_program_confidence"])
        components.append((confidence - 1) / 4)

    if "referral_likelihood" in df.columns:
        referral = encode_referral(df["referral_likelihood"])
        components.append(referral / 10)

    if not components:
        return pd.Series([np.nan] * len(df), index=df.index, name="affective_index")

    stacked = pd.concat(components, axis=1)
    aei = stacked.mean(axis=1, skipna=True)
    aei.name = "affective_index"
    return aei


def compute_total_engagement(df: pd.DataFrame) -> pd.Series:
    bei = compute_behavioral_index(df)
    cei = compute_cognitive_index(df)
    aei = compute_affective_index(df)

    stacked = pd.concat([bei, cei, aei], axis=1)
    tei = stacked.mean(axis=1, skipna=True)
    tei.name = "total_engagement_index"
    return tei
