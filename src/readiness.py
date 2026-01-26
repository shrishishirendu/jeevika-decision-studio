from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.encoding import encode_likert_5
from src.engagement_indices import (
    compute_affective_index,
    compute_behavioral_index,
    compute_cognitive_index,
    compute_total_engagement,
)
from src.features import compute_access_ease_avg, compute_awareness_flags_count, compute_barriers_count


def safe_quantiles(series: pd.Series) -> Dict[str, float | None]:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return {"q25": None, "q50": None, "q75": None}
    if values.min() == values.max():
        return {"q25": None, "q50": None, "q75": None}
    return {
        "q25": float(values.quantile(0.25)),
        "q50": float(values.quantile(0.5)),
        "q75": float(values.quantile(0.75)),
    }


def minmax_norm(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series([np.nan] * len(series), index=series.index)
    min_val = values.min()
    max_val = values.max()
    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        return pd.Series([np.nan] * len(series), index=series.index)
    return (values - min_val) / (max_val - min_val)


def compute_engagement_readiness(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    if "total_engagement_index" in df.columns:
        tei = pd.to_numeric(df["total_engagement_index"], errors="coerce")
    else:
        bei = compute_behavioral_index(df)
        cei = compute_cognitive_index(df)
        aei = compute_affective_index(df)
        tei = pd.concat([bei, cei, aei], axis=1).mean(axis=1, skipna=True)

    quant = safe_quantiles(tei)
    q25, q75 = quant["q25"], quant["q75"]
    readiness = pd.Series(["Unknown"] * len(df), index=df.index)

    if q25 is not None and q75 is not None:
        readiness = pd.Series(
            np.where(
                tei >= q75,
                "High",
                np.where(tei <= q25, "Low", "Medium"),
            ),
            index=df.index,
        )

    score_map = {"High": 1.0, "Medium": 0.5, "Low": 0.0}
    readiness_score = readiness.map(score_map)
    readiness.name = "engagement_readiness"
    readiness_score.name = "engagement_readiness_score"
    return readiness, readiness_score


def compute_awareness_readiness(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    if "scheme_awareness_count" in df.columns:
        awareness = pd.to_numeric(df["scheme_awareness_count"], errors="coerce")
    else:
        awareness = compute_awareness_flags_count(df)

    clarity_components = []
    if "scheme_clarity_level" in df.columns:
        clarity_components.append(minmax_norm(encode_likert_5(df["scheme_clarity_level"])))
    if "application_process_knowledge" in df.columns:
        clarity_components.append(minmax_norm(encode_likert_5(df["application_process_knowledge"])))

    clarity_score = None
    if clarity_components:
        clarity_score = pd.concat(clarity_components, axis=1).mean(axis=1, skipna=True)

    awareness_norm = minmax_norm(awareness)

    aw_med = awareness_norm.median(skipna=True)
    cl_med = clarity_score.median(skipna=True) if clarity_score is not None else None

    readiness = pd.Series(["Unknown"] * len(df), index=df.index)

    if awareness_norm.notna().any() or (clarity_score is not None and clarity_score.notna().any()):
        awareness_high = awareness_norm >= aw_med if not np.isnan(aw_med) else pd.Series(False, index=df.index)
        clarity_high = (
            clarity_score >= cl_med if clarity_score is not None and cl_med is not None else pd.Series(False, index=df.index)
        )
        readiness = pd.Series(
            np.where(
                awareness_high & clarity_high,
                "High",
                np.where(awareness_high ^ clarity_high, "Medium", "Low"),
            ),
            index=df.index,
        )

    score_map = {"High": 1.0, "Medium": 0.5, "Low": 0.0}
    readiness_score = readiness.map(score_map)
    readiness.name = "awareness_readiness"
    readiness_score.name = "awareness_readiness_score"
    return readiness, readiness_score


def compute_access_readiness(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    access = compute_access_ease_avg(df)
    barriers = compute_barriers_count(df)

    access_q = safe_quantiles(access)
    barriers_q = safe_quantiles(barriers)

    q25_access, q75_access = access_q["q25"], access_q["q75"]
    q25_bar, q75_bar = barriers_q["q25"], barriers_q["q75"]

    readiness = pd.Series(["Unknown"] * len(df), index=df.index)

    if q25_access is not None or q25_bar is not None:
        access_high = access >= q75_access if q75_access is not None else pd.Series(False, index=df.index)
        access_low = access <= q25_access if q25_access is not None else pd.Series(False, index=df.index)
        bar_high = barriers >= q75_bar if q75_bar is not None else pd.Series(False, index=df.index)
        bar_low = barriers <= q25_bar if q25_bar is not None else pd.Series(False, index=df.index)

        readiness = pd.Series(
            np.where(
                (access_high & ~bar_high) | (bar_low & ~access_low),
                "High",
                np.where(access_low | bar_high, "Low", "Medium"),
            ),
            index=df.index,
        )

    score_map = {"High": 1.0, "Medium": 0.5, "Low": 0.0}
    readiness_score = readiness.map(score_map)
    readiness.name = "access_readiness"
    readiness_score.name = "access_readiness_score"
    return readiness, readiness_score


def assign_decision_profile(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    e, _ = compute_engagement_readiness(df)
    a, _ = compute_awareness_readiness(df)
    x, _ = compute_access_readiness(df)

    profile = pd.Series(["Mainstream / Monitor"] * len(df), index=df.index)
    action = pd.Series(["Maintain engagement and monitor"] * len(df), index=df.index)

    ready = (e == "High") & (a == "High") & (x == "High")
    profile.loc[ready] = "Ready-to-Scale"
    action.loc[ready] = "Scale services and expand offerings"

    aware_blocked = (a == "High") & (e.isin(["High", "Medium"])) & (x == "Low")
    profile.loc[aware_blocked] = "Aware-but-Blocked"
    action.loc[aware_blocked] = "Remove access barriers, streamline processes"

    motivated_unaware = (e == "High") & (a == "Low")
    profile.loc[motivated_unaware] = "Motivated-but-Unaware"
    action.loc[motivated_unaware] = "Targeted IEC and outreach"

    informed_passive = (a == "High") & (x == "High") & (e == "Low")
    profile.loc[informed_passive] = "Informed-but-Passive"
    action.loc[informed_passive] = "Activation and behavior nudges"

    disengaged = (e == "Low") & (a == "Low")
    profile.loc[disengaged] = "Disengaged / Intensive Support"
    action.loc[disengaged] = "Trust building and intensive support"

    profile.name = "decision_profile"
    action.name = "decision_action"
    return profile, action


def build_readiness_frame(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["behavioral_index"] = compute_behavioral_index(data)
    data["cognitive_index"] = compute_cognitive_index(data)
    data["affective_index"] = compute_affective_index(data)
    data["total_engagement_index"] = compute_total_engagement(data)

    e, e_score = compute_engagement_readiness(data)
    a, a_score = compute_awareness_readiness(data)
    x, x_score = compute_access_readiness(data)

    profile, action = assign_decision_profile(data)

    data["engagement_readiness"] = e
    data["awareness_readiness"] = a
    data["access_readiness"] = x
    data["engagement_readiness_score"] = e_score
    data["awareness_readiness_score"] = a_score
    data["access_readiness_score"] = x_score
    data["decision_profile"] = profile
    data["decision_action"] = action

    return data