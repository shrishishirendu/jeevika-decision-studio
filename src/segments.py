from __future__ import annotations

import numpy as np
import pandas as pd

from src.encoding import encode_likert_5
from src.engagement_indices import (
    compute_affective_index,
    compute_behavioral_index,
    compute_cognitive_index,
    compute_total_engagement,
)


def assign_population_segments(df: pd.DataFrame) -> pd.Series:
    data = df.copy()
    data["behavioral_index"] = compute_behavioral_index(data)
    data["cognitive_index"] = compute_cognitive_index(data)
    data["affective_index"] = compute_affective_index(data)
    data["total_engagement_index"] = compute_total_engagement(data)

    bei = pd.to_numeric(data["behavioral_index"], errors="coerce")
    cei = pd.to_numeric(data["cognitive_index"], errors="coerce")
    aei = pd.to_numeric(data["affective_index"], errors="coerce")

    bei_q25, bei_q75 = bei.quantile(0.25), bei.quantile(0.75)
    cei_q25, cei_q75 = cei.quantile(0.25), cei.quantile(0.75)
    aei_q25, aei_q75 = aei.quantile(0.25), aei.quantile(0.75)

    bei_high = bei >= bei_q75
    bei_low = bei <= bei_q25
    cei_high = cei >= cei_q75
    cei_low = cei <= cei_q25
    aei_high = aei >= aei_q75
    aei_low = aei <= aei_q25

    segments = pd.Series(["Mainstream"] * len(data), index=data.index, name="segment_label")

    segments.loc[bei_low & cei_low & aei_low] = "Disengaged"
    segments.loc[bei_high & cei_high] = "Champions"
    segments.loc[bei_low & cei_high] = "Aspirers (High Priority)"
    segments.loc[bei_high & cei_low] = "Busy but Blind"
    segments.loc[aei_high & bei_low] = "Motivated but Blocked"

    return segments


def summarize_segments(df: pd.DataFrame, segment_col: str = "segment_label") -> pd.DataFrame:
    data = df.copy()
    if "behavioral_index" not in data.columns:
        data["behavioral_index"] = compute_behavioral_index(data)
    if "cognitive_index" not in data.columns:
        data["cognitive_index"] = compute_cognitive_index(data)
    if "affective_index" not in data.columns:
        data["affective_index"] = compute_affective_index(data)
    if "total_engagement_index" not in data.columns:
        data["total_engagement_index"] = compute_total_engagement(data)

    if segment_col not in data.columns:
        data[segment_col] = assign_population_segments(data)

    data["household_income_increase_percent"] = pd.to_numeric(
        data.get("household_income_increase_percent"), errors="coerce"
    )
    data["overall_impact_rating"] = (
        encode_likert_5(data["overall_impact_rating"])
        if "overall_impact_rating" in data.columns
        else np.nan
    )
    data["satisfaction_level"] = (
        encode_likert_5(data["satisfaction_level"])
        if "satisfaction_level" in data.columns
        else np.nan
    )
    data["future_program_confidence"] = (
        encode_likert_5(data["future_program_confidence"])
        if "future_program_confidence" in data.columns
        else np.nan
    )
    data["referral_likelihood"] = pd.to_numeric(
        data.get("referral_likelihood"), errors="coerce"
    )

    summary = (
        data.groupby(segment_col)
        .agg(
            count=(segment_col, "size"),
            mean_bei=("behavioral_index", "mean"),
            mean_cei=("cognitive_index", "mean"),
            mean_aei=("affective_index", "mean"),
            mean_tei=("total_engagement_index", "mean"),
            mean_income_uplift=("household_income_increase_percent", "mean"),
            mean_impact=("overall_impact_rating", "mean"),
            mean_satisfaction=("satisfaction_level", "mean"),
            mean_confidence=("future_program_confidence", "mean"),
            mean_referral=("referral_likelihood", "mean"),
        )
        .reset_index()
    )

    total = summary["count"].sum()
    summary["pct"] = (summary["count"] / total * 100).round(2)
    summary = summary.sort_values(by="count", ascending=False)
    return summary


def segment_quality_checks(summary_df: pd.DataFrame) -> dict:
    num_segments = int(summary_df.shape[0])
    largest_pct = float(summary_df["pct"].max()) if not summary_df.empty else 0.0
    smallest_pct = float(summary_df["pct"].min()) if not summary_df.empty else 0.0

    warnings = []
    if num_segments < 4:
        warnings.append("Fewer than 4 segments detected.")
    if largest_pct > 75:
        warnings.append("Largest segment exceeds 75% of population.")

    return {
        "num_segments": num_segments,
        "largest_segment_pct": largest_pct,
        "smallest_segment_pct": smallest_pct,
        "warnings": warnings,
    }