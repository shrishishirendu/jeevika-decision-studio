from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.ui_filters import apply_filters, load_clean_df
from src.encoding import coerce_numeric, encode_likert_5, encode_referral
from src.engagement_indices import (
    compute_affective_index,
    compute_behavioral_index,
    compute_cognitive_index,
    compute_total_engagement,
)
from src.features import (
    compute_awareness_flags_count,
    compute_empowerment_score,
    compute_participation_count,
)


def safe_float(value):
    import pandas as pd

    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except TypeError:
        return None


def compute_district_correlations(df: pd.DataFrame, min_n: int = 50) -> pd.DataFrame:
    if "district" not in df.columns:
        return pd.DataFrame()
    if "household_income_increase_percent" not in df.columns:
        return pd.DataFrame()

    data = df.copy()
    data["behavioral_index"] = pd.to_numeric(data["behavioral_index"], errors="coerce")
    data["cognitive_index"] = pd.to_numeric(data["cognitive_index"], errors="coerce")
    data["affective_index"] = pd.to_numeric(data["affective_index"], errors="coerce")
    data["income"] = pd.to_numeric(data["household_income_increase_percent"], errors="coerce")

    rows = []
    for district, subset in data.groupby("district"):
        if subset.shape[0] < min_n:
            continue
        if subset["income"].dropna().empty:
            continue
        rho_bei = subset[["behavioral_index", "income"]].corr(method="spearman").iloc[0, 1]
        rho_cei = subset[["cognitive_index", "income"]].corr(method="spearman").iloc[0, 1]
        rho_aei = subset[["affective_index", "income"]].corr(method="spearman").iloc[0, 1]
        rows.append(
            {
                "district": district,
                "n": int(subset.shape[0]),
                "rho_BEI": safe_float(rho_bei),
                "rho_CEI": safe_float(rho_cei),
                "rho_AEI": safe_float(rho_aei),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(by="n", ascending=False)

CONFIG_PATH = ROOT / "config" / "config.yaml"

st.set_page_config(page_title="Engagement", page_icon="Engagement", layout="wide")

st.title("Engagement")

if not CONFIG_PATH.exists():
    st.error("Missing config/config.yaml. Please add it to continue.")
    st.stop()

with CONFIG_PATH.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

data_config = config.get("data") or {}
processed_path = data_config.get("processed_path")
raw_path = data_config.get("raw_path")
if not processed_path:
    st.error("config.yaml is missing data.processed_path")
    st.stop()

parquet_path = ROOT / processed_path
csv_path = ROOT / raw_path if raw_path else None

try:
    df = load_clean_df(str(parquet_path), str(csv_path) if csv_path else None)
except Exception as exc:  # pragma: no cover
    st.error(f"Failed to load cleaned dataset: {exc}")
    st.stop()

if not parquet_path.exists():
    st.warning("Processed parquet not found. Using cleaned raw CSV in memory.")

filtered_df, selections = apply_filters(df)
filtered_df = filtered_df.copy()

st.sidebar.markdown("### Active Filters")
for key, value in selections.items():
    st.sidebar.write(f"{key}: {value}")

if "meeting_attendance" not in filtered_df.columns and "shg_meetings_attended_monthly" in filtered_df.columns:
    filtered_df["meeting_attendance"] = filtered_df["shg_meetings_attended_monthly"]

if "involvement_level" not in filtered_df.columns and "engagement_level" in filtered_df.columns:
    filtered_df["involvement_level"] = filtered_df["engagement_level"]

filtered_df["participation_count"] = compute_participation_count(filtered_df)
filtered_df["awareness_flags_count"] = compute_awareness_flags_count(filtered_df)
filtered_df["empowerment_score"] = compute_empowerment_score(filtered_df)

filtered_df["behavioral_index"] = compute_behavioral_index(filtered_df)
filtered_df["cognitive_index"] = compute_cognitive_index(filtered_df)
filtered_df["affective_index"] = compute_affective_index(filtered_df)
filtered_df["total_engagement_index"] = compute_total_engagement(filtered_df)

for col in [
    "behavioral_index",
    "cognitive_index",
    "affective_index",
    "total_engagement_index",
]:
    if col in filtered_df.columns:
        filtered_df[col] = pd.to_numeric(filtered_df[col], errors="coerce")

st.subheader("Method: Engagement Indices")
st.markdown(
    """
**Behavioral Index (BEI)** = 0.5 * normalized meeting attendance + 0.5 * normalized participation count.
Attendance is clipped to 0..4 and scaled to 0..1. Participation count is scaled by 8 activities.

**Cognitive Index (CEI)** = row-wise mean of available: scheme awareness count (min-max), awareness flags (min-max),
clarity level (if numeric), and application process knowledge (if numeric).

**Affective Index (AEI)** = row-wise mean of available: satisfaction, confidence, and referral likelihood,
scaled to 0..1.

**Total Engagement Index (TEI)** = row-wise mean of BEI, CEI, and AEI.
Missing components are skipped per row; indices are reported on available data.
"""
)

st.subheader("Engagement Dispersion")
dispersion_rows = []
for col, label in [
    ("behavioral_index", "BEI"),
    ("cognitive_index", "CEI"),
    ("affective_index", "AEI"),
    ("total_engagement_index", "TEI"),
]:
    if col in filtered_df.columns:
        std_val = safe_float(pd.to_numeric(filtered_df[col], errors="coerce").std(ddof=0))
        if std_val is None:
            interpretation = "NA"
        elif std_val < 0.05:
            interpretation = "Low discrimination"
        elif std_val <= 0.12:
            interpretation = "Moderate discrimination"
        else:
            interpretation = "Good discrimination"
        dispersion_rows.append(
            {
                "index_name": label,
                "std_dev": std_val,
                "interpretation": interpretation,
            }
        )

if dispersion_rows:
    dispersion_df = pd.DataFrame(dispersion_rows)
    dispersion_df["std_dev"] = dispersion_df["std_dev"].apply(
        lambda x: f"{x:.4f}" if x is not None else "NA"
    )
    st.dataframe(dispersion_df, use_container_width=True, height=200)
else:
    st.info("No engagement indices available for dispersion analysis.")

st.subheader("Which Dimension Drives Income?")
if "household_income_increase_percent" in filtered_df.columns:
    income = pd.to_numeric(filtered_df["household_income_increase_percent"], errors="coerce")
    bei_rank = pd.to_numeric(filtered_df["behavioral_index"], errors="coerce").rank(method="average")
    cei_rank = pd.to_numeric(filtered_df["cognitive_index"], errors="coerce").rank(method="average")
    aei_rank = pd.to_numeric(filtered_df["affective_index"], errors="coerce").rank(method="average")
    income_rank = income.rank(method="average")

    regression_df = pd.DataFrame(
        {
            "income_rank": income_rank,
            "BEI_rank": bei_rank,
            "CEI_rank": cei_rank,
            "AEI_rank": aei_rank,
        }
    ).dropna()

    if regression_df.shape[0] >= 10:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler

        X = regression_df[["BEI_rank", "CEI_rank", "AEI_rank"]].to_numpy()
        y = regression_df["income_rank"].to_numpy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LinearRegression()
        model.fit(X_scaled, y)
        coefs = model.coef_
        max_abs = max(abs(coef) for coef in coefs) if coefs.size else 1
        strength = [abs(coef) / max_abs * 100 for coef in coefs]

        coef_df = pd.DataFrame(
            {
                "dimension": ["BEI", "CEI", "AEI"],
                "standardized_coefficient": coefs,
                "relative_strength": strength,
            }
        )
        coef_df["standardized_coefficient"] = coef_df["standardized_coefficient"].apply(
            lambda x: f"{x:.3f}"
        )
        coef_df["relative_strength"] = coef_df["relative_strength"].apply(
            lambda x: f"{x:.1f}%"
        )
        st.dataframe(coef_df, use_container_width=True, height=200)
        st.caption("Rank-based regression used for robustness to non-normality.")
    else:
        st.info("Not enough rows for regression analysis.")
else:
    st.info("Income increase column not available for driver analysis.")

st.subheader("Action Quadrants")
bei = pd.to_numeric(filtered_df["behavioral_index"], errors="coerce")
cei = pd.to_numeric(filtered_df["cognitive_index"], errors="coerce")
if bei.notna().any() and cei.notna().any():
    q25_bei, q75_bei = bei.quantile(0.25), bei.quantile(0.75)
    q25_cei, q75_cei = cei.quantile(0.25), cei.quantile(0.75)

    quadrant_df = filtered_df.copy()
    quadrant_df["bei_level"] = np.where(bei >= q75_bei, "High", np.where(bei <= q25_bei, "Low", "Mid"))
    quadrant_df["cei_level"] = np.where(cei >= q75_cei, "High", np.where(cei <= q25_cei, "Low", "Mid"))

    def quadrant_label(row):
        if row["bei_level"] == "High" and row["cei_level"] == "High":
            return "Champions"
        if row["bei_level"] == "Low" and row["cei_level"] == "High":
            return "Aspirers (High Priority)"
        if row["bei_level"] == "High" and row["cei_level"] == "Low":
            return "Busy but Blind"
        if row["bei_level"] == "Low" and row["cei_level"] == "Low":
            return "Disengaged"
        return "Middle"

    quadrant_df["quadrant"] = quadrant_df.apply(quadrant_label, axis=1)

    if "household_income_increase_percent" in quadrant_df.columns:
        color_col = "income_quartile"
        outcome_vals = pd.to_numeric(quadrant_df["household_income_increase_percent"], errors="coerce")
        quadrant_df[color_col] = pd.qcut(outcome_vals, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop").astype(str)
    else:
        color_col = "tei_quartile"
        outcome_vals = pd.to_numeric(quadrant_df["total_engagement_index"], errors="coerce")
        quadrant_df[color_col] = pd.qcut(outcome_vals, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop").astype(str)

    fig = px.scatter(
        quadrant_df,
        x="behavioral_index",
        y="cognitive_index",
        color=color_col,
        title="BEI vs CEI (colored by outcome quartile)",
    )
    st.plotly_chart(fig, use_container_width=True)

    quadrant_summary = (
        quadrant_df.groupby("quadrant")
        .agg(
            count=("quadrant", "size"),
            avg_income=("household_income_increase_percent", "mean")
            if "household_income_increase_percent" in quadrant_df.columns
            else ("total_engagement_index", "mean"),
        )
        .reset_index()
    )
    quadrant_summary["recommended_action"] = quadrant_summary["quadrant"].map(
        {
            "Champions": "Leverage as peer leaders.",
            "Aspirers (High Priority)": "Targeted support and barrier removal.",
            "Busy but Blind": "Boost awareness and clarity.",
            "Disengaged": "Re-engage with outreach.",
            "Middle": "Maintain steady engagement.",
        }
    )
    st.dataframe(quadrant_summary, use_container_width=True, height=240)
else:
    st.info("Insufficient BEI/CEI coverage for quadrant analysis.")

st.subheader("District-Level Correlations")
district_corr = compute_district_correlations(filtered_df, min_n=50)
if district_corr.empty:
    st.info("District-level analysis unavailable or insufficient data (n>=50 required).")
else:
    st.caption(
        "Spearman correlation computed at district level; districts with n < min_n excluded."
    )
    st.dataframe(district_corr, use_container_width=True, height=300)
    if {"rho_BEI", "rho_CEI", "rho_AEI"}.issubset(district_corr.columns):
        district_corr["max_rho"] = district_corr[["rho_BEI", "rho_CEI", "rho_AEI"]].max(axis=1)
        district_corr["min_rho"] = district_corr[["rho_BEI", "rho_CEI", "rho_AEI"]].min(axis=1)
        strongest = district_corr.sort_values(by="max_rho", ascending=False).head(5)
        weakest = district_corr.sort_values(by="min_rho").head(5)
        st.markdown("**Strongest positive correlations (any index)**")
        st.dataframe(strongest[["district", "n", "rho_BEI", "rho_CEI", "rho_AEI"]], use_container_width=True, height=200)
        st.markdown("**Near-zero or negative correlations (any index)**")
        st.dataframe(weakest[["district", "n", "rho_BEI", "rho_CEI", "rho_AEI"]], use_container_width=True, height=200)

st.subheader("Input Analytics")
input_cols = [
    "meeting_attendance",
    "shg_meetings_attended_monthly",
    "participation_count",
    "scheme_awareness_count",
    "awareness_flags_count",
    "scheme_clarity_level",
    "application_process_knowledge",
    "satisfaction_level",
    "future_program_confidence",
    "referral_likelihood",
]

rows = []
for col in input_cols:
    if col not in filtered_df.columns:
        continue
    series = filtered_df[col]
    missing_pct = float(series.isna().mean() * 100)
    unique_count = int(series.nunique(dropna=True))
    row = {
        "column": col,
        "missing_pct": round(missing_pct, 2),
        "dtype": str(series.dtype),
        "unique_count": unique_count,
    }

    encoded = series
    if col in {"satisfaction_level", "future_program_confidence", "scheme_clarity_level", "application_process_knowledge"}:
        encoded = encode_likert_5(series)
    elif col == "referral_likelihood":
        encoded = encode_referral(series)
    else:
        encoded = coerce_numeric(series)

    if encoded.notna().any():
        row["min"] = float(encoded.min())
        row["median"] = float(encoded.median())
        row["max"] = float(encoded.max())
    else:
        row["min"] = np.nan
        row["median"] = np.nan
        row["max"] = np.nan

    if unique_count <= 20:
        row["top_values"] = series.dropna().astype(str).value_counts().head(5).to_dict()

    rows.append(row)

if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=360)
else:
    st.info("No input columns found for analytics.")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Rows", f"{len(filtered_df):,}")
avg_bei = safe_float(pd.to_numeric(filtered_df["behavioral_index"], errors="coerce").mean())
avg_cei = safe_float(pd.to_numeric(filtered_df["cognitive_index"], errors="coerce").mean())
avg_aei = safe_float(pd.to_numeric(filtered_df["affective_index"], errors="coerce").mean())
avg_tei = safe_float(pd.to_numeric(filtered_df["total_engagement_index"], errors="coerce").mean())
col2.metric("Avg BEI", f"{avg_bei:.3f}" if avg_bei is not None else "NA")
col3.metric("Avg CEI", f"{avg_cei:.3f}" if avg_cei is not None else "NA")
col4.metric("Avg AEI", f"{avg_aei:.3f}" if avg_aei is not None else "NA")
col5.metric("Avg TEI", f"{avg_tei:.3f}" if avg_tei is not None else "NA")

st.subheader("Engagement Distributions")
indices_df = filtered_df[["behavioral_index", "cognitive_index", "affective_index", "total_engagement_index"]].melt(
    var_name="index", value_name="score"
)
fig = px.box(
    indices_df,
    x="index",
    y="score",
    title="Engagement Index Distributions",
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Involvement Composition")
if "involvement_level" in filtered_df.columns:
    involvement_counts = (
        filtered_df["involvement_level"].astype("string").value_counts().reset_index()
    )
    involvement_counts.columns = ["involvement_level", "count"]
    fig = px.pie(
        involvement_counts,
        names="involvement_level",
        values="count",
        title="Involvement Composition",
        hole=0.5,
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Involvement column not available.")

st.subheader("Meeting Attendance Distribution")
if "meeting_attendance" in filtered_df.columns:
    fig = px.histogram(
        filtered_df,
        x="meeting_attendance",
        title="Meeting Attendance",
        nbins=5,
        range_x=[0, 4],
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Attendance column not available.")

st.subheader("Engagement Correlations (Spearman)")
correlation_targets = [
    "household_income_increase_percent",
    "overall_impact_rating",
    "satisfaction_level",
    "referral_likelihood",
    "empowerment_score",
]
correlation_cols = [
    "behavioral_index",
    "cognitive_index",
    "affective_index",
    "total_engagement_index",
]

corr_data = {
    "behavioral_index": filtered_df["behavioral_index"],
    "cognitive_index": filtered_df["cognitive_index"],
    "affective_index": filtered_df["affective_index"],
    "total_engagement_index": filtered_df["total_engagement_index"],
}

if "household_income_increase_percent" in filtered_df.columns:
    corr_data["household_income_increase_percent"] = coerce_numeric(
        filtered_df["household_income_increase_percent"]
    )

if "overall_impact_rating" in filtered_df.columns:
    corr_data["overall_impact_rating"] = encode_likert_5(
        filtered_df["overall_impact_rating"]
    )

if "satisfaction_level" in filtered_df.columns:
    corr_data["satisfaction_level"] = encode_likert_5(filtered_df["satisfaction_level"])

if "future_program_confidence" in filtered_df.columns:
    corr_data["future_program_confidence"] = encode_likert_5(
        filtered_df["future_program_confidence"]
    )

if "referral_likelihood" in filtered_df.columns:
    corr_data["referral_likelihood"] = encode_referral(
        filtered_df["referral_likelihood"]
    )

if "empowerment_score" in filtered_df.columns:
    corr_data["empowerment_score"] = coerce_numeric(filtered_df["empowerment_score"])

corr_df = pd.DataFrame(corr_data)

if corr_df.shape[1] < 2 or corr_df.dropna().shape[0] < 5:
    st.info("Not enough numeric data for correlation analysis.")
else:
    corr_matrix = corr_df.corr(method="spearman", numeric_only=True)
    corr_subset = corr_matrix.loc[
        correlation_cols, [col for col in corr_matrix.columns if col not in correlation_cols]
    ]
    if corr_subset.empty:
        st.info("Not enough numeric columns for correlation analysis.")
    else:
        fig = px.imshow(
            corr_subset,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            title="Spearman Correlation: Engagement vs Outcomes",
        )
        st.plotly_chart(fig, use_container_width=True)

st.subheader("Engagement vs Outcomes")
if "household_income_increase_percent" in filtered_df.columns:
    outcome_col = "household_income_increase_percent"
    outcome_label = "Income Increase (%)"
elif "overall_impact_rating" in filtered_df.columns:
    outcome_col = "overall_impact_rating"
    outcome_label = "Overall Impact Rating"
else:
    outcome_col = None
    outcome_label = None

if outcome_col:
    scatter_df = filtered_df[["total_engagement_index", outcome_col]].dropna()
    fig = px.scatter(
        scatter_df,
        x="total_engagement_index",
        y=outcome_col,
        title=f"TEI vs {outcome_label}",
    )
    st.plotly_chart(fig, use_container_width=True)

    if len(scatter_df) > 10:
        scatter_df = scatter_df.copy()
        scatter_df["tei_decile"] = pd.qcut(
            scatter_df["total_engagement_index"], 10, duplicates="drop"
        )
        scatter_df["tei_decile_label"] = scatter_df["tei_decile"].astype(str)
        centers = scatter_df["tei_decile"].apply(
            lambda val: (val.left + val.right) / 2 if pd.notna(val) else None
        )
        scatter_df["tei_bin_center"] = pd.to_numeric(centers, errors="coerce")
        decile_means = (
            scatter_df.groupby("tei_bin_center")[outcome_col].mean().reset_index()
        )
        fig = px.line(
            decile_means,
            x="tei_bin_center",
            y=outcome_col,
            title=f"Mean {outcome_label} by TEI Decile (bin centers)",
        )
        st.plotly_chart(fig, use_container_width=True)

        scatter_df["tei_quartile"] = pd.qcut(
            scatter_df["total_engagement_index"],
            4,
            labels=["Q1", "Q2", "Q3", "Q4"],
        )
        scatter_df["tei_quartile_label"] = scatter_df["tei_quartile"].astype(str)
        scatter_df["tei_quartile_label"] = pd.Categorical(
            scatter_df["tei_quartile_label"],
            ordered=True,
            categories=sorted(scatter_df["tei_quartile_label"].unique()),
        )
        fig = px.box(
            scatter_df,
            x="tei_quartile_label",
            y=outcome_col,
            title=f"{outcome_label} by TEI Quartile",
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if any(
            isinstance(val, pd.Interval)
            for val in scatter_df.get("tei_decile", pd.Series(dtype=object)).dropna()
        ):
            scatter_df["tei_decile"] = scatter_df["tei_decile"].astype(str)
else:
    st.info("No outcome column available for bivariate analysis.")

with st.expander("Diagnostics", expanded=False):
    input_cols = [
        "meeting_attendance",
        "shg_meetings_attended_monthly",
        "involvement_level",
        "scheme_awareness_count",
        "scheme_clarity_level",
        "application_process_knowledge",
        "satisfaction_level",
        "future_program_confidence",
        "referral_likelihood",
    ]
    input_cols = [col for col in input_cols if col in filtered_df.columns]
    missing_pct = (
        filtered_df[input_cols].isna().mean().sort_values(ascending=False) * 100
        if input_cols
        else pd.Series(dtype=float)
    )
    if not missing_pct.empty:
        st.markdown("**Missing % of inputs**")
        st.dataframe(missing_pct.reset_index().rename(columns={"index": "column", 0: "missing_pct"}))

    pair_counts = {}
    for col in correlation_targets:
        if col in filtered_df.columns:
            pair_counts[col] = int(
                filtered_df[["total_engagement_index", col]].dropna().shape[0]
            )
    if pair_counts:
        st.markdown("**Non-null pairs for correlations**")
        st.json(pair_counts)

    index_std = {
        "behavioral_index": safe_float(pd.to_numeric(filtered_df["behavioral_index"], errors="coerce").std(ddof=0)),
        "cognitive_index": safe_float(pd.to_numeric(filtered_df["cognitive_index"], errors="coerce").std(ddof=0)),
        "affective_index": safe_float(pd.to_numeric(filtered_df["affective_index"], errors="coerce").std(ddof=0)),
        "total_engagement_index": safe_float(pd.to_numeric(filtered_df["total_engagement_index"], errors="coerce").std(ddof=0)),
    }
    st.markdown("**Index standard deviations**")
    st.json(index_std)
    if any(val is not None and val < 0.02 for val in index_std.values()):
        st.warning("One or more indices have near-zero variance; review inputs.")

with st.expander("Debug: encoding", expanded=False):
    for col in ["satisfaction_level", "future_program_confidence", "overall_impact_rating"]:
        if col in filtered_df.columns:
            st.write(f"Raw values ({col}):", filtered_df[col].dropna().astype(str).unique()[:10].tolist())
            if col in {"satisfaction_level", "future_program_confidence", "scheme_clarity_level", "application_process_knowledge"}:
                encoded = encode_likert_5(filtered_df[col])
            else:
                encoded = encode_likert_5(filtered_df[col])
            st.write(f"Encoded values ({col}):", encoded.dropna().unique()[:10].tolist())
