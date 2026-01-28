from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.ui_filters import apply_filters, load_clean_df
from src.encoding import encode_likert_5
from src.engagement_indices import (
    compute_affective_index,
    compute_behavioral_index,
    compute_cognitive_index,
    compute_total_engagement,
)
from src.features import compute_empowerment_score
from src.segments import assign_population_segments
from app.utils.layout import render_page_header

CONFIG_PATH = ROOT / "config" / "config.yaml"

def safe_float(value):
    import pandas as pd

    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except TypeError:
        return None

st.set_page_config(page_title="Impact & Outcomes", page_icon="Impact & Outcomes", layout="wide")

render_page_header("Impact & Outcomes")

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
if selections:
    for key, value in selections.items():
        st.sidebar.write(f"{key}: {value}")
else:
    st.sidebar.write("Showing full dataset (no filters selected)")

filtered_df["behavioral_index"] = compute_behavioral_index(filtered_df)
filtered_df["cognitive_index"] = compute_cognitive_index(filtered_df)
filtered_df["affective_index"] = compute_affective_index(filtered_df)
filtered_df["total_engagement_index"] = compute_total_engagement(filtered_df)
filtered_df["segment_label"] = assign_population_segments(filtered_df)
filtered_df["empowerment_score"] = compute_empowerment_score(filtered_df)

income_col = "household_income_increase_percent"
impact_col = "overall_impact_rating"

if income_col in filtered_df.columns:
    filtered_df[income_col] = pd.to_numeric(filtered_df[income_col], errors="coerce")

if impact_col in filtered_df.columns:
    filtered_df[impact_col] = encode_likert_5(filtered_df[impact_col])

st.subheader("Key Outcome KPIs")
col1, col2, col3, col4 = st.columns(4)

avg_income = (
    safe_float(pd.to_numeric(filtered_df[income_col], errors="coerce").mean())
    if income_col in filtered_df.columns
    else None
)

pct_income_10 = (
    safe_float((pd.to_numeric(filtered_df[income_col], errors="coerce") >= 10).mean() * 100)
    if income_col in filtered_df.columns
    else None
)

pct_income_20 = (
    safe_float((pd.to_numeric(filtered_df[income_col], errors="coerce") >= 20).mean() * 100)
    if income_col in filtered_df.columns
    else None
)

avg_impact = (
    safe_float(pd.to_numeric(filtered_df[impact_col], errors="coerce").mean())
    if impact_col in filtered_df.columns
    else None
)

avg_empowerment = (
    safe_float(pd.to_numeric(filtered_df["empowerment_score"], errors="coerce").mean())
    if "empowerment_score" in filtered_df.columns
    else None
)

col1.metric("Avg income increase %", f"{avg_income:.2f}" if avg_income is not None else "NA")
col2.metric(">=10% income uplift", f"{pct_income_10:.1f}%" if pct_income_10 is not None else "NA")
col3.metric(">=20% income uplift", f"{pct_income_20:.1f}%" if pct_income_20 is not None else "NA")
col4.metric("Avg empowerment score", f"{avg_empowerment:.2f}" if avg_empowerment is not None else "NA")

st.subheader("Impact by Segment")
if income_col in filtered_df.columns:
    fig = px.box(
        filtered_df,
        x="segment_label",
        y=income_col,
        title="Income Increase by Segment",
    )
    st.plotly_chart(fig, use_container_width=True)
    impact_metric_col = income_col
else:
    if impact_col in filtered_df.columns:
        fig = px.box(
            filtered_df,
            x="segment_label",
            y=impact_col,
            title="Overall Impact by Segment",
        )
        st.plotly_chart(fig, use_container_width=True)
        impact_metric_col = impact_col
    else:
        impact_metric_col = None
        st.info("No income or impact rating available for segment analysis.")

segment_summary = (
    filtered_df.groupby("segment_label")
    .agg(
        n=("segment_label", "size"),
        avg_income=(income_col, "mean") if income_col in filtered_df.columns else ("total_engagement_index", "mean"),
        avg_impact=(impact_col, "mean") if impact_col in filtered_df.columns else ("total_engagement_index", "mean"),
        avg_empowerment=("empowerment_score", "mean"),
        avg_tei=("total_engagement_index", "mean"),
    )
    .reset_index()
    .sort_values(by="n", ascending=False)
)

st.dataframe(segment_summary, use_container_width=True, height=320)

st.subheader("Impact by District")
if "district" in filtered_df.columns:
    district_df = filtered_df.copy()
    district_df["tei"] = pd.to_numeric(district_df["total_engagement_index"], errors="coerce")

    district_summary = (
        district_df.groupby("district")
        .agg(
            n=("district", "size"),
            avg_income=(income_col, "mean") if income_col in district_df.columns else ("tei", "mean"),
            pct_20=(income_col, lambda s: float((s >= 20).mean() * 100)) if income_col in district_df.columns else ("tei", "mean"),
            avg_tei=("tei", "mean"),
        )
        .reset_index()
    )

    district_summary = district_summary[district_summary["n"] >= 50]
    if not district_summary.empty:
        st.dataframe(district_summary, use_container_width=True, height=320)

        top10 = district_summary.sort_values(by="avg_income", ascending=False).head(10)
        bottom10 = district_summary.sort_values(by="avg_income", ascending=True).head(10)
        st.markdown("**Top 10 districts by average income uplift**")
        st.dataframe(top10, use_container_width=True, height=240)
        st.markdown("**Bottom 10 districts by average income uplift**")
        st.dataframe(bottom10, use_container_width=True, height=240)

        csv_bytes = district_summary.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download district summary (CSV)",
            csv_bytes,
            file_name="district_impact_summary.csv",
            mime="text/csv",
        )
    else:
        st.info("No districts meet the minimum sample size (n>=50).")
else:
    st.info("District-level analysis unavailable.")

st.subheader("Engagement ? Impact")
if income_col in filtered_df.columns:
    outcome_col = income_col
    outcome_label = "Income Increase (%)"
elif impact_col in filtered_df.columns:
    outcome_col = impact_col
    outcome_label = "Overall Impact Rating"
else:
    outcome_col = None
    outcome_label = None

if outcome_col:
    trend_df = filtered_df[["total_engagement_index", outcome_col]].dropna().copy()
    if not trend_df.empty:
        trend_df["tei_decile"] = pd.qcut(
            trend_df["total_engagement_index"], 10, duplicates="drop"
        )
        trend_df["tei_decile_label"] = trend_df["tei_decile"].astype(str)
        trend_summary = (
            trend_df.groupby("tei_decile_label")
            .agg(mean_outcome=(outcome_col, "mean"), n=(outcome_col, "size"))
            .reset_index()
        )
        fig = px.line(
            trend_summary,
            x="tei_decile_label",
            y="mean_outcome",
            title=f"Mean {outcome_label} by TEI Decile",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Binned averages; not causal.")
        st.dataframe(trend_summary, use_container_width=True, height=240)
    else:
        st.info("No data for TEI decile analysis.")
else:
    st.info("No outcome column available for engagement impact analysis.")

st.subheader("Levers / Diagnostics")
if "district" in filtered_df.columns and income_col in filtered_df.columns:
    district_summary = (
        filtered_df.groupby("district")
        .agg(n=("district", "size"), avg_income=(income_col, "mean"))
        .reset_index()
    )
    bottom_districts = district_summary.sort_values(by="avg_income").head(5)
    bottom_list = bottom_districts["district"].tolist()

    if bottom_list:
        lever_df = filtered_df[filtered_df["district"].isin(bottom_list)].copy()
        barrier_cols = [col for col in lever_df.columns if col.startswith("barrier_")]
        if barrier_cols:
            barrier_rates = (
                lever_df[barrier_cols]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .mean()
                .sort_values(ascending=False)
                .head(5)
            )
            barrier_table = barrier_rates.reset_index()
            barrier_table.columns = ["barrier", "prevalence"]
            st.markdown("**Top barriers (bottom 5 districts)**")
            st.dataframe(barrier_table, use_container_width=True, height=220)
        else:
            st.info("No barrier columns available for diagnostics.")

        metrics = {
            "avg_cognitive_index": float(pd.to_numeric(lever_df["cognitive_index"], errors="coerce").mean()),
            "avg_access_ease": float(pd.to_numeric(lever_df.get("access_ease_avg"), errors="coerce").mean())
            if "access_ease_avg" in lever_df.columns
            else np.nan,
        }
        st.markdown("**Additional diagnostics**")
        st.json(metrics)
    else:
        st.info("No districts available for lever diagnostics.")
else:
    st.info("District-level levers unavailable (missing district or income).")
