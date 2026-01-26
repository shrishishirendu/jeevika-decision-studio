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
from src.segments import assign_population_segments, segment_quality_checks, summarize_segments

CONFIG_PATH = ROOT / "config" / "config.yaml"

st.set_page_config(page_title="Population Structure", page_icon="Population Structure", layout="wide")

st.title("Population Structure")
st.caption(
    "Segments are defined using engagement indices (Behavioral, Cognitive, Affective) and quartiles for robustness."
)

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

filtered_df["behavioral_index"] = compute_behavioral_index(filtered_df)
filtered_df["cognitive_index"] = compute_cognitive_index(filtered_df)
filtered_df["affective_index"] = compute_affective_index(filtered_df)
filtered_df["total_engagement_index"] = compute_total_engagement(filtered_df)

filtered_df["segment_label"] = assign_population_segments(filtered_df)
summary_df = summarize_segments(filtered_df, segment_col="segment_label")
quality = segment_quality_checks(summary_df)

if quality["warnings"]:
    st.warning(" | ".join(quality["warnings"]))

st.subheader("Segment Composition")
segment_counts = summary_df[["segment_label", "count"]]
fig = px.pie(
    segment_counts,
    names="segment_label",
    values="count",
    title="Segment Composition",
    hole=0.5,
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("BEI vs CEI Quadrant")
q25_bei = pd.to_numeric(filtered_df["behavioral_index"], errors="coerce").quantile(0.25)
q75_bei = pd.to_numeric(filtered_df["behavioral_index"], errors="coerce").quantile(0.75)
q25_cei = pd.to_numeric(filtered_df["cognitive_index"], errors="coerce").quantile(0.25)
q75_cei = pd.to_numeric(filtered_df["cognitive_index"], errors="coerce").quantile(0.75)

hover_cols = [
    col
    for col in [
        "respondent_id",
        "district",
        "behavioral_index",
        "cognitive_index",
        "affective_index",
        "total_engagement_index",
    ]
    if col in filtered_df.columns
]

fig = px.scatter(
    filtered_df,
    x="behavioral_index",
    y="cognitive_index",
    color="segment_label",
    title="Behavioral vs Cognitive Engagement",
    hover_data=hover_cols,
)
fig.add_vline(x=q75_bei, line_dash="dash", line_color="gray")
fig.add_hline(y=q75_cei, line_dash="dash", line_color="gray")
fig.add_vline(x=q25_bei, line_dash="dot", line_color="lightgray")
fig.add_hline(y=q25_cei, line_dash="dot", line_color="lightgray")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Segment Profile Summary")
st.dataframe(summary_df, use_container_width=True, height=340)

st.subheader("Field Focus Priorities")
focus_rows = [
    {
        "segment": "Aspirers (High Priority)",
        "action": "Enable participation; remove access friction; targeted facilitation",
    },
    {
        "segment": "Motivated but Blocked",
        "action": "Convert intent to behavior; local activation; peer support",
    },
    {
        "segment": "Busy but Blind",
        "action": "Improve clarity; scheme navigation; process knowledge",
    },
    {
        "segment": "Disengaged",
        "action": "Trust building; low-cost reactivation",
    },
]
focus_df = pd.DataFrame(focus_rows)
focus_df["count"] = focus_df["segment"].map(
    summary_df.set_index("segment_label")["count"]
).fillna(0).astype(int)
st.dataframe(focus_df[["segment", "count", "action"]], use_container_width=True, height=220)

st.subheader("Stability by Geography (District)")
if "district" in filtered_df.columns:
    district_counts = filtered_df["district"].value_counts()
    min_n = 200
    eligible_districts = district_counts[district_counts >= min_n].index.tolist()
    if eligible_districts:
        district_df = filtered_df[filtered_df["district"].isin(eligible_districts)].copy()
        global_means = {
            "behavioral_index": pd.to_numeric(filtered_df["behavioral_index"], errors="coerce").mean(),
            "cognitive_index": pd.to_numeric(filtered_df["cognitive_index"], errors="coerce").mean(),
            "affective_index": pd.to_numeric(filtered_df["affective_index"], errors="coerce").mean(),
        }

        district_stats = (
            district_df.groupby("district")[["behavioral_index", "cognitive_index", "affective_index"]]
            .agg(["mean", "std"])
        )
        district_stats.columns = ["_".join(col) for col in district_stats.columns]
        district_stats = district_stats.reset_index()
        for idx in ["behavioral_index", "cognitive_index", "affective_index"]:
            mean_col = f"{idx}_mean"
            district_stats[f"{idx}_mean_delta_pct"] = (
                (district_stats[mean_col] - global_means[idx]) / global_means[idx] * 100
            )
            district_stats[f"{idx}_flag"] = district_stats[f"{idx}_mean_delta_pct"].abs() > 20

        st.markdown("**District Mean Deviation (% vs global mean)**")
        st.dataframe(
            district_stats[
                [
                    "district",
                    "behavioral_index_mean",
                    "cognitive_index_mean",
                    "affective_index_mean",
                    "behavioral_index_mean_delta_pct",
                    "cognitive_index_mean_delta_pct",
                    "affective_index_mean_delta_pct",
                    "behavioral_index_flag",
                    "cognitive_index_flag",
                    "affective_index_flag",
                ]
            ],
            use_container_width=True,
            height=300,
        )

        district_total = district_df.groupby("district").size().rename("total")
        district_segment = (
            district_df.groupby(["district", "segment_label"])
            .size()
            .rename("count")
            .reset_index()
        )
        district_segment = district_segment.merge(district_total, on="district", how="left")
        district_segment["pct"] = (district_segment["count"] / district_segment["total"] * 100).round(2)

        segment_table = district_segment.pivot_table(
            index="district", columns="segment_label", values="pct", fill_value=0
        ).reset_index()
        segment_table["flag_any_segment_gt_70"] = segment_table.drop(columns=["district"]).max(axis=1) > 70
        st.markdown("**Segment Mix by District (pct)**")
        st.dataframe(segment_table, use_container_width=True, height=320)

        top_districts = district_counts.loc[eligible_districts].head(8).index.tolist()
        district_bar = district_segment[district_segment["district"].isin(top_districts)]
        fig = px.bar(
            district_bar,
            x="district",
            y="pct",
            color="segment_label",
            title="Segment % by District (Top 8 by N)",
        )
        st.plotly_chart(fig, use_container_width=True)

        if "household_income_increase_percent" in district_df.columns:
            outcome_df = district_df.copy()
            outcome_df["income"] = pd.to_numeric(
                outcome_df["household_income_increase_percent"], errors="coerce"
            )
            outcome_summary = (
                outcome_df.groupby(["district", "segment_label"])
                .agg(count=("segment_label", "size"), avg_income_uplift=("income", "mean"))
                .reset_index()
            )
            outcome_summary = outcome_summary[outcome_summary["count"] >= 30]
            if not outcome_summary.empty:
                st.markdown("**Outcome alignment (avg income uplift) by district & segment**")
                st.dataframe(outcome_summary, use_container_width=True, height=320)
    else:
        st.info("Not enough district-level sample size (n>=200) for stability analysis.")
else:
    st.info("District-level analysis unavailable.")

if any(val < 0.02 for val in [
    pd.to_numeric(filtered_df["behavioral_index"], errors="coerce").std(ddof=0),
    pd.to_numeric(filtered_df["cognitive_index"], errors="coerce").std(ddof=0),
    pd.to_numeric(filtered_df["affective_index"], errors="coerce").std(ddof=0),
]):
    st.warning("One or more engagement indices have near-zero variance; review inputs.")

if st.button("Save segmented dataset"):
    output_path = ROOT / "data" / "processed" / "segmented_population.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_parquet(output_path, index=False)
    st.success(f"Segmented dataset saved to {output_path}")
