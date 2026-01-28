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
from src.readiness import build_readiness_frame
from src.segments import assign_population_segments
from app.utils.layout import render_page_header

CONFIG_PATH = ROOT / "config" / "config.yaml"

st.set_page_config(page_title="Decision Readiness & Targeting", page_icon="Decision Readiness", layout="wide")

render_page_header("Decision Readiness & Targeting")
st.caption(
    "Readiness combines Engagement, Awareness, and Access to target field actions with deterministic rules."
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
if selections:
    for key, value in selections.items():
        st.sidebar.write(f"{key}: {value}")
else:
    st.sidebar.write("Showing full dataset (no filters selected)")

df_r = build_readiness_frame(filtered_df)

unknown_counts = {
    "engagement": int((df_r["engagement_readiness"] == "Unknown").sum()),
    "awareness": int((df_r["awareness_readiness"] == "Unknown").sum()),
    "access": int((df_r["access_readiness"] == "Unknown").sum()),
}

unknown_pct = {k: v / len(df_r) * 100 if len(df_r) else 0 for k, v in unknown_counts.items()}

if any(val > 20 for val in unknown_pct.values()):
    st.warning("Unknown readiness exceeds 20% for at least one dimension. Review input coverage.")

st.subheader("Decision Profile Distribution")
profile_counts = df_r["decision_profile"].value_counts().reset_index()
profile_counts.columns = ["profile", "count"]
fig = px.pie(
    profile_counts,
    names="profile",
    values="count",
    title="Decision Profile Distribution",
    hole=0.5,
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Decision Profile Summary")
summary_df = (
    df_r.groupby("decision_profile")
    .agg(
        count=("decision_profile", "size"),
        avg_tei=("total_engagement_index", "mean"),
        avg_income=("household_income_increase_percent", "mean")
        if "household_income_increase_percent" in df_r.columns
        else ("total_engagement_index", "mean"),
    )
    .reset_index()
)
summary_df["pct"] = (summary_df["count"] / summary_df["count"].sum() * 100).round(2)
st.dataframe(summary_df, use_container_width=True, height=300)

st.subheader("Decision Matrix")
matrix_df = (
    df_r.groupby(["engagement_readiness", "awareness_readiness", "access_readiness"])
    .size()
    .reset_index(name="count")
)
st.dataframe(matrix_df, use_container_width=True, height=260)

st.subheader("District Readiness Breakdown")
if "district" in df_r.columns:
    district_summary = (
        df_r.groupby("district")
        .agg(
            n=("district", "size"),
            avg_tei=("total_engagement_index", "mean"),
            ready=("decision_profile", lambda s: (s == "Ready-to-Scale").mean() * 100),
            aware_blocked=("decision_profile", lambda s: (s == "Aware-but-Blocked").mean() * 100),
            motivated_unaware=("decision_profile", lambda s: (s == "Motivated-but-Unaware").mean() * 100),
            disengaged=("decision_profile", lambda s: (s == "Disengaged / Intensive Support").mean() * 100),
        )
        .reset_index()
    )

    district_summary = district_summary[district_summary["n"] >= 50]
    if not district_summary.empty:
        st.dataframe(district_summary, use_container_width=True, height=320)

        districts = sorted(district_summary["district"].tolist())
        selected_district = st.selectbox("Select district", districts)
        dist_df = df_r[df_r["district"] == selected_district]
        dist_profile = dist_df["decision_profile"].value_counts().reset_index()
        dist_profile.columns = ["profile", "count"]
        st.markdown("**Profile distribution**")
        st.dataframe(dist_profile, use_container_width=True, height=200)

        actions = (
            dist_df["decision_action"].value_counts().head(3).reset_index()
        )
        actions.columns = ["action", "count"]
        st.markdown("**Top actions**")
        st.dataframe(actions, use_container_width=True, height=160)
    else:
        st.info("No districts meet the minimum sample size (n>=50).")
else:
    st.info("District column not available.")

st.subheader("Segment x Decision Profile")
try:
    df_r["segment_label"] = assign_population_segments(df_r)
    crosstab = (
        pd.crosstab(df_r["segment_label"], df_r["decision_profile"], normalize="index")
        * 100
    ).round(2)
    st.dataframe(crosstab.reset_index(), use_container_width=True, height=300)
except Exception:
    st.info("Segment cross-tab unavailable.")

st.subheader("Field Action Summary")
if "district" in df_r.columns:
    action_df = district_summary.copy()
    st.markdown("**Top districts by Ready-to-Scale**")
    st.dataframe(action_df.sort_values(by="ready", ascending=False).head(5), use_container_width=True, height=220)
    st.markdown("**Top districts by Aware-but-Blocked**")
    st.dataframe(action_df.sort_values(by="aware_blocked", ascending=False).head(5), use_container_width=True, height=220)
    st.markdown("**Top districts by Motivated-but-Unaware**")
    st.dataframe(action_df.sort_values(by="motivated_unaware", ascending=False).head(5), use_container_width=True, height=220)

if st.button("Save Decision Readiness dataset"):
    output_path = ROOT / "data" / "processed" / "decision_readiness.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_r.to_parquet(output_path, index=False)
    st.success(f"Decision Readiness dataset saved to {output_path}")
