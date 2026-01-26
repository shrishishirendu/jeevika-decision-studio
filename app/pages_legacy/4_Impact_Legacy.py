from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.ui_filters import apply_filters, load_clean_df
from src.features import compute_engagement_index

CONFIG_PATH = ROOT / "config" / "config.yaml"

st.set_page_config(page_title="Impact (Legacy)", page_icon="??", layout="wide")

st.title("Impact (Legacy)")

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

st.sidebar.markdown("### Active Filters")
for key, value in selections.items():
    st.sidebar.write(f"{key}: {value}")

income_col = "household_income_increase_percent"

st.subheader("Income Increase Distribution")
if income_col in filtered_df.columns:
    fig = px.histogram(
        filtered_df,
        x=income_col,
        nbins=20,
        title="Income Increase (%)",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Income increase column not available.")

st.subheader("Income Increase by Pre-Income Bucket")
if income_col in filtered_df.columns and "pre_jeevika_income_bucket_clean" in filtered_df.columns:
    fig = px.box(
        filtered_df,
        x="pre_jeevika_income_bucket_clean",
        y=income_col,
        title="Income Increase by Pre-Income Bucket",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Pre-income bucket or income increase column not available.")

st.subheader("Income Increase vs Engagement")
engagement_index = compute_engagement_index(filtered_df)
if income_col in filtered_df.columns and engagement_index.notna().any():
    scatter_df = filtered_df.copy()
    scatter_df["engagement_index"] = engagement_index
    fig = px.scatter(
        scatter_df,
        x="engagement_index",
        y=income_col,
        title="Income Increase vs Engagement Index",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Income increase or engagement index not available.")
