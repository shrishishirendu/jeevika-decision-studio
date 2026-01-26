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
from src.features import compute_kpis

CONFIG_PATH = ROOT / "config" / "config.yaml"

st.set_page_config(page_title="Overview", page_icon="??", layout="wide")

st.title("Overview")

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
except Exception as exc:  # pragma: no cover - streamlit error display
    st.error(f"Failed to load cleaned dataset: {exc}")
    st.stop()

if not parquet_path.exists():
    st.warning("Processed parquet not found. Using cleaned raw CSV in memory.")

filtered_df, selections = apply_filters(df)

st.sidebar.markdown("### Active Filters")
for key, value in selections.items():
    st.sidebar.write(f"{key}: {value}")

kpis = compute_kpis(filtered_df)

col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{kpis.get('rows', 0):,}")
if "avg_membership_duration" in kpis:
    col2.metric("Avg Membership Duration", f"{kpis['avg_membership_duration']:.2f}")
if "avg_income_increase" in kpis:
    col3.metric("Avg Income Increase %", f"{kpis['avg_income_increase']:.2f}")

st.subheader("Pre-Jeevika Income Distribution")
if "pre_jeevika_income_bucket_clean" in filtered_df.columns:
    fig = px.histogram(
        filtered_df,
        x="pre_jeevika_income_bucket_clean",
        title="Pre-Jeevika Income Distribution",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Pre-Jeevika income bucket column not available.")

st.subheader("Membership Duration")
if "membership_duration" in filtered_df.columns:
    fig = px.histogram(
        filtered_df,
        x="membership_duration",
        nbins=20,
        title="Membership Duration Distribution",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Membership duration column not available.")
