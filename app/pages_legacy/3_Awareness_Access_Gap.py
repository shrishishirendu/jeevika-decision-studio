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

st.set_page_config(page_title="Awareness & Access Gap", page_icon="??", layout="wide")

st.title("Awareness & Access Gap")

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

awareness_cols = [
    col
    for col in filtered_df.columns
    if col.lower() in {"awareness", "awareness_proxy", "program_awareness"}
]
access_cols = [
    col
    for col in filtered_df.columns
    if col.lower() in {"access", "access_proxy", "program_access"}
]

st.subheader("Awareness Distribution")
if awareness_cols:
    fig = px.histogram(filtered_df, x=awareness_cols[0], title="Awareness Proxy")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Awareness proxy column not available.")

st.subheader("Access Distribution")
if access_cols:
    fig = px.histogram(filtered_df, x=access_cols[0], title="Access Proxy")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Access proxy column not available.")

st.subheader("Awareness vs Access Gap")
if awareness_cols and access_cols:
    gap_df = (
        filtered_df.groupby([awareness_cols[0], access_cols[0]])
        .size()
        .reset_index(name="count")
    )
    gap_df["pct"] = (gap_df["count"] / gap_df["count"].sum() * 100).round(2)
    st.dataframe(gap_df, use_container_width=True)
else:
    st.info("Gap table requires both awareness and access columns.")
