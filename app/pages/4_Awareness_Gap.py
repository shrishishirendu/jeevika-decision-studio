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
from src.engagement_indices import compute_cognitive_index
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

st.set_page_config(
    page_title="Awareness Gaps & Information Barriers",
    page_icon="Awareness Gaps",
    layout="wide",
)

render_page_header("Awareness Gaps & Information Barriers")

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

filtered_df["segment_label"] = assign_population_segments(filtered_df)

awareness_cols = [col for col in filtered_df.columns if col.endswith("_awareness")]

st.subheader("Awareness Landscape")
if "scheme_awareness_count" in filtered_df.columns:
    fig = px.histogram(
        filtered_df,
        x="scheme_awareness_count",
        title="Scheme Awareness Count",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("scheme_awareness_count not available.")

st.subheader("Awareness by Segment")
if "segment_label" in filtered_df.columns:
    segment_summary = (
        filtered_df.groupby("segment_label")
        .agg(
            avg_awareness=("scheme_awareness_count", "mean")
            if "scheme_awareness_count" in filtered_df.columns
            else ("segment_label", "size"),
            avg_clarity=("scheme_clarity_level", lambda s: encode_likert_5(s).mean())
            if "scheme_clarity_level" in filtered_df.columns
            else ("segment_label", "size"),
        )
        .reset_index()
    )

    if awareness_cols:
        top_unaware = {}
        for segment in filtered_df["segment_label"].dropna().unique():
            seg_df = filtered_df[filtered_df["segment_label"] == segment]
            rates = {}
            for col in awareness_cols:
                series = pd.to_numeric(seg_df[col], errors="coerce")
                rates[col] = safe_float((series == 0).mean() * 100)
            top_unaware[segment] = [
                key
                for key, value in sorted(
                    rates.items(),
                    key=lambda item: item[1] if item[1] is not None else -1,
                    reverse=True,
                )
            ][:3]
        segment_summary["top_unaware_schemes"] = segment_summary["segment_label"].map(top_unaware)

    st.dataframe(segment_summary, use_container_width=True, height=300)
    st.caption("Focus segments: Aspirers (High Priority) and Busy but Blind.")
else:
    st.info("Segment labels unavailable.")

st.subheader("Awareness by District")
if "district" in filtered_df.columns and "scheme_awareness_count" in filtered_df.columns:
    district_summary = (
        filtered_df.groupby("district")
        .agg(
            n=("district", "size"),
            avg_awareness=("scheme_awareness_count", "mean"),
        )
        .reset_index()
    )
    district_summary = district_summary[district_summary["n"] >= 50]
    if not district_summary.empty:
        global_mean = district_summary["avg_awareness"].mean()
        district_summary["flag_low"] = district_summary["avg_awareness"] < global_mean * 0.8

        top_schemes = awareness_cols[:5]
        for col in top_schemes:
            if col in filtered_df.columns:
                pct = (
                    filtered_df.groupby("district")[col]
                    .apply(lambda s: safe_float((pd.to_numeric(s, errors="coerce") == 1).mean() * 100))
                    .rename(f"{col}_aware_pct")
                )
                district_summary = district_summary.merge(pct, on="district", how="left")

        st.dataframe(district_summary, use_container_width=True, height=320)

        if top_schemes:
            matrix_cols = ["district"] + [f"{col}_aware_pct" for col in top_schemes if col in filtered_df.columns]
            matrix_df = district_summary[matrix_cols]
            csv_bytes = matrix_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download district awareness matrix (CSV)",
                csv_bytes,
                file_name="district_awareness_matrix.csv",
                mime="text/csv",
            )
    else:
        st.info("No districts meet the minimum sample size (n>=50).")
else:
    st.info("District-level awareness analysis unavailable.")
