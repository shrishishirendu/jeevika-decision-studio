from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.ui_filters import apply_filters, load_clean_df
from src.ml_high_uplift import get_predictions_for_df
from src import recommendations as recommendations
from src.segments import assign_population_segments
from app.utils.layout import render_page_header

CONFIG_PATH = ROOT / "config" / "config.yaml"
PRED_PATH = ROOT / "data" / "processed" / "predicted_high_uplift.parquet"
LOG_PATH = ROOT / "data" / "recommendation_logs.csv"

st.set_page_config(page_title="Recommendations", page_icon="Recommendations", layout="wide")

render_page_header("Recommendations")
st.caption("Explainable recommendations based on prediction, segment, and key drivers.")

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

filtered_df, selections = apply_filters(df)
filtered_df = filtered_df.copy()

st.sidebar.markdown("### Active Filters")
if selections:
    for key, value in selections.items():
        st.sidebar.write(f"{key}: {value}")
else:
    st.sidebar.write("Showing full dataset (no filters selected)")

id_col = "respondent_id" if "respondent_id" in filtered_df.columns else None
if id_col is None:
    filtered_df["respondent_id"] = filtered_df.index.astype(str)
    id_col = "respondent_id"

pred_df = pd.DataFrame()
if PRED_PATH.exists():
    pred_df = pd.read_parquet(PRED_PATH)
else:
    st.info("Predictions file not found. Please run the Prediction page and save predictions.")

if not pred_df.empty:
    merged = filtered_df.merge(pred_df, left_on=id_col, right_on="respondent_id", how="left")
else:
    merged = filtered_df.copy()

if "segment_label" not in merged.columns:
    merged["segment_label"] = assign_population_segments(merged)

ids = merged[id_col].dropna().astype(str).unique().tolist()
if not ids:
    st.info("No IDs available to recommend.")
    st.stop()

selected_id = st.selectbox("Select respondent ID", ids)
row = merged[merged[id_col].astype(str) == str(selected_id)].iloc[0]

prediction = int(row.get("predicted_class", 0)) if pd.notna(row.get("predicted_class")) else 0
proba_candidates = [
    row.get("predicted_prob"),
    row.get("probability"),
    row.get("predicted_probability"),
]
proba_value = next((val for val in proba_candidates if pd.notna(val)), 0.0)
proba = float(proba_value) if proba_value is not None else 0.0
threshold = 0.5

generate_recommendation = getattr(recommendations, "generate_recommendation", None)
log_feedback = getattr(recommendations, "log_feedback", None)
risk_label = getattr(recommendations, "risk_label", None)
if generate_recommendation is None or log_feedback is None or risk_label is None:
    st.error("Recommendations module did not load correctly. Restart Streamlit.")
    st.stop()

risk_status = risk_label(prediction, proba)

top_drivers = row.get("top_drivers")
if isinstance(top_drivers, str):
    try:
        top_drivers = json.loads(top_drivers)
    except Exception:
        top_drivers = []
if not isinstance(top_drivers, list):
    top_drivers = []

rec = generate_recommendation(row, prediction, proba, top_drivers, threshold)

st.subheader("Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("ID", str(selected_id))
col2.metric("Risk status", risk_status)
col3.metric("Probability", f"{proba:.2f}")
col4.metric("Segment", rec["segment"])

st.subheader("Why")
for item in rec["drivers"]:
    st.write(f"- {item}")

st.subheader("Recommended Actions")
primary = rec["primary"]
st.markdown(
    f"**Primary:** {primary['action']}  \
Owner: {primary['owner']} | Urgency: {primary['urgency']} | Timeframe: {primary['timeframe']}"
)
if rec["secondary"]:
    st.markdown("**Secondary options:**")
    for option in rec["secondary"]:
        st.write(f"- {option}")

st.subheader("Feedback Logging")
notes = st.text_area("Notes (optional)")

col_a, col_b, col_c = st.columns(3)
if col_a.button("Action taken"):
    log_feedback(
        str(LOG_PATH),
        {
            "id": selected_id,
            "predicted_class": prediction,
            "probability": proba,
            "segment": rec["segment"],
            "primary_action": primary["action"],
            "user_feedback": "action_taken",
            "notes": notes,
            "threshold": threshold,
            "model_name": "prediction_v1",
        },
    )
    st.success("Logged action taken.")

if col_b.button("Reviewed ? no action"):
    log_feedback(
        str(LOG_PATH),
        {
            "id": selected_id,
            "predicted_class": prediction,
            "probability": proba,
            "segment": rec["segment"],
            "primary_action": primary["action"],
            "user_feedback": "reviewed_no_action",
            "notes": notes,
            "threshold": threshold,
            "model_name": "prediction_v1",
        },
    )
    st.success("Logged review with no action.")

if col_c.button("Override ? incorrect"):
    log_feedback(
        str(LOG_PATH),
        {
            "id": selected_id,
            "predicted_class": prediction,
            "probability": proba,
            "segment": rec["segment"],
            "primary_action": primary["action"],
            "user_feedback": "override_incorrect",
            "notes": notes,
            "threshold": threshold,
            "model_name": "prediction_v1",
        },
    )
    st.success("Logged override feedback.")
