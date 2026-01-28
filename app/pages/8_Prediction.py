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
from src import ml_high_uplift as ml_high_uplift
from src import recommendations as recommendations
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

st.set_page_config(page_title="High Uplift Prediction", page_icon="High Uplift", layout="wide")

render_page_header("High Uplift Prediction")

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

threshold = st.slider("Uplift threshold (%)", min_value=10, max_value=30, value=20)
model_choice = st.selectbox("Model", ["LogisticRegression", "RandomForest"])

if "household_income_increase_percent" not in filtered_df.columns:
    st.info("Income column missing; prediction cannot run.")
    st.stop()

X_preview, _, _ = ml_high_uplift.build_features(filtered_df, return_meta=False)
if not isinstance(X_preview, pd.DataFrame):
    st.error("Feature builder returned unexpected output.")
    st.stop()
missingness = X_preview.isna().mean().sort_values(ascending=False) * 100
if not missingness.empty:
    st.markdown("**Top 10 missing feature columns**")
    st.dataframe(missingness.head(10).reset_index().rename(columns={"index": "feature", 0: "missing_pct"}))

@st.cache_data(show_spinner=False)
def train_cached(data: pd.DataFrame, threshold_value: int):
    y = ml_high_uplift.make_target(data, threshold=threshold_value)
    X, numeric_cols, categorical_cols = ml_high_uplift.build_features(data, return_meta=True)
    if numeric_cols is None or categorical_cols is None:
        raise ValueError("Feature builder did not return metadata.")
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]
    results = ml_high_uplift.train_models(X, y, numeric_cols, categorical_cols)
    return results, int(mask.sum()), int((~mask).sum())

results, used_rows, dropped_rows = train_cached(filtered_df, threshold)

metrics = results.metrics.get(model_choice, {})
confusion = results.confusion.get(model_choice, [])

st.subheader("Model Performance")
st.caption(f"Rows used for training: {used_rows} | Dropped due to missing target: {dropped_rows}")
metric_df = pd.DataFrame([metrics])
if not metric_df.empty:
    st.dataframe(metric_df, use_container_width=True, height=120)

if confusion:
    st.markdown("**Confusion Matrix (threshold=0.5)**")
    st.dataframe(pd.DataFrame(confusion, columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"]))

fi_df = results.feature_importance.get(model_choice)
if fi_df is not None and not fi_df.empty:
    st.subheader("Feature Importance (Top 15)")
    top_fi = fi_df.head(15)
    fig = px.bar(top_fi, x="importance", y="feature", orientation="h")
    st.plotly_chart(fig, use_container_width=True)

model = results.models.get(model_choice)
if model is None:
    st.stop()

if not hasattr(ml_high_uplift, "predict_with_model"):
    st.error("Prediction helper not available. Restart Streamlit to reload code.")
    st.stop()

pred_probs, pred_classes = ml_high_uplift.predict_with_model(filtered_df, model, threshold=0.5)

scored_df = filtered_df.copy()
scored_df["segment_label"] = assign_population_segments(scored_df)
scored_df["predicted_prob"] = pred_probs
scored_df["predicted_class"] = pred_classes
risk_label = getattr(recommendations, "risk_label", None)
if risk_label is None:
    st.error("Recommendations module did not load correctly. Restart Streamlit.")
    st.stop()

scored_df["risk_status"] = scored_df.apply(
    lambda row: risk_label(
        int(row.get("predicted_class", 0)) if pd.notna(row.get("predicted_class")) else 0,
        float(row.get("predicted_prob", 0)) if pd.notna(row.get("predicted_prob")) else None,
    ),
    axis=1,
)

st.subheader("Top Candidates")
cols = ["respondent_id", "district", "segment_label", "risk_status", "predicted_prob", "predicted_class"]
if "household_income_increase_percent" in scored_df.columns:
    cols.append("household_income_increase_percent")

candidate_df = scored_df[cols].sort_values(by="predicted_prob", ascending=False).head(50)
st.dataframe(candidate_df, use_container_width=True, height=360)

with st.expander("Debug: Prediction distribution", expanded=False):
    st.write("Predicted class counts:", scored_df["predicted_class"].value_counts(dropna=False))
    st.write(
        "Unique probabilities (rounded):",
        scored_df["predicted_prob"].round(4).nunique(),
    )

st.subheader("District Summary")
if "district" in scored_df.columns:
    district_summary = (
        scored_df.groupby("district")
        .agg(
            n=("district", "size"),
            avg_predicted_prob=("predicted_prob", "mean"),
            pct_above=("predicted_prob", lambda s: safe_float((s >= 0.7).mean() * 100)),
        )
        .reset_index()
    )
    st.dataframe(district_summary, use_container_width=True, height=320)
else:
    st.info("District column not available for summary.")

if st.button("Save predictions dataset"):
    output_path = ROOT / "data" / "processed" / "predicted_high_uplift.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_parquet(output_path, index=False)
    st.success(f"Predictions saved to {output_path}")
