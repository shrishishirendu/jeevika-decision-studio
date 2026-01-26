from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.cleaning import clean_survey_df
from src.imputation import compute_missing_profile, impute_dataframe
from src import io as io_utils
from src.schema import load_columns_config

COLUMNS_PATH = ROOT / "config" / "columns.yaml"
CONFIG_PATH = ROOT / "config" / "config.yaml"

st.set_page_config(page_title="Data Health", page_icon="Data Health", layout="wide")

st.title("Data Health")
st.caption("Basic dataset statistics and missing value report.")

if not CONFIG_PATH.exists():
    st.error("Missing config/config.yaml. Please add it to continue.")
    st.stop()

with CONFIG_PATH.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

data_config = config.get("data") or {}
raw_path = data_config.get("raw_path")
processed_path = data_config.get("processed_path")

if not raw_path:
    st.error("config.yaml is missing data.raw_path")
    st.stop()

csv_path = ROOT / raw_path
if not csv_path.exists():
    st.error(f"CSV not found: {csv_path}")
    st.stop()

try:
    df = pd.read_csv(csv_path)
except Exception as exc:  # pragma: no cover - streamlit error display
    st.error(f"Failed to load CSV: {exc}")
    st.stop()

if not COLUMNS_PATH.exists():
    st.error("Missing config/columns.yaml. Please add it to continue.")
    st.stop()

schema = load_columns_config(COLUMNS_PATH)
cleaned_df, profile = clean_survey_df(df, schema)

st.subheader("Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{profile['row_count']:,}")
col2.metric("Columns", f"{profile['col_count']:,}")
col3.metric("Missing Cells", f"{int(cleaned_df.isna().sum().sum()):,}")

st.subheader("Preview")
st.dataframe(cleaned_df.head(20), use_container_width=True)

st.subheader("Column Types")
dtype_df = pd.DataFrame(profile["dtypes"])
st.dataframe(dtype_df, use_container_width=True, height=320)

with st.expander("Debug: Column Names", expanded=False):
    st.write("Raw columns:")
    st.write(list(df.columns))
    st.write("Cleaned columns:")
    st.write(list(cleaned_df.columns))

st.subheader("Missing Value Report")
missing_df = pd.DataFrame(profile["missingness"]).sort_values(
    by="missing_count", ascending=False
)
st.dataframe(missing_df.head(20), use_container_width=True, height=400)

st.subheader("Missingness Diagnostics")
missing_profile_df = compute_missing_profile(cleaned_df)
st.dataframe(missing_profile_df, use_container_width=True, height=400)

st.subheader("Imputation Policy")
st.markdown(
    """
- Numeric columns: impute with mean if missing <= 20%; otherwise keep NaN.
- Categorical columns: impute with mode if missing <= 20%; otherwise fill with "Unknown".
- Binary columns: impute with mode if missing <= 20%; otherwise keep NaN.
- All imputed columns get a `<col>__is_imputed` flag.
"""
)

st.subheader("Quality Flags")
flag_counts = profile.get("flag_counts", {})
if flag_counts:
    flag_df = pd.DataFrame(
        [{"flag": key, "count": value} for key, value in flag_counts.items()]
    )
    st.dataframe(flag_df, use_container_width=True, height=200)
else:
    st.info("No quality flags generated for this dataset.")

if "run_id" not in st.session_state:
    st.session_state.run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]

run_dir = ROOT / "outputs" / "runs" / st.session_state.run_id

if st.button("Apply Data Quality Improvements"):
    imputed_df, imputation_report = impute_dataframe(cleaned_df)
    imputed_path = ROOT / "data" / "processed" / "cleaned_imputed.parquet"
    imputed_path.parent.mkdir(parents=True, exist_ok=True)
    imputed_df.to_parquet(imputed_path, index=False)

    report_path = run_dir / "imputation_report.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(imputation_report, indent=2), encoding="utf-8")

    st.success(
        f"Imputed dataset saved to {imputed_path}. Total imputed cells: {imputation_report['total_imputed_cells']}"
    )
    st.info(f"Imputation report saved to {report_path}")

    before_missing = compute_missing_profile(cleaned_df)[
        ["column_name", "missing_pct"]
    ].rename(columns={"missing_pct": "missing_before_pct"})
    after_missing = compute_missing_profile(imputed_df)[
        ["column_name", "missing_pct"]
    ].rename(columns={"missing_pct": "missing_after_pct"})
    merged_missing = before_missing.merge(after_missing, on="column_name", how="left")

    st.subheader("Post-Imputation Validation")
    st.dataframe(merged_missing, use_container_width=True, height=360)

    top_imputed = (
        pd.DataFrame(
            list(imputation_report["per_column_imputed_count"].items()),
            columns=["column", "imputed_count"],
        )
        .sort_values(by="imputed_count", ascending=False)
        .head(10)
    )
    st.markdown("**Top 10 columns by imputed count**")
    st.dataframe(top_imputed, use_container_width=True, height=280)

if st.button("Save cleaned dataset"):
    if not processed_path:
        st.error("config.yaml is missing data.processed_path")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        save_fn = getattr(io_utils, "save_clean_parquet", io_utils.write_parquet)
        save_fn(cleaned_df, ROOT / processed_path)

        profile_path = run_dir / "profile.json"
        profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

        sample_path = run_dir / "cleaned_sample.csv"
        cleaned_df.head(200).to_csv(sample_path, index=False)

        st.success(f"Cleaned dataset saved to {processed_path}")
        st.info(f"Profile saved to {profile_path}")
        st.info(f"Sample saved to {sample_path}")