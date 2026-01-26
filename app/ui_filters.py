from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import streamlit as st

from src.cleaning import clean_survey_df
from src.mock_data import save_mock_csv
from src.schema import load_columns_config


DATA_SOURCE_OPTIONS = ["REAL (cleaned.parquet)", "DEMO (mock)"]


def load_clean_df(processed_path: str, raw_path: str | None = None) -> pd.DataFrame:
    """Load cleaned parquet, or clean raw CSV if parquet is missing."""
    data_source = st.sidebar.selectbox("Data source", DATA_SOURCE_OPTIONS, key="data_source")

    if data_source == "DEMO (mock)":
        mock_path = save_mock_csv()
        df = pd.read_csv(mock_path)
        schema = load_columns_config()
        cleaned, _ = clean_survey_df(df, schema)
        return cleaned

    processed = Path(processed_path)
    if processed.exists():
        return pd.read_parquet(processed)

    if not raw_path:
        raise FileNotFoundError(f"Processed parquet not found: {processed}")

    raw = Path(raw_path)
    if not raw.exists():
        raise FileNotFoundError(f"Raw CSV not found: {raw}")

    df = pd.read_csv(raw)
    schema = load_columns_config()
    cleaned, _ = clean_survey_df(df, schema)
    return cleaned


def apply_filters(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Apply sidebar filters if supported columns exist."""
    selected: Dict[str, object] = {}
    filtered = df.copy()

    schema = load_columns_config()
    geo_col = schema.geography_col or "village_or_block"

    if geo_col in filtered.columns:
        options = sorted(filtered[geo_col].dropna().unique().tolist())
        choices = st.sidebar.multiselect("Village / Block", options, default=options)
        selected[geo_col] = choices
        if choices:
            filtered = filtered[filtered[geo_col].isin(choices)]

    duration_col = "membership_duration"
    if duration_col in filtered.columns:
        if pd.api.types.is_numeric_dtype(filtered[duration_col]):
            min_val = float(filtered[duration_col].min())
            max_val = float(filtered[duration_col].max())
            if min_val != max_val:
                picked = st.sidebar.slider(
                    "Membership Duration",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                )
                selected[duration_col] = picked
                filtered = filtered[
                    (filtered[duration_col] >= picked[0])
                    & (filtered[duration_col] <= picked[1])
                ]
        else:
            options = sorted(filtered[duration_col].dropna().unique().tolist())
            choices = st.sidebar.multiselect(
                "Membership Duration", options, default=options
            )
            selected[duration_col] = choices
            if choices:
                filtered = filtered[filtered[duration_col].isin(choices)]

    income_col = "pre_jeevika_income_bucket_clean"
    if income_col in filtered.columns:
        options = sorted(filtered[income_col].dropna().unique().tolist())
        choices = st.sidebar.multiselect("Pre-Jeevika Income", options, default=options)
        selected[income_col] = choices
        if choices:
            filtered = filtered[filtered[income_col].isin(choices)]

    return filtered, selected