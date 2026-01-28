from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import streamlit as st

from src.cleaning import clean_survey_df
from src.mock_data import save_mock_csv
from src.schema import load_columns_config


DATA_SOURCE_OPTIONS = ["REAL (cleaned.parquet)", "DEMO (mock)"]


def normalize_filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    aliases = {
        "district_name": "district",
        "district_name_clean": "district",
        "block_name": "village_block",
        "block": "village_block",
        "village_or_block": "village_block",
        "village_block_name": "village_block",
        "shg_meetings_attended_monthly": "meeting_attendance",
        "engagement_level": "involvement_level",
        "membership_duration_months": "membership_duration",
        "membership_duration_years": "membership_duration",
        "pre_jeevika_income_bucket": "pre_jeevika_income_bucket_clean",
    }
    for src, tgt in aliases.items():
        if tgt not in df.columns and src in df.columns:
            df[tgt] = df[src]
    return df


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
    filtered = normalize_filter_columns(df)

    schema = load_columns_config()
    geo_col = schema.geography_col or "village_or_block"

    if geo_col in filtered.columns:
        series = filtered[geo_col].astype("string").fillna("Unknown")
        options = sorted(series.unique().tolist())
        choices = st.sidebar.multiselect(
            "Village / Block",
            options,
            default=options,
            key="filter_village_block",
        )
        if choices and len(choices) != len(options):
            selected["Village / Block"] = choices
            filtered = filtered[series.isin(choices)]

    if "district" in filtered.columns:
        series = filtered["district"].astype("string").fillna("Unknown")
        options = sorted(series.unique().tolist())
        choices = st.sidebar.multiselect(
            "District",
            options,
            default=options,
            key="filter_district",
        )
        if choices and len(choices) != len(options):
            selected["district"] = choices
            filtered = filtered[series.isin(choices)]

    duration_col = "membership_duration"
    if duration_col in filtered.columns:
        duration_numeric = pd.to_numeric(filtered[duration_col], errors="coerce")
        if duration_numeric.notna().any():
            min_val = float(duration_numeric.min())
            max_val = float(duration_numeric.max())
            if min_val != max_val:
                picked = st.sidebar.slider(
                    "Membership Duration",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    key="filter_membership_duration",
                )
                if picked[0] != min_val or picked[1] != max_val:
                    selected["Membership Duration"] = picked
                filtered = filtered[
                    (duration_numeric >= picked[0]) & (duration_numeric <= picked[1])
                ]
        else:
            series = filtered[duration_col].astype("string").fillna("Unknown")
            options = sorted(series.unique().tolist())
            choices = st.sidebar.multiselect(
                "Membership Duration",
                options,
                default=options,
                key="filter_membership_duration_cat",
            )
            if choices and len(choices) != len(options):
                selected["Membership Duration"] = choices
                filtered = filtered[series.isin(choices)]

    income_col = "pre_jeevika_income_bucket_clean"
    if income_col in filtered.columns:
        series = filtered[income_col].astype("string").fillna("Unknown")
        options = sorted(series.unique().tolist())
        choices = st.sidebar.multiselect(
            "Pre-Jeevika Income",
            options,
            default=options,
            key="filter_pre_jeevika_income",
        )
        if choices and len(choices) != len(options):
            selected["Pre-Jeevika Income"] = choices
            filtered = filtered[series.isin(choices)]

    with st.sidebar.expander("Debug: Filters (temporary)", expanded=False):
        st.write("Columns available:", sorted(filtered.columns.tolist())[:80])
        st.write("Selections dict:", selected)
        st.write("Filtered rows:", len(filtered))

    return filtered, selected
