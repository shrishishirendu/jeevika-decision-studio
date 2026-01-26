from __future__ import annotations

import pandas as pd


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def encode_likert_5(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    values = series.astype("string").str.strip().str.lower()
    mapping = {
        "very dissatisfied": 1,
        "dissatisfied": 2,
        "neutral": 3,
        "satisfied": 4,
        "very satisfied": 5,
        "not confident at all": 1,
        "slightly confident": 2,
        "moderately confident": 3,
        "very confident": 4,
        "extremely confident": 5,
        "strongly disagree": 1,
        "disagree": 2,
        "neither agree nor disagree": 3,
        "agree": 4,
        "strongly agree": 5,
    }
    mapped = values.map(mapping)
    if mapped.notna().any():
        return mapped

    return pd.to_numeric(values, errors="coerce")


def encode_referral(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    values = series.astype("string").str.strip().str.lower()
    mapping = {
        "very unlikely": 1,
        "unlikely": 3,
        "neutral": 5,
        "likely": 7,
        "very likely": 9,
    }
    mapped = values.map(mapping)
    if mapped.notna().any():
        return mapped

    return pd.to_numeric(values, errors="coerce")