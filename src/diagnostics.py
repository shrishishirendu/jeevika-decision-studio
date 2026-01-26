from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


def infer_binary_columns(
    df: pd.DataFrame,
    include_prefixes: Sequence[str] = ("barrier_",),
    include_names: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> List[str]:
    prefixes = tuple(include_prefixes)
    include_set = set(include_names)
    exclude_set = set(exclude)

    cols = [
        col
        for col in df.columns
        if (col.startswith(prefixes) or col in include_set) and col not in exclude_set
    ]
    return cols


def coerce_binary_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)

    if pd.api.types.is_numeric_dtype(s):
        coerced = pd.to_numeric(s, errors="coerce")
        return coerced.where(coerced.isin([0, 1]))

    lower = s.astype("string").str.strip().str.lower()
    mapping = {
        "yes": 1.0,
        "y": 1.0,
        "true": 1.0,
        "1": 1.0,
        "no": 0.0,
        "n": 0.0,
        "false": 0.0,
        "0": 0.0,
    }
    mapped = lower.map(mapping)
    return pd.to_numeric(mapped, errors="coerce")


def binary_profile(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        series = coerce_binary_series(df[col])
        non_null = series.notna().sum()
        total = len(series)
        ones = (series == 1).sum()
        zeros = (series == 0).sum()
        sample_vals = df[col].dropna().astype(str).unique().tolist()[:5]
        rows.append(
            {
                "column": col,
                "non_null_pct": round(non_null / total * 100, 2) if total else 0.0,
                "unique_values_sample": sample_vals,
                "ones_pct": round(ones / non_null * 100, 2) if non_null else 0.0,
                "zeros_pct": round(zeros / non_null * 100, 2) if non_null else 0.0,
                "nan_pct": round((total - non_null) / total * 100, 2) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def numeric_profile(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        non_null = series.notna().sum()
        total = len(series)
        if non_null:
            rows.append(
                {
                    "column": col,
                    "non_null_pct": round(non_null / total * 100, 2),
                    "min": float(series.min()),
                    "p25": float(series.quantile(0.25)),
                    "median": float(series.median()),
                    "p75": float(series.quantile(0.75)),
                    "max": float(series.max()),
                    "mean": float(series.mean()),
                    "std": float(series.std(ddof=0)),
                }
            )
        else:
            rows.append(
                {
                    "column": col,
                    "non_null_pct": 0.0,
                    "min": np.nan,
                    "p25": np.nan,
                    "median": np.nan,
                    "p75": np.nan,
                    "max": np.nan,
                    "mean": np.nan,
                    "std": np.nan,
                }
            )
    return pd.DataFrame(rows)


def categorical_profile(df: pd.DataFrame, cols: List[str], top_n: int = 10) -> Dict[str, Dict[str, int]]:
    result: Dict[str, Dict[str, int]] = {}
    for col in cols:
        if col not in df.columns:
            continue
        counts = df[col].astype("string").value_counts(dropna=False).head(top_n)
        result[col] = {str(idx): int(val) for idx, val in counts.items()}
    return result