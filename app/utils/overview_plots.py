from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


LIKERT_TEXT = {
    "strongly disagree",
    "disagree",
    "neutral",
    "agree",
    "strongly agree",
    "very dissatisfied",
    "dissatisfied",
    "satisfied",
    "very satisfied",
    "very low",
    "low",
    "medium",
    "high",
    "very high",
}


def get_existing_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [col for col in cols if col in df.columns]


def apply_filters(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    filtered = df.copy()
    selections: Dict[str, object] = {}

    col1, col2, col3 = st.columns(3)

    if "village_block" in filtered.columns:
        options = sorted(filtered["village_block"].dropna().unique().tolist())
        choices = col1.multiselect("Village / Block", options, default=options)
        selections["village_block"] = choices
        if choices:
            filtered = filtered[filtered["village_block"].isin(choices)]

    if "membership_duration_months" in filtered.columns:
        series = filtered["membership_duration_months"]
        if pd.api.types.is_numeric_dtype(series):
            min_val = float(pd.to_numeric(series, errors="coerce").min())
            max_val = float(pd.to_numeric(series, errors="coerce").max())
            if np.isfinite(min_val) and np.isfinite(max_val) and min_val != max_val:
                picked = col2.slider(
                    "Membership Duration (months)",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                )
                selections["membership_duration_months"] = picked
                filtered = filtered[
                    (pd.to_numeric(filtered["membership_duration_months"], errors="coerce") >= picked[0])
                    & (pd.to_numeric(filtered["membership_duration_months"], errors="coerce") <= picked[1])
                ]
        else:
            options = sorted(series.dropna().unique().tolist())
            choices = col2.multiselect("Membership Duration (months)", options, default=options)
            selections["membership_duration_months"] = choices
            if choices:
                filtered = filtered[filtered["membership_duration_months"].isin(choices)]

    if "engagement_level" in filtered.columns:
        options = sorted(filtered["engagement_level"].dropna().unique().tolist())
        choices = col3.multiselect("Engagement Level", options, default=options)
        selections["engagement_level"] = choices
        if choices:
            filtered = filtered[filtered["engagement_level"].isin(choices)]

    return filtered, selections


def _normalize_binary_value(value: object) -> Optional[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    if isinstance(value, (np.bool_, bool)):
        return 1 if bool(value) else 0

    if isinstance(value, (int, np.integer)) and value in {0, 1}:
        return int(value)

    if isinstance(value, (float, np.floating)) and value in {0.0, 1.0}:
        return int(value)

    if isinstance(value, str):
        val = value.strip().lower()
        if val in {"yes", "y", "true", "1"}:
            return 1
        if val in {"no", "n", "false", "0"}:
            return 0
    return None


def binary_positive_rate(series: pd.Series) -> Optional[float]:
    if series.dropna().empty:
        return None

    normalized = series.dropna().apply(_normalize_binary_value)
    if normalized.isna().mean() > 0.2:
        return None

    unique_vals = set(normalized.dropna().unique().tolist())
    if not unique_vals.issubset({0, 1}):
        return None

    return float(normalized.mean() * 100)


def plot_coverage_bar(
    df: pd.DataFrame, cols: List[str], title: str, label: str = "% Active"
) -> Optional[Tuple[px.Figure, pd.DataFrame]]:
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        rate = binary_positive_rate(df[col])
        if rate is None:
            continue
        rows.append({"variable": col, "positive_rate": rate})

    if not rows:
        return None

    plot_df = pd.DataFrame(rows).sort_values("positive_rate", ascending=True)
    fig = px.bar(
        plot_df,
        x="positive_rate",
        y="variable",
        orientation="h",
        title=title,
        labels={"positive_rate": label, "variable": ""},
        text="positive_rate",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    return fig, plot_df


def plot_hist_box(df: pd.DataFrame, col: str, title: str) -> Optional[px.Figure]:
    if col not in df.columns:
        return None
    numeric = pd.to_numeric(df[col], errors="coerce")
    if numeric.dropna().empty:
        return None
    fig = px.histogram(
        df,
        x=col,
        title=title,
        marginal="box",
        labels={col: col.replace("_", " ").title()},
    )
    median_val = float(numeric.median())
    fig.add_vline(x=median_val, line_dash="dash", line_color="gray")
    return fig


def plot_cat_bar(
    df: pd.DataFrame, col: str, title: str, top_n: int = 15
) -> Optional[px.Figure]:
    if col not in df.columns:
        return None
    counts = df[col].astype("string").value_counts(dropna=True)
    if counts.empty:
        return None

    if len(counts) > top_n:
        top_counts = counts.head(top_n)
        others = counts.iloc[top_n:].sum()
        top_counts["Others"] = others
        counts = top_counts

    plot_df = counts.reset_index()
    plot_df.columns = [col, "count"]
    fig = px.bar(
        plot_df,
        x="count",
        y=col,
        orientation="h",
        title=title,
        labels={"count": "Responses", col: ""},
    )
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    return fig


def _is_likert(series: pd.Series) -> bool:
    if series.dropna().empty:
        return False
    unique_vals = series.dropna().astype("string").str.lower().unique().tolist()
    if len(unique_vals) <= 7:
        return True
    return any(val in LIKERT_TEXT for val in unique_vals)


def _infer_series_type(series: pd.Series) -> str:
    if series.dropna().empty:
        return "unknown"

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_ratio = numeric.notna().mean()
    if numeric_ratio >= 0.6 and numeric.nunique(dropna=True) > 2:
        return "numeric"
    if _is_likert(series):
        return "categorical"
    return "categorical"


def smart_bivariate_plot(
    df: pd.DataFrame, x: str, y: str
) -> Tuple[Optional[px.Figure], str, Optional[pd.DataFrame]]:
    if x not in df.columns or y not in df.columns:
        return None, "Selected columns not available.", None

    x_type = _infer_series_type(df[x])
    y_type = _infer_series_type(df[y])
    subset = df[[x, y]].dropna()
    if subset.empty:
        return None, "Not enough data after filtering.", None

    if x_type == "numeric" and y_type == "numeric":
        fig = px.scatter(
            subset,
            x=x,
            y=y,
            trendline="ols",
            title=f"{x.replace('_', ' ').title()} vs {y.replace('_', ' ').title()}",
        )
        corr_val = subset[[x, y]].corr(method="spearman").iloc[0, 1]
        explanation = f"Spearman correlation: {corr_val:.2f} (higher absolute values suggest stronger association)."
        return fig, explanation, None

    if x_type == "numeric" and y_type == "categorical":
        fig = px.box(
            subset,
            x=y,
            y=x,
            title=f"{x.replace('_', ' ').title()} by {y.replace('_', ' ').title()}",
            labels={x: x.replace("_", " ").title(), y: y.replace("_", " ").title()},
        )
        medians = (
            subset.groupby(y)[x]
            .median()
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={x: "median"})
        )
        explanation = "Compare medians and spread across groups to spot gaps."
        return fig, explanation, medians

    if x_type == "categorical" and y_type == "numeric":
        fig = px.box(
            subset,
            x=x,
            y=y,
            title=f"{y.replace('_', ' ').title()} by {x.replace('_', ' ').title()}",
            labels={x: x.replace("_", " ").title(), y: y.replace("_", " ").title()},
        )
        medians = (
            subset.groupby(x)[y]
            .median()
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={y: "median"})
        )
        explanation = "Compare medians and spread across groups to spot gaps."
        return fig, explanation, medians

    crosstab = pd.crosstab(subset[y], subset[x], normalize="index") * 100
    fig = px.imshow(
        crosstab,
        text_auto=True,
        aspect="auto",
        title=f"{y.replace('_', ' ').title()} by {x.replace('_', ' ').title()} (% within {y})",
        labels={"color": "%"},
    )
    explanation = "Rows are normalized to show within-group composition."
    return fig, explanation, None
