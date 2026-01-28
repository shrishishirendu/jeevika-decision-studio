from __future__ import annotations

import re
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
from app.utils.overview_plots import (
    get_existing_cols,
    plot_cat_bar,
    plot_coverage_bar,
    plot_hist_box,
    smart_bivariate_plot,
)
from app.utils.layout import render_page_header

CONFIG_PATH = ROOT / "config" / "config.yaml"

st.set_page_config(page_title="Overview", page_icon="Overview", layout="wide")

render_page_header("Overview")
st.caption(
    "Charts-first snapshot of participation, awareness, impact, and sentiment. "
    "Use the filters to focus on specific blocks, tenure bands, or engagement levels."
)

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

# Align raw vs cleaned schema for meeting attendance fields.
if (
    "shg_meetings_attended_monthly" not in filtered_df.columns
    and "meeting_attendance" in filtered_df.columns
):
    filtered_df["shg_meetings_attended_monthly"] = filtered_df["meeting_attendance"]
if (
    "meeting_attendance" not in filtered_df.columns
    and "shg_meetings_attended_monthly" in filtered_df.columns
):
    filtered_df["meeting_attendance"] = filtered_df["shg_meetings_attended_monthly"]

st.sidebar.markdown("### Active Filters")
if selections:
    for key, value in selections.items():
        st.sidebar.write(f"{key}: {value}")
else:
    st.sidebar.write("Showing full dataset (no filters selected)")


def pick_first_existing(df_: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df_.columns:
            return col
    return None


def chart_explainer(what: str, look_for: str) -> None:
    st.caption(f"**What this shows:** {what}  \n**What to look for:** {look_for}")


def plot_discrete_or_hist(df_: pd.DataFrame, col: str, title: str) -> None:
    if col not in df_.columns:
        st.info(f"{col} not available.")
        return
    numeric = pd.to_numeric(df_[col], errors="coerce")
    unique_vals = numeric.dropna().unique()
    if numeric.notna().any() and len(unique_vals) <= 10:
        counts = (
            numeric.dropna()
            .value_counts()
            .sort_index()
            .rename("value_count")
            .reset_index()
            .rename(columns={"index": col})
        )
        fig = px.bar(
            counts,
            x=col,
            y="value_count",
            title=title,
            labels={col: col.replace("_", " ").title(), "value_count": "Responses"},
        )
        median_val = float(numeric.median())
        fig.add_vline(x=median_val, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
        chart_explainer(
            "The count of respondents at each attendance level.",
            "Skew toward low attendance or a tight cluster around a typical value.",
        )
        return

    fig = plot_hist_box(df_, col, title)
    if fig is None:
        st.info(f"{col} not available for plotting.")
        return
    st.plotly_chart(fig, use_container_width=True)
    chart_explainer(
        "The distribution of meeting attendance with the median line.",
        "Long tails or multi-modal patterns that signal uneven participation.",
    )


def build_likert_stack(df_: pd.DataFrame, cols: list[str], title: str) -> bool:
    existing = get_existing_cols(df_, cols)
    if not existing:
        return False

    melted = df_[existing].melt(var_name="question", value_name="response").dropna()
    if melted.empty:
        return False

    melted["response"] = melted["response"].astype("string")
    counts = (
        melted.groupby(["question", "response"]).size().reset_index(name="count")
    )
    totals = counts.groupby("question")["count"].transform("sum")
    counts["percent"] = counts["count"] / totals * 100

    likert_order = [
        "Strongly disagree",
        "Disagree",
        "Neutral",
        "Agree",
        "Strongly agree",
    ]
    responses = counts["response"].unique().tolist()
    ordered = [r for r in likert_order if r in responses]
    if not ordered:
        ordered = sorted(responses)

    counts["response"] = pd.Categorical(counts["response"], categories=ordered, ordered=True)
    counts["question"] = counts["question"].apply(lambda x: x.replace("_", " ").title())

    fig = px.bar(
        counts,
        x="percent",
        y="question",
        color="response",
        orientation="h",
        title=title,
        labels={"percent": "Share of responses (%)", "question": ""},
    )
    st.plotly_chart(fig, use_container_width=True)
    chart_explainer(
        "How sentiment responses distribute across key questions.",
        "Large neutral blocks or polarized responses that suggest mixed experience.",
    )
    return True


def extract_keywords(series: pd.Series, top_n: int = 15) -> pd.DataFrame:
    stopwords = {
        "and",
        "or",
        "the",
        "to",
        "of",
        "in",
        "for",
        "a",
        "an",
        "on",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "it",
        "this",
        "that",
        "as",
        "we",
        "our",
        "us",
        "you",
        "your",
        "they",
        "their",
        "from",
        "by",
        "more",
        "less",
        "not",
        "no",
        "yes",
    }
    tokens = []
    for text in series.dropna().astype(str):
        words = re.findall(r"[A-Za-z]+", text.lower())
        tokens.extend([w for w in words if len(w) > 2 and w not in stopwords])

    if not tokens:
        return pd.DataFrame(columns=["keyword", "count"])

    counts = pd.Series(tokens).value_counts().head(top_n).reset_index()
    counts.columns = ["keyword", "count"]
    return counts


tabs = st.tabs(["Univariate", "Bivariate", "Key Stories"])

with tabs[0]:
    st.subheader("Participation Snapshot")
    st.caption("Baseline view of tenure, geography, and meeting attendance.")

    col1, col2 = st.columns(2)
    with col1:
        fig = plot_hist_box(
            filtered_df,
            "membership_duration_months",
            "Membership Duration (months)",
        )
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            chart_explainer(
                "The spread of member tenure and the median duration.",
                "Whether new members dominate or tenure is broadly distributed.",
            )
        else:
            st.info("membership_duration_months not available.")
    with col2:
        fig = plot_cat_bar(filtered_df, "village_block", "Village/Block Distribution", top_n=12)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            chart_explainer(
                "Which blocks contribute the most respondents.",
                "Over-representation or thin coverage in specific blocks.",
            )
        else:
            st.info("village_block not available.")

    plot_discrete_or_hist(
        filtered_df,
        "shg_meetings_attended_monthly",
        "SHG Meetings Attended (monthly)",
    )

    st.subheader("Engagement Participation Mix")
    st.caption("Coverage of key participation activities across the sample.")
    participation_cols = [
        "savings_participation",
        "credit_participation",
        "agricultural_intervention",
        "livestock_activity",
        "nonfarm_enterprise",
        "didi_ki_rasoi",
        "producer_groups",
        "vo_participation",
        "income_generating_activity_frequency",
        "loan_access_knowledge",
    ]
    participation_cols = get_existing_cols(filtered_df, participation_cols)
    participation_plot = plot_coverage_bar(
        filtered_df, participation_cols, "Participation Coverage", label="% Active"
    )
    if participation_plot:
        fig, _ = participation_plot
        st.plotly_chart(fig, use_container_width=True)
        chart_explainer(
            "The share of respondents actively participating in each intervention.",
            "Programs with low uptake that may need targeted outreach.",
        )
    else:
        st.info("No binary participation columns available for coverage chart.")

    st.subheader("Awareness Coverage")
    st.caption("Visibility of Jeevika benefits and trainings.")
    awareness_cols = [
        "mukhyamantri_awareness",
        "bank_linkage_awareness",
        "agricultural_training_awareness",
        "livestock_training_awareness",
        "skill_training_awareness",
        "didi_ki_rasoi_awareness",
        "gender_rights_awareness",
        "health_nutrition_awareness",
        "welfare_linkage_awareness",
    ]
    awareness_cols = get_existing_cols(filtered_df, awareness_cols)
    awareness_plot = plot_coverage_bar(
        filtered_df, awareness_cols, "Awareness Coverage", label="% Aware"
    )
    if awareness_plot:
        fig, _ = awareness_plot
        st.plotly_chart(fig, use_container_width=True)
        chart_explainer(
            "The share of respondents aware of each benefit or training.",
            "Blind spots where awareness is weak compared with core programs.",
        )
    else:
        st.info("No binary awareness columns available for coverage chart.")

    st.subheader("Impact & Sentiment Distributions")
    st.caption("Outcome signals on income change, perceived impact, and sentiment.")

    fig = plot_hist_box(
        filtered_df,
        "household_income_increase_percent",
        "Household Income Increase (%)",
    )
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        chart_explainer(
            "The distribution of reported income change with a median marker.",
            "How much of the sample reports meaningful positive change.",
        )
    else:
        st.info("household_income_increase_percent not available.")

    impact_col = pick_first_existing(
        filtered_df,
        ["overall_impact_rating", "impact_rating", "overall_impact"],
    )
    if impact_col:
        series = filtered_df[impact_col]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            fig = px.histogram(
                filtered_df,
                x=impact_col,
                title="Overall Impact Rating",
                labels={impact_col: "Impact Rating"},
            )
        else:
            counts = series.astype("string").value_counts().reset_index()
            counts.columns = [impact_col, "count"]
            fig = px.bar(
                counts,
                x=impact_col,
                y="count",
                title="Overall Impact Rating",
                labels={impact_col: "Impact Rating", "count": "Responses"},
            )
        st.plotly_chart(fig, use_container_width=True)
        chart_explainer(
            "How respondents rate the overall program impact.",
            "Whether ratings skew toward the positive end or cluster in the middle.",
        )
    else:
        st.info("overall_impact_rating not available.")

    sentiment_cols = ["satisfaction_level", "future_program_confidence", "referral_likelihood"]
    if not build_likert_stack(filtered_df, sentiment_cols, "Sentiment Snapshot"):
        for col in get_existing_cols(filtered_df, sentiment_cols):
            counts = filtered_df[col].astype("string").value_counts().reset_index()
            counts.columns = [col, "count"]
            fig = px.bar(
                counts,
                x=col,
                y="count",
                title=col.replace("_", " ").title(),
                labels={col: col.replace("_", " ").title(), "count": "Responses"},
            )
            st.plotly_chart(fig, use_container_width=True)
            chart_explainer(
                f"Distribution of {col.replace('_', ' ')} responses.",
                "Skew toward favorable or unfavorable responses.",
            )

    st.subheader("Single Best Improvement")
    st.caption("Most common themes from open-ended feedback.")
    improvement_col = pick_first_existing(
        filtered_df,
        ["single_best_improvement", "best_improvement", "improvement_suggestion", "improvement_area"],
    )
    if improvement_col:
        keywords_df = extract_keywords(filtered_df[improvement_col])
        if not keywords_df.empty:
            fig = px.bar(
                keywords_df,
                x="count",
                y="keyword",
                orientation="h",
                title="Top Improvement Keywords",
                labels={"count": "Mentions", "keyword": ""},
            )
            fig.update_layout(yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig, use_container_width=True)
            chart_explainer(
                "The most common keywords in improvement suggestions.",
                "Repeated themes that signal priority areas for action.",
            )
            with st.expander("Sample responses", expanded=False):
                sample = (
                    filtered_df[improvement_col]
                    .dropna()
                    .sample(n=min(10, filtered_df[improvement_col].dropna().shape[0]), random_state=42)
                )
                st.dataframe(sample.to_frame(name=improvement_col), use_container_width=True)
        else:
            st.info("No keyword signals found in improvement responses.")
    else:
        st.info("single_best_improvement not available.")

with tabs[1]:
    st.subheader("Curated Highlights")
    st.caption("Targeted bivariate views for key engagement and outcome relationships.")

    if "engagement_level" in filtered_df.columns and "overall_impact_rating" in filtered_df.columns:
        impact_series = pd.to_numeric(filtered_df["overall_impact_rating"], errors="coerce")
        if impact_series.notna().any():
            fig = px.box(
                filtered_df,
                x="engagement_level",
                y="overall_impact_rating",
                title="Overall Impact by Engagement Level",
                labels={
                    "engagement_level": "Engagement Level",
                    "overall_impact_rating": "Overall Impact Rating",
                },
            )
        else:
            ctab = pd.crosstab(
                filtered_df["engagement_level"],
                filtered_df["overall_impact_rating"],
                normalize="index",
            ) * 100
            fig = px.imshow(
                ctab,
                text_auto=True,
                aspect="auto",
                title="Impact Rating by Engagement Level (% within level)",
                labels={"color": "%"},
            )
        st.plotly_chart(fig, use_container_width=True)
        chart_explainer(
            "How perceived impact varies across engagement levels.",
            "Upward shifts in ratings as engagement increases.",
        )

    clarity_col = pick_first_existing(
        filtered_df,
        ["scheme_clarity_level", "scheme_clarity", "clarity_level", "scheme_clarity_rating"],
    )
    if "scheme_awareness_count" in filtered_df.columns and clarity_col:
        awareness_numeric = pd.to_numeric(filtered_df["scheme_awareness_count"], errors="coerce")
        clarity_series = filtered_df[clarity_col].astype("string")
        if awareness_numeric.notna().any():
            bins = pd.qcut(
                awareness_numeric.dropna(),
                q=min(5, awareness_numeric.nunique()),
                duplicates="drop",
            )
            bin_labels = bins.astype("string")
            temp = pd.DataFrame(
                {
                    "awareness_band": bin_labels,
                    "clarity": clarity_series.loc[bin_labels.index],
                }
            )
            ctab = pd.crosstab(temp["clarity"], temp["awareness_band"], normalize="index") * 100
            fig = px.imshow(
                ctab,
                text_auto=True,
                aspect="auto",
                title="Scheme Awareness vs Clarity (% within clarity)",
                labels={"color": "%"},
            )
            st.plotly_chart(fig, use_container_width=True)
            chart_explainer(
                "Whether higher awareness aligns with clearer understanding of schemes.",
                "Clarity groups with low awareness bands indicate a gap.",
            )

    if "shg_meetings_attended_monthly" in filtered_df.columns and "satisfaction_level" in filtered_df.columns:
        attendance_numeric = pd.to_numeric(filtered_df["shg_meetings_attended_monthly"], errors="coerce")
        if attendance_numeric.notna().any():
            fig = px.box(
                filtered_df,
                x="satisfaction_level",
                y="shg_meetings_attended_monthly",
                title="Meeting Attendance by Satisfaction Level",
                labels={
                    "satisfaction_level": "Satisfaction Level",
                    "shg_meetings_attended_monthly": "Meetings Attended (monthly)",
                },
            )
            st.plotly_chart(fig, use_container_width=True)
            chart_explainer(
                "How attendance varies across satisfaction responses.",
                "Whether more satisfied respondents attend more meetings.",
            )

    if "membership_duration_months" in filtered_df.columns and "engagement_level" in filtered_df.columns:
        duration_numeric = pd.to_numeric(filtered_df["membership_duration_months"], errors="coerce")
        if duration_numeric.notna().any():
            fig = px.box(
                filtered_df,
                x="engagement_level",
                y="membership_duration_months",
                title="Membership Duration by Engagement Level",
                labels={
                    "engagement_level": "Engagement Level",
                    "membership_duration_months": "Membership Duration (months)",
                },
            )
            st.plotly_chart(fig, use_container_width=True)
            chart_explainer(
                "Whether longer membership tenure aligns with higher engagement.",
                "Groups where long-tenure members are still low engagement.",
            )

    st.subheader("Smart Bivariate Explorer")
    st.caption("Choose any two variables and the chart adapts based on data types.")

    allowed_cols = [
        "membership_duration_months",
        "village_block",
        "shg_meetings_attended_monthly",
        "savings_participation",
        "credit_participation",
        "agricultural_intervention",
        "livestock_activity",
        "nonfarm_enterprise",
        "didi_ki_rasoi",
        "producer_groups",
        "vo_participation",
        "engagement_level",
        "income_generating_activity_frequency",
        "loan_access_knowledge",
        "scheme_awareness_count",
        "mukhyamantri_awareness",
        "bank_linkage_awareness",
        "agricultural_training_awareness",
        "livestock_training_awareness",
        "skill_training_awareness",
        "didi_ki_rasoi_awareness",
        "gender_rights_awareness",
        "health_nutrition_awareness",
        "welfare_linkage_awareness",
        clarity_col or "",
        "household_income_increase_percent",
        "overall_impact_rating",
        "satisfaction_level",
        "future_program_confidence",
        "referral_likelihood",
    ]
    allowed_cols = [col for col in allowed_cols if col and col in filtered_df.columns]
    if len(allowed_cols) >= 2:
        col1, col2 = st.columns(2)
        x_choice = col1.selectbox("X-axis", allowed_cols, index=0)
        y_choice = col2.selectbox(
            "Y-axis", [c for c in allowed_cols if c != x_choice], index=0
        )
        fig, explanation, medians = smart_bivariate_plot(filtered_df, x_choice, y_choice)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            chart_explainer(
                "Relationship between the two selected variables.",
                explanation,
            )
        else:
            st.info(explanation)
        if medians is not None and not medians.empty:
            st.markdown("**Group medians**")
            st.dataframe(medians, use_container_width=True, height=220)
    else:
        st.info("Not enough columns available for the bivariate explorer.")

with tabs[2]:
    st.subheader("Key Stories")
    st.caption("Auto-generated highlights based on simple distribution rules.")

    stories = []
    if participation_plot:
        _, pdata = participation_plot
        if not pdata.empty:
            top = pdata.sort_values("positive_rate", ascending=False).iloc[0]
            low = pdata.sort_values("positive_rate", ascending=True).iloc[0]
            stories.append(
                f"Highest participation: {top['variable']} ({top['positive_rate']:.1f}%)."
            )
            stories.append(
                f"Lowest participation: {low['variable']} ({low['positive_rate']:.1f}%)."
            )

    if awareness_plot:
        _, adata = awareness_plot
        if not adata.empty:
            top = adata.sort_values("positive_rate", ascending=False).iloc[0]
            low = adata.sort_values("positive_rate", ascending=True).iloc[0]
            stories.append(
                f"Most recognized benefit: {top['variable']} ({top['positive_rate']:.1f}%)."
            )
            stories.append(
                f"Least recognized benefit: {low['variable']} ({low['positive_rate']:.1f}%)."
            )

    if "household_income_increase_percent" in filtered_df.columns:
        income = pd.to_numeric(filtered_df["household_income_increase_percent"], errors="coerce")
        if income.notna().any():
            median_income = income.median()
            positive_share = (income > 0).mean() * 100
            stories.append(
                f"Median reported income increase: {median_income:.1f}%."
            )
            stories.append(
                f"{positive_share:.1f}% report a positive income change."
            )

    if "satisfaction_level" in filtered_df.columns:
        top_val = filtered_df["satisfaction_level"].astype("string").value_counts().idxmax()
        stories.append(f"Most common satisfaction response: {top_val}.")

    if stories:
        for story in stories:
            st.write(f"- {story}")
    else:
        st.info("Not enough data available to generate highlights.")
