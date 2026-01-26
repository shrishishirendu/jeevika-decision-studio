from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from src.reporting import build_executive_narration


@dataclass
class ReportAgent:
    """Create a lightweight executive report and narration text."""

    def run(self, state: Dict) -> Dict:
        df: Optional[pd.DataFrame] = state.get("df")
        if df is None:
            raise ValueError("state['df'] is required for ReportAgent")

        row_count = len(df)
        col_count = len(df.columns)

        missing_counts = df.isna().sum().sort_values(ascending=False)
        missing_pct = (missing_counts / max(row_count, 1) * 100).round(2)
        missing_cols = missing_counts[missing_counts > 0]

        if missing_cols.empty:
            missing_highlights = "No missing values were detected in the current dataset."
        else:
            top_missing = missing_cols.head(3)
            parts = []
            for col in top_missing.index:
                parts.append(f"{col} ({missing_pct[col]}%)")
            missing_highlights = (
                "The highest missingness appears in " + ", ".join(parts) + "."
            )

        engagement_summary = _summarize_metric(df, ["engagement", "attendance", "involvement"])
        satisfaction_summary = _summarize_metric(df, ["satisfaction", "sentiment"])

        report_markdown = _build_report_markdown(
            row_count=row_count,
            col_count=col_count,
            missing_highlights=missing_highlights,
            engagement_summary=engagement_summary,
            satisfaction_summary=satisfaction_summary,
        )

        narration_text = build_executive_narration(
            {
                "row_count": row_count,
                "col_count": col_count,
                "missing_highlights": missing_highlights,
                "engagement_summary": engagement_summary,
                "satisfaction_summary": satisfaction_summary,
            }
        )

        state["report_markdown"] = report_markdown
        state["narration_text"] = narration_text
        return state


def _summarize_metric(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    matches = [
        col
        for col in df.columns
        if any(keyword in col.lower() for keyword in keywords)
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not matches:
        return None

    col = matches[0]
    series = df[col].dropna()
    if series.empty:
        return None

    mean_val = series.mean()
    return f"{col} averages {mean_val:.2f} across available records."


def _build_report_markdown(
    *,
    row_count: int,
    col_count: int,
    missing_highlights: str,
    engagement_summary: Optional[str],
    satisfaction_summary: Optional[str],
) -> str:
    lines = [
        "# Executive Report",
        "",
        f"**Dataset size:** {row_count:,} rows ? {col_count:,} columns.",
        "",
        "## Data Health",
        missing_highlights,
        "",
        "## Engagement",
        engagement_summary or "Engagement metrics will be added once configured.",
        "",
        "## Satisfaction",
        satisfaction_summary or "Satisfaction metrics will be added once configured.",
        "",
        "## Next Steps",
        "- Finalize segment definitions and risk thresholds.",
        "- Add impact and access gap charts to the report.",
        "- Validate model readiness for predictive scoring.",
    ]
    return "\n".join(lines)