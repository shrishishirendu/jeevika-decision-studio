from __future__ import annotations

from typing import Dict, Optional


def build_executive_narration(state: Dict) -> str:
    """Create a concise narration for the executive report."""
    row_count = state.get("row_count")
    col_count = state.get("col_count")
    missing_highlights = state.get("missing_highlights")
    engagement_summary = state.get("engagement_summary")
    satisfaction_summary = state.get("satisfaction_summary")

    size_sentence = ""
    if row_count is not None and col_count is not None:
        size_sentence = (
            f"The current dataset includes {row_count:,} records across {col_count:,} columns. "
        )

    if not missing_highlights:
        missing_highlights = (
            "Missingness highlights are still being prepared as the data pipeline is finalized."
        )

    if not engagement_summary:
        engagement_summary = (
            "Engagement indicators are not yet fully configured, so this summary will be updated soon."
        )

    if not satisfaction_summary:
        satisfaction_summary = (
            "Satisfaction signals are not yet fully configured, and will be added after score validation."
        )

    narration = (
        "Here is a brief executive narration for the Jeevika Decision Studio. "
        f"{size_sentence}"
        f"{missing_highlights} "
        f"{engagement_summary} "
        f"{satisfaction_summary} "
        "Next, we will refine the segment-level insights and integrate recommendations for program action."
    )

    return narration