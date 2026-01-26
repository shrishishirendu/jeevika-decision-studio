from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

__all__ = [
    "RECOMMENDATION_POLICY",
    "risk_label",
    "assign_segment",
    "format_drivers",
    "generate_recommendation",
    "log_feedback",
]

RECOMMENDATION_POLICY = {
    "Champions": {
        "primary": {
            "action": "Scale advanced services and leadership roles",
            "owner": "Cluster Lead",
            "urgency": "High",
            "timeframe": "0-30 days",
        },
        "secondary": ["Peer mentorship assignments"],
    },
    "Aspirers (High Priority)": {
        "primary": {
            "action": "Remove access friction and enable participation",
            "owner": "Block Coordinator",
            "urgency": "High",
            "timeframe": "0-45 days",
        },
        "secondary": ["Targeted facilitation sessions"],
    },
    "Busy but Blind": {
        "primary": {
            "action": "Improve clarity and process knowledge",
            "owner": "Field Trainer",
            "urgency": "Medium",
            "timeframe": "30-60 days",
        },
        "secondary": ["Simplified communication materials"],
    },
    "Motivated but Blocked": {
        "primary": {
            "action": "Convert intent to action via peer support",
            "owner": "Field Facilitator",
            "urgency": "High",
            "timeframe": "0-45 days",
        },
        "secondary": ["Local activation drives"],
    },
    "Disengaged": {
        "primary": {
            "action": "Trust building and low-cost reactivation",
            "owner": "Community Mobilizer",
            "urgency": "High",
            "timeframe": "0-60 days",
        },
        "secondary": ["Home visits and listening sessions"],
    },
    "Mainstream": {
        "primary": {
            "action": "Maintain engagement and monitor",
            "owner": "Block Coordinator",
            "urgency": "Low",
            "timeframe": "60-90 days",
        },
        "secondary": ["Periodic check-ins"],
    },
}

HIGH_RISK_THRESHOLD = 0.75
MODERATE_RISK_THRESHOLD = 0.55


def risk_label(predicted_class: int, probability: float | None = None) -> str:
    if predicted_class == 1:
        if probability is not None and probability >= HIGH_RISK_THRESHOLD:
            return "Elevated Risk – Intervention Recommended"
        if probability is not None and probability >= MODERATE_RISK_THRESHOLD:
            return "Moderate Risk – Review Suggested"
        return "Risk Signal Detected – Review Suggested"
    return "Low Risk – Monitoring Only"


def assign_segment(row: pd.Series, prediction: int, proba: float, threshold: float = 0.5) -> str:
    if "segment_label" in row and pd.notna(row["segment_label"]):
        return str(row["segment_label"])

    bei = row.get("behavioral_index", 0)
    cei = row.get("cognitive_index", 0)

    if proba >= threshold and bei >= 0.5 and cei >= 0.5:
        return "Champions"
    if proba >= threshold and bei < 0.5 and cei >= 0.5:
        return "Aspirers (High Priority)"
    if proba >= threshold and bei >= 0.5 and cei < 0.5:
        return "Busy but Blind"
    if prediction == 0 and proba < threshold * 0.5:
        return "Disengaged"
    return "Mainstream"


def format_drivers(top_drivers: List[str]) -> List[str]:
    mapping = {
        "behavioral_index": "Behavioral engagement",
        "cognitive_index": "Cognitive awareness",
        "affective_index": "Affective commitment",
        "total_engagement_index": "Overall engagement",
        "participation_count": "Participation breadth",
        "clarity_score": "Clarity and process knowledge",
        "access_ease_avg": "Access ease",
        "barriers_count": "Access barriers",
        "empowerment_score": "Empowerment indicators",
    }
    formatted = []
    for item in top_drivers:
        formatted.append(mapping.get(item, item.replace("_", " ")))
    return formatted


def generate_recommendation(
    row: pd.Series,
    prediction: int,
    proba: float,
    top_drivers: List[str],
    threshold: float,
) -> Dict:
    segment = assign_segment(row, prediction, proba, threshold)
    policy = RECOMMENDATION_POLICY.get(segment, RECOMMENDATION_POLICY["Mainstream"])
    drivers = format_drivers(top_drivers) if top_drivers else ["Drivers unavailable"]

    return {
        "segment": segment,
        "primary": policy["primary"],
        "secondary": policy.get("secondary", []),
        "drivers": drivers,
    }


def log_feedback(
    path: str,
    record: Dict,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    record = {**record, "timestamp": datetime.utcnow().isoformat()}
    df = pd.DataFrame([record])

    if output_path.exists():
        df.to_csv(output_path, mode="a", header=False, index=False)
    else:
        df.to_csv(output_path, mode="w", header=True, index=False)
