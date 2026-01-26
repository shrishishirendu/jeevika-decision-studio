from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BARRIER_COLUMNS = [
    "barrier_lack_of_info",
    "barrier_access_difficulty",
    "barrier_inadequate_capital",
    "barrier_poor_market",
    "barrier_skill_gap",
    "barrier_family_opposition",
]


def generate_mock_survey(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    segment_sizes = {
        "Empowered Leaders": int(n * 0.15),
        "Active Participants": int(n * 0.40),
        "Silent Savers": int(n * 0.15),
        "Aspirers (Aware but Stuck)": int(n * 0.20),
        "At-Risk / Disengaging": n - int(n * 0.90),
    }

    frames = []
    respondent_id = 1

    for segment, size in segment_sizes.items():
        if size <= 0:
            continue

        if segment == "Empowered Leaders":
            attendance = rng.integers(3, 5, size=size)
            engagement = rng.choice(["active"], size=size)
            awareness = rng.integers(6, 11, size=size)
            participation_prob = 0.7
            barrier_prob = 0.1
            income = rng.integers(20, 61, size=size)
            satisfaction = rng.integers(4, 6, size=size)
            referral = rng.integers(7, 11, size=size)
            confidence = rng.integers(4, 6, size=size)
        elif segment == "Active Participants":
            attendance = rng.integers(2, 4, size=size)
            engagement = rng.choice(["active", "moderate"], size=size, p=[0.5, 0.5])
            awareness = rng.integers(4, 8, size=size)
            participation_prob = 0.45
            barrier_prob = 0.3
            income = rng.integers(10, 31, size=size)
            satisfaction = rng.integers(3, 5, size=size)
            referral = rng.integers(5, 9, size=size)
            confidence = rng.integers(3, 5, size=size)
        elif segment == "Silent Savers":
            attendance = rng.integers(1, 3, size=size)
            engagement = rng.choice(["moderate", "passive"], size=size, p=[0.6, 0.4])
            awareness = rng.integers(3, 7, size=size)
            participation_prob = 0.25
            barrier_prob = 0.35
            income = rng.integers(5, 21, size=size)
            satisfaction = rng.integers(3, 5, size=size)
            referral = rng.integers(4, 8, size=size)
            confidence = rng.integers(3, 5, size=size)
        elif segment == "Aspirers (Aware but Stuck)":
            attendance = rng.integers(1, 3, size=size)
            engagement = rng.choice(["moderate"], size=size)
            awareness = rng.integers(7, 11, size=size)
            participation_prob = 0.2
            barrier_prob = 0.6
            income = rng.integers(0, 11, size=size)
            satisfaction = rng.integers(2, 4, size=size)
            referral = rng.integers(3, 7, size=size)
            confidence = rng.integers(2, 4, size=size)
        else:
            attendance = rng.integers(0, 2, size=size)
            engagement = rng.choice(["passive"], size=size)
            awareness = rng.integers(0, 5, size=size)
            participation_prob = 0.1
            barrier_prob = 0.7
            income = rng.integers(0, 9, size=size)
            satisfaction = rng.integers(1, 3, size=size)
            referral = rng.integers(0, 5, size=size)
            confidence = rng.integers(1, 3, size=size)

        participation_cols = {
            "savings_participation": rng.binomial(1, 0.7 if segment == "Silent Savers" else participation_prob, size=size),
            "credit_participation": rng.binomial(1, participation_prob, size=size),
            "agricultural_intervention": rng.binomial(1, participation_prob, size=size),
            "livestock_activity": rng.binomial(1, participation_prob, size=size),
            "nonfarm_enterprise": rng.binomial(1, participation_prob, size=size),
            "didi_ki_rasoi": rng.binomial(1, participation_prob, size=size),
            "producer_groups": rng.binomial(1, participation_prob, size=size),
            "vo_participation": rng.binomial(1, participation_prob, size=size),
        }

        barriers = {
            col: rng.binomial(1, barrier_prob, size=size) for col in BARRIER_COLUMNS
        }

        frame = pd.DataFrame(
            {
                "respondent_id": np.arange(respondent_id, respondent_id + size),
                "village_block": rng.choice(
                    ["Block A", "Block B", "Block C", "Block D"], size=size
                ),
                "membership_duration_months": rng.integers(6, 61, size=size),
                "membership_duration": rng.integers(6, 61, size=size),
                "shg_meetings_attended_monthly": attendance,
                "engagement_level": engagement,
                "scheme_awareness_count": awareness,
                "awareness_score": awareness,
                "household_income_increase_percent": income,
                "satisfaction_level": satisfaction,
                "future_program_confidence": confidence,
                "referral_likelihood": referral,
            }
        )

        for col, values in participation_cols.items():
            frame[col] = values

        for col, values in barriers.items():
            frame[col] = values

        frames.append(frame)
        respondent_id += size

    df = pd.concat(frames, ignore_index=True)
    return df


def save_mock_csv(path: str = "data/mock/jeevika_mock.csv") -> Path:
    output_path = Path(path)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parents[1] / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return output_path

    df = generate_mock_survey()
    df.to_csv(output_path, index=False)
    return output_path