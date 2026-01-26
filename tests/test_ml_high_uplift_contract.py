from __future__ import annotations

import pandas as pd

from src.ml_high_uplift import build_features


def test_build_features_contract():
    df = pd.DataFrame(
        {
            "household_income_increase_percent": [0, 10, 25],
            "meeting_attendance": [1, 2, 3],
            "involvement_level": ["passive", "moderate", "active"],
            "scheme_awareness_count": [2, 4, 6],
            "district": ["A", "B", "A"],
        }
    )

    X, num_cols, cat_cols = build_features(df, return_meta=True)
    assert isinstance(X, pd.DataFrame)
    assert num_cols is not None and cat_cols is not None
    assert "behavioral_index" in X.columns
    assert "cognitive_index" in X.columns
    assert "affective_index" in X.columns
    assert "total_engagement_index" in X.columns

    X2, num2, cat2 = build_features(df, return_meta=False)
    assert isinstance(X2, pd.DataFrame)
    assert num2 is None and cat2 is None