from __future__ import annotations

import pandas as pd

from src.ml_high_uplift import build_features, make_target, train_models


def main():
    df = pd.DataFrame(
        {
            "household_income_increase_percent": [5, 12, 25, 40, 18, 30],
            "meeting_attendance": [1, 2, 3, 4, 2, 3],
            "involvement_level": ["passive", "moderate", "active", "active", "moderate", "passive"],
            "scheme_awareness_count": [2, 4, 6, 7, 3, 5],
            "district": ["A", "B", "A", "C", "B", "C"],
        }
    )

    X, numeric_cols, categorical_cols = build_features(df, return_meta=True)
    y = make_target(df, threshold=20)
    results = train_models(X, y, numeric_cols, categorical_cols)
    fi = results.feature_importance.get("LogisticRegression")
    assert fi is not None and not fi.empty
    print("OK: feature importance rows", len(fi))


if __name__ == "__main__":
    main()