from __future__ import annotations

import pandas as pd

from src.ml_high_uplift import build_features, get_predictions_for_df, train_models, make_target
from src.recommendations import generate_recommendation, risk_label


def main():
    df = pd.DataFrame(
        {
            "respondent_id": ["A1", "A2", "A3", "A4"],
            "household_income_increase_percent": [5, 25, 30, 12],
            "meeting_attendance": [1, 3, 4, 2],
            "involvement_level": ["passive", "active", "active", "moderate"],
            "scheme_awareness_count": [2, 6, 7, 3],
            "district": ["D1", "D2", "D1", "D3"],
        }
    )

    X, num_cols, cat_cols = build_features(df, return_meta=True)
    y = make_target(df, threshold=20)
    results = train_models(X, y, num_cols, cat_cols)
    model = results.models["LogisticRegression"]

    preds = get_predictions_for_df(df, model, threshold=0.5, id_col="respondent_id")
    assert {"id", "predicted_class", "probability", "top_drivers"}.issubset(preds.columns)

    row = df.iloc[0]
    rec = generate_recommendation(row, 0, 0.4, ["behavioral_index", "clarity_score"], 0.5)
    assert "primary" in rec and "segment" in rec
    assert risk_label(1, 0.9) == "Elevated Risk – Intervention Recommended"
    assert risk_label(1, 0.6) == "Moderate Risk – Review Suggested"
    assert risk_label(0, 0.2) == "Low Risk – Monitoring Only"
    print("OK: recommendation pipeline")


if __name__ == "__main__":
    main()
