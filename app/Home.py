from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

st.set_page_config(
    page_title="Jeevika Decision Studio",
    page_icon="??",
    layout="wide",
)

st.title("Jeevika Decision Studio")
st.markdown(
    """
A Streamlit workspace for exploring Jeevika survey data, assessing data health,
segmenting beneficiaries, and generating decision-ready insights.

Use the sidebar to navigate through the modules. Start with **Data Health** to
validate the dataset before moving into analytics and recommendations.
"""
)

st.subheader("Navigation")
st.markdown(
    """
- **0_Data_Health**: Basic dataset stats, missing values, and readiness checks
- **1_Overview**: High-level summaries and key indicators
- **2_Engagement**: Participation and engagement trends
- **3_Awareness_Access_Gap**: Awareness vs. access analysis
- **4_Impact**: Outcome and impact signals
- **5_Segments**: Rule-based or clustering segments
- **6_Risk_Opportunity**: At-risk groups and growth opportunities
- **7_Recommendations**: Prescriptive actions and rules
- **8_Executive_Analyst**: Executive summary and analyst view
"""
)

st.info("Tip: Ensure `data/raw/jeevika_survey.csv` is present before analysis.")