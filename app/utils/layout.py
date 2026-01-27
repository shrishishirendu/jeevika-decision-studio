from __future__ import annotations

import streamlit as st

from app.utils.paths import asset_path

ISOFT_LOGO = asset_path("isoftpic.png")
JEEVIKA_LOGO = asset_path("Jeevikapic.png")


def render_page_header(page_title: str) -> None:
    st.markdown(
        """
        <style>
        #global-header-anchor + div[data-testid="stHorizontalBlock"] {
            background: #0e1117;
            padding: 18px 30px;
            border-radius: 6px;
            margin-bottom: 25px;
            align-items: center;
        }
        #global-header-anchor + div[data-testid="stHorizontalBlock"] > div {
            display: flex;
            align-items: center;
        }
        #global-header-anchor + div[data-testid="stHorizontalBlock"] > div:nth-child(1) {
            justify-content: flex-start;
        }
        #global-header-anchor + div[data-testid="stHorizontalBlock"] > div:nth-child(2) {
            justify-content: center;
        }
        #global-header-anchor + div[data-testid="stHorizontalBlock"] > div:nth-child(3) {
            justify-content: flex-end;
        }
        .global-header-title {
            color: #ffffff;
            font-size: 30px;
            font-weight: 700;
            text-align: center;
            line-height: 1.2;
            margin: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div id='global-header-anchor'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        if ISOFT_LOGO.exists():
            st.image(str(ISOFT_LOGO), width=130)
        else:
            st.warning("iSoft logo not found")

    with col2:
        st.markdown(
            f"<div class='global-header-title'>{page_title}</div>",
            unsafe_allow_html=True,
        )

    with col3:
        if JEEVIKA_LOGO.exists():
            st.image(str(JEEVIKA_LOGO), width=130)
        else:
            st.warning("Jeevika logo not found")
