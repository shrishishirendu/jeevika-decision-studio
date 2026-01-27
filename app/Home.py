from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.utils.layout import render_page_header
from app.utils.paths import asset_path

st.set_page_config(page_title="Jeevika Survey", layout="wide")

HTML_PATH = asset_path("jeevika-survey-hindi.html")

render_page_header("Jeevika Survey")

try:
    html = HTML_PATH.read_text(encoding="utf-8")
    components.html(html, height=1400, scrolling=True)
except FileNotFoundError:
    st.error(f"Survey HTML file not found at: {HTML_PATH}")
except Exception as exc:  # pragma: no cover
    st.error(f"Failed to load survey HTML: {exc}")
