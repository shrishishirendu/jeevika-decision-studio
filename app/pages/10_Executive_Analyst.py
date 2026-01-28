from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.ui_filters import apply_filters, load_clean_df
from src import recommendations as recommendations
from src.voice_agent import generate_executive_answer, speak_text, transcribe_audio
from app.utils.layout import render_page_header

CONFIG_PATH = ROOT / "config" / "config.yaml"
PRED_PATH = ROOT / "data" / "processed" / "predicted_high_uplift.parquet"

st.set_page_config(page_title="Executive Analyst", page_icon="Executive Analyst", layout="wide")

render_page_header("Executive Analyst")
st.caption("Ask a question by voice, get an executive brief and audio response.")

if not CONFIG_PATH.exists():
    st.error("Missing config/config.yaml. Please add it to continue.")
    st.stop()

with CONFIG_PATH.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

data_config = config.get("data") or {}
processed_path = data_config.get("processed_path")
raw_path = data_config.get("raw_path")
if not processed_path:
    st.error("config.yaml is missing data.processed_path")
    st.stop()

parquet_path = ROOT / processed_path
csv_path = ROOT / raw_path if raw_path else None

try:
    df = load_clean_df(str(parquet_path), str(csv_path) if csv_path else None)
except Exception as exc:  # pragma: no cover
    st.error(f"Failed to load cleaned dataset: {exc}")
    st.stop()

filtered_df, selections = apply_filters(df)

@st.cache_data(show_spinner=False)
def cached_transcribe(audio_bytes: bytes) -> str:
    return transcribe_audio(audio_bytes)

@st.cache_data(show_spinner=False)
def cached_tts(text: str) -> bytes:
    return speak_text(text)

@st.cache_data(show_spinner=False)
def cached_answer(prompt: str) -> str:
    return generate_executive_answer(prompt)

st.subheader("Voice Input")
audio_bytes = None
if hasattr(st, "audio_input"):
    audio_file = st.audio_input("Ask Executive Analyst (voice)")
    if audio_file is not None:
        audio_bytes = audio_file.getvalue()
else:
    audio_file = st.file_uploader("Upload a WAV/MP3 file", type=["wav", "mp3", "m4a"]) 
    if audio_file is not None:
        audio_bytes = audio_file.getvalue()

transcript = st.session_state.get("executive_transcript", "")
if audio_bytes:
    try:
        transcript = cached_transcribe(audio_bytes)
        st.session_state.executive_transcript = transcript
    except Exception as exc:
        st.error(f"Transcription failed: {exc}")

st.text_area("Transcript", value=transcript, height=120)

if "executive_answer" not in st.session_state:
    st.session_state.executive_answer = ""

if st.button("Generate Answer"):
    if not transcript:
        st.warning("Please record or upload audio first.")
    else:
        pred_summary = "Predictions file not found."
        if PRED_PATH.exists():
            pred_df = pd.read_parquet(PRED_PATH)
            if "predicted_class" in pred_df.columns:
                counts = pred_df["predicted_class"].value_counts().to_dict()
                pred_summary = f"Predicted class counts: {counts}"

        policy_summary = ", ".join(recommendations.RECOMMENDATION_POLICY.keys())
        filter_summary = json.dumps(selections)

        prompt = (
            "You are an executive analyst. Use this context:\n"
            f"Filters: {filter_summary}\n"
            f"Prediction summary: {pred_summary}\n"
            f"Recommendation segments: {policy_summary}\n"
            "Respond with:\n"
            "- 2-3 sentence executive summary\n"
            "- 3-5 bullets: What we see\n"
            "- 3 bullets: What to do next\n"
            "- 1 bullet: Risks / caveats\n"
            f"Question: {transcript}"
        )
        try:
            answer = cached_answer(prompt)
            st.session_state.executive_answer = answer
        except Exception as exc:
            st.error(f"Answer generation failed: {exc}")

answer = st.session_state.get("executive_answer", "")
if answer:
    st.subheader("Answer")
    st.write(answer)
    try:
        audio_out = cached_tts(answer)
        if audio_out:
            st.audio(audio_out, format="audio/mp3")
        else:
            st.error("TTS failed: no audio returned.")
    except Exception as exc:
        st.error(f"TTS failed: {exc}")
