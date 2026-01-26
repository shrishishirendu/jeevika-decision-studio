from __future__ import annotations

from pathlib import Path

import pyttsx3


def synthesize_voiceover(text: str, out_path: str, rate: int = 175) -> str:
    """Generate a WAV voiceover at out_path using pyttsx3."""
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.save_to_file(text, str(output_path))
    engine.runAndWait()

    return str(output_path)