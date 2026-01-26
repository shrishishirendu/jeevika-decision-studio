from __future__ import annotations

import os

from src.voice_agent import speak_text


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("SKIP: OPENAI_API_KEY not set")
        return
    audio = speak_text("Hello")
    assert audio and len(audio) > 1000
    print("OK: TTS bytes", len(audio))


if __name__ == "__main__":
    main()