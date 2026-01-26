from __future__ import annotations

import io
import os

from openai import OpenAI


def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment.")
    return OpenAI(api_key=api_key)


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    client = _client()
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename
    result = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=audio_file,
    )
    return result.text


def speak_text(text: str, voice: str = "alloy") -> bytes:
    try:
        client = _client()
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text,
        )
        return response.read()
    except Exception as exc:
        raise RuntimeError(f"TTS failed: {exc}") from exc


def generate_executive_answer(prompt: str, model: str = "gpt-4o-mini") -> str:
    client = _client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an executive analyst. Provide concise, boardroom-ready responses.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()
