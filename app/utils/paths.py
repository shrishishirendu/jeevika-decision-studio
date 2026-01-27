from __future__ import annotations

from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = APP_DIR / "assets"


def asset_path(name: str) -> Path:
    return ASSETS_DIR / name
