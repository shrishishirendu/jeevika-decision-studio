from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class SurveySchema:
    rename_map: Dict[str, str]
    categoricals: List[str]
    numerics: List[str]
    geography_col: Optional[str]


def load_columns_config(path: str | Path = "config/columns.yaml") -> SurveySchema:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parents[1] / config_path

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    rename_map = data.get("rename") or {}
    categoricals = data.get("categoricals") or []
    numerics = data.get("numerics") or []
    geography = data.get("geography") or {}
    geography_col = geography.get("village_block")

    return SurveySchema(
        rename_map=rename_map,
        categoricals=categoricals,
        numerics=numerics,
        geography_col=geography_col,
    )
