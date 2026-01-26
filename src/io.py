from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

__all__ = [
    "load_raw_csv",
    "save_clean_parquet",
    "load_clean_parquet",
    "load_csv",
    "write_parquet",
]

_CSV_CACHE: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], pd.DataFrame] = {}


def load_raw_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load a raw CSV with a simple in-memory cache keyed by path + kwargs."""
    csv_path = Path(path).expanduser()
    key = (str(csv_path.resolve()), tuple(sorted(kwargs.items())))
    if key not in _CSV_CACHE:
        _CSV_CACHE[key] = pd.read_csv(csv_path, **kwargs)
    return _CSV_CACHE[key]


def save_clean_parquet(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> Path:
    """Write a DataFrame to parquet, creating parent directories if needed."""
    parquet_path = Path(path).expanduser()
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, **kwargs)
    return parquet_path


def load_clean_parquet(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load a cleaned parquet dataset."""
    parquet_path = Path(path).expanduser()
    return pd.read_parquet(parquet_path, **kwargs)


def load_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Backward-compatible alias for load_raw_csv."""
    return load_raw_csv(path, **kwargs)


def write_parquet(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> Path:
    """Backward-compatible alias for save_clean_parquet."""
    return save_clean_parquet(df, path, **kwargs)
