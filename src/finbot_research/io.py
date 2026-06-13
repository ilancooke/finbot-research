from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def read_parquet(path: Path, columns: list[str] | tuple[str, ...] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet input: {path}")
    return pd.read_parquet(path, columns=list(columns) if columns is not None else None)


def parquet_columns(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet input: {path}")
    return pq.read_schema(path).names


def write_parquet(dataframe: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(path, index=False)
    return path


def write_csv(dataframe: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)
    return path
