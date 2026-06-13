from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from finbot_research.io import parquet_columns, read_parquet, write_parquet


def test_write_and_read_parquet(tmp_path: Path) -> None:
    path = tmp_path / "research" / "sample.parquet"
    frame = pd.DataFrame({"symbol": ["AAPL"], "date": ["2026-01-02"], "value": [1.0]})

    write_parquet(frame, path)

    assert path.exists()
    assert read_parquet(path).equals(frame)
    assert read_parquet(path, columns=["symbol"]).equals(frame[["symbol"]])
    assert parquet_columns(path) == ["symbol", "date", "value"]


def test_read_parquet_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing parquet input"):
        read_parquet(tmp_path / "missing.parquet")
