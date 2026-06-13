from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd


def metadata_path_for(parquet_path: Path) -> Path:
    return parquet_path.with_suffix(".metadata.json")


def write_metadata(
    *,
    dataset_name: str,
    output_path: Path,
    input_paths: list[Path],
    dataframe: pd.DataFrame,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    metadata_path = metadata_path_for(output_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = build_metadata(
        dataset_name=dataset_name,
        output_path=output_path,
        input_paths=input_paths,
        dataframe=dataframe,
        extra_metadata=extra_metadata,
    )
    with tempfile.NamedTemporaryFile(
        dir=metadata_path.parent,
        suffix=".json",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as temp_metadata:
        temp_path = Path(temp_metadata.name)
        json.dump(metadata, temp_metadata, indent=2, sort_keys=True)
        temp_metadata.write("\n")
    try:
        temp_path.replace(metadata_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return metadata_path


def build_metadata(
    *,
    dataset_name: str,
    output_path: Path,
    input_paths: list[Path],
    dataframe: pd.DataFrame,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    generated_at = datetime.now(UTC)
    metadata: dict[str, Any] = {
        "dataset": dataset_name,
        "generated_at_utc": generated_at.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_path": str(output_path),
        "input_paths": [str(path) for path in input_paths],
        "rows": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "columns": list(dataframe.columns),
    }
    metadata.update(extra_metadata or {})
    return metadata
