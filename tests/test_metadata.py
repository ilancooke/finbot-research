from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from finbot_research.metadata import metadata_path_for, write_metadata


def test_write_metadata_records_shape_paths_and_extra_fields(tmp_path: Path) -> None:
    output_path = tmp_path / "research" / "diagnostics.parquet"
    input_path = tmp_path / "features" / "equity_price_features.parquet"
    frame = pd.DataFrame({"metric_type": ["coverage"], "feature_name": ["return_63d"]})

    metadata_path = write_metadata(
        dataset_name="equity_price_signal_diagnostics",
        output_path=output_path,
        input_paths=[input_path],
        dataframe=frame,
        extra_metadata={"primary_label": "forward_63d_sector_relative_return"},
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata_path == metadata_path_for(output_path)
    assert metadata["dataset"] == "equity_price_signal_diagnostics"
    assert metadata["output_path"] == str(output_path)
    assert metadata["input_paths"] == [str(input_path)]
    assert metadata["rows"] == 1
    assert metadata["column_count"] == 2
    assert metadata["primary_label"] == "forward_63d_sector_relative_return"
