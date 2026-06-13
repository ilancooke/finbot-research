from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from finbot_research.feature_label_diagnostics import (
    build_feature_label_diagnostics,
    compute_feature_label_diagnostics,
)
from finbot_research.validation import ValidationError


def test_compute_feature_label_diagnostics_records_coverage_corr_deciles_and_spread() -> None:
    price_features, relative_features, labels = _sample_inputs()

    diagnostics, summary = compute_feature_label_diagnostics(
        price_features,
        relative_features,
        labels,
        selected_feature_columns=("return_63d", "return_63d_market_pct_rank", "missing_feature"),
    )

    assert summary["selected_feature_columns"] == ["return_63d", "return_63d_market_pct_rank"]
    assert summary["skipped_feature_columns"] == ["missing_feature"]
    coverage = diagnostics[
        (diagnostics["metric_type"] == "coverage")
        & (diagnostics["feature_name"] == "return_63d")
    ].iloc[0]
    assert coverage["non_null_count"] == 20
    assert coverage["null_count"] == 0
    assert coverage["label_non_null_count"] == 20
    assert coverage["classification_label_non_null_count"] == 20
    assert coverage["pearson_corr_with_forward_63d_sector_relative_return"] == pytest.approx(1.0)

    deciles = diagnostics[
        (diagnostics["metric_type"] == "decile")
        & (diagnostics["feature_name"] == "return_63d")
    ]
    assert set(deciles["decile"]) == set(range(1, 11))
    top_decile = deciles[deciles["decile"] == 10].iloc[0]
    bottom_decile = deciles[deciles["decile"] == 1].iloc[0]
    assert top_decile["avg_forward_63d_sector_relative_return"] > bottom_decile["avg_forward_63d_sector_relative_return"]

    spread = diagnostics[
        (diagnostics["metric_type"] == "spread")
        & (diagnostics["feature_name"] == "return_63d")
    ].iloc[0]
    assert spread["top_minus_bottom_avg_forward_return"] > 0
    assert spread["top_minus_bottom_top_30pct_flag_rate"] > 0

    year_deciles = diagnostics[diagnostics["metric_type"] == "year_decile"]
    assert set(year_deciles["calendar_year"].dropna()) == {2026}


def test_compute_feature_label_diagnostics_rejects_no_usable_features() -> None:
    price_features, relative_features, labels = _sample_inputs()

    with pytest.raises(ValidationError, match="No selected feature columns"):
        compute_feature_label_diagnostics(
            price_features,
            relative_features,
            labels,
            selected_feature_columns=("missing_feature",),
        )


def test_build_feature_label_diagnostics_writes_output_and_metadata(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    price_features, relative_features, labels = _sample_inputs()
    _write_parquet(price_features, data_root / "features" / "equity_price_features.parquet")
    _write_parquet(relative_features, data_root / "features" / "equity_relative_features.parquet")
    _write_parquet(labels, data_root / "labels" / "equity_forward_return_labels.parquet")

    output_path, metadata_path, summary = build_feature_label_diagnostics(data_root)

    assert output_path == data_root / "research" / "feature_label_diagnostics" / "equity_price_signal_diagnostics.parquet"
    assert output_path.exists()
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["dataset"] == "equity_price_signal_diagnostics"
    assert metadata["dataset_type"] == "research_diagnostics"
    assert metadata["primary_regression_label"] == "forward_63d_sector_relative_return"
    assert metadata["joined_rows"] == 20
    assert summary["joined_rows"] == 20


def _sample_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = [pd.Timestamp("2026-01-02").date(), pd.Timestamp("2026-01-03").date()]
    symbols = [f"S{idx:02d}" for idx in range(1, 11)]
    price_rows = []
    relative_rows = []
    label_rows = []
    for date in dates:
        for idx, symbol in enumerate(symbols, start=1):
            price_rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "return_63d": float(idx),
                }
            )
            relative_rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "return_63d_market_pct_rank": idx / 10,
                }
            )
            label_rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "forward_63d_sector_relative_return": float(idx) / 100,
                    "forward_63d_top_30pct_sector_flag": 1 if idx >= 8 else 0,
                }
            )
    return pd.DataFrame(price_rows), pd.DataFrame(relative_rows), pd.DataFrame(label_rows)


def _write_parquet(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(path, index=False)
