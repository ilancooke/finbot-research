from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from finbot_research.bucket_signal_diagnostics import (
    build_bucket_signal_diagnostics,
    compute_bucket_signal_diagnostics,
    compute_candidate_rule_stability,
)
from finbot_research.cli import app


def test_bucket_assignment_and_metric_calculation() -> None:
    price_features, relative_features, labels = _sample_inputs()

    summary, candidate_rules, candidate_rule_years, candidate_rule_stability, diagnostics_summary = compute_bucket_signal_diagnostics(
        price_features,
        relative_features,
        labels,
    )
    rows = summary.set_index(["time_window", "combo_name", "bucket_1_value", "bucket_2_value"])
    high_strong = rows.loc[
        (
            "full_completed_history",
            "volatility_63d_bucket_x_momentum_63d_sector_bucket",
            "high_volatility",
            "strong_momentum",
        )
    ]

    assert high_strong["row_count"] == 9
    assert high_strong["symbol_count"] == 3
    assert high_strong["avg_forward_63d_sector_relative_return"] == pytest.approx(0.09)
    assert high_strong["median_forward_63d_sector_relative_return"] == pytest.approx(0.09)
    assert high_strong["avg_minus_median_forward_63d_sector_relative_return"] == pytest.approx(0.0)
    assert high_strong["top_30pct_sector_flag_rate"] == pytest.approx(1.0)
    assert high_strong["bottom_30pct_sector_flag_rate"] == pytest.approx(0.0)
    drawdown_values = set(summary.loc[summary["combo_name"].str.contains("drawdown"), "bucket_2_value"])
    assert "sector_relative_deep_drawdown" in drawdown_values
    assert "sector_relative_mid_drawdown" in drawdown_values
    assert "sector_relative_near_52w_high" in drawdown_values
    assert not {"drawdown_low_bucket", "drawdown_middle_bucket", "drawdown_high_bucket"} & drawdown_values
    assert not candidate_rules.empty
    assert not candidate_rule_years.empty
    assert not candidate_rule_stability.empty
    assert diagnostics_summary["bottom_label_available"]


def test_only_specified_combinations_are_generated() -> None:
    summary, candidate_rules, _candidate_rule_years, _candidate_rule_stability, diagnostics_summary = compute_bucket_signal_diagnostics(*_sample_inputs())

    assert set(summary["combo_name"]) == {
        "volatility_63d_bucket_x_momentum_63d_sector_bucket",
        "volatility_63d_bucket_x_drawdown_52w_sector_bucket",
        "momentum_63d_sector_bucket_x_drawdown_52w_sector_bucket",
        "dollar_volume_63d_bucket_x_momentum_63d_sector_bucket",
        "dollar_volume_63d_bucket_x_volatility_63d_bucket",
    }
    assert candidate_rules["rule_name"].nunique() == 6
    assert diagnostics_summary["skipped_combinations"] == []


def test_missing_features_skip_dependent_combinations() -> None:
    price_features, relative_features, labels = _sample_inputs()
    relative_features = relative_features.drop(columns=["drawdown_from_52w_high_sector_pct_rank"])

    summary, candidate_rules, _candidate_rule_years, _candidate_rule_stability, diagnostics_summary = compute_bucket_signal_diagnostics(price_features, relative_features, labels)

    assert "drawdown_from_52w_high_sector_pct_rank" in diagnostics_summary["skipped_features"]
    assert not summary["combo_name"].str.contains("drawdown").any()
    skipped_names = {item["combo_name"] for item in diagnostics_summary["skipped_combinations"]}
    assert skipped_names == {
        "volatility_63d_bucket_x_drawdown_52w_sector_bucket",
        "momentum_63d_sector_bucket_x_drawdown_52w_sector_bucket",
    }
    skipped_rules = {item["rule_name"] for item in diagnostics_summary["skipped_candidate_rules"]}
    assert skipped_rules == {
        "high_volatility_plus_sector_relative_near_52w_high",
        "sector_relative_near_52w_high_plus_strong_momentum",
        "high_volatility_plus_sector_relative_near_52w_high_plus_strong_momentum",
        "high_volatility_plus_sector_relative_deep_drawdown",
    }
    assert not candidate_rules["rule_name"].str.contains("near_52w_high|deep_drawdown").any()


def test_time_windows_exclude_current_partial_year_by_default() -> None:
    summary, candidate_rules, candidate_rule_years, _candidate_rule_stability, diagnostics_summary = compute_bucket_signal_diagnostics(*_sample_inputs())
    windows = {row["time_window"]: row for row in diagnostics_summary["time_windows"]}

    assert windows["full_completed_history"]["start_year"] == 2023
    assert windows["full_completed_history"]["end_year"] == 2025
    assert windows["last_10y_completed"]["start_year"] == 2023
    assert windows["last_10y_completed"]["end_year"] == 2025
    assert windows["last_5y_completed"]["start_year"] == 2023
    assert windows["last_5y_completed"]["end_year"] == 2025
    assert windows["current_partial_year"]["start_year"] == 2026
    assert windows["current_partial_year"]["is_current_partial_year_window"]
    assert diagnostics_summary["partial_year_handling"]["current_partial_year"] == 2026
    assert diagnostics_summary["partial_year_handling"]["latest_completed_stability_year"] == 2025
    assert set(summary["time_window"]) == set(windows)
    assert set(candidate_rules["time_window"]) == set(windows)
    partial_years = candidate_rule_years[candidate_rule_years["calendar_year"] == 2026]
    completed_years = candidate_rule_years[candidate_rule_years["calendar_year"] < 2026]
    assert partial_years["is_partial_year"].all()
    assert not partial_years["included_in_completed_year_stability"].any()
    assert completed_years["included_in_completed_year_stability"].all()


def test_candidate_rule_generation_includes_three_way_rule() -> None:
    _summary, candidate_rules, _candidate_rule_years, _candidate_rule_stability, _diagnostics_summary = compute_bucket_signal_diagnostics(*_sample_inputs())
    rules = candidate_rules.set_index(["rule_name", "time_window"])
    rule = rules.loc[
        (
            "high_volatility_plus_sector_relative_near_52w_high_plus_strong_momentum",
            "full_completed_history",
        )
    ]

    assert rule["row_count"] == 9
    assert rule["symbol_count"] == 3
    assert rule["avg_forward_63d_sector_relative_return"] == pytest.approx(0.09)
    assert rule["median_forward_63d_sector_relative_return"] == pytest.approx(0.09)
    assert rule["top_30pct_sector_flag_rate"] == pytest.approx(1.0)
    assert rule["bottom_30pct_sector_flag_rate"] == pytest.approx(0.0)
    assert rule["avg_forward_return_vs_baseline"] == pytest.approx(0.035)
    assert rule["median_forward_return_vs_baseline"] == pytest.approx(0.035)
    assert rule["top_30pct_flag_rate_vs_baseline"] == pytest.approx(0.7)
    assert rule["bottom_30pct_flag_rate_vs_baseline"] == pytest.approx(-0.3)


def test_candidate_rule_years_and_stability_use_baselines() -> None:
    _summary, _candidate_rules, candidate_rule_years, candidate_rule_stability, _diagnostics_summary = (
        compute_bucket_signal_diagnostics(*_sample_inputs())
    )
    years = candidate_rule_years.set_index(["rule_name", "calendar_year"])
    year = years.loc[("high_volatility_plus_sector_relative_near_52w_high_plus_strong_momentum", 2023)]

    assert year["row_count"] == 3
    assert not bool(year["is_partial_year"])
    assert bool(year["included_in_completed_year_stability"])
    assert year["avg_forward_return_vs_baseline"] == pytest.approx(0.035)
    assert year["median_forward_return_vs_baseline"] == pytest.approx(0.035)
    assert year["top_30pct_flag_rate_vs_baseline"] == pytest.approx(0.7)
    assert year["bottom_30pct_flag_rate_vs_baseline"] == pytest.approx(-0.3)

    stability = candidate_rule_stability.set_index("rule_name")
    primary = stability.loc["high_volatility_plus_sector_relative_near_52w_high_plus_strong_momentum"]
    assert primary["completed_years_count"] == 3
    assert primary["pct_years_median_above_baseline"] == pytest.approx(1.0)
    assert primary["stability_assessment"] == "broadly_consistent"
    empty_rule = stability.loc["high_volatility_plus_weak_momentum"]
    assert empty_rule["stability_assessment"] == "insufficient_data"


def test_candidate_rule_stability_assessment_labels() -> None:
    years = pd.DataFrame(
        [
            _stability_row("tail", year, avg=0.02, median=-0.01, top=-0.01) for year in (2021, 2022, 2023)
        ]
        + [
            _stability_row("weak", year, avg=-0.01, median=-0.02, top=-0.02) for year in (2021, 2022, 2023)
        ]
        + [
            _stability_row("mixed", 2021, avg=0.01, median=0.01, top=-0.01),
            _stability_row("mixed", 2022, avg=-0.01, median=-0.01, top=0.01),
            _stability_row("mixed", 2023, avg=-0.01, median=-0.01, top=0.01),
        ]
    )

    stability = compute_candidate_rule_stability(years).set_index("rule_name")

    assert stability.loc["tail", "stability_assessment"] == "average_only_tail_driven"
    assert stability.loc["weak", "stability_assessment"] == "weak_or_negative"
    assert stability.loc["mixed", "stability_assessment"] == "mixed_or_regime_dependent"


def test_build_bucket_signal_diagnostics_writes_outputs_report_and_metadata(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    price_features, relative_features, labels = _sample_inputs()
    _write_parquet(price_features, data_root / "features" / "equity_price_features.parquet")
    _write_parquet(relative_features, data_root / "features" / "equity_relative_features.parquet")
    _write_parquet(labels, data_root / "labels" / "equity_forward_return_labels.parquet")

    paths, diagnostics_summary = build_bucket_signal_diagnostics(data_root)

    assert paths["summary_parquet"].exists()
    assert paths["summary_csv"].exists()
    assert paths["candidate_rules_parquet"].exists()
    assert paths["candidate_rules_csv"].exists()
    assert paths["candidate_rule_years_parquet"].exists()
    assert paths["candidate_rule_years_csv"].exists()
    assert paths["candidate_rule_stability_parquet"].exists()
    assert paths["candidate_rule_stability_csv"].exists()
    assert paths["markdown_report"].exists()
    assert paths["metadata"].exists()
    assert diagnostics_summary["combinations_analyzed"] == 5
    report = paths["markdown_report"].read_text(encoding="utf-8")
    assert "## Bucket Definitions" in report
    assert "## Candidate Rule Diagnostics" in report
    assert "## Candidate Rule Stability" in report
    assert "## Current Partial Year Snapshot" in report
    assert "## Important Caveats" in report
    assert "## Output File Guide" in report
    assert "size/liquidity/crowding proxy" in report
    assert "Values closer to 0 indicate stocks closer to their 52-week highs" in report
    assert "sector_relative_near_52w_high" in report

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["dataset"] == "equity_bucket_signal_summary"
    assert metadata["dataset_type"] == "research_bucket_diagnostics"
    assert "bucket_definitions" in metadata
    assert "combination_definitions" in metadata
    assert "candidate_rule_definitions" in metadata
    assert "candidate_rules_parquet" in metadata["output_paths"]
    assert "candidate_rule_years_parquet" in metadata["output_paths"]
    assert "candidate_rule_stability_parquet" in metadata["output_paths"]
    assert "baseline_definition" in metadata
    assert "stability_assessment_rules" in metadata
    assert "partial_year_handling" in metadata
    assert "dollar_volume_interpretation" in metadata
    assert metadata["parquet_is_canonical"]
    assert metadata["csv_is_convenience_export"]


def test_cli_bucket_signal_diagnostics_writes_expected_outputs(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    price_features, relative_features, labels = _sample_inputs()
    _write_parquet(price_features, data_root / "features" / "equity_price_features.parquet")
    _write_parquet(relative_features, data_root / "features" / "equity_relative_features.parquet")
    _write_parquet(labels, data_root / "labels" / "equity_forward_return_labels.parquet")
    monkeypatch.setenv("FINBOT_DATA_ROOT", str(data_root))
    runner = CliRunner()

    result = runner.invoke(app, ["bucket-signal-diagnostics"])

    assert result.exit_code == 0, result.output
    assert "Bucket signal diagnostics complete." in result.output
    assert "Canonical machine-readable output:" in result.output
    assert (data_root / "research" / "bucket_signal_diagnostics" / "equity_bucket_signal_summary.parquet").exists()
    assert (data_root / "research" / "bucket_signal_diagnostics" / "equity_bucket_signal_summary.csv").exists()
    assert (data_root / "research" / "bucket_signal_diagnostics" / "equity_bucket_candidate_rules.parquet").exists()
    assert (data_root / "research" / "bucket_signal_diagnostics" / "equity_bucket_candidate_rules.csv").exists()
    assert (data_root / "research" / "bucket_signal_diagnostics" / "equity_bucket_candidate_rule_years.parquet").exists()
    assert (data_root / "research" / "bucket_signal_diagnostics" / "equity_bucket_candidate_rule_years.csv").exists()
    assert (data_root / "research" / "bucket_signal_diagnostics" / "equity_bucket_candidate_rule_stability.parquet").exists()
    assert (data_root / "research" / "bucket_signal_diagnostics" / "equity_bucket_candidate_rule_stability.csv").exists()
    assert (data_root / "research" / "bucket_signal_diagnostics" / "equity_bucket_signal_report.md").exists()
    assert (data_root / "research" / "bucket_signal_diagnostics" / "equity_bucket_signal_summary.metadata.json").exists()


def _sample_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = [pd.Timestamp(f"{year}-06-30").date() for year in (2023, 2024, 2025, 2026)]
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
                    "volatility_63d": float(idx),
                    "average_dollar_volume_63d": float(idx * 1000),
                }
            )
            relative_rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "return_63d_sector_pct_rank": idx / 10,
                    "drawdown_from_52w_high_sector_pct_rank": idx / 10,
                }
            )
            label_rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "forward_63d_sector_relative_return": idx / 100,
                    "forward_63d_top_30pct_sector_flag": 1 if idx >= 8 else 0,
                    "forward_63d_bottom_30pct_sector_flag": 1 if idx <= 3 else 0,
                }
            )
    return pd.DataFrame(price_rows), pd.DataFrame(relative_rows), pd.DataFrame(label_rows)


def _write_parquet(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(path, index=False)


def _stability_row(rule_name: str, year: int, *, avg: float, median: float, top: float) -> dict[str, object]:
    return {
        "rule_name": rule_name,
        "calendar_year": year,
        "included_in_completed_year_stability": True,
        "row_count": 10,
        "avg_forward_return_vs_baseline": avg,
        "median_forward_return_vs_baseline": median,
        "top_30pct_flag_rate_vs_baseline": top,
        "bottom_30pct_flag_rate_vs_baseline": -top,
    }
