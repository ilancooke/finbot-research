from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from finbot_research.cli import app
from finbot_research.price_strength_scorecard import (
    assess_stability,
    build_price_strength_scorecard,
    compute_scorecard_stability,
    compute_scorecard_years,
    compute_price_strength_scorecard,
)
from finbot_research.validation import ValidationError


def test_compute_price_strength_scorecard_assigns_buckets_flags_and_scores() -> None:
    price_features, relative_features, labels = _sample_inputs()

    scorecard, summary, build_summary = compute_price_strength_scorecard(price_features, relative_features, labels)
    rows = scorecard[scorecard["date"] == pd.Timestamp("2023-06-30").date()].set_index("symbol")

    assert rows.loc["S10", "volatility_63d_bucket"] == "high_volatility"
    assert rows.loc["S10", "momentum_63d_sector_bucket"] == "strong_momentum"
    assert rows.loc["S10", "drawdown_52w_sector_bucket"] == "sector_relative_near_52w_high"
    assert bool(rows.loc["S10", "higher_conviction_price_strength_candidate"])
    assert rows.loc["S10", "price_strength_scorecard_bucket"] == "higher_conviction_price_strength"
    assert rows.loc["S10", "price_strength_score_v0"] == 3
    assert bool(rows.loc["S10", "is_scorecard_bucket_eligible"])

    assert bool(rows.loc["S08", "price_strength_candidate"])
    assert bool(rows.loc["S08", "high_volatility_weak_momentum_trap"])
    assert rows.loc["S08", "price_strength_scorecard_bucket"] == "price_strength_candidate"
    assert rows.loc["S08", "price_strength_score_v0"] == 2

    assert rows.loc["S07", "price_strength_scorecard_bucket"] == "momentum_resilience_candidate"
    assert rows.loc["S07", "price_strength_score_v0"] == 1
    assert rows.loc["S09", "price_strength_scorecard_bucket"] == "high_volatility_trap"
    assert rows.loc["S09", "price_strength_score_v0"] == -1
    assert rows.loc["S01", "price_strength_scorecard_bucket"] == "neutral"
    assert rows.loc["S01", "price_strength_score_v0"] == 0
    assert build_summary["bottom_label_available"]
    assert build_summary["scorecard_bucket_eligible_rows"] == 40
    assert build_summary["scorecard_bucket_ineligible_rows"] == 0
    assert not summary.empty


def test_summary_metrics_windows_and_baselines() -> None:
    scorecard, summary, build_summary = compute_price_strength_scorecard(*_sample_inputs())
    windows = set(summary["time_window"])

    assert windows == {"full_completed_history", "last_10y_completed", "last_5y_completed", "current_partial_year"}
    assert build_summary["partial_year_handling"]["current_partial_year"] == 2026
    assert build_summary["partial_year_handling"]["latest_completed_stability_year"] == 2025

    rows = summary.set_index(["time_window", "price_strength_scorecard_bucket"])
    higher = rows.loc[("full_completed_history", "higher_conviction_price_strength")]
    assert higher["row_count"] == 3
    assert higher["symbol_count"] == 1
    assert higher["avg_forward_63d_sector_relative_return"] == pytest.approx(0.10)
    assert higher["median_forward_63d_sector_relative_return"] == pytest.approx(0.10)
    assert higher["top_30pct_sector_flag_rate"] == pytest.approx(1.0)
    assert higher["bottom_30pct_sector_flag_rate"] == pytest.approx(0.0)
    assert higher["avg_forward_return_vs_baseline"] == pytest.approx(0.045)
    assert higher["median_forward_return_vs_baseline"] == pytest.approx(0.045)
    assert higher["top_30pct_flag_rate_vs_baseline"] == pytest.approx(0.7)
    assert higher["bottom_30pct_flag_rate_vs_baseline"] == pytest.approx(-0.3)

    current = summary[summary["time_window"] == "current_partial_year"]
    assert set(current["price_strength_scorecard_bucket"]) == set(scorecard["price_strength_scorecard_bucket"])


def test_yearly_scorecard_bucket_metrics_baselines_and_partial_year_policy() -> None:
    scorecard, _summary, _build_summary = compute_price_strength_scorecard(*_sample_inputs())

    years = compute_scorecard_years(scorecard, bottom_label_available=True)

    assert set(years["calendar_year"]) == {2023, 2024, 2025, 2026}
    assert years[years["calendar_year"] == 2026]["is_partial_year"].all()
    assert not years[years["calendar_year"] == 2026]["included_in_completed_year_stability"].any()
    assert years[years["calendar_year"] == 2025]["included_in_completed_year_stability"].all()

    rows = years.set_index(["calendar_year", "price_strength_scorecard_bucket"])
    higher = rows.loc[(2023, "higher_conviction_price_strength")]
    assert higher["row_count"] == 1
    assert higher["symbol_count"] == 1
    assert higher["avg_forward_63d_sector_relative_return"] == pytest.approx(0.10)
    assert higher["median_forward_63d_sector_relative_return"] == pytest.approx(0.10)
    assert higher["top_30pct_sector_flag_rate"] == pytest.approx(1.0)
    assert higher["bottom_30pct_sector_flag_rate"] == pytest.approx(0.0)
    assert higher["avg_forward_return_vs_baseline"] == pytest.approx(0.045)
    assert higher["median_forward_return_vs_baseline"] == pytest.approx(0.045)
    assert higher["top_30pct_flag_rate_vs_baseline"] == pytest.approx(0.7)
    assert higher["bottom_30pct_flag_rate_vs_baseline"] == pytest.approx(-0.3)


def test_sparse_year_with_missing_bucket_inputs_is_excluded_from_diagnostics() -> None:
    price_features, relative_features, labels = _sample_inputs()
    sparse_date = pd.Timestamp("2023-06-30").date()
    price_features.loc[price_features["date"] == sparse_date, ["volatility_63d", "average_dollar_volume_63d"]] = pd.NA
    relative_features.loc[
        relative_features["date"] == sparse_date,
        ["return_63d_sector_pct_rank", "drawdown_from_52w_high_sector_pct_rank"],
    ] = pd.NA

    scorecard, summary, build_summary = compute_price_strength_scorecard(price_features, relative_features, labels)
    years = compute_scorecard_years(scorecard, bottom_label_available=True)
    stability = compute_scorecard_stability(years, bottom_label_available=True)

    sparse_rows = scorecard[pd.to_datetime(scorecard["date"]).dt.year == 2023]
    assert len(sparse_rows) == 10
    assert not sparse_rows["is_scorecard_bucket_eligible"].any()
    assert set(sparse_rows["price_strength_scorecard_bucket"]) == {"neutral"}
    assert build_summary["scorecard_bucket_eligible_rows"] == 30
    assert build_summary["scorecard_bucket_ineligible_rows"] == 10
    assert 2023 not in set(years["calendar_year"])
    assert "full_completed_history" in set(summary["time_window"])
    assert stability["completed_years_count"].max() == 2


def test_scorecard_bucket_stability_summary_excludes_partial_year() -> None:
    scorecard, _summary, _build_summary = compute_price_strength_scorecard(*_sample_inputs())
    years = compute_scorecard_years(scorecard, bottom_label_available=True)

    stability = compute_scorecard_stability(years, bottom_label_available=True)

    rows = stability.set_index("price_strength_scorecard_bucket")
    higher = rows.loc["higher_conviction_price_strength"]
    assert higher["completed_years_count"] == 3
    assert higher["years_avg_above_baseline"] == 3
    assert higher["years_median_above_baseline"] == 3
    assert higher["years_top_rate_above_baseline"] == 3
    assert higher["years_bottom_rate_below_baseline"] == 3
    assert higher["pct_years_bottom_rate_below_baseline"] == pytest.approx(1.0)
    assert higher["mean_avg_return_vs_baseline"] == pytest.approx(0.045)
    assert higher["stability_assessment"] == "broadly_positive"

    trap = rows.loc["high_volatility_trap"]
    assert trap["completed_years_count"] == 3
    assert trap["stability_assessment"] in {
        "broadly_positive",
        "positive_but_high_risk",
        "tail_driven",
        "neutral_or_defensive",
        "negative_or_trap",
        "mixed_or_regime_dependent",
    }


def test_stability_assessment_labels() -> None:
    base = {
        "completed_years_count": 5,
        "pct_years_avg_above_baseline": 0.8,
        "pct_years_median_above_baseline": 0.8,
        "pct_years_top_rate_above_baseline": 0.8,
        "pct_years_bottom_rate_below_baseline": 0.8,
        "mean_avg_return_vs_baseline": 0.03,
        "mean_median_return_vs_baseline": 0.02,
        "mean_top_rate_vs_baseline": 0.10,
        "mean_bottom_rate_vs_baseline": -0.05,
    }

    assert assess_stability({"completed_years_count": 2}, bottom_label_available=True) == "insufficient_data"
    assert assess_stability(base, bottom_label_available=True) == "broadly_positive"
    assert (
        assess_stability(
            {
                **base,
                "pct_years_median_above_baseline": 0.6,
                "pct_years_bottom_rate_below_baseline": 0.2,
                "mean_median_return_vs_baseline": 0.01,
                "mean_bottom_rate_vs_baseline": 0.04,
            },
            bottom_label_available=True,
        )
        == "positive_but_high_risk"
    )
    assert (
        assess_stability(
            {
                **base,
                "pct_years_median_above_baseline": 0.2,
                "mean_median_return_vs_baseline": -0.01,
                "pct_years_bottom_rate_below_baseline": 0.2,
                "mean_bottom_rate_vs_baseline": 0.04,
            },
            bottom_label_available=True,
        )
        == "tail_driven"
    )
    assert (
        assess_stability(
            {
                **base,
                "pct_years_avg_above_baseline": 0.2,
                "pct_years_median_above_baseline": 0.2,
                "pct_years_top_rate_above_baseline": 0.2,
                "pct_years_bottom_rate_below_baseline": 0.8,
                "mean_avg_return_vs_baseline": -0.01,
                "mean_median_return_vs_baseline": -0.01,
                "mean_top_rate_vs_baseline": -0.05,
                "mean_bottom_rate_vs_baseline": -0.05,
            },
            bottom_label_available=True,
        )
        == "neutral_or_defensive"
    )
    assert (
        assess_stability(
            {
                **base,
                "pct_years_avg_above_baseline": 0.2,
                "pct_years_median_above_baseline": 0.2,
                "pct_years_top_rate_above_baseline": 0.2,
                "pct_years_bottom_rate_below_baseline": 0.2,
                "mean_avg_return_vs_baseline": -0.01,
                "mean_median_return_vs_baseline": -0.01,
                "mean_top_rate_vs_baseline": -0.05,
                "mean_bottom_rate_vs_baseline": 0.05,
            },
            bottom_label_available=True,
        )
        == "negative_or_trap"
    )


def test_missing_required_feature_raises_helpful_error() -> None:
    price_features, relative_features, labels = _sample_inputs()
    price_features = price_features.drop(columns=["average_dollar_volume_63d"])

    with pytest.raises(ValidationError, match="missing required source features"):
        compute_price_strength_scorecard(price_features, relative_features, labels)


def test_build_price_strength_scorecard_writes_outputs_report_and_metadata(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    price_features, relative_features, labels = _sample_inputs()
    _write_parquet(price_features, data_root / "features" / "equity_price_features.parquet")
    _write_parquet(relative_features, data_root / "features" / "equity_relative_features.parquet")
    _write_parquet(labels, data_root / "labels" / "equity_forward_return_labels.parquet")

    paths, build_summary = build_price_strength_scorecard(data_root)

    assert paths["scorecard_parquet"].exists()
    assert paths["scorecard_csv"].exists()
    assert paths["summary_parquet"].exists()
    assert paths["summary_csv"].exists()
    assert paths["years_parquet"].exists()
    assert paths["years_csv"].exists()
    assert paths["stability_parquet"].exists()
    assert paths["stability_csv"].exists()
    assert paths["markdown_report"].exists()
    assert paths["metadata"].exists()
    assert build_summary["rows_analyzed"] == 40
    report = paths["markdown_report"].read_text(encoding="utf-8")
    assert "# Equity Price Strength Scorecard v0" in report
    assert "## Scorecard Logic" in report
    assert "## Current Partial Year Snapshot" in report
    assert "## Scorecard Bucket Stability" in report
    assert "research-only" in report

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["dataset"] == "equity_price_strength_scorecard_v0"
    assert metadata["dataset_type"] == "research_scorecard_prototype"
    assert metadata["research_only"] is True
    assert metadata["scorecard_version"] == "v0"
    assert "scorecard_rules" in metadata
    assert "score_values" in metadata
    assert "scorecard_bucket_eligibility" in metadata
    assert "baseline_definition" in metadata
    assert "partial_year_handling" in metadata
    assert "completed_year_stability_logic" in metadata
    assert "stability_assessment_labels" in metadata
    assert metadata["output_paths"]["scorecard_parquet"] == str(paths["scorecard_parquet"])
    assert metadata["output_paths"]["years_parquet"] == str(paths["years_parquet"])
    assert metadata["output_paths"]["stability_parquet"] == str(paths["stability_parquet"])


def test_cli_price_strength_scorecard_v0_writes_expected_outputs(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    price_features, relative_features, labels = _sample_inputs()
    _write_parquet(price_features, data_root / "features" / "equity_price_features.parquet")
    _write_parquet(relative_features, data_root / "features" / "equity_relative_features.parquet")
    _write_parquet(labels, data_root / "labels" / "equity_forward_return_labels.parquet")
    monkeypatch.setenv("FINBOT_DATA_ROOT", str(data_root))
    runner = CliRunner()

    result = runner.invoke(app, ["price-strength-scorecard-v0"])

    assert result.exit_code == 0, result.output
    assert "Price strength scorecard v0 complete." in result.output
    assert (data_root / "research" / "price_strength_scorecard_v0" / "equity_price_strength_scorecard_v0.parquet").exists()
    assert (data_root / "research" / "price_strength_scorecard_v0" / "equity_price_strength_scorecard_v0.csv").exists()
    assert (data_root / "research" / "price_strength_scorecard_v0" / "equity_price_strength_scorecard_v0_summary.parquet").exists()
    assert (data_root / "research" / "price_strength_scorecard_v0" / "equity_price_strength_scorecard_v0_summary.csv").exists()
    assert (data_root / "research" / "price_strength_scorecard_v0" / "equity_price_strength_scorecard_v0_years.parquet").exists()
    assert (data_root / "research" / "price_strength_scorecard_v0" / "equity_price_strength_scorecard_v0_years.csv").exists()
    assert (data_root / "research" / "price_strength_scorecard_v0" / "equity_price_strength_scorecard_v0_stability.parquet").exists()
    assert (data_root / "research" / "price_strength_scorecard_v0" / "equity_price_strength_scorecard_v0_stability.csv").exists()
    assert (data_root / "research" / "price_strength_scorecard_v0" / "equity_price_strength_scorecard_v0_report.md").exists()
    assert (data_root / "research" / "price_strength_scorecard_v0" / "equity_price_strength_scorecard_v0.metadata.json").exists()


def _sample_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = [pd.Timestamp(f"{year}-06-30").date() for year in (2023, 2024, 2025, 2026)]
    symbols = [f"S{idx:02d}" for idx in range(1, 11)]
    price_rows = []
    relative_rows = []
    label_rows = []
    momentum = {
        "S07": 0.90,
        "S08": 0.20,
        "S09": 0.50,
        "S10": 0.90,
    }
    drawdown = {
        "S07": 0.90,
        "S08": 0.90,
        "S09": 0.20,
        "S10": 0.90,
    }
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
                    "return_63d_sector_pct_rank": momentum.get(symbol, 0.50),
                    "drawdown_from_52w_high_sector_pct_rank": drawdown.get(symbol, 0.50),
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
