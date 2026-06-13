from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from finbot_research.cli import app
from finbot_research.price_strength_scorecard_v1 import (
    DEFAULT_RESEARCH_EXPRESSION,
    PRACTICAL_DEFAULT_EXPRESSION,
    build_evidence_summary,
    build_price_strength_scorecard_v1,
    compute_current_snapshot,
    compute_scorecard_v1,
    compute_scorecard_v1_current_summary,
    compute_scorecard_v1_summary,
)
from finbot_research.validation import ValidationError


def test_v0_to_v1_bucket_mapping_scores_labels_and_flags() -> None:
    scorecard = compute_scorecard_v1(_sample_scorecard_v0())
    rows = scorecard.set_index("symbol")

    assert rows.loc["AAA", "price_strength_score_v1"] == 3
    assert rows.loc["AAA", "price_strength_bucket_v1"] == "high_conviction_price_strength"
    assert rows.loc["AAA", "price_strength_confidence_v1"] == "high"
    assert rows.loc["AAA", "price_strength_risk_label_v1"] == "aggressive_upside_high_drawdown"
    assert rows.loc["AAA", "price_strength_research_label_v1"] == "primary_price_strength_signal"
    assert bool(rows.loc["AAA", "is_high_conviction_price_strength_v1"])
    assert rows.loc["BBB", "price_strength_bucket_v1"] == "moderate_price_strength"
    assert bool(rows.loc["BBB", "is_moderate_price_strength_v1"])
    assert rows.loc["CCC", "price_strength_bucket_v1"] == "momentum_resilience"
    assert bool(rows.loc["CCC", "is_momentum_resilience_v1"])
    assert rows.loc["DDD", "price_strength_bucket_v1"] == "high_volatility_trap"
    assert bool(rows.loc["DDD", "is_high_volatility_trap_v1"])
    assert rows.loc["EEE", "price_strength_bucket_v1"] == "neutral"
    assert bool(rows.loc["EEE", "is_neutral_v1"])
    assert rows.loc["AAA", "default_research_expression_v1"] == DEFAULT_RESEARCH_EXPRESSION
    assert rows.loc["AAA", "practical_default_expression_v1"] == PRACTICAL_DEFAULT_EXPRESSION
    assert "volatility_63d_bucket" in scorecard.columns


def test_eligible_filtering_and_missing_sector_failure() -> None:
    scorecard = compute_scorecard_v1(_sample_scorecard_v0())

    assert "ZZZ" not in set(scorecard["symbol"])

    missing_sector = _sample_scorecard_v0()
    missing_sector.loc[missing_sector["symbol"] == "AAA", "sector"] = pd.NA
    with pytest.raises(ValidationError, match="requires sector"):
        compute_scorecard_v1(missing_sector)


def test_current_snapshot_and_summaries() -> None:
    v1 = compute_scorecard_v1(_sample_scorecard_v0(include_later=True))
    current = compute_current_snapshot(v1)
    summary = compute_scorecard_v1_summary(v1, current)
    current_summary = compute_scorecard_v1_current_summary(current)

    aaa = current[current["symbol"] == "AAA"].iloc[0]
    assert aaa["date"] == pd.Timestamp("2024-02-29").date()
    assert current.iloc[0]["price_strength_score_v1"] >= current.iloc[-1]["price_strength_score_v1"]
    high = summary[summary["price_strength_bucket_v1"] == "high_conviction_price_strength"].iloc[0]
    assert high["row_count"] == 2
    assert high["symbol_count"] == 1
    assert high["current_symbol_count"] == 1
    assert high["pct_of_current_universe"] > 0
    assert "current_symbol_count" in current_summary.columns


def test_evidence_summary_with_and_without_optional_files(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    evidence, sources = build_evidence_summary(data_root)

    assert len(evidence) == 6
    assert sources["found_paths"] == []
    assert sources["missing_paths"]

    _write_parquet(_sample_best_variants(), _best_variants_path(data_root))
    _write_parquet(_sample_focus(), _focus_path(data_root))
    evidence, sources = build_evidence_summary(data_root)
    primary = evidence.set_index("evidence_topic").loc["primary_research_expression"]
    practical = evidence.set_index("evidence_topic").loc["practical_default_expression"]
    assert float(primary["supporting_metric_1_value"]) == pytest.approx(0.072)
    assert practical["evidence_label"] == "quarterly_126d_sector_capped_50bps_practical_default"
    assert len(sources["found_paths"]) == 2


def test_build_writes_outputs_metadata_and_report(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_parquet(_sample_scorecard_v0(include_later=True), _scorecard_path(data_root))
    _write_parquet(_sample_best_variants(), _best_variants_path(data_root))
    _write_parquet(_sample_focus(), _focus_path(data_root))

    paths, run_summary = build_price_strength_scorecard_v1(data_root)

    assert run_summary["scorecard_rows"] == 6
    assert run_summary["current_rows"] == 5
    assert paths["scorecard_parquet"].exists()
    assert paths["scorecard_csv"].exists()
    assert paths["current_parquet"].exists()
    assert paths["current_csv"].exists()
    assert paths["summary_parquet"].exists()
    assert paths["summary_csv"].exists()
    assert paths["current_summary_parquet"].exists()
    assert paths["current_summary_csv"].exists()
    assert paths["evidence_summary_parquet"].exists()
    assert paths["evidence_summary_csv"].exists()
    assert paths["markdown_report"].exists()
    assert paths["metadata"].name == "equity_price_strength_scorecard_v1.metadata.json"
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["default_research_expression"] == DEFAULT_RESEARCH_EXPRESSION
    assert metadata["practical_default_expression"] == PRACTICAL_DEFAULT_EXPRESSION
    assert "scorecard_parquet" in metadata["output_paths"]
    report = paths["markdown_report"].read_text(encoding="utf-8")
    for section in [
        "## Purpose",
        "## What Changed from v0",
        "## Scorecard v1 Fields",
        "## Bucket Definitions",
        "## Current Snapshot Summary",
        "## Research Evidence Summary",
        "## Default Research Expression",
        "## Practical Default Expression",
        "## Risk Labels and Caveats",
        "## Output File Guide",
        "## Suggested Next Step",
    ]:
        assert section in report


def test_cli_price_strength_scorecard_v1_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    _write_parquet(_sample_scorecard_v0(include_later=True), _scorecard_path(data_root))
    monkeypatch.setenv("FINBOT_DATA_ROOT", str(data_root))
    runner = CliRunner()

    result = runner.invoke(app, ["price-strength-scorecard-v1"])

    assert result.exit_code == 0, result.output
    assert "Price strength scorecard v1 complete." in result.output
    assert (
        data_root
        / "research"
        / "price_strength_scorecard_v1"
        / "equity_price_strength_scorecard_v1.parquet"
    ).exists()


def _sample_scorecard_v0(*, include_later: bool = False) -> pd.DataFrame:
    rows = [
        _row("AAA", "2024-01-31", "Technology", "higher_conviction_price_strength", 3),
        _row("BBB", "2024-01-31", "Healthcare", "price_strength_candidate", 2),
        _row("CCC", "2024-01-31", "Financials", "momentum_resilience_candidate", 1),
        _row("DDD", "2024-01-31", "Energy", "high_volatility_trap", -1),
        _row("EEE", "2024-01-31", "Utilities", "neutral", 0),
        _row("ZZZ", "2024-01-31", "Technology", "neutral", 0, eligible=False),
    ]
    if include_later:
        rows.append(_row("AAA", "2024-02-29", "Technology", "higher_conviction_price_strength", 3))
    return pd.DataFrame(rows)


def _row(symbol: str, date: str, sector: str, bucket: str, score: int, *, eligible: bool = True) -> dict:
    return {
        "symbol": symbol,
        "date": pd.Timestamp(date).date(),
        "sector": sector,
        "price_strength_scorecard_bucket": bucket,
        "price_strength_score_v0": score,
        "is_scorecard_bucket_eligible": eligible,
        "volatility_63d_bucket": "high_volatility",
        "momentum_63d_sector_bucket": "strong_momentum",
        "drawdown_52w_sector_bucket": "sector_relative_near_52w_high",
        "return_63d_sector_pct_rank": 0.90,
        "drawdown_from_52w_high_sector_pct_rank": 0.90,
        "volatility_63d": 0.40,
    }


def _sample_best_variants() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "selection_criterion": "highest_net_excess_annualized_return",
                "portfolio_name": "higher_conviction_raw",
                "rebalance_frequency": "monthly",
                "holding_period_days": 63,
                "transaction_cost_bps": 25.0,
                "net_excess_annualized_return": 0.072,
                "net_annualized_return": 0.263,
                "net_sharpe_like": 0.93,
                "net_max_drawdown": -0.59,
            },
            {
                "selection_criterion": "highest_net_sharpe_like",
                "portfolio_name": "higher_conviction_sector_capped",
                "rebalance_frequency": "monthly",
                "holding_period_days": 63,
                "transaction_cost_bps": 25.0,
                "net_excess_annualized_return": 0.071,
                "net_annualized_return": 0.262,
                "net_sharpe_like": 0.94,
                "net_max_drawdown": -0.58,
            },
            {
                "selection_criterion": "best_practical_default_candidate",
                "portfolio_name": "higher_conviction_sector_capped",
                "rebalance_frequency": "quarterly",
                "holding_period_days": 126,
                "transaction_cost_bps": 50.0,
                "cost_efficiency_score": -0.002,
                "net_excess_annualized_return": 0.052,
                "estimated_annualized_cost_drag": 0.017,
                "net_max_drawdown": -0.64,
            },
        ]
    )


def _sample_focus() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rebalance_frequency": "monthly",
                "holding_period_days": 63,
                "transaction_cost_bps": 25.0,
                "portfolio_name": "higher_conviction_sector_capped",
                "net_sharpe_like": 0.94,
                "net_max_drawdown": -0.58,
                "annualized_one_way_turnover": 7.7,
            }
        ]
    )


def _scorecard_path(data_root: Path) -> Path:
    return data_root / "research" / "price_strength_scorecard_v0" / "equity_price_strength_scorecard_v0.parquet"


def _best_variants_path(data_root: Path) -> Path:
    return (
        data_root
        / "research"
        / "price_strength_turnover_cost_efficiency"
        / "equity_price_strength_turnover_cost_efficiency_best_variants.parquet"
    )


def _focus_path(data_root: Path) -> Path:
    return (
        data_root
        / "research"
        / "price_strength_turnover_cost_efficiency"
        / "equity_price_strength_turnover_cost_efficiency_focus.parquet"
    )


def _write_parquet(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(path, index=False)
