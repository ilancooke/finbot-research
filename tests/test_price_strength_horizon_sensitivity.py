from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from finbot_research.cli import app
from finbot_research.price_strength_equity_curve_backtest import (
    build_equity_curve_constituents,
    build_vintages,
    compute_daily_equity_curve,
    compute_vintage_daily_returns,
    load_adjusted_close_returns,
    prepare_equity_curve_input,
)
from finbot_research.price_strength_horizon_sensitivity import (
    apply_transaction_cost,
    build_price_strength_horizon_sensitivity,
    compute_horizon_summary,
    compute_turnover_summary,
    prepare_daily_output,
    select_best_variants,
    select_horizon_rebalance_rows,
)


def test_monthly_and_quarterly_rebalance_selection() -> None:
    eligible = prepare_equity_curve_input(_sample_scorecard())

    monthly = select_horizon_rebalance_rows(eligible, rebalance_frequency="monthly")
    quarterly = select_horizon_rebalance_rows(eligible, rebalance_frequency="quarterly")

    assert sorted(monthly["rebalance_date"].unique()) == [
        pd.Timestamp("2024-01-31").date(),
        pd.Timestamp("2024-02-29").date(),
        pd.Timestamp("2024-03-28").date(),
    ]
    assert sorted(quarterly["rebalance_date"].unique()) == [pd.Timestamp("2024-03-28").date()]


def test_holding_period_end_active_vintages_and_costs(tmp_path: Path) -> None:
    data_root = _write_inputs(tmp_path)
    eligible = prepare_equity_curve_input(_sample_scorecard())
    rebalance_rows = select_horizon_rebalance_rows(eligible, rebalance_frequency="monthly")
    constituents, sector_exposure = build_equity_curve_constituents(rebalance_rows, sector_cap=0.30)
    constituents = constituents[constituents["portfolio_name"].isin(["higher_conviction_raw", "eligible_universe_baseline"])]
    sector_exposure = sector_exposure[sector_exposure["portfolio_name"].isin(["higher_conviction_raw", "eligible_universe_baseline"])]
    returns, trading_dates = load_adjusted_close_returns(_daily_bars_path(data_root), symbols=["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"])

    vintages, constituents, _, holding_calendar = build_vintages(
        constituents,
        sector_exposure,
        trading_dates=trading_dates,
        holding_period_days=2,
        sector_cap=0.30,
        transaction_cost_bps=0,
    )
    jan = vintages.set_index(["rebalance_date", "portfolio_name"]).loc[
        (pd.Timestamp("2024-01-31").date(), "higher_conviction_raw")
    ]
    assert jan["holding_end_date"] == pd.Timestamp("2024-02-02").date()
    assert holding_calendar.groupby("vintage_id")["date"].size().max() == 2

    costed = apply_transaction_cost(vintages, transaction_cost_bps=50)
    assert costed["transaction_cost_bps"].eq(50).all()
    assert costed["transaction_cost"].max() > 0
    vintage_daily = compute_vintage_daily_returns(constituents, holding_calendar, returns)
    daily = compute_daily_equity_curve(vintage_daily, costed)
    output = prepare_daily_output(
        daily,
        rebalance_frequency="monthly",
        holding_period_days=2,
        transaction_cost_bps=50,
    )
    assert output["active_vintage_count"].max() >= 1
    assert "net_excess_cumulative_return_index" in output.columns


def test_sector_cap_summary_and_best_variant_selection(tmp_path: Path) -> None:
    data_root = _write_inputs(tmp_path)
    eligible = prepare_equity_curve_input(_sample_scorecard())
    rebalance_rows = select_horizon_rebalance_rows(eligible, rebalance_frequency="monthly")
    constituents, sector_exposure = build_equity_curve_constituents(rebalance_rows, sector_cap=0.30)
    constituents = constituents[
        constituents["portfolio_name"].isin(
            ["higher_conviction_raw", "higher_conviction_sector_capped", "eligible_universe_baseline"]
        )
    ]
    sector_exposure = sector_exposure[
        sector_exposure["portfolio_name"].isin(
            ["higher_conviction_raw", "higher_conviction_sector_capped", "eligible_universe_baseline"]
        )
    ]
    returns, trading_dates = load_adjusted_close_returns(_daily_bars_path(data_root), symbols=["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"])
    vintages, constituents, sector_exposure, holding_calendar = build_vintages(
        constituents,
        sector_exposure,
        trading_dates=trading_dates,
        holding_period_days=4,
        sector_cap=0.30,
        transaction_cost_bps=25,
    )
    vintage_daily = compute_vintage_daily_returns(constituents, holding_calendar, returns)
    daily = compute_daily_equity_curve(vintage_daily, vintages)

    summary = compute_horizon_summary(
        daily,
        vintages,
        sector_exposure,
        rebalance_frequency="monthly",
        holding_period_days=4,
        transaction_cost_bps=25,
    )
    turnover = compute_turnover_summary(
        vintages,
        rebalance_frequency="monthly",
        holding_period_days=4,
        transaction_cost_bps=25,
    )
    best = select_best_variants(summary)

    capped = summary[summary["portfolio_name"] == "higher_conviction_sector_capped"]
    assert capped["median_max_sector_weight"].max() <= 0.3000001
    assert {"mean_one_way_turnover", "p75_one_way_turnover"}.issubset(set(turnover.columns))
    assert "best_turnover_adjusted_variant" in set(best["selection_criterion"])


def test_build_writes_outputs_metadata_and_report(tmp_path: Path) -> None:
    data_root = _write_inputs(tmp_path)

    paths, run_summary = build_price_strength_horizon_sensitivity(data_root)

    assert run_summary["summary_rows"] == 72
    assert run_summary["best_variant_rows"] == 5
    assert paths["summary_parquet"].exists()
    assert paths["summary_csv"].exists()
    assert paths["best_variants_parquet"].exists()
    assert paths["best_variants_csv"].exists()
    assert paths["daily_parquet"].exists()
    assert paths["turnover_parquet"].exists()
    assert paths["turnover_csv"].exists()
    assert paths["markdown_report"].exists()
    assert paths["metadata"].name == "equity_price_strength_horizon_sensitivity.metadata.json"
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["rebalance_frequencies"] == ["monthly", "quarterly"]
    assert "daily_parquet" in metadata["output_paths"]
    report = paths["markdown_report"].read_text(encoding="utf-8")
    for section in [
        "## Purpose",
        "## Method",
        "## Sensitivity Grid",
        "## Executive Summary",
        "## Monthly vs Quarterly Results",
        "## Holding-Period Sensitivity",
        "## Transaction-Cost Interaction",
        "## Raw vs Sector-Capped Comparison",
        "## Turnover Summary",
        "## Best Variants",
        "## Interpretation",
        "## Important Caveats",
        "## Output File Guide",
        "## Suggested Next Step",
    ]:
        assert section in report


def test_cli_price_strength_horizon_sensitivity_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    data_root = _write_inputs(tmp_path)
    monkeypatch.setenv("FINBOT_DATA_ROOT", str(data_root))
    runner = CliRunner()

    result = runner.invoke(app, ["price-strength-horizon-sensitivity"])

    assert result.exit_code == 0, result.output
    assert "Price strength horizon sensitivity complete." in result.output
    assert (
        data_root
        / "research"
        / "price_strength_horizon_sensitivity"
        / "equity_price_strength_horizon_sensitivity_summary.parquet"
    ).exists()


def _sample_scorecard() -> pd.DataFrame:
    rows = []
    specs = {
        "2024-01-31": {
            "AAA": ("higher_conviction_price_strength", "Technology"),
            "BBB": ("higher_conviction_price_strength", "Technology"),
            "CCC": ("higher_conviction_price_strength", "Healthcare"),
            "DDD": ("higher_conviction_price_strength", "Financials"),
            "EEE": ("higher_conviction_price_strength", "Industrials"),
            "FFF": ("price_strength_candidate", "Utilities"),
        },
        "2024-02-29": {
            "AAA": ("higher_conviction_price_strength", "Technology"),
            "CCC": ("higher_conviction_price_strength", "Healthcare"),
            "DDD": ("higher_conviction_price_strength", "Financials"),
            "EEE": ("higher_conviction_price_strength", "Industrials"),
            "BBB": ("high_volatility_trap", "Technology"),
            "FFF": ("price_strength_candidate", "Utilities"),
        },
        "2024-03-28": {
            "AAA": ("higher_conviction_price_strength", "Technology"),
            "BBB": ("higher_conviction_price_strength", "Technology"),
            "CCC": ("higher_conviction_price_strength", "Healthcare"),
            "DDD": ("higher_conviction_price_strength", "Financials"),
            "EEE": ("higher_conviction_price_strength", "Industrials"),
            "FFF": ("price_strength_candidate", "Utilities"),
        },
    }
    for date, symbols in specs.items():
        for symbol, (bucket, sector) in symbols.items():
            rows.append(
                {
                    "symbol": symbol,
                    "date": pd.Timestamp(date).date(),
                    "price_strength_scorecard_bucket": bucket,
                    "price_strength_score_v0": {
                        "higher_conviction_price_strength": 3,
                        "price_strength_candidate": 2,
                        "high_volatility_trap": -1,
                    }[bucket],
                    "is_scorecard_bucket_eligible": True,
                    "sector": sector,
                }
            )
    return pd.DataFrame(rows)


def _sample_daily_bars() -> pd.DataFrame:
    rows = []
    close = {symbol: 100.0 for symbol in ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]}
    returns_by_date = {
        "2024-01-31": 0.00,
        "2024-02-01": 0.02,
        "2024-02-02": -0.01,
        "2024-02-29": 0.03,
        "2024-03-01": 0.01,
        "2024-03-04": -0.02,
        "2024-03-28": 0.04,
        "2024-03-29": 0.01,
        "2024-04-01": 0.02,
    }
    for date, daily_return in returns_by_date.items():
        for symbol in close:
            symbol_adjustment = 0.01 if symbol in {"AAA", "CCC"} else 0.0
            close[symbol] *= 1 + daily_return + symbol_adjustment
            rows.append({"date": pd.Timestamp(date).date(), "symbol": symbol, "closeadj": close[symbol]})
    return pd.DataFrame(rows)


def _write_inputs(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    _write_parquet(_sample_scorecard(), _scorecard_path(data_root))
    _write_parquet(_sample_daily_bars(), _daily_bars_path(data_root))
    return data_root


def _scorecard_path(data_root: Path) -> Path:
    return data_root / "research" / "price_strength_scorecard_v0" / "equity_price_strength_scorecard_v0.parquet"


def _daily_bars_path(data_root: Path) -> Path:
    return data_root / "market" / "daily_bars" / "historical.parquet"


def _write_parquet(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(path, index=False)
