from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from finbot_research.cli import app
from finbot_research.price_strength_equity_curve_backtest import build_price_strength_equity_curve_backtest
from finbot_research.price_strength_equity_curve_robustness import (
    aggregate_sector_contributions,
    aggregate_symbol_contributions,
    build_price_strength_equity_curve_robustness,
    compute_cost_sensitivity,
    compute_daily_contributions,
    compute_regime_performance,
    compute_rolling_performance,
    compute_sector_cap_sensitivity,
    read_backtest_outputs,
)
from finbot_research.price_strength_equity_curve_backtest import load_adjusted_close_returns


def test_cost_sensitivity_generation(tmp_path: Path) -> None:
    data_root = _build_base_backtest(tmp_path)
    inputs = read_backtest_outputs(data_root)

    cost = compute_cost_sensitivity(
        inputs["daily_curve"],
        inputs["vintages"],
        transaction_cost_bps_values=[0, 50],
        portfolio_names=["higher_conviction_raw", "eligible_universe_baseline"],
    )

    assert set(cost["transaction_cost_bps"]) == {0, 50}
    assert set(cost["portfolio_name"]) == {"higher_conviction_raw", "eligible_universe_baseline"}
    raw = cost[cost["portfolio_name"] == "higher_conviction_raw"].set_index("transaction_cost_bps")
    assert raw.loc[50, "mean_transaction_cost"] > raw.loc[0, "mean_transaction_cost"]
    assert raw.loc[50, "net_total_return"] < raw.loc[0, "net_total_return"]


def test_sector_cap_sensitivity_generation(tmp_path: Path) -> None:
    data_root = _build_base_backtest(tmp_path)
    metadata = json.loads(
        (
            data_root
            / "research"
            / "price_strength_equity_curve_backtest"
            / "equity_price_strength_equity_curve_backtest.metadata.json"
        ).read_text(encoding="utf-8")
    )

    inputs = read_backtest_outputs(data_root)
    sensitivity = compute_sector_cap_sensitivity(
        data_root,
        metadata,
        base_daily_curve=inputs["daily_curve"],
        base_summary=inputs["summary"],
        base_vintages=inputs["vintages"],
    )

    assert {"none", "0.20", "0.25", "0.30", "0.40"}.issubset(set(sensitivity["sector_cap"]))
    capped = sensitivity[sensitivity["portfolio_name"] == "higher_conviction_sector_capped"]
    assert capped["median_max_sector_weight"].max() <= 0.4000001
    assert "mean_one_way_turnover" in sensitivity.columns


def test_regime_and_rolling_calculations(tmp_path: Path) -> None:
    data_root = _build_base_backtest(tmp_path)
    daily = read_backtest_outputs(data_root)["daily_curve"]

    regimes = compute_regime_performance(
        daily,
        regimes=[
            ("sample_early", "2024-01-01", "2024-02-15"),
            ("sample_late", "2024-02-16", "2024-12-31"),
        ],
    )
    assert set(regimes["regime_name"]) == {"sample_early", "sample_late"}
    assert "net_excess_annualized_return" in regimes.columns

    rolling = compute_rolling_performance(daily, windows=[2, 3])
    assert set(rolling["window_trading_days"]) == {2, 3}
    assert "rolling_net_max_drawdown" in rolling.columns
    assert rolling["rolling_net_total_return"].notna().all()


def test_contribution_aggregation(tmp_path: Path) -> None:
    data_root = _build_base_backtest(tmp_path)
    inputs = read_backtest_outputs(data_root)
    constituents = inputs["constituents"]
    vintages = inputs["vintages"]
    daily = inputs["daily_curve"]
    returns, _ = load_adjusted_close_returns(
        _daily_bars_path(data_root),
        symbols=sorted(constituents["symbol"].astype(str).unique()),
    )
    selected_vintages = vintages[vintages["portfolio_name"] == "higher_conviction_raw"]
    selected_constituents = constituents[constituents["portfolio_name"] == "higher_conviction_raw"]
    contribution_daily = compute_daily_contributions(
        daily_curve=daily,
        vintages=selected_vintages,
        constituents=selected_constituents,
        adjusted_returns=returns,
        daily_dates=sorted(pd.to_datetime(daily["date"].drop_duplicates())),
    )

    sector = aggregate_sector_contributions(contribution_daily, daily)
    symbol = aggregate_symbol_contributions(contribution_daily, daily)

    assert not sector.empty
    assert not symbol.empty
    assert {"Technology", "Healthcare"}.issubset(set(sector["sector"]))
    aaa = symbol[symbol["symbol"] == "AAA"]
    assert aaa["active_day_count"].max() > 0
    assert "mean_weight_when_active" in symbol.columns


def test_build_writes_outputs_metadata_report_and_summary(tmp_path: Path) -> None:
    data_root = _build_base_backtest(tmp_path)

    paths, run_summary = build_price_strength_equity_curve_robustness(data_root)

    assert run_summary["cost_sensitivity_rows"] > 0
    assert run_summary["sector_cap_sensitivity_rows"] > 0
    assert run_summary["regime_rows"] > 0
    assert paths["cost_sensitivity_parquet"].exists()
    assert paths["sector_cap_sensitivity_parquet"].exists()
    assert paths["regime_performance_parquet"].exists()
    assert paths["rolling_performance_parquet"].exists()
    assert paths["sector_contribution_parquet"].exists()
    assert paths["symbol_contribution_parquet"].exists()
    assert paths["summary_parquet"].exists()
    assert paths["metadata"].name == "equity_price_strength_equity_curve_robustness.metadata.json"
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert "cost_sensitivity_parquet" in metadata["output_paths"]
    assert metadata["transaction_cost_assumptions_bps"] == [0.0, 25.0, 50.0, 100.0, 150.0]
    report = paths["markdown_report"].read_text(encoding="utf-8")
    for section in [
        "## Purpose",
        "## Inputs",
        "## Executive Summary",
        "## Transaction-Cost Sensitivity",
        "## Sector-Cap Sensitivity",
        "## Regime Performance",
        "## Rolling Performance",
        "## Sector Contribution",
        "## Symbol Contribution",
        "## Interpretation",
        "## Important Caveats",
        "## Output File Guide",
        "## Suggested Next Step",
    ]:
        assert section in report


def test_cli_price_strength_equity_curve_robustness_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    data_root = _build_base_backtest(tmp_path)
    monkeypatch.setenv("FINBOT_DATA_ROOT", str(data_root))
    runner = CliRunner()

    result = runner.invoke(app, ["price-strength-equity-curve-robustness"])

    assert result.exit_code == 0, result.output
    assert "Price strength equity-curve robustness diagnostics complete." in result.output
    assert (
        data_root
        / "research"
        / "price_strength_equity_curve_robustness"
        / "equity_price_strength_equity_curve_robustness_summary.parquet"
    ).exists()


def _build_base_backtest(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    _write_parquet(_sample_scorecard(), _scorecard_path(data_root))
    _write_parquet(_sample_daily_bars(), _daily_bars_path(data_root))
    build_price_strength_equity_curve_backtest(
        data_root,
        holding_period_days=4,
        sector_cap=0.30,
        transaction_cost_bps=25,
    )
    return data_root


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
    close = {
        "AAA": 100.0,
        "BBB": 100.0,
        "CCC": 100.0,
        "DDD": 100.0,
        "EEE": 100.0,
        "FFF": 100.0,
    }
    returns_by_date = {
        "2024-01-31": 0.00,
        "2024-02-01": 0.03,
        "2024-02-02": -0.01,
        "2024-02-29": 0.02,
        "2024-03-01": 0.04,
        "2024-03-04": -0.02,
        "2024-03-05": 0.01,
    }
    for date, daily_return in returns_by_date.items():
        for symbol in close:
            symbol_adjustment = 0.01 if symbol in {"AAA", "CCC"} else 0.0
            close[symbol] *= 1 + daily_return + symbol_adjustment
            rows.append({"date": pd.Timestamp(date).date(), "symbol": symbol, "closeadj": close[symbol]})
    return pd.DataFrame(rows)


def _scorecard_path(data_root: Path) -> Path:
    return data_root / "research" / "price_strength_scorecard_v0" / "equity_price_strength_scorecard_v0.parquet"


def _daily_bars_path(data_root: Path) -> Path:
    return data_root / "market" / "daily_bars" / "historical.parquet"


def _write_parquet(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(path, index=False)
