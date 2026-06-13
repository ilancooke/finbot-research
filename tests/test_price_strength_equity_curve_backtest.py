from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from finbot_research.cli import app
from finbot_research.price_strength_equity_curve_backtest import (
    annualized_return,
    annualized_volatility,
    build_equity_curve_constituents,
    build_price_strength_equity_curve_backtest,
    build_vintages,
    compute_daily_equity_curve,
    compute_drawdown,
    compute_performance_summary,
    compute_vintage_daily_returns,
    cumulative_return_index,
    load_adjusted_close_returns,
    prepare_equity_curve_input,
)
from finbot_research.price_strength_portfolio_simulation import one_way_turnover
from finbot_research.price_strength_rebalance_feasibility import select_rebalance_rows


def test_rebalance_vintages_and_weight_rules(tmp_path: Path) -> None:
    rebalance_rows = select_rebalance_rows(prepare_equity_curve_input(_sample_scorecard()))

    assert sorted(rebalance_rows["rebalance_date"].unique()) == [
        pd.Timestamp("2024-01-31").date(),
        pd.Timestamp("2024-02-29").date(),
    ]

    constituents, sector_exposure = build_equity_curve_constituents(rebalance_rows, sector_cap=0.30)
    raw = constituents[
        (constituents["rebalance_date"] == pd.Timestamp("2024-01-31").date())
        & (constituents["portfolio_name"] == "higher_conviction_raw")
    ]
    assert raw["weight"].sum() == pytest.approx(1.0)
    assert set(raw["weight"]) == {0.20}
    capped = sector_exposure[
        (sector_exposure["rebalance_date"] == pd.Timestamp("2024-01-31").date())
        & (sector_exposure["portfolio_name"] == "higher_conviction_sector_capped")
    ]
    assert capped["sector_weight"].sum() == pytest.approx(1.0)
    assert capped["sector_weight"].max() <= 0.3000001

    vintages, enriched_constituents, _, holding_calendar = build_vintages(
        constituents,
        sector_exposure,
        trading_dates=_trading_dates(),
        holding_period_days=4,
        sector_cap=0.30,
        transaction_cost_bps=25,
    )

    jan_vintage = vintages.set_index(["rebalance_date", "portfolio_name"]).loc[
        (pd.Timestamp("2024-01-31").date(), "higher_conviction_raw")
    ]
    assert jan_vintage["holding_start_date"] == pd.Timestamp("2024-02-01").date()
    assert jan_vintage["holding_end_date"] == pd.Timestamp("2024-03-01").date()
    assert jan_vintage["one_way_turnover"] == pytest.approx(1.0)
    assert jan_vintage["transaction_cost"] == pytest.approx(0.0025)
    assert enriched_constituents["vintage_id"].notna().all()
    assert holding_calendar.groupby("vintage_id")["date"].size().max() == 4


def test_daily_returns_overlap_costs_cumulative_and_benchmark(tmp_path: Path) -> None:
    bars_path = tmp_path / "daily_bars.parquet"
    _write_parquet(_sample_daily_bars(), bars_path)
    returns, trading_dates = load_adjusted_close_returns(bars_path, symbols=["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"])
    assert trading_dates[0] == pd.Timestamp("2024-01-31")
    aaa_feb1 = returns[(returns["symbol"] == "AAA") & (returns["date"] == pd.Timestamp("2024-02-01"))]
    assert aaa_feb1["daily_return"].iloc[0] == pytest.approx(0.10)

    rebalance_rows = select_rebalance_rows(prepare_equity_curve_input(_sample_scorecard()))
    constituents, sector_exposure = build_equity_curve_constituents(rebalance_rows, sector_cap=0.30)
    vintages, constituents, _, holding_calendar = build_vintages(
        constituents,
        sector_exposure,
        trading_dates=trading_dates,
        holding_period_days=4,
        sector_cap=0.30,
        transaction_cost_bps=25,
    )
    vintage_daily = compute_vintage_daily_returns(constituents, holding_calendar, returns)
    daily_curve = compute_daily_equity_curve(vintage_daily, vintages)

    raw = daily_curve[daily_curve["portfolio_name"] == "higher_conviction_raw"].set_index("date")
    assert raw.loc[pd.Timestamp("2024-01-31").date(), "active_vintage_count"] == 0
    assert raw.loc[pd.Timestamp("2024-01-31").date(), "net_daily_return"] == pytest.approx(-0.0025)
    assert raw.loc[pd.Timestamp("2024-02-01").date(), "gross_daily_return"] == pytest.approx(0.10)
    assert raw.loc[pd.Timestamp("2024-03-01").date(), "active_vintage_count"] == 2
    assert raw.loc[pd.Timestamp("2024-03-01").date(), "gross_daily_return"] == pytest.approx(0.10)
    assert "benchmark_net_daily_return" in daily_curve.columns
    assert "net_excess_daily_return" in daily_curve.columns
    assert cumulative_return_index(pd.Series([0.10, -0.05])).iloc[-1] == pytest.approx(1.045)
    assert compute_drawdown(pd.Series([0.10, -0.05])) == pytest.approx(-0.05)
    assert annualized_return(pd.Series([0.01] * 252)) == pytest.approx((1.01**252) - 1)
    assert annualized_volatility(pd.Series([0.01, -0.01])) > 0

    summary = compute_performance_summary(daily_curve, vintages)
    assert "gross_excess_total_return" in summary.columns
    assert "net_daily_win_rate_vs_benchmark" in summary.columns
    higher = summary.set_index("portfolio_name").loc["higher_conviction_raw"]
    assert higher["rebalance_count"] == 2
    assert higher["mean_active_vintage_count"] > 0
    assert one_way_turnover({"AAA": 0.5, "BBB": 0.5}, {"AAA": 0.25, "CCC": 0.75}) == pytest.approx(0.75)


def test_build_writes_outputs_metadata_and_report(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_parquet(_sample_scorecard(), _scorecard_path(data_root))
    _write_parquet(_sample_daily_bars(), _daily_bars_path(data_root))

    paths, run_summary = build_price_strength_equity_curve_backtest(
        data_root,
        holding_period_days=4,
        sector_cap=0.30,
        transaction_cost_bps=25,
    )

    assert run_summary["rebalance_date_count"] == 2
    assert paths["daily_parquet"].exists()
    assert paths["daily_csv"].exists()
    assert paths["vintages_parquet"].exists()
    assert paths["vintages_csv"].exists()
    assert paths["constituents_parquet"].exists()
    assert paths["summary_parquet"].exists()
    assert paths["summary_csv"].exists()
    assert paths["sector_exposure_parquet"].exists()
    assert paths["sector_exposure_csv"].exists()
    assert paths["markdown_report"].exists()
    assert paths["metadata"].name == "equity_price_strength_equity_curve_backtest.metadata.json"
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["holding_period_days"] == 4
    assert metadata["transaction_cost_bps"] == 25
    assert "daily_parquet" in metadata["output_paths"]
    report = paths["markdown_report"].read_text(encoding="utf-8")
    for section in [
        "## Purpose",
        "## Method",
        "## Portfolio Definitions",
        "## Executive Summary",
        "## Performance Summary",
        "## Raw vs Sector-Capped Comparison",
        "## Benchmark Comparison",
        "## Drawdown and Volatility",
        "## Turnover and Transaction Costs",
        "## Sector Exposure",
        "## Important Caveats",
        "## Output File Guide",
        "## Suggested Next Step",
    ]:
        assert section in report


def test_build_fails_when_sector_unavailable(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_parquet(_sample_scorecard().drop(columns=["sector"]), _scorecard_path(data_root))
    _write_parquet(_sample_daily_bars(), _daily_bars_path(data_root))

    with pytest.raises(Exception, match="requires sector"):
        build_price_strength_equity_curve_backtest(data_root, holding_period_days=4)


def test_cli_price_strength_equity_curve_backtest_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    _write_parquet(_sample_scorecard(), _scorecard_path(data_root))
    _write_parquet(_sample_daily_bars(), _daily_bars_path(data_root))
    monkeypatch.setenv("FINBOT_DATA_ROOT", str(data_root))
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "price-strength-equity-curve-backtest",
            "--holding-period-days",
            "4",
            "--sector-cap",
            "0.30",
            "--transaction-cost-bps",
            "25",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Price strength equity-curve backtest complete." in result.output
    assert (
        data_root
        / "research"
        / "price_strength_equity_curve_backtest"
        / "equity_price_strength_equity_curve_daily.parquet"
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
        "2024-02-01": 0.10,
        "2024-02-02": 0.10,
        "2024-02-29": 0.10,
        "2024-03-01": 0.10,
        "2024-03-04": 0.10,
        "2024-03-05": 0.10,
    }
    for date, daily_return in returns_by_date.items():
        for symbol in close:
            close[symbol] *= 1 + daily_return
            rows.append({"date": pd.Timestamp(date).date(), "symbol": symbol, "closeadj": close[symbol]})
    return pd.DataFrame(rows)


def _trading_dates() -> list[pd.Timestamp]:
    return [pd.Timestamp(date) for date in ["2024-01-31", "2024-02-01", "2024-02-02", "2024-02-29", "2024-03-01"]]


def _scorecard_path(data_root: Path) -> Path:
    return data_root / "research" / "price_strength_scorecard_v0" / "equity_price_strength_scorecard_v0.parquet"


def _daily_bars_path(data_root: Path) -> Path:
    return data_root / "market" / "daily_bars" / "historical.parquet"


def _write_parquet(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(path, index=False)
