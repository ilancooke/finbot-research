from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from finbot_research.cli import app
from finbot_research.price_strength_portfolio_simulation import (
    build_portfolio_constituents,
    build_price_strength_portfolio_simulation,
    compute_rebalance_results,
    compute_summary,
    equal_weights,
    max_drawdown,
    one_way_turnover,
    prepare_portfolio_input,
    sector_capped_weights,
)
from finbot_research.price_strength_rebalance_feasibility import select_rebalance_rows


def test_portfolio_membership_and_weights() -> None:
    rebalance_rows = select_rebalance_rows(prepare_portfolio_input(_sample_scorecard()))

    constituents, sector_weights = build_portfolio_constituents(rebalance_rows, sector_cap=0.30)

    jan = constituents[
        (constituents["rebalance_date"] == pd.Timestamp("2024-01-31").date())
        & (constituents["portfolio_name"] == "higher_conviction_raw")
    ]
    assert set(jan["symbol"]) == {"AAA", "BBB", "CCC", "DDD", "EEE"}
    assert jan["weight"].sum() == pytest.approx(1.0)
    assert set(jan["weight"]) == {0.20}
    capped = sector_weights[
        (sector_weights["rebalance_date"] == pd.Timestamp("2024-01-31").date())
        & (sector_weights["portfolio_name"] == "higher_conviction_sector_capped")
    ]
    assert capped["sector_weight"].sum() == pytest.approx(1.0)
    assert capped["sector_weight"].max() <= 0.3000001


def test_weight_helpers_and_turnover() -> None:
    rows = _sample_scorecard()
    group = rows[rows["date"] == pd.Timestamp("2024-01-31").date()]

    assert sum(equal_weights(group).values()) == pytest.approx(1.0)
    capped = sector_capped_weights(group, sector_cap=0.30)
    assert sum(capped.values()) == pytest.approx(1.0)
    sector_weights = group.assign(weight=group["symbol"].map(capped)).groupby("sector")["weight"].sum()
    assert sector_weights.max() <= 0.3000001
    assert one_way_turnover({"AAA": 0.5, "BBB": 0.5}, {"AAA": 0.25, "CCC": 0.75}) == pytest.approx(0.75)


def test_rebalance_results_costs_net_returns_and_baseline_fields() -> None:
    rebalance_rows = select_rebalance_rows(prepare_portfolio_input(_sample_scorecard()))
    constituents, sector_weights = build_portfolio_constituents(rebalance_rows, sector_cap=0.30)

    results = compute_rebalance_results(constituents, sector_weights, sector_cap=0.30, transaction_cost_bps=25)

    row = results.set_index(["rebalance_date", "portfolio_name"]).loc[
        (pd.Timestamp("2024-01-31").date(), "higher_conviction_raw")
    ]
    assert row["one_way_turnover"] == pytest.approx(1.0)
    assert row["transaction_cost"] == pytest.approx(0.0025)
    assert row["net_portfolio_forward_63d_sector_relative_return"] == pytest.approx(
        row["portfolio_forward_63d_sector_relative_return"] - 0.0025
    )
    assert "portfolio_forward_return_vs_baseline" in results.columns
    assert "net_portfolio_forward_return_vs_baseline" in results.columns


def test_summary_metrics() -> None:
    rebalance_rows = select_rebalance_rows(prepare_portfolio_input(_sample_scorecard()))
    constituents, sector_weights = build_portfolio_constituents(rebalance_rows, sector_cap=0.30)
    results = compute_rebalance_results(constituents, sector_weights, sector_cap=0.30, transaction_cost_bps=25)

    summary = compute_summary(results)

    higher = summary.set_index("portfolio_name").loc["higher_conviction_sector_capped"]
    assert higher["rebalance_date_count"] == 2
    assert higher["median_symbol_count"] == pytest.approx(4.5)
    assert "mean_net_vs_baseline" in summary.columns
    assert "std_net_forward_return" in summary.columns
    assert "max_drawdown_net_forward_return_index" in summary.columns
    assert max_drawdown(pd.Series([0.10, -0.05])) == pytest.approx(-0.05)


def test_build_writes_outputs_metadata_and_report(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_parquet(_sample_scorecard().drop(columns=["sector"]), _scorecard_path(data_root))
    _write_parquet(
        pd.DataFrame(
            [
                {"ticker": "AAA", "sector": "Technology"},
                {"ticker": "BBB", "sector": "Technology"},
                {"ticker": "CCC", "sector": "Healthcare"},
                {"ticker": "DDD", "sector": "Financials"},
                {"ticker": "EEE", "sector": "Industrials"},
                {"ticker": "FFF", "sector": "Utilities"},
            ]
        ),
        data_root / "reference" / "tickers.parquet",
    )

    paths, summary = build_price_strength_portfolio_simulation(data_root, sector_cap=0.30, transaction_cost_bps=25)

    assert summary["rebalance_date_count"] == 2
    assert paths["rebalance_results_parquet"].exists()
    assert paths["summary_parquet"].exists()
    assert paths["constituents_parquet"].exists()
    assert paths["sector_weights_parquet"].exists()
    assert paths["markdown_report"].exists()
    assert paths["metadata"].name == "equity_price_strength_portfolio_simulation.metadata.json"
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["sector_cap"] == 0.30
    assert metadata["transaction_cost_bps"] == 25
    assert "rebalance_results_parquet" in metadata["output_paths"]
    report = paths["markdown_report"].read_text(encoding="utf-8")
    assert "# Equity Price Strength Portfolio Simulation" in report
    assert "## Raw vs Sector-Capped Comparison" in report
    assert "## Suggested Next Step" in report


def test_build_fails_when_sector_unavailable(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_parquet(_sample_scorecard().drop(columns=["sector"]), _scorecard_path(data_root))

    with pytest.raises(Exception, match="requires sector"):
        build_price_strength_portfolio_simulation(data_root)


def test_cli_price_strength_portfolio_simulation_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    _write_parquet(_sample_scorecard(), _scorecard_path(data_root))
    monkeypatch.setenv("FINBOT_DATA_ROOT", str(data_root))
    runner = CliRunner()

    result = runner.invoke(app, ["price-strength-portfolio-simulation", "--sector-cap", "0.30", "--transaction-cost-bps", "25"])

    assert result.exit_code == 0, result.output
    assert "Price strength portfolio simulation complete." in result.output
    assert (
        data_root
        / "research"
        / "price_strength_portfolio_simulation"
        / "equity_price_strength_portfolio_summary.parquet"
    ).exists()


def _sample_scorecard() -> pd.DataFrame:
    rows = []
    specs = {
        "2024-01-31": {
            "AAA": ("higher_conviction_price_strength", "Technology", 0.10),
            "BBB": ("higher_conviction_price_strength", "Technology", 0.05),
            "CCC": ("higher_conviction_price_strength", "Healthcare", 0.04),
            "DDD": ("higher_conviction_price_strength", "Financials", 0.03),
            "EEE": ("higher_conviction_price_strength", "Industrials", 0.02),
            "FFF": ("price_strength_candidate", "Utilities", 0.01),
        },
        "2024-02-29": {
            "AAA": ("higher_conviction_price_strength", "Technology", 0.06),
            "CCC": ("higher_conviction_price_strength", "Healthcare", 0.04),
            "DDD": ("higher_conviction_price_strength", "Financials", 0.02),
            "EEE": ("higher_conviction_price_strength", "Industrials", 0.01),
            "BBB": ("high_volatility_trap", "Technology", -0.03),
        },
    }
    for date, symbols in specs.items():
        for symbol, (bucket, sector, forward_return) in symbols.items():
            rows.append(
                {
                    "symbol": symbol,
                    "date": pd.Timestamp(date).date(),
                    "price_strength_scorecard_bucket": bucket,
                    "price_strength_score_v0": {"higher_conviction_price_strength": 3, "price_strength_candidate": 2, "high_volatility_trap": -1}[bucket],
                    "is_scorecard_bucket_eligible": True,
                    "sector": sector,
                    "forward_63d_sector_relative_return": forward_return,
                    "forward_63d_top_30pct_sector_flag": 1 if forward_return >= 0.05 else 0,
                    "forward_63d_bottom_30pct_sector_flag": 1 if forward_return < 0 else 0,
                }
            )
    return pd.DataFrame(rows)


def _scorecard_path(data_root: Path) -> Path:
    return data_root / "research" / "price_strength_scorecard_v0" / "equity_price_strength_scorecard_v0.parquet"


def _write_parquet(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(path, index=False)
