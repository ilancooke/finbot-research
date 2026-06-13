from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from finbot_research.cli import app
from finbot_research.price_strength_holding_period_simulation import (
    build_basket_memberships,
    build_price_strength_holding_period_simulation,
    compute_rebalance_results,
    compute_sector_composition,
    compute_sector_concentration,
    compute_summary,
    compute_turnover,
    prepare_simulation_input,
)
from finbot_research.price_strength_rebalance_feasibility import select_rebalance_rows


def test_basket_membership_and_monthly_rebalance_selection() -> None:
    eligible = prepare_simulation_input(_sample_scorecard())

    rebalance_rows = select_rebalance_rows(eligible)
    basket_rows = build_basket_memberships(rebalance_rows)

    assert rebalance_rows["rebalance_date"].drop_duplicates().tolist() == [
        pd.Timestamp("2024-01-31").date(),
        pd.Timestamp("2024-02-29").date(),
        pd.Timestamp("2024-03-28").date(),
    ]
    jan_positive = basket_rows[
        (basket_rows["rebalance_date"] == pd.Timestamp("2024-01-31").date())
        & (basket_rows["basket_name"] == "positive_combined")
    ]
    assert set(jan_positive["symbol"]) == {"AAA", "BBB"}
    assert "ZZZ" not in set(basket_rows["symbol"])


def test_rebalance_metrics_and_baseline_comparisons() -> None:
    basket_rows = build_basket_memberships(select_rebalance_rows(prepare_simulation_input(_sample_scorecard())))

    results = compute_rebalance_results(basket_rows)

    rows = results.set_index(["rebalance_date", "basket_name"])
    higher = rows.loc[(pd.Timestamp("2024-01-31").date(), "higher_conviction_price_strength")]
    assert higher["symbol_count"] == 2
    assert higher["avg_forward_63d_sector_relative_return"] == pytest.approx(0.075)
    assert higher["median_forward_63d_sector_relative_return"] == pytest.approx(0.075)
    assert higher["top_30pct_sector_flag_rate"] == pytest.approx(1.0)
    assert higher["bottom_30pct_sector_flag_rate"] == pytest.approx(0.0)
    assert higher["avg_forward_return_vs_baseline"] == pytest.approx(0.075 - ((0.10 + 0.05 - 0.04) / 3))
    baseline = rows.loc[(pd.Timestamp("2024-01-31").date(), "eligible_universe_baseline")]
    assert baseline["avg_forward_return_vs_baseline"] == pytest.approx(0.0)


def test_summary_metrics() -> None:
    results = compute_rebalance_results(
        build_basket_memberships(select_rebalance_rows(prepare_simulation_input(_sample_scorecard())))
    )

    summary = compute_summary(results)

    higher = summary.set_index("basket_name").loc["higher_conviction_price_strength"]
    assert higher["rebalance_date_count"] == 3
    assert higher["median_symbol_count"] == pytest.approx(2.0)
    assert "pct_rebalances_avg_above_baseline" in summary.columns


def test_turnover_calculations() -> None:
    basket_rows = build_basket_memberships(select_rebalance_rows(prepare_simulation_input(_sample_scorecard())))

    turnover = compute_turnover(basket_rows)

    higher = turnover.set_index(["previous_rebalance_date", "rebalance_date", "basket_name"]).loc[
        (
            pd.Timestamp("2024-01-31").date(),
            pd.Timestamp("2024-02-29").date(),
            "higher_conviction_price_strength",
        )
    ]
    assert higher["previous_symbol_count"] == 2
    assert higher["current_symbol_count"] == 2
    assert higher["common_symbol_count"] == 1
    assert higher["added_symbol_count"] == 1
    assert higher["removed_symbol_count"] == 1
    assert higher["jaccard_similarity"] == pytest.approx(1 / 3)
    assert higher["turnover_rate"] == pytest.approx(0.5)


def test_sector_concentration_when_sector_available() -> None:
    basket_rows = build_basket_memberships(select_rebalance_rows(prepare_simulation_input(_sample_scorecard())))

    composition = compute_sector_composition(basket_rows)
    concentration = compute_sector_concentration(composition)

    higher = concentration.set_index(["rebalance_date", "basket_name"]).loc[
        (pd.Timestamp("2024-01-31").date(), "higher_conviction_price_strength")
    ]
    assert higher["sector_count"] == 2
    assert higher["max_sector_share"] == pytest.approx(0.5)
    assert higher["top_3_sector_share"] == pytest.approx(1.0)
    assert higher["herfindahl_sector_concentration"] == pytest.approx(0.5)


def test_build_writes_outputs_metadata_and_report_with_sector(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_parquet(_sample_scorecard().drop(columns=["sector"]), _scorecard_path(data_root))
    _write_parquet(
        pd.DataFrame(
            [
                {"ticker": "AAA", "sector": "Technology"},
                {"ticker": "BBB", "sector": "Healthcare"},
                {"ticker": "CCC", "sector": "Technology"},
                {"ticker": "DDD", "sector": "Financials"},
            ]
        ),
        data_root / "reference" / "tickers.parquet",
    )

    paths, run_summary = build_price_strength_holding_period_simulation(data_root)

    assert run_summary["sector_available"] is True
    assert paths["rebalance_results_parquet"].exists()
    assert paths["summary_parquet"].exists()
    assert paths["sector_composition_parquet"].exists()
    assert paths["sector_concentration_parquet"].exists()
    assert paths["turnover_parquet"].exists()
    assert paths["markdown_report"].exists()
    assert paths["metadata"].name == "equity_price_strength_holding_period_simulation.metadata.json"
    assert paths["metadata"].exists()
    report = paths["markdown_report"].read_text(encoding="utf-8")
    assert "# Equity Price Strength Holding-Period Simulation" in report
    assert "## Basket Performance Summary" in report
    assert "## Suggested Next Step" in report
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["dataset"] == "equity_price_strength_holding_period_simulation"
    assert metadata["sector_availability"]["available"] is True
    assert "rebalance_results_parquet" in metadata["output_paths"]
    assert "sector_composition_parquet" in metadata["output_paths"]


def test_build_skips_sector_outputs_when_sector_unavailable(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_parquet(_sample_scorecard().drop(columns=["sector"]), _scorecard_path(data_root))

    paths, run_summary = build_price_strength_holding_period_simulation(data_root)

    assert run_summary["sector_available"] is False
    assert paths["summary_parquet"].exists()
    assert not paths["sector_composition_parquet"].exists()
    assert not paths["sector_concentration_parquet"].exists()
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["sector_availability"]["available"] is False
    assert "sector_composition_parquet" not in metadata["output_paths"]


def test_cli_price_strength_holding_period_simulation_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    _write_parquet(_sample_scorecard(), _scorecard_path(data_root))
    monkeypatch.setenv("FINBOT_DATA_ROOT", str(data_root))
    runner = CliRunner()

    result = runner.invoke(app, ["price-strength-holding-period-simulation", "--start-date", "2024-01-01"])

    assert result.exit_code == 0, result.output
    assert "Price strength holding-period simulation complete." in result.output
    assert (
        data_root
        / "research"
        / "price_strength_holding_period_simulation"
        / "equity_price_strength_holding_period_summary.parquet"
    ).exists()


def _sample_scorecard() -> pd.DataFrame:
    rows = []
    specs = [
        ("2024-01-30", {"AAA": ("neutral", 0.01), "BBB": ("neutral", 0.02), "CCC": ("high_volatility_trap", -0.01)}),
        (
            "2024-01-31",
            {
                "AAA": ("higher_conviction_price_strength", 0.10),
                "BBB": ("higher_conviction_price_strength", 0.05),
                "CCC": ("high_volatility_trap", -0.04),
            },
        ),
        (
            "2024-02-29",
            {
                "AAA": ("higher_conviction_price_strength", 0.07),
                "CCC": ("higher_conviction_price_strength", 0.03),
                "DDD": ("price_strength_candidate", 0.02),
            },
        ),
        (
            "2024-03-28",
            {
                "AAA": ("price_strength_candidate", 0.01),
                "BBB": ("high_volatility_trap", -0.03),
                "DDD": ("higher_conviction_price_strength", 0.08),
            },
        ),
    ]
    sectors = {"AAA": "Technology", "BBB": "Healthcare", "CCC": "Technology", "DDD": "Financials"}
    for date, symbols in specs:
        for symbol, (bucket, forward_return) in symbols.items():
            rows.append(
                {
                    "symbol": symbol,
                    "date": pd.Timestamp(date).date(),
                    "price_strength_scorecard_bucket": bucket,
                    "price_strength_score_v0": {
                        "higher_conviction_price_strength": 3,
                        "price_strength_candidate": 2,
                        "neutral": 0,
                        "high_volatility_trap": -1,
                    }[bucket],
                    "is_scorecard_bucket_eligible": True,
                    "forward_63d_sector_relative_return": forward_return,
                    "forward_63d_top_30pct_sector_flag": 1 if forward_return >= 0.05 else 0,
                    "forward_63d_bottom_30pct_sector_flag": 1 if forward_return < 0 else 0,
                    "sector": sectors[symbol],
                }
            )
    rows.append(
        {
            "symbol": "ZZZ",
            "date": pd.Timestamp("2024-03-28").date(),
            "price_strength_scorecard_bucket": "neutral",
            "price_strength_score_v0": 0,
            "is_scorecard_bucket_eligible": False,
            "forward_63d_sector_relative_return": 0.99,
            "forward_63d_top_30pct_sector_flag": 1,
            "forward_63d_bottom_30pct_sector_flag": 0,
            "sector": "Utilities",
        }
    )
    return pd.DataFrame(rows)


def _scorecard_path(data_root: Path) -> Path:
    return data_root / "research" / "price_strength_scorecard_v0" / "equity_price_strength_scorecard_v0.parquet"


def _write_parquet(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(path, index=False)
