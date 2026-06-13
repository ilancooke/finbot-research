from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from finbot_research.cli import app
from finbot_research.price_strength_turnover_cost_efficiency import (
    build_price_strength_turnover_cost_efficiency,
    compute_efficiency_table,
    filter_focus_variants,
    select_best_efficiency_variants,
)


def test_efficiency_formulas_and_rebalance_mapping() -> None:
    efficiency = compute_efficiency_table(_sample_summary(), _sample_turnover())

    row = efficiency[
        (efficiency["rebalance_frequency"] == "monthly")
        & (efficiency["holding_period_days"] == 63)
        & (efficiency["transaction_cost_bps"] == 50)
        & (efficiency["portfolio_name"] == "higher_conviction_raw")
    ].iloc[0]
    assert row["rebalances_per_year"] == pytest.approx(12)
    assert row["annualized_one_way_turnover"] == pytest.approx(0.50 * 12)
    assert row["annualized_two_way_turnover"] == pytest.approx(2 * 0.50 * 12)
    assert row["estimated_annualized_cost_drag"] == pytest.approx(0.0025 * 12)
    assert row["net_excess_return_per_annualized_one_way_turnover"] == pytest.approx(0.06 / 6.0)
    assert row["net_excess_return_per_annualized_two_way_turnover"] == pytest.approx(0.06 / 12.0)
    assert row["net_excess_return_per_annualized_cost_drag"] == pytest.approx(0.06 / 0.03)
    expected_score = 0.06 - 0.03 - (abs(-0.60) - 0.50) * 0.25
    assert row["cost_efficiency_score"] == pytest.approx(expected_score)


def test_focus_variant_filtering() -> None:
    efficiency = compute_efficiency_table(_sample_summary(), _sample_turnover())
    focus = filter_focus_variants(efficiency)

    assert set(focus["portfolio_name"]) == {"higher_conviction_raw", "higher_conviction_sector_capped"}
    assert set(focus["holding_period_days"]) == {63, 126}
    assert set(focus["rebalance_frequency"]) == {"monthly", "quarterly"}


def test_best_variant_selection_logic() -> None:
    efficiency = compute_efficiency_table(_sample_summary(), _sample_turnover())
    best = select_best_efficiency_variants(efficiency)

    assert {
        "highest_net_excess_annualized_return",
        "highest_net_sharpe_like",
        "lowest_annualized_one_way_turnover_among_positive_excess",
        "highest_excess_per_annualized_turnover",
        "highest_excess_per_annualized_cost_drag",
        "highest_cost_efficiency_score",
        "best_practical_default_candidate",
    } == set(best["selection_criterion"])
    practical = best.set_index("selection_criterion").loc["best_practical_default_candidate"]
    assert practical["constraints_met"] == True
    assert practical["transaction_cost_bps"] >= 50


def test_build_writes_outputs_metadata_and_report(tmp_path: Path) -> None:
    data_root = _write_inputs(tmp_path)

    paths, run_summary = build_price_strength_turnover_cost_efficiency(data_root)

    assert run_summary["efficiency_rows"] == len(_sample_summary())
    assert run_summary["focus_rows"] > 0
    assert run_summary["best_variant_rows"] == 7
    assert paths["efficiency_parquet"].exists()
    assert paths["efficiency_csv"].exists()
    assert paths["focus_parquet"].exists()
    assert paths["focus_csv"].exists()
    assert paths["best_variants_parquet"].exists()
    assert paths["best_variants_csv"].exists()
    assert paths["markdown_report"].exists()
    assert paths["metadata"].name == "equity_price_strength_turnover_cost_efficiency.metadata.json"
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["rebalances_per_year_assumptions"] == {"monthly": 12.0, "quarterly": 4.0}
    assert "efficiency_parquet" in metadata["output_paths"]
    report = paths["markdown_report"].read_text(encoding="utf-8")
    for section in [
        "## Purpose",
        "## Inputs",
        "## Method",
        "## Executive Summary",
        "## Focus Variant Comparison",
        "## Monthly vs Quarterly Annualized Turnover",
        "## Cost-Adjusted Results",
        "## Best Variants",
        "## Interpretation",
        "## Important Caveats",
        "## Output File Guide",
        "## Suggested Decision",
    ]:
        assert section in report


def test_cli_price_strength_turnover_cost_efficiency_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    data_root = _write_inputs(tmp_path)
    monkeypatch.setenv("FINBOT_DATA_ROOT", str(data_root))
    runner = CliRunner()

    result = runner.invoke(app, ["price-strength-turnover-cost-efficiency"])

    assert result.exit_code == 0, result.output
    assert "Price strength turnover/cost efficiency complete." in result.output
    assert (
        data_root
        / "research"
        / "price_strength_turnover_cost_efficiency"
        / "equity_price_strength_turnover_cost_efficiency.parquet"
    ).exists()


def _sample_summary() -> pd.DataFrame:
    rows = []
    specs = [
        ("monthly", 63, 50, "higher_conviction_raw", 0.22, 0.25, 0.88, -0.60, 0.06, 0.53, 0.50, 0.45, 0.0025),
        ("monthly", 63, 50, "higher_conviction_sector_capped", 0.21, 0.23, 0.91, -0.57, 0.055, 0.52, 0.52, 0.47, 0.0026),
        ("monthly", 63, 50, "eligible_universe_baseline", 0.16, 0.20, 0.80, -0.50, 0.0, 0.0, 0.10, 0.08, 0.0005),
        ("quarterly", 63, 50, "higher_conviction_raw", 0.20, 0.22, 0.90, -0.55, 0.07, 0.54, 0.80, 0.75, 0.0040),
        ("quarterly", 63, 50, "higher_conviction_sector_capped", 0.19, 0.21, 0.89, -0.54, 0.065, 0.53, 0.82, 0.77, 0.0041),
        ("quarterly", 126, 50, "higher_conviction_raw", 0.18, 0.20, 0.92, -0.49, 0.055, 0.55, 0.70, 0.65, 0.0035),
        ("quarterly", 126, 50, "higher_conviction_sector_capped", 0.17, 0.19, 0.87, -0.48, 0.045, 0.54, 0.72, 0.67, 0.0036),
        ("monthly", 21, 50, "higher_conviction_raw", 0.15, 0.24, 0.62, -0.66, -0.01, 0.49, 0.50, 0.45, 0.0025),
    ]
    for (
        frequency,
        holding_days,
        cost_bps,
        portfolio_name,
        ann_return,
        ann_vol,
        sharpe,
        drawdown,
        excess,
        win_rate,
        mean_turnover,
        median_turnover,
        mean_cost,
    ) in specs:
        rows.append(
            {
                "rebalance_frequency": frequency,
                "holding_period_days": holding_days,
                "transaction_cost_bps": cost_bps,
                "portfolio_name": portfolio_name,
                "net_annualized_return": ann_return,
                "net_annualized_volatility": ann_vol,
                "net_sharpe_like": sharpe,
                "net_max_drawdown": drawdown,
                "net_excess_annualized_return": excess,
                "net_daily_win_rate_vs_benchmark": win_rate,
                "mean_one_way_turnover": mean_turnover,
                "median_one_way_turnover": median_turnover,
                "mean_transaction_cost": mean_cost,
            }
        )
    return pd.DataFrame(rows)


def _sample_turnover() -> pd.DataFrame:
    return _sample_summary()[
        [
            "rebalance_frequency",
            "holding_period_days",
            "transaction_cost_bps",
            "portfolio_name",
            "mean_one_way_turnover",
            "mean_transaction_cost",
        ]
    ].copy()


def _write_inputs(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    summary_path = (
        data_root
        / "research"
        / "price_strength_horizon_sensitivity"
        / "equity_price_strength_horizon_sensitivity_summary.parquet"
    )
    turnover_path = (
        data_root
        / "research"
        / "price_strength_horizon_sensitivity"
        / "equity_price_strength_horizon_sensitivity_turnover.parquet"
    )
    _write_parquet(_sample_summary(), summary_path)
    _write_parquet(_sample_turnover(), turnover_path)
    return data_root


def _write_parquet(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(path, index=False)
