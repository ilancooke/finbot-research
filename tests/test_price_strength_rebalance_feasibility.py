from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from finbot_research.cli import app
from finbot_research.price_strength_rebalance_feasibility import (
    assess_feasibility,
    build_price_strength_rebalance_feasibility,
    compute_bucket_counts,
    compute_feasibility,
    compute_sector_composition,
    compute_sector_concentration,
    compute_turnover,
    prepare_rebalance_input,
    select_rebalance_rows,
)


def test_monthly_rebalance_date_selection_uses_last_available_trading_date() -> None:
    scorecard = prepare_rebalance_input(_sample_scorecard())

    rows = select_rebalance_rows(scorecard)

    assert rows["rebalance_date"].drop_duplicates().tolist() == [
        pd.Timestamp("2024-01-31").date(),
        pd.Timestamp("2024-02-29").date(),
        pd.Timestamp("2024-03-28").date(),
    ]
    assert set(rows[rows["rebalance_date"] == pd.Timestamp("2024-01-31").date()]["symbol"]) == {"AAA", "BBB", "CCC"}


def test_bucket_counts_and_pct_of_eligible_universe() -> None:
    rows = select_rebalance_rows(prepare_rebalance_input(_sample_scorecard()))

    counts = compute_bucket_counts(rows)

    jan = counts.set_index(["rebalance_date", "price_strength_scorecard_bucket"]).loc[
        (pd.Timestamp("2024-01-31").date(), "higher_conviction_price_strength")
    ]
    assert jan["symbol_count"] == 2
    assert jan["pct_of_eligible_universe"] == pytest.approx(2 / 3)


def test_turnover_calculations() -> None:
    rows = select_rebalance_rows(prepare_rebalance_input(_sample_scorecard()))

    turnover = compute_turnover(rows)

    higher = turnover.set_index(
        ["previous_rebalance_date", "rebalance_date", "price_strength_scorecard_bucket"]
    ).loc[
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


def test_sector_concentration_calculations() -> None:
    rows = select_rebalance_rows(prepare_rebalance_input(_sample_scorecard()))

    composition = compute_sector_composition(rows)
    concentration = compute_sector_concentration(composition)

    higher = concentration.set_index(["rebalance_date", "price_strength_scorecard_bucket"]).loc[
        (pd.Timestamp("2024-01-31").date(), "higher_conviction_price_strength")
    ]
    assert higher["sector_count"] == 2
    assert higher["max_sector_share"] == pytest.approx(0.5)
    assert higher["top_3_sector_share"] == pytest.approx(1.0)
    assert higher["herfindahl_sector_concentration"] == pytest.approx(0.5)


def test_feasibility_labels() -> None:
    assert (
        assess_feasibility(
            median_symbol_count=60,
            p10_symbol_count=30,
            median_max_sector_share=0.30,
            median_top_3_sector_share=0.60,
            median_turnover_rate=0.50,
            sector_available=True,
        )
        == "feasible_basket_candidate"
    )
    assert (
        assess_feasibility(
            median_symbol_count=20,
            p10_symbol_count=12,
            median_max_sector_share=0.30,
            median_top_3_sector_share=0.60,
            median_turnover_rate=0.50,
            sector_available=True,
        )
        == "sparse_but_usable"
    )
    assert (
        assess_feasibility(
            median_symbol_count=8,
            p10_symbol_count=4,
            median_max_sector_share=0.30,
            median_top_3_sector_share=0.60,
            median_turnover_rate=0.50,
            sector_available=True,
        )
        == "too_sparse"
    )
    assert (
        assess_feasibility(
            median_symbol_count=60,
            p10_symbol_count=30,
            median_max_sector_share=0.50,
            median_top_3_sector_share=0.60,
            median_turnover_rate=0.50,
            sector_available=True,
        )
        == "sector_concentrated"
    )
    assert (
        assess_feasibility(
            median_symbol_count=60,
            p10_symbol_count=30,
            median_max_sector_share=None,
            median_top_3_sector_share=None,
            median_turnover_rate=0.90,
            sector_available=False,
        )
        == "high_turnover"
    )


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

    paths, summary = build_price_strength_rebalance_feasibility(data_root)

    assert summary["sector_available"] is True
    assert paths["bucket_counts_parquet"].exists()
    assert paths["bucket_count_summary_parquet"].exists()
    assert paths["sector_composition_parquet"].exists()
    assert paths["sector_concentration_parquet"].exists()
    assert paths["turnover_parquet"].exists()
    assert paths["feasibility_parquet"].exists()
    assert paths["markdown_report"].exists()
    assert paths["metadata"].exists()
    report = paths["markdown_report"].read_text(encoding="utf-8")
    assert "# Equity Price Strength Rebalance Feasibility" in report
    assert "## Rebalance Method" in report
    assert "## Sector Concentration" in report

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["dataset"] == "equity_price_strength_rebalance_feasibility"
    assert metadata["rebalance_frequency"] == "monthly"
    assert metadata["sector_availability"]["available"] is True
    assert "bucket_counts_parquet" in metadata["output_paths"]
    assert "sector_composition_parquet" in metadata["output_paths"]


def test_build_skips_sector_outputs_when_sector_unavailable(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_parquet(_sample_scorecard().drop(columns=["sector"]), _scorecard_path(data_root))

    paths, summary = build_price_strength_rebalance_feasibility(data_root)

    assert summary["sector_available"] is False
    assert paths["feasibility_parquet"].exists()
    assert not paths["sector_composition_parquet"].exists()
    assert not paths["sector_concentration_parquet"].exists()
    report = paths["markdown_report"].read_text(encoding="utf-8")
    assert "Sector diagnostics were skipped" in report
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["sector_availability"]["available"] is False
    assert "sector_composition_parquet" not in metadata["output_paths"]


def test_cli_price_strength_rebalance_feasibility_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    _write_parquet(_sample_scorecard(), _scorecard_path(data_root))
    monkeypatch.setenv("FINBOT_DATA_ROOT", str(data_root))
    runner = CliRunner()

    result = runner.invoke(app, ["price-strength-rebalance-feasibility", "--start-date", "2024-01-01"])

    assert result.exit_code == 0, result.output
    assert "Price strength rebalance feasibility complete." in result.output
    assert (
        data_root
        / "research"
        / "price_strength_rebalance_feasibility"
        / "equity_price_strength_rebalance_feasibility.parquet"
    ).exists()


def test_compute_feasibility_without_sector_uses_counts_and_turnover_only() -> None:
    rows = select_rebalance_rows(prepare_rebalance_input(_sample_scorecard().drop(columns=["sector"])))
    counts = compute_bucket_counts(rows)
    summary = counts.groupby("price_strength_scorecard_bucket", as_index=False).agg(
        rebalance_date_count=("rebalance_date", "nunique"),
        mean_symbol_count=("symbol_count", "mean"),
        median_symbol_count=("symbol_count", "median"),
        min_symbol_count=("symbol_count", "min"),
        max_symbol_count=("symbol_count", "max"),
        p10_symbol_count=("symbol_count", lambda values: values.quantile(0.10)),
        p25_symbol_count=("symbol_count", lambda values: values.quantile(0.25)),
        p75_symbol_count=("symbol_count", lambda values: values.quantile(0.75)),
        p90_symbol_count=("symbol_count", lambda values: values.quantile(0.90)),
        mean_pct_of_eligible_universe=("pct_of_eligible_universe", "mean"),
    )

    feasibility = compute_feasibility(summary, compute_turnover(rows), pd.DataFrame(), sector_available=False)

    assert "feasibility_assessment" in feasibility.columns
    assert feasibility["median_max_sector_share"].isna().all()


def _sample_scorecard() -> pd.DataFrame:
    rows = []
    specs = [
        ("2024-01-30", {"AAA": "neutral", "BBB": "neutral", "CCC": "high_volatility_trap"}),
        (
            "2024-01-31",
            {
                "AAA": "higher_conviction_price_strength",
                "BBB": "higher_conviction_price_strength",
                "CCC": "high_volatility_trap",
            },
        ),
        (
            "2024-02-29",
            {
                "AAA": "higher_conviction_price_strength",
                "CCC": "higher_conviction_price_strength",
                "DDD": "price_strength_candidate",
            },
        ),
        (
            "2024-03-28",
            {
                "AAA": "price_strength_candidate",
                "BBB": "high_volatility_trap",
                "DDD": "higher_conviction_price_strength",
            },
        ),
    ]
    sectors = {
        "AAA": "Technology",
        "BBB": "Healthcare",
        "CCC": "Technology",
        "DDD": "Financials",
    }
    for date, buckets in specs:
        for symbol, bucket in buckets.items():
            rows.append(
                {
                    "symbol": symbol,
                    "date": pd.Timestamp(date).date(),
                    "price_strength_scorecard_bucket": bucket,
                    "price_strength_score_v0": {"higher_conviction_price_strength": 3, "price_strength_candidate": 2, "neutral": 0, "high_volatility_trap": -1}[bucket],
                    "sector": sectors[symbol],
                    "is_scorecard_bucket_eligible": True,
                }
            )
    rows.append(
        {
            "symbol": "ZZZ",
            "date": pd.Timestamp("2024-03-28").date(),
            "price_strength_scorecard_bucket": "neutral",
            "price_strength_score_v0": 0,
            "sector": "Utilities",
            "is_scorecard_bucket_eligible": False,
        }
    )
    return pd.DataFrame(rows)


def _scorecard_path(data_root: Path) -> Path:
    return data_root / "research" / "price_strength_scorecard_v0" / "equity_price_strength_scorecard_v0.parquet"


def _write_parquet(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(path, index=False)
