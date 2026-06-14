from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from finbot_research.cli import app
from finbot_research.fundamental_research_v0 import (
    OUTPUT_FILENAMES,
    assign_fundamental_buckets,
    build_fundamental_research_v0,
    compute_feature_label_diagnostics,
    compute_feature_direction_audit,
    compute_rebalance_and_holding_outputs,
    compute_scorecard_v0,
    current_snapshot,
    summarize_bucket_outcomes,
)


def test_assign_fundamental_buckets_uses_directional_ranks() -> None:
    fundamentals = _sample_fundamentals()

    bucketed = assign_fundamental_buckets(fundamentals)
    rows = bucketed.set_index("symbol")

    assert rows.loc["HIGH", "quality_bucket"] == "high_quality"
    assert rows.loc["HIGH", "growth_bucket"] == "high_growth"
    assert rows.loc["HIGH", "balance_sheet_bucket"] == "strong_balance_sheet"
    assert rows.loc["LEV", "balance_sheet_bucket"] == "distressed_or_high_risk"
    assert rows.loc["OLD", "staleness_bucket"] == "very_stale_fundamentals"


def test_feature_diagnostics_marks_higher_is_worse_features() -> None:
    panel = _sample_panel()

    diagnostics = compute_feature_label_diagnostics(panel)
    rows = diagnostics.set_index(["feature", "horizon_days"])

    assert rows.loc[("debt_to_assets", 63), "direction"] == "higher_is_worse"
    assert rows.loc[("gross_margin_ttm", 63), "direction"] == "higher_is_better_or_contextual"
    assert rows.loc[("debt_to_assets", 63), "row_count"] > 0
    assert "strong_minus_weak_median_return" in diagnostics.columns


def test_feature_direction_audit_reports_consistency_status() -> None:
    audit = compute_feature_direction_audit(compute_feature_label_diagnostics(_sample_panel()))
    rows = audit.set_index(["feature", "horizon_days"])

    assert rows.loc[("debt_to_assets", 63), "assumed_direction"] == "higher_generally_worse"
    assert rows.loc[("gross_margin_ttm", 63), "assumed_direction"] == "higher_generally_better"
    assert rows.loc[("debt_to_assets", 63), "direction_consistency_status"] in {
        "consistent",
        "mixed",
        "opposite",
        "insufficient_data",
    }


def test_bucket_outcomes_use_eligible_universe_baseline_not_self_baseline() -> None:
    panel = assign_fundamental_buckets(_sample_panel())

    summary = summarize_bucket_outcomes(panel)
    rows = summary[(summary["bucket_name"] == "quality_bucket") & (summary["horizon_days"] == 63)]

    assert rows["baseline_avg_forward_sector_relative_return"].nunique() == 1
    high_quality = rows[rows["bucket_value"] == "high_quality"].iloc[0]
    assert high_quality["avg_forward_return_vs_baseline"] != pytest.approx(0.0)
    assert "median_forward_return_vs_baseline" in rows.columns
    assert "behavior_label" in rows.columns


def test_scorecard_v0_mapping_and_current_snapshot() -> None:
    scorecard = compute_scorecard_v0(assign_fundamental_buckets(_sample_fundamentals()))
    rows = scorecard.set_index("symbol")

    assert rows.loc["HIGH", "scorecard_label_v0"] == "high_quality_growth"
    assert rows.loc["HIGH", "fundamental_score_v0"] == 3
    assert rows.loc["LEV", "scorecard_label_v0"] == "levered_growth_risk"
    assert rows.loc["OLD", "scorecard_label_v0"] == "insufficient_data"

    latest = current_snapshot(scorecard, "effective_date")
    assert set(latest["symbol"]) == set(scorecard["symbol"])


def test_rebalance_and_holding_outputs_are_generated() -> None:
    scorecard_panel = _sample_scorecard_panel()

    feasibility, holding_summary, turnover = compute_rebalance_and_holding_outputs(scorecard_panel)

    assert {"monthly", "quarterly"}.issubset(set(feasibility["rebalance_frequency"]))
    assert not holding_summary.empty
    assert not turnover.empty
    assert "turnover_rate" in turnover.columns


def test_build_fundamental_research_v0_writes_numbered_parquet_and_reports_only(tmp_path: Path) -> None:
    data_root = _write_sample_inputs(tmp_path)

    paths, summary = build_fundamental_research_v0(data_root)
    output_dir = data_root / "research" / "fundamental_research_v0"

    assert set(paths) == set(OUTPUT_FILENAMES)
    assert summary["output_count"] == len(OUTPUT_FILENAMES)
    assert summary["research_panel_grain"] == "symbol,effective_date"
    assert summary["event_panel_rows"] == summary["fundamental_rows"]
    assert summary["event_panel_rows"] < summary["daily_label_rows"]
    assert summary["event_panel_label_match_rows"] == summary["event_panel_rows"]
    assert (output_dir / "01_fundamental_feature_coverage.parquet").exists()
    assert (output_dir / "02b_fundamental_feature_direction_audit.parquet").exists()
    assert (output_dir / "12b_fundamental_scorecard_behavior_summary.parquet").exists()
    assert (output_dir / "12c_fundamental_scorecard_relabeling_recommendation.parquet").exists()
    assert (output_dir / "18_fundamental_scorecard_v0_1_current.parquet").exists()
    assert (output_dir / "fundamental_feature_diagnostics_report.md").exists()
    assert not list(output_dir.glob("*.csv"))
    assert not list(output_dir.glob("*.json"))

    current = pd.read_parquet(output_dir / "10_fundamental_scorecard_v0_current.parquet")
    assert set(current["symbol"]) == {"HIGH", "GOOD", "LEV", "OLD"}
    recommendation = pd.read_parquet(output_dir / "17_fundamental_scorecard_v1_recommendation.parquet")
    assert not bool(recommendation.iloc[0]["v1_ready"])
    assert recommendation.iloc[0]["recommended_next_stage"] == "fundamental_scorecard_v0_1_review"
    assert recommendation.iloc[0]["relabeling_output"] == "12c_fundamental_scorecard_relabeling_recommendation.parquet"

    relabeling = pd.read_parquet(output_dir / "12c_fundamental_scorecard_relabeling_recommendation.parquet")
    assert set(relabeling["scorecard_label_v0"]) == {
        "high_quality_growth",
        "quality_cashflow_compounder",
        "cashflow_supported_growth",
        "speculative_growth",
        "levered_growth_risk",
        "fundamental_deterioration",
        "fundamental_trap",
        "insufficient_data",
        "neutral",
    }
    assert {
        "recommended_label_v0_1",
        "recommended_score_v0_1",
        "recommended_role_v0_1",
        "fundamental_quality_score_v0_1",
        "fundamental_opportunity_score_v0_1",
        "fundamental_risk_label_v0_1",
        "fundamental_data_quality_flag_v0_1",
        "score_policy_notes",
        "recommended_keep_drop",
        "primary_reason",
    }.issubset(relabeling.columns)
    relabeling_rows = relabeling.set_index("scorecard_label_v0")
    insufficient = relabeling_rows.loc["insufficient_data"]
    assert pd.isna(insufficient["fundamental_quality_score_v0_1"])
    assert pd.isna(insufficient["fundamental_opportunity_score_v0_1"])
    assert insufficient["fundamental_risk_label_v0_1"] == "insufficient_data"
    assert bool(insufficient["fundamental_data_quality_flag_v0_1"])
    high_quality = relabeling_rows.loc["high_quality_growth"]
    assert high_quality["fundamental_quality_score_v0_1"] > 0
    assert high_quality["fundamental_opportunity_score_v0_1"] == 0
    speculative = relabeling_rows.loc["speculative_growth"]
    assert speculative["fundamental_opportunity_score_v0_1"] > 0
    assert speculative["fundamental_risk_label_v0_1"] == "high_downside_tail_risk"

    current_v0_1 = pd.read_parquet(output_dir / "18_fundamental_scorecard_v0_1_current.parquet")
    assert {
        "symbol",
        "effective_date",
        "sector",
        "scorecard_label_v0",
        "recommended_label_v0_1",
        "fundamental_quality_score_v0_1",
        "fundamental_opportunity_score_v0_1",
        "fundamental_risk_label_v0_1",
        "fundamental_data_quality_flag_v0_1",
        "ready_for_v1",
    }.issubset(current_v0_1.columns)
    assert not current_v0_1["ready_for_v1"].any()

    for report_path in output_dir.glob("*.md"):
        contents = report_path.read_text(encoding="utf-8")
        assert "## Output File Guide" in contents
        assert ".parquet" in contents
    scorecard_report = (output_dir / "fundamental_scorecard_v0_report.md").read_text(encoding="utf-8")
    v1_report = (output_dir / "fundamental_scorecard_v1_recommendation_report.md").read_text(encoding="utf-8")
    assert "quality score, opportunity score, risk label, and data-quality flag" in scorecard_report
    assert "single fundamentals score is misleading" in v1_report


def test_cli_runs_fundamental_research_v0(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = _write_sample_inputs(tmp_path)
    monkeypatch.setenv("FINBOT_DATA_ROOT", str(data_root))

    result = CliRunner().invoke(app, ["fundamental-research-v0"])

    assert result.exit_code == 0
    assert "[fundamental-research-v0] Loading fundamental features" in result.output
    assert "grain=symbol,effective_date" in result.output
    assert "Wrote fundamental research v0" in result.output


def _sample_fundamentals() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _fundamental_row("HIGH", "2023-01-10", 0.60, 0.30, 0.50, 0.10, 0.05, 20, 120),
            _fundamental_row("GOOD", "2023-01-10", 0.45, 0.20, 0.30, 0.15, 0.20, 15, 100),
            _fundamental_row("LEV", "2023-01-10", 0.25, 0.30, 0.25, 0.85, 0.95, 4, 100),
            _fundamental_row("OLD", "2023-01-10", 0.10, -0.20, 0.05, 0.70, 0.20, -2, 500),
        ]
    )


def _fundamental_row(
    symbol: str,
    effective_date: str,
    margin: float,
    growth: float,
    cash: float,
    debt: float,
    accruals: float,
    eps: float,
    age: int,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "effective_date": pd.Timestamp(effective_date),
        "sector": "Technology",
        "gross_margin_ttm": margin,
        "ebitda_margin_ttm": margin,
        "net_margin_ttm": margin,
        "fcf_margin_ttm": margin,
        "roa_ttm": margin,
        "roe_ttm": margin,
        "roic_ttm": margin,
        "asset_turnover_ttm": margin,
        "revenue_yoy_ttm": growth,
        "ebitda_yoy_ttm": growth,
        "netinc_yoy_ttm": growth,
        "fcf_yoy_ttm": growth,
        "revenue_qoq": growth / 4,
        "revenue_yoy_quarter": growth,
        "gross_margin_ttm_change_yoy": growth / 10,
        "net_margin_ttm_change_yoy": growth / 10,
        "fcf_to_netinc_ttm": cash,
        "ncfo_to_netinc_ttm": cash,
        "capex_to_revenue_ttm": -0.05,
        "accruals_to_assets": accruals,
        "debt_to_assets": debt,
        "debt_to_equity": debt * 2,
        "cash_to_assets": 1 - debt,
        "net_debt_to_ebitda": debt * 3,
        "equity_to_assets": 1 - debt,
        "liabilities_to_assets": debt,
        "workingcapital_to_assets": 1 - debt,
        "eps_ttm": eps,
        "bvps": eps * 2 if eps > 0 else -1,
        "fcfps_ttm": eps / 2 if eps > 0 else -1,
        "shareswa_yoy_change": 0.01 if symbol != "LEV" else 0.30,
        "fundamentals_age_days": age,
    }


def _sample_panel() -> pd.DataFrame:
    rows = []
    for date in pd.to_datetime(["2023-01-10", "2023-02-15", "2023-02-28", "2023-03-31", "2023-04-30"]):
        for row in _sample_fundamentals().to_dict(orient="records"):
            symbol = row["symbol"]
            forward_return = {"HIGH": 0.08, "GOOD": 0.03, "LEV": -0.04, "OLD": -0.08}[symbol]
            rows.append(
                {
                    **row,
                    "date": date,
                    "forward_63d_sector_relative_return": forward_return,
                    "forward_126d_sector_relative_return": forward_return * 1.5,
                    "forward_252d_sector_relative_return": forward_return * 2,
                    "forward_63d_top_30pct_sector_flag": symbol == "HIGH",
                    "forward_126d_top_30pct_sector_flag": symbol == "HIGH",
                    "forward_252d_top_30pct_sector_flag": symbol == "HIGH",
                    "forward_63d_bottom_30pct_sector_flag": symbol == "OLD",
                    "forward_126d_bottom_30pct_sector_flag": symbol == "OLD",
                    "forward_252d_bottom_30pct_sector_flag": symbol == "OLD",
                }
            )
    return pd.DataFrame(rows)


def _sample_scorecard_panel() -> pd.DataFrame:
    bucketed = assign_fundamental_buckets(_sample_fundamentals())
    scorecard = compute_scorecard_v0(bucketed)
    panel = _sample_panel()[["symbol", "date", "forward_63d_sector_relative_return", "forward_126d_sector_relative_return", "forward_252d_sector_relative_return", "forward_63d_top_30pct_sector_flag", "forward_126d_top_30pct_sector_flag", "forward_252d_top_30pct_sector_flag", "forward_63d_bottom_30pct_sector_flag", "forward_126d_bottom_30pct_sector_flag", "forward_252d_bottom_30pct_sector_flag"]]
    return panel.merge(scorecard.drop(columns=["effective_date"]), on="symbol", how="left")


def _write_sample_inputs(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    (data_root / "features").mkdir(parents=True)
    (data_root / "labels").mkdir(parents=True)
    (data_root / "reference").mkdir(parents=True)

    _sample_fundamentals().drop(columns=["sector"]).to_parquet(
        data_root / "features" / "equity_fundamental_features.parquet",
        index=False,
    )
    labels = _sample_panel()[
        [
            "symbol",
            "date",
            "sector",
            "forward_63d_sector_relative_return",
            "forward_126d_sector_relative_return",
            "forward_252d_sector_relative_return",
            "forward_63d_top_30pct_sector_flag",
            "forward_126d_top_30pct_sector_flag",
            "forward_252d_top_30pct_sector_flag",
            "forward_63d_bottom_30pct_sector_flag",
            "forward_126d_bottom_30pct_sector_flag",
            "forward_252d_bottom_30pct_sector_flag",
        ]
    ]
    labels.to_parquet(data_root / "labels" / "equity_forward_return_labels.parquet", index=False)
    pd.DataFrame(
        {
            "symbol": ["HIGH", "GOOD", "LEV", "OLD"],
            "sector": ["Technology", "Technology", "Technology", "Technology"],
            "industry": ["Software", "Software", "Hardware", "Hardware"],
        }
    ).to_parquet(data_root / "reference" / "tickers.parquet", index=False)
    return data_root
