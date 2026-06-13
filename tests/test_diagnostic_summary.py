from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from finbot_research.cli import app
from finbot_research.diagnostic_summary import (
    MAX_DECILE_TABLES,
    MAX_YEAR_DETAIL_FEATURES,
    build_diagnostic_summary,
    compute_current_year_snapshot,
    compute_lookback_summary,
    compute_year_spreads,
    summarize_diagnostics,
    write_markdown_report,
)


def test_summarize_diagnostics_builds_one_row_per_feature() -> None:
    summary = summarize_diagnostics(_fake_diagnostics())

    assert set(summary["feature_name"]) == {
        "momentum",
        "risk",
        "disagreement",
        "average_dollar_volume_63d",
        "mixed",
        "sparse",
        "recent_momentum",
        "old_momentum",
        "short_history",
    }
    assert {
        "feature_name",
        "non_null_count",
        "null_rate",
        "label_non_null_count",
        "pearson_corr",
        "spearman_corr",
        "top_minus_bottom_avg_forward_return",
        "top_minus_bottom_median_forward_return",
        "top_minus_bottom_avg_median_gap",
        "tail_effect_assessment",
        "top_minus_bottom_top_30pct_flag_rate",
        "avg_return_decile_spearman_corr",
        "flag_rate_decile_spearman_corr",
        "avg_return_monotonicity_direction",
        "flag_rate_monotonicity_direction",
        "spread_correlation_agreement",
        "suggested_scorecard_use",
        "years_with_valid_spread",
        "pct_years_positive_spread",
        "stability_assessment",
        "recent_signal_assessment",
        "last_10y_mean_year_spread",
        "last_5y_mean_year_spread",
        "current_partial_year",
        "current_year_signal_direction",
        "signal_direction",
        "candidate_category",
        "summary_rank",
        "notes",
    }.issubset(summary.columns)


def test_summarize_diagnostics_assigns_candidate_categories() -> None:
    summary = summarize_diagnostics(_fake_diagnostics()).set_index("feature_name")

    assert summary.loc["momentum", "candidate_category"] == "bullish_candidate"
    assert summary.loc["momentum", "suggested_scorecard_use"] == "positive_component_candidate"
    assert summary.loc["momentum", "recent_signal_assessment"] == "persistent_positive"
    assert summary.loc["momentum", "signal_direction"] == "positive"
    assert summary.loc["momentum", "spread_correlation_agreement"] == "positive_agreement"
    assert summary.loc["risk", "candidate_category"] == "risk_penalty_candidate"
    assert summary.loc["risk", "suggested_scorecard_use"] == "risk_penalty_candidate"
    assert summary.loc["risk", "recent_signal_assessment"] == "persistent_negative"
    assert summary.loc["risk", "signal_direction"] == "negative"
    assert summary.loc["risk", "spread_correlation_agreement"] == "negative_agreement"
    assert summary.loc["disagreement", "candidate_category"] == "nonlinear_or_unstable"
    assert summary.loc["disagreement", "suggested_scorecard_use"] == "nonlinear_feature_requires_review"
    assert summary.loc["disagreement", "spread_correlation_agreement"] == "disagreement"
    assert summary.loc["average_dollar_volume_63d", "suggested_scorecard_use"] == "liquidity_filter_candidate"
    assert summary.loc["mixed", "candidate_category"] == "weak_or_noisy"
    assert summary.loc["mixed", "signal_direction"] == "mixed"
    assert summary.loc["sparse", "candidate_category"] == "coverage_problem"
    assert summary.loc["sparse", "suggested_scorecard_use"] == "coverage_issue"


def test_summarize_diagnostics_adds_tail_effect_assessments() -> None:
    summary = summarize_diagnostics(_fake_diagnostics()).set_index("feature_name")

    assert summary.loc["momentum", "tail_effect_assessment"] == "broad_positive"
    assert summary.loc["risk", "tail_effect_assessment"] == "broad_negative"
    assert summary.loc["momentum", "top_minus_bottom_median_forward_return"] == pytest.approx(0.09)
    assert summary.loc["momentum", "top_minus_bottom_avg_median_gap"] == pytest.approx(0.0)


def test_summarize_diagnostics_flags_right_tail_positive() -> None:
    diagnostics = _fake_diagnostics()
    mask = (diagnostics["metric_type"] == "decile") & (diagnostics["feature_name"] == "momentum")
    diagnostics.loc[mask & (diagnostics["decile"] == 1), "median_forward_63d_sector_relative_return"] = 0.03
    diagnostics.loc[mask & (diagnostics["decile"] == 10), "median_forward_63d_sector_relative_return"] = 0.02

    summary = summarize_diagnostics(diagnostics).set_index("feature_name")

    assert summary.loc["momentum", "tail_effect_assessment"] == "right_tail_positive"
    assert summary.loc["momentum", "suggested_scorecard_use"] == "upside_optional_component_requires_review"


def test_compute_year_spreads_handles_valid_and_missing_deciles() -> None:
    diagnostics = pd.DataFrame(
        [
            _year_decile_row("complete", 2023, 1, avg_return=0.01, flag_rate=0.20),
            _year_decile_row("complete", 2023, 10, avg_return=0.04, flag_rate=0.35),
            _year_decile_row("missing_top", 2023, 1, avg_return=0.01, flag_rate=0.20),
        ]
    )

    year_spreads = compute_year_spreads(diagnostics).set_index(["feature_name", "calendar_year"])

    assert year_spreads.loc[("complete", 2023), "year_top_minus_bottom_avg_forward_return"] == 0.03
    assert year_spreads.loc[("complete", 2023), "year_top_minus_bottom_top_30pct_flag_rate"] == pytest.approx(0.15)
    assert bool(year_spreads.loc[("complete", 2023), "year_valid_spread_flag"])
    assert not bool(year_spreads.loc[("missing_top", 2023), "year_valid_spread_flag"])


def test_compute_lookback_summary_uses_max_calendar_year_windows() -> None:
    year_spreads = compute_year_spreads(_fake_diagnostics())

    lookback = compute_lookback_summary(year_spreads)
    momentum = lookback[lookback["feature_name"] == "momentum"].set_index("lookback_window")

    assert momentum.loc["full_history", "start_year"] == 2006
    assert momentum.loc["full_history", "end_year"] == 2025
    assert momentum.loc["last_10y", "start_year"] == 2016
    assert momentum.loc["last_10y", "end_year"] == 2025
    assert momentum.loc["last_5y", "start_year"] == 2021
    assert momentum.loc["last_5y", "end_year"] == 2025


def test_partial_current_year_is_visible_but_excluded_from_stability_by_default() -> None:
    year_spreads = compute_year_spreads(_fake_diagnostics())
    partial_rows = year_spreads[year_spreads["calendar_year"] == 2026]

    assert not partial_rows.empty
    assert partial_rows["is_partial_year"].all()
    assert not partial_rows["included_in_completed_year_stability"].any()

    lookback = compute_lookback_summary(year_spreads)
    momentum = lookback[lookback["feature_name"] == "momentum"].set_index("lookback_window")
    assert momentum.loc["last_5y", "start_year"] == 2021
    assert momentum.loc["last_5y", "end_year"] == 2025


def test_partial_current_year_can_be_included_in_stability() -> None:
    year_spreads = compute_year_spreads(
        _fake_diagnostics(),
        include_partial_current_year_in_stability=True,
    )
    partial_rows = year_spreads[year_spreads["calendar_year"] == 2026]

    assert partial_rows["is_partial_year"].all()
    assert partial_rows["included_in_completed_year_stability"].all()

    lookback = compute_lookback_summary(year_spreads)
    momentum = lookback[lookback["feature_name"] == "momentum"].set_index("lookback_window")
    assert momentum.loc["last_5y", "start_year"] == 2022
    assert momentum.loc["last_5y", "end_year"] == 2026


def test_compute_current_year_snapshot_adds_current_direction() -> None:
    snapshot = compute_current_year_snapshot(compute_year_spreads(_fake_diagnostics())).set_index("feature_name")

    assert snapshot.loc["momentum", "current_partial_year"] == 2026
    assert bool(snapshot.loc["momentum", "current_year_is_partial"])
    assert snapshot.loc["momentum", "current_year_signal_direction"] == "positive"
    assert snapshot.loc["risk", "current_year_signal_direction"] == "negative"


def test_summarize_diagnostics_classifies_recent_signal_assessments() -> None:
    summary = summarize_diagnostics(_fake_diagnostics()).set_index("feature_name")

    assert summary.loc["momentum", "recent_signal_assessment"] == "persistent_positive"
    assert summary.loc["risk", "recent_signal_assessment"] == "persistent_negative"
    assert summary.loc["recent_momentum", "recent_signal_assessment"] == "recent_positive_only"
    assert summary.loc["old_momentum", "recent_signal_assessment"] == "historical_positive_but_recent_weak"
    assert summary.loc["mixed", "recent_signal_assessment"] == "mixed_or_regime_dependent"
    assert summary.loc["short_history", "recent_signal_assessment"] == "insufficient_year_data"


def test_summarize_diagnostics_adds_recent_window_fields() -> None:
    summary = summarize_diagnostics(_fake_diagnostics()).set_index("feature_name")

    assert "last_10y_mean_year_spread" in summary.columns
    assert "last_5y_pct_years_positive_spread" in summary.columns
    assert "last_10y_mean_year_flag_lift" in summary.columns
    assert summary.loc["momentum", "last_5y_mean_year_spread"] > 0


def test_summarize_diagnostics_computes_decile_monotonicity() -> None:
    summary = summarize_diagnostics(_fake_diagnostics()).set_index("feature_name")

    assert summary.loc["momentum", "avg_return_decile_spearman_corr"] > 0.9
    assert summary.loc["momentum", "avg_return_monotonicity_direction"] == "positive"
    assert summary.loc["risk", "avg_return_decile_spearman_corr"] < -0.9
    assert summary.loc["risk", "avg_return_monotonicity_direction"] == "negative"


def test_summarize_diagnostics_ranks_stronger_features_higher() -> None:
    summary = summarize_diagnostics(_fake_diagnostics()).set_index("feature_name")

    assert summary.loc["momentum", "summary_rank_score"] > summary.loc["mixed", "summary_rank_score"]
    assert summary.loc["momentum", "summary_rank"] < summary.loc["mixed", "summary_rank"]


def test_summarize_diagnostics_computes_year_stability() -> None:
    summary = summarize_diagnostics(_fake_diagnostics()).set_index("feature_name")

    assert summary.loc["momentum", "stability_assessment"] == "mostly_positive"
    assert summary.loc["risk", "stability_assessment"] == "mostly_negative"
    assert summary.loc["mixed", "stability_assessment"] == "mixed_by_year"


def test_write_markdown_report_includes_expected_sections(tmp_path: Path) -> None:
    summary = summarize_diagnostics(_fake_diagnostics())
    report_path = tmp_path / "research" / "report.md"

    write_markdown_report(summary, report_path, diagnostics_path=tmp_path / "diagnostics.parquet", diagnostics=_fake_diagnostics())

    text = report_path.read_text(encoding="utf-8")
    assert "## Executive Summary" in text
    assert "## Key Findings" in text
    assert "## Feature Candidates for Review" in text
    assert "## Recent and Regime Signal Summary" in text
    assert "## Current Partial Year Snapshot" in text
    assert "## Selected Decile Curves" in text
    assert "## Selected Year-by-Year Details" in text
    assert "## Output File Guide" in text
    assert "## Important Caveats" in text
    assert "## Suggested Next Step" in text
    assert "## Top Risk Penalty Candidate Features" not in text
    assert "## Nonlinear or Unstable Features Requiring Review" not in text
    assert "median_forward_63d_sector_relative_return" in text
    assert "avg_minus_median_forward_63d_sector_relative_return" in text
    assert _section_heading_count(text, "## Selected Decile Curves", "## Selected Year-by-Year Details") <= MAX_DECILE_TABLES
    assert _section_heading_count(text, "## Selected Year-by-Year Details", "## Output File Guide") <= MAX_YEAR_DETAIL_FEATURES


def test_build_diagnostic_summary_writes_outputs_and_metadata(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    diagnostics_path = data_root / "research" / "feature_label_diagnostics" / "equity_price_signal_diagnostics.parquet"
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    _fake_diagnostics().to_parquet(diagnostics_path, index=False)

    paths, stats = build_diagnostic_summary(data_root)

    assert paths["summary_parquet"].exists()
    assert paths["summary_csv"].exists()
    assert paths["markdown_report"].exists()
    assert paths["year_spreads_parquet"].exists()
    assert paths["year_spreads_csv"].exists()
    assert paths["lookback_summary_parquet"].exists()
    assert paths["lookback_summary_csv"].exists()
    assert paths["current_year_snapshot_parquet"].exists()
    assert paths["current_year_snapshot_csv"].exists()
    assert paths["metadata"].exists()
    assert stats["features_analyzed"] == 9
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["dataset"] == "equity_price_signal_summary"
    assert metadata["dataset_type"] == "research_summary"
    assert metadata["candidate_category_counts"]["bullish_candidate"] == 4
    assert metadata["candidate_category_counts"]["nonlinear_or_unstable"] == 1
    assert metadata["suggested_scorecard_use_counts"]["liquidity_filter_candidate"] == 1
    assert "monotonicity_thresholds" in metadata
    assert "stability_assessment_logic" in metadata
    assert "year_spreads_output_path" in metadata
    assert "lookback_summary_output_path" in metadata
    assert "current_year_snapshot_parquet" in metadata["output_paths"]
    assert "tail_effect_fields" in metadata
    assert "tail_effect_assessment_rules" in metadata
    assert "partial_year_handling" in metadata
    assert metadata["partial_year_handling"]["current_partial_year"] == 2026
    assert metadata["partial_year_handling"]["latest_completed_stability_year"] == 2025
    assert "recent_signal_assessment_rules" in metadata
    assert "recent_signal_assessment_counts" in metadata
    assert "output_file_purposes" in metadata
    assert "default_output_policy" in metadata
    assert "ranking_formula" in metadata


def test_cli_summarize_diagnostics_writes_expected_outputs(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    diagnostics_path = data_root / "research" / "feature_label_diagnostics" / "equity_price_signal_diagnostics.parquet"
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    _fake_diagnostics().to_parquet(diagnostics_path, index=False)
    monkeypatch.setenv("FINBOT_DATA_ROOT", str(data_root))
    runner = CliRunner()

    result = runner.invoke(app, ["summarize-diagnostics"])

    assert result.exit_code == 0, result.output
    assert "Wrote diagnostics summary:" in result.output
    assert "Human-readable output:" in result.output
    assert "Canonical machine-readable outputs:" in result.output
    assert "Convenience exports:" in result.output
    assert "Current year snapshot parquet:" in result.output
    assert (data_root / "research" / "feature_label_diagnostics" / "equity_price_signal_summary.parquet").exists()
    assert (data_root / "research" / "feature_label_diagnostics" / "equity_price_signal_summary.csv").exists()
    assert (data_root / "research" / "feature_label_diagnostics" / "equity_price_signal_report.md").exists()
    assert (data_root / "research" / "feature_label_diagnostics" / "equity_price_signal_year_spreads.parquet").exists()
    assert (data_root / "research" / "feature_label_diagnostics" / "equity_price_signal_year_spreads.csv").exists()
    assert (data_root / "research" / "feature_label_diagnostics" / "equity_price_signal_lookback_summary.parquet").exists()
    assert (data_root / "research" / "feature_label_diagnostics" / "equity_price_signal_lookback_summary.csv").exists()
    assert (data_root / "research" / "feature_label_diagnostics" / "equity_price_signal_current_year_snapshot.parquet").exists()
    assert (data_root / "research" / "feature_label_diagnostics" / "equity_price_signal_current_year_snapshot.csv").exists()
    assert (data_root / "research" / "feature_label_diagnostics" / "equity_price_signal_summary.metadata.json").exists()


def test_readme_documents_summarize_diagnostics_outputs() -> None:
    text = Path("README.md").read_text(encoding="utf-8")

    assert "## `summarize-diagnostics` outputs" in text
    assert "Parquet files are the canonical machine-readable outputs" in text
    assert "CSV files are convenience exports" in text


def _section_heading_count(text: str, start_heading: str, end_heading: str) -> int:
    start = text.index(start_heading)
    end = text.index(end_heading, start)
    return text[start:end].count("\n### ")


def _fake_diagnostics() -> pd.DataFrame:
    rows = []
    rows.extend(
        [
            _coverage("momentum", null_rate=0.02, pearson=0.12, spearman=0.15),
            _spread("momentum", return_spread=0.08, flag_lift=0.20),
            *_deciles("momentum", return_direction="positive", flag_direction="positive"),
            *_year_deciles("momentum", spreads=[0.05] * 21),
            _coverage("risk", null_rate=0.03, pearson=-0.10, spearman=-0.14),
            _spread("risk", return_spread=-0.06, flag_lift=-0.18),
            *_deciles("risk", return_direction="negative", flag_direction="negative"),
            *_year_deciles("risk", spreads=[-0.04] * 21),
            _coverage("disagreement", null_rate=0.01, pearson=-0.08, spearman=-0.12),
            _spread("disagreement", return_spread=0.07, flag_lift=0.16),
            *_deciles("disagreement", return_direction="negative", flag_direction="positive"),
            *_year_deciles("disagreement", spreads=[0.03, -0.02] * 10 + [0.03]),
            _coverage("average_dollar_volume_63d", null_rate=0.01, pearson=-0.09, spearman=-0.11),
            _spread("average_dollar_volume_63d", return_spread=-0.04, flag_lift=-0.10),
            *_deciles("average_dollar_volume_63d", return_direction="negative", flag_direction="negative"),
            *_year_deciles("average_dollar_volume_63d", spreads=[-0.02] * 21),
            _coverage("mixed", null_rate=0.01, pearson=0.01, spearman=0.02),
            _spread("mixed", return_spread=0.04, flag_lift=-0.05),
            *_deciles("mixed", return_direction="flat", flag_direction="flat"),
            *_year_deciles("mixed", spreads=[0.03, -0.02] * 10 + [0.03]),
            _coverage("sparse", null_rate=0.80, pearson=0.20, spearman=0.25),
            _spread("sparse", return_spread=0.20, flag_lift=0.30),
            *_deciles("sparse", return_direction="positive", flag_direction="positive"),
            *_year_deciles("sparse", spreads=[0.04] * 21),
            _coverage("recent_momentum", null_rate=0.02, pearson=0.04, spearman=0.05),
            _spread("recent_momentum", return_spread=0.03, flag_lift=0.06),
            *_deciles("recent_momentum", return_direction="positive", flag_direction="positive"),
            *_year_deciles("recent_momentum", spreads=[-0.08] * 10 + [0.04] * 11),
            _coverage("old_momentum", null_rate=0.02, pearson=0.04, spearman=0.05),
            _spread("old_momentum", return_spread=0.03, flag_lift=0.06),
            *_deciles("old_momentum", return_direction="positive", flag_direction="positive"),
            *_year_deciles("old_momentum", spreads=[0.04] * 13 + [-0.02] * 8),
            _coverage("short_history", null_rate=0.02, pearson=0.04, spearman=0.05),
            _spread("short_history", return_spread=0.03, flag_lift=0.06),
            *_deciles("short_history", return_direction="positive", flag_direction="positive"),
            *_year_deciles("short_history", spreads=[0.04, 0.04, 0.04], start_year=2024),
        ]
    )
    return pd.DataFrame(rows)


def _coverage(feature_name: str, *, null_rate: float, pearson: float, spearman: float) -> dict[str, object]:
    return {
        "metric_type": "coverage",
        "feature_name": feature_name,
        "non_null_count": int(1000 * (1 - null_rate)),
        "null_count": int(1000 * null_rate),
        "null_rate": null_rate,
        "label_non_null_count": 1000,
        "classification_label_non_null_count": 1000,
        "pearson_corr_with_forward_63d_sector_relative_return": pearson,
        "spearman_corr_with_forward_63d_sector_relative_return": spearman,
    }


def _spread(feature_name: str, *, return_spread: float, flag_lift: float) -> dict[str, object]:
    return {
        "metric_type": "spread",
        "feature_name": feature_name,
        "top_decile_avg_forward_return": return_spread / 2,
        "bottom_decile_avg_forward_return": -return_spread / 2,
        "top_minus_bottom_avg_forward_return": return_spread,
        "top_decile_top_30pct_flag_rate": 0.30 + flag_lift / 2,
        "bottom_decile_top_30pct_flag_rate": 0.30 - flag_lift / 2,
        "top_minus_bottom_top_30pct_flag_rate": flag_lift,
    }


def _deciles(feature_name: str, *, return_direction: str, flag_direction: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for decile in range(1, 11):
        if return_direction == "positive":
            avg_return = decile / 100
        elif return_direction == "negative":
            avg_return = (11 - decile) / 100
        else:
            avg_return = 0.03 + (0.001 if decile % 2 == 0 else -0.001)

        if flag_direction == "positive":
            flag_rate = 0.20 + decile / 50
        elif flag_direction == "negative":
            flag_rate = 0.20 + (11 - decile) / 50
        else:
            flag_rate = 0.30 + (0.01 if decile % 2 == 0 else -0.01)

        rows.append(
            {
                "metric_type": "decile",
                "feature_name": feature_name,
                "decile": decile,
                "row_count": 100,
                "avg_forward_63d_sector_relative_return": avg_return,
                "median_forward_63d_sector_relative_return": avg_return,
                "top_30pct_sector_flag_rate": flag_rate,
            }
        )
    return rows


def _year_deciles(feature_name: str, *, spreads: list[float], start_year: int = 2006) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for offset, spread in enumerate(spreads):
        year = start_year + offset
        bottom_return = 0.01
        top_return = bottom_return + spread
        bottom_flag = 0.25
        top_flag = bottom_flag + spread
        for decile, avg_return, flag_rate in [(1, bottom_return, bottom_flag), (10, top_return, top_flag)]:
            rows.append(
                {
                    "metric_type": "year_decile",
                    "feature_name": feature_name,
                    "calendar_year": year,
                    "decile": decile,
                    "row_count": 100,
                    "avg_forward_63d_sector_relative_return": avg_return,
                    "median_forward_63d_sector_relative_return": avg_return,
                    "top_30pct_sector_flag_rate": flag_rate,
                }
            )
    return rows


def _year_decile_row(
    feature_name: str,
    calendar_year: int,
    decile: int,
    *,
    avg_return: float,
    flag_rate: float,
) -> dict[str, object]:
    return {
        "metric_type": "year_decile",
        "feature_name": feature_name,
        "calendar_year": calendar_year,
        "decile": decile,
        "row_count": 100,
        "avg_forward_63d_sector_relative_return": avg_return,
        "median_forward_63d_sector_relative_return": avg_return,
        "top_30pct_sector_flag_rate": flag_rate,
    }
