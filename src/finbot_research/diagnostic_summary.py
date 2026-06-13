from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from finbot_research.config import (
    feature_label_current_year_snapshot_csv_path,
    feature_label_current_year_snapshot_path,
    feature_label_diagnostics_path,
    feature_label_lookback_summary_csv_path,
    feature_label_lookback_summary_path,
    feature_label_report_path,
    feature_label_summary_csv_path,
    feature_label_summary_path,
    feature_label_year_spreads_csv_path,
    feature_label_year_spreads_path,
)
from finbot_research.io import read_parquet, write_csv, write_parquet
from finbot_research.metadata import write_metadata
from finbot_research.schemas import PRIMARY_CLASSIFICATION_LABEL, PRIMARY_REGRESSION_LABEL
from finbot_research.validation import ValidationError

DATASET_NAME = "equity_price_signal_summary"

COVERAGE_PROBLEM_NULL_RATE = 0.50
MIN_LABEL_NON_NULL_COUNT = 1
TOP_N_REPORT_FEATURES = 10
MAX_DECILE_TABLES = 5
MAX_YEAR_DETAIL_FEATURES = 4
MARKDOWN_YEAR_DETAIL_RECENT_YEARS = 15
MONOTONICITY_CORR_THRESHOLD = 0.30
MIN_VALID_DECILES = 5
MEANINGFUL_RETURN_SPREAD = 0.01
MEANINGFUL_FLAG_LIFT = 0.05
MIN_STABILITY_YEARS = 3
STABILITY_DIRECTION_THRESHOLD = 0.65
RECENT_SIGNAL_PERSISTENCE_THRESHOLD = 0.60
RECENT_SIGNAL_NEGATIVE_THRESHOLD = 0.40
MIN_RECENT_ASSESSMENT_YEARS = 3
TAIL_CLOSE_TO_ZERO_THRESHOLD = 0.005
LOOKBACK_WINDOWS = {
    "full_history": None,
    "last_20y": 20,
    "last_15y": 15,
    "last_10y": 10,
    "last_5y": 5,
}
LOOKBACK_SUMMARY_COLUMNS = (
    "full_history_years_with_valid_spread",
    "last_20y_years_with_valid_spread",
    "last_15y_years_with_valid_spread",
    "last_10y_years_with_valid_spread",
    "last_5y_years_with_valid_spread",
    "full_history_mean_year_spread",
    "last_20y_mean_year_spread",
    "last_15y_mean_year_spread",
    "last_10y_mean_year_spread",
    "last_5y_mean_year_spread",
    "full_history_pct_years_positive_spread",
    "last_20y_pct_years_positive_spread",
    "last_15y_pct_years_positive_spread",
    "last_10y_pct_years_positive_spread",
    "last_5y_pct_years_positive_spread",
    "full_history_mean_year_flag_lift",
    "last_20y_mean_year_flag_lift",
    "last_15y_mean_year_flag_lift",
    "last_10y_mean_year_flag_lift",
    "last_5y_mean_year_flag_lift",
    "full_history_pct_years_positive_flag_lift",
    "last_20y_pct_years_positive_flag_lift",
    "last_15y_pct_years_positive_flag_lift",
    "last_10y_pct_years_positive_flag_lift",
    "last_5y_pct_years_positive_flag_lift",
)

RECENT_SIGNAL_RULES = {
    "persistent_positive": "full_history, last_10y, and last_5y are positive by mean spread and pct-positive thresholds",
    "persistent_negative": "full_history, last_10y, and last_5y are negative by mean spread and pct-positive thresholds",
    "recent_positive_only": "full-history signal is mixed or weak, but last_10y and last_5y are positive",
    "recent_negative_only": "full-history signal is mixed or weak, but last_10y and last_5y are negative",
    "historical_positive_but_recent_weak": "full-history signal is positive, but last_10y or last_5y is not positive",
    "historical_negative_but_recent_weak": "full-history signal is negative, but last_10y or last_5y is not negative",
    "mixed_or_regime_dependent": "enough year data exists but no clear recent/full-history classification applies",
    "insufficient_year_data": f"fewer than {MIN_RECENT_ASSESSMENT_YEARS} valid years in required windows",
}

RANKING_FORMULA = (
    "abs(top_minus_bottom_avg_forward_return) + "
    "abs(top_minus_bottom_top_30pct_flag_rate) + "
    "0.25 * abs(avg_return_decile_spearman_corr) + "
    "0.25 * abs(flag_rate_decile_spearman_corr) + "
    "0.10 * max(pct_years_positive_spread, pct_years_negative_spread) - null_rate"
)

CATEGORY_RULES = {
    "coverage_problem": f"null_rate > {COVERAGE_PROBLEM_NULL_RATE} or label_non_null_count < {MIN_LABEL_NON_NULL_COUNT}",
    "bullish_candidate": "positive spread, positive hit-rate lift, positive Spearman correlation, positive decile monotonicity, and positive agreement",
    "risk_penalty_candidate": "negative spread, negative hit-rate lift, negative Spearman correlation, negative decile monotonicity, and negative agreement",
    "nonlinear_or_unstable": "meaningful top-minus-bottom spread or hit-rate lift with correlation or monotonicity disagreement",
    "weak_or_noisy": "all remaining features",
}

MONOTONICITY_RULES = {
    "positive": f"decile Spearman correlation >= {MONOTONICITY_CORR_THRESHOLD}",
    "negative": f"decile Spearman correlation <= -{MONOTONICITY_CORR_THRESHOLD}",
    "weak_or_nonmonotonic": f"absolute decile Spearman correlation < {MONOTONICITY_CORR_THRESHOLD}",
    "insufficient_deciles": f"fewer than {MIN_VALID_DECILES} valid deciles",
}

STABILITY_RULES = {
    "mostly_positive": f"pct_years_positive_spread >= {STABILITY_DIRECTION_THRESHOLD}",
    "mostly_negative": f"pct_years_negative_spread >= {STABILITY_DIRECTION_THRESHOLD}",
    "mixed_by_year": "enough valid years exist but neither direction dominates",
    "insufficient_year_data": f"fewer than {MIN_STABILITY_YEARS} valid years",
}

REPORT_SECTIONS = [
    "Purpose",
    "Executive Summary",
    "Key Findings",
    "Feature Candidates for Review",
    "Recent Signal Summary",
    "Selected Decile Curves",
    "Selected Year-by-Year Details",
    "Output File Guide",
    "Important Caveats",
    "Suggested Next Step",
]

DEFAULT_OUTPUT_POLICY = (
    "Parquet is canonical for machine-readable outputs, Markdown is canonical for human-readable review, "
    "CSV is optional convenience."
)

OUTPUT_FILE_PURPOSES = {
    "equity_price_signal_report.md": {
        "format": "markdown",
        "purpose": "Human-readable executive research summary for manual review.",
        "canonical": True,
    },
    "equity_price_signal_summary.parquet": {
        "format": "parquet",
        "purpose": "Canonical feature-level summary for downstream research code.",
        "canonical": True,
    },
    "equity_price_signal_summary.csv": {
        "format": "csv",
        "purpose": "Convenience export for manual inspection in spreadsheet tools.",
        "canonical": False,
    },
    "equity_price_signal_summary.metadata.json": {
        "format": "json",
        "purpose": "Documents inputs, outputs, rules, thresholds, and generation metadata.",
        "canonical": True,
    },
    "equity_price_signal_year_spreads.parquet": {
        "format": "parquet",
        "purpose": "Canonical year-by-year feature spread details.",
        "canonical": True,
    },
    "equity_price_signal_year_spreads.csv": {
        "format": "csv",
        "purpose": "Convenience export for manual inspection of year-by-year spread details.",
        "canonical": False,
    },
    "equity_price_signal_lookback_summary.parquet": {
        "format": "parquet",
        "purpose": "Canonical lookback-window summary by feature and window.",
        "canonical": True,
    },
    "equity_price_signal_lookback_summary.csv": {
        "format": "csv",
        "purpose": "Convenience export for manual inspection of lookback-window summaries.",
        "canonical": False,
    },
    "equity_price_signal_current_year_snapshot.parquet": {
        "format": "parquet",
        "purpose": "Canonical current partial-year/current-regime feature spread snapshot.",
        "canonical": True,
    },
    "equity_price_signal_current_year_snapshot.csv": {
        "format": "csv",
        "purpose": "Convenience export for manual inspection of the current partial-year snapshot.",
        "canonical": False,
    },
}

TAIL_EFFECT_FIELDS = [
    "top_decile_avg_forward_return",
    "top_decile_median_forward_return",
    "top_decile_avg_minus_median_forward_return",
    "bottom_decile_avg_forward_return",
    "bottom_decile_median_forward_return",
    "bottom_decile_avg_minus_median_forward_return",
    "top_minus_bottom_median_forward_return",
    "top_minus_bottom_avg_median_gap",
    "tail_effect_assessment",
]

TAIL_EFFECT_RULES = {
    "broad_positive": "top-minus-bottom average and median forward returns are both positive",
    "broad_negative": "top-minus-bottom average and median forward returns are both negative",
    "right_tail_positive": "average spread is positive while median spread is non-positive",
    "left_tail_negative": "average spread is negative while median spread is non-negative",
    "mean_median_disagreement": "average and median signs disagree outside the named tail cases",
    "limited_tail_evidence": f"average and median spreads are both within +/-{TAIL_CLOSE_TO_ZERO_THRESHOLD}",
    "insufficient_data": "required average or median decile fields are missing",
}


def build_diagnostic_summary(
    data_root: Path,
    *,
    include_partial_current_year_in_stability: bool = False,
) -> tuple[dict[str, Path], dict[str, Any]]:
    diagnostics_path = feature_label_diagnostics_path(data_root)
    summary_path = feature_label_summary_path(data_root)
    summary_csv_path = feature_label_summary_csv_path(data_root)
    report_path = feature_label_report_path(data_root)
    year_spreads_path = feature_label_year_spreads_path(data_root)
    year_spreads_csv_path = feature_label_year_spreads_csv_path(data_root)
    lookback_summary_path = feature_label_lookback_summary_path(data_root)
    lookback_summary_csv_path = feature_label_lookback_summary_csv_path(data_root)
    current_snapshot_path = feature_label_current_year_snapshot_path(data_root)
    current_snapshot_csv_path = feature_label_current_year_snapshot_csv_path(data_root)

    diagnostics = read_parquet(diagnostics_path)
    year_spreads = compute_year_spreads(
        diagnostics,
        include_partial_current_year_in_stability=include_partial_current_year_in_stability,
    )
    lookback_summary = compute_lookback_summary(year_spreads)
    current_snapshot = compute_current_year_snapshot(year_spreads)
    summary = summarize_diagnostics(
        diagnostics,
        year_spreads=year_spreads,
        lookback_summary=lookback_summary,
        current_snapshot=current_snapshot,
    )
    write_parquet(summary, summary_path)
    write_csv(summary, summary_csv_path)
    write_parquet(year_spreads, year_spreads_path)
    write_csv(year_spreads, year_spreads_csv_path)
    write_parquet(lookback_summary, lookback_summary_path)
    write_csv(lookback_summary, lookback_summary_csv_path)
    write_parquet(current_snapshot, current_snapshot_path)
    write_csv(current_snapshot, current_snapshot_csv_path)
    write_markdown_report(
        summary,
        report_path,
        diagnostics_path=diagnostics_path,
        diagnostics=diagnostics,
        year_spreads=year_spreads,
        lookback_summary=lookback_summary,
        current_snapshot=current_snapshot,
    )

    stats = summary_stats(summary, year_spreads=year_spreads)
    max_calendar_year = _max_calendar_year(year_spreads)
    metadata_path = write_metadata(
        dataset_name=DATASET_NAME,
        output_path=summary_path,
        input_paths=[diagnostics_path],
        dataframe=summary,
        extra_metadata={
            "dataset_type": "research_summary",
            "input_diagnostics_path": str(diagnostics_path),
            "output_paths": {
                "summary_parquet": str(summary_path),
                "summary_csv": str(summary_csv_path),
                "markdown_report": str(report_path),
                "year_spreads_parquet": str(year_spreads_path),
                "year_spreads_csv": str(year_spreads_csv_path),
                "lookback_summary_parquet": str(lookback_summary_path),
                "lookback_summary_csv": str(lookback_summary_csv_path),
                "current_year_snapshot_parquet": str(current_snapshot_path),
                "current_year_snapshot_csv": str(current_snapshot_csv_path),
            },
            "year_spreads_output_path": str(year_spreads_path),
            "lookback_summary_output_path": str(lookback_summary_path),
            "output_file_purposes": OUTPUT_FILE_PURPOSES,
            "default_output_policy": DEFAULT_OUTPUT_POLICY,
            "tail_effect_fields": TAIL_EFFECT_FIELDS,
            "tail_effect_assessment_rules": TAIL_EFFECT_RULES,
            "partial_year_handling": _partial_year_metadata(
                year_spreads,
                current_snapshot_path,
                include_partial_current_year_in_stability=include_partial_current_year_in_stability,
            ),
            "lookback_windows": _lookback_window_metadata(lookback_summary),
            "max_calendar_year": max_calendar_year,
            "features_analyzed": stats["features_analyzed"],
            "primary_regression_label": PRIMARY_REGRESSION_LABEL,
            "primary_classification_label": PRIMARY_CLASSIFICATION_LABEL,
            "candidate_category_counts": stats["category_counts"],
            "suggested_scorecard_use_counts": stats["suggested_use_counts"],
            "recent_signal_assessment_counts": stats["recent_signal_counts"],
            "ranking_formula": RANKING_FORMULA,
            "category_rules": CATEGORY_RULES,
            "monotonicity_thresholds": {
                "direction_threshold": MONOTONICITY_CORR_THRESHOLD,
                "min_valid_deciles": MIN_VALID_DECILES,
            },
            "monotonicity_rules": MONOTONICITY_RULES,
            "spread_correlation_agreement_logic": "Requires top-minus-bottom spread, hit-rate lift, Spearman correlation, and average-return decile monotonicity to point in the same direction for positive or negative agreement.",
            "stability_assessment_logic": STABILITY_RULES,
            "recent_signal_assessment_rules": RECENT_SIGNAL_RULES,
            "year_spread_method": "For each feature and calendar year, subtract decile 1 from decile 10 for average forward sector-relative return and top-30pct flag rate.",
            "report_sections_included": REPORT_SECTIONS,
            "max_decile_tables_included": MAX_DECILE_TABLES,
            "markdown_year_detail_feature_limit": MAX_YEAR_DETAIL_FEATURES,
            "top_features_by_summary_rank": _top_features_for_metadata(summary),
            "package_version": _package_version(),
        },
    )
    paths = {
        "summary_parquet": summary_path,
        "summary_csv": summary_csv_path,
        "markdown_report": report_path,
        "year_spreads_parquet": year_spreads_path,
        "year_spreads_csv": year_spreads_csv_path,
        "lookback_summary_parquet": lookback_summary_path,
        "lookback_summary_csv": lookback_summary_csv_path,
        "current_year_snapshot_parquet": current_snapshot_path,
        "current_year_snapshot_csv": current_snapshot_csv_path,
        "metadata": metadata_path,
    }
    return paths, stats


def compute_year_spreads(
    diagnostics: pd.DataFrame,
    *,
    include_partial_current_year_in_stability: bool = False,
) -> pd.DataFrame:
    _validate_diagnostics(diagnostics)
    year_deciles = diagnostics[diagnostics["metric_type"] == "year_decile"].copy()
    columns = [
        "feature_name",
        "calendar_year",
        "year_top_minus_bottom_avg_forward_return",
        "year_top_minus_bottom_top_30pct_flag_rate",
        "year_decile_1_row_count",
        "year_decile_10_row_count",
        "year_valid_spread_flag",
        "is_partial_year",
        "included_in_completed_year_stability",
    ]
    if year_deciles.empty:
        return pd.DataFrame(columns=columns)

    required = {"feature_name", "calendar_year", "decile", "avg_forward_63d_sector_relative_return"}
    if not required.issubset(year_deciles.columns):
        return pd.DataFrame(columns=columns)

    frame = year_deciles.copy()
    frame["calendar_year"] = pd.to_numeric(frame["calendar_year"], errors="coerce").astype("Int64")
    frame["decile"] = pd.to_numeric(frame["decile"], errors="coerce").astype("Int64")
    frame["avg_forward_63d_sector_relative_return"] = pd.to_numeric(
        frame["avg_forward_63d_sector_relative_return"],
        errors="coerce",
    )
    if "top_30pct_sector_flag_rate" in frame.columns:
        frame["top_30pct_sector_flag_rate"] = pd.to_numeric(frame["top_30pct_sector_flag_rate"], errors="coerce")
    else:
        frame["top_30pct_sector_flag_rate"] = pd.NA
    if "row_count" in frame.columns:
        frame["row_count"] = pd.to_numeric(frame["row_count"], errors="coerce")
    else:
        frame["row_count"] = pd.NA

    edge_rows = frame[frame["decile"].isin([1, 10])].copy()
    pivot = edge_rows.pivot_table(
        index=["feature_name", "calendar_year"],
        columns="decile",
        values=["avg_forward_63d_sector_relative_return", "top_30pct_sector_flag_rate", "row_count"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_decile_{int(decile)}" for metric, decile in pivot.columns]
    spreads = pivot.reset_index()
    for column in (
        "avg_forward_63d_sector_relative_return_decile_1",
        "avg_forward_63d_sector_relative_return_decile_10",
        "top_30pct_sector_flag_rate_decile_1",
        "top_30pct_sector_flag_rate_decile_10",
        "row_count_decile_1",
        "row_count_decile_10",
    ):
        if column not in spreads.columns:
            spreads[column] = pd.NA
    spreads["year_top_minus_bottom_avg_forward_return"] = (
        spreads["avg_forward_63d_sector_relative_return_decile_10"]
        - spreads["avg_forward_63d_sector_relative_return_decile_1"]
    )
    spreads["year_top_minus_bottom_top_30pct_flag_rate"] = (
        spreads["top_30pct_sector_flag_rate_decile_10"]
        - spreads["top_30pct_sector_flag_rate_decile_1"]
    )
    spreads["year_decile_1_row_count"] = spreads["row_count_decile_1"]
    spreads["year_decile_10_row_count"] = spreads["row_count_decile_10"]
    spreads["year_valid_spread_flag"] = (
        spreads["year_top_minus_bottom_avg_forward_return"].notna()
        & spreads["year_top_minus_bottom_top_30pct_flag_rate"].notna()
    )
    max_year = _max_calendar_year(spreads)
    spreads["is_partial_year"] = spreads["calendar_year"] == max_year if max_year is not None else False
    spreads["included_in_completed_year_stability"] = (
        True if include_partial_current_year_in_stability else ~spreads["is_partial_year"]
    )
    result = spreads[columns].sort_values(["feature_name", "calendar_year"]).reset_index(drop=True)
    return result


def compute_lookback_summary(year_spreads: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "feature_name",
        "lookback_window",
        "start_year",
        "end_year",
        "years_with_valid_spread",
        "years_positive_spread",
        "years_negative_spread",
        "pct_years_positive_spread",
        "pct_years_negative_spread",
        "mean_year_top_minus_bottom_avg_forward_return",
        "median_year_top_minus_bottom_avg_forward_return",
        "std_year_top_minus_bottom_avg_forward_return",
        "mean_year_top_minus_bottom_top_30pct_flag_rate",
        "median_year_top_minus_bottom_top_30pct_flag_rate",
        "std_year_top_minus_bottom_top_30pct_flag_rate",
        "years_positive_flag_lift",
        "years_negative_flag_lift",
        "pct_years_positive_flag_lift",
        "pct_years_negative_flag_lift",
    ]
    if year_spreads.empty:
        return pd.DataFrame(columns=columns)

    frame = year_spreads.copy()
    frame["calendar_year"] = pd.to_numeric(frame["calendar_year"], errors="coerce")
    if "included_in_completed_year_stability" not in frame.columns:
        frame["included_in_completed_year_stability"] = True
    frame = frame[frame["included_in_completed_year_stability"].fillna(False)]
    if frame.empty or frame["calendar_year"].dropna().empty:
        return pd.DataFrame(columns=columns)
    max_year = int(frame["calendar_year"].dropna().max())
    min_year = int(frame["calendar_year"].dropna().min())
    rows: list[dict[str, Any]] = []
    for feature_name, feature_rows in frame.groupby("feature_name", sort=False):
        for window_name, years in LOOKBACK_WINDOWS.items():
            start_year = min_year if years is None else max_year - years + 1
            end_year = max_year
            window_rows = feature_rows[
                (feature_rows["calendar_year"] >= start_year)
                & (feature_rows["calendar_year"] <= end_year)
                & (feature_rows["year_valid_spread_flag"])
            ].copy()
            rows.append(_lookback_row(feature_name, window_name, start_year, end_year, window_rows))
    return pd.DataFrame(rows, columns=columns)


def compute_current_year_snapshot(year_spreads: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "feature_name",
        "current_partial_year",
        "current_year_top_minus_bottom_avg_forward_return",
        "current_year_top_minus_bottom_top_30pct_flag_rate",
        "current_year_decile_1_row_count",
        "current_year_decile_10_row_count",
        "current_year_is_partial",
        "current_year_signal_direction",
    ]
    if year_spreads.empty:
        return pd.DataFrame(columns=columns)

    max_year = _max_calendar_year(year_spreads)
    if max_year is None:
        return pd.DataFrame(columns=columns)

    rows = year_spreads[pd.to_numeric(year_spreads["calendar_year"], errors="coerce") == max_year].copy()
    if rows.empty:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(
        {
            "feature_name": rows["feature_name"],
            "current_partial_year": max_year,
            "current_year_top_minus_bottom_avg_forward_return": rows["year_top_minus_bottom_avg_forward_return"],
            "current_year_top_minus_bottom_top_30pct_flag_rate": rows["year_top_minus_bottom_top_30pct_flag_rate"],
            "current_year_decile_1_row_count": rows["year_decile_1_row_count"],
            "current_year_decile_10_row_count": rows["year_decile_10_row_count"],
            "current_year_is_partial": rows.get("is_partial_year", True),
        }
    )
    result["current_year_signal_direction"] = result.apply(_current_year_signal_direction, axis=1)
    return result[columns].sort_values("feature_name").reset_index(drop=True)


def _lookback_row(
    feature_name: str,
    window_name: str,
    start_year: int,
    end_year: int,
    window_rows: pd.DataFrame,
) -> dict[str, Any]:
    spread_values = pd.to_numeric(window_rows["year_top_minus_bottom_avg_forward_return"], errors="coerce").dropna()
    flag_values = pd.to_numeric(window_rows["year_top_minus_bottom_top_30pct_flag_rate"], errors="coerce").dropna()
    years = int(len(spread_values))
    positive_spread = int((spread_values > 0).sum())
    negative_spread = int((spread_values < 0).sum())
    positive_flag = int((flag_values > 0).sum())
    negative_flag = int((flag_values < 0).sum())
    return {
        "feature_name": feature_name,
        "lookback_window": window_name,
        "start_year": int(start_year),
        "end_year": int(end_year),
        "years_with_valid_spread": years,
        "years_positive_spread": positive_spread,
        "years_negative_spread": negative_spread,
        "pct_years_positive_spread": positive_spread / years if years else None,
        "pct_years_negative_spread": negative_spread / years if years else None,
        "mean_year_top_minus_bottom_avg_forward_return": _finite_or_none(spread_values.mean()) if years else None,
        "median_year_top_minus_bottom_avg_forward_return": _finite_or_none(spread_values.median()) if years else None,
        "std_year_top_minus_bottom_avg_forward_return": _finite_or_none(spread_values.std(ddof=0)) if years else None,
        "mean_year_top_minus_bottom_top_30pct_flag_rate": _finite_or_none(flag_values.mean()) if len(flag_values) else None,
        "median_year_top_minus_bottom_top_30pct_flag_rate": _finite_or_none(flag_values.median()) if len(flag_values) else None,
        "std_year_top_minus_bottom_top_30pct_flag_rate": _finite_or_none(flag_values.std(ddof=0)) if len(flag_values) else None,
        "years_positive_flag_lift": positive_flag,
        "years_negative_flag_lift": negative_flag,
        "pct_years_positive_flag_lift": positive_flag / len(flag_values) if len(flag_values) else None,
        "pct_years_negative_flag_lift": negative_flag / len(flag_values) if len(flag_values) else None,
    }


def _lookback_wide_summary(lookback_summary: pd.DataFrame) -> pd.DataFrame:
    columns = ["feature_name", *LOOKBACK_SUMMARY_COLUMNS]
    if lookback_summary.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for feature_name, group in lookback_summary.groupby("feature_name", sort=False):
        row: dict[str, Any] = {"feature_name": feature_name}
        for _, window in group.iterrows():
            prefix = str(window["lookback_window"])
            row[f"{prefix}_years_with_valid_spread"] = window["years_with_valid_spread"]
            row[f"{prefix}_mean_year_spread"] = window["mean_year_top_minus_bottom_avg_forward_return"]
            row[f"{prefix}_pct_years_positive_spread"] = window["pct_years_positive_spread"]
            row[f"{prefix}_mean_year_flag_lift"] = window["mean_year_top_minus_bottom_top_30pct_flag_rate"]
            row[f"{prefix}_pct_years_positive_flag_lift"] = window["pct_years_positive_flag_lift"]
        rows.append(row)
    result = pd.DataFrame(rows)
    for column in LOOKBACK_SUMMARY_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    return result[columns]


def summarize_diagnostics(
    diagnostics: pd.DataFrame,
    *,
    year_spreads: pd.DataFrame | None = None,
    lookback_summary: pd.DataFrame | None = None,
    current_snapshot: pd.DataFrame | None = None,
) -> pd.DataFrame:
    _validate_diagnostics(diagnostics)
    coverage = diagnostics[diagnostics["metric_type"] == "coverage"].copy()
    spread = diagnostics[diagnostics["metric_type"] == "spread"].copy()
    deciles = diagnostics[diagnostics["metric_type"] == "decile"].copy()
    year_spreads = compute_year_spreads(diagnostics) if year_spreads is None else year_spreads
    lookback_summary = compute_lookback_summary(year_spreads) if lookback_summary is None else lookback_summary
    current_snapshot = compute_current_year_snapshot(year_spreads) if current_snapshot is None else current_snapshot
    if coverage.empty:
        raise ValidationError("diagnostics contains no coverage rows")
    if spread.empty:
        raise ValidationError("diagnostics contains no spread rows")

    coverage_summary = coverage[
        [
            "feature_name",
            "non_null_count",
            "null_rate",
            "label_non_null_count",
            "pearson_corr_with_forward_63d_sector_relative_return",
            "spearman_corr_with_forward_63d_sector_relative_return",
        ]
    ].rename(
        columns={
            "pearson_corr_with_forward_63d_sector_relative_return": "pearson_corr",
            "spearman_corr_with_forward_63d_sector_relative_return": "spearman_corr",
        }
    )
    spread_summary = spread[
        [
            "feature_name",
            "top_decile_avg_forward_return",
            "bottom_decile_avg_forward_return",
            "top_minus_bottom_avg_forward_return",
            "top_decile_top_30pct_flag_rate",
            "bottom_decile_top_30pct_flag_rate",
            "top_minus_bottom_top_30pct_flag_rate",
        ]
    ]
    summary = coverage_summary.merge(spread_summary, on="feature_name", how="inner")
    if summary.empty:
        raise ValidationError("diagnostics coverage and spread rows do not overlap by feature_name")
    summary = summary.merge(_monotonicity_summary(deciles), on="feature_name", how="left")
    summary = summary.merge(_year_stability_summary_from_spreads(year_spreads), on="feature_name", how="left")
    summary = summary.merge(_lookback_wide_summary(lookback_summary), on="feature_name", how="left")
    summary = summary.merge(_tail_effect_summary(deciles), on="feature_name", how="left")
    summary = summary.merge(current_snapshot, on="feature_name", how="left")

    for column in _NUMERIC_COLUMNS:
        if column in summary.columns:
            summary[column] = pd.to_numeric(summary[column], errors="coerce")
    summary["avg_return_monotonicity_direction"] = summary["avg_return_monotonicity_direction"].fillna("insufficient_deciles")
    summary["flag_rate_monotonicity_direction"] = summary["flag_rate_monotonicity_direction"].fillna("insufficient_deciles")
    summary["stability_assessment"] = summary["stability_assessment"].fillna("insufficient_year_data")
    summary["tail_effect_assessment"] = summary["tail_effect_assessment"].fillna("insufficient_data")
    summary["current_year_signal_direction"] = summary["current_year_signal_direction"].fillna("insufficient_data")

    summary["signal_direction"] = summary.apply(_signal_direction, axis=1)
    summary["spread_correlation_agreement"] = summary.apply(_spread_correlation_agreement, axis=1)
    summary["candidate_category"] = summary.apply(_candidate_category, axis=1)
    summary["recent_signal_assessment"] = summary.apply(_recent_signal_assessment, axis=1)
    summary["suggested_scorecard_use"] = summary.apply(_suggested_scorecard_use, axis=1)
    summary["summary_rank_score"] = summary.apply(_summary_rank_score, axis=1)
    summary["notes"] = summary.apply(_notes, axis=1)
    summary = summary.sort_values("summary_rank_score", ascending=False, na_position="last").reset_index(drop=True)
    summary["summary_rank"] = range(1, len(summary) + 1)
    columns = [
        "feature_name",
        "non_null_count",
        "null_rate",
        "label_non_null_count",
        "pearson_corr",
        "spearman_corr",
        "top_decile_avg_forward_return",
        "top_decile_median_forward_return",
        "top_decile_avg_minus_median_forward_return",
        "bottom_decile_avg_forward_return",
        "bottom_decile_median_forward_return",
        "bottom_decile_avg_minus_median_forward_return",
        "top_minus_bottom_avg_forward_return",
        "top_minus_bottom_median_forward_return",
        "top_minus_bottom_avg_median_gap",
        "top_decile_top_30pct_flag_rate",
        "bottom_decile_top_30pct_flag_rate",
        "top_minus_bottom_top_30pct_flag_rate",
        "avg_return_decile_spearman_corr",
        "flag_rate_decile_spearman_corr",
        "avg_return_monotonicity_direction",
        "flag_rate_monotonicity_direction",
        "spread_correlation_agreement",
        "years_with_valid_spread",
        "years_positive_spread",
        "years_negative_spread",
        "pct_years_positive_spread",
        "pct_years_negative_spread",
        "year_spread_mean",
        "year_spread_std",
        "stability_assessment",
        "signal_direction",
        "candidate_category",
        "recent_signal_assessment",
        "tail_effect_assessment",
        "suggested_scorecard_use",
        *LOOKBACK_SUMMARY_COLUMNS,
        "current_partial_year",
        "current_year_top_minus_bottom_avg_forward_return",
        "current_year_top_minus_bottom_top_30pct_flag_rate",
        "current_year_decile_1_row_count",
        "current_year_decile_10_row_count",
        "current_year_is_partial",
        "current_year_signal_direction",
        "summary_rank_score",
        "summary_rank",
        "notes",
    ]
    return summary[columns]


def write_markdown_report(
    summary: pd.DataFrame,
    path: Path,
    *,
    diagnostics_path: Path,
    diagnostics: pd.DataFrame | None = None,
    year_spreads: pd.DataFrame | None = None,
    lookback_summary: pd.DataFrame | None = None,
    current_snapshot: pd.DataFrame | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = summary_stats(summary)
    decile_rows = diagnostics[diagnostics["metric_type"] == "decile"].copy() if diagnostics is not None else pd.DataFrame()
    year_spreads = year_spreads if year_spreads is not None else pd.DataFrame()
    current_snapshot = current_snapshot if current_snapshot is not None else pd.DataFrame()
    lines = [
        "# Equity Price Signal Diagnostics Summary",
        "",
        "## Purpose",
        "",
        "This report summarizes feature/label diagnostics for price-based and relative equity features. It is intended to guide future scorecard research, not to make trading recommendations.",
        "",
        "## Executive Summary",
        "",
        f"- Features analyzed: {stats['features_analyzed']}",
        f"- Candidate categories: {_counts_sentence(stats['category_counts'])}",
        f"- Recent signal assessments: {_counts_sentence(stats['recent_signal_counts'])}",
        f"- Current partial year: {_current_year_sentence(current_snapshot)}",
        "",
        _executive_interpretation(stats),
        "",
        "## Key Findings",
        "",
        *_key_findings(summary, stats),
        "",
        "## Feature Candidates for Review",
        "",
        _markdown_table(_feature_candidates_for_review(summary)),
        "",
        "## Recent and Regime Signal Summary",
        "",
        _markdown_table(_recent_regime_rows(summary)),
        "",
        "## Current Partial Year Snapshot",
        "",
        "The current calendar year is shown separately because it may be incomplete. Completed-year stability calculations exclude it by default.",
        "",
        _markdown_table(_current_year_snapshot_rows(summary)),
        "",
        "## Selected Decile Curves",
        "",
        *_decile_curve_sections(summary, decile_rows),
        "",
        "## Selected Year-by-Year Details",
        "",
        *_year_detail_sections(summary, year_spreads),
        "",
        "## Output File Guide",
        "",
        "Parquet files are canonical machine-readable research outputs. Markdown is the human-readable summary. CSV files are convenience exports and can be disabled later behind a flag.",
        "",
        _markdown_table(_output_file_guide_rows()),
        "",
        "## Important Caveats",
        "",
        "- Candidate categories are research labels, not trading rules.",
        "- The diagnostics parquet is the source for the report: `" + str(diagnostics_path) + "`.",
        f"- The primary labels are `{PRIMARY_REGRESSION_LABEL}` and `{PRIMARY_CLASSIFICATION_LABEL}`.",
        "- The available history begins in the late 1990s and spans multiple market regimes; recent-window diagnostics are included so older periods do not silently dominate decisions.",
        "- Top-minus-bottom spreads can be misleading when decile curves are non-monotonic or rank correlations disagree.",
        "- Average-minus-median gaps are shown to highlight possible right-tail or left-tail effects; broad median support is stronger evidence than a mean-only result.",
        "- The current calendar year remains visible in the report and snapshot but is excluded from completed-year stability and lookback calculations by default.",
        "- Positive volatility results may indicate nonlinear behavior, regime effects, or universe effects, not necessarily that volatility should be rewarded.",
        "- Liquidity may be better used as a tradability filter than an alpha component.",
        "- Diagnostics do not replace backtesting.",
        "",
        "## Suggested Next Step",
        "",
        "Review the selected volatility, liquidity, drawdown, and momentum/reversal diagnostics manually, then prototype a simple price-based scorecard in `finbot-research` using only features with interpretable behavior and acceptable recent-window evidence. Keep the parquet outputs as the source of full detail.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def summary_stats(summary: pd.DataFrame, year_spreads: pd.DataFrame | None = None) -> dict[str, Any]:
    counts = summary["candidate_category"].value_counts().to_dict() if not summary.empty else {}
    for category in ("bullish_candidate", "risk_penalty_candidate", "nonlinear_or_unstable", "weak_or_noisy", "coverage_problem"):
        counts.setdefault(category, 0)
    suggested_use_counts = summary["suggested_scorecard_use"].value_counts().to_dict() if "suggested_scorecard_use" in summary.columns else {}
    recent_signal_counts = summary["recent_signal_assessment"].value_counts().to_dict() if "recent_signal_assessment" in summary.columns else {}
    for assessment in RECENT_SIGNAL_RULES:
        recent_signal_counts.setdefault(assessment, 0)
    stats = {
        "features_analyzed": int(len(summary)),
        "category_counts": {key: int(value) for key, value in counts.items()},
        "suggested_use_counts": {key: int(value) for key, value in suggested_use_counts.items()},
        "recent_signal_counts": {key: int(value) for key, value in recent_signal_counts.items()},
        "top_bullish": _top_feature_names(summary, "bullish_candidate"),
        "top_risk_penalty": _top_feature_names(summary, "risk_penalty_candidate"),
        "top_nonlinear": _top_feature_names(summary, "nonlinear_or_unstable"),
    }
    stats.update(_partial_year_stats(year_spreads if year_spreads is not None else pd.DataFrame()))
    return stats


def terminal_summary(paths: dict[str, Path], stats: dict[str, Any]) -> str:
    lines = [
        "Wrote diagnostics summary:",
        "",
        "Human-readable output:",
        f"- Markdown report: {paths['markdown_report']}",
        "",
        "Canonical machine-readable outputs:",
        f"- Summary parquet: {paths['summary_parquet']}",
        f"- Year spreads parquet: {paths['year_spreads_parquet']}",
        f"- Lookback summary parquet: {paths['lookback_summary_parquet']}",
        f"- Current year snapshot parquet: {paths['current_year_snapshot_parquet']}",
        f"- Metadata JSON: {paths['metadata']}",
        "",
        "Convenience exports:",
        f"- Summary CSV: {paths['summary_csv']}",
        f"- Year spreads CSV: {paths['year_spreads_csv']}",
        f"- Lookback summary CSV: {paths['lookback_summary_csv']}",
        f"- Current year snapshot CSV: {paths['current_year_snapshot_csv']}",
        "",
        f"Features analyzed: {stats['features_analyzed']}",
        f"Current partial year: {stats.get('current_partial_year', 'none')}",
        f"Partial year included in stability: {stats.get('partial_year_included_in_stability', False)}",
        f"Persistent positive: {stats['recent_signal_counts'].get('persistent_positive', 0)}",
        f"Persistent negative: {stats['recent_signal_counts'].get('persistent_negative', 0)}",
        f"Recent positive only: {stats['recent_signal_counts'].get('recent_positive_only', 0)}",
        f"Recent negative only: {stats['recent_signal_counts'].get('recent_negative_only', 0)}",
        f"Historical positive but recent weak: {stats['recent_signal_counts'].get('historical_positive_but_recent_weak', 0)}",
        f"Historical negative but recent weak: {stats['recent_signal_counts'].get('historical_negative_but_recent_weak', 0)}",
        f"Mixed/regime-dependent: {stats['recent_signal_counts'].get('mixed_or_regime_dependent', 0)}",
        f"Insufficient year data: {stats['recent_signal_counts'].get('insufficient_year_data', 0)}",
        "",
        "Top positive candidates:",
        *_numbered(stats["top_bullish"]),
        "",
        "Top risk penalty candidates:",
        *_numbered(stats["top_risk_penalty"]),
        "",
        "Top nonlinear/unstable candidates:",
        *_numbered(stats["top_nonlinear"]),
    ]
    return "\n".join(lines)


_NUMERIC_COLUMNS = (
    "non_null_count",
    "null_rate",
    "label_non_null_count",
    "pearson_corr",
    "spearman_corr",
    "top_decile_avg_forward_return",
    "top_decile_median_forward_return",
    "top_decile_avg_minus_median_forward_return",
    "bottom_decile_avg_forward_return",
    "bottom_decile_median_forward_return",
    "bottom_decile_avg_minus_median_forward_return",
    "top_minus_bottom_avg_forward_return",
    "top_minus_bottom_median_forward_return",
    "top_minus_bottom_avg_median_gap",
    "top_decile_top_30pct_flag_rate",
    "bottom_decile_top_30pct_flag_rate",
    "top_minus_bottom_top_30pct_flag_rate",
    "avg_return_decile_spearman_corr",
    "flag_rate_decile_spearman_corr",
    "years_with_valid_spread",
    "years_positive_spread",
    "years_negative_spread",
    "pct_years_positive_spread",
    "pct_years_negative_spread",
    "year_spread_mean",
    "year_spread_std",
    *LOOKBACK_SUMMARY_COLUMNS,
    "current_partial_year",
    "current_year_top_minus_bottom_avg_forward_return",
    "current_year_top_minus_bottom_top_30pct_flag_rate",
    "current_year_decile_1_row_count",
    "current_year_decile_10_row_count",
)


def _validate_diagnostics(diagnostics: pd.DataFrame) -> None:
    missing = [column for column in ("metric_type", "feature_name") if column not in diagnostics.columns]
    if missing:
        raise ValidationError(f"diagnostics missing required columns: {missing}")


def _monotonicity_summary(deciles: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "feature_name",
        "avg_return_decile_spearman_corr",
        "flag_rate_decile_spearman_corr",
        "avg_return_monotonicity_direction",
        "flag_rate_monotonicity_direction",
    ]
    if deciles.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for feature_name, group in deciles.groupby("feature_name", sort=False):
        avg_corr = _decile_spearman(group, "avg_forward_63d_sector_relative_return")
        flag_corr = _decile_spearman(group, "top_30pct_sector_flag_rate")
        rows.append(
            {
                "feature_name": feature_name,
                "avg_return_decile_spearman_corr": avg_corr,
                "flag_rate_decile_spearman_corr": flag_corr,
                "avg_return_monotonicity_direction": _monotonicity_direction(avg_corr),
                "flag_rate_monotonicity_direction": _monotonicity_direction(flag_corr),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _decile_spearman(group: pd.DataFrame, value_column: str) -> float | None:
    if value_column not in group.columns or "decile" not in group.columns:
        return None
    paired = group[["decile", value_column]].copy()
    paired["decile"] = pd.to_numeric(paired["decile"], errors="coerce")
    paired[value_column] = pd.to_numeric(paired[value_column], errors="coerce")
    paired = paired.dropna().drop_duplicates(subset=["decile"])
    if len(paired) < MIN_VALID_DECILES:
        return None
    value = paired["decile"].rank(method="average").corr(paired[value_column].rank(method="average"), method="pearson")
    return _finite_or_none(value)


def _monotonicity_direction(correlation: float | None) -> str:
    if correlation is None or pd.isna(correlation):
        return "insufficient_deciles"
    if correlation >= MONOTONICITY_CORR_THRESHOLD:
        return "positive"
    if correlation <= -MONOTONICITY_CORR_THRESHOLD:
        return "negative"
    return "weak_or_nonmonotonic"


def _tail_effect_summary(deciles: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "feature_name",
        "top_decile_median_forward_return",
        "top_decile_avg_minus_median_forward_return",
        "bottom_decile_median_forward_return",
        "bottom_decile_avg_minus_median_forward_return",
        "top_minus_bottom_median_forward_return",
        "top_minus_bottom_avg_median_gap",
        "tail_effect_assessment",
    ]
    required = {
        "feature_name",
        "decile",
        "avg_forward_63d_sector_relative_return",
        "median_forward_63d_sector_relative_return",
    }
    if deciles.empty or not required.issubset(deciles.columns):
        return pd.DataFrame(columns=columns)

    frame = deciles.copy()
    frame["decile"] = pd.to_numeric(frame["decile"], errors="coerce").astype("Int64")
    frame["avg_forward_63d_sector_relative_return"] = pd.to_numeric(
        frame["avg_forward_63d_sector_relative_return"],
        errors="coerce",
    )
    frame["median_forward_63d_sector_relative_return"] = pd.to_numeric(
        frame["median_forward_63d_sector_relative_return"],
        errors="coerce",
    )
    edge_rows = frame[frame["decile"].isin([1, 10])].copy()
    pivot = edge_rows.pivot_table(
        index="feature_name",
        columns="decile",
        values=["avg_forward_63d_sector_relative_return", "median_forward_63d_sector_relative_return"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_decile_{int(decile)}" for metric, decile in pivot.columns]
    result = pivot.reset_index()
    for column in (
        "avg_forward_63d_sector_relative_return_decile_1",
        "avg_forward_63d_sector_relative_return_decile_10",
        "median_forward_63d_sector_relative_return_decile_1",
        "median_forward_63d_sector_relative_return_decile_10",
    ):
        if column not in result.columns:
            result[column] = pd.NA
    result["top_decile_median_forward_return"] = result["median_forward_63d_sector_relative_return_decile_10"]
    result["bottom_decile_median_forward_return"] = result["median_forward_63d_sector_relative_return_decile_1"]
    result["top_decile_avg_minus_median_forward_return"] = (
        result["avg_forward_63d_sector_relative_return_decile_10"]
        - result["median_forward_63d_sector_relative_return_decile_10"]
    )
    result["bottom_decile_avg_minus_median_forward_return"] = (
        result["avg_forward_63d_sector_relative_return_decile_1"]
        - result["median_forward_63d_sector_relative_return_decile_1"]
    )
    result["top_minus_bottom_median_forward_return"] = (
        result["top_decile_median_forward_return"] - result["bottom_decile_median_forward_return"]
    )
    result["top_minus_bottom_avg_median_gap"] = (
        result["avg_forward_63d_sector_relative_return_decile_10"]
        - result["avg_forward_63d_sector_relative_return_decile_1"]
        - result["top_minus_bottom_median_forward_return"]
    )
    result["tail_effect_assessment"] = result.apply(_tail_effect_assessment, axis=1)
    return result[columns]


def _tail_effect_assessment(row: pd.Series) -> str:
    avg_spread = row.get("avg_forward_63d_sector_relative_return_decile_10") - row.get(
        "avg_forward_63d_sector_relative_return_decile_1"
    )
    median_spread = row.get("top_minus_bottom_median_forward_return")
    if pd.isna(avg_spread) or pd.isna(median_spread):
        return "insufficient_data"
    avg_spread = float(avg_spread)
    median_spread = float(median_spread)
    if abs(avg_spread) < TAIL_CLOSE_TO_ZERO_THRESHOLD and abs(median_spread) < TAIL_CLOSE_TO_ZERO_THRESHOLD:
        return "limited_tail_evidence"
    if avg_spread > 0 and median_spread > 0:
        return "broad_positive"
    if avg_spread < 0 and median_spread < 0:
        return "broad_negative"
    if avg_spread > 0 and median_spread <= 0:
        return "right_tail_positive"
    if avg_spread < 0 and median_spread >= 0:
        return "left_tail_negative"
    return "mean_median_disagreement"


def _year_stability_summary_from_spreads(year_spreads: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "feature_name",
        "years_with_valid_spread",
        "years_positive_spread",
        "years_negative_spread",
        "pct_years_positive_spread",
        "pct_years_negative_spread",
        "year_spread_mean",
        "year_spread_std",
        "stability_assessment",
    ]
    if year_spreads.empty:
        return pd.DataFrame(columns=columns)

    required = {"feature_name", "year_valid_spread_flag", "year_top_minus_bottom_avg_forward_return"}
    if not required.issubset(year_spreads.columns):
        return pd.DataFrame(columns=columns)

    frame = year_spreads.copy()
    if "included_in_completed_year_stability" not in frame.columns:
        frame["included_in_completed_year_stability"] = True
    frame = frame[frame["year_valid_spread_flag"] & frame["included_in_completed_year_stability"].fillna(False)].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for feature_name, group in frame.groupby("feature_name", sort=False):
        spread_values = pd.to_numeric(group["year_top_minus_bottom_avg_forward_return"], errors="coerce").dropna()
        years = int(len(spread_values))
        positive = int((spread_values > 0).sum())
        negative = int((spread_values < 0).sum())
        pct_positive = positive / years if years else None
        pct_negative = negative / years if years else None
        rows.append(
            {
                "feature_name": feature_name,
                "years_with_valid_spread": years,
                "years_positive_spread": positive,
                "years_negative_spread": negative,
                "pct_years_positive_spread": pct_positive,
                "pct_years_negative_spread": pct_negative,
                "year_spread_mean": _finite_or_none(spread_values.mean()) if years else None,
                "year_spread_std": _finite_or_none(spread_values.std(ddof=0)) if years else None,
                "stability_assessment": _stability_assessment(years, pct_positive, pct_negative),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _year_spreads(group: pd.DataFrame) -> pd.DataFrame:
    frame = group.copy()
    frame["decile"] = pd.to_numeric(frame["decile"], errors="coerce")
    frame["avg_forward_63d_sector_relative_return"] = pd.to_numeric(
        frame["avg_forward_63d_sector_relative_return"],
        errors="coerce",
    )
    bottom = frame[frame["decile"] == 1][["calendar_year", "avg_forward_63d_sector_relative_return"]].rename(
        columns={"avg_forward_63d_sector_relative_return": "bottom_return"}
    )
    top = frame[frame["decile"] == 10][["calendar_year", "avg_forward_63d_sector_relative_return"]].rename(
        columns={"avg_forward_63d_sector_relative_return": "top_return"}
    )
    spreads = top.merge(bottom, on="calendar_year", how="inner")
    spreads["year_top_minus_bottom_avg_forward_return"] = spreads["top_return"] - spreads["bottom_return"]
    if "top_30pct_sector_flag_rate" in frame.columns:
        frame["top_30pct_sector_flag_rate"] = pd.to_numeric(frame["top_30pct_sector_flag_rate"], errors="coerce")
        bottom_flag = frame[frame["decile"] == 1][["calendar_year", "top_30pct_sector_flag_rate"]].rename(
            columns={"top_30pct_sector_flag_rate": "bottom_flag_rate"}
        )
        top_flag = frame[frame["decile"] == 10][["calendar_year", "top_30pct_sector_flag_rate"]].rename(
            columns={"top_30pct_sector_flag_rate": "top_flag_rate"}
        )
        flag_spreads = top_flag.merge(bottom_flag, on="calendar_year", how="inner")
        spreads = spreads.merge(flag_spreads, on="calendar_year", how="left")
        spreads["year_top_minus_bottom_top_30pct_flag_rate"] = spreads["top_flag_rate"] - spreads["bottom_flag_rate"]
    return spreads


def _stability_assessment(years: int, pct_positive: float | None, pct_negative: float | None) -> str:
    if years < MIN_STABILITY_YEARS:
        return "insufficient_year_data"
    if pct_positive is not None and pct_positive >= STABILITY_DIRECTION_THRESHOLD:
        return "mostly_positive"
    if pct_negative is not None and pct_negative >= STABILITY_DIRECTION_THRESHOLD:
        return "mostly_negative"
    return "mixed_by_year"


def _signal_direction(row: pd.Series) -> str:
    return_spread = row["top_minus_bottom_avg_forward_return"]
    flag_lift = row["top_minus_bottom_top_30pct_flag_rate"]
    if pd.notna(return_spread) and pd.notna(flag_lift):
        if return_spread > 0 and flag_lift > 0:
            return "positive"
        if return_spread < 0 and flag_lift < 0:
            return "negative"
    return "mixed"


def _spread_correlation_agreement(row: pd.Series) -> str:
    required = [
        "top_minus_bottom_avg_forward_return",
        "top_minus_bottom_top_30pct_flag_rate",
        "spearman_corr",
        "avg_return_decile_spearman_corr",
    ]
    values = [row.get(column) for column in required]
    if any(pd.isna(value) for value in values):
        return "insufficient_data"
    return_spread, flag_lift, spearman_corr, decile_corr = [float(value) for value in values]
    if return_spread > 0 and flag_lift > 0 and spearman_corr > 0 and decile_corr > 0:
        return "positive_agreement"
    if return_spread < 0 and flag_lift < 0 and spearman_corr < 0 and decile_corr < 0:
        return "negative_agreement"
    spread_direction = _direction_from_pair(return_spread, flag_lift)
    corr_direction = _direction_from_pair(spearman_corr, decile_corr)
    if spread_direction in {"positive", "negative"} and corr_direction in {"positive", "negative"} and spread_direction != corr_direction:
        return "disagreement"
    if spread_direction in {"positive", "negative"} and _has_opposing_value(spread_direction, [spearman_corr, decile_corr]):
        return "disagreement"
    if corr_direction in {"positive", "negative"} and _has_opposing_value(corr_direction, [return_spread, flag_lift]):
        return "disagreement"
    return "weak_or_mixed"


def _candidate_category(row: pd.Series) -> str:
    null_rate = row["null_rate"]
    label_count = row["label_non_null_count"]
    if pd.notna(null_rate) and null_rate > COVERAGE_PROBLEM_NULL_RATE:
        return "coverage_problem"
    if pd.isna(label_count) or label_count < MIN_LABEL_NON_NULL_COUNT:
        return "coverage_problem"
    if row["spread_correlation_agreement"] == "positive_agreement":
        return "bullish_candidate"
    if row["spread_correlation_agreement"] == "negative_agreement":
        return "risk_penalty_candidate"
    if (
        row["spread_correlation_agreement"] == "disagreement"
        and row["signal_direction"] in {"positive", "negative"}
        and _has_meaningful_spread(row)
    ):
        return "nonlinear_or_unstable"
    return "weak_or_noisy"


def _suggested_scorecard_use(row: pd.Series) -> str:
    tail_effect = row.get("tail_effect_assessment")
    if row["candidate_category"] == "coverage_problem":
        return "coverage_issue"
    if _is_liquidity_feature(row["feature_name"]):
        return "liquidity_filter_candidate"
    if row["candidate_category"] == "bullish_candidate" and row["recent_signal_assessment"] in {
        "persistent_positive",
        "recent_positive_only",
    }:
        if tail_effect == "broad_positive":
            return "positive_component_candidate"
        if tail_effect == "right_tail_positive":
            return "upside_optional_component_requires_review"
    if row["candidate_category"] == "risk_penalty_candidate" and row["recent_signal_assessment"] in {
        "persistent_negative",
        "recent_negative_only",
    }:
        if tail_effect == "broad_negative":
            return "risk_penalty_candidate"
    if row["candidate_category"] == "nonlinear_or_unstable" and tail_effect in {
        "right_tail_positive",
        "left_tail_negative",
        "mean_median_disagreement",
    }:
        return "nonlinear_or_tail_feature_requires_review"
    if row["candidate_category"] == "nonlinear_or_unstable" or row["recent_signal_assessment"] == "mixed_or_regime_dependent":
        return "nonlinear_feature_requires_review"
    return "ignore_for_v1"


def _recent_signal_assessment(row: pd.Series) -> str:
    full_state = _window_signal_state(row, "full_history")
    last_10y_state = _window_signal_state(row, "last_10y")
    last_5y_state = _window_signal_state(row, "last_5y")
    if "insufficient" in {full_state, last_10y_state, last_5y_state}:
        return "insufficient_year_data"
    if full_state == "positive" and last_10y_state == "positive" and last_5y_state == "positive":
        return "persistent_positive"
    if full_state == "negative" and last_10y_state == "negative" and last_5y_state == "negative":
        return "persistent_negative"
    if full_state == "mixed" and last_10y_state == "positive" and last_5y_state == "positive":
        return "recent_positive_only"
    if full_state == "mixed" and last_10y_state == "negative" and last_5y_state == "negative":
        return "recent_negative_only"
    if full_state == "positive" and (last_10y_state != "positive" or last_5y_state != "positive"):
        return "historical_positive_but_recent_weak"
    if full_state == "negative" and (last_10y_state != "negative" or last_5y_state != "negative"):
        return "historical_negative_but_recent_weak"
    return "mixed_or_regime_dependent"


def _window_signal_state(row: pd.Series, prefix: str) -> str:
    years = row.get(f"{prefix}_years_with_valid_spread")
    mean_spread = row.get(f"{prefix}_mean_year_spread")
    pct_positive = row.get(f"{prefix}_pct_years_positive_spread")
    if pd.isna(years) or pd.isna(mean_spread) or pd.isna(pct_positive):
        return "insufficient"
    if int(years) < MIN_RECENT_ASSESSMENT_YEARS:
        return "insufficient"
    mean_spread = float(mean_spread)
    pct_positive = float(pct_positive)
    if pct_positive >= RECENT_SIGNAL_PERSISTENCE_THRESHOLD and mean_spread > 0:
        return "positive"
    if pct_positive <= RECENT_SIGNAL_NEGATIVE_THRESHOLD and mean_spread < 0:
        return "negative"
    return "mixed"


def _summary_rank_score(row: pd.Series) -> float:
    return_spread = _abs_or_zero(row["top_minus_bottom_avg_forward_return"])
    flag_lift = _abs_or_zero(row["top_minus_bottom_top_30pct_flag_rate"])
    avg_mono = 0.25 * _abs_or_zero(row.get("avg_return_decile_spearman_corr"))
    flag_mono = 0.25 * _abs_or_zero(row.get("flag_rate_decile_spearman_corr"))
    stability = 0.10 * max(
        _zero_if_missing(row.get("pct_years_positive_spread")),
        _zero_if_missing(row.get("pct_years_negative_spread")),
    )
    null_penalty = float(row["null_rate"]) if pd.notna(row["null_rate"]) else 1.0
    return return_spread + flag_lift + avg_mono + flag_mono + stability - null_penalty


def _notes(row: pd.Series) -> str:
    if _is_liquidity_feature(row["feature_name"]):
        return "Liquidity may be more useful as a tradability filter than as a positive alpha component."
    if row["candidate_category"] == "coverage_problem":
        return "Coverage is too sparse for reliable interpretation."
    if row["candidate_category"] == "weak_or_noisy":
        return "Feature does not show a clear or stable relationship with the primary label."
    if row["spread_correlation_agreement"] == "positive_agreement":
        return "Feature shows positive spread, positive hit-rate lift, positive rank correlation, and positive decile monotonicity."
    if row["spread_correlation_agreement"] == "negative_agreement":
        return "High feature values are associated with weaker forward returns and lower top-sector outcome rates."
    if row["spread_correlation_agreement"] == "disagreement":
        return "Top-minus-bottom spread and rank/decile monotonicity disagree; inspect decile curve before using in a scorecard."
    return "Feature does not show a clear or stable relationship with the primary label."


def _abs_or_zero(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    return abs(float(value))


def _zero_if_missing(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    return float(value)


def _finite_or_none(value: Any) -> float | None:
    if pd.isna(value):
        return None
    numeric = float(value)
    if numeric == float("inf") or numeric == float("-inf"):
        return None
    return numeric


def _direction_from_pair(left: float, right: float) -> str:
    if left > 0 and right > 0:
        return "positive"
    if left < 0 and right < 0:
        return "negative"
    return "mixed"


def _has_opposing_value(direction: str, values: list[float]) -> bool:
    if direction == "positive":
        return any(value < 0 for value in values)
    if direction == "negative":
        return any(value > 0 for value in values)
    return False


def _has_meaningful_spread(row: pd.Series) -> bool:
    return (
        _abs_or_zero(row.get("top_minus_bottom_avg_forward_return")) >= MEANINGFUL_RETURN_SPREAD
        or _abs_or_zero(row.get("top_minus_bottom_top_30pct_flag_rate")) >= MEANINGFUL_FLAG_LIFT
    )


def _is_liquidity_feature(feature_name: Any) -> bool:
    return "average_dollar_volume" in str(feature_name)


def _category_rows(summary: pd.DataFrame, category: str, limit: int = TOP_N_REPORT_FEATURES) -> pd.DataFrame:
    columns = [
        "feature_name",
        "top_minus_bottom_avg_forward_return",
        "top_minus_bottom_top_30pct_flag_rate",
        "spearman_corr",
        "avg_return_decile_spearman_corr",
        "stability_assessment",
        "recent_signal_assessment",
        "suggested_scorecard_use",
        "null_rate",
        "notes",
    ]
    rows = summary[summary["candidate_category"] == category].head(limit)
    return rows[columns]


def _feature_candidates_for_review(summary: pd.DataFrame, limit: int = TOP_N_REPORT_FEATURES) -> pd.DataFrame:
    columns = [
        "feature_name",
        "candidate_category",
        "recent_signal_assessment",
        "suggested_scorecard_use",
        "top_minus_bottom_avg_forward_return",
        "top_minus_bottom_median_forward_return",
        "tail_effect_assessment",
        "top_minus_bottom_top_30pct_flag_rate",
        "spearman_corr",
        "avg_return_decile_spearman_corr",
        "last_10y_mean_year_spread",
        "last_5y_mean_year_spread",
        "current_year_signal_direction",
        "current_year_top_minus_bottom_avg_forward_return",
        "notes",
    ]
    selected = _select_review_features(summary, limit=limit)
    rows = summary[summary["feature_name"].isin(selected)].copy()
    rows["_order"] = rows["feature_name"].map({feature: idx for idx, feature in enumerate(selected)})
    rows = rows.sort_values("_order").drop(columns=["_order"])
    return rows[columns]


def _recent_regime_rows(summary: pd.DataFrame, limit: int = 12) -> pd.DataFrame:
    columns = [
        "feature_name",
        "recent_signal_assessment",
        "full_history_mean_year_spread",
        "last_10y_mean_year_spread",
        "last_5y_mean_year_spread",
        "full_history_pct_years_positive_spread",
        "last_10y_pct_years_positive_spread",
        "last_5y_pct_years_positive_spread",
        "current_year_signal_direction",
        "suggested_scorecard_use",
    ]
    selected = _select_recent_regime_features(summary, limit=limit)
    rows = summary[summary["feature_name"].isin(selected)].copy()
    rows["_order"] = rows["feature_name"].map({feature: idx for idx, feature in enumerate(selected)})
    rows = rows.sort_values("_order").drop(columns=["_order"])
    return rows[columns]


def _current_year_snapshot_rows(summary: pd.DataFrame, limit: int = 12) -> pd.DataFrame:
    columns = [
        "feature_name",
        "current_partial_year",
        "current_year_signal_direction",
        "current_year_top_minus_bottom_avg_forward_return",
        "current_year_top_minus_bottom_top_30pct_flag_rate",
        "current_year_decile_1_row_count",
        "current_year_decile_10_row_count",
        "candidate_category",
        "tail_effect_assessment",
        "suggested_scorecard_use",
    ]
    available = [column for column in columns if column in summary.columns]
    rows = summary[summary["current_year_signal_direction"] != "insufficient_data"].head(limit)
    return rows[available]


def _recent_signal_rows(summary: pd.DataFrame, limit: int = TOP_N_REPORT_FEATURES) -> pd.DataFrame:
    columns = [
        "feature_name",
        "candidate_category",
        "recent_signal_assessment",
        "full_history_mean_year_spread",
        "last_10y_mean_year_spread",
        "last_5y_mean_year_spread",
        "full_history_pct_years_positive_spread",
        "last_10y_pct_years_positive_spread",
        "last_5y_pct_years_positive_spread",
        "suggested_scorecard_use",
    ]
    return summary[columns].head(limit)


def _recent_assessment_rows(summary: pd.DataFrame, assessment: str, limit: int = TOP_N_REPORT_FEATURES) -> pd.DataFrame:
    columns = [
        "feature_name",
        "candidate_category",
        "recent_signal_assessment",
        "full_history_mean_year_spread",
        "last_10y_mean_year_spread",
        "last_5y_mean_year_spread",
        "full_history_pct_years_positive_spread",
        "last_10y_pct_years_positive_spread",
        "last_5y_pct_years_positive_spread",
        "suggested_scorecard_use",
    ]
    rows = summary[summary["recent_signal_assessment"] == assessment].head(limit)
    return rows[columns]


def _suggested_use_rows(summary: pd.DataFrame, suggested_use: str, limit: int = TOP_N_REPORT_FEATURES) -> pd.DataFrame:
    columns = [
        "feature_name",
        "candidate_category",
        "top_minus_bottom_avg_forward_return",
        "top_minus_bottom_top_30pct_flag_rate",
        "spearman_corr",
        "avg_return_decile_spearman_corr",
        "stability_assessment",
        "notes",
    ]
    rows = summary[summary["suggested_scorecard_use"] == suggested_use].head(limit)
    return rows[columns]


def _decile_curve_sections(summary: pd.DataFrame, decile_rows: pd.DataFrame) -> list[str]:
    if decile_rows.empty:
        return ["_No decile rows were available in the diagnostics input._"]

    selected_features = _select_decile_curve_features(summary)
    if not selected_features:
        return ["_No features selected for decile curve review._"]

    lines: list[str] = []
    for feature_name in selected_features:
        feature_summary = summary[summary["feature_name"] == feature_name].iloc[0]
        feature_deciles = decile_rows[decile_rows["feature_name"] == feature_name].copy()
        if feature_deciles.empty:
            continue
        feature_deciles["decile"] = pd.to_numeric(feature_deciles["decile"], errors="coerce")
        feature_deciles["avg_forward_63d_sector_relative_return"] = pd.to_numeric(
            feature_deciles["avg_forward_63d_sector_relative_return"],
            errors="coerce",
        )
        feature_deciles["median_forward_63d_sector_relative_return"] = pd.to_numeric(
            feature_deciles["median_forward_63d_sector_relative_return"],
            errors="coerce",
        )
        feature_deciles["avg_minus_median_forward_63d_sector_relative_return"] = (
            feature_deciles["avg_forward_63d_sector_relative_return"]
            - feature_deciles["median_forward_63d_sector_relative_return"]
        )
        feature_deciles = feature_deciles.sort_values("decile")
        table = feature_deciles[
            [
                "decile",
                "avg_forward_63d_sector_relative_return",
                "median_forward_63d_sector_relative_return",
                "avg_minus_median_forward_63d_sector_relative_return",
                "top_30pct_sector_flag_rate",
                "row_count",
            ]
        ]
        lines.extend(
            [
                f"### {feature_name}",
                "",
                f"- Category: `{feature_summary['candidate_category']}`",
                f"- Suggested use: `{feature_summary['suggested_scorecard_use']}`",
                f"- Interpretation: {_decile_curve_interpretation(feature_summary)}",
                "",
                _markdown_table(table),
                "",
            ]
        )
    return lines or ["_No decile rows were available for selected features._"]


def _year_detail_sections(summary: pd.DataFrame, year_spreads: pd.DataFrame) -> list[str]:
    if year_spreads.empty:
        return ["_No year_decile rows were available in the diagnostics input._"]
    selected_features = _select_year_detail_features(summary)
    if not selected_features:
        return ["_No features selected for year-by-year detail review._"]

    max_year = _max_calendar_year(year_spreads)
    min_year = max_year - MARKDOWN_YEAR_DETAIL_RECENT_YEARS + 1 if max_year is not None else None
    lines: list[str] = []
    for feature_name in selected_features:
        feature_summary = summary[summary["feature_name"] == feature_name].iloc[0]
        rows = year_spreads[
            (year_spreads["feature_name"] == feature_name)
            & (year_spreads["year_valid_spread_flag"])
        ].copy()
        if min_year is not None:
            rows = rows[pd.to_numeric(rows["calendar_year"], errors="coerce") >= min_year]
        if rows.empty:
            continue
        rows = rows.sort_values("calendar_year")
        table = rows[
            [
                "calendar_year",
                "is_partial_year",
                "included_in_completed_year_stability",
                "year_top_minus_bottom_avg_forward_return",
                "year_top_minus_bottom_top_30pct_flag_rate",
            ]
        ]
        lines.extend(
            [
                f"### {feature_name}",
                "",
                f"- Category: `{feature_summary['candidate_category']}`",
                f"- Recent signal: `{feature_summary['recent_signal_assessment']}`",
                f"- Suggested use: `{feature_summary['suggested_scorecard_use']}`",
                "",
                _markdown_table(table),
                "",
            ]
        )
    return lines or ["_No valid year spreads were available for selected features._"]


def _select_decile_curve_features(summary: pd.DataFrame) -> list[str]:
    return _select_representative_features(summary, limit=MAX_DECILE_TABLES)


def _select_year_detail_features(summary: pd.DataFrame) -> list[str]:
    return _select_representative_features(summary, limit=MAX_YEAR_DETAIL_FEATURES)


def _select_review_features(summary: pd.DataFrame, limit: int) -> list[str]:
    selected = _select_representative_features(summary, limit=limit)
    for feature_name in summary["feature_name"].head(limit).tolist():
        selected = _append_unique_feature(selected, feature_name, limit)
    return selected[:limit]


def _select_recent_regime_features(summary: pd.DataFrame, limit: int) -> list[str]:
    selected: list[str] = []
    for assessment in (
        "persistent_positive",
        "persistent_negative",
        "recent_positive_only",
        "recent_negative_only",
        "mixed_or_regime_dependent",
        "historical_positive_but_recent_weak",
        "historical_negative_but_recent_weak",
    ):
        for feature_name in summary[summary["recent_signal_assessment"] == assessment]["feature_name"].head(3).tolist():
            selected = _append_unique_feature(selected, feature_name, limit)
            if len(selected) >= limit:
                return selected
    for feature_name in summary["feature_name"].head(limit).tolist():
        selected = _append_unique_feature(selected, feature_name, limit)
    return selected[:limit]


def _select_representative_features(summary: pd.DataFrame, limit: int) -> list[str]:
    selected: list[str] = []
    for pattern in ("volatility", "average_dollar_volume", "drawdown", "return_21d|distance_from_50dma|return_63d"):
        matches = summary[summary["feature_name"].astype(str).str.contains(pattern, case=False, na=False)]
        for feature_name in matches["feature_name"].tolist():
            selected = _append_unique_feature(selected, feature_name, limit)
            if _feature_bucket(feature_name) == _feature_bucket(selected[-1]):
                break
        if len(selected) >= limit:
            return selected
    mixed = summary[summary["recent_signal_assessment"] == "mixed_or_regime_dependent"]
    for feature_name in mixed["feature_name"].tolist():
        selected = _append_unique_feature(selected, feature_name, limit)
        if len(selected) >= limit:
            return selected
    for feature_name in summary["feature_name"].tolist():
        selected = _append_unique_feature(selected, feature_name, limit)
        if len(selected) >= limit:
            break
    return selected


def _append_unique_feature(selected: list[str], feature_name: str, limit: int) -> list[str]:
    if len(selected) >= limit:
        return selected
    bucket = _feature_bucket(feature_name)
    existing_buckets = {_feature_bucket(existing) for existing in selected}
    if feature_name not in selected and bucket not in existing_buckets:
        selected.append(feature_name)
    return selected


def _feature_bucket(feature_name: str) -> str:
    name = str(feature_name)
    for suffix in ("_market_pct_rank", "_sector_pct_rank"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _decile_curve_interpretation(feature_summary: pd.Series) -> str:
    feature_name = str(feature_summary["feature_name"])
    if "volatility" in feature_name:
        return "High-volatility deciles show notable spread, but correlation and regime diagnostics should be reviewed before treating volatility as a positive component."
    if "average_dollar_volume" in feature_name:
        return "Liquidity deciles are shown for tradability-filter review rather than as a direct alpha signal."
    if "drawdown" in feature_name:
        return "Drawdown deciles require sign-convention review before assigning a scorecard role."
    return str(feature_summary["notes"])


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_None._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = [_format_markdown_value(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _counts_sentence(counts: dict[str, int]) -> str:
    visible = [(key, value) for key, value in counts.items() if value]
    if not visible:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in visible)


def _executive_interpretation(stats: dict[str, Any]) -> str:
    positive_count = stats["category_counts"].get("bullish_candidate", 0)
    nonlinear_count = stats["category_counts"].get("nonlinear_or_unstable", 0)
    liquidity_count = stats["suggested_use_counts"].get("liquidity_filter_candidate", 0)
    if positive_count == 0:
        return (
            f"The report analyzed {stats['features_analyzed']} price and relative-strength features against the "
            "63-trading-day sector-relative return label. No features currently qualify as simple positive scorecard "
            "components. The strongest patterns are concentrated in volatility, liquidity, and drawdown-style features, "
            f"with {nonlinear_count} nonlinear/unstable features and {liquidity_count} liquidity filter candidates requiring manual review."
        )
    return (
        f"The report analyzed {stats['features_analyzed']} price and relative-strength features against the "
        "63-trading-day sector-relative return label. Some features pass the positive-candidate screen, but the selected "
        "decile curves and recent-window diagnostics should be reviewed before using them in scorecard research."
    )


def _key_findings(summary: pd.DataFrame, stats: dict[str, Any]) -> list[str]:
    findings = []
    if stats["category_counts"].get("bullish_candidate", 0) == 0:
        findings.append("- No clean positive component candidates were identified under the current agreement rules.")
    if _has_feature_like(summary, "volatility"):
        findings.append("- Volatility features show large top-minus-bottom spreads, but are marked nonlinear/unstable and require manual review.")
    if _has_feature_like(summary, "average_dollar_volume"):
        findings.append("- Liquidity features show persistent relationships, but are best reviewed as tradability filters rather than alpha components.")
    if _has_feature_like(summary, "drawdown"):
        findings.append("- Drawdown features show meaningful patterns, but sign convention and scorecard interpretation need review.")
    recent_only = stats["recent_signal_counts"].get("recent_positive_only", 0) + stats["recent_signal_counts"].get("recent_negative_only", 0)
    if recent_only:
        findings.append(f"- {recent_only} features show recent-only behavior, so full-history averages should not be used alone.")
    weak_count = stats["category_counts"].get("weak_or_noisy", 0)
    if weak_count:
        findings.append(f"- {weak_count} features are weak/noisy and are likely v1 ignore candidates unless later research finds a clearer role.")
    return findings or ["- No standout findings were identified; inspect the machine-readable summaries for details."]


def _has_feature_like(summary: pd.DataFrame, pattern: str) -> bool:
    return summary["feature_name"].astype(str).str.contains(pattern, case=False, na=False).any()


def _output_file_guide_rows() -> pd.DataFrame:
    rows = []
    for filename, details in OUTPUT_FILE_PURPOSES.items():
        rows.append(
            {
                "File": filename,
                "Format": details["format"],
                "Purpose": details["purpose"],
                "Canonical?": "Yes" if details["canonical"] else "No",
            }
        )
    return pd.DataFrame(rows)


def _current_year_signal_direction(row: pd.Series) -> str:
    return_spread = row.get("current_year_top_minus_bottom_avg_forward_return")
    flag_lift = row.get("current_year_top_minus_bottom_top_30pct_flag_rate")
    if pd.isna(return_spread) or pd.isna(flag_lift):
        return "insufficient_data"
    return_spread = float(return_spread)
    flag_lift = float(flag_lift)
    if return_spread > 0 and flag_lift > 0:
        return "positive"
    if return_spread < 0 and flag_lift < 0:
        return "negative"
    return "mixed"


def _current_year_sentence(current_snapshot: pd.DataFrame) -> str:
    if current_snapshot.empty or "current_partial_year" not in current_snapshot.columns:
        return "none"
    years = pd.to_numeric(current_snapshot["current_partial_year"], errors="coerce").dropna()
    if years.empty:
        return "none"
    year = int(years.max())
    is_partial = bool(current_snapshot["current_year_is_partial"].fillna(False).any())
    return f"{year} ({'partial' if is_partial else 'complete'})"


def _partial_year_stats(year_spreads: pd.DataFrame) -> dict[str, Any]:
    if year_spreads.empty:
        return {
            "current_partial_year": None,
            "partial_year_included_in_stability": False,
            "latest_completed_stability_year": None,
        }
    max_year = _max_calendar_year(year_spreads)
    included_years = pd.to_numeric(
        year_spreads.loc[
            year_spreads.get("included_in_completed_year_stability", pd.Series(False, index=year_spreads.index)).fillna(
                False
            ),
            "calendar_year",
        ],
        errors="coerce",
    ).dropna()
    partial_rows = year_spreads[
        pd.to_numeric(year_spreads["calendar_year"], errors="coerce") == max_year
    ] if max_year is not None else pd.DataFrame()
    return {
        "current_partial_year": max_year,
        "partial_year_included_in_stability": bool(
            not partial_rows.empty and partial_rows["included_in_completed_year_stability"].fillna(False).all()
        ),
        "latest_completed_stability_year": int(included_years.max()) if not included_years.empty else None,
    }


def _partial_year_metadata(
    year_spreads: pd.DataFrame,
    current_snapshot_path: Path,
    *,
    include_partial_current_year_in_stability: bool,
) -> dict[str, Any]:
    stats = _partial_year_stats(year_spreads)
    return {
        "max_calendar_year_is_treated_as_partial": True,
        "include_partial_current_year_in_stability": include_partial_current_year_in_stability,
        "current_partial_year": stats["current_partial_year"],
        "latest_completed_stability_year": stats["latest_completed_stability_year"],
        "current_year_snapshot_output_path": str(current_snapshot_path),
        "default_behavior": (
            "The max calendar year remains visible in year spreads, Markdown, and the current-year snapshot, "
            "but is excluded from completed-year stability and lookback calculations unless "
            "--include-partial-current-year-in-stability is set."
        ),
    }


def _format_markdown_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


def _top_feature_names(summary: pd.DataFrame, category: str, limit: int = 3) -> list[str]:
    return summary[summary["candidate_category"] == category]["feature_name"].head(limit).tolist()


def _numbered(values: list[str]) -> list[str]:
    if not values:
        return ["None"]
    return [f"{idx}. {value}" for idx, value in enumerate(values, start=1)]


def _top_features_for_metadata(summary: pd.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
    columns = ["summary_rank", "feature_name", "candidate_category", "suggested_scorecard_use", "summary_rank_score"]
    return summary[columns].head(limit).to_dict(orient="records")


def _max_calendar_year(year_spreads: pd.DataFrame) -> int | None:
    if year_spreads.empty or "calendar_year" not in year_spreads.columns:
        return None
    years = pd.to_numeric(year_spreads["calendar_year"], errors="coerce").dropna()
    if years.empty:
        return None
    return int(years.max())


def _lookback_window_metadata(lookback_summary: pd.DataFrame) -> list[dict[str, Any]]:
    if lookback_summary.empty:
        return []
    columns = ["lookback_window", "start_year", "end_year"]
    return (
        lookback_summary[columns]
        .drop_duplicates()
        .sort_values(["end_year", "start_year", "lookback_window"])
        .to_dict(orient="records")
    )


def _package_version() -> str:
    try:
        from finbot_research import __version__
    except Exception:
        return "unknown"
    return __version__
