from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from finbot_research.bucket_signal_diagnostics import (
    BOTTOM_CLASSIFICATION_LABEL,
    BUCKET_DEFINITIONS,
    DOLLAR_VOLUME_INTERPRETATION,
    DRAWDOWN_INTERPRETATION_NOTE,
    assign_feature_buckets,
)
from finbot_research.config import (
    labels_path,
    price_features_path,
    price_strength_scorecard_csv_path,
    price_strength_scorecard_path,
    price_strength_scorecard_report_path,
    price_strength_scorecard_summary_csv_path,
    price_strength_scorecard_summary_path,
    price_strength_scorecard_stability_csv_path,
    price_strength_scorecard_stability_path,
    price_strength_scorecard_years_csv_path,
    price_strength_scorecard_years_path,
    relative_features_path,
)
from finbot_research.io import parquet_columns, read_parquet, write_csv, write_parquet
from finbot_research.metadata import write_metadata
from finbot_research.schemas import PRIMARY_CLASSIFICATION_LABEL, PRIMARY_REGRESSION_LABEL
from finbot_research.validation import ValidationError, validate_joinable_dataset

DATASET_NAME = "equity_price_strength_scorecard_v0"
SCORECARD_VERSION = "v0"

REQUIRED_FEATURES = [
    "volatility_63d",
    "return_63d_sector_pct_rank",
    "drawdown_from_52w_high_sector_pct_rank",
    "average_dollar_volume_63d",
]

SCORECARD_RULES = {
    "price_strength_candidate": "is_high_volatility and is_sector_relative_near_52w_high",
    "higher_conviction_price_strength_candidate": "is_high_volatility and is_sector_relative_near_52w_high and is_strong_momentum",
    "momentum_resilience_candidate": "is_sector_relative_near_52w_high and is_strong_momentum",
    "high_volatility_weak_momentum_trap": "is_high_volatility and is_weak_momentum",
    "high_volatility_deep_drawdown_trap": "is_high_volatility and is_sector_relative_deep_drawdown",
}

BUCKET_PRIORITY_RULES = [
    ("higher_conviction_price_strength", 3),
    ("price_strength_candidate", 2),
    ("momentum_resilience_candidate", 1),
    ("high_volatility_trap", -1),
    ("neutral", 0),
]

SCORE_VALUES = {bucket: score for bucket, score in BUCKET_PRIORITY_RULES}

OUTPUT_FILE_PURPOSES = {
    "equity_price_strength_scorecard_v0.parquet": {
        "format": "parquet",
        "purpose": "Canonical row-level research scorecard prototype by symbol/date.",
        "canonical": True,
    },
    "equity_price_strength_scorecard_v0.csv": {
        "format": "csv",
        "purpose": "Convenience export for manual inspection of row-level scorecard output.",
        "canonical": False,
    },
    "equity_price_strength_scorecard_v0_summary.parquet": {
        "format": "parquet",
        "purpose": "Canonical scorecard bucket summary by time window.",
        "canonical": True,
    },
    "equity_price_strength_scorecard_v0_summary.csv": {
        "format": "csv",
        "purpose": "Convenience export for manual inspection of scorecard summaries.",
        "canonical": False,
    },
    "equity_price_strength_scorecard_v0_years.parquet": {
        "format": "parquet",
        "purpose": "Canonical year-by-year scorecard bucket diagnostics with same-year baseline comparisons.",
        "canonical": True,
    },
    "equity_price_strength_scorecard_v0_years.csv": {
        "format": "csv",
        "purpose": "Convenience export for manual inspection of year-by-year scorecard bucket diagnostics.",
        "canonical": False,
    },
    "equity_price_strength_scorecard_v0_stability.parquet": {
        "format": "parquet",
        "purpose": "Canonical completed-year scorecard bucket stability summary.",
        "canonical": True,
    },
    "equity_price_strength_scorecard_v0_stability.csv": {
        "format": "csv",
        "purpose": "Convenience export for manual inspection of scorecard bucket stability.",
        "canonical": False,
    },
    "equity_price_strength_scorecard_v0_report.md": {
        "format": "markdown",
        "purpose": "Human-readable research summary for the price-strength scorecard prototype.",
        "canonical": True,
    },
    "equity_price_strength_scorecard_v0.metadata.json": {
        "format": "json",
        "purpose": "Documents inputs, outputs, rules, buckets, and generation metadata.",
        "canonical": True,
    },
}

SCORECARD_COLUMNS = [
    "symbol",
    "date",
    "volatility_63d",
    "return_63d_sector_pct_rank",
    "drawdown_from_52w_high_sector_pct_rank",
    "average_dollar_volume_63d",
    "volatility_63d_bucket",
    "momentum_63d_sector_bucket",
    "drawdown_52w_sector_bucket",
    "dollar_volume_63d_bucket",
    "is_high_volatility",
    "is_medium_volatility",
    "is_low_volatility",
    "is_strong_momentum",
    "is_middle_momentum",
    "is_weak_momentum",
    "is_sector_relative_near_52w_high",
    "is_sector_relative_mid_drawdown",
    "is_sector_relative_deep_drawdown",
    "is_lower_dollar_volume",
    "is_middle_dollar_volume",
    "is_highest_dollar_volume",
    "price_strength_candidate",
    "higher_conviction_price_strength_candidate",
    "momentum_resilience_candidate",
    "high_volatility_weak_momentum_trap",
    "high_volatility_deep_drawdown_trap",
    "is_scorecard_bucket_eligible",
    "price_strength_scorecard_bucket",
    "price_strength_score_v0",
    PRIMARY_REGRESSION_LABEL,
    PRIMARY_CLASSIFICATION_LABEL,
    BOTTOM_CLASSIFICATION_LABEL,
]

SUMMARY_COLUMNS = [
    "price_strength_scorecard_bucket",
    "time_window",
    "start_year",
    "end_year",
    "is_current_partial_year_window",
    "row_count",
    "symbol_count",
    "avg_forward_63d_sector_relative_return",
    "median_forward_63d_sector_relative_return",
    "avg_minus_median_forward_63d_sector_relative_return",
    "top_30pct_sector_flag_rate",
    "bottom_30pct_sector_flag_rate",
    "avg_forward_return_vs_baseline",
    "median_forward_return_vs_baseline",
    "top_30pct_flag_rate_vs_baseline",
    "bottom_30pct_flag_rate_vs_baseline",
]

YEAR_COLUMNS = [
    "price_strength_scorecard_bucket",
    "calendar_year",
    "is_partial_year",
    "included_in_completed_year_stability",
    "row_count",
    "symbol_count",
    "avg_forward_63d_sector_relative_return",
    "median_forward_63d_sector_relative_return",
    "top_30pct_sector_flag_rate",
    "bottom_30pct_sector_flag_rate",
    "avg_forward_return_vs_baseline",
    "median_forward_return_vs_baseline",
    "top_30pct_flag_rate_vs_baseline",
    "bottom_30pct_flag_rate_vs_baseline",
]

STABILITY_COLUMNS = [
    "price_strength_scorecard_bucket",
    "completed_years_count",
    "years_avg_above_baseline",
    "years_median_above_baseline",
    "years_top_rate_above_baseline",
    "years_bottom_rate_below_baseline",
    "pct_years_avg_above_baseline",
    "pct_years_median_above_baseline",
    "pct_years_top_rate_above_baseline",
    "pct_years_bottom_rate_below_baseline",
    "mean_avg_return_vs_baseline",
    "mean_median_return_vs_baseline",
    "mean_top_rate_vs_baseline",
    "mean_bottom_rate_vs_baseline",
    "stability_assessment",
]

STABILITY_ASSESSMENT_LABELS = [
    "broadly_positive",
    "positive_but_high_risk",
    "tail_driven",
    "neutral_or_defensive",
    "negative_or_trap",
    "mixed_or_regime_dependent",
    "insufficient_data",
]


def build_price_strength_scorecard(data_root: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    price_path = price_features_path(data_root)
    relative_path = relative_features_path(data_root)
    label_path = labels_path(data_root)
    output_path = price_strength_scorecard_path(data_root)
    output_csv_path = price_strength_scorecard_csv_path(data_root)
    summary_path = price_strength_scorecard_summary_path(data_root)
    summary_csv_path = price_strength_scorecard_summary_csv_path(data_root)
    years_path = price_strength_scorecard_years_path(data_root)
    years_csv_path = price_strength_scorecard_years_csv_path(data_root)
    stability_path = price_strength_scorecard_stability_path(data_root)
    stability_csv_path = price_strength_scorecard_stability_csv_path(data_root)
    report_path = price_strength_scorecard_report_path(data_root)

    price_columns = _existing_columns(price_path, ["symbol", "date", *REQUIRED_FEATURES])
    relative_columns = _existing_columns(relative_path, ["symbol", "date", *REQUIRED_FEATURES])
    label_columns = _existing_columns(
        label_path,
        ["symbol", "date", PRIMARY_REGRESSION_LABEL, PRIMARY_CLASSIFICATION_LABEL, BOTTOM_CLASSIFICATION_LABEL],
    )
    price_features = read_parquet(price_path, columns=price_columns)
    relative_features = read_parquet(relative_path, columns=relative_columns)
    labels = read_parquet(label_path, columns=label_columns)
    scorecard, summary, build_summary = compute_price_strength_scorecard(price_features, relative_features, labels)
    bottom_label_available = build_summary["bottom_label_available"]
    years = compute_scorecard_years(scorecard, bottom_label_available=bottom_label_available)
    stability = compute_scorecard_stability(years, bottom_label_available=bottom_label_available)
    build_summary["stability_assessment_counts"] = {
        str(key): int(value) for key, value in stability["stability_assessment"].value_counts(dropna=False).to_dict().items()
    }

    write_parquet(scorecard, output_path)
    write_csv(scorecard, output_csv_path)
    write_parquet(summary, summary_path)
    write_csv(summary, summary_csv_path)
    write_parquet(years, years_path)
    write_csv(years, years_csv_path)
    write_parquet(stability, stability_path)
    write_csv(stability, stability_csv_path)
    write_markdown_report(summary, report_path, build_summary=build_summary, years=years, stability=stability)
    metadata_path = write_metadata(
        dataset_name=DATASET_NAME,
        output_path=output_path,
        input_paths=[price_path, relative_path, label_path],
        dataframe=scorecard,
        extra_metadata={
            "dataset_type": "research_scorecard_prototype",
            "research_only": True,
            "scorecard_version": SCORECARD_VERSION,
            "output_paths": {
                "scorecard_parquet": str(output_path),
                "scorecard_csv": str(output_csv_path),
                "summary_parquet": str(summary_path),
                "summary_csv": str(summary_csv_path),
                "years_parquet": str(years_path),
                "years_csv": str(years_csv_path),
                "stability_parquet": str(stability_path),
                "stability_csv": str(stability_csv_path),
                "markdown_report": str(report_path),
                "metadata": str(metadata_path_placeholder(output_path)),
            },
            "output_file_purposes": OUTPUT_FILE_PURPOSES,
            "bucket_definitions": BUCKET_DEFINITIONS,
            "scorecard_rules": SCORECARD_RULES,
            "score_values": SCORE_VALUES,
            "primary_regression_label": PRIMARY_REGRESSION_LABEL,
            "primary_classification_label": PRIMARY_CLASSIFICATION_LABEL,
            "bottom_classification_label": BOTTOM_CLASSIFICATION_LABEL if build_summary["bottom_label_available"] else None,
            "scorecard_bucket_eligibility": (
                "Row-level outputs retain all joined rows and include is_scorecard_bucket_eligible. "
                "Summary, yearly, and stability diagnostics only include rows with all required bucket inputs present."
            ),
            "baseline_definition": "The full eligible joined feature/label universe for the same calendar year or summary time window.",
            "partial_year_handling": build_summary["partial_year_handling"],
            "completed_year_stability_logic": (
                "Year-by-year diagnostics include the max calendar year as partial, but completed-year stability "
                "excludes that partial year."
            ),
            "stability_assessment_labels": STABILITY_ASSESSMENT_LABELS,
            "dollar_volume_interpretation": DOLLAR_VOLUME_INTERPRETATION,
            "drawdown_interpretation_note": DRAWDOWN_INTERPRETATION_NOTE,
            "parquet_is_canonical": True,
            "csv_is_convenience_export": True,
        },
    )
    return (
        {
            "scorecard_parquet": output_path,
            "scorecard_csv": output_csv_path,
            "summary_parquet": summary_path,
            "summary_csv": summary_csv_path,
            "years_parquet": years_path,
            "years_csv": years_csv_path,
            "stability_parquet": stability_path,
            "stability_csv": stability_csv_path,
            "markdown_report": report_path,
            "metadata": metadata_path,
        },
        build_summary,
    )


def metadata_path_placeholder(output_path: Path) -> Path:
    return output_path.with_suffix(".metadata.json")


def _existing_columns(path: Path, requested_columns: list[str]) -> list[str]:
    available = set(parquet_columns(path))
    return [column for column in requested_columns if column in available]


def compute_price_strength_scorecard(
    price_features: pd.DataFrame,
    relative_features: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    validate_joinable_dataset(price_features, "price_features")
    validate_joinable_dataset(relative_features, "relative_features")
    validate_joinable_dataset(labels, "labels")
    if PRIMARY_REGRESSION_LABEL not in labels.columns or PRIMARY_CLASSIFICATION_LABEL not in labels.columns:
        raise ValidationError("labels missing required forward-return columns")

    features = _merge_feature_inputs(price_features, relative_features)
    missing_features = [feature for feature in REQUIRED_FEATURES if feature not in features.columns]
    if missing_features:
        raise ValidationError(f"price-strength-scorecard-v0 missing required source features: {missing_features}")

    labels = _normalize_symbol_date(labels)
    bottom_label_available = BOTTOM_CLASSIFICATION_LABEL in labels.columns
    label_columns = ["symbol", "date", PRIMARY_REGRESSION_LABEL, PRIMARY_CLASSIFICATION_LABEL]
    if bottom_label_available:
        label_columns.append(BOTTOM_CLASSIFICATION_LABEL)
    joined = features[["symbol", "date", *REQUIRED_FEATURES]].merge(labels[label_columns], on=["symbol", "date"], how="inner")
    if joined.empty:
        raise ValidationError("Joined feature/label dataset is empty")
    if not bottom_label_available:
        joined[BOTTOM_CLASSIFICATION_LABEL] = pd.NA

    bucketed, _bucket_summary = assign_feature_buckets(joined)
    scorecard = add_scorecard_flags(bucketed)
    columns = SCORECARD_COLUMNS if bottom_label_available else [column for column in SCORECARD_COLUMNS if column != BOTTOM_CLASSIFICATION_LABEL]
    scorecard = scorecard[columns].sort_values(["symbol", "date"]).reset_index(drop=True)
    summary = compute_scorecard_summary(scorecard, bottom_label_available=bottom_label_available)
    build_summary = {
        "rows_analyzed": int(len(scorecard)),
        "scorecard_bucket_eligible_rows": int(scorecard["is_scorecard_bucket_eligible"].sum()),
        "scorecard_bucket_ineligible_rows": int((~scorecard["is_scorecard_bucket_eligible"]).sum()),
        "symbols_analyzed": int(scorecard["symbol"].nunique()),
        "bottom_label_available": bottom_label_available,
        "time_windows_analyzed": summary["time_window"].drop_duplicates().tolist(),
        "current_partial_year": _max_calendar_year(scorecard),
        "latest_completed_stability_year": _latest_completed_year(scorecard),
        "partial_year_handling": _partial_year_metadata(scorecard),
        "scorecard_bucket_counts": {
            str(key): int(value)
            for key, value in scorecard["price_strength_scorecard_bucket"].value_counts(dropna=False).to_dict().items()
        },
    }
    return scorecard, summary, build_summary


def _merge_feature_inputs(price_features: pd.DataFrame, relative_features: pd.DataFrame) -> pd.DataFrame:
    price = _normalize_symbol_date(price_features)
    relative = _normalize_symbol_date(relative_features)
    merged = price.merge(relative, on=["symbol", "date"], how="outer", suffixes=("_price", "_relative"))
    for feature in REQUIRED_FEATURES:
        price_column = f"{feature}_price"
        relative_column = f"{feature}_relative"
        if feature in merged.columns:
            continue
        if price_column in merged.columns and relative_column in merged.columns:
            merged[feature] = merged[price_column].combine_first(merged[relative_column])
            merged = merged.drop(columns=[price_column, relative_column])
        elif price_column in merged.columns:
            merged = merged.rename(columns={price_column: feature})
        elif relative_column in merged.columns:
            merged = merged.rename(columns={relative_column: feature})
    return merged


def _normalize_symbol_date(dataframe: pd.DataFrame) -> pd.DataFrame:
    normalized = dataframe.copy()
    normalized["symbol"] = normalized["symbol"].astype("string").str.upper()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="raise").dt.date
    return normalized.sort_values(["symbol", "date"]).reset_index(drop=True)


def add_scorecard_flags(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["is_high_volatility"] = result["volatility_63d_bucket"] == "high_volatility"
    result["is_medium_volatility"] = result["volatility_63d_bucket"] == "medium_volatility"
    result["is_low_volatility"] = result["volatility_63d_bucket"] == "low_volatility"
    result["is_strong_momentum"] = result["momentum_63d_sector_bucket"] == "strong_momentum"
    result["is_middle_momentum"] = result["momentum_63d_sector_bucket"] == "middle_momentum"
    result["is_weak_momentum"] = result["momentum_63d_sector_bucket"] == "weak_momentum"
    result["is_sector_relative_near_52w_high"] = result["drawdown_52w_sector_bucket"] == "sector_relative_near_52w_high"
    result["is_sector_relative_mid_drawdown"] = result["drawdown_52w_sector_bucket"] == "sector_relative_mid_drawdown"
    result["is_sector_relative_deep_drawdown"] = result["drawdown_52w_sector_bucket"] == "sector_relative_deep_drawdown"
    result["is_lower_dollar_volume"] = result["dollar_volume_63d_bucket"] == "lower_dollar_volume"
    result["is_middle_dollar_volume"] = result["dollar_volume_63d_bucket"] == "middle_dollar_volume"
    result["is_highest_dollar_volume"] = result["dollar_volume_63d_bucket"] == "highest_dollar_volume"

    result["price_strength_candidate"] = result["is_high_volatility"] & result["is_sector_relative_near_52w_high"]
    result["higher_conviction_price_strength_candidate"] = (
        result["is_high_volatility"] & result["is_sector_relative_near_52w_high"] & result["is_strong_momentum"]
    )
    result["momentum_resilience_candidate"] = result["is_sector_relative_near_52w_high"] & result["is_strong_momentum"]
    result["high_volatility_weak_momentum_trap"] = result["is_high_volatility"] & result["is_weak_momentum"]
    result["high_volatility_deep_drawdown_trap"] = result["is_high_volatility"] & result["is_sector_relative_deep_drawdown"]
    result["is_scorecard_bucket_eligible"] = result[
        ["volatility_63d_bucket", "momentum_63d_sector_bucket", "drawdown_52w_sector_bucket", "dollar_volume_63d_bucket"]
    ].notna().all(axis=1)

    result["price_strength_scorecard_bucket"] = "neutral"
    result.loc[
        result["high_volatility_weak_momentum_trap"] | result["high_volatility_deep_drawdown_trap"],
        "price_strength_scorecard_bucket",
    ] = "high_volatility_trap"
    result.loc[result["momentum_resilience_candidate"], "price_strength_scorecard_bucket"] = "momentum_resilience_candidate"
    result.loc[result["price_strength_candidate"], "price_strength_scorecard_bucket"] = "price_strength_candidate"
    result.loc[
        result["higher_conviction_price_strength_candidate"],
        "price_strength_scorecard_bucket",
    ] = "higher_conviction_price_strength"
    result["price_strength_score_v0"] = result["price_strength_scorecard_bucket"].map(SCORE_VALUES).astype("Int64")
    return result


def compute_scorecard_summary(scorecard: pd.DataFrame, *, bottom_label_available: bool) -> pd.DataFrame:
    scorecard = _eligible_scorecard_rows(scorecard)
    windows = _time_windows(scorecard)
    rows: list[dict[str, Any]] = []
    for window in windows:
        window_rows = _window_rows(scorecard, window)
        baseline = _metric_fields(window_rows, bottom_label_available=bottom_label_available)
        for bucket, group in window_rows.groupby("price_strength_scorecard_bucket", sort=True, dropna=False):
            metrics = _with_baseline_differences(_metric_fields(group, bottom_label_available=bottom_label_available), baseline)
            rows.append({"price_strength_scorecard_bucket": str(bucket), **window, **metrics})
    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    if not bottom_label_available and "bottom_30pct_sector_flag_rate" in summary.columns:
        summary = summary.drop(columns=["bottom_30pct_sector_flag_rate", "bottom_30pct_flag_rate_vs_baseline"])
    return summary


def compute_scorecard_years(scorecard: pd.DataFrame, *, bottom_label_available: bool) -> pd.DataFrame:
    scorecard = _eligible_scorecard_rows(scorecard)
    max_year = _max_calendar_year(scorecard)
    rows: list[dict[str, Any]] = []
    frame = scorecard.copy()
    frame["calendar_year"] = pd.to_datetime(frame["date"]).dt.year
    for calendar_year, year_rows in frame.groupby("calendar_year", sort=True):
        baseline = _metric_fields(year_rows, bottom_label_available=bottom_label_available)
        is_partial_year = calendar_year == max_year
        for bucket, group in year_rows.groupby("price_strength_scorecard_bucket", sort=True, dropna=False):
            metrics = _with_baseline_differences(_metric_fields(group, bottom_label_available=bottom_label_available), baseline)
            rows.append(
                {
                    "price_strength_scorecard_bucket": str(bucket),
                    "calendar_year": int(calendar_year),
                    "is_partial_year": bool(is_partial_year),
                    "included_in_completed_year_stability": bool(not is_partial_year),
                    **metrics,
                }
            )
    years = pd.DataFrame(rows, columns=YEAR_COLUMNS)
    if not bottom_label_available and "bottom_30pct_sector_flag_rate" in years.columns:
        years = years.drop(columns=["bottom_30pct_sector_flag_rate", "bottom_30pct_flag_rate_vs_baseline"])
    return years


def compute_scorecard_stability(years: pd.DataFrame, *, bottom_label_available: bool) -> pd.DataFrame:
    completed = years[years["included_in_completed_year_stability"]].copy()
    rows: list[dict[str, Any]] = []
    for bucket in SCORE_VALUES:
        group = completed[completed["price_strength_scorecard_bucket"] == bucket]
        completed_years_count = int(group["calendar_year"].nunique())
        avg_count = _positive_count(group.get("avg_forward_return_vs_baseline"))
        median_count = _positive_count(group.get("median_forward_return_vs_baseline"))
        top_count = _positive_count(group.get("top_30pct_flag_rate_vs_baseline"))
        bottom_below_count = _negative_count(group.get("bottom_30pct_flag_rate_vs_baseline")) if bottom_label_available else None
        row = {
            "price_strength_scorecard_bucket": bucket,
            "completed_years_count": completed_years_count,
            "years_avg_above_baseline": avg_count,
            "years_median_above_baseline": median_count,
            "years_top_rate_above_baseline": top_count,
            "years_bottom_rate_below_baseline": bottom_below_count,
            "pct_years_avg_above_baseline": _share(avg_count, completed_years_count),
            "pct_years_median_above_baseline": _share(median_count, completed_years_count),
            "pct_years_top_rate_above_baseline": _share(top_count, completed_years_count),
            "pct_years_bottom_rate_below_baseline": _share(bottom_below_count, completed_years_count)
            if bottom_label_available
            else None,
            "mean_avg_return_vs_baseline": _mean_or_none(group.get("avg_forward_return_vs_baseline")),
            "mean_median_return_vs_baseline": _mean_or_none(group.get("median_forward_return_vs_baseline")),
            "mean_top_rate_vs_baseline": _mean_or_none(group.get("top_30pct_flag_rate_vs_baseline")),
            "mean_bottom_rate_vs_baseline": _mean_or_none(group.get("bottom_30pct_flag_rate_vs_baseline"))
            if bottom_label_available
            else None,
        }
        row["stability_assessment"] = assess_stability(row, bottom_label_available=bottom_label_available)
        rows.append(row)
    stability = pd.DataFrame(rows, columns=STABILITY_COLUMNS)
    if not bottom_label_available:
        stability = stability.drop(
            columns=[
                "years_bottom_rate_below_baseline",
                "pct_years_bottom_rate_below_baseline",
                "mean_bottom_rate_vs_baseline",
            ]
        )
    return stability


def _eligible_scorecard_rows(scorecard: pd.DataFrame) -> pd.DataFrame:
    if "is_scorecard_bucket_eligible" not in scorecard.columns:
        return scorecard
    return scorecard[scorecard["is_scorecard_bucket_eligible"].fillna(False)].copy()


def assess_stability(row: dict[str, Any] | pd.Series, *, bottom_label_available: bool) -> str:
    completed_years_count = int(row.get("completed_years_count") or 0)
    if completed_years_count < 3:
        return "insufficient_data"

    pct_avg = float(row.get("pct_years_avg_above_baseline") or 0)
    pct_median = float(row.get("pct_years_median_above_baseline") or 0)
    pct_top = float(row.get("pct_years_top_rate_above_baseline") or 0)
    pct_bottom_better = float(row.get("pct_years_bottom_rate_below_baseline") or 0) if bottom_label_available else 0.5
    mean_avg = float(row.get("mean_avg_return_vs_baseline") or 0)
    mean_median = float(row.get("mean_median_return_vs_baseline") or 0)
    mean_top = float(row.get("mean_top_rate_vs_baseline") or 0)
    mean_bottom = float(row.get("mean_bottom_rate_vs_baseline") or 0) if bottom_label_available else 0
    bottom_often_worse = bottom_label_available and pct_bottom_better < 0.40
    bottom_mean_worse = bottom_label_available and mean_bottom > 0
    bottom_not_meaningfully_worse = (not bottom_label_available) or (pct_bottom_better >= 0.40 and mean_bottom <= 0.02)

    if pct_median >= 0.60 and pct_top >= 0.60 and mean_median > 0 and mean_top > 0 and bottom_not_meaningfully_worse:
        return "broadly_positive"
    if pct_avg >= 0.60 and pct_top >= 0.60 and mean_avg > 0 and mean_top > 0 and (bottom_often_worse or bottom_mean_worse):
        if pct_median < 0.50 or mean_median <= 0:
            return "tail_driven"
        return "positive_but_high_risk"
    if pct_median <= 0.40 and mean_median < 0 and (bottom_often_worse or bottom_mean_worse):
        return "negative_or_trap"
    if pct_top <= 0.40 and pct_bottom_better >= 0.60 and mean_bottom < 0:
        return "neutral_or_defensive"
    return "mixed_or_regime_dependent"


def _time_windows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    years = pd.to_datetime(frame["date"]).dt.year
    max_year = int(years.max())
    completed = years[years < max_year]
    windows: list[dict[str, Any]] = []
    if not completed.empty:
        min_completed_year = int(completed.min())
        max_completed_year = int(completed.max())
        windows.append(_window("full_completed_history", min_completed_year, max_completed_year, False))
        windows.append(_window("last_10y_completed", max(max_completed_year - 9, min_completed_year), max_completed_year, False))
        windows.append(_window("last_5y_completed", max(max_completed_year - 4, min_completed_year), max_completed_year, False))
    windows.append(_window("current_partial_year", max_year, max_year, True))
    return windows


def _window(time_window: str, start_year: int, end_year: int, is_current_partial_year_window: bool) -> dict[str, Any]:
    return {
        "time_window": time_window,
        "start_year": int(start_year),
        "end_year": int(end_year),
        "is_current_partial_year_window": bool(is_current_partial_year_window),
    }


def _window_rows(frame: pd.DataFrame, window: dict[str, Any]) -> pd.DataFrame:
    years = pd.to_datetime(frame["date"]).dt.year
    return frame[(years >= window["start_year"]) & (years <= window["end_year"])].dropna(
        subset=[PRIMARY_REGRESSION_LABEL, PRIMARY_CLASSIFICATION_LABEL]
    )


def _metric_fields(group: pd.DataFrame, *, bottom_label_available: bool) -> dict[str, Any]:
    if group.empty:
        row = {
            "row_count": 0,
            "symbol_count": 0,
            "avg_forward_63d_sector_relative_return": None,
            "median_forward_63d_sector_relative_return": None,
            "avg_minus_median_forward_63d_sector_relative_return": None,
            "top_30pct_sector_flag_rate": None,
        }
        if bottom_label_available:
            row["bottom_30pct_sector_flag_rate"] = None
        return row
    returns = pd.to_numeric(group[PRIMARY_REGRESSION_LABEL], errors="coerce")
    top_flags = pd.to_numeric(group[PRIMARY_CLASSIFICATION_LABEL], errors="coerce")
    avg_return = _finite_or_none(returns.mean())
    median_return = _finite_or_none(returns.median())
    row = {
        "row_count": int(len(group)),
        "symbol_count": int(group["symbol"].nunique()),
        "avg_forward_63d_sector_relative_return": avg_return,
        "median_forward_63d_sector_relative_return": median_return,
        "avg_minus_median_forward_63d_sector_relative_return": _difference(avg_return, median_return),
        "top_30pct_sector_flag_rate": _finite_or_none(top_flags.mean()),
    }
    if bottom_label_available:
        bottom_flags = pd.to_numeric(group[BOTTOM_CLASSIFICATION_LABEL], errors="coerce")
        row["bottom_30pct_sector_flag_rate"] = _finite_or_none(bottom_flags.mean())
    return row


def _with_baseline_differences(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    result["avg_forward_return_vs_baseline"] = _difference(
        metrics.get("avg_forward_63d_sector_relative_return"),
        baseline.get("avg_forward_63d_sector_relative_return"),
    )
    result["median_forward_return_vs_baseline"] = _difference(
        metrics.get("median_forward_63d_sector_relative_return"),
        baseline.get("median_forward_63d_sector_relative_return"),
    )
    result["top_30pct_flag_rate_vs_baseline"] = _difference(
        metrics.get("top_30pct_sector_flag_rate"),
        baseline.get("top_30pct_sector_flag_rate"),
    )
    result["bottom_30pct_flag_rate_vs_baseline"] = _difference(
        metrics.get("bottom_30pct_sector_flag_rate"),
        baseline.get("bottom_30pct_sector_flag_rate"),
    )
    return result


def write_markdown_report(
    summary: pd.DataFrame,
    path: Path,
    *,
    build_summary: dict[str, Any],
    years: pd.DataFrame | None = None,
    stability: pd.DataFrame | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    bottom_available = build_summary["bottom_label_available"]
    years = years if years is not None else pd.DataFrame()
    stability = stability if stability is not None else pd.DataFrame()
    lines = [
        "# Equity Price Strength Scorecard v0",
        "",
        "## Purpose",
        "",
        "This is a research-only prototype that classifies each symbol/date into interpretable price-strength and risk buckets. It is not a production signal, model, backtest, portfolio tool, or trading recommendation.",
        "",
        "## Scorecard Logic",
        "",
        "- Primary hypothesis: high volatility plus sector-relative price resilience near 52-week highs.",
        "- Strong momentum is a confirmation boost, especially in the current partial year.",
        "- High volatility plus weak momentum and high volatility plus sector-relative deep drawdown are risk/trap buckets.",
        f"- {DOLLAR_VOLUME_INTERPRETATION}",
        "",
        "## Executive Summary",
        "",
        f"- Rows analyzed: {build_summary['rows_analyzed']}",
        f"- Scorecard-bucket eligible rows: {build_summary['scorecard_bucket_eligible_rows']}",
        f"- Scorecard-bucket ineligible rows: {build_summary['scorecard_bucket_ineligible_rows']}",
        f"- Symbols analyzed: {build_summary['symbols_analyzed']}",
        f"- Current partial year: {build_summary['current_partial_year']}",
        f"- Scorecard bucket counts: {_counts_sentence(build_summary['scorecard_bucket_counts'])}",
        "",
        "The strongest historical rule from bucket diagnostics is high volatility combined with sector-relative near-52-week-high resilience. Strong momentum acts as a confirmation boost, while weak momentum and deep drawdown variants remain risk/trap research buckets.",
        "",
        "## Scorecard Bucket Summary",
        "",
        _markdown_table(_summary_rows(summary, bottom_available=bottom_available)),
        "",
        "## Current Partial Year Snapshot",
        "",
        _markdown_table(_summary_rows(summary[summary["time_window"] == "current_partial_year"], bottom_available=bottom_available)),
        "",
        "## Completed-Year Evidence",
        "",
        _markdown_table(
            _summary_rows(
                summary[summary["time_window"].isin(["full_completed_history", "last_10y_completed", "last_5y_completed"])],
                bottom_available=bottom_available,
            )
        ),
        "",
        "## Scorecard Bucket Stability",
        "",
        _markdown_table(_stability_rows(stability, bottom_available=bottom_available)),
        "",
        _markdown_table(_year_detail_rows(years, bottom_available=bottom_available)),
        "",
        "Interpretation: `higher_conviction_price_strength` is the main positive candidate when year-by-year evidence supports it. `price_strength_candidate` is the weaker/lower-conviction positive bucket. `high_volatility_trap` should be treated as a risk bucket if median return underperforms and bottom-rate is worse than baseline.",
        "",
        "## Important Caveats",
        "",
        "- This is research-only and does not produce production signals.",
        "- This is not a backtest and does not include portfolio construction.",
        "- Score values are simple research ranking labels, not trading recommendations.",
        "- Summary, yearly, and stability diagnostics exclude rows with missing required bucket inputs; those rows remain visible in the row-level output with `is_scorecard_bucket_eligible=false`.",
        "- Current partial-year behavior is shown separately because it is incomplete.",
        f"- {DRAWDOWN_INTERPRETATION_NOTE}",
        "",
        "## Output File Guide",
        "",
        _markdown_table(_output_file_guide_rows()),
        "",
        "## Suggested Next Step",
        "",
        "Compare scorecard bucket stability against the candidate-rule diagnostics before deciding whether any component deserves formal scorecard research.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _summary_rows(summary: pd.DataFrame, *, bottom_available: bool, limit: int = 24) -> pd.DataFrame:
    columns = [
        "time_window",
        "price_strength_scorecard_bucket",
        "row_count",
        "symbol_count",
        "avg_forward_63d_sector_relative_return",
        "median_forward_63d_sector_relative_return",
        "avg_forward_return_vs_baseline",
        "median_forward_return_vs_baseline",
        "top_30pct_sector_flag_rate",
        "top_30pct_flag_rate_vs_baseline",
    ]
    if bottom_available and "bottom_30pct_sector_flag_rate" in summary.columns:
        columns.extend(["bottom_30pct_sector_flag_rate", "bottom_30pct_flag_rate_vs_baseline"])
    if summary.empty:
        return pd.DataFrame(columns=columns)
    rows = summary.copy()
    rows["_window_order"] = rows["time_window"].map(
        {"current_partial_year": 0, "last_5y_completed": 1, "last_10y_completed": 2, "full_completed_history": 3}
    ).fillna(99)
    rows["_score"] = rows["price_strength_scorecard_bucket"].map(SCORE_VALUES).fillna(0)
    return rows.sort_values(["_window_order", "_score"], ascending=[True, False]).head(limit)[columns]


def _stability_rows(stability: pd.DataFrame, *, bottom_available: bool) -> pd.DataFrame:
    columns = [
        "price_strength_scorecard_bucket",
        "completed_years_count",
        "pct_years_avg_above_baseline",
        "pct_years_median_above_baseline",
        "pct_years_top_rate_above_baseline",
        "mean_avg_return_vs_baseline",
        "mean_median_return_vs_baseline",
        "mean_top_rate_vs_baseline",
        "stability_assessment",
    ]
    if bottom_available and "pct_years_bottom_rate_below_baseline" in stability.columns:
        columns.insert(4, "pct_years_bottom_rate_below_baseline")
        columns.insert(-1, "mean_bottom_rate_vs_baseline")
    if stability.empty:
        return pd.DataFrame(columns=columns)
    rows = stability.copy()
    rows["_score"] = rows["price_strength_scorecard_bucket"].map(SCORE_VALUES).fillna(0)
    return rows.sort_values("_score", ascending=False)[columns]


def _year_detail_rows(years: pd.DataFrame, *, bottom_available: bool) -> pd.DataFrame:
    columns = [
        "calendar_year",
        "is_partial_year",
        "price_strength_scorecard_bucket",
        "row_count",
        "median_forward_return_vs_baseline",
        "top_30pct_flag_rate_vs_baseline",
    ]
    if bottom_available and "bottom_30pct_flag_rate_vs_baseline" in years.columns:
        columns.append("bottom_30pct_flag_rate_vs_baseline")
    if years.empty:
        return pd.DataFrame(columns=columns)
    buckets = ["higher_conviction_price_strength", "price_strength_candidate", "high_volatility_trap"]
    rows = years[years["price_strength_scorecard_bucket"].isin(buckets)].copy()
    rows["_score"] = rows["price_strength_scorecard_bucket"].map(SCORE_VALUES).fillna(0)
    return rows.sort_values(["_score", "calendar_year"], ascending=[False, True])[columns]


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_None._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_format_markdown_value(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _format_markdown_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


def _output_file_guide_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "File": filename,
                "Format": details["format"],
                "Purpose": details["purpose"],
                "Canonical?": "Yes" if details["canonical"] else "No",
            }
            for filename, details in OUTPUT_FILE_PURPOSES.items()
        ]
    )


def terminal_summary(paths: dict[str, Path], summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Price strength scorecard v0 complete.",
            "",
            f"Rows analyzed: {summary['rows_analyzed']}",
            f"Scorecard-bucket eligible rows: {summary['scorecard_bucket_eligible_rows']}",
            f"Scorecard-bucket ineligible rows: {summary['scorecard_bucket_ineligible_rows']}",
            f"Symbols analyzed: {summary['symbols_analyzed']}",
            f"Time windows: {', '.join(summary['time_windows_analyzed'])}",
            "",
            "Human-readable output:",
            f"- Markdown report: {paths['markdown_report']}",
            "",
            "Canonical machine-readable outputs:",
            f"- Scorecard parquet: {paths['scorecard_parquet']}",
            f"- Summary parquet: {paths['summary_parquet']}",
            f"- Year diagnostics parquet: {paths['years_parquet']}",
            f"- Stability parquet: {paths['stability_parquet']}",
            f"- Metadata JSON: {paths['metadata']}",
            "",
            "Convenience exports:",
            f"- Scorecard CSV: {paths['scorecard_csv']}",
            f"- Summary CSV: {paths['summary_csv']}",
            f"- Year diagnostics CSV: {paths['years_csv']}",
            f"- Stability CSV: {paths['stability_csv']}",
        ]
    )


def _partial_year_metadata(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "max_calendar_year_is_treated_as_partial": True,
        "current_partial_year": _max_calendar_year(frame),
        "latest_completed_stability_year": _latest_completed_year(frame),
        "default_behavior": (
            "The max calendar year is summarized as current_partial_year and excluded from completed-history "
            "windows by default."
        ),
    }


def _max_calendar_year(frame: pd.DataFrame) -> int | None:
    if frame.empty:
        return None
    years = pd.to_datetime(frame["date"]).dt.year.dropna()
    return int(years.max()) if not years.empty else None


def _latest_completed_year(frame: pd.DataFrame) -> int | None:
    max_year = _max_calendar_year(frame)
    if max_year is None:
        return None
    years = pd.to_datetime(frame["date"]).dt.year
    completed = years[years < max_year]
    return int(completed.max()) if not completed.empty else None


def _counts_sentence(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in counts.items()) if counts else "none"


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _positive_count(values: pd.Series | None) -> int:
    if values is None:
        return 0
    return int((pd.to_numeric(values, errors="coerce") > 0).sum())


def _negative_count(values: pd.Series | None) -> int:
    if values is None:
        return 0
    return int((pd.to_numeric(values, errors="coerce") < 0).sum())


def _share(count: int | None, total: int) -> float | None:
    if count is None or total == 0:
        return None
    return count / total


def _mean_or_none(values: pd.Series | None) -> float | None:
    if values is None:
        return None
    return _finite_or_none(pd.to_numeric(values, errors="coerce").mean())


def _finite_or_none(value: Any) -> float | None:
    if pd.isna(value):
        return None
    numeric = float(value)
    if numeric == float("inf") or numeric == float("-inf"):
        return None
    return numeric
