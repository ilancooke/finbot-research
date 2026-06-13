from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from finbot_research.config import (
    bucket_candidate_rule_stability_csv_path,
    bucket_candidate_rule_stability_path,
    bucket_candidate_rule_years_csv_path,
    bucket_candidate_rule_years_path,
    bucket_candidate_rules_csv_path,
    bucket_candidate_rules_path,
    bucket_signal_report_path,
    bucket_signal_summary_csv_path,
    bucket_signal_summary_path,
    labels_path,
    price_features_path,
    relative_features_path,
)
from finbot_research.io import parquet_columns, read_parquet, write_csv, write_parquet
from finbot_research.metadata import write_metadata
from finbot_research.schemas import PRIMARY_CLASSIFICATION_LABEL, PRIMARY_REGRESSION_LABEL
from finbot_research.validation import ValidationError, validate_joinable_dataset

DATASET_NAME = "equity_bucket_signal_summary"
BOTTOM_CLASSIFICATION_LABEL = "forward_63d_bottom_30pct_sector_flag"

BUCKET_FEATURES = {
    "volatility_63d_bucket": "volatility_63d",
    "momentum_63d_sector_bucket": "return_63d_sector_pct_rank",
    "drawdown_52w_sector_bucket": "drawdown_from_52w_high_sector_pct_rank",
    "dollar_volume_63d_bucket": "average_dollar_volume_63d",
}

BUCKET_DEFINITIONS = {
    "volatility_63d_bucket": {
        "source_feature": "volatility_63d",
        "method": "cross_sectional_date_percentile_rank",
        "buckets": {
            "low_volatility": "<= 0.30",
            "medium_volatility": "> 0.30 and <= 0.70",
            "high_volatility": "> 0.70",
        },
    },
    "momentum_63d_sector_bucket": {
        "source_feature": "return_63d_sector_pct_rank",
        "method": "existing_percentile_rank_or_decile_scaled_to_0_1",
        "buckets": {
            "weak_momentum": "<= 0.30",
            "middle_momentum": "> 0.30 and < 0.70",
            "strong_momentum": ">= 0.70",
        },
    },
    "drawdown_52w_sector_bucket": {
        "source_feature": "drawdown_from_52w_high_sector_pct_rank",
        "method": "existing_percentile_rank_or_decile_scaled_to_0_1",
        "buckets": {
            "sector_relative_deep_drawdown": "<= 0.30",
            "sector_relative_mid_drawdown": "> 0.30 and < 0.70",
            "sector_relative_near_52w_high": ">= 0.70",
        },
    },
    "dollar_volume_63d_bucket": {
        "source_feature": "average_dollar_volume_63d",
        "method": "cross_sectional_date_percentile_rank",
        "buckets": {
            "lower_dollar_volume": "<= 0.30",
            "middle_dollar_volume": "> 0.30 and <= 0.70",
            "highest_dollar_volume": "> 0.70",
        },
    },
}

COMBINATION_DEFINITIONS = [
    ("volatility_63d_bucket", "momentum_63d_sector_bucket"),
    ("volatility_63d_bucket", "drawdown_52w_sector_bucket"),
    ("momentum_63d_sector_bucket", "drawdown_52w_sector_bucket"),
    ("dollar_volume_63d_bucket", "momentum_63d_sector_bucket"),
    ("dollar_volume_63d_bucket", "volatility_63d_bucket"),
]

DOLLAR_VOLUME_INTERPRETATION = (
    "Dollar volume is treated as a size/liquidity/crowding proxy within a mid-cap+ tradable universe, "
    "not as a simple illiquidity signal."
)
DRAWDOWN_INTERPRETATION_NOTE = (
    "Drawdown is computed as adjusted_close / rolling_52w_high - 1. Values closer to 0 indicate stocks "
    "closer to their 52-week highs. Sector percentile ranks are ascending, so higher percentile rank means "
    "stronger price resilience relative to sector peers."
)

CANDIDATE_RULE_DEFINITIONS = [
    {
        "rule_name": "high_volatility_plus_sector_relative_near_52w_high",
        "conditions": {
            "volatility_63d_bucket": "high_volatility",
            "drawdown_52w_sector_bucket": "sector_relative_near_52w_high",
        },
    },
    {
        "rule_name": "high_volatility_plus_strong_momentum",
        "conditions": {
            "volatility_63d_bucket": "high_volatility",
            "momentum_63d_sector_bucket": "strong_momentum",
        },
    },
    {
        "rule_name": "sector_relative_near_52w_high_plus_strong_momentum",
        "conditions": {
            "drawdown_52w_sector_bucket": "sector_relative_near_52w_high",
            "momentum_63d_sector_bucket": "strong_momentum",
        },
    },
    {
        "rule_name": "high_volatility_plus_sector_relative_near_52w_high_plus_strong_momentum",
        "conditions": {
            "volatility_63d_bucket": "high_volatility",
            "drawdown_52w_sector_bucket": "sector_relative_near_52w_high",
            "momentum_63d_sector_bucket": "strong_momentum",
        },
    },
    {
        "rule_name": "high_volatility_plus_weak_momentum",
        "conditions": {
            "volatility_63d_bucket": "high_volatility",
            "momentum_63d_sector_bucket": "weak_momentum",
        },
    },
    {
        "rule_name": "high_volatility_plus_sector_relative_deep_drawdown",
        "conditions": {
            "volatility_63d_bucket": "high_volatility",
            "drawdown_52w_sector_bucket": "sector_relative_deep_drawdown",
        },
    },
]

OUTPUT_FILE_PURPOSES = {
    "equity_bucket_signal_report.md": {
        "format": "markdown",
        "purpose": "Human-readable bucket-level interaction diagnostics report.",
        "canonical": True,
    },
    "equity_bucket_signal_summary.parquet": {
        "format": "parquet",
        "purpose": "Canonical bucket-level interaction metrics by window and bucket pair.",
        "canonical": True,
    },
    "equity_bucket_signal_summary.csv": {
        "format": "csv",
        "purpose": "Convenience export for manual inspection of bucket interaction metrics.",
        "canonical": False,
    },
    "equity_bucket_candidate_rules.parquet": {
        "format": "parquet",
        "purpose": "Canonical focused candidate-rule diagnostics by time window.",
        "canonical": True,
    },
    "equity_bucket_candidate_rules.csv": {
        "format": "csv",
        "purpose": "Convenience export for manual inspection of candidate-rule diagnostics.",
        "canonical": False,
    },
    "equity_bucket_candidate_rule_years.parquet": {
        "format": "parquet",
        "purpose": "Canonical year-by-year candidate-rule diagnostics with baseline comparisons.",
        "canonical": True,
    },
    "equity_bucket_candidate_rule_years.csv": {
        "format": "csv",
        "purpose": "Convenience export for manual inspection of year-by-year candidate-rule diagnostics.",
        "canonical": False,
    },
    "equity_bucket_candidate_rule_stability.parquet": {
        "format": "parquet",
        "purpose": "Canonical completed-year stability summary for candidate rules.",
        "canonical": True,
    },
    "equity_bucket_candidate_rule_stability.csv": {
        "format": "csv",
        "purpose": "Convenience export for manual inspection of candidate-rule stability.",
        "canonical": False,
    },
    "equity_bucket_signal_summary.metadata.json": {
        "format": "json",
        "purpose": "Documents inputs, outputs, bucket definitions, windows, and generation metadata.",
        "canonical": True,
    },
}

SUMMARY_COLUMNS = [
    "time_window",
    "start_year",
    "end_year",
    "start_date",
    "end_date",
    "is_current_partial_year_window",
    "combo_name",
    "bucket_1_feature",
    "bucket_1_value",
    "bucket_2_feature",
    "bucket_2_value",
    "row_count",
    "symbol_count",
    "avg_forward_63d_sector_relative_return",
    "median_forward_63d_sector_relative_return",
    "avg_minus_median_forward_63d_sector_relative_return",
    "top_30pct_sector_flag_rate",
    "bottom_30pct_sector_flag_rate",
]

CANDIDATE_RULE_COLUMNS = [
    "rule_name",
    "time_window",
    "start_year",
    "end_year",
    "start_date",
    "end_date",
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

CANDIDATE_RULE_YEAR_COLUMNS = [
    "rule_name",
    "calendar_year",
    "is_partial_year",
    "included_in_completed_year_stability",
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

CANDIDATE_RULE_STABILITY_COLUMNS = [
    "rule_name",
    "completed_years_count",
    "years_avg_above_baseline",
    "years_median_above_baseline",
    "years_top_rate_above_baseline",
    "pct_years_avg_above_baseline",
    "pct_years_median_above_baseline",
    "pct_years_top_rate_above_baseline",
    "mean_avg_return_vs_baseline",
    "mean_median_return_vs_baseline",
    "mean_top_rate_vs_baseline",
    "mean_bottom_rate_vs_baseline",
    "stability_assessment",
]

STABILITY_ASSESSMENT_RULES = {
    "broadly_consistent": "completed-year median and top-rate lift are positive in at least 60% of years, with positive mean median and top-rate lift",
    "average_only_tail_driven": "average lift is positive in at least 60% of completed years, but median or top-rate support is weaker",
    "mixed_or_regime_dependent": "completed-year evidence is mixed across average, median, and top-rate lift",
    "weak_or_negative": "median and top-rate support are weak or negative across completed years",
    "insufficient_data": "fewer than three completed years with candidate-rule observations",
}


def build_bucket_signal_diagnostics(data_root: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    price_path = price_features_path(data_root)
    relative_path = relative_features_path(data_root)
    label_path = labels_path(data_root)
    output_path = bucket_signal_summary_path(data_root)
    output_csv_path = bucket_signal_summary_csv_path(data_root)
    candidate_rules_path = bucket_candidate_rules_path(data_root)
    candidate_rules_csv_path = bucket_candidate_rules_csv_path(data_root)
    candidate_rule_years_path = bucket_candidate_rule_years_path(data_root)
    candidate_rule_years_csv_path = bucket_candidate_rule_years_csv_path(data_root)
    candidate_rule_stability_path = bucket_candidate_rule_stability_path(data_root)
    candidate_rule_stability_csv_path = bucket_candidate_rule_stability_csv_path(data_root)
    report_path = bucket_signal_report_path(data_root)

    requested_features = list(BUCKET_FEATURES.values())
    price_columns = _existing_columns(price_path, ["symbol", "date", *requested_features])
    relative_columns = _existing_columns(relative_path, ["symbol", "date", *requested_features])
    label_columns = _existing_columns(
        label_path,
        ["symbol", "date", PRIMARY_REGRESSION_LABEL, PRIMARY_CLASSIFICATION_LABEL, BOTTOM_CLASSIFICATION_LABEL],
    )
    price_features = read_parquet(price_path, columns=price_columns)
    relative_features = read_parquet(relative_path, columns=relative_columns)
    labels = read_parquet(label_path, columns=label_columns)

    summary, candidate_rules, candidate_rule_years, candidate_rule_stability, diagnostics_summary = compute_bucket_signal_diagnostics(
        price_features,
        relative_features,
        labels,
    )
    write_parquet(summary, output_path)
    write_csv(summary, output_csv_path)
    write_parquet(candidate_rules, candidate_rules_path)
    write_csv(candidate_rules, candidate_rules_csv_path)
    write_parquet(candidate_rule_years, candidate_rule_years_path)
    write_csv(candidate_rule_years, candidate_rule_years_csv_path)
    write_parquet(candidate_rule_stability, candidate_rule_stability_path)
    write_csv(candidate_rule_stability, candidate_rule_stability_csv_path)
    write_markdown_report(
        summary,
        report_path,
        diagnostics_summary=diagnostics_summary,
        candidate_rules=candidate_rules,
        candidate_rule_years=candidate_rule_years,
        candidate_rule_stability=candidate_rule_stability,
    )
    metadata_path = write_metadata(
        dataset_name=DATASET_NAME,
        output_path=output_path,
        input_paths=[price_path, relative_path, label_path],
        dataframe=summary,
        extra_metadata={
            "dataset_type": "research_bucket_diagnostics",
            "research_domain": "bucket_signal_diagnostics",
            "primary_regression_label": PRIMARY_REGRESSION_LABEL,
            "primary_classification_label": PRIMARY_CLASSIFICATION_LABEL,
            "bottom_classification_label": BOTTOM_CLASSIFICATION_LABEL if diagnostics_summary["bottom_label_available"] else None,
            "output_paths": {
                "summary_parquet": str(output_path),
                "summary_csv": str(output_csv_path),
                "candidate_rules_parquet": str(candidate_rules_path),
                "candidate_rules_csv": str(candidate_rules_csv_path),
                "candidate_rule_years_parquet": str(candidate_rule_years_path),
                "candidate_rule_years_csv": str(candidate_rule_years_csv_path),
                "candidate_rule_stability_parquet": str(candidate_rule_stability_path),
                "candidate_rule_stability_csv": str(candidate_rule_stability_csv_path),
                "markdown_report": str(report_path),
                "metadata": str(output_path.with_suffix(".metadata.json")),
            },
            "output_file_purposes": OUTPUT_FILE_PURPOSES,
            "bucket_features": diagnostics_summary["bucket_features"],
            "skipped_features": diagnostics_summary["skipped_features"],
            "bucket_definitions": BUCKET_DEFINITIONS,
            "combination_definitions": diagnostics_summary["combination_definitions"],
            "skipped_combinations": diagnostics_summary["skipped_combinations"],
            "candidate_rule_definitions": diagnostics_summary["candidate_rule_definitions"],
            "skipped_candidate_rules": diagnostics_summary["skipped_candidate_rules"],
            "baseline_definition": diagnostics_summary["baseline_definition"],
            "stability_assessment_rules": STABILITY_ASSESSMENT_RULES,
            "time_windows": diagnostics_summary["time_windows"],
            "partial_year_handling": diagnostics_summary["partial_year_handling"],
            "dollar_volume_interpretation": DOLLAR_VOLUME_INTERPRETATION,
            "drawdown_interpretation_note": DRAWDOWN_INTERPRETATION_NOTE,
            "csv_is_convenience_export": True,
            "parquet_is_canonical": True,
        },
    )
    paths = {
        "summary_parquet": output_path,
        "summary_csv": output_csv_path,
        "candidate_rules_parquet": candidate_rules_path,
        "candidate_rules_csv": candidate_rules_csv_path,
        "candidate_rule_years_parquet": candidate_rule_years_path,
        "candidate_rule_years_csv": candidate_rule_years_csv_path,
        "candidate_rule_stability_parquet": candidate_rule_stability_path,
        "candidate_rule_stability_csv": candidate_rule_stability_csv_path,
        "markdown_report": report_path,
        "metadata": metadata_path,
    }
    return paths, diagnostics_summary


def _existing_columns(path: Path, requested_columns: list[str]) -> list[str]:
    available = set(parquet_columns(path))
    return [column for column in requested_columns if column in available]


def compute_bucket_signal_diagnostics(
    price_features: pd.DataFrame,
    relative_features: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    validate_joinable_dataset(price_features, "price_features")
    validate_joinable_dataset(relative_features, "relative_features")
    validate_joinable_dataset(labels, "labels")
    if PRIMARY_REGRESSION_LABEL not in labels.columns or PRIMARY_CLASSIFICATION_LABEL not in labels.columns:
        raise ValidationError("labels missing required forward-return columns")

    features = _merge_feature_inputs(price_features, relative_features)
    labels = _normalize_symbol_date(labels)
    bottom_label_available = BOTTOM_CLASSIFICATION_LABEL in labels.columns
    label_columns = ["symbol", "date", PRIMARY_REGRESSION_LABEL, PRIMARY_CLASSIFICATION_LABEL]
    if bottom_label_available:
        label_columns.append(BOTTOM_CLASSIFICATION_LABEL)
    merged = features.merge(labels[label_columns], on=["symbol", "date"], how="inner")
    if merged.empty:
        raise ValidationError("Joined feature/label dataset is empty")

    bucketed, bucket_summary = assign_feature_buckets(merged)
    windows = _time_windows(bucketed)
    rows: list[dict[str, Any]] = []
    available_buckets = set(bucket_summary["bucket_features"])
    skipped_combinations: list[dict[str, Any]] = []
    combination_definitions: list[dict[str, str]] = []
    for bucket_1, bucket_2 in COMBINATION_DEFINITIONS:
        combo_name = f"{bucket_1}_x_{bucket_2}"
        definition = {"combo_name": combo_name, "bucket_1_feature": bucket_1, "bucket_2_feature": bucket_2}
        combination_definitions.append(definition)
        missing = [bucket for bucket in (bucket_1, bucket_2) if bucket not in available_buckets]
        if missing:
            skipped_combinations.append({**definition, "missing_bucket_features": missing})
            continue
        for window in windows:
            rows.extend(_combination_rows(bucketed, bucket_1, bucket_2, window, bottom_label_available=bottom_label_available))

    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    if not bottom_label_available and "bottom_30pct_sector_flag_rate" in summary.columns:
        summary = summary.drop(columns=["bottom_30pct_sector_flag_rate"])
    candidate_rules, skipped_candidate_rules = compute_candidate_rule_diagnostics(
        bucketed,
        windows,
        available_buckets=available_buckets,
        bottom_label_available=bottom_label_available,
    )
    candidate_rule_years, yearly_skipped_candidate_rules = compute_candidate_rule_year_diagnostics(
        bucketed,
        available_buckets=available_buckets,
        bottom_label_available=bottom_label_available,
    )
    candidate_rule_stability = compute_candidate_rule_stability(candidate_rule_years)
    if not bottom_label_available and "bottom_30pct_sector_flag_rate" in candidate_rules.columns:
        candidate_rules = candidate_rules.drop(columns=["bottom_30pct_sector_flag_rate"])
    if not bottom_label_available and "bottom_30pct_sector_flag_rate" in candidate_rule_years.columns:
        candidate_rule_years = candidate_rule_years.drop(columns=["bottom_30pct_sector_flag_rate"])
    diagnostics_summary = {
        "rows_analyzed": int(len(bucketed)),
        "combinations_analyzed": int(summary["combo_name"].nunique()) if not summary.empty else 0,
        "time_windows_analyzed": [window["time_window"] for window in windows],
        "current_partial_year": _max_calendar_year(bucketed),
        "latest_completed_stability_year": _latest_completed_year(bucketed),
        "bottom_label_available": bottom_label_available,
        "bucket_features": bucket_summary["bucket_features"],
        "skipped_features": bucket_summary["skipped_features"],
        "combination_definitions": combination_definitions,
        "skipped_combinations": skipped_combinations,
        "candidate_rule_definitions": CANDIDATE_RULE_DEFINITIONS,
        "skipped_candidate_rules": _merge_skipped_rules(skipped_candidate_rules, yearly_skipped_candidate_rules),
        "baseline_definition": "The baseline is the full eligible joined feature/label universe for the same time window or calendar year.",
        "time_windows": _window_metadata(windows),
        "partial_year_handling": _partial_year_metadata(bucketed),
        "top_average_combination": _top_combination(summary, "avg_forward_63d_sector_relative_return"),
        "top_median_combination": _top_combination(summary, "median_forward_63d_sector_relative_return"),
        "largest_avg_median_gap_combination": _top_combination(
            summary,
            "avg_minus_median_forward_63d_sector_relative_return",
            absolute=True,
        ),
    }
    return summary, candidate_rules, candidate_rule_years, candidate_rule_stability, diagnostics_summary


def assign_feature_buckets(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    bucketed = _normalize_symbol_date(frame)
    bucket_features: list[str] = []
    skipped_features: list[str] = []
    for bucket_name, source_feature in BUCKET_FEATURES.items():
        if source_feature not in bucketed.columns:
            skipped_features.append(source_feature)
            continue
        if bucket_name in {"volatility_63d_bucket", "dollar_volume_63d_bucket"}:
            percentile = _date_percentile_rank(bucketed, source_feature)
            bucketed[bucket_name] = _three_bucket_from_percentile(percentile, bucket_name)
        else:
            percentile = _rank_like_percentile(bucketed, source_feature)
            bucketed[bucket_name] = _three_bucket_from_percentile(percentile, bucket_name)
        bucket_features.append(bucket_name)
    return bucketed, {"bucket_features": bucket_features, "skipped_features": skipped_features}


def _merge_feature_inputs(price_features: pd.DataFrame, relative_features: pd.DataFrame) -> pd.DataFrame:
    price = _normalize_symbol_date(price_features)
    relative = _normalize_symbol_date(relative_features)
    merged = price.merge(relative, on=["symbol", "date"], how="outer", suffixes=("_price", "_relative"))
    for feature in BUCKET_FEATURES.values():
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


def _date_percentile_rank(frame: pd.DataFrame, feature: str) -> pd.Series:
    values = pd.to_numeric(frame[feature], errors="coerce")
    return values.groupby(frame["date"], sort=False).transform(lambda group: group.rank(method="first", pct=True, na_option="keep"))


def _rank_like_percentile(frame: pd.DataFrame, feature: str) -> pd.Series:
    values = pd.to_numeric(frame[feature], errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return values
    if valid.min() >= 0 and valid.max() <= 1:
        return values
    if valid.min() >= 1 and valid.max() <= 10:
        return values / 10
    return _date_percentile_rank(frame, feature)


def _three_bucket_from_percentile(percentile: pd.Series, bucket_name: str) -> pd.Series:
    values = pd.to_numeric(percentile, errors="coerce")
    result = pd.Series(pd.NA, index=values.index, dtype="string")
    if bucket_name == "volatility_63d_bucket":
        labels = ("low_volatility", "medium_volatility", "high_volatility")
        result[values <= 0.30] = labels[0]
        result[(values > 0.30) & (values <= 0.70)] = labels[1]
        result[values > 0.70] = labels[2]
    elif bucket_name == "momentum_63d_sector_bucket":
        result[values <= 0.30] = "weak_momentum"
        result[(values > 0.30) & (values < 0.70)] = "middle_momentum"
        result[values >= 0.70] = "strong_momentum"
    elif bucket_name == "drawdown_52w_sector_bucket":
        result[values <= 0.30] = "sector_relative_deep_drawdown"
        result[(values > 0.30) & (values < 0.70)] = "sector_relative_mid_drawdown"
        result[values >= 0.70] = "sector_relative_near_52w_high"
    elif bucket_name == "dollar_volume_63d_bucket":
        result[values <= 0.30] = "lower_dollar_volume"
        result[(values > 0.30) & (values <= 0.70)] = "middle_dollar_volume"
        result[values > 0.70] = "highest_dollar_volume"
    else:
        raise ValidationError(f"Unknown bucket name: {bucket_name}")
    return result


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


def _combination_rows(
    frame: pd.DataFrame,
    bucket_1: str,
    bucket_2: str,
    window: dict[str, Any],
    *,
    bottom_label_available: bool,
) -> list[dict[str, Any]]:
    years = pd.to_datetime(frame["date"]).dt.year
    window_rows = frame[(years >= window["start_year"]) & (years <= window["end_year"])].dropna(
        subset=[bucket_1, bucket_2, PRIMARY_REGRESSION_LABEL, PRIMARY_CLASSIFICATION_LABEL]
    )
    if window_rows.empty:
        return []
    combo_name = f"{bucket_1}_x_{bucket_2}"
    rows: list[dict[str, Any]] = []
    grouped = window_rows.groupby([bucket_1, bucket_2], sort=True, dropna=True)
    for (bucket_1_value, bucket_2_value), group in grouped:
        returns = pd.to_numeric(group[PRIMARY_REGRESSION_LABEL], errors="coerce")
        top_flags = pd.to_numeric(group[PRIMARY_CLASSIFICATION_LABEL], errors="coerce")
        avg_return = _finite_or_none(returns.mean())
        median_return = _finite_or_none(returns.median())
        row = {
            **window,
            "start_date": str(min(group["date"])),
            "end_date": str(max(group["date"])),
            "combo_name": combo_name,
            "bucket_1_feature": bucket_1,
            "bucket_1_value": str(bucket_1_value),
            "bucket_2_feature": bucket_2,
            "bucket_2_value": str(bucket_2_value),
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
        rows.append(row)
    return rows


def compute_candidate_rule_diagnostics(
    frame: pd.DataFrame,
    windows: list[dict[str, Any]],
    *,
    available_buckets: set[str],
    bottom_label_available: bool,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skipped_rules: list[dict[str, Any]] = []
    for rule in CANDIDATE_RULE_DEFINITIONS:
        conditions = dict(rule["conditions"])
        missing_buckets = [bucket for bucket in conditions if bucket not in available_buckets]
        if missing_buckets:
            skipped_rules.append(
                {
                    "rule_name": rule["rule_name"],
                    "missing_bucket_features": missing_buckets,
                }
            )
            continue
        for window in windows:
            baseline = _window_baseline_metrics(frame, window, bottom_label_available=bottom_label_available)
            rows.append(
                _candidate_rule_row(
                    frame,
                    str(rule["rule_name"]),
                    conditions,
                    window,
                    baseline=baseline,
                    bottom_label_available=bottom_label_available,
                )
            )
    return pd.DataFrame(rows, columns=CANDIDATE_RULE_COLUMNS), skipped_rules


def _candidate_rule_row(
    frame: pd.DataFrame,
    rule_name: str,
    conditions: dict[str, str],
    window: dict[str, Any],
    *,
    baseline: dict[str, Any],
    bottom_label_available: bool,
) -> dict[str, Any]:
    years = pd.to_datetime(frame["date"]).dt.year
    mask = (years >= window["start_year"]) & (years <= window["end_year"])
    for bucket_name, bucket_value in conditions.items():
        mask &= frame[bucket_name] == bucket_value
    window_rows = frame[mask].dropna(subset=[PRIMARY_REGRESSION_LABEL, PRIMARY_CLASSIFICATION_LABEL])
    return {
        "rule_name": rule_name,
        **window,
        **_with_baseline_differences(_metric_fields(window_rows, bottom_label_available=bottom_label_available), baseline),
    }


def compute_candidate_rule_year_diagnostics(
    frame: pd.DataFrame,
    *,
    available_buckets: set[str],
    bottom_label_available: bool,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if frame.empty:
        return pd.DataFrame(columns=CANDIDATE_RULE_YEAR_COLUMNS), []

    working = frame.copy()
    working["calendar_year"] = pd.to_datetime(working["date"]).dt.year
    max_year = int(working["calendar_year"].max())
    baseline_by_year = {
        int(calendar_year): _metric_fields(group, bottom_label_available=bottom_label_available)
        for calendar_year, group in working.groupby("calendar_year", sort=True)
    }

    rows: list[dict[str, Any]] = []
    skipped_rules: list[dict[str, Any]] = []
    for rule in CANDIDATE_RULE_DEFINITIONS:
        conditions = dict(rule["conditions"])
        missing_buckets = [bucket for bucket in conditions if bucket not in available_buckets]
        if missing_buckets:
            skipped_rules.append(
                {
                    "rule_name": rule["rule_name"],
                    "missing_bucket_features": missing_buckets,
                }
            )
            continue
        mask = pd.Series(True, index=working.index)
        for bucket_name, bucket_value in conditions.items():
            mask &= working[bucket_name] == bucket_value
        rule_rows = working[mask]
        for calendar_year in sorted(baseline_by_year):
            group = rule_rows[rule_rows["calendar_year"] == calendar_year]
            metrics = _with_baseline_differences(
                _metric_fields(group, bottom_label_available=bottom_label_available),
                baseline_by_year[calendar_year],
            )
            rows.append(
                {
                    "rule_name": str(rule["rule_name"]),
                    "calendar_year": int(calendar_year),
                    "is_partial_year": bool(calendar_year == max_year),
                    "included_in_completed_year_stability": bool(calendar_year != max_year),
                    **metrics,
                }
            )
    return pd.DataFrame(rows, columns=CANDIDATE_RULE_YEAR_COLUMNS), skipped_rules


def compute_candidate_rule_stability(candidate_rule_years: pd.DataFrame) -> pd.DataFrame:
    if candidate_rule_years.empty:
        return pd.DataFrame(columns=CANDIDATE_RULE_STABILITY_COLUMNS)

    frame = candidate_rule_years.copy()
    if "included_in_completed_year_stability" not in frame.columns:
        frame["included_in_completed_year_stability"] = True
    frame = frame[frame["included_in_completed_year_stability"].fillna(False)].copy()
    if frame.empty:
        return pd.DataFrame(columns=CANDIDATE_RULE_STABILITY_COLUMNS)

    rows: list[dict[str, Any]] = []
    for rule_name, group in frame.groupby("rule_name", sort=False):
        valid = group[pd.to_numeric(group["row_count"], errors="coerce").fillna(0) > 0].copy()
        completed_years = int(len(valid))
        avg_lift = pd.to_numeric(valid["avg_forward_return_vs_baseline"], errors="coerce").dropna()
        median_lift = pd.to_numeric(valid["median_forward_return_vs_baseline"], errors="coerce").dropna()
        top_lift = pd.to_numeric(valid["top_30pct_flag_rate_vs_baseline"], errors="coerce").dropna()
        bottom_lift = pd.to_numeric(valid.get("bottom_30pct_flag_rate_vs_baseline"), errors="coerce").dropna()
        years_avg = int((avg_lift > 0).sum())
        years_median = int((median_lift > 0).sum())
        years_top = int((top_lift > 0).sum())
        row = {
            "rule_name": rule_name,
            "completed_years_count": completed_years,
            "years_avg_above_baseline": years_avg,
            "years_median_above_baseline": years_median,
            "years_top_rate_above_baseline": years_top,
            "pct_years_avg_above_baseline": years_avg / completed_years if completed_years else None,
            "pct_years_median_above_baseline": years_median / completed_years if completed_years else None,
            "pct_years_top_rate_above_baseline": years_top / completed_years if completed_years else None,
            "mean_avg_return_vs_baseline": _finite_or_none(avg_lift.mean()) if len(avg_lift) else None,
            "mean_median_return_vs_baseline": _finite_or_none(median_lift.mean()) if len(median_lift) else None,
            "mean_top_rate_vs_baseline": _finite_or_none(top_lift.mean()) if len(top_lift) else None,
            "mean_bottom_rate_vs_baseline": _finite_or_none(bottom_lift.mean()) if len(bottom_lift) else None,
        }
        row["stability_assessment"] = _stability_assessment(row)
        rows.append(row)
    return pd.DataFrame(rows, columns=CANDIDATE_RULE_STABILITY_COLUMNS)


def _window_baseline_metrics(
    frame: pd.DataFrame,
    window: dict[str, Any],
    *,
    bottom_label_available: bool,
) -> dict[str, Any]:
    years = pd.to_datetime(frame["date"]).dt.year
    window_rows = frame[(years >= window["start_year"]) & (years <= window["end_year"])].dropna(
        subset=[PRIMARY_REGRESSION_LABEL, PRIMARY_CLASSIFICATION_LABEL]
    )
    return _metric_fields(window_rows, bottom_label_available=bottom_label_available)


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


def _stability_assessment(row: dict[str, Any]) -> str:
    years = row.get("completed_years_count") or 0
    if years < 3:
        return "insufficient_data"
    pct_avg = row.get("pct_years_avg_above_baseline")
    pct_median = row.get("pct_years_median_above_baseline")
    pct_top = row.get("pct_years_top_rate_above_baseline")
    mean_avg = row.get("mean_avg_return_vs_baseline")
    mean_median = row.get("mean_median_return_vs_baseline")
    mean_top = row.get("mean_top_rate_vs_baseline")
    if (
        _positive_rate(pct_median)
        and _positive_rate(pct_top)
        and _positive_value(mean_median)
        and _positive_value(mean_top)
    ):
        return "broadly_consistent"
    if _positive_rate(pct_avg) and _positive_value(mean_avg):
        return "average_only_tail_driven"
    if (
        pct_median is not None
        and pct_top is not None
        and float(pct_median) <= 0.40
        and float(pct_top) <= 0.40
        and not _positive_value(mean_median)
        and not _positive_value(mean_top)
    ):
        return "weak_or_negative"
    return "mixed_or_regime_dependent"


def _positive_rate(value: Any) -> bool:
    return value is not None and not pd.isna(value) and float(value) >= 0.60


def _positive_value(value: Any) -> bool:
    return value is not None and not pd.isna(value) and float(value) > 0


def _metric_fields(group: pd.DataFrame, *, bottom_label_available: bool) -> dict[str, Any]:
    if group.empty:
        row = {
            "start_date": None,
            "end_date": None,
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
        "start_date": str(min(group["date"])),
        "end_date": str(max(group["date"])),
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


def write_markdown_report(
    summary: pd.DataFrame,
    path: Path,
    *,
    diagnostics_summary: dict[str, Any],
    candidate_rules: pd.DataFrame | None = None,
    candidate_rule_years: pd.DataFrame | None = None,
    candidate_rule_stability: pd.DataFrame | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    bottom_available = diagnostics_summary["bottom_label_available"]
    candidate_rules = candidate_rules if candidate_rules is not None else pd.DataFrame(columns=CANDIDATE_RULE_COLUMNS)
    candidate_rule_years = (
        candidate_rule_years if candidate_rule_years is not None else pd.DataFrame(columns=CANDIDATE_RULE_YEAR_COLUMNS)
    )
    candidate_rule_stability = (
        candidate_rule_stability
        if candidate_rule_stability is not None
        else pd.DataFrame(columns=CANDIDATE_RULE_STABILITY_COLUMNS)
    )
    lines = [
        "# Equity Bucket Signal Diagnostics",
        "",
        "## Purpose",
        "",
        "This report tests simple, interpretable bucket interactions across volatility, momentum, drawdown, and dollar-volume features before any scorecard is defined.",
        "",
        "## Executive Summary",
        "",
        f"- Bucket combinations analyzed: {diagnostics_summary['combinations_analyzed']}",
        f"- Rows analyzed: {diagnostics_summary['rows_analyzed']}",
        f"- Time windows analyzed: {', '.join(diagnostics_summary['time_windows_analyzed'])}",
        f"- Current partial year: {diagnostics_summary['current_partial_year']}",
        f"- Highest average-return combination: {_combo_sentence(diagnostics_summary['top_average_combination'])}",
        f"- Highest median-return combination: {_combo_sentence(diagnostics_summary['top_median_combination'])}",
        f"- Largest average-minus-median gap: {_combo_sentence(diagnostics_summary['largest_avg_median_gap_combination'])}",
        "",
        "## Key Findings",
        "",
        *_key_findings(summary),
        "",
        "## Bucket Definitions",
        "",
        *_bucket_definition_lines(),
        "",
        "## Top Bucket Combinations by Average Forward Return",
        "",
        _markdown_table(_top_rows(summary, "avg_forward_63d_sector_relative_return", bottom_available=bottom_available)),
        "",
        "## Top Bucket Combinations by Median Forward Return",
        "",
        _markdown_table(_top_rows(summary, "median_forward_63d_sector_relative_return", bottom_available=bottom_available)),
        "",
        "## Bucket Combinations With Large Average-Median Gaps",
        "",
        _markdown_table(_top_rows(summary, "avg_minus_median_forward_63d_sector_relative_return", bottom_available=bottom_available, absolute=True)),
        "",
        "## Candidate Rule Diagnostics",
        "",
        _markdown_table(_candidate_rule_report_rows(candidate_rules, bottom_available=bottom_available)),
        "",
        "The strongest current-regime hypothesis is high volatility combined with sector-relative price resilience near 52-week highs. Full completed-history evidence should be compared before using this in scorecard research.",
        "",
        "## Candidate Rule Stability",
        "",
        _markdown_table(_candidate_rule_stability_rows(candidate_rule_stability)),
        "",
        *_candidate_rule_year_detail_lines(candidate_rule_years, candidate_rule_stability, bottom_available=bottom_available),
        "",
        "## Current Partial Year Snapshot",
        "",
        _markdown_table(
            _top_rows(
                summary[summary["time_window"] == "current_partial_year"],
                "avg_forward_63d_sector_relative_return",
                bottom_available=bottom_available,
            )
        ),
        "",
        "## Combination Details",
        "",
        *_combination_detail_lines(summary, bottom_available=bottom_available),
        "",
        "## Output File Guide",
        "",
        _markdown_table(_output_file_guide_rows()),
        "",
        "## Important Caveats",
        "",
        "- This is research, not a trading rule.",
        "- Bucket diagnostics do not replace backtesting.",
        "- Mean returns may be driven by tails; compare average and median.",
        "- Current partial year is shown separately because it may be relevant to the current regime but incomplete.",
        f"- {DOLLAR_VOLUME_INTERPRETATION}",
        f"- {DRAWDOWN_INTERPRETATION_NOTE}",
        "",
        "## Suggested Next Step",
        "",
        "Review the strongest and most tail-driven bucket pairs manually, then decide whether any interpretable interactions deserve scorecard research.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def terminal_summary(paths: dict[str, Path], summary: dict[str, Any]) -> str:
    lines = [
        "Bucket signal diagnostics complete.",
        "",
        f"Rows analyzed: {summary['rows_analyzed']}",
        f"Combinations analyzed: {summary['combinations_analyzed']}",
        f"Time windows: {', '.join(summary['time_windows_analyzed'])}",
        "",
        "Human-readable output:",
        f"- Markdown report: {paths['markdown_report']}",
        "",
        "Canonical machine-readable output:",
        f"- Summary parquet: {paths['summary_parquet']}",
        f"- Candidate rules parquet: {paths['candidate_rules_parquet']}",
        f"- Candidate rule years parquet: {paths['candidate_rule_years_parquet']}",
        f"- Candidate rule stability parquet: {paths['candidate_rule_stability_parquet']}",
        f"- Metadata JSON: {paths['metadata']}",
        "",
        "Convenience exports:",
        f"- Summary CSV: {paths['summary_csv']}",
        f"- Candidate rules CSV: {paths['candidate_rules_csv']}",
        f"- Candidate rule years CSV: {paths['candidate_rule_years_csv']}",
        f"- Candidate rule stability CSV: {paths['candidate_rule_stability_csv']}",
    ]
    return "\n".join(lines)


def _top_rows(
    summary: pd.DataFrame,
    sort_column: str,
    *,
    bottom_available: bool,
    limit: int = 10,
    absolute: bool = False,
) -> pd.DataFrame:
    columns = [
        "time_window",
        "combo_name",
        "bucket_1_value",
        "bucket_2_value",
        "row_count",
        "symbol_count",
        "avg_forward_63d_sector_relative_return",
        "median_forward_63d_sector_relative_return",
        "avg_minus_median_forward_63d_sector_relative_return",
        "top_30pct_sector_flag_rate",
    ]
    if bottom_available and "bottom_30pct_sector_flag_rate" in summary.columns:
        columns.append("bottom_30pct_sector_flag_rate")
    if summary.empty:
        return pd.DataFrame(columns=columns)
    rows = summary.copy()
    rows["_sort_value"] = pd.to_numeric(rows[sort_column], errors="coerce").abs() if absolute else pd.to_numeric(
        rows[sort_column],
        errors="coerce",
    )
    return rows.sort_values("_sort_value", ascending=False, na_position="last").head(limit)[columns]


def _candidate_rule_report_rows(
    candidate_rules: pd.DataFrame,
    *,
    bottom_available: bool,
) -> pd.DataFrame:
    columns = [
        "rule_name",
        "time_window",
        "row_count",
        "symbol_count",
        "avg_forward_63d_sector_relative_return",
        "median_forward_63d_sector_relative_return",
        "avg_minus_median_forward_63d_sector_relative_return",
        "top_30pct_sector_flag_rate",
    ]
    if bottom_available and "bottom_30pct_sector_flag_rate" in candidate_rules.columns:
        columns.append("bottom_30pct_sector_flag_rate")
    if candidate_rules.empty:
        return pd.DataFrame(columns=columns)

    current = candidate_rules[candidate_rules["time_window"] == "current_partial_year"].copy()
    current["_median_sort"] = pd.to_numeric(current["median_forward_63d_sector_relative_return"], errors="coerce")
    current["_avg_sort"] = pd.to_numeric(current["avg_forward_63d_sector_relative_return"], errors="coerce")
    rule_order = (
        current.sort_values(["_median_sort", "_avg_sort"], ascending=False, na_position="last")["rule_name"]
        .drop_duplicates()
        .tolist()
    )
    window_order = {
        "current_partial_year": 0,
        "last_5y_completed": 1,
        "last_10y_completed": 2,
        "full_completed_history": 3,
    }
    rows = candidate_rules[candidate_rules["rule_name"].isin(rule_order)].copy()
    rows["_rule_order"] = rows["rule_name"].map({rule_name: idx for idx, rule_name in enumerate(rule_order)})
    rows["_window_order"] = rows["time_window"].map(window_order).fillna(99)
    rows = rows.sort_values(["_rule_order", "_window_order"]).drop(columns=["_rule_order", "_window_order"])
    return rows[columns].head(24)


def _candidate_rule_stability_rows(candidate_rule_stability: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "rule_name",
        "completed_years_count",
        "pct_years_avg_above_baseline",
        "pct_years_median_above_baseline",
        "pct_years_top_rate_above_baseline",
        "mean_avg_return_vs_baseline",
        "mean_median_return_vs_baseline",
        "mean_top_rate_vs_baseline",
        "mean_bottom_rate_vs_baseline",
        "stability_assessment",
    ]
    if candidate_rule_stability.empty:
        return pd.DataFrame(columns=columns)
    rows = candidate_rule_stability.copy()
    rows["_sort_assessment"] = rows["stability_assessment"].map(
        {
            "broadly_consistent": 0,
            "average_only_tail_driven": 1,
            "mixed_or_regime_dependent": 2,
            "weak_or_negative": 3,
            "insufficient_data": 4,
        }
    ).fillna(9)
    rows["_sort_median"] = pd.to_numeric(rows["mean_median_return_vs_baseline"], errors="coerce")
    rows["_sort_top"] = pd.to_numeric(rows["mean_top_rate_vs_baseline"], errors="coerce")
    rows = rows.sort_values(["_sort_assessment", "_sort_median", "_sort_top"], ascending=[True, False, False])
    return rows[columns]


def _candidate_rule_year_detail_lines(
    candidate_rule_years: pd.DataFrame,
    candidate_rule_stability: pd.DataFrame,
    *,
    bottom_available: bool,
) -> list[str]:
    if candidate_rule_years.empty or candidate_rule_stability.empty:
        return ["_No candidate-rule year details were available._"]

    positive = candidate_rule_stability[
        candidate_rule_stability["stability_assessment"].isin(["broadly_consistent", "average_only_tail_driven"])
    ].copy()
    positive["_sort"] = pd.to_numeric(positive["mean_median_return_vs_baseline"], errors="coerce")
    positive_rules = positive.sort_values("_sort", ascending=False)["rule_name"].head(2).tolist()

    risk = candidate_rule_stability[
        candidate_rule_stability["rule_name"].str.contains("weak_momentum|deep_drawdown", regex=True, na=False)
    ].copy()
    risk["_sort"] = pd.to_numeric(risk["mean_median_return_vs_baseline"], errors="coerce")
    risk_rules = risk.sort_values("_sort", ascending=True)["rule_name"].head(2).tolist()

    selected_rules = list(dict.fromkeys([*positive_rules, *risk_rules]))
    columns = [
        "calendar_year",
        "is_partial_year",
        "row_count",
        "avg_forward_return_vs_baseline",
        "median_forward_return_vs_baseline",
        "top_30pct_flag_rate_vs_baseline",
    ]
    if bottom_available and "bottom_30pct_flag_rate_vs_baseline" in candidate_rule_years.columns:
        columns.append("bottom_30pct_flag_rate_vs_baseline")
    lines: list[str] = ["### Year-by-Year Rule Details", ""]
    for rule_name in selected_rules:
        rows = candidate_rule_years[candidate_rule_years["rule_name"] == rule_name].copy()
        if rows.empty:
            continue
        rows = rows.sort_values("calendar_year", ascending=False).head(12).sort_values("calendar_year")
        lines.extend([f"#### {rule_name}", "", _markdown_table(rows[columns]), ""])
    return lines if len(lines) > 2 else ["_No selected candidate-rule year details were available._"]


def _combination_detail_lines(summary: pd.DataFrame, *, bottom_available: bool) -> list[str]:
    if summary.empty:
        return ["_No bucket combination rows were available._"]
    lines: list[str] = []
    for combo_name in (
        "volatility_63d_bucket_x_momentum_63d_sector_bucket",
        "dollar_volume_63d_bucket_x_momentum_63d_sector_bucket",
    ):
        rows = summary[(summary["combo_name"] == combo_name) & (summary["time_window"] == "full_completed_history")]
        if rows.empty:
            continue
        lines.extend(
            [
                f"### {combo_name}",
                "",
                _markdown_table(_top_rows(rows, "avg_forward_63d_sector_relative_return", bottom_available=bottom_available, limit=9)),
                "",
            ]
        )
    return lines or ["_No compact combination details were available._"]


def _bucket_definition_lines() -> list[str]:
    return [
        "- `volatility_63d_bucket`: `low_volatility`, `medium_volatility`, `high_volatility` from date-level percentile buckets.",
        "- `momentum_63d_sector_bucket`: `weak_momentum`, `middle_momentum`, `strong_momentum` from `return_63d_sector_pct_rank`.",
        "- `drawdown_52w_sector_bucket`: `sector_relative_deep_drawdown`, `sector_relative_mid_drawdown`, `sector_relative_near_52w_high` from `drawdown_from_52w_high_sector_pct_rank`.",
        "- `dollar_volume_63d_bucket`: `lower_dollar_volume`, `middle_dollar_volume`, `highest_dollar_volume` from date-level percentile buckets.",
        f"- {DOLLAR_VOLUME_INTERPRETATION}",
        f"- {DRAWDOWN_INTERPRETATION_NOTE}",
    ]


def _key_findings(summary: pd.DataFrame) -> list[str]:
    findings = [
        "- Dollar-volume buckets should be interpreted as size, crowding, or mega-cap exposure, not an illiquidity preference.",
        "- Current partial-year behavior is shown separately and should not be treated as completed-year evidence.",
    ]
    if summary.empty:
        return [*findings, "- No valid bucket combinations were available."]
    high_strong = _find_combo_row(
        summary,
        "volatility_63d_bucket_x_momentum_63d_sector_bucket",
        "high_volatility",
        "strong_momentum",
    )
    high_weak = _find_combo_row(
        summary,
        "volatility_63d_bucket_x_momentum_63d_sector_bucket",
        "high_volatility",
        "weak_momentum",
    )
    if high_strong is not None and high_weak is not None:
        comparison = "had better" if high_strong > high_weak else "did not have better"
        findings.append(f"- High volatility plus strong momentum {comparison} average outcomes than high volatility plus weak momentum.")
    largest_gap = summary.iloc[pd.to_numeric(summary["avg_minus_median_forward_63d_sector_relative_return"], errors="coerce").abs().idxmax()]
    if abs(float(largest_gap["avg_minus_median_forward_63d_sector_relative_return"])) > 0:
        findings.append("- Some bucket combinations have average-minus-median gaps, so tail effects need manual review.")
    return findings


def _find_combo_row(summary: pd.DataFrame, combo_name: str, bucket_1_value: str, bucket_2_value: str) -> float | None:
    rows = summary[
        (summary["time_window"] == "full_completed_history")
        & (summary["combo_name"] == combo_name)
        & (summary["bucket_1_value"] == bucket_1_value)
        & (summary["bucket_2_value"] == bucket_2_value)
    ]
    if rows.empty:
        return None
    value = rows.iloc[0]["avg_forward_63d_sector_relative_return"]
    return None if pd.isna(value) else float(value)


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


def _merge_skipped_rules(*skipped_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for skipped_group in skipped_groups:
        for item in skipped_group:
            rule_name = str(item["rule_name"])
            existing = merged.setdefault(rule_name, {"rule_name": rule_name, "missing_bucket_features": []})
            existing["missing_bucket_features"] = sorted(
                set(existing["missing_bucket_features"]) | set(item.get("missing_bucket_features", []))
            )
    return list(merged.values())


def _combo_sentence(row: dict[str, Any] | None) -> str:
    if not row:
        return "none"
    return f"{row['time_window']} {row['combo_name']} {row['bucket_1_value']} x {row['bucket_2_value']}"


def _top_combination(summary: pd.DataFrame, column: str, *, absolute: bool = False) -> dict[str, Any] | None:
    if summary.empty or column not in summary.columns:
        return None
    values = pd.to_numeric(summary[column], errors="coerce")
    if values.dropna().empty:
        return None
    if absolute:
        values = values.abs()
    return summary.loc[values.idxmax()].to_dict()


def _window_metadata(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(window) for window in windows]


def _partial_year_metadata(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "max_calendar_year_is_treated_as_partial": True,
        "current_partial_year": _max_calendar_year(frame),
        "latest_completed_stability_year": _latest_completed_year(frame),
        "default_behavior": (
            "The max calendar year is analyzed as current_partial_year and excluded from completed-history "
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


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _finite_or_none(value: Any) -> float | None:
    if pd.isna(value):
        return None
    numeric = float(value)
    if numeric == float("inf") or numeric == float("-inf"):
        return None
    return numeric
