from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from finbot_research.config import (
    feature_label_diagnostics_path,
    labels_path,
    price_features_path,
    relative_features_path,
)
from finbot_research.io import parquet_columns, read_parquet, write_parquet
from finbot_research.metadata import write_metadata
from finbot_research.schemas import (
    PRIMARY_CLASSIFICATION_LABEL,
    PRIMARY_REGRESSION_LABEL,
    SELECTED_FEATURE_COLUMNS,
)
from finbot_research.validation import ValidationError, validate_joinable_dataset

DATASET_NAME = "equity_price_signal_diagnostics"


def build_feature_label_diagnostics(data_root: Path) -> tuple[Path, Path, dict[str, Any]]:
    price_path = price_features_path(data_root)
    relative_path = relative_features_path(data_root)
    label_path = labels_path(data_root)
    output_path = feature_label_diagnostics_path(data_root)

    price_columns = _existing_columns(price_path, ["symbol", "date", *SELECTED_FEATURE_COLUMNS])
    relative_columns = _existing_columns(relative_path, ["symbol", "date", *SELECTED_FEATURE_COLUMNS])
    label_columns = _existing_columns(label_path, ["symbol", "date", PRIMARY_REGRESSION_LABEL, PRIMARY_CLASSIFICATION_LABEL])
    price_features = read_parquet(price_path, columns=price_columns)
    relative_features = read_parquet(relative_path, columns=relative_columns)
    labels = read_parquet(label_path, columns=label_columns)
    diagnostics, summary = compute_feature_label_diagnostics(price_features, relative_features, labels)
    write_parquet(diagnostics, output_path)
    metadata_path = write_metadata(
        dataset_name=DATASET_NAME,
        output_path=output_path,
        input_paths=[price_path, relative_path, label_path],
        dataframe=diagnostics,
        extra_metadata={
            "dataset_type": "research_diagnostics",
            "research_domain": "feature_label_diagnostics",
            "primary_regression_label": PRIMARY_REGRESSION_LABEL,
            "primary_classification_label": PRIMARY_CLASSIFICATION_LABEL,
            "selected_feature_columns": summary["selected_feature_columns"],
            "skipped_feature_columns": summary["skipped_feature_columns"],
            "joined_rows": summary["joined_rows"],
            "source_rows": summary["source_rows"],
            "decile_method": "cross_sectional_by_date_rank_pct_ceil",
            "decile_fallback": "Rows on dates with fewer than two non-null values for a feature are left without a decile for that feature.",
            "included_metric_types": sorted(diagnostics["metric_type"].dropna().unique().tolist()),
        },
    )
    return output_path, metadata_path, summary


def _existing_columns(path: Path, requested_columns: list[str]) -> list[str]:
    available = set(parquet_columns(path))
    return [column for column in requested_columns if column in available]


def compute_feature_label_diagnostics(
    price_features: pd.DataFrame,
    relative_features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    selected_feature_columns: tuple[str, ...] = SELECTED_FEATURE_COLUMNS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    validate_joinable_dataset(price_features, "price_features")
    validate_joinable_dataset(relative_features, "relative_features")
    validate_joinable_dataset(labels, "labels")
    _validate_label_columns(labels)

    price = _normalize_symbol_date(price_features)
    relative = _normalize_symbol_date(relative_features)
    label_frame = _normalize_symbol_date(labels)

    usable_price_columns = [column for column in selected_feature_columns if column in price.columns]
    usable_relative_columns = [column for column in selected_feature_columns if column in relative.columns]
    usable_columns = list(dict.fromkeys([*usable_price_columns, *usable_relative_columns]))
    skipped_columns = [column for column in selected_feature_columns if column not in usable_columns]
    if not usable_columns:
        raise ValidationError("No selected feature columns are present in the input feature datasets")

    merged = price[["symbol", "date", *usable_price_columns]].merge(
        relative[["symbol", "date", *usable_relative_columns]],
        on=["symbol", "date"],
        how="inner",
    )
    merged = merged.merge(
        label_frame[["symbol", "date", PRIMARY_REGRESSION_LABEL, PRIMARY_CLASSIFICATION_LABEL]],
        on=["symbol", "date"],
        how="inner",
    )
    if merged.empty:
        raise ValidationError("Joined feature/label dataset is empty")

    rows: list[dict[str, Any]] = []
    for feature in usable_columns:
        rows.append(_coverage_row(merged, feature))
        rows.append(_spread_row(merged, feature))
        rows.extend(_decile_rows(merged, feature))
        rows.extend(_year_decile_rows(merged, feature))

    diagnostics = pd.DataFrame(rows)
    diagnostics = diagnostics.sort_values(["feature_name", "metric_type", "calendar_year", "decile"], na_position="last")
    diagnostics = diagnostics.reset_index(drop=True)
    summary = {
        "selected_feature_columns": usable_columns,
        "skipped_feature_columns": skipped_columns,
        "joined_rows": int(len(merged)),
        "source_rows": {
            "price_features": int(len(price_features)),
            "relative_features": int(len(relative_features)),
            "labels": int(len(labels)),
        },
    }
    return diagnostics, summary


def _validate_label_columns(labels: pd.DataFrame) -> None:
    missing = [
        column
        for column in (PRIMARY_REGRESSION_LABEL, PRIMARY_CLASSIFICATION_LABEL)
        if column not in labels.columns
    ]
    if missing:
        raise ValidationError(f"labels missing required columns: {missing}")


def _normalize_symbol_date(dataframe: pd.DataFrame) -> pd.DataFrame:
    normalized = dataframe.copy()
    normalized["symbol"] = normalized["symbol"].astype("string").str.upper()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="raise").dt.date
    return normalized.sort_values(["symbol", "date"]).reset_index(drop=True)


def _coverage_row(frame: pd.DataFrame, feature: str) -> dict[str, Any]:
    feature_values = pd.to_numeric(frame[feature], errors="coerce")
    label_values = pd.to_numeric(frame[PRIMARY_REGRESSION_LABEL], errors="coerce")
    classification_values = pd.to_numeric(frame[PRIMARY_CLASSIFICATION_LABEL], errors="coerce")
    paired = pd.DataFrame({"feature": feature_values, "label": label_values}).dropna()
    return {
        "metric_type": "coverage",
        "feature_name": feature,
        "non_null_count": int(feature_values.notna().sum()),
        "null_count": int(feature_values.isna().sum()),
        "null_rate": _safe_rate(int(feature_values.isna().sum()), len(feature_values)),
        "label_non_null_count": int(label_values.notna().sum()),
        "classification_label_non_null_count": int(classification_values.notna().sum()),
        "pearson_corr_with_forward_63d_sector_relative_return": _corr(paired, method="pearson"),
        "spearman_corr_with_forward_63d_sector_relative_return": _corr(paired, method="spearman"),
    }


def _decile_rows(frame: pd.DataFrame, feature: str) -> list[dict[str, Any]]:
    working = _with_cross_sectional_deciles(frame, feature)
    rows: list[dict[str, Any]] = []
    for decile, group in working.dropna(subset=["decile"]).groupby("decile", sort=True):
        rows.append(_decile_metric_row(feature, group, int(decile), calendar_year=None, metric_type="decile"))
    return rows


def _year_decile_rows(frame: pd.DataFrame, feature: str) -> list[dict[str, Any]]:
    working = _with_cross_sectional_deciles(frame, feature)
    working["calendar_year"] = pd.to_datetime(working["date"]).dt.year
    rows: list[dict[str, Any]] = []
    grouped = working.dropna(subset=["decile"]).groupby(["calendar_year", "decile"], sort=True)
    for (calendar_year, decile), group in grouped:
        rows.append(
            _decile_metric_row(
                feature,
                group,
                int(decile),
                calendar_year=int(calendar_year),
                metric_type="year_decile",
            )
        )
    return rows


def _decile_metric_row(
    feature: str,
    group: pd.DataFrame,
    decile: int,
    *,
    calendar_year: int | None,
    metric_type: str,
) -> dict[str, Any]:
    feature_values = pd.to_numeric(group[feature], errors="coerce")
    label_values = pd.to_numeric(group[PRIMARY_REGRESSION_LABEL], errors="coerce")
    classification_values = pd.to_numeric(group[PRIMARY_CLASSIFICATION_LABEL], errors="coerce")
    return {
        "metric_type": metric_type,
        "feature_name": feature,
        "calendar_year": calendar_year,
        "decile": decile,
        "row_count": int(len(group)),
        "feature_min": _finite_or_none(feature_values.min()),
        "feature_max": _finite_or_none(feature_values.max()),
        "feature_mean": _finite_or_none(feature_values.mean()),
        "avg_forward_63d_sector_relative_return": _finite_or_none(label_values.mean()),
        "median_forward_63d_sector_relative_return": _finite_or_none(label_values.median()),
        "top_30pct_sector_flag_rate": _finite_or_none(classification_values.mean()),
    }


def _spread_row(frame: pd.DataFrame, feature: str) -> dict[str, Any]:
    working = _with_cross_sectional_deciles(frame, feature)
    bottom = working[working["decile"] == 1]
    top = working[working["decile"] == 10]
    top_return = _finite_or_none(pd.to_numeric(top[PRIMARY_REGRESSION_LABEL], errors="coerce").mean())
    bottom_return = _finite_or_none(pd.to_numeric(bottom[PRIMARY_REGRESSION_LABEL], errors="coerce").mean())
    top_flag = _finite_or_none(pd.to_numeric(top[PRIMARY_CLASSIFICATION_LABEL], errors="coerce").mean())
    bottom_flag = _finite_or_none(pd.to_numeric(bottom[PRIMARY_CLASSIFICATION_LABEL], errors="coerce").mean())
    return {
        "metric_type": "spread",
        "feature_name": feature,
        "top_decile_avg_forward_return": top_return,
        "bottom_decile_avg_forward_return": bottom_return,
        "top_minus_bottom_avg_forward_return": _difference(top_return, bottom_return),
        "top_decile_top_30pct_flag_rate": top_flag,
        "bottom_decile_top_30pct_flag_rate": bottom_flag,
        "top_minus_bottom_top_30pct_flag_rate": _difference(top_flag, bottom_flag),
    }


def _with_cross_sectional_deciles(frame: pd.DataFrame, feature: str) -> pd.DataFrame:
    working = frame[["symbol", "date", feature, PRIMARY_REGRESSION_LABEL, PRIMARY_CLASSIFICATION_LABEL]].copy()
    working[feature] = pd.to_numeric(working[feature], errors="coerce")
    ranks = working.groupby("date", sort=False)[feature].transform(_date_decile)
    working["decile"] = ranks.astype("Int64")
    return working


def _date_decile(values: pd.Series) -> pd.Series:
    valid_count = int(values.notna().sum())
    if valid_count < 2:
        return pd.Series(pd.NA, index=values.index, dtype="Int64")
    ranks = values.rank(method="first", pct=True, na_option="keep")
    deciles = np.ceil(ranks * 10)
    deciles = deciles.clip(lower=1, upper=10)
    return deciles.astype("Int64")


def _corr(paired: pd.DataFrame, *, method: str) -> float | None:
    if len(paired) < 2:
        return None
    if method == "spearman":
        value = paired["feature"].rank(method="average").corr(paired["label"].rank(method="average"), method="pearson")
    else:
        value = paired["feature"].corr(paired["label"], method=method)
    return _finite_or_none(value)


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _finite_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric
