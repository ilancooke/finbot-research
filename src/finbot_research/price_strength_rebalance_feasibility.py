from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from finbot_research.config import (
    price_strength_rebalance_bucket_count_summary_csv_path,
    price_strength_rebalance_bucket_count_summary_path,
    price_strength_rebalance_bucket_counts_csv_path,
    price_strength_rebalance_bucket_counts_path,
    price_strength_rebalance_feasibility_csv_path,
    price_strength_rebalance_feasibility_path,
    price_strength_rebalance_feasibility_report_path,
    price_strength_rebalance_sector_composition_csv_path,
    price_strength_rebalance_sector_composition_path,
    price_strength_rebalance_sector_concentration_csv_path,
    price_strength_rebalance_sector_concentration_path,
    price_strength_rebalance_turnover_csv_path,
    price_strength_rebalance_turnover_path,
    price_strength_scorecard_path,
    reference_tickers_path,
)
from finbot_research.io import parquet_columns, read_parquet, write_csv, write_parquet
from finbot_research.metadata import write_metadata
from finbot_research.price_strength_scorecard import SCORE_VALUES
from finbot_research.validation import ValidationError

DATASET_NAME = "equity_price_strength_rebalance_feasibility"
REBALANCE_FREQUENCIES = {"monthly"}
REQUIRED_SCORECARD_COLUMNS = [
    "symbol",
    "date",
    "price_strength_scorecard_bucket",
    "price_strength_score_v0",
    "is_scorecard_bucket_eligible",
]

OUTPUT_FILE_PURPOSES = {
    "equity_price_strength_rebalance_bucket_counts.parquet": "Canonical bucket counts by rebalance date.",
    "equity_price_strength_rebalance_bucket_counts.csv": "Convenience export for bucket counts by rebalance date.",
    "equity_price_strength_rebalance_bucket_count_summary.parquet": "Canonical bucket count distribution summary.",
    "equity_price_strength_rebalance_bucket_count_summary.csv": "Convenience export for bucket count summary.",
    "equity_price_strength_rebalance_sector_composition.parquet": "Canonical sector composition by rebalance date and bucket.",
    "equity_price_strength_rebalance_sector_composition.csv": "Convenience export for sector composition.",
    "equity_price_strength_rebalance_sector_concentration.parquet": "Canonical sector concentration metrics by rebalance date and bucket.",
    "equity_price_strength_rebalance_sector_concentration.csv": "Convenience export for sector concentration metrics.",
    "equity_price_strength_rebalance_turnover.parquet": "Canonical bucket membership change between consecutive rebalance dates.",
    "equity_price_strength_rebalance_turnover.csv": "Convenience export for bucket turnover diagnostics.",
    "equity_price_strength_rebalance_feasibility.parquet": "Canonical bucket-level rebalance feasibility assessment.",
    "equity_price_strength_rebalance_feasibility.csv": "Convenience export for rebalance feasibility assessment.",
    "equity_price_strength_rebalance_feasibility_report.md": "Human-readable rebalance feasibility report.",
    "equity_price_strength_rebalance_feasibility.metadata.json": "Documents inputs, outputs, assumptions, and generation metadata.",
}


def build_price_strength_rebalance_feasibility(
    data_root: Path,
    *,
    rebalance_frequency: str = "monthly",
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    if rebalance_frequency not in REBALANCE_FREQUENCIES:
        raise ValidationError(f"Unsupported rebalance frequency: {rebalance_frequency}")

    scorecard_path = price_strength_scorecard_path(data_root)
    scorecard_columns = _scorecard_columns(scorecard_path)
    scorecard = read_parquet(scorecard_path, columns=scorecard_columns)
    scorecard, sector_metadata = attach_sector(scorecard, data_root)
    filtered = prepare_rebalance_input(scorecard, start_date=start_date, end_date=end_date)
    rebalance_frame = select_rebalance_rows(filtered, rebalance_frequency=rebalance_frequency)
    if rebalance_frame.empty:
        raise ValidationError("No eligible scorecard rows found for rebalance feasibility diagnostics")

    bucket_counts = compute_bucket_counts(rebalance_frame)
    bucket_count_summary = compute_bucket_count_summary(bucket_counts)
    turnover = compute_turnover(rebalance_frame)
    sector_available = bool("sector" in rebalance_frame.columns and rebalance_frame["sector"].notna().any())
    if sector_available:
        sector_composition = compute_sector_composition(rebalance_frame)
        sector_concentration = compute_sector_concentration(sector_composition)
    else:
        sector_composition = pd.DataFrame()
        sector_concentration = pd.DataFrame()
    feasibility = compute_feasibility(
        bucket_count_summary,
        turnover,
        sector_concentration,
        sector_available=sector_available,
    )

    paths = _output_paths(data_root)
    write_parquet(bucket_counts, paths["bucket_counts_parquet"])
    write_csv(bucket_counts, paths["bucket_counts_csv"])
    write_parquet(bucket_count_summary, paths["bucket_count_summary_parquet"])
    write_csv(bucket_count_summary, paths["bucket_count_summary_csv"])
    if sector_available:
        write_parquet(sector_composition, paths["sector_composition_parquet"])
        write_csv(sector_composition, paths["sector_composition_csv"])
        write_parquet(sector_concentration, paths["sector_concentration_parquet"])
        write_csv(sector_concentration, paths["sector_concentration_csv"])
    write_parquet(turnover, paths["turnover_parquet"])
    write_csv(turnover, paths["turnover_csv"])
    write_parquet(feasibility, paths["feasibility_parquet"])
    write_csv(feasibility, paths["feasibility_csv"])
    write_markdown_report(
        paths["markdown_report"],
        bucket_count_summary=bucket_count_summary,
        feasibility=feasibility,
        turnover=turnover,
        sector_concentration=sector_concentration,
        sector_available=sector_available,
        summary={
            "rebalance_frequency": rebalance_frequency,
            "rebalance_date_count": int(rebalance_frame["rebalance_date"].nunique()),
            "start_date": str(rebalance_frame["rebalance_date"].min()),
            "end_date": str(rebalance_frame["rebalance_date"].max()),
        },
    )
    input_paths = [scorecard_path]
    reference_path = reference_tickers_path(data_root)
    if sector_metadata.get("source") == "reference/tickers.parquet":
        input_paths.append(reference_path)
    metadata_path = write_metadata(
        dataset_name=DATASET_NAME,
        output_path=paths["feasibility_parquet"],
        input_paths=input_paths,
        dataframe=feasibility,
        extra_metadata={
            "dataset_type": "research_rebalance_feasibility",
            "research_only": True,
            "output_paths": _metadata_output_paths(paths, sector_available=sector_available),
            "output_file_purposes": OUTPUT_FILE_PURPOSES,
            "rebalance_frequency": rebalance_frequency,
            "rebalance_date_rule": "Monthly rebalance date is the last available trading date in each calendar month.",
            "eligible_row_filter": "is_scorecard_bucket_eligible == true",
            "start_date": start_date,
            "end_date": end_date,
            "sector_availability": {"available": sector_available, **sector_metadata},
            "turnover_definition": "(added_symbol_count + removed_symbol_count) / max(previous_symbol_count + current_symbol_count, 1)",
            "feasibility_assessment_rules": _feasibility_rule_text(sector_available),
            "parquet_is_canonical": True,
            "csv_is_convenience_export": True,
        },
    )
    paths["metadata"] = metadata_path
    summary = {
        "rebalance_frequency": rebalance_frequency,
        "rebalance_date_count": int(rebalance_frame["rebalance_date"].nunique()),
        "eligible_rows": int(len(filtered)),
        "rebalance_rows": int(len(rebalance_frame)),
        "sector_available": sector_available,
    }
    return paths, summary


def _scorecard_columns(path: Path) -> list[str]:
    columns = parquet_columns(path)
    missing = [column for column in REQUIRED_SCORECARD_COLUMNS if column not in columns]
    if missing:
        raise ValidationError(f"scorecard missing required columns: {missing}")
    optional = ["sector"]
    return [column for column in [*REQUIRED_SCORECARD_COLUMNS, *optional] if column in columns]


def attach_sector(scorecard: pd.DataFrame, data_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "sector" in scorecard.columns and scorecard["sector"].notna().any():
        return scorecard, {"source": "scorecard", "reason": None}

    reference_path = reference_tickers_path(data_root)
    if not reference_path.exists():
        return scorecard, {"source": None, "reason": f"Missing reference file: {reference_path}"}
    columns = parquet_columns(reference_path)
    if "sector" not in columns or "ticker" not in columns:
        return scorecard, {"source": None, "reason": "reference/tickers.parquet lacks ticker and sector columns"}
    reference = read_parquet(reference_path, columns=["ticker", "sector"]).dropna(subset=["ticker"])
    reference["symbol"] = reference["ticker"].astype("string").str.upper()
    reference = reference.drop(columns=["ticker"]).drop_duplicates(subset=["symbol"], keep="first")
    joined = scorecard.merge(reference, on="symbol", how="left")
    source = "reference/tickers.parquet" if joined["sector"].notna().any() else None
    reason = None if source else "reference/tickers.parquet did not provide matched sectors"
    return joined, {"source": source, "reason": reason}


def prepare_rebalance_input(
    scorecard: pd.DataFrame,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    frame = scorecard.copy()
    frame["symbol"] = frame["symbol"].astype("string").str.upper()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame[frame["is_scorecard_bucket_eligible"].fillna(False)].copy()
    if start_date:
        frame = frame[frame["date"] >= pd.Timestamp(start_date)]
    if end_date:
        frame = frame[frame["date"] <= pd.Timestamp(end_date)]
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def select_rebalance_rows(scorecard: pd.DataFrame, *, rebalance_frequency: str = "monthly") -> pd.DataFrame:
    if rebalance_frequency != "monthly":
        raise ValidationError(f"Unsupported rebalance frequency: {rebalance_frequency}")
    frame = scorecard.copy()
    frame["month"] = frame["date"].dt.to_period("M")
    rebalance_dates = frame.groupby("month", sort=True)["date"].max().rename("rebalance_date")
    frame = frame.merge(rebalance_dates, on="month", how="left")
    frame = frame[frame["date"] == frame["rebalance_date"]].drop(columns=["month"])
    frame["rebalance_date"] = frame["rebalance_date"].dt.date
    frame["date"] = frame["date"].dt.date
    return frame.reset_index(drop=True)


def compute_bucket_counts(rebalance_rows: pd.DataFrame) -> pd.DataFrame:
    universe = rebalance_rows.groupby("rebalance_date", sort=True)["symbol"].nunique().rename("eligible_universe_count")
    counts = (
        rebalance_rows.groupby(["rebalance_date", "price_strength_scorecard_bucket"], sort=True)["symbol"]
        .nunique()
        .rename("symbol_count")
        .reset_index()
    )
    counts = counts.merge(universe, on="rebalance_date", how="left")
    counts["pct_of_eligible_universe"] = counts["symbol_count"] / counts["eligible_universe_count"]
    return counts.drop(columns=["eligible_universe_count"])


def compute_bucket_count_summary(bucket_counts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bucket, group in bucket_counts.groupby("price_strength_scorecard_bucket", sort=True):
        counts = pd.to_numeric(group["symbol_count"], errors="coerce")
        rows.append(
            {
                "price_strength_scorecard_bucket": bucket,
                "rebalance_date_count": int(group["rebalance_date"].nunique()),
                "mean_symbol_count": _finite_or_none(counts.mean()),
                "median_symbol_count": _finite_or_none(counts.median()),
                "min_symbol_count": int(counts.min()),
                "max_symbol_count": int(counts.max()),
                "p10_symbol_count": _finite_or_none(counts.quantile(0.10)),
                "p25_symbol_count": _finite_or_none(counts.quantile(0.25)),
                "p75_symbol_count": _finite_or_none(counts.quantile(0.75)),
                "p90_symbol_count": _finite_or_none(counts.quantile(0.90)),
                "mean_pct_of_eligible_universe": _finite_or_none(group["pct_of_eligible_universe"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("price_strength_scorecard_bucket").reset_index(drop=True)


def compute_sector_composition(rebalance_rows: pd.DataFrame) -> pd.DataFrame:
    frame = rebalance_rows.dropna(subset=["sector"]).copy()
    composition = (
        frame.groupby(["rebalance_date", "price_strength_scorecard_bucket", "sector"], sort=True)["symbol"]
        .nunique()
        .rename("symbol_count")
        .reset_index()
    )
    totals = (
        composition.groupby(["rebalance_date", "price_strength_scorecard_bucket"], sort=True)["symbol_count"]
        .sum()
        .rename("bucket_symbol_count")
        .reset_index()
    )
    composition = composition.merge(totals, on=["rebalance_date", "price_strength_scorecard_bucket"], how="left")
    composition["sector_share_within_bucket"] = composition["symbol_count"] / composition["bucket_symbol_count"]
    return composition.drop(columns=["bucket_symbol_count"])


def compute_sector_concentration(sector_composition: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (rebalance_date, bucket), group in sector_composition.groupby(
        ["rebalance_date", "price_strength_scorecard_bucket"], sort=True
    ):
        shares = pd.to_numeric(group["sector_share_within_bucket"], errors="coerce").sort_values(ascending=False)
        rows.append(
            {
                "rebalance_date": rebalance_date,
                "price_strength_scorecard_bucket": bucket,
                "sector_count": int(group["sector"].nunique()),
                "max_sector_share": _finite_or_none(shares.max()),
                "top_3_sector_share": _finite_or_none(shares.head(3).sum()),
                "herfindahl_sector_concentration": _finite_or_none((shares**2).sum()),
            }
        )
    return pd.DataFrame(rows)


def compute_turnover(rebalance_rows: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(rebalance_rows["rebalance_date"].drop_duplicates())
    buckets = sorted(rebalance_rows["price_strength_scorecard_bucket"].drop_duplicates())
    memberships = {
        (date, bucket): set(group["symbol"].astype(str))
        for (date, bucket), group in rebalance_rows.groupby(["rebalance_date", "price_strength_scorecard_bucket"], sort=True)
    }
    rows = []
    for previous_date, current_date in zip(dates, dates[1:]):
        for bucket in buckets:
            previous = memberships.get((previous_date, bucket), set())
            current = memberships.get((current_date, bucket), set())
            common = previous & current
            added = current - previous
            removed = previous - current
            denominator = max(len(previous) + len(current), 1)
            union = previous | current
            rows.append(
                {
                    "previous_rebalance_date": previous_date,
                    "rebalance_date": current_date,
                    "price_strength_scorecard_bucket": bucket,
                    "previous_symbol_count": len(previous),
                    "current_symbol_count": len(current),
                    "common_symbol_count": len(common),
                    "added_symbol_count": len(added),
                    "removed_symbol_count": len(removed),
                    "jaccard_similarity": len(common) / len(union) if union else None,
                    "turnover_rate": (len(added) + len(removed)) / denominator,
                }
            )
    return pd.DataFrame(rows)


def compute_feasibility(
    bucket_count_summary: pd.DataFrame,
    turnover: pd.DataFrame,
    sector_concentration: pd.DataFrame,
    *,
    sector_available: bool,
) -> pd.DataFrame:
    rows = []
    turnover_summary = turnover.groupby("price_strength_scorecard_bucket", sort=True)["turnover_rate"].median()
    if sector_available and not sector_concentration.empty:
        max_sector = sector_concentration.groupby("price_strength_scorecard_bucket", sort=True)["max_sector_share"].median()
        top3_sector = sector_concentration.groupby("price_strength_scorecard_bucket", sort=True)["top_3_sector_share"].median()
    else:
        max_sector = pd.Series(dtype="float64")
        top3_sector = pd.Series(dtype="float64")

    for _, row in bucket_count_summary.iterrows():
        bucket = row["price_strength_scorecard_bucket"]
        median_count = row["median_symbol_count"]
        p10_count = row["p10_symbol_count"]
        median_max_sector_share = _series_get(max_sector, bucket)
        median_top_3_sector_share = _series_get(top3_sector, bucket)
        median_turnover_rate = _series_get(turnover_summary, bucket)
        rows.append(
            {
                "price_strength_scorecard_bucket": bucket,
                "median_symbol_count": median_count,
                "p10_symbol_count": p10_count,
                "median_max_sector_share": median_max_sector_share,
                "median_top_3_sector_share": median_top_3_sector_share,
                "median_turnover_rate": median_turnover_rate,
                "feasibility_assessment": assess_feasibility(
                    median_symbol_count=median_count,
                    p10_symbol_count=p10_count,
                    median_max_sector_share=median_max_sector_share,
                    median_top_3_sector_share=median_top_3_sector_share,
                    median_turnover_rate=median_turnover_rate,
                    sector_available=sector_available,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("price_strength_scorecard_bucket").reset_index(drop=True)


def assess_feasibility(
    *,
    median_symbol_count: float | None,
    p10_symbol_count: float | None,
    median_max_sector_share: float | None,
    median_top_3_sector_share: float | None,
    median_turnover_rate: float | None,
    sector_available: bool,
) -> str:
    median_count = median_symbol_count or 0
    p10_count = p10_symbol_count or 0
    turnover = median_turnover_rate or 0
    if median_count < 10 or p10_count < 5:
        return "too_sparse"
    if median_count < 25 or p10_count < 10:
        return "sparse_but_usable"
    if turnover > 0.75:
        return "high_turnover"
    if sector_available and ((median_max_sector_share or 0) > 0.40 or (median_top_3_sector_share or 0) > 0.75):
        return "sector_concentrated"
    if median_count >= 50 and p10_count >= 25 and turnover <= 0.60:
        return "feasible_basket_candidate"
    return "needs_review"


def write_markdown_report(
    path: Path,
    *,
    bucket_count_summary: pd.DataFrame,
    feasibility: pd.DataFrame,
    turnover: pd.DataFrame,
    sector_concentration: pd.DataFrame,
    sector_available: bool,
    summary: dict[str, Any],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Equity Price Strength Rebalance Feasibility",
        "",
        "## Purpose",
        "",
        "Evaluate whether scorecard v0 buckets are operationally usable as rebalance-date baskets before any holding-period simulation.",
        "",
        "## Rebalance Method",
        "",
        f"- Frequency: {summary['rebalance_frequency']}",
        "- Monthly rebalance date: last available trading date in each calendar month.",
        "- Eligible rows: `is_scorecard_bucket_eligible=true`.",
        f"- Rebalance dates: {summary['rebalance_date_count']} from {summary['start_date']} to {summary['end_date']}.",
        "",
        "## Executive Summary",
        "",
        _markdown_table(_feasibility_rows(feasibility)),
        "",
        "## Bucket Count Summary",
        "",
        _markdown_table(_count_summary_rows(bucket_count_summary)),
        "",
        "## Positive Bucket Feasibility",
        "",
        _markdown_table(_bucket_rows(feasibility, ["higher_conviction_price_strength", "price_strength_candidate"])),
        "",
        "## Risk Bucket Feasibility",
        "",
        _markdown_table(_bucket_rows(feasibility, ["high_volatility_trap"])),
        "",
    ]
    if sector_available:
        lines.extend(
            [
                "## Sector Concentration",
                "",
                _markdown_table(_sector_rows(sector_concentration)),
                "",
            ]
        )
    else:
        lines.extend(["## Sector Concentration", "", "Sector diagnostics were skipped because sector was unavailable.", ""])
    lines.extend(
        [
            "## Turnover Summary",
            "",
            _markdown_table(_turnover_rows(turnover)),
            "",
            "## Important Caveats",
            "",
            "- This does not test returns, P&L, execution, or portfolio construction.",
            "- This only tests whether scorecard buckets are operationally feasible as rebalance-date baskets.",
            "- The main question is whether `higher_conviction_price_strength` has enough names and acceptable concentration/turnover.",
            "",
            "## Output File Guide",
            "",
            _markdown_table(_output_file_guide_rows(sector_available=sector_available)),
            "",
            "## Suggested Next Step",
            "",
            "If feasibility is acceptable, the next research step is a simple monthly rebalance / 63-trading-day holding-period simulation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _output_paths(data_root: Path) -> dict[str, Path]:
    return {
        "bucket_counts_parquet": price_strength_rebalance_bucket_counts_path(data_root),
        "bucket_counts_csv": price_strength_rebalance_bucket_counts_csv_path(data_root),
        "bucket_count_summary_parquet": price_strength_rebalance_bucket_count_summary_path(data_root),
        "bucket_count_summary_csv": price_strength_rebalance_bucket_count_summary_csv_path(data_root),
        "sector_composition_parquet": price_strength_rebalance_sector_composition_path(data_root),
        "sector_composition_csv": price_strength_rebalance_sector_composition_csv_path(data_root),
        "sector_concentration_parquet": price_strength_rebalance_sector_concentration_path(data_root),
        "sector_concentration_csv": price_strength_rebalance_sector_concentration_csv_path(data_root),
        "turnover_parquet": price_strength_rebalance_turnover_path(data_root),
        "turnover_csv": price_strength_rebalance_turnover_csv_path(data_root),
        "feasibility_parquet": price_strength_rebalance_feasibility_path(data_root),
        "feasibility_csv": price_strength_rebalance_feasibility_csv_path(data_root),
        "markdown_report": price_strength_rebalance_feasibility_report_path(data_root),
    }


def _metadata_output_paths(paths: dict[str, Path], *, sector_available: bool) -> dict[str, str]:
    output_keys = [
        "bucket_counts_parquet",
        "bucket_counts_csv",
        "bucket_count_summary_parquet",
        "bucket_count_summary_csv",
        "turnover_parquet",
        "turnover_csv",
        "feasibility_parquet",
        "feasibility_csv",
        "markdown_report",
    ]
    if sector_available:
        output_keys.extend(
            [
                "sector_composition_parquet",
                "sector_composition_csv",
                "sector_concentration_parquet",
                "sector_concentration_csv",
            ]
        )
    outputs = {key: str(paths[key]) for key in output_keys}
    outputs["metadata"] = str(paths["feasibility_parquet"].with_suffix(".metadata.json"))
    return outputs


def _feasibility_rule_text(sector_available: bool) -> dict[str, str]:
    rules = {
        "too_sparse": "median count < 10 or p10 count < 5",
        "sparse_but_usable": "median count < 25 or p10 count < 10",
        "high_turnover": "median turnover > 0.75",
        "feasible_basket_candidate": "median count >= 50, p10 count >= 25, and median turnover <= 0.60",
        "needs_review": "does not match a more specific label",
    }
    if sector_available:
        rules["sector_concentrated"] = "median max-sector share > 0.40 or median top-3-sector share > 0.75"
    return rules


def _feasibility_rows(feasibility: pd.DataFrame) -> pd.DataFrame:
    return feasibility.sort_values("price_strength_scorecard_bucket")


def _count_summary_rows(bucket_count_summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "price_strength_scorecard_bucket",
        "rebalance_date_count",
        "median_symbol_count",
        "p10_symbol_count",
        "mean_pct_of_eligible_universe",
    ]
    return bucket_count_summary[columns].sort_values("price_strength_scorecard_bucket")


def _bucket_rows(feasibility: pd.DataFrame, buckets: list[str]) -> pd.DataFrame:
    rows = feasibility[feasibility["price_strength_scorecard_bucket"].isin(buckets)].copy()
    return rows if not rows.empty else pd.DataFrame(columns=feasibility.columns)


def _sector_rows(sector_concentration: pd.DataFrame) -> pd.DataFrame:
    if sector_concentration.empty:
        return pd.DataFrame()
    rows = sector_concentration.copy()
    rows["_score"] = rows["price_strength_scorecard_bucket"].map(SCORE_VALUES).fillna(0)
    return rows.sort_values(["rebalance_date", "_score"], ascending=[False, False]).head(20).drop(columns=["_score"])


def _turnover_rows(turnover: pd.DataFrame) -> pd.DataFrame:
    if turnover.empty:
        return pd.DataFrame()
    return (
        turnover.groupby("price_strength_scorecard_bucket", sort=True)["turnover_rate"]
        .agg(median_turnover_rate="median", p75_turnover_rate=lambda values: values.quantile(0.75))
        .reset_index()
    )


def _output_file_guide_rows(sector_available: bool) -> pd.DataFrame:
    skipped = set()
    if not sector_available:
        skipped = {
            "equity_price_strength_rebalance_sector_composition.parquet",
            "equity_price_strength_rebalance_sector_composition.csv",
            "equity_price_strength_rebalance_sector_concentration.parquet",
            "equity_price_strength_rebalance_sector_concentration.csv",
        }
    return pd.DataFrame(
        [
            {"File": filename, "Purpose": purpose, "Written?": "No" if filename in skipped else "Yes"}
            for filename, purpose in OUTPUT_FILE_PURPOSES.items()
        ]
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_None._"
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
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


def _series_get(series: pd.Series, key: str) -> float | None:
    if key not in series.index:
        return None
    return _finite_or_none(series.loc[key])


def _finite_or_none(value: Any) -> float | None:
    if pd.isna(value):
        return None
    numeric = float(value)
    if numeric == float("inf") or numeric == float("-inf"):
        return None
    return numeric


def terminal_summary(paths: dict[str, Path], summary: dict[str, Any]) -> str:
    lines = [
        "Price strength rebalance feasibility complete.",
        "",
        f"Rebalance frequency: {summary['rebalance_frequency']}",
        f"Eligible scorecard rows: {summary['eligible_rows']}",
        f"Rebalance rows: {summary['rebalance_rows']}",
        f"Rebalance dates: {summary['rebalance_date_count']}",
        f"Sector diagnostics available: {summary['sector_available']}",
        "",
        "Human-readable output:",
        f"- Markdown report: {paths['markdown_report']}",
        "",
        "Canonical machine-readable outputs:",
        f"- Bucket counts parquet: {paths['bucket_counts_parquet']}",
        f"- Bucket count summary parquet: {paths['bucket_count_summary_parquet']}",
        f"- Turnover parquet: {paths['turnover_parquet']}",
        f"- Feasibility parquet: {paths['feasibility_parquet']}",
        f"- Metadata JSON: {paths['metadata']}",
    ]
    if summary["sector_available"]:
        lines.extend(
            [
                f"- Sector composition parquet: {paths['sector_composition_parquet']}",
                f"- Sector concentration parquet: {paths['sector_concentration_parquet']}",
            ]
        )
    return "\n".join(lines)
