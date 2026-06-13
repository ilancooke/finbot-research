from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from finbot_research.config import (
    price_strength_holding_period_rebalance_results_csv_path,
    price_strength_holding_period_rebalance_results_path,
    price_strength_holding_period_metadata_path,
    price_strength_holding_period_report_path,
    price_strength_holding_period_sector_composition_csv_path,
    price_strength_holding_period_sector_composition_path,
    price_strength_holding_period_sector_concentration_csv_path,
    price_strength_holding_period_sector_concentration_path,
    price_strength_holding_period_summary_csv_path,
    price_strength_holding_period_summary_path,
    price_strength_holding_period_turnover_csv_path,
    price_strength_holding_period_turnover_path,
    price_strength_scorecard_path,
    reference_tickers_path,
)
from finbot_research.io import parquet_columns, read_parquet, write_csv, write_parquet
from finbot_research.metadata import write_metadata
from finbot_research.price_strength_rebalance_feasibility import attach_sector, select_rebalance_rows
from finbot_research.validation import ValidationError

DATASET_NAME = "equity_price_strength_holding_period_simulation"
REBALANCE_FREQUENCIES = {"monthly"}

PRIMARY_RETURN = "forward_63d_sector_relative_return"
TOP_FLAG = "forward_63d_top_30pct_sector_flag"
BOTTOM_FLAG = "forward_63d_bottom_30pct_sector_flag"

REQUIRED_COLUMNS = [
    "symbol",
    "date",
    "price_strength_scorecard_bucket",
    "price_strength_score_v0",
    "is_scorecard_bucket_eligible",
    PRIMARY_RETURN,
    TOP_FLAG,
    BOTTOM_FLAG,
]

BASKET_DEFINITIONS = {
    "higher_conviction_price_strength": ["higher_conviction_price_strength"],
    "positive_combined": ["higher_conviction_price_strength", "price_strength_candidate"],
    "momentum_resilience_candidate": ["momentum_resilience_candidate"],
    "high_volatility_trap": ["high_volatility_trap"],
    "eligible_universe_baseline": None,
}

OUTPUT_FILE_PURPOSES = {
    "equity_price_strength_holding_period_rebalance_results.parquet": "Canonical per-rebalance 63-trading-day basket outcome metrics.",
    "equity_price_strength_holding_period_rebalance_results.csv": "Convenience export for per-rebalance basket outcome metrics.",
    "equity_price_strength_holding_period_summary.parquet": "Canonical basket summary across rebalance dates.",
    "equity_price_strength_holding_period_summary.csv": "Convenience export for basket summary.",
    "equity_price_strength_holding_period_sector_composition.parquet": "Canonical sector exposure by rebalance date and basket.",
    "equity_price_strength_holding_period_sector_composition.csv": "Convenience export for sector exposure.",
    "equity_price_strength_holding_period_sector_concentration.parquet": "Canonical sector concentration by rebalance date and basket.",
    "equity_price_strength_holding_period_sector_concentration.csv": "Convenience export for sector concentration.",
    "equity_price_strength_holding_period_turnover.parquet": "Canonical basket membership change between consecutive rebalance dates.",
    "equity_price_strength_holding_period_turnover.csv": "Convenience export for turnover diagnostics.",
    "equity_price_strength_holding_period_simulation_report.md": "Human-readable holding-period simulation report.",
    "equity_price_strength_holding_period_simulation.metadata.json": "Documents inputs, outputs, assumptions, and generation metadata.",
}


def build_price_strength_holding_period_simulation(
    data_root: Path,
    *,
    rebalance_frequency: str = "monthly",
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    if rebalance_frequency not in REBALANCE_FREQUENCIES:
        raise ValidationError(f"Unsupported rebalance frequency: {rebalance_frequency}")

    scorecard_path = price_strength_scorecard_path(data_root)
    scorecard = read_parquet(scorecard_path, columns=_scorecard_columns(scorecard_path))
    scorecard, sector_metadata = attach_sector(scorecard, data_root)
    eligible = prepare_simulation_input(scorecard, start_date=start_date, end_date=end_date)
    rebalance_rows = select_rebalance_rows(eligible, rebalance_frequency=rebalance_frequency)
    if rebalance_rows.empty:
        raise ValidationError("No eligible scorecard rows found for holding-period simulation")

    basket_rows = build_basket_memberships(rebalance_rows)
    rebalance_results = compute_rebalance_results(basket_rows)
    summary = compute_summary(rebalance_results)
    turnover = compute_turnover(basket_rows)
    sector_available = bool("sector" in basket_rows.columns and basket_rows["sector"].notna().any())
    if sector_available:
        sector_composition = compute_sector_composition(basket_rows)
        sector_concentration = compute_sector_concentration(sector_composition)
    else:
        sector_composition = pd.DataFrame()
        sector_concentration = pd.DataFrame()

    paths = _output_paths(data_root)
    write_parquet(rebalance_results, paths["rebalance_results_parquet"])
    write_csv(rebalance_results, paths["rebalance_results_csv"])
    write_parquet(summary, paths["summary_parquet"])
    write_csv(summary, paths["summary_csv"])
    if sector_available:
        write_parquet(sector_composition, paths["sector_composition_parquet"])
        write_csv(sector_composition, paths["sector_composition_csv"])
        write_parquet(sector_concentration, paths["sector_concentration_parquet"])
        write_csv(sector_concentration, paths["sector_concentration_csv"])
    write_parquet(turnover, paths["turnover_parquet"])
    write_csv(turnover, paths["turnover_csv"])
    write_markdown_report(
        paths["markdown_report"],
        rebalance_results=rebalance_results,
        summary=summary,
        turnover=turnover,
        sector_concentration=sector_concentration,
        sector_available=sector_available,
        run_summary={
            "rebalance_frequency": rebalance_frequency,
            "rebalance_date_count": int(rebalance_rows["rebalance_date"].nunique()),
            "start_date": str(rebalance_rows["rebalance_date"].min()),
            "end_date": str(rebalance_rows["rebalance_date"].max()),
        },
    )

    input_paths = [scorecard_path]
    if sector_metadata.get("source") == "reference/tickers.parquet":
        input_paths.append(reference_tickers_path(data_root))
    metadata_path = write_metadata(
        dataset_name=DATASET_NAME,
        output_path=paths["summary_parquet"],
        input_paths=input_paths,
        dataframe=summary,
        extra_metadata={
            "dataset_type": "research_holding_period_outcome_simulation",
            "research_only": True,
            "output_paths": _metadata_output_paths(paths, sector_available=sector_available),
            "output_file_purposes": OUTPUT_FILE_PURPOSES,
            "basket_definitions": BASKET_DEFINITIONS,
            "rebalance_frequency": rebalance_frequency,
            "rebalance_date_rule": "Monthly rebalance date is the last available trading date in each calendar month.",
            "eligible_row_filter": "is_scorecard_bucket_eligible == true",
            "baseline_definition": "eligible_universe_baseline is all eligible rows on the same rebalance date.",
            "sector_availability": {"available": sector_available, **sector_metadata},
            "turnover_definition": "(added_symbol_count + removed_symbol_count) / max(previous_symbol_count + current_symbol_count, 1)",
            "limitations": [
                "This is not a full portfolio backtest.",
                "No transaction costs, slippage, position sizing, overlapping portfolios, or equity curves are modeled.",
                "high_volatility_trap is included as a risk-bucket comparison, not as a long recommendation.",
            ],
            "parquet_is_canonical": True,
            "csv_is_convenience_export": True,
        },
    )
    desired_metadata_path = paths["metadata"]
    if metadata_path != desired_metadata_path:
        desired_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.replace(desired_metadata_path)
        metadata_path = desired_metadata_path
    paths["metadata"] = metadata_path
    return paths, {
        "rebalance_frequency": rebalance_frequency,
        "eligible_rows": int(len(eligible)),
        "rebalance_rows": int(len(rebalance_rows)),
        "basket_rows": int(len(basket_rows)),
        "rebalance_date_count": int(rebalance_rows["rebalance_date"].nunique()),
        "sector_available": sector_available,
    }


def _scorecard_columns(path: Path) -> list[str]:
    columns = parquet_columns(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing:
        raise ValidationError(f"scorecard missing required holding-period columns: {missing}")
    return [column for column in [*REQUIRED_COLUMNS, "sector"] if column in columns]


def prepare_simulation_input(
    scorecard: pd.DataFrame,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    frame = scorecard.copy()
    frame["symbol"] = frame["symbol"].astype("string").str.upper()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame[frame["is_scorecard_bucket_eligible"].fillna(False)].copy()
    frame = frame.dropna(subset=[PRIMARY_RETURN, TOP_FLAG, BOTTOM_FLAG])
    if start_date:
        frame = frame[frame["date"] >= pd.Timestamp(start_date)]
    if end_date:
        frame = frame[frame["date"] <= pd.Timestamp(end_date)]
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def build_basket_memberships(rebalance_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for basket_name, buckets in BASKET_DEFINITIONS.items():
        if buckets is None:
            basket = rebalance_rows.copy()
        else:
            basket = rebalance_rows[rebalance_rows["price_strength_scorecard_bucket"].isin(buckets)].copy()
        if basket.empty:
            continue
        basket["basket_name"] = basket_name
        rows.append(basket)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def compute_rebalance_results(basket_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (rebalance_date, basket_name), group in basket_rows.groupby(["rebalance_date", "basket_name"], sort=True):
        rows.append({"rebalance_date": rebalance_date, "basket_name": basket_name, **_metric_fields(group)})
    result = pd.DataFrame(rows)
    baseline = result[result["basket_name"] == "eligible_universe_baseline"][
        [
            "rebalance_date",
            "avg_forward_63d_sector_relative_return",
            "median_forward_63d_sector_relative_return",
            "top_30pct_sector_flag_rate",
            "bottom_30pct_sector_flag_rate",
        ]
    ].rename(
        columns={
            "avg_forward_63d_sector_relative_return": "baseline_avg_forward_63d_sector_relative_return",
            "median_forward_63d_sector_relative_return": "baseline_median_forward_63d_sector_relative_return",
            "top_30pct_sector_flag_rate": "baseline_top_30pct_sector_flag_rate",
            "bottom_30pct_sector_flag_rate": "baseline_bottom_30pct_sector_flag_rate",
        }
    )
    result = result.merge(baseline, on="rebalance_date", how="left")
    result["avg_forward_return_vs_baseline"] = (
        result["avg_forward_63d_sector_relative_return"] - result["baseline_avg_forward_63d_sector_relative_return"]
    )
    result["median_forward_return_vs_baseline"] = (
        result["median_forward_63d_sector_relative_return"] - result["baseline_median_forward_63d_sector_relative_return"]
    )
    result["top_30pct_flag_rate_vs_baseline"] = (
        result["top_30pct_sector_flag_rate"] - result["baseline_top_30pct_sector_flag_rate"]
    )
    result["bottom_30pct_flag_rate_vs_baseline"] = (
        result["bottom_30pct_sector_flag_rate"] - result["baseline_bottom_30pct_sector_flag_rate"]
    )
    baseline_mask = result["basket_name"] == "eligible_universe_baseline"
    comparison_columns = [
        "avg_forward_return_vs_baseline",
        "median_forward_return_vs_baseline",
        "top_30pct_flag_rate_vs_baseline",
        "bottom_30pct_flag_rate_vs_baseline",
    ]
    result.loc[baseline_mask, comparison_columns] = 0.0
    return result.drop(
        columns=[
            "baseline_avg_forward_63d_sector_relative_return",
            "baseline_median_forward_63d_sector_relative_return",
            "baseline_top_30pct_sector_flag_rate",
            "baseline_bottom_30pct_sector_flag_rate",
        ]
    )


def compute_summary(rebalance_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for basket_name, group in rebalance_results.groupby("basket_name", sort=True):
        counts = pd.to_numeric(group["symbol_count"], errors="coerce")
        rows.append(
            {
                "basket_name": basket_name,
                "rebalance_date_count": int(group["rebalance_date"].nunique()),
                "mean_symbol_count": _finite_or_none(counts.mean()),
                "median_symbol_count": _finite_or_none(counts.median()),
                "p10_symbol_count": _finite_or_none(counts.quantile(0.10)),
                "p25_symbol_count": _finite_or_none(counts.quantile(0.25)),
                "p75_symbol_count": _finite_or_none(counts.quantile(0.75)),
                "p90_symbol_count": _finite_or_none(counts.quantile(0.90)),
                "mean_avg_forward_63d_sector_relative_return": _finite_or_none(
                    group["avg_forward_63d_sector_relative_return"].mean()
                ),
                "median_avg_forward_63d_sector_relative_return": _finite_or_none(
                    group["avg_forward_63d_sector_relative_return"].median()
                ),
                "mean_median_forward_63d_sector_relative_return": _finite_or_none(
                    group["median_forward_63d_sector_relative_return"].mean()
                ),
                "mean_avg_forward_return_vs_baseline": _finite_or_none(group["avg_forward_return_vs_baseline"].mean()),
                "median_avg_forward_return_vs_baseline": _finite_or_none(
                    group["avg_forward_return_vs_baseline"].median()
                ),
                "mean_median_forward_return_vs_baseline": _finite_or_none(
                    group["median_forward_return_vs_baseline"].mean()
                ),
                "mean_top_30pct_sector_flag_rate": _finite_or_none(group["top_30pct_sector_flag_rate"].mean()),
                "mean_top_30pct_flag_rate_vs_baseline": _finite_or_none(
                    group["top_30pct_flag_rate_vs_baseline"].mean()
                ),
                "mean_bottom_30pct_sector_flag_rate": _finite_or_none(group["bottom_30pct_sector_flag_rate"].mean()),
                "mean_bottom_30pct_flag_rate_vs_baseline": _finite_or_none(
                    group["bottom_30pct_flag_rate_vs_baseline"].mean()
                ),
                "pct_rebalances_avg_above_baseline": _share_positive(group["avg_forward_return_vs_baseline"]),
                "pct_rebalances_median_above_baseline": _share_positive(group["median_forward_return_vs_baseline"]),
                "pct_rebalances_top_rate_above_baseline": _share_positive(group["top_30pct_flag_rate_vs_baseline"]),
                "pct_rebalances_bottom_rate_below_baseline": _share_negative(group["bottom_30pct_flag_rate_vs_baseline"]),
            }
        )
    return pd.DataFrame(rows)


def compute_sector_composition(basket_rows: pd.DataFrame) -> pd.DataFrame:
    frame = basket_rows.dropna(subset=["sector"]).copy()
    composition = (
        frame.groupby(["rebalance_date", "basket_name", "sector"], sort=True)["symbol"]
        .nunique()
        .rename("symbol_count")
        .reset_index()
    )
    totals = (
        composition.groupby(["rebalance_date", "basket_name"], sort=True)["symbol_count"]
        .sum()
        .rename("basket_symbol_count")
        .reset_index()
    )
    composition = composition.merge(totals, on=["rebalance_date", "basket_name"], how="left")
    composition["sector_share_within_basket"] = composition["symbol_count"] / composition["basket_symbol_count"]
    return composition.drop(columns=["basket_symbol_count"])


def compute_sector_concentration(sector_composition: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (rebalance_date, basket_name), group in sector_composition.groupby(["rebalance_date", "basket_name"], sort=True):
        shares = pd.to_numeric(group["sector_share_within_basket"], errors="coerce").sort_values(ascending=False)
        rows.append(
            {
                "rebalance_date": rebalance_date,
                "basket_name": basket_name,
                "sector_count": int(group["sector"].nunique()),
                "max_sector_share": _finite_or_none(shares.max()),
                "top_3_sector_share": _finite_or_none(shares.head(3).sum()),
                "herfindahl_sector_concentration": _finite_or_none((shares**2).sum()),
            }
        )
    return pd.DataFrame(rows)


def compute_turnover(basket_rows: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(basket_rows["rebalance_date"].drop_duplicates())
    baskets = sorted(basket_rows["basket_name"].drop_duplicates())
    memberships = {
        (date, basket): set(group["symbol"].astype(str))
        for (date, basket), group in basket_rows.groupby(["rebalance_date", "basket_name"], sort=True)
    }
    rows = []
    for previous_date, current_date in zip(dates, dates[1:]):
        for basket in baskets:
            previous = memberships.get((previous_date, basket), set())
            current = memberships.get((current_date, basket), set())
            common = previous & current
            added = current - previous
            removed = previous - current
            denominator = max(len(previous) + len(current), 1)
            union = previous | current
            rows.append(
                {
                    "previous_rebalance_date": previous_date,
                    "rebalance_date": current_date,
                    "basket_name": basket,
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


def write_markdown_report(
    path: Path,
    *,
    rebalance_results: pd.DataFrame,
    summary: pd.DataFrame,
    turnover: pd.DataFrame,
    sector_concentration: pd.DataFrame,
    sector_available: bool,
    run_summary: dict[str, Any],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Equity Price Strength Holding-Period Simulation",
        "",
        "## Purpose",
        "",
        "Evaluate monthly rebalance scorecard buckets using 63-trading-day forward outcomes.",
        "",
        "## Method",
        "",
        f"- Frequency: {run_summary['rebalance_frequency']}",
        "- Monthly rebalance date: last available trading date in each calendar month.",
        "- Eligible rows: `is_scorecard_bucket_eligible=true` with non-null 63-trading-day forward labels.",
        f"- Rebalance dates: {run_summary['rebalance_date_count']} from {run_summary['start_date']} to {run_summary['end_date']}.",
        "",
        "## Executive Summary",
        "",
        _markdown_table(_summary_rows(summary)),
        "",
        "## Basket Performance Summary",
        "",
        _markdown_table(_summary_rows(summary)),
        "",
        "## Positive Basket Results",
        "",
        _markdown_table(_basket_rows(summary, ["higher_conviction_price_strength", "positive_combined"])),
        "",
        "## Risk Bucket Results",
        "",
        _markdown_table(_basket_rows(summary, ["high_volatility_trap"])),
        "",
        "## Sector Exposure",
        "",
        _markdown_table(_sector_rows(sector_concentration)) if sector_available else "Sector exposure was skipped because sector was unavailable.",
        "",
        "## Turnover",
        "",
        _markdown_table(_turnover_rows(turnover)),
        "",
        "## Important Caveats",
        "",
        "- This is not a full portfolio backtest.",
        "- It does not model transaction costs, slippage, position sizing, overlapping portfolios, or realistic equity curves.",
        "- `high_volatility_trap` is a risk-bucket comparison, not a long recommendation.",
        "",
        "## Output File Guide",
        "",
        _markdown_table(_output_file_guide_rows(sector_available)),
        "",
        "## Suggested Next Step",
        "",
        "If `higher_conviction_price_strength` looks good but has high turnover/concentration, the next step is a more realistic portfolio simulation with sector caps and transaction-cost assumptions.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _metric_fields(group: pd.DataFrame) -> dict[str, Any]:
    return {
        "symbol_count": int(group["symbol"].nunique()),
        "avg_forward_63d_sector_relative_return": _finite_or_none(pd.to_numeric(group[PRIMARY_RETURN], errors="coerce").mean()),
        "median_forward_63d_sector_relative_return": _finite_or_none(
            pd.to_numeric(group[PRIMARY_RETURN], errors="coerce").median()
        ),
        "top_30pct_sector_flag_rate": _finite_or_none(pd.to_numeric(group[TOP_FLAG], errors="coerce").mean()),
        "bottom_30pct_sector_flag_rate": _finite_or_none(pd.to_numeric(group[BOTTOM_FLAG], errors="coerce").mean()),
    }


def _output_paths(data_root: Path) -> dict[str, Path]:
    return {
        "rebalance_results_parquet": price_strength_holding_period_rebalance_results_path(data_root),
        "rebalance_results_csv": price_strength_holding_period_rebalance_results_csv_path(data_root),
        "summary_parquet": price_strength_holding_period_summary_path(data_root),
        "summary_csv": price_strength_holding_period_summary_csv_path(data_root),
        "sector_composition_parquet": price_strength_holding_period_sector_composition_path(data_root),
        "sector_composition_csv": price_strength_holding_period_sector_composition_csv_path(data_root),
        "sector_concentration_parquet": price_strength_holding_period_sector_concentration_path(data_root),
        "sector_concentration_csv": price_strength_holding_period_sector_concentration_csv_path(data_root),
        "turnover_parquet": price_strength_holding_period_turnover_path(data_root),
        "turnover_csv": price_strength_holding_period_turnover_csv_path(data_root),
        "markdown_report": price_strength_holding_period_report_path(data_root),
        "metadata": price_strength_holding_period_metadata_path(data_root),
    }


def _metadata_output_paths(paths: dict[str, Path], *, sector_available: bool) -> dict[str, str]:
    keys = [
        "rebalance_results_parquet",
        "rebalance_results_csv",
        "summary_parquet",
        "summary_csv",
        "turnover_parquet",
        "turnover_csv",
        "markdown_report",
    ]
    if sector_available:
        keys.extend(["sector_composition_parquet", "sector_composition_csv", "sector_concentration_parquet", "sector_concentration_csv"])
    outputs = {key: str(paths[key]) for key in keys}
    outputs["metadata"] = str(paths["metadata"])
    return outputs


def _summary_rows(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "basket_name",
        "rebalance_date_count",
        "median_symbol_count",
        "mean_avg_forward_return_vs_baseline",
        "mean_median_forward_return_vs_baseline",
        "mean_top_30pct_flag_rate_vs_baseline",
        "mean_bottom_30pct_flag_rate_vs_baseline",
        "pct_rebalances_avg_above_baseline",
    ]
    return summary[columns].sort_values("basket_name")


def _basket_rows(summary: pd.DataFrame, baskets: list[str]) -> pd.DataFrame:
    rows = summary[summary["basket_name"].isin(baskets)].copy()
    return rows if not rows.empty else pd.DataFrame(columns=summary.columns)


def _sector_rows(sector_concentration: pd.DataFrame) -> pd.DataFrame:
    if sector_concentration.empty:
        return pd.DataFrame()
    return sector_concentration.sort_values("rebalance_date", ascending=False).head(20)


def _turnover_rows(turnover: pd.DataFrame) -> pd.DataFrame:
    if turnover.empty:
        return pd.DataFrame()
    return (
        turnover.groupby("basket_name", sort=True)["turnover_rate"]
        .agg(median_turnover_rate="median", p75_turnover_rate=lambda values: values.quantile(0.75))
        .reset_index()
    )


def _output_file_guide_rows(sector_available: bool) -> pd.DataFrame:
    skipped = set()
    if not sector_available:
        skipped = {
            "equity_price_strength_holding_period_sector_composition.parquet",
            "equity_price_strength_holding_period_sector_composition.csv",
            "equity_price_strength_holding_period_sector_concentration.parquet",
            "equity_price_strength_holding_period_sector_concentration.csv",
        }
    return pd.DataFrame(
        [{"File": filename, "Purpose": purpose, "Written?": "No" if filename in skipped else "Yes"} for filename, purpose in OUTPUT_FILE_PURPOSES.items()]
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


def _share_positive(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float((numeric > 0).mean())


def _share_negative(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float((numeric < 0).mean())


def _finite_or_none(value: Any) -> float | None:
    if pd.isna(value):
        return None
    numeric = float(value)
    if numeric == float("inf") or numeric == float("-inf"):
        return None
    return numeric


def terminal_summary(paths: dict[str, Path], summary: dict[str, Any]) -> str:
    lines = [
        "Price strength holding-period simulation complete.",
        "",
        f"Rebalance frequency: {summary['rebalance_frequency']}",
        f"Eligible scorecard rows with labels: {summary['eligible_rows']}",
        f"Rebalance rows: {summary['rebalance_rows']}",
        f"Basket rows: {summary['basket_rows']}",
        f"Rebalance dates: {summary['rebalance_date_count']}",
        f"Sector exposure available: {summary['sector_available']}",
        "",
        "Human-readable output:",
        f"- Markdown report: {paths['markdown_report']}",
        "",
        "Canonical machine-readable outputs:",
        f"- Rebalance results parquet: {paths['rebalance_results_parquet']}",
        f"- Summary parquet: {paths['summary_parquet']}",
        f"- Turnover parquet: {paths['turnover_parquet']}",
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
