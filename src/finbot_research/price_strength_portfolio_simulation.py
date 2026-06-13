from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from finbot_research.config import (
    price_strength_portfolio_constituents_path,
    price_strength_portfolio_metadata_path,
    price_strength_portfolio_rebalance_results_csv_path,
    price_strength_portfolio_rebalance_results_path,
    price_strength_portfolio_report_path,
    price_strength_portfolio_sector_weights_csv_path,
    price_strength_portfolio_sector_weights_path,
    price_strength_portfolio_summary_csv_path,
    price_strength_portfolio_summary_path,
    price_strength_scorecard_path,
    reference_tickers_path,
)
from finbot_research.io import parquet_columns, read_parquet, write_csv, write_parquet
from finbot_research.metadata import write_metadata
from finbot_research.price_strength_rebalance_feasibility import attach_sector, select_rebalance_rows
from finbot_research.validation import ValidationError

DATASET_NAME = "equity_price_strength_portfolio_simulation"
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
PORTFOLIO_DEFINITIONS = {
    "higher_conviction_raw": ["higher_conviction_price_strength"],
    "higher_conviction_sector_capped": ["higher_conviction_price_strength"],
    "positive_combined_raw": ["higher_conviction_price_strength", "price_strength_candidate"],
    "positive_combined_sector_capped": ["higher_conviction_price_strength", "price_strength_candidate"],
    "eligible_universe_baseline": None,
}


def build_price_strength_portfolio_simulation(
    data_root: Path,
    *,
    rebalance_frequency: str = "monthly",
    start_date: str | None = None,
    end_date: str | None = None,
    sector_cap: float = 0.30,
    transaction_cost_bps: float = 25.0,
) -> tuple[dict[str, Path], dict[str, Any]]:
    if rebalance_frequency != "monthly":
        raise ValidationError(f"Unsupported rebalance frequency: {rebalance_frequency}")
    scorecard_path = price_strength_scorecard_path(data_root)
    scorecard = read_parquet(scorecard_path, columns=_scorecard_columns(scorecard_path))
    scorecard, sector_metadata = attach_sector(scorecard, data_root)
    if "sector" not in scorecard.columns or not scorecard["sector"].notna().any():
        raise ValidationError("price-strength-portfolio-simulation requires sector; scorecard/reference sector unavailable")
    eligible = prepare_portfolio_input(scorecard, start_date=start_date, end_date=end_date)
    rebalance_rows = select_rebalance_rows(eligible, rebalance_frequency=rebalance_frequency)
    constituents, sector_weights = build_portfolio_constituents(rebalance_rows, sector_cap=sector_cap)
    if constituents.empty:
        raise ValidationError("No portfolio constituents found")
    rebalance_results = compute_rebalance_results(
        constituents,
        sector_weights,
        sector_cap=sector_cap,
        transaction_cost_bps=transaction_cost_bps,
    )
    summary = compute_summary(rebalance_results)
    paths = _output_paths(data_root)
    write_parquet(rebalance_results, paths["rebalance_results_parquet"])
    write_csv(rebalance_results, paths["rebalance_results_csv"])
    write_parquet(summary, paths["summary_parquet"])
    write_csv(summary, paths["summary_csv"])
    write_parquet(constituents, paths["constituents_parquet"])
    write_parquet(sector_weights, paths["sector_weights_parquet"])
    write_csv(sector_weights, paths["sector_weights_csv"])
    write_markdown_report(paths["markdown_report"], summary=summary, run_summary={
        "rebalance_frequency": rebalance_frequency,
        "sector_cap": sector_cap,
        "transaction_cost_bps": transaction_cost_bps,
        "rebalance_date_count": int(rebalance_rows["rebalance_date"].nunique()),
    })
    input_paths = [scorecard_path]
    if sector_metadata.get("source") == "reference/tickers.parquet":
        input_paths.append(reference_tickers_path(data_root))
    metadata_path = write_metadata(
        dataset_name=DATASET_NAME,
        output_path=paths["summary_parquet"],
        input_paths=input_paths,
        dataframe=summary,
        extra_metadata={
            "dataset_type": "research_portfolio_simulation",
            "research_only": True,
            "output_paths": {key: str(value) for key, value in paths.items()},
            "rebalance_frequency": rebalance_frequency,
            "rebalance_date_rule": "Monthly rebalance date is the last available trading date in each calendar month.",
            "portfolio_definitions": PORTFOLIO_DEFINITIONS,
            "sector_cap": sector_cap,
            "sector_cap_method": "Cap sector-level weights, redistribute remaining weight across uncapped sectors proportionally, equal weight within sector.",
            "transaction_cost_bps": transaction_cost_bps,
            "turnover_definition": "0.5 * sum(abs(current_weight - previous_weight)); first rebalance uses one-way turnover of 1.0.",
            "return_definition": "Weighted average forward_63d_sector_relative_return less simple transaction cost for net return.",
            "baseline_definition": "eligible_universe_baseline equal-weights all eligible names on the same rebalance date.",
            "sector_availability": {"available": True, **sector_metadata},
            "limitations": [
                "Research simulation only, not production trading logic.",
                "Uses 63-trading-day forward labels, not an overlapping equity curve.",
                "No slippage model beyond simple transaction-cost bps.",
            ],
            "parquet_is_canonical": True,
            "csv_is_convenience_export": True,
        },
    )
    desired_metadata = paths["metadata"]
    if metadata_path != desired_metadata:
        metadata_path.replace(desired_metadata)
    paths["metadata"] = desired_metadata
    return paths, {
        "rebalance_frequency": rebalance_frequency,
        "rebalance_date_count": int(rebalance_rows["rebalance_date"].nunique()),
        "eligible_rows": int(len(eligible)),
        "constituent_rows": int(len(constituents)),
        "sector_cap": sector_cap,
        "transaction_cost_bps": transaction_cost_bps,
    }


def _scorecard_columns(path: Path) -> list[str]:
    columns = parquet_columns(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing:
        raise ValidationError(f"scorecard missing required portfolio columns: {missing}")
    return [column for column in [*REQUIRED_COLUMNS, "sector"] if column in columns]


def prepare_portfolio_input(scorecard: pd.DataFrame, *, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    frame = scorecard.copy()
    frame["symbol"] = frame["symbol"].astype("string").str.upper()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame[frame["is_scorecard_bucket_eligible"].fillna(False)].dropna(subset=[PRIMARY_RETURN, TOP_FLAG, BOTTOM_FLAG, "sector"])
    if start_date:
        frame = frame[frame["date"] >= pd.Timestamp(start_date)]
    if end_date:
        frame = frame[frame["date"] <= pd.Timestamp(end_date)]
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def build_portfolio_constituents(rebalance_rows: pd.DataFrame, *, sector_cap: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    constituent_frames = []
    sector_frames = []
    for (rebalance_date, portfolio_name), group in _portfolio_groups(rebalance_rows):
        capped = portfolio_name.endswith("_sector_capped")
        weights = sector_capped_weights(group, sector_cap=sector_cap) if capped else equal_weights(group)
        frame = group.copy()
        frame["portfolio_name"] = portfolio_name
        frame["weight"] = frame["symbol"].map(weights)
        frame = frame[frame["weight"] > 0].copy()
        constituent_frames.append(
            frame[
                [
                    "rebalance_date",
                    "portfolio_name",
                    "symbol",
                    "sector",
                    "weight",
                    "price_strength_scorecard_bucket",
                    PRIMARY_RETURN,
                    TOP_FLAG,
                    BOTTOM_FLAG,
                ]
            ]
        )
        sectors = frame.groupby("sector", sort=True).agg(sector_weight=("weight", "sum"), symbol_count=("symbol", "nunique")).reset_index()
        sectors["rebalance_date"] = rebalance_date
        sectors["portfolio_name"] = portfolio_name
        sector_frames.append(sectors[["rebalance_date", "portfolio_name", "sector", "sector_weight", "symbol_count"]])
    return pd.concat(constituent_frames, ignore_index=True), pd.concat(sector_frames, ignore_index=True)


def _portfolio_groups(rebalance_rows: pd.DataFrame):
    for rebalance_date, date_rows in rebalance_rows.groupby("rebalance_date", sort=True):
        for portfolio_name, buckets in PORTFOLIO_DEFINITIONS.items():
            group = date_rows if buckets is None else date_rows[date_rows["price_strength_scorecard_bucket"].isin(buckets)]
            if not group.empty:
                yield (rebalance_date, portfolio_name), group.copy()


def equal_weights(group: pd.DataFrame) -> dict[str, float]:
    symbols = sorted(group["symbol"].astype(str).unique())
    return {symbol: 1 / len(symbols) for symbol in symbols}


def sector_capped_weights(group: pd.DataFrame, *, sector_cap: float) -> dict[str, float]:
    sector_counts = group.groupby("sector", sort=True)["symbol"].nunique()
    sector_weights = pd.Series(1 / len(group), index=group.index).groupby(group["sector"]).sum()
    if sector_cap * len(sector_weights) < 1:
        capped = pd.Series(1 / len(sector_weights), index=sector_weights.index)
    else:
        capped = pd.Series(0.0, index=sector_weights.index)
        remaining_sectors = list(sector_weights.index)
        remaining_weight = 1.0
        while remaining_sectors:
            targets = sector_weights.loc[remaining_sectors] / sector_weights.loc[remaining_sectors].sum() * remaining_weight
            over = targets > sector_cap
            if not over.any():
                capped.loc[remaining_sectors] = targets
                break
            over_sectors = targets[over].index
            capped.loc[over_sectors] = sector_cap
            remaining_weight -= sector_cap * len(over_sectors)
            remaining_sectors = [sector for sector in remaining_sectors if sector not in set(over_sectors)]
    weights = {}
    for sector, sector_weight in capped.items():
        symbols = sorted(group.loc[group["sector"] == sector, "symbol"].astype(str).unique())
        for symbol in symbols:
            weights[symbol] = float(sector_weight / sector_counts.loc[sector])
    total = sum(weights.values())
    return {symbol: weight / total for symbol, weight in weights.items()}


def compute_rebalance_results(
    constituents: pd.DataFrame,
    sector_weights: pd.DataFrame,
    *,
    sector_cap: float,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    metrics = []
    previous_weights: dict[str, dict[str, float]] = {}
    for (rebalance_date, portfolio_name), group in constituents.groupby(["rebalance_date", "portfolio_name"], sort=True):
        current = dict(zip(group["symbol"].astype(str), group["weight"]))
        previous = previous_weights.get(portfolio_name, {})
        turnover = one_way_turnover(previous, current) if previous else 1.0
        previous_weights[portfolio_name] = current
        sectors = sector_weights[(sector_weights["rebalance_date"] == rebalance_date) & (sector_weights["portfolio_name"] == portfolio_name)]
        gross = float((group["weight"] * group[PRIMARY_RETURN]).sum())
        cost = turnover * transaction_cost_bps / 10000
        metrics.append({
            "rebalance_date": rebalance_date,
            "portfolio_name": portfolio_name,
            "symbol_count": int(group["symbol"].nunique()),
            "sector_count": int(sectors["sector"].nunique()),
            "max_sector_weight": float(sectors["sector_weight"].max()),
            "top_3_sector_weight": float(sectors["sector_weight"].sort_values(ascending=False).head(3).sum()),
            "sector_cap": sector_cap if portfolio_name.endswith("_sector_capped") else None,
            "sector_cap_binding": bool(portfolio_name.endswith("_sector_capped") and sectors["sector_weight"].max() >= sector_cap - 1e-9),
            "one_way_turnover": turnover,
            "transaction_cost_bps": transaction_cost_bps,
            "transaction_cost": cost,
            "portfolio_forward_63d_sector_relative_return": gross,
            "net_portfolio_forward_63d_sector_relative_return": gross - cost,
            "median_constituent_forward_63d_sector_relative_return": float(group[PRIMARY_RETURN].median()),
            "top_30pct_sector_flag_rate": float(group[TOP_FLAG].mean()),
            "bottom_30pct_sector_flag_rate": float(group[BOTTOM_FLAG].mean()),
        })
    result = pd.DataFrame(metrics)
    baseline = result[result["portfolio_name"] == "eligible_universe_baseline"][
        ["rebalance_date", "portfolio_forward_63d_sector_relative_return", "net_portfolio_forward_63d_sector_relative_return", "median_constituent_forward_63d_sector_relative_return", "top_30pct_sector_flag_rate", "bottom_30pct_sector_flag_rate"]
    ].rename(columns={
        "portfolio_forward_63d_sector_relative_return": "baseline_gross",
        "net_portfolio_forward_63d_sector_relative_return": "baseline_net",
        "median_constituent_forward_63d_sector_relative_return": "baseline_median",
        "top_30pct_sector_flag_rate": "baseline_top",
        "bottom_30pct_sector_flag_rate": "baseline_bottom",
    })
    result = result.merge(baseline, on="rebalance_date", how="left")
    result["portfolio_forward_return_vs_baseline"] = result["portfolio_forward_63d_sector_relative_return"] - result["baseline_gross"]
    result["net_portfolio_forward_return_vs_baseline"] = result["net_portfolio_forward_63d_sector_relative_return"] - result["baseline_net"]
    result["median_constituent_forward_return_vs_baseline"] = result["median_constituent_forward_63d_sector_relative_return"] - result["baseline_median"]
    result["top_30pct_flag_rate_vs_baseline"] = result["top_30pct_sector_flag_rate"] - result["baseline_top"]
    result["bottom_30pct_flag_rate_vs_baseline"] = result["bottom_30pct_sector_flag_rate"] - result["baseline_bottom"]
    return result.drop(columns=["baseline_gross", "baseline_net", "baseline_median", "baseline_top", "baseline_bottom"])


def one_way_turnover(previous: dict[str, float], current: dict[str, float]) -> float:
    symbols = set(previous) | set(current)
    return 0.5 * sum(abs(current.get(symbol, 0.0) - previous.get(symbol, 0.0)) for symbol in symbols)


def compute_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for portfolio_name, group in results.groupby("portfolio_name", sort=True):
        rows.append({
            "portfolio_name": portfolio_name,
            "rebalance_date_count": int(group["rebalance_date"].nunique()),
            "mean_symbol_count": float(group["symbol_count"].mean()),
            "median_symbol_count": float(group["symbol_count"].median()),
            "p10_symbol_count": float(group["symbol_count"].quantile(0.10)),
            "mean_sector_count": float(group["sector_count"].mean()),
            "median_max_sector_weight": float(group["max_sector_weight"].median()),
            "median_top_3_sector_weight": float(group["top_3_sector_weight"].median()),
            "pct_rebalances_sector_cap_binding": float(group["sector_cap_binding"].mean()),
            "mean_one_way_turnover": float(group["one_way_turnover"].mean()),
            "median_one_way_turnover": float(group["one_way_turnover"].median()),
            "mean_transaction_cost": float(group["transaction_cost"].mean()),
            "mean_gross_forward_return": float(group["portfolio_forward_63d_sector_relative_return"].mean()),
            "mean_net_forward_return": float(group["net_portfolio_forward_63d_sector_relative_return"].mean()),
            "median_gross_forward_return": float(group["portfolio_forward_63d_sector_relative_return"].median()),
            "median_net_forward_return": float(group["net_portfolio_forward_63d_sector_relative_return"].median()),
            "std_gross_forward_return": float(group["portfolio_forward_63d_sector_relative_return"].std()),
            "std_net_forward_return": float(group["net_portfolio_forward_63d_sector_relative_return"].std()),
            "max_drawdown_net_forward_return_index": max_drawdown(group.sort_values("rebalance_date")["net_portfolio_forward_63d_sector_relative_return"]),
            "mean_gross_vs_baseline": float(group["portfolio_forward_return_vs_baseline"].mean()),
            "mean_net_vs_baseline": float(group["net_portfolio_forward_return_vs_baseline"].mean()),
            "median_gross_vs_baseline": float(group["portfolio_forward_return_vs_baseline"].median()),
            "median_net_vs_baseline": float(group["net_portfolio_forward_return_vs_baseline"].median()),
            "pct_rebalances_gross_above_baseline": float((group["portfolio_forward_return_vs_baseline"] > 0).mean()),
            "pct_rebalances_net_above_baseline": float((group["net_portfolio_forward_return_vs_baseline"] > 0).mean()),
            "mean_top_30pct_flag_rate_vs_baseline": float(group["top_30pct_flag_rate_vs_baseline"].mean()),
            "mean_bottom_30pct_flag_rate_vs_baseline": float(group["bottom_30pct_flag_rate_vs_baseline"].mean()),
        })
    return pd.DataFrame(rows)


def max_drawdown(returns: pd.Series) -> float:
    numeric = pd.to_numeric(returns, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    index = (1 + numeric).cumprod()
    running_max = index.cummax()
    drawdown = index / running_max - 1
    return float(drawdown.min())


def write_markdown_report(path: Path, *, summary: pd.DataFrame, run_summary: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Equity Price Strength Portfolio Simulation",
        "",
        "## Purpose",
        "",
        "Test whether the higher-conviction price-strength bucket remains attractive after basic portfolio mechanics.",
        "",
        "## Method",
        "",
        f"- Monthly rebalance dates; sector cap {run_summary['sector_cap']}; transaction cost {run_summary['transaction_cost_bps']} bps.",
        "- Equal-weight raw portfolios; sector-capped portfolios redistribute weight across uncapped sectors.",
        "- First rebalance assumes one-way turnover of 1.0.",
        "",
        "## Portfolio Definitions",
        "",
        _markdown_table(pd.DataFrame([{"portfolio_name": key, "buckets": str(value)} for key, value in PORTFOLIO_DEFINITIONS.items()])),
        "",
        "## Executive Summary",
        "",
        _markdown_table(_summary_rows(summary)),
        "",
        "## Portfolio Performance Summary",
        "",
        _markdown_table(_summary_rows(summary)),
        "",
        "## Raw vs Sector-Capped Comparison",
        "",
        _markdown_table(_summary_rows(summary[summary["portfolio_name"].str.contains("higher_conviction|positive_combined", regex=True)])),
        "",
        "## Turnover and Transaction-Cost Summary",
        "",
        _markdown_table(summary[["portfolio_name", "mean_one_way_turnover", "median_one_way_turnover", "mean_transaction_cost"]]),
        "",
        "## Sector Exposure",
        "",
        _markdown_table(summary[["portfolio_name", "median_max_sector_weight", "median_top_3_sector_weight", "pct_rebalances_sector_cap_binding"]]),
        "",
        "## Important Caveats",
        "",
        "- Research simulation only, not production trading logic.",
        "- Returns are based on 63-trading-day forward labels.",
        "- No overlapping portfolio equity curve yet.",
        "- No slippage model beyond simple transaction-cost bps.",
        "",
        "## Output File Guide",
        "",
        _markdown_table(pd.DataFrame([{"File": path.name, "Purpose": "Portfolio simulation output"}])),
        "",
        "## Suggested Next Step",
        "",
        "If sector-capped performance remains attractive after transaction costs, the next step is an equity-curve backtest with overlapping holdings.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _summary_rows(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "portfolio_name",
        "rebalance_date_count",
        "median_symbol_count",
        "mean_net_forward_return",
        "std_net_forward_return",
        "max_drawdown_net_forward_return_index",
        "mean_net_vs_baseline",
        "pct_rebalances_net_above_baseline",
        "mean_one_way_turnover",
    ]
    return summary[columns].sort_values("portfolio_name")


def _output_paths(data_root: Path) -> dict[str, Path]:
    return {
        "rebalance_results_parquet": price_strength_portfolio_rebalance_results_path(data_root),
        "rebalance_results_csv": price_strength_portfolio_rebalance_results_csv_path(data_root),
        "summary_parquet": price_strength_portfolio_summary_path(data_root),
        "summary_csv": price_strength_portfolio_summary_csv_path(data_root),
        "constituents_parquet": price_strength_portfolio_constituents_path(data_root),
        "sector_weights_parquet": price_strength_portfolio_sector_weights_path(data_root),
        "sector_weights_csv": price_strength_portfolio_sector_weights_csv_path(data_root),
        "markdown_report": price_strength_portfolio_report_path(data_root),
        "metadata": price_strength_portfolio_metadata_path(data_root),
    }


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_None._"
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_format_value(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


def terminal_summary(paths: dict[str, Path], summary: dict[str, Any]) -> str:
    return "\n".join([
        "Price strength portfolio simulation complete.",
        "",
        f"Rebalance frequency: {summary['rebalance_frequency']}",
        f"Eligible rows: {summary['eligible_rows']}",
        f"Constituent rows: {summary['constituent_rows']}",
        f"Rebalance dates: {summary['rebalance_date_count']}",
        f"Sector cap: {summary['sector_cap']}",
        f"Transaction cost bps: {summary['transaction_cost_bps']}",
        "",
        f"- Rebalance results parquet: {paths['rebalance_results_parquet']}",
        f"- Summary parquet: {paths['summary_parquet']}",
        f"- Constituents parquet: {paths['constituents_parquet']}",
        f"- Sector weights parquet: {paths['sector_weights_parquet']}",
        f"- Markdown report: {paths['markdown_report']}",
        f"- Metadata JSON: {paths['metadata']}",
    ])
