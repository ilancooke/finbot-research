from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from finbot_research.config import (
    daily_bars_path,
    price_strength_equity_curve_constituents_path,
    price_strength_equity_curve_daily_csv_path,
    price_strength_equity_curve_daily_path,
    price_strength_equity_curve_metadata_path,
    price_strength_equity_curve_report_path,
    price_strength_equity_curve_sector_exposure_csv_path,
    price_strength_equity_curve_sector_exposure_path,
    price_strength_equity_curve_summary_csv_path,
    price_strength_equity_curve_summary_path,
    price_strength_equity_curve_vintages_csv_path,
    price_strength_equity_curve_vintages_path,
    price_strength_scorecard_path,
    reference_tickers_path,
)
from finbot_research.io import parquet_columns, read_parquet, write_csv, write_parquet
from finbot_research.metadata import write_metadata
from finbot_research.price_strength_portfolio_simulation import (
    equal_weights,
    one_way_turnover,
    sector_capped_weights,
)
from finbot_research.price_strength_rebalance_feasibility import attach_sector, select_rebalance_rows
from finbot_research.validation import ValidationError

DATASET_NAME = "equity_price_strength_equity_curve_backtest"
ADJUSTED_CLOSE_COLUMN = "closeadj"
ANNUALIZATION_DAYS = 252
REQUIRED_SCORECARD_COLUMNS = [
    "symbol",
    "date",
    "price_strength_scorecard_bucket",
    "price_strength_score_v0",
    "is_scorecard_bucket_eligible",
]
PORTFOLIO_DEFINITIONS = {
    "higher_conviction_raw": ["higher_conviction_price_strength"],
    "higher_conviction_sector_capped": ["higher_conviction_price_strength"],
    "positive_combined_raw": ["higher_conviction_price_strength", "price_strength_candidate"],
    "positive_combined_sector_capped": ["higher_conviction_price_strength", "price_strength_candidate"],
    "eligible_universe_baseline": None,
}
BENCHMARK_PORTFOLIO = "eligible_universe_baseline"


def build_price_strength_equity_curve_backtest(
    data_root: Path,
    *,
    rebalance_frequency: str = "monthly",
    holding_period_days: int = 63,
    sector_cap: float = 0.30,
    transaction_cost_bps: float = 25.0,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    if rebalance_frequency != "monthly":
        raise ValidationError(f"Unsupported rebalance frequency: {rebalance_frequency}")
    if holding_period_days <= 0:
        raise ValidationError("holding_period_days must be positive")

    scorecard_path = price_strength_scorecard_path(data_root)
    bars_path = daily_bars_path(data_root)
    scorecard = read_parquet(scorecard_path, columns=_scorecard_columns(scorecard_path))
    scorecard, sector_metadata = attach_sector(scorecard, data_root)
    eligible = prepare_equity_curve_input(scorecard, start_date=start_date, end_date=end_date)
    if eligible.empty:
        raise ValidationError("No eligible scorecard rows found for equity-curve backtest")
    if "sector" not in eligible.columns or eligible["sector"].isna().any():
        missing_count = int(eligible["sector"].isna().sum()) if "sector" in eligible.columns else len(eligible)
        raise ValidationError(
            "price-strength-equity-curve-backtest requires sector for all eligible rows; "
            f"missing sector rows: {missing_count}"
        )

    rebalance_rows = select_rebalance_rows(eligible, rebalance_frequency=rebalance_frequency)
    if rebalance_rows.empty:
        raise ValidationError("No monthly rebalance rows found for equity-curve backtest")

    constituents, sector_exposure = build_equity_curve_constituents(rebalance_rows, sector_cap=sector_cap)
    if constituents.empty:
        raise ValidationError("No portfolio constituents found for equity-curve backtest")

    adjusted_returns, trading_dates = load_adjusted_close_returns(
        bars_path,
        symbols=sorted(constituents["symbol"].astype(str).unique()),
        start_date=start_date,
        end_date=end_date,
    )
    vintages, constituents, sector_exposure, holding_calendar = build_vintages(
        constituents,
        sector_exposure,
        trading_dates=trading_dates,
        holding_period_days=holding_period_days,
        sector_cap=sector_cap,
        transaction_cost_bps=transaction_cost_bps,
    )
    vintage_daily_returns = compute_vintage_daily_returns(
        constituents,
        holding_calendar,
        adjusted_returns,
    )
    daily_curve = compute_daily_equity_curve(
        vintage_daily_returns,
        vintages,
        benchmark_portfolio=BENCHMARK_PORTFOLIO,
    )
    summary = compute_performance_summary(daily_curve, vintages)
    sector_exposure = add_sector_exposure_summary_fields(sector_exposure, vintages)

    paths = _output_paths(data_root)
    write_parquet(daily_curve, paths["daily_parquet"])
    write_csv(daily_curve, paths["daily_csv"])
    write_parquet(vintages, paths["vintages_parquet"])
    write_csv(vintages, paths["vintages_csv"])
    write_parquet(constituents, paths["constituents_parquet"])
    write_parquet(summary, paths["summary_parquet"])
    write_csv(summary, paths["summary_csv"])
    write_parquet(sector_exposure, paths["sector_exposure_parquet"])
    write_csv(sector_exposure, paths["sector_exposure_csv"])
    write_markdown_report(paths["markdown_report"], summary=summary, vintages=vintages)

    input_paths = [scorecard_path, bars_path]
    if sector_metadata.get("source") == "reference/tickers.parquet":
        input_paths.append(reference_tickers_path(data_root))
    metadata_path = write_metadata(
        dataset_name=DATASET_NAME,
        output_path=paths["summary_parquet"],
        input_paths=input_paths,
        dataframe=summary,
        extra_metadata={
            "dataset_type": "research_equity_curve_backtest",
            "research_only": True,
            "output_paths": {key: str(value) for key, value in paths.items()},
            "portfolio_definitions": PORTFOLIO_DEFINITIONS,
            "rebalance_frequency": rebalance_frequency,
            "rebalance_rule": "Monthly rebalance date is the last available trading date in each calendar month.",
            "holding_period_days": holding_period_days,
            "vintage_overlap_method": "Each rebalance creates a vintage held for the next trading days; daily portfolio return is the equal-weight average of active vintage returns.",
            "daily_return_source": str(bars_path),
            "daily_return_definition": f"Symbol-level close-to-close returns from {ADJUSTED_CLOSE_COLUMN}.",
            "sector_cap": sector_cap,
            "sector_cap_method": "Same helper as price-strength-portfolio-simulation: cap sector-level weights, redistribute remaining weight across uncapped sectors proportionally, equal weight within sector.",
            "transaction_cost_bps": transaction_cost_bps,
            "turnover_definition": "0.5 * sum(abs(current_weight - previous_weight)); first rebalance uses one-way turnover of 1.0.",
            "transaction_cost_method": "one_way_turnover * transaction_cost_bps / 10000, subtracted from net daily return on the rebalance date.",
            "benchmark_definition": f"{BENCHMARK_PORTFOLIO} equal-weights all eligible names on the same rebalance dates.",
            "annualization_assumption": f"{ANNUALIZATION_DAYS} trading days, risk-free rate 0.",
            "sector_availability": {"available": True, **sector_metadata},
            "limitations": [
                "Research backtest only, not production trading logic.",
                "Uses realized adjusted-close returns and simplified transaction costs.",
                "No slippage/spread model beyond bps assumption.",
                "No taxes, borrow constraints, limit orders, cash drag, execution constraints, broker integration, or live trading logic.",
                "Missing constituent return observations are excluded from that vintage-day weighted return.",
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
        "holding_period_days": holding_period_days,
        "eligible_rows": int(len(eligible)),
        "rebalance_date_count": int(rebalance_rows["rebalance_date"].nunique()),
        "vintage_count": int(len(vintages)),
        "daily_rows": int(len(daily_curve)),
        "sector_cap": sector_cap,
        "transaction_cost_bps": transaction_cost_bps,
    }


def _scorecard_columns(path: Path) -> list[str]:
    columns = parquet_columns(path)
    missing = [column for column in REQUIRED_SCORECARD_COLUMNS if column not in columns]
    if missing:
        raise ValidationError(f"scorecard missing required equity-curve columns: {missing}")
    return [column for column in [*REQUIRED_SCORECARD_COLUMNS, "sector"] if column in columns]


def prepare_equity_curve_input(
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


def load_adjusted_close_returns(
    path: Path,
    *,
    symbols: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    columns = parquet_columns(path)
    required = ["date", "symbol", ADJUSTED_CLOSE_COLUMN]
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValidationError(f"daily bars missing required adjusted-close columns: {missing}")
    bars = read_parquet(path, columns=required)
    bars["symbol"] = bars["symbol"].astype("string").str.upper()
    bars["date"] = pd.to_datetime(bars["date"], errors="raise")
    bars = bars[bars["symbol"].isin(set(symbols))].dropna(subset=[ADJUSTED_CLOSE_COLUMN]).copy()
    if start_date:
        bars = bars[bars["date"] >= pd.Timestamp(start_date)]
    if end_date:
        bars = bars[bars["date"] <= pd.Timestamp(end_date)]
    if bars.empty:
        raise ValidationError("No adjusted-close price rows found for selected constituents")
    bars = bars.sort_values(["symbol", "date"])
    bars["daily_return"] = bars.groupby("symbol", sort=True)[ADJUSTED_CLOSE_COLUMN].pct_change()
    returns = bars.dropna(subset=["daily_return"])[["date", "symbol", "daily_return"]].reset_index(drop=True)
    trading_dates = sorted(pd.to_datetime(bars["date"].drop_duplicates()))
    if not trading_dates:
        raise ValidationError("No trading dates found in adjusted-close price data")
    return returns, trading_dates


def build_equity_curve_constituents(
    rebalance_rows: pd.DataFrame,
    *,
    sector_cap: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    constituent_frames = []
    sector_frames = []
    for rebalance_date, date_rows in rebalance_rows.groupby("rebalance_date", sort=True):
        for portfolio_name, buckets in PORTFOLIO_DEFINITIONS.items():
            group = date_rows if buckets is None else date_rows[date_rows["price_strength_scorecard_bucket"].isin(buckets)]
            if group.empty:
                continue
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
                    ]
                ]
            )
            sectors = (
                frame.groupby("sector", sort=True)
                .agg(sector_weight=("weight", "sum"), symbol_count=("symbol", "nunique"))
                .reset_index()
            )
            sectors["rebalance_date"] = rebalance_date
            sectors["portfolio_name"] = portfolio_name
            sector_frames.append(sectors[["rebalance_date", "portfolio_name", "sector", "sector_weight", "symbol_count"]])
    if not constituent_frames:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(constituent_frames, ignore_index=True), pd.concat(sector_frames, ignore_index=True)


def build_vintages(
    constituents: pd.DataFrame,
    sector_exposure: pd.DataFrame,
    *,
    trading_dates: list[pd.Timestamp],
    holding_period_days: int,
    sector_cap: float,
    transaction_cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vintage_rows = []
    holding_rows = []
    previous_weights: dict[str, dict[str, float]] = {}
    constituent_frames = []
    sector_frames = []
    trading_index = pd.DatetimeIndex(trading_dates)

    for (rebalance_date, portfolio_name), group in constituents.groupby(["rebalance_date", "portfolio_name"], sort=True):
        rebalance_ts = pd.Timestamp(rebalance_date)
        vintage_id = f"{portfolio_name}_{rebalance_ts.strftime('%Y%m%d')}"
        holding_dates = list(trading_index[trading_index > rebalance_ts][:holding_period_days])
        current = dict(zip(group["symbol"].astype(str), group["weight"]))
        previous = previous_weights.get(portfolio_name, {})
        turnover = one_way_turnover(previous, current) if previous else 1.0
        previous_weights[portfolio_name] = current
        sectors = sector_exposure[
            (sector_exposure["rebalance_date"] == rebalance_date)
            & (sector_exposure["portfolio_name"] == portfolio_name)
        ].copy()
        max_sector_weight = float(sectors["sector_weight"].max())
        vintage_rows.append(
            {
                "rebalance_date": rebalance_ts.date(),
                "portfolio_name": portfolio_name,
                "vintage_id": vintage_id,
                "holding_start_date": holding_dates[0].date() if holding_dates else pd.NaT,
                "holding_end_date": holding_dates[-1].date() if holding_dates else pd.NaT,
                "symbol_count": int(group["symbol"].nunique()),
                "sector_count": int(sectors["sector"].nunique()),
                "max_sector_weight": max_sector_weight,
                "top_3_sector_weight": float(sectors["sector_weight"].sort_values(ascending=False).head(3).sum()),
                "sector_cap": sector_cap if portfolio_name.endswith("_sector_capped") else None,
                "sector_cap_binding": bool(portfolio_name.endswith("_sector_capped") and max_sector_weight >= sector_cap - 1e-9),
                "one_way_turnover": float(turnover),
                "transaction_cost_bps": float(transaction_cost_bps),
                "transaction_cost": float(turnover * transaction_cost_bps / 10000),
            }
        )
        enriched = group.copy()
        enriched["vintage_id"] = vintage_id
        constituent_frames.append(enriched)
        sectors["vintage_id"] = vintage_id
        sector_frames.append(sectors)
        holding_rows.extend({"vintage_id": vintage_id, "date": date} for date in holding_dates)

    vintages = pd.DataFrame(vintage_rows)
    enriched_constituents = pd.concat(constituent_frames, ignore_index=True)
    enriched_constituents = enriched_constituents[
        [
            "rebalance_date",
            "vintage_id",
            "portfolio_name",
            "symbol",
            "sector",
            "weight",
            "price_strength_scorecard_bucket",
        ]
    ]
    enriched_constituents["rebalance_date"] = pd.to_datetime(enriched_constituents["rebalance_date"]).dt.date
    enriched_sector = pd.concat(sector_frames, ignore_index=True)[
        ["rebalance_date", "portfolio_name", "vintage_id", "sector", "sector_weight", "symbol_count"]
    ]
    enriched_sector["rebalance_date"] = pd.to_datetime(enriched_sector["rebalance_date"]).dt.date
    holding_calendar = pd.DataFrame(holding_rows)
    return vintages, enriched_constituents, enriched_sector, holding_calendar


def compute_vintage_daily_returns(
    constituents: pd.DataFrame,
    holding_calendar: pd.DataFrame,
    adjusted_returns: pd.DataFrame,
) -> pd.DataFrame:
    if holding_calendar.empty:
        return pd.DataFrame(columns=["date", "portfolio_name", "vintage_id", "vintage_daily_return"])
    returns = adjusted_returns.set_index(["date", "symbol"])["daily_return"].sort_index()
    rows = []
    for vintage_id, calendar in holding_calendar.groupby("vintage_id", sort=True):
        weights = constituents[constituents["vintage_id"] == vintage_id][["symbol", "weight", "portfolio_name"]].copy()
        if weights.empty:
            continue
        portfolio_name = str(weights["portfolio_name"].iloc[0])
        dates = sorted(pd.to_datetime(calendar["date"].drop_duplicates()))
        index = pd.MultiIndex.from_product([dates, weights["symbol"].astype(str)], names=["date", "symbol"])
        observations = returns.reindex(index).rename("daily_return").reset_index()
        observations = observations.merge(weights[["symbol", "weight"]], on="symbol", how="left")
        for date, group in observations.groupby("date", sort=True):
            valid = group.dropna(subset=["daily_return", "weight"])
            if valid.empty or valid["weight"].sum() <= 0:
                daily_return = 0.0
            else:
                daily_return = float((valid["daily_return"] * valid["weight"]).sum() / valid["weight"].sum())
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "portfolio_name": portfolio_name,
                    "vintage_id": vintage_id,
                    "vintage_daily_return": daily_return,
                }
            )
    return pd.DataFrame(rows)


def compute_daily_equity_curve(
    vintage_daily_returns: pd.DataFrame,
    vintages: pd.DataFrame,
    *,
    benchmark_portfolio: str = BENCHMARK_PORTFOLIO,
) -> pd.DataFrame:
    daily = (
        vintage_daily_returns.groupby(["date", "portfolio_name"], sort=True)
        .agg(
            gross_daily_return=("vintage_daily_return", "mean"),
            active_vintage_count=("vintage_id", "nunique"),
        )
        .reset_index()
        if not vintage_daily_returns.empty
        else pd.DataFrame(columns=["date", "portfolio_name", "gross_daily_return", "active_vintage_count"])
    )
    costs = (
        vintages.assign(date=pd.to_datetime(vintages["rebalance_date"]))
        .groupby(["date", "portfolio_name"], sort=True)["transaction_cost"]
        .sum()
        .rename("transaction_cost")
        .reset_index()
    )
    portfolio_names = sorted(vintages["portfolio_name"].unique())
    dates = sorted(set(pd.to_datetime(daily["date"])) | set(pd.to_datetime(costs["date"])))
    grid = pd.MultiIndex.from_product([dates, portfolio_names], names=["date", "portfolio_name"]).to_frame(index=False)
    frame = grid.merge(daily, on=["date", "portfolio_name"], how="left").merge(costs, on=["date", "portfolio_name"], how="left")
    frame["gross_daily_return"] = frame["gross_daily_return"].fillna(0.0)
    frame["active_vintage_count"] = frame["active_vintage_count"].fillna(0).astype(int)
    frame["transaction_cost"] = frame["transaction_cost"].fillna(0.0)
    frame["net_daily_return"] = frame["gross_daily_return"] - frame["transaction_cost"]
    frame = frame.sort_values(["portfolio_name", "date"]).reset_index(drop=True)
    frame["gross_cumulative_return_index"] = frame.groupby("portfolio_name")["gross_daily_return"].transform(cumulative_return_index)
    frame["net_cumulative_return_index"] = frame.groupby("portfolio_name")["net_daily_return"].transform(cumulative_return_index)

    benchmark = frame[frame["portfolio_name"] == benchmark_portfolio][
        ["date", "gross_daily_return", "net_daily_return"]
    ].rename(
        columns={
            "gross_daily_return": "benchmark_gross_daily_return",
            "net_daily_return": "benchmark_net_daily_return",
        }
    )
    frame = frame.merge(benchmark, on="date", how="left")
    frame["benchmark_gross_daily_return"] = frame["benchmark_gross_daily_return"].fillna(0.0)
    frame["benchmark_net_daily_return"] = frame["benchmark_net_daily_return"].fillna(0.0)
    frame["gross_excess_daily_return"] = frame["gross_daily_return"] - frame["benchmark_gross_daily_return"]
    frame["net_excess_daily_return"] = frame["net_daily_return"] - frame["benchmark_net_daily_return"]
    frame = frame.sort_values(["portfolio_name", "date"]).reset_index(drop=True)
    frame["gross_excess_cumulative_return_index"] = frame.groupby("portfolio_name")["gross_excess_daily_return"].transform(
        cumulative_return_index
    )
    frame["net_excess_cumulative_return_index"] = frame.groupby("portfolio_name")["net_excess_daily_return"].transform(
        cumulative_return_index
    )
    output_columns = [
        "date",
        "portfolio_name",
        "active_vintage_count",
        "gross_daily_return",
        "net_daily_return",
        "gross_cumulative_return_index",
        "net_cumulative_return_index",
        "benchmark_gross_daily_return",
        "benchmark_net_daily_return",
        "gross_excess_daily_return",
        "net_excess_daily_return",
        "gross_excess_cumulative_return_index",
        "net_excess_cumulative_return_index",
    ]
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    return frame[output_columns]


def cumulative_return_index(returns: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    return (1 + numeric).cumprod()


def compute_drawdown(returns: pd.Series) -> float:
    index = cumulative_return_index(returns)
    if index.empty:
        return 0.0
    running_max = index.cummax()
    return float((index / running_max - 1).min())


def annualized_return(returns: pd.Series, *, trading_days: int = ANNUALIZATION_DAYS) -> float:
    numeric = pd.to_numeric(returns, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    total = float((1 + numeric).prod() - 1)
    return float((1 + total) ** (trading_days / len(numeric)) - 1)


def annualized_volatility(returns: pd.Series, *, trading_days: int = ANNUALIZATION_DAYS) -> float:
    numeric = pd.to_numeric(returns, errors="coerce").dropna()
    if len(numeric) < 2:
        return 0.0
    return float(numeric.std() * (trading_days**0.5))


def compute_performance_summary(daily_curve: pd.DataFrame, vintages: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for portfolio_name, daily in daily_curve.groupby("portfolio_name", sort=True):
        daily = daily.sort_values("date")
        vintage_group = vintages[vintages["portfolio_name"] == portfolio_name]
        gross_vol = annualized_volatility(daily["gross_daily_return"])
        net_vol = annualized_volatility(daily["net_daily_return"])
        gross_ann = annualized_return(daily["gross_daily_return"])
        net_ann = annualized_return(daily["net_daily_return"])
        rows.append(
            {
                "portfolio_name": portfolio_name,
                "start_date": daily["date"].min(),
                "end_date": daily["date"].max(),
                "daily_observation_count": int(len(daily)),
                "rebalance_count": int(vintage_group["rebalance_date"].nunique()),
                "mean_active_vintage_count": float(daily["active_vintage_count"].mean()),
                "mean_symbol_count_per_vintage": float(vintage_group["symbol_count"].mean()),
                "median_symbol_count_per_vintage": float(vintage_group["symbol_count"].median()),
                "mean_one_way_turnover": float(vintage_group["one_way_turnover"].mean()),
                "median_one_way_turnover": float(vintage_group["one_way_turnover"].median()),
                "mean_transaction_cost": float(vintage_group["transaction_cost"].mean()),
                "gross_total_return": float(daily["gross_cumulative_return_index"].iloc[-1] - 1),
                "net_total_return": float(daily["net_cumulative_return_index"].iloc[-1] - 1),
                "gross_annualized_return": gross_ann,
                "net_annualized_return": net_ann,
                "gross_annualized_volatility": gross_vol,
                "net_annualized_volatility": net_vol,
                "gross_sharpe_like": gross_ann / gross_vol if gross_vol else None,
                "net_sharpe_like": net_ann / net_vol if net_vol else None,
                "gross_max_drawdown": compute_drawdown(daily["gross_daily_return"]),
                "net_max_drawdown": compute_drawdown(daily["net_daily_return"]),
                "gross_excess_total_return": float(daily["gross_excess_cumulative_return_index"].iloc[-1] - 1),
                "net_excess_total_return": float(daily["net_excess_cumulative_return_index"].iloc[-1] - 1),
                "gross_excess_annualized_return": annualized_return(daily["gross_excess_daily_return"]),
                "net_excess_annualized_return": annualized_return(daily["net_excess_daily_return"]),
                "net_daily_win_rate_vs_benchmark": float((daily["net_excess_daily_return"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def add_sector_exposure_summary_fields(sector_exposure: pd.DataFrame, vintages: pd.DataFrame) -> pd.DataFrame:
    summary = (
        vintages.groupby("portfolio_name", sort=True)
        .agg(
            median_max_sector_weight=("max_sector_weight", "median"),
            median_top_3_sector_weight=("top_3_sector_weight", "median"),
            pct_rebalances_sector_cap_binding=("sector_cap_binding", "mean"),
        )
        .reset_index()
    )
    return sector_exposure.merge(summary, on="portfolio_name", how="left")


def write_markdown_report(path: Path, *, summary: pd.DataFrame, vintages: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Equity Price Strength Equity-Curve Backtest",
        "",
        "## Purpose",
        "",
        "Approximate a realistic research equity curve for the price-strength scorecard using overlapping monthly vintages.",
        "",
        "## Method",
        "",
        "- Monthly rebalance dates are the last available trading date in each calendar month.",
        "- Each rebalance creates a vintage whose realized adjusted-close returns begin on the next trading date.",
        "- Daily portfolio returns equal the average return of active vintages for that portfolio.",
        "- Transaction costs are charged on rebalance dates using vintage-level one-way turnover.",
        "",
        "## Portfolio Definitions",
        "",
        _markdown_table(pd.DataFrame([{"portfolio_name": key, "buckets": str(value)} for key, value in PORTFOLIO_DEFINITIONS.items()])),
        "",
        "## Executive Summary",
        "",
        _markdown_table(_summary_rows(summary)),
        "",
        "## Performance Summary",
        "",
        _markdown_table(_summary_rows(summary)),
        "",
        "## Raw vs Sector-Capped Comparison",
        "",
        _markdown_table(_summary_rows(summary[summary["portfolio_name"].str.contains("higher_conviction", regex=False)])),
        "",
        "## Benchmark Comparison",
        "",
        _markdown_table(summary[["portfolio_name", "net_excess_total_return", "net_excess_annualized_return", "net_daily_win_rate_vs_benchmark"]]),
        "",
        "## Drawdown and Volatility",
        "",
        _markdown_table(summary[["portfolio_name", "net_annualized_volatility", "gross_max_drawdown", "net_max_drawdown"]]),
        "",
        "## Turnover and Transaction Costs",
        "",
        _markdown_table(summary[["portfolio_name", "mean_one_way_turnover", "median_one_way_turnover", "mean_transaction_cost"]]),
        "",
        "## Sector Exposure",
        "",
        _markdown_table(
            vintages.groupby("portfolio_name", sort=True)
            .agg(
                median_max_sector_weight=("max_sector_weight", "median"),
                median_top_3_sector_weight=("top_3_sector_weight", "median"),
                pct_rebalances_sector_cap_binding=("sector_cap_binding", "mean"),
            )
            .reset_index()
        ),
        "",
        "## Important Caveats",
        "",
        "- This is a research backtest, not production trading logic or a trading recommendation.",
        "- It uses realized adjusted-close daily returns.",
        "- Monthly vintages overlap for the requested holding period, 63 trading days by default.",
        "- Transaction costs are simplified and do not model slippage, spreads, taxes, borrow constraints, limit orders, cash drag, or execution constraints.",
        "- Missing constituent return observations are excluded from that vintage-day weighted return.",
        "",
        "## Output File Guide",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {"File": "equity_price_strength_equity_curve_daily.parquet/csv", "Purpose": "Daily gross/net and excess equity curves."},
                    {"File": "equity_price_strength_equity_curve_vintages.parquet/csv", "Purpose": "Rebalance vintage diagnostics."},
                    {"File": "equity_price_strength_equity_curve_constituents.parquet", "Purpose": "Vintage constituent weights."},
                    {"File": "equity_price_strength_equity_curve_summary.parquet/csv", "Purpose": "Portfolio performance summary."},
                    {"File": "equity_price_strength_equity_curve_sector_exposure.parquet/csv", "Purpose": "Vintage sector exposure."},
                    {"File": "equity_price_strength_equity_curve_backtest.metadata.json", "Purpose": "Inputs, assumptions, and output paths."},
                ]
            )
        ),
        "",
        "## Suggested Next Step",
        "",
        "Review whether higher-conviction raw results survive transaction costs, drawdowns, and sector concentration before considering a more formal strategy/backtesting package.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _summary_rows(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "portfolio_name",
        "rebalance_count",
        "mean_active_vintage_count",
        "median_symbol_count_per_vintage",
        "net_total_return",
        "net_annualized_return",
        "net_annualized_volatility",
        "net_max_drawdown",
        "net_excess_total_return",
        "net_daily_win_rate_vs_benchmark",
        "mean_one_way_turnover",
    ]
    return summary[columns].sort_values("portfolio_name")


def _output_paths(data_root: Path) -> dict[str, Path]:
    return {
        "daily_parquet": price_strength_equity_curve_daily_path(data_root),
        "daily_csv": price_strength_equity_curve_daily_csv_path(data_root),
        "vintages_parquet": price_strength_equity_curve_vintages_path(data_root),
        "vintages_csv": price_strength_equity_curve_vintages_csv_path(data_root),
        "constituents_parquet": price_strength_equity_curve_constituents_path(data_root),
        "summary_parquet": price_strength_equity_curve_summary_path(data_root),
        "summary_csv": price_strength_equity_curve_summary_csv_path(data_root),
        "sector_exposure_parquet": price_strength_equity_curve_sector_exposure_path(data_root),
        "sector_exposure_csv": price_strength_equity_curve_sector_exposure_csv_path(data_root),
        "markdown_report": price_strength_equity_curve_report_path(data_root),
        "metadata": price_strength_equity_curve_metadata_path(data_root),
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
    return "\n".join(
        [
            "Price strength equity-curve backtest complete.",
            "",
            f"Rebalance frequency: {summary['rebalance_frequency']}",
            f"Holding period days: {summary['holding_period_days']}",
            f"Eligible rows: {summary['eligible_rows']}",
            f"Rebalance dates: {summary['rebalance_date_count']}",
            f"Vintages: {summary['vintage_count']}",
            f"Daily rows: {summary['daily_rows']}",
            f"Sector cap: {summary['sector_cap']}",
            f"Transaction cost bps: {summary['transaction_cost_bps']}",
            "",
            f"- Daily equity curve parquet: {paths['daily_parquet']}",
            f"- Vintages parquet: {paths['vintages_parquet']}",
            f"- Constituents parquet: {paths['constituents_parquet']}",
            f"- Summary parquet: {paths['summary_parquet']}",
            f"- Sector exposure parquet: {paths['sector_exposure_parquet']}",
            f"- Markdown report: {paths['markdown_report']}",
            f"- Metadata JSON: {paths['metadata']}",
        ]
    )
