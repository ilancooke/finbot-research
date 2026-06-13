from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import pandas as pd

from finbot_research.config import (
    daily_bars_path,
    price_strength_cost_sensitivity_csv_path,
    price_strength_cost_sensitivity_path,
    price_strength_equity_curve_constituents_path,
    price_strength_equity_curve_daily_path,
    price_strength_equity_curve_metadata_path,
    price_strength_equity_curve_sector_exposure_path,
    price_strength_equity_curve_summary_path,
    price_strength_equity_curve_vintages_path,
    price_strength_regime_performance_csv_path,
    price_strength_regime_performance_path,
    price_strength_robustness_metadata_path,
    price_strength_robustness_report_path,
    price_strength_robustness_summary_csv_path,
    price_strength_robustness_summary_path,
    price_strength_rolling_performance_csv_path,
    price_strength_rolling_performance_path,
    price_strength_scorecard_path,
    price_strength_sector_cap_sensitivity_csv_path,
    price_strength_sector_cap_sensitivity_path,
    price_strength_sector_contribution_csv_path,
    price_strength_sector_contribution_path,
    price_strength_symbol_contribution_csv_path,
    price_strength_symbol_contribution_path,
    reference_tickers_path,
)
from finbot_research.io import read_parquet, write_csv, write_parquet
from finbot_research.metadata import write_metadata
from finbot_research.price_strength_equity_curve_backtest import (
    ANNUALIZATION_DAYS,
    BENCHMARK_PORTFOLIO,
    add_sector_exposure_summary_fields,
    annualized_return,
    annualized_volatility,
    build_equity_curve_constituents,
    build_vintages,
    compute_drawdown,
    compute_performance_summary,
    compute_vintage_daily_returns,
    cumulative_return_index,
    load_adjusted_close_returns,
    prepare_equity_curve_input,
)
from finbot_research.price_strength_rebalance_feasibility import attach_sector, select_rebalance_rows
from finbot_research.validation import ValidationError

DATASET_NAME = "equity_price_strength_equity_curve_robustness"
TRANSACTION_COST_BPS_VALUES = [0.0, 25.0, 50.0, 100.0, 150.0]
SECTOR_CAP_VALUES = [0.20, 0.25, 0.30, 0.40]
COST_SENSITIVITY_PORTFOLIOS = [
    "higher_conviction_raw",
    "higher_conviction_sector_capped",
    BENCHMARK_PORTFOLIO,
]
CONTRIBUTION_PORTFOLIOS = ["higher_conviction_raw", "higher_conviction_sector_capped"]
ROLLING_WINDOWS = [252, 756]
REGIME_DEFINITIONS = [
    ("pre_gfc", "1998-12-31", "2007-12-31"),
    ("gfc", "2008-01-01", "2009-12-31"),
    ("post_gfc_pre_covid", "2010-01-01", "2019-12-31"),
    ("covid_and_liquidity_boom", "2020-01-01", "2021-12-31"),
    ("rate_hike_drawdown", "2022-01-01", "2022-12-31"),
    ("ai_growth_rebound", "2023-01-01", None),
]
ROLLING_PERFORMANCE_COLUMNS = [
    "date",
    "portfolio_name",
    "window_trading_days",
    "rolling_net_total_return",
    "rolling_net_excess_return",
    "rolling_net_annualized_return",
    "rolling_net_annualized_volatility",
    "rolling_net_sharpe_like",
    "rolling_net_max_drawdown",
]
SECTOR_CONTRIBUTION_COLUMNS = [
    "portfolio_name",
    "period_name",
    "period_start_date",
    "period_end_date",
    "sector",
    "mean_daily_weight",
    "total_return_contribution",
    "avg_daily_return_contribution",
    "share_of_positive_contribution",
    "share_of_absolute_contribution",
]
SYMBOL_CONTRIBUTION_COLUMNS = [
    "portfolio_name",
    "period_name",
    "period_start_date",
    "period_end_date",
    "symbol",
    "sector",
    "active_day_count",
    "mean_weight_when_active",
    "total_return_contribution",
    "avg_daily_return_contribution",
]


def build_price_strength_equity_curve_robustness(data_root: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    inputs = read_backtest_outputs(data_root)
    daily_curve = inputs["daily_curve"]
    vintages = inputs["vintages"]
    constituents = inputs["constituents"]
    sector_exposure = inputs["sector_exposure"]
    metadata = inputs["metadata"]

    cost_sensitivity = compute_cost_sensitivity(daily_curve, vintages)
    sector_cap_sensitivity = compute_sector_cap_sensitivity(
        data_root,
        metadata,
        base_daily_curve=daily_curve,
        base_summary=inputs["summary"],
        base_vintages=vintages,
    )
    regime_performance = compute_regime_performance(daily_curve)
    rolling_performance = compute_rolling_performance(daily_curve, windows=ROLLING_WINDOWS)
    sector_contribution, symbol_contribution = compute_contribution_outputs(
        data_root,
        daily_curve=daily_curve,
        vintages=vintages,
        constituents=constituents,
    )
    robustness_summary = build_robustness_summary(
        cost_sensitivity,
        sector_cap_sensitivity,
        regime_performance,
        rolling_performance,
    )

    paths = _output_paths(data_root)
    write_parquet(cost_sensitivity, paths["cost_sensitivity_parquet"])
    write_csv(cost_sensitivity, paths["cost_sensitivity_csv"])
    write_parquet(sector_cap_sensitivity, paths["sector_cap_sensitivity_parquet"])
    write_csv(sector_cap_sensitivity, paths["sector_cap_sensitivity_csv"])
    write_parquet(regime_performance, paths["regime_performance_parquet"])
    write_csv(regime_performance, paths["regime_performance_csv"])
    write_parquet(rolling_performance, paths["rolling_performance_parquet"])
    write_csv(rolling_performance, paths["rolling_performance_csv"])
    write_parquet(sector_contribution, paths["sector_contribution_parquet"])
    write_csv(sector_contribution, paths["sector_contribution_csv"])
    write_parquet(symbol_contribution, paths["symbol_contribution_parquet"])
    write_csv(symbol_contribution, paths["symbol_contribution_csv"])
    write_parquet(robustness_summary, paths["summary_parquet"])
    write_csv(robustness_summary, paths["summary_csv"])
    write_markdown_report(
        paths["markdown_report"],
        cost_sensitivity=cost_sensitivity,
        sector_cap_sensitivity=sector_cap_sensitivity,
        regime_performance=regime_performance,
        rolling_performance=rolling_performance,
        sector_contribution=sector_contribution,
        symbol_contribution=symbol_contribution,
        robustness_summary=robustness_summary,
    )

    input_paths = [
        price_strength_equity_curve_daily_path(data_root),
        price_strength_equity_curve_summary_path(data_root),
        price_strength_equity_curve_vintages_path(data_root),
        price_strength_equity_curve_constituents_path(data_root),
        price_strength_equity_curve_sector_exposure_path(data_root),
        price_strength_equity_curve_metadata_path(data_root),
        price_strength_scorecard_path(data_root),
        daily_bars_path(data_root),
    ]
    reference_path = reference_tickers_path(data_root)
    if reference_path.exists():
        input_paths.append(reference_path)
    metadata_path = write_metadata(
        dataset_name=DATASET_NAME,
        output_path=paths["summary_parquet"],
        input_paths=input_paths,
        dataframe=robustness_summary,
        extra_metadata={
            "dataset_type": "research_equity_curve_robustness",
            "research_only": True,
            "output_paths": {key: str(value) for key, value in paths.items()},
            "transaction_cost_assumptions_bps": TRANSACTION_COST_BPS_VALUES,
            "sector_cap_assumptions": ["none", *SECTOR_CAP_VALUES],
            "regime_definitions": [
                {"regime_name": name, "start_date": start, "end_date": end}
                for name, start, end in REGIME_DEFINITIONS
            ],
            "rolling_window_definitions": ROLLING_WINDOWS,
            "contribution_methodology": (
                "Approximate sector/symbol contribution uses constituent weight times realized adjusted-close "
                "daily return for each vintage-day, then averages contributions across active vintages."
            ),
            "benchmark_definition": f"{BENCHMARK_PORTFOLIO} from the base equity-curve backtest.",
            "limitations": [
                "Research diagnostics only, not production trading logic.",
                "Cost sensitivity recomputes net returns from base gross returns and vintage turnover.",
                "Sector-cap sensitivity reruns the in-memory equity-curve helper logic and does not write alternate backtest directories.",
                "Contribution outputs are approximate and focus on higher-conviction portfolios.",
                "No slippage/spread model beyond bps transaction-cost assumptions.",
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
        "cost_sensitivity_rows": int(len(cost_sensitivity)),
        "sector_cap_sensitivity_rows": int(len(sector_cap_sensitivity)),
        "regime_rows": int(len(regime_performance)),
        "rolling_rows": int(len(rolling_performance)),
        "sector_contribution_rows": int(len(sector_contribution)),
        "symbol_contribution_rows": int(len(symbol_contribution)),
        "summary_rows": int(len(robustness_summary)),
    }


def read_backtest_outputs(data_root: Path) -> dict[str, Any]:
    required = {
        "daily_curve": price_strength_equity_curve_daily_path(data_root),
        "summary": price_strength_equity_curve_summary_path(data_root),
        "vintages": price_strength_equity_curve_vintages_path(data_root),
        "constituents": price_strength_equity_curve_constituents_path(data_root),
        "sector_exposure": price_strength_equity_curve_sector_exposure_path(data_root),
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise ValidationError(f"Missing equity-curve backtest inputs for robustness diagnostics: {missing}")
    metadata_path = price_strength_equity_curve_metadata_path(data_root)
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        key: read_parquet(path) for key, path in required.items()
    } | {"metadata": metadata}


def compute_cost_sensitivity(
    daily_curve: pd.DataFrame,
    vintages: pd.DataFrame,
    *,
    transaction_cost_bps_values: list[float] | None = None,
    portfolio_names: list[str] | None = None,
) -> pd.DataFrame:
    bps_values = transaction_cost_bps_values or TRANSACTION_COST_BPS_VALUES
    selected = portfolio_names or COST_SENSITIVITY_PORTFOLIOS
    rows = []
    for bps in bps_values:
        adjusted_vintages = vintages.copy()
        adjusted_vintages["transaction_cost_bps"] = float(bps)
        adjusted_vintages["transaction_cost"] = adjusted_vintages["one_way_turnover"] * float(bps) / 10000
        adjusted_daily = recompute_net_daily_from_gross(daily_curve, adjusted_vintages)
        summary = compute_performance_summary(adjusted_daily, adjusted_vintages)
        summary = summary[summary["portfolio_name"].isin(selected)].copy()
        summary["transaction_cost_bps"] = float(bps)
        rows.append(
            summary[
                [
                    "transaction_cost_bps",
                    "portfolio_name",
                    "net_total_return",
                    "net_annualized_return",
                    "net_annualized_volatility",
                    "net_sharpe_like",
                    "net_max_drawdown",
                    "net_excess_total_return",
                    "net_excess_annualized_return",
                    "net_daily_win_rate_vs_benchmark",
                    "mean_one_way_turnover",
                    "mean_transaction_cost",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def recompute_net_daily_from_gross(daily_curve: pd.DataFrame, vintages: pd.DataFrame) -> pd.DataFrame:
    frame = daily_curve.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    costs = (
        vintages.assign(date=pd.to_datetime(vintages["rebalance_date"]))
        .groupby(["date", "portfolio_name"], sort=True)["transaction_cost"]
        .sum()
        .rename("transaction_cost")
        .reset_index()
    )
    frame = frame.drop(
        columns=[
            "net_daily_return",
            "net_cumulative_return_index",
            "benchmark_net_daily_return",
            "net_excess_daily_return",
            "net_excess_cumulative_return_index",
        ],
        errors="ignore",
    )
    frame = frame.merge(costs, on=["date", "portfolio_name"], how="left")
    frame["transaction_cost"] = frame["transaction_cost"].fillna(0.0)
    frame["net_daily_return"] = frame["gross_daily_return"] - frame["transaction_cost"]
    frame = frame.sort_values(["portfolio_name", "date"]).reset_index(drop=True)
    frame["gross_cumulative_return_index"] = frame.groupby("portfolio_name")["gross_daily_return"].transform(
        cumulative_return_index
    )
    frame["net_cumulative_return_index"] = frame.groupby("portfolio_name")["net_daily_return"].transform(
        cumulative_return_index
    )
    benchmark = frame[frame["portfolio_name"] == BENCHMARK_PORTFOLIO][["date", "gross_daily_return", "net_daily_return"]]
    benchmark = benchmark.rename(
        columns={
            "gross_daily_return": "benchmark_gross_daily_return",
            "net_daily_return": "benchmark_net_daily_return",
        }
    )
    frame = frame.drop(columns=["benchmark_gross_daily_return"], errors="ignore").merge(benchmark, on="date", how="left")
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
    frame["date"] = frame["date"].dt.date
    return frame[
        [
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
    ]


def compute_sector_cap_sensitivity(
    data_root: Path,
    metadata: dict[str, Any],
    *,
    base_daily_curve: pd.DataFrame,
    base_summary: pd.DataFrame,
    base_vintages: pd.DataFrame,
) -> pd.DataFrame:
    scorecard_path = price_strength_scorecard_path(data_root)
    scorecard = read_parquet(scorecard_path)
    scorecard, _ = attach_sector(scorecard, data_root)
    start_date = metadata.get("start_date")
    end_date = metadata.get("end_date")
    holding_period_days = int(metadata.get("holding_period_days", 63))
    transaction_cost_bps = float(metadata.get("transaction_cost_bps", 25.0))
    eligible = prepare_equity_curve_input(scorecard, start_date=start_date, end_date=end_date)
    if "sector" not in eligible.columns or eligible["sector"].isna().any():
        raise ValidationError("price-strength-equity-curve-robustness requires sector for sector-cap sensitivity")
    rebalance_rows = select_rebalance_rows(eligible, rebalance_frequency="monthly")
    signal_rows = rebalance_rows[
        rebalance_rows["price_strength_scorecard_bucket"].isin(
            ["higher_conviction_price_strength", "price_strength_candidate"]
        )
    ]
    symbols = sorted(signal_rows["symbol"].astype(str).unique())
    adjusted_returns, trading_dates = load_adjusted_close_returns(
        daily_bars_path(data_root),
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
    )

    output_frames = []
    raw = base_summary[base_summary["portfolio_name"].isin(["higher_conviction_raw", "positive_combined_raw"])].copy()
    raw_vintages = base_vintages[base_vintages["portfolio_name"].isin(raw["portfolio_name"])].copy()
    raw_exposure = (
        raw_vintages.groupby("portfolio_name", sort=True)
        .agg(
            median_max_sector_weight=("max_sector_weight", "median"),
            median_top_3_sector_weight=("top_3_sector_weight", "median"),
            pct_rebalances_sector_cap_binding=("sector_cap_binding", "mean"),
        )
        .reset_index()
    )
    raw = raw.merge(raw_exposure, on="portfolio_name", how="left")
    raw["sector_cap"] = "none"
    output_frames.append(raw)
    for sector_cap in SECTOR_CAP_VALUES:
        constituents, sector_exposure = build_equity_curve_constituents(rebalance_rows, sector_cap=sector_cap)
        constituents = constituents[
            constituents["portfolio_name"].isin(["higher_conviction_sector_capped", "positive_combined_sector_capped"])
        ].copy()
        sector_exposure = sector_exposure[
            sector_exposure["portfolio_name"].isin(["higher_conviction_sector_capped", "positive_combined_sector_capped"])
        ].copy()
        vintages, constituents, sector_exposure, holding_calendar = build_vintages(
            constituents,
            sector_exposure,
            trading_dates=trading_dates,
            holding_period_days=holding_period_days,
            sector_cap=sector_cap,
            transaction_cost_bps=transaction_cost_bps,
        )
        vintage_daily = compute_vintage_daily_returns(constituents, holding_calendar, adjusted_returns)
        daily_curve = compute_daily_with_external_benchmark(vintage_daily, vintages, base_daily_curve)
        summary = compute_performance_summary(daily_curve, vintages)
        sector_exposure = add_sector_exposure_summary_fields(sector_exposure, vintages)
        exposure_summary = sector_exposure[
            [
                "portfolio_name",
                "median_max_sector_weight",
                "median_top_3_sector_weight",
                "pct_rebalances_sector_cap_binding",
            ]
        ].drop_duplicates()
        summary = summary.merge(exposure_summary, on="portfolio_name", how="left")
        capped = summary.copy()
        capped["sector_cap"] = f"{sector_cap:.2f}"
        output_frames.append(capped)

    result = pd.concat(output_frames, ignore_index=True)
    return result[
        [
            "sector_cap",
            "portfolio_name",
            "net_total_return",
            "net_annualized_return",
            "net_annualized_volatility",
            "net_sharpe_like",
            "net_max_drawdown",
            "net_excess_total_return",
            "net_excess_annualized_return",
            "net_daily_win_rate_vs_benchmark",
            "median_max_sector_weight",
            "median_top_3_sector_weight",
            "pct_rebalances_sector_cap_binding",
            "mean_one_way_turnover",
        ]
    ]


def compute_daily_with_external_benchmark(
    vintage_daily_returns: pd.DataFrame,
    vintages: pd.DataFrame,
    benchmark_daily_curve: pd.DataFrame,
) -> pd.DataFrame:
    daily = (
        vintage_daily_returns.groupby(["date", "portfolio_name"], sort=True)
        .agg(gross_daily_return=("vintage_daily_return", "mean"), active_vintage_count=("vintage_id", "nunique"))
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
    benchmark = benchmark_daily_curve[benchmark_daily_curve["portfolio_name"] == BENCHMARK_PORTFOLIO].copy()
    dates = sorted(set(pd.to_datetime(benchmark["date"])) | set(pd.to_datetime(daily["date"])) | set(pd.to_datetime(costs["date"])))
    grid = pd.MultiIndex.from_product([dates, portfolio_names], names=["date", "portfolio_name"]).to_frame(index=False)
    frame = grid.merge(daily, on=["date", "portfolio_name"], how="left").merge(costs, on=["date", "portfolio_name"], how="left")
    frame["gross_daily_return"] = frame["gross_daily_return"].fillna(0.0)
    frame["active_vintage_count"] = frame["active_vintage_count"].fillna(0).astype(int)
    frame["transaction_cost"] = frame["transaction_cost"].fillna(0.0)
    frame["net_daily_return"] = frame["gross_daily_return"] - frame["transaction_cost"]
    frame = frame.sort_values(["portfolio_name", "date"]).reset_index(drop=True)
    frame["gross_cumulative_return_index"] = frame.groupby("portfolio_name")["gross_daily_return"].transform(
        cumulative_return_index
    )
    frame["net_cumulative_return_index"] = frame.groupby("portfolio_name")["net_daily_return"].transform(
        cumulative_return_index
    )
    benchmark = benchmark[
        ["date", "gross_daily_return", "net_daily_return"]
    ].rename(
        columns={
            "gross_daily_return": "benchmark_gross_daily_return",
            "net_daily_return": "benchmark_net_daily_return",
        }
    )
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    frame = frame.merge(benchmark, on="date", how="left")
    frame["benchmark_gross_daily_return"] = frame["benchmark_gross_daily_return"].fillna(0.0)
    frame["benchmark_net_daily_return"] = frame["benchmark_net_daily_return"].fillna(0.0)
    frame["gross_excess_daily_return"] = frame["gross_daily_return"] - frame["benchmark_gross_daily_return"]
    frame["net_excess_daily_return"] = frame["net_daily_return"] - frame["benchmark_net_daily_return"]
    frame["gross_excess_cumulative_return_index"] = frame.groupby("portfolio_name")["gross_excess_daily_return"].transform(
        cumulative_return_index
    )
    frame["net_excess_cumulative_return_index"] = frame.groupby("portfolio_name")["net_excess_daily_return"].transform(
        cumulative_return_index
    )
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    return frame[
        [
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
    ]


def compute_regime_performance(daily_curve: pd.DataFrame, *, regimes: list[tuple[str, str, str | None]] | None = None) -> pd.DataFrame:
    regime_defs = regimes or REGIME_DEFINITIONS
    frame = daily_curve.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    min_date = frame["date"].min()
    max_date = frame["date"].max()
    rows = []
    for regime_name, start, end in regime_defs:
        start_ts = max(pd.Timestamp(start), min_date)
        end_ts = min(pd.Timestamp(end) if end else max_date, max_date)
        if start_ts > end_ts:
            continue
        period = frame[(frame["date"] >= start_ts) & (frame["date"] <= end_ts)]
        if period.empty:
            continue
        for portfolio_name, group in period.groupby("portfolio_name", sort=True):
            rows.append({"regime_name": regime_name, "regime_start_date": start_ts.date(), "regime_end_date": end_ts.date()} | _daily_metrics(portfolio_name, group))
    return pd.DataFrame(rows)


def compute_rolling_performance(daily_curve: pd.DataFrame, *, windows: list[int]) -> pd.DataFrame:
    frame = daily_curve.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    rows = []
    for portfolio_name, group in frame.groupby("portfolio_name", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        for window in windows:
            if len(group) < window:
                continue
            for end_idx in range(window - 1, len(group)):
                sample = group.iloc[end_idx - window + 1 : end_idx + 1]
                net = sample["net_daily_return"]
                excess = sample["net_excess_daily_return"]
                ann_return = annualized_return(net)
                ann_vol = annualized_volatility(net)
                rows.append(
                    {
                        "date": sample["date"].iloc[-1].date(),
                        "portfolio_name": portfolio_name,
                        "window_trading_days": int(window),
                        "rolling_net_total_return": float((1 + net).prod() - 1),
                        "rolling_net_excess_return": float((1 + excess).prod() - 1),
                        "rolling_net_annualized_return": ann_return,
                        "rolling_net_annualized_volatility": ann_vol,
                        "rolling_net_sharpe_like": ann_return / ann_vol if ann_vol else None,
                        "rolling_net_max_drawdown": compute_drawdown(net),
                    }
                )
    return pd.DataFrame(rows, columns=ROLLING_PERFORMANCE_COLUMNS)


def compute_contribution_outputs(
    data_root: Path,
    *,
    daily_curve: pd.DataFrame,
    vintages: pd.DataFrame,
    constituents: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_constituents = constituents[constituents["portfolio_name"].isin(CONTRIBUTION_PORTFOLIOS)].copy()
    if selected_constituents.empty:
        empty_sector = pd.DataFrame(columns=SECTOR_CONTRIBUTION_COLUMNS)
        empty_symbol = pd.DataFrame(columns=SYMBOL_CONTRIBUTION_COLUMNS)
        return empty_sector, empty_symbol
    symbols = sorted(selected_constituents["symbol"].astype(str).unique())
    end_date = str(pd.to_datetime(vintages["holding_end_date"]).max().date())
    adjusted_returns, _ = load_adjusted_close_returns(
        daily_bars_path(data_root),
        symbols=symbols,
        start_date=None,
        end_date=end_date,
    )
    daily_dates = sorted(pd.to_datetime(daily_curve["date"].drop_duplicates()))
    contribution_daily = compute_daily_contributions(
        daily_curve=daily_curve,
        vintages=vintages[vintages["portfolio_name"].isin(CONTRIBUTION_PORTFOLIOS)],
        constituents=selected_constituents,
        adjusted_returns=adjusted_returns,
        daily_dates=daily_dates,
    )
    sector_contribution = aggregate_sector_contributions(contribution_daily, daily_curve)
    symbol_contribution = aggregate_symbol_contributions(contribution_daily, daily_curve)
    return sector_contribution, symbol_contribution


def compute_daily_contributions(
    *,
    daily_curve: pd.DataFrame,
    vintages: pd.DataFrame,
    constituents: pd.DataFrame,
    adjusted_returns: pd.DataFrame,
    daily_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    returns = adjusted_returns.set_index(["date", "symbol"])["daily_return"].sort_index()
    active_counts = daily_curve[["date", "portfolio_name", "active_vintage_count"]].copy()
    active_counts["date"] = pd.to_datetime(active_counts["date"])
    rows = []
    for _, vintage in vintages.iterrows():
        start = pd.Timestamp(vintage["holding_start_date"])
        end = pd.Timestamp(vintage["holding_end_date"])
        if pd.isna(start) or pd.isna(end):
            continue
        portfolio_name = str(vintage["portfolio_name"])
        vintage_id = str(vintage["vintage_id"])
        dates = [date for date in daily_dates if start <= date <= end]
        if not dates:
            continue
        weights = constituents[constituents["vintage_id"] == vintage_id][["symbol", "sector", "weight"]].copy()
        index = pd.MultiIndex.from_product([dates, weights["symbol"].astype(str)], names=["date", "symbol"])
        observations = returns.reindex(index).rename("daily_return").reset_index()
        observations = observations.merge(weights, on="symbol", how="left").dropna(subset=["daily_return", "weight"])
        if observations.empty:
            continue
        observations["portfolio_name"] = portfolio_name
        observations["vintage_id"] = vintage_id
        observations["weighted_return_contribution"] = observations["weight"] * observations["daily_return"]
        observations = observations.merge(active_counts, on=["date", "portfolio_name"], how="left")
        observations["active_vintage_count"] = observations["active_vintage_count"].fillna(1).clip(lower=1)
        observations["daily_return_contribution"] = (
            observations["weighted_return_contribution"] / observations["active_vintage_count"]
        )
        observations["daily_weight"] = observations["weight"] / observations["active_vintage_count"]
        rows.append(
            observations[
                [
                    "date",
                    "portfolio_name",
                    "vintage_id",
                    "symbol",
                    "sector",
                    "daily_weight",
                    "daily_return_contribution",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def aggregate_sector_contributions(contribution_daily: pd.DataFrame, daily_curve: pd.DataFrame) -> pd.DataFrame:
    if contribution_daily.empty:
        return pd.DataFrame(columns=SECTOR_CONTRIBUTION_COLUMNS)
    daily_sector = (
        contribution_daily.groupby(["date", "portfolio_name", "sector"], sort=True)
        .agg(
            daily_weight=("daily_weight", "sum"),
            daily_return_contribution=("daily_return_contribution", "sum"),
        )
        .reset_index()
    )
    periods = _period_slices(daily_curve)
    rows = []
    for period_name, start, end in periods:
        period = daily_sector[(daily_sector["date"] >= start) & (daily_sector["date"] <= end)]
        if period.empty:
            continue
        grouped = (
            period.groupby(["portfolio_name", "sector"], sort=True)
            .agg(
                mean_daily_weight=("daily_weight", "mean"),
                total_return_contribution=("daily_return_contribution", "sum"),
                avg_daily_return_contribution=("daily_return_contribution", "mean"),
            )
            .reset_index()
        )
        for portfolio_name, group in grouped.groupby("portfolio_name", sort=True):
            positive_total = group["total_return_contribution"].clip(lower=0).sum()
            absolute_total = group["total_return_contribution"].abs().sum()
            for _, row in group.iterrows():
                rows.append(
                    {
                        "portfolio_name": portfolio_name,
                        "period_name": period_name,
                        "period_start_date": start.date(),
                        "period_end_date": end.date(),
                        "sector": row["sector"],
                        "mean_daily_weight": float(row["mean_daily_weight"]),
                        "total_return_contribution": float(row["total_return_contribution"]),
                        "avg_daily_return_contribution": float(row["avg_daily_return_contribution"]),
                        "share_of_positive_contribution": (
                            float(max(row["total_return_contribution"], 0) / positive_total) if positive_total else 0.0
                        ),
                        "share_of_absolute_contribution": (
                            float(abs(row["total_return_contribution"]) / absolute_total) if absolute_total else 0.0
                        ),
                    }
                )
    return pd.DataFrame(rows, columns=SECTOR_CONTRIBUTION_COLUMNS)


def aggregate_symbol_contributions(contribution_daily: pd.DataFrame, daily_curve: pd.DataFrame) -> pd.DataFrame:
    if contribution_daily.empty:
        return pd.DataFrame(columns=SYMBOL_CONTRIBUTION_COLUMNS)
    filtered = contribution_daily[contribution_daily["portfolio_name"] == "higher_conviction_raw"].copy()
    periods = _period_slices(daily_curve)
    rows = []
    for period_name, start, end in periods:
        period = filtered[(filtered["date"] >= start) & (filtered["date"] <= end)]
        if period.empty:
            continue
        grouped = (
            period.groupby(["portfolio_name", "symbol", "sector"], sort=True)
            .agg(
                active_day_count=("date", "nunique"),
                mean_weight_when_active=("daily_weight", "mean"),
                total_return_contribution=("daily_return_contribution", "sum"),
                avg_daily_return_contribution=("daily_return_contribution", "mean"),
            )
            .reset_index()
        )
        grouped["period_name"] = period_name
        grouped["period_start_date"] = start.date()
        grouped["period_end_date"] = end.date()
        rows.append(grouped[SYMBOL_CONTRIBUTION_COLUMNS])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=SYMBOL_CONTRIBUTION_COLUMNS)


def build_robustness_summary(
    cost_sensitivity: pd.DataFrame,
    sector_cap_sensitivity: pd.DataFrame,
    regime_performance: pd.DataFrame,
    rolling_performance: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, row in cost_sensitivity.iterrows():
        if row["portfolio_name"] == BENCHMARK_PORTFOLIO:
            continue
        rows.append(_summary_row("transaction_cost_sensitivity", f"{row['transaction_cost_bps']:.0f}_bps", row))
    for _, row in sector_cap_sensitivity.iterrows():
        rows.append(_summary_row("sector_cap_sensitivity", str(row["sector_cap"]), row))
    for _, row in regime_performance.iterrows():
        if row["portfolio_name"] in ["higher_conviction_raw", "higher_conviction_sector_capped"]:
            rows.append(_summary_row("regime_performance", str(row["regime_name"]), row))
    if not rolling_performance.empty:
        latest = rolling_performance.sort_values("date").groupby(["portfolio_name", "window_trading_days"], sort=True).tail(1)
        for _, row in latest.iterrows():
            rows.append(
                {
                    "diagnostic_type": "rolling_performance",
                    "variant": f"{int(row['window_trading_days'])}_days_latest",
                    "portfolio_name": row["portfolio_name"],
                    "net_annualized_return": row["rolling_net_annualized_return"],
                    "net_annualized_volatility": row["rolling_net_annualized_volatility"],
                    "net_sharpe_like": row["rolling_net_sharpe_like"],
                    "net_max_drawdown": row["rolling_net_max_drawdown"],
                    "net_excess_annualized_return": None,
                    "net_daily_win_rate_vs_benchmark": None,
                    "mean_one_way_turnover": None,
                    "notes": "Latest completed rolling window.",
                }
            )
    return pd.DataFrame(rows)


def _summary_row(diagnostic_type: str, variant: str, row: pd.Series) -> dict[str, Any]:
    return {
        "diagnostic_type": diagnostic_type,
        "variant": variant,
        "portfolio_name": row["portfolio_name"],
        "net_annualized_return": row.get("net_annualized_return"),
        "net_annualized_volatility": row.get("net_annualized_volatility"),
        "net_sharpe_like": row.get("net_sharpe_like"),
        "net_max_drawdown": row.get("net_max_drawdown"),
        "net_excess_annualized_return": row.get("net_excess_annualized_return"),
        "net_daily_win_rate_vs_benchmark": row.get("net_daily_win_rate_vs_benchmark"),
        "mean_one_way_turnover": row.get("mean_one_way_turnover"),
        "notes": "",
    }


def _daily_metrics(portfolio_name: str, group: pd.DataFrame) -> dict[str, Any]:
    net = group["net_daily_return"]
    excess = group["net_excess_daily_return"]
    ann = annualized_return(net)
    vol = annualized_volatility(net)
    return {
        "portfolio_name": portfolio_name,
        "daily_observation_count": int(len(group)),
        "net_total_return": float((1 + net).prod() - 1),
        "net_annualized_return": ann,
        "net_annualized_volatility": vol,
        "net_sharpe_like": ann / vol if vol else None,
        "net_max_drawdown": compute_drawdown(net),
        "net_excess_total_return": float((1 + excess).prod() - 1),
        "net_excess_annualized_return": annualized_return(excess),
        "net_daily_win_rate_vs_benchmark": float((excess > 0).mean()),
    }


def _period_slices(daily_curve: pd.DataFrame) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    dates = pd.to_datetime(daily_curve["date"])
    min_date = dates.min()
    max_date = dates.max()
    periods = [("full_period", min_date, max_date)]
    for name, start, end in REGIME_DEFINITIONS:
        start_ts = max(pd.Timestamp(start), min_date)
        end_ts = min(pd.Timestamp(end) if end else max_date, max_date)
        if start_ts <= end_ts:
            periods.append((name, start_ts, end_ts))
    return periods


def write_markdown_report(
    path: Path,
    *,
    cost_sensitivity: pd.DataFrame,
    sector_cap_sensitivity: pd.DataFrame,
    regime_performance: pd.DataFrame,
    rolling_performance: pd.DataFrame,
    sector_contribution: pd.DataFrame,
    symbol_contribution: pd.DataFrame,
    robustness_summary: pd.DataFrame,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cost_focus = cost_sensitivity[
        (cost_sensitivity["portfolio_name"] == "higher_conviction_raw")
        & (cost_sensitivity["transaction_cost_bps"].isin([50.0, 100.0, 150.0]))
    ]
    sector_focus = sector_cap_sensitivity[sector_cap_sensitivity["portfolio_name"].str.contains("higher_conviction")]
    regime_focus = regime_performance[regime_performance["portfolio_name"] == "higher_conviction_raw"].sort_values(
        "net_excess_annualized_return",
        ascending=False,
    )
    rolling_focus = (
        rolling_performance[rolling_performance["portfolio_name"] == "higher_conviction_raw"]
        .sort_values("rolling_net_excess_return")
        .head(8)
        if not rolling_performance.empty
        else pd.DataFrame()
    )
    top_symbols = _top_bottom_symbols(symbol_contribution, largest=True)
    bottom_symbols = _top_bottom_symbols(symbol_contribution, largest=False)
    lines = [
        "# Equity Price Strength Equity-Curve Robustness Diagnostics",
        "",
        "## Purpose",
        "",
        "Test whether the higher-conviction price-strength equity curve is robust across costs, sector caps, regimes, rolling windows, and return contributors.",
        "",
        "## Inputs",
        "",
        "- Existing equity-curve daily, summary, vintage, constituent, and sector exposure outputs.",
        "- Source scorecard and adjusted-close daily bars for sector-cap sensitivity and approximate contribution analysis.",
        "",
        "## Executive Summary",
        "",
        _markdown_table(robustness_summary.head(20)),
        "",
        "## Transaction-Cost Sensitivity",
        "",
        "Higher-conviction raw at 50, 100, and 150 bps:",
        "",
        _markdown_table(cost_focus),
        "",
        "## Sector-Cap Sensitivity",
        "",
        _markdown_table(sector_focus),
        "",
        "## Regime Performance",
        "",
        _markdown_table(regime_focus),
        "",
        "## Rolling Performance",
        "",
        "Worst completed rolling windows for higher-conviction raw by net excess return:",
        "",
        _markdown_table(rolling_focus),
        "",
        "## Sector Contribution",
        "",
        _markdown_table(
            sector_contribution[
                (sector_contribution["portfolio_name"] == "higher_conviction_raw")
                & (sector_contribution["period_name"] == "full_period")
            ].sort_values("total_return_contribution", ascending=False)
        ),
        "",
        "## Symbol Contribution",
        "",
        "Top contributors:",
        "",
        _markdown_table(top_symbols),
        "",
        "Bottom contributors:",
        "",
        _markdown_table(bottom_symbols),
        "",
        "## Interpretation",
        "",
        "- Cost sensitivity shows whether higher-conviction raw remains positive after 50, 100, and 150 bps assumptions.",
        "- Sector-cap sensitivity separates concentration reduction from risk-adjusted performance improvement.",
        "- Regime and rolling diagnostics identify whether outperformance is broad or concentrated in a few market environments.",
        "- Contribution diagnostics are approximate and intended to reveal sector or symbol concentration, not exact performance attribution.",
        "",
        "## Important Caveats",
        "",
        "- Research diagnostics only, not production trading logic or trading recommendations.",
        "- Contribution outputs use a simple weighted daily return approximation averaged across active vintages.",
        "- No slippage/spread model beyond transaction-cost bps assumptions.",
        "- No taxes, borrow constraints, limit orders, cash drag, execution constraints, broker integration, or live trading logic.",
        "",
        "## Output File Guide",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {"File": "equity_price_strength_cost_sensitivity.parquet/csv", "Purpose": "Transaction-cost sensitivity."},
                    {"File": "equity_price_strength_sector_cap_sensitivity.parquet/csv", "Purpose": "Sector-cap sensitivity."},
                    {"File": "equity_price_strength_regime_performance.parquet/csv", "Purpose": "Subperiod performance."},
                    {"File": "equity_price_strength_rolling_performance.parquet/csv", "Purpose": "Rolling performance windows."},
                    {"File": "equity_price_strength_sector_contribution.parquet/csv", "Purpose": "Approximate sector contribution."},
                    {"File": "equity_price_strength_symbol_contribution.parquet/csv", "Purpose": "Approximate symbol contribution."},
                    {"File": "equity_price_strength_equity_curve_robustness_summary.parquet/csv", "Purpose": "Compact comparison summary."},
                    {"File": "equity_price_strength_equity_curve_robustness.metadata.json", "Purpose": "Inputs, assumptions, and outputs."},
                ]
            )
        ),
        "",
        "## Suggested Next Step",
        "",
        "Use these diagnostics to decide whether the signal is broad enough for deeper strategy research with fuller execution and risk modeling.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _top_bottom_symbols(symbol_contribution: pd.DataFrame, *, largest: bool) -> pd.DataFrame:
    if symbol_contribution.empty:
        return symbol_contribution
    full = symbol_contribution[
        (symbol_contribution["portfolio_name"] == "higher_conviction_raw")
        & (symbol_contribution["period_name"] == "full_period")
    ]
    return full.sort_values("total_return_contribution", ascending=not largest).head(10)


def _output_paths(data_root: Path) -> dict[str, Path]:
    return {
        "cost_sensitivity_parquet": price_strength_cost_sensitivity_path(data_root),
        "cost_sensitivity_csv": price_strength_cost_sensitivity_csv_path(data_root),
        "sector_cap_sensitivity_parquet": price_strength_sector_cap_sensitivity_path(data_root),
        "sector_cap_sensitivity_csv": price_strength_sector_cap_sensitivity_csv_path(data_root),
        "regime_performance_parquet": price_strength_regime_performance_path(data_root),
        "regime_performance_csv": price_strength_regime_performance_csv_path(data_root),
        "rolling_performance_parquet": price_strength_rolling_performance_path(data_root),
        "rolling_performance_csv": price_strength_rolling_performance_csv_path(data_root),
        "sector_contribution_parquet": price_strength_sector_contribution_path(data_root),
        "sector_contribution_csv": price_strength_sector_contribution_csv_path(data_root),
        "symbol_contribution_parquet": price_strength_symbol_contribution_path(data_root),
        "symbol_contribution_csv": price_strength_symbol_contribution_csv_path(data_root),
        "summary_parquet": price_strength_robustness_summary_path(data_root),
        "summary_csv": price_strength_robustness_summary_csv_path(data_root),
        "markdown_report": price_strength_robustness_report_path(data_root),
        "metadata": price_strength_robustness_metadata_path(data_root),
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
            "Price strength equity-curve robustness diagnostics complete.",
            "",
            f"Cost sensitivity rows: {summary['cost_sensitivity_rows']}",
            f"Sector-cap sensitivity rows: {summary['sector_cap_sensitivity_rows']}",
            f"Regime rows: {summary['regime_rows']}",
            f"Rolling rows: {summary['rolling_rows']}",
            f"Sector contribution rows: {summary['sector_contribution_rows']}",
            f"Symbol contribution rows: {summary['symbol_contribution_rows']}",
            f"Summary rows: {summary['summary_rows']}",
            "",
            f"- Cost sensitivity parquet: {paths['cost_sensitivity_parquet']}",
            f"- Sector-cap sensitivity parquet: {paths['sector_cap_sensitivity_parquet']}",
            f"- Regime performance parquet: {paths['regime_performance_parquet']}",
            f"- Rolling performance parquet: {paths['rolling_performance_parquet']}",
            f"- Sector contribution parquet: {paths['sector_contribution_parquet']}",
            f"- Symbol contribution parquet: {paths['symbol_contribution_parquet']}",
            f"- Robustness summary parquet: {paths['summary_parquet']}",
            f"- Markdown report: {paths['markdown_report']}",
            f"- Metadata JSON: {paths['metadata']}",
        ]
    )
