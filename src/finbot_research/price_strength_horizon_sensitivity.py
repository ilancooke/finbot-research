from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from finbot_research.config import (
    daily_bars_path,
    price_strength_horizon_sensitivity_best_variants_csv_path,
    price_strength_horizon_sensitivity_best_variants_path,
    price_strength_horizon_sensitivity_daily_path,
    price_strength_horizon_sensitivity_metadata_path,
    price_strength_horizon_sensitivity_report_path,
    price_strength_horizon_sensitivity_summary_csv_path,
    price_strength_horizon_sensitivity_summary_path,
    price_strength_horizon_sensitivity_turnover_csv_path,
    price_strength_horizon_sensitivity_turnover_path,
    price_strength_scorecard_path,
    reference_tickers_path,
)
from finbot_research.io import parquet_columns, read_parquet, write_csv, write_parquet
from finbot_research.metadata import write_metadata
from finbot_research.price_strength_equity_curve_backtest import (
    ANNUALIZATION_DAYS,
    BENCHMARK_PORTFOLIO,
    add_sector_exposure_summary_fields,
    build_equity_curve_constituents,
    build_vintages,
    compute_daily_equity_curve,
    compute_performance_summary,
    compute_vintage_daily_returns,
    load_adjusted_close_returns,
    prepare_equity_curve_input,
)
from finbot_research.price_strength_rebalance_feasibility import attach_sector
from finbot_research.validation import ValidationError

DATASET_NAME = "equity_price_strength_horizon_sensitivity"
REBALANCE_FREQUENCIES = ["monthly", "quarterly"]
HOLDING_PERIOD_DAYS = [21, 63, 126, 252]
TRANSACTION_COST_BPS_VALUES = [25.0, 50.0, 100.0]
SECTOR_CAP = 0.30
PORTFOLIO_NAMES = [
    "higher_conviction_raw",
    "higher_conviction_sector_capped",
    BENCHMARK_PORTFOLIO,
]
REQUIRED_SCORECARD_COLUMNS = [
    "symbol",
    "date",
    "price_strength_scorecard_bucket",
    "price_strength_score_v0",
    "is_scorecard_bucket_eligible",
]


def build_price_strength_horizon_sensitivity(data_root: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    scorecard_path = price_strength_scorecard_path(data_root)
    bars_path = daily_bars_path(data_root)
    scorecard = read_parquet(scorecard_path, columns=_scorecard_columns(scorecard_path))
    scorecard, sector_metadata = attach_sector(scorecard, data_root)
    eligible = prepare_equity_curve_input(scorecard)
    if eligible.empty:
        raise ValidationError("No eligible scorecard rows found for horizon sensitivity")
    if "sector" not in eligible.columns or eligible["sector"].isna().any():
        missing_count = int(eligible["sector"].isna().sum()) if "sector" in eligible.columns else len(eligible)
        raise ValidationError(f"price-strength-horizon-sensitivity requires sector for all eligible rows; missing: {missing_count}")

    all_symbols = sorted(eligible["symbol"].astype(str).unique())
    adjusted_returns, trading_dates = load_adjusted_close_returns(bars_path, symbols=all_symbols)

    summary_frames = []
    turnover_frames = []
    daily_frames = []
    for rebalance_frequency in REBALANCE_FREQUENCIES:
        rebalance_rows = select_horizon_rebalance_rows(eligible, rebalance_frequency=rebalance_frequency)
        constituents, sector_exposure = build_equity_curve_constituents(rebalance_rows, sector_cap=SECTOR_CAP)
        constituents = constituents[constituents["portfolio_name"].isin(PORTFOLIO_NAMES)].copy()
        sector_exposure = sector_exposure[sector_exposure["portfolio_name"].isin(PORTFOLIO_NAMES)].copy()
        for holding_period_days in HOLDING_PERIOD_DAYS:
            base_vintages, base_constituents, base_sector_exposure, holding_calendar = build_vintages(
                constituents,
                sector_exposure,
                trading_dates=trading_dates,
                holding_period_days=holding_period_days,
                sector_cap=SECTOR_CAP,
                transaction_cost_bps=0.0,
            )
            vintage_daily = compute_vintage_daily_returns(base_constituents, holding_calendar, adjusted_returns)
            for transaction_cost_bps in TRANSACTION_COST_BPS_VALUES:
                vintages = apply_transaction_cost(base_vintages, transaction_cost_bps=transaction_cost_bps)
                daily_curve = compute_daily_equity_curve(vintage_daily, vintages)
                summary = compute_horizon_summary(
                    daily_curve,
                    vintages,
                    base_sector_exposure,
                    rebalance_frequency=rebalance_frequency,
                    holding_period_days=holding_period_days,
                    transaction_cost_bps=transaction_cost_bps,
                )
                turnover = compute_turnover_summary(
                    vintages,
                    rebalance_frequency=rebalance_frequency,
                    holding_period_days=holding_period_days,
                    transaction_cost_bps=transaction_cost_bps,
                )
                daily = prepare_daily_output(
                    daily_curve,
                    rebalance_frequency=rebalance_frequency,
                    holding_period_days=holding_period_days,
                    transaction_cost_bps=transaction_cost_bps,
                )
                summary_frames.append(summary)
                turnover_frames.append(turnover)
                daily_frames.append(daily)

    summary = pd.concat(summary_frames, ignore_index=True)
    turnover = pd.concat(turnover_frames, ignore_index=True)
    daily = pd.concat(daily_frames, ignore_index=True)
    best_variants = select_best_variants(summary)

    paths = _output_paths(data_root)
    write_parquet(summary, paths["summary_parquet"])
    write_csv(summary, paths["summary_csv"])
    write_parquet(best_variants, paths["best_variants_parquet"])
    write_csv(best_variants, paths["best_variants_csv"])
    write_parquet(daily, paths["daily_parquet"])
    write_parquet(turnover, paths["turnover_parquet"])
    write_csv(turnover, paths["turnover_csv"])
    write_markdown_report(
        paths["markdown_report"],
        summary=summary,
        best_variants=best_variants,
        turnover=turnover,
        daily_written=True,
    )

    input_paths = [scorecard_path, bars_path]
    reference_path = reference_tickers_path(data_root)
    if sector_metadata.get("source") == "reference/tickers.parquet":
        input_paths.append(reference_path)
    metadata_path = write_metadata(
        dataset_name=DATASET_NAME,
        output_path=paths["summary_parquet"],
        input_paths=input_paths,
        dataframe=summary,
        extra_metadata={
            "dataset_type": "research_horizon_sensitivity",
            "research_only": True,
            "output_paths": {key: str(value) for key, value in paths.items()},
            "portfolio_definitions": {
                "higher_conviction_raw": ["higher_conviction_price_strength"],
                "higher_conviction_sector_capped": ["higher_conviction_price_strength"],
                BENCHMARK_PORTFOLIO: "All eligible rows on rebalance dates.",
            },
            "rebalance_frequencies": REBALANCE_FREQUENCIES,
            "rebalance_rules": {
                "monthly": "Last available trading date in each calendar month.",
                "quarterly": "Last available trading date in each calendar quarter.",
            },
            "holding_period_days": HOLDING_PERIOD_DAYS,
            "transaction_cost_bps": TRANSACTION_COST_BPS_VALUES,
            "sector_cap": SECTOR_CAP,
            "turnover_definition": "0.5 * sum(abs(current_weight - previous_weight)); first rebalance uses one-way turnover of 1.0.",
            "benchmark_definition": f"{BENCHMARK_PORTFOLIO} equal-weights all eligible names on the same rebalance dates.",
            "daily_return_source": str(bars_path),
            "annualization_assumption": f"{ANNUALIZATION_DAYS} trading days, risk-free rate 0.",
            "best_turnover_adjusted_variant_formula": "net_excess_annualized_return / max(mean_one_way_turnover, 0.01).",
            "limitations": [
                "Research diagnostics only, not production trading logic.",
                "Uses realized adjusted-close daily returns and simplified transaction costs.",
                "No slippage/spread model beyond transaction-cost bps assumptions.",
                "No taxes, borrow constraints, limit orders, cash drag, execution constraints, broker integration, or live trading logic.",
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
        "summary_rows": int(len(summary)),
        "best_variant_rows": int(len(best_variants)),
        "daily_rows": int(len(daily)),
        "turnover_rows": int(len(turnover)),
        "rebalance_frequencies": REBALANCE_FREQUENCIES,
        "holding_period_days": HOLDING_PERIOD_DAYS,
        "transaction_cost_bps": TRANSACTION_COST_BPS_VALUES,
    }


def _scorecard_columns(path: Path) -> list[str]:
    columns = parquet_columns(path)
    missing = [column for column in REQUIRED_SCORECARD_COLUMNS if column not in columns]
    if missing:
        raise ValidationError(f"scorecard missing required horizon-sensitivity columns: {missing}")
    return [column for column in [*REQUIRED_SCORECARD_COLUMNS, "sector"] if column in columns]


def select_horizon_rebalance_rows(scorecard: pd.DataFrame, *, rebalance_frequency: str) -> pd.DataFrame:
    if rebalance_frequency not in REBALANCE_FREQUENCIES:
        raise ValidationError(f"Unsupported rebalance frequency: {rebalance_frequency}")
    frame = scorecard.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if rebalance_frequency == "monthly":
        frame["rebalance_period"] = frame["date"].dt.to_period("M")
    else:
        frame["rebalance_period"] = frame["date"].dt.to_period("Q")
    rebalance_dates = frame.groupby("rebalance_period", sort=True)["date"].max().rename("rebalance_date")
    frame = frame.merge(rebalance_dates, on="rebalance_period", how="left")
    frame = frame[frame["date"] == frame["rebalance_date"]].drop(columns=["rebalance_period"])
    frame["rebalance_date"] = frame["rebalance_date"].dt.date
    frame["date"] = frame["date"].dt.date
    return frame.sort_values(["rebalance_date", "symbol"]).reset_index(drop=True)


def apply_transaction_cost(vintages: pd.DataFrame, *, transaction_cost_bps: float) -> pd.DataFrame:
    frame = vintages.copy()
    frame["transaction_cost_bps"] = float(transaction_cost_bps)
    frame["transaction_cost"] = frame["one_way_turnover"] * float(transaction_cost_bps) / 10000
    return frame


def compute_horizon_summary(
    daily_curve: pd.DataFrame,
    vintages: pd.DataFrame,
    sector_exposure: pd.DataFrame,
    *,
    rebalance_frequency: str,
    holding_period_days: int,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    summary = compute_performance_summary(daily_curve, vintages)
    enriched_sector = add_sector_exposure_summary_fields(sector_exposure, vintages)
    sector_summary = enriched_sector[
        [
            "portfolio_name",
            "median_max_sector_weight",
            "median_top_3_sector_weight",
            "pct_rebalances_sector_cap_binding",
        ]
    ].drop_duplicates()
    summary = summary.merge(sector_summary, on="portfolio_name", how="left")
    summary["rebalance_frequency"] = rebalance_frequency
    summary["holding_period_days"] = int(holding_period_days)
    summary["transaction_cost_bps"] = float(transaction_cost_bps)
    return summary[
        [
            "rebalance_frequency",
            "holding_period_days",
            "transaction_cost_bps",
            "portfolio_name",
            "rebalance_count",
            "daily_observation_count",
            "mean_active_vintage_count",
            "median_symbol_count_per_vintage",
            "mean_symbol_count_per_vintage",
            "mean_one_way_turnover",
            "median_one_way_turnover",
            "mean_transaction_cost",
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
        ]
    ].sort_values(["rebalance_frequency", "holding_period_days", "transaction_cost_bps", "portfolio_name"]).reset_index(drop=True)


def compute_turnover_summary(
    vintages: pd.DataFrame,
    *,
    rebalance_frequency: str,
    holding_period_days: int,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    rows = []
    for portfolio_name, group in vintages.groupby("portfolio_name", sort=True):
        rows.append(
            {
                "rebalance_frequency": rebalance_frequency,
                "holding_period_days": int(holding_period_days),
                "transaction_cost_bps": float(transaction_cost_bps),
                "portfolio_name": portfolio_name,
                "mean_one_way_turnover": float(group["one_way_turnover"].mean()),
                "median_one_way_turnover": float(group["one_way_turnover"].median()),
                "p75_one_way_turnover": float(group["one_way_turnover"].quantile(0.75)),
                "rebalance_count": int(group["rebalance_date"].nunique()),
                "mean_transaction_cost": float(group["transaction_cost"].mean()),
            }
        )
    return pd.DataFrame(rows)


def prepare_daily_output(
    daily_curve: pd.DataFrame,
    *,
    rebalance_frequency: str,
    holding_period_days: int,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    frame = daily_curve.copy()
    frame["rebalance_frequency"] = rebalance_frequency
    frame["holding_period_days"] = int(holding_period_days)
    frame["transaction_cost_bps"] = float(transaction_cost_bps)
    return frame[
        [
            "date",
            "rebalance_frequency",
            "holding_period_days",
            "transaction_cost_bps",
            "portfolio_name",
            "active_vintage_count",
            "gross_daily_return",
            "net_daily_return",
            "gross_cumulative_return_index",
            "net_cumulative_return_index",
            "benchmark_net_daily_return",
            "net_excess_daily_return",
            "net_excess_cumulative_return_index",
        ]
    ]


def select_best_variants(summary: pd.DataFrame) -> pd.DataFrame:
    candidates = summary[summary["portfolio_name"] != BENCHMARK_PORTFOLIO].copy()
    candidates["turnover_adjusted_score"] = candidates["net_excess_annualized_return"] / candidates[
        "mean_one_way_turnover"
    ].clip(lower=0.01)
    criteria = [
        ("highest_net_annualized_return", "net_annualized_return", False, candidates),
        ("highest_net_excess_annualized_return", "net_excess_annualized_return", False, candidates),
        ("highest_net_sharpe_like", "net_sharpe_like", False, candidates),
        (
            "lowest_net_max_drawdown_among_positive_excess",
            "net_max_drawdown",
            False,
            candidates[candidates["net_excess_annualized_return"] > 0],
        ),
        ("best_turnover_adjusted_variant", "turnover_adjusted_score", False, candidates),
    ]
    rows = []
    for criterion, column, ascending, frame in criteria:
        if frame.empty:
            continue
        row = frame.sort_values(column, ascending=ascending).iloc[0].to_dict()
        row["selection_criterion"] = criterion
        row["selection_metric"] = column
        rows.append(row)
    columns = [
        "selection_criterion",
        "selection_metric",
        "rebalance_frequency",
        "holding_period_days",
        "transaction_cost_bps",
        "portfolio_name",
        "net_annualized_return",
        "net_annualized_volatility",
        "net_sharpe_like",
        "net_max_drawdown",
        "net_excess_annualized_return",
        "net_daily_win_rate_vs_benchmark",
        "mean_one_way_turnover",
        "turnover_adjusted_score",
    ]
    return pd.DataFrame(rows)[columns]


def write_markdown_report(
    path: Path,
    *,
    summary: pd.DataFrame,
    best_variants: pd.DataFrame,
    turnover: pd.DataFrame,
    daily_written: bool,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    high_conviction = summary[summary["portfolio_name"].str.contains("higher_conviction", regex=False)]
    monthly_vs_quarterly = (
        high_conviction.groupby(["rebalance_frequency", "holding_period_days", "transaction_cost_bps"], sort=True)
        .agg(
            best_net_excess_annualized_return=("net_excess_annualized_return", "max"),
            best_net_sharpe_like=("net_sharpe_like", "max"),
            lowest_mean_one_way_turnover=("mean_one_way_turnover", "min"),
        )
        .reset_index()
    )
    cost_stop = high_conviction[
        [
            "rebalance_frequency",
            "holding_period_days",
            "transaction_cost_bps",
            "portfolio_name",
            "net_excess_annualized_return",
        ]
    ].sort_values(["portfolio_name", "rebalance_frequency", "holding_period_days", "transaction_cost_bps"])
    lines = [
        "# Equity Price Strength Horizon Sensitivity",
        "",
        "## Purpose",
        "",
        "Test whether the higher-conviction price-strength signal can work with longer holding periods or lower rebalance frequency.",
        "",
        "## Method",
        "",
        "- Reuses the overlapping-vintage equity-curve method with adjusted-close daily returns.",
        "- Monthly rebalances use the last available trading date in each calendar month.",
        "- Quarterly rebalances use the last available trading date in each calendar quarter.",
        "- Transaction costs are applied on rebalance dates from vintage-level one-way turnover.",
        "",
        "## Sensitivity Grid",
        "",
        f"- Rebalance frequencies: {', '.join(REBALANCE_FREQUENCIES)}.",
        f"- Holding periods: {', '.join(str(value) for value in HOLDING_PERIOD_DAYS)} trading days.",
        f"- Transaction costs: {', '.join(str(int(value)) for value in TRANSACTION_COST_BPS_VALUES)} bps.",
        f"- Daily curve parquet written: {daily_written}.",
        "",
        "## Executive Summary",
        "",
        _markdown_table(best_variants),
        "",
        "## Monthly vs Quarterly Results",
        "",
        _markdown_table(monthly_vs_quarterly),
        "",
        "## Holding-Period Sensitivity",
        "",
        _markdown_table(
            high_conviction[
                [
                    "rebalance_frequency",
                    "holding_period_days",
                    "transaction_cost_bps",
                    "portfolio_name",
                    "net_annualized_return",
                    "net_excess_annualized_return",
                    "net_sharpe_like",
                ]
            ]
        ),
        "",
        "## Transaction-Cost Interaction",
        "",
        _markdown_table(cost_stop),
        "",
        "## Raw vs Sector-Capped Comparison",
        "",
        _markdown_table(
            high_conviction[
                [
                    "rebalance_frequency",
                    "holding_period_days",
                    "transaction_cost_bps",
                    "portfolio_name",
                    "net_sharpe_like",
                    "net_max_drawdown",
                    "median_max_sector_weight",
                    "pct_rebalances_sector_cap_binding",
                ]
            ]
        ),
        "",
        "## Turnover Summary",
        "",
        _markdown_table(turnover[turnover["portfolio_name"].str.contains("higher_conviction", regex=False)]),
        "",
        "## Best Variants",
        "",
        _markdown_table(best_variants),
        "",
        "## Interpretation",
        "",
        "- Quarterly rows show whether lower rebalance frequency preserves excess return while reducing rebalance count.",
        "- Longer holding periods show whether lower responsiveness is compensated by lower turnover and smoother vintage overlap.",
        "- The turnover-adjusted score is net excess annualized return divided by max(mean one-way turnover, 0.01).",
        "- Compare raw and sector-capped rows to see whether the 30% cap improves risk-adjusted results or mainly controls concentration.",
        "",
        "## Important Caveats",
        "",
        "- Research diagnostics only, not production trading logic or trading recommendations.",
        "- Uses simplified transaction costs and adjusted-close returns.",
        "- No slippage/spread model beyond bps assumptions.",
        "- No taxes, borrow constraints, limit orders, cash drag, execution constraints, broker integration, or live trading logic.",
        "",
        "## Output File Guide",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {"File": "equity_price_strength_horizon_sensitivity_summary.parquet/csv", "Purpose": "Main grid summary."},
                    {"File": "equity_price_strength_horizon_sensitivity_best_variants.parquet/csv", "Purpose": "Best variants by selection criterion."},
                    {"File": "equity_price_strength_horizon_sensitivity_daily.parquet", "Purpose": "Daily curves for the grid."},
                    {"File": "equity_price_strength_horizon_sensitivity_turnover.parquet/csv", "Purpose": "Turnover diagnostics."},
                    {"File": "equity_price_strength_horizon_sensitivity.metadata.json", "Purpose": "Inputs, assumptions, and outputs."},
                ]
            )
        ),
        "",
        "## Suggested Next Step",
        "",
        "Use the best turnover-adjusted and risk-adjusted variants to decide whether deeper strategy research should focus on monthly or quarterly expression.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _output_paths(data_root: Path) -> dict[str, Path]:
    return {
        "summary_parquet": price_strength_horizon_sensitivity_summary_path(data_root),
        "summary_csv": price_strength_horizon_sensitivity_summary_csv_path(data_root),
        "best_variants_parquet": price_strength_horizon_sensitivity_best_variants_path(data_root),
        "best_variants_csv": price_strength_horizon_sensitivity_best_variants_csv_path(data_root),
        "daily_parquet": price_strength_horizon_sensitivity_daily_path(data_root),
        "turnover_parquet": price_strength_horizon_sensitivity_turnover_path(data_root),
        "turnover_csv": price_strength_horizon_sensitivity_turnover_csv_path(data_root),
        "markdown_report": price_strength_horizon_sensitivity_report_path(data_root),
        "metadata": price_strength_horizon_sensitivity_metadata_path(data_root),
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
            "Price strength horizon sensitivity complete.",
            "",
            f"Summary rows: {summary['summary_rows']}",
            f"Best variant rows: {summary['best_variant_rows']}",
            f"Daily rows: {summary['daily_rows']}",
            f"Turnover rows: {summary['turnover_rows']}",
            f"Rebalance frequencies: {', '.join(summary['rebalance_frequencies'])}",
            f"Holding periods: {', '.join(str(value) for value in summary['holding_period_days'])}",
            f"Transaction cost bps: {', '.join(str(int(value)) for value in summary['transaction_cost_bps'])}",
            "",
            f"- Summary parquet: {paths['summary_parquet']}",
            f"- Best variants parquet: {paths['best_variants_parquet']}",
            f"- Daily curves parquet: {paths['daily_parquet']}",
            f"- Turnover parquet: {paths['turnover_parquet']}",
            f"- Markdown report: {paths['markdown_report']}",
            f"- Metadata JSON: {paths['metadata']}",
        ]
    )
