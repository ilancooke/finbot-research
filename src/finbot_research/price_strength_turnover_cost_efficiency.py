from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from finbot_research.config import (
    price_strength_horizon_sensitivity_summary_path,
    price_strength_horizon_sensitivity_turnover_path,
    price_strength_turnover_cost_efficiency_best_variants_csv_path,
    price_strength_turnover_cost_efficiency_best_variants_path,
    price_strength_turnover_cost_efficiency_csv_path,
    price_strength_turnover_cost_efficiency_focus_csv_path,
    price_strength_turnover_cost_efficiency_focus_path,
    price_strength_turnover_cost_efficiency_metadata_path,
    price_strength_turnover_cost_efficiency_path,
    price_strength_turnover_cost_efficiency_report_path,
)
from finbot_research.io import read_parquet, write_csv, write_parquet
from finbot_research.metadata import write_metadata
from finbot_research.validation import ValidationError

DATASET_NAME = "equity_price_strength_turnover_cost_efficiency"
REBALANCES_PER_YEAR = {"monthly": 12.0, "quarterly": 4.0}
FOCUS_VARIANTS = [
    ("monthly", 63, "higher_conviction_raw"),
    ("monthly", 63, "higher_conviction_sector_capped"),
    ("quarterly", 63, "higher_conviction_raw"),
    ("quarterly", 63, "higher_conviction_sector_capped"),
    ("quarterly", 126, "higher_conviction_raw"),
    ("quarterly", 126, "higher_conviction_sector_capped"),
]
EFFICIENCY_COLUMNS = [
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
    "median_one_way_turnover",
    "rebalances_per_year",
    "annualized_one_way_turnover",
    "annualized_two_way_turnover",
    "mean_transaction_cost",
    "estimated_annualized_cost_drag",
    "net_excess_return_per_annualized_one_way_turnover",
    "net_excess_return_per_annualized_two_way_turnover",
    "net_excess_return_per_annualized_cost_drag",
    "cost_efficiency_score",
]


def build_price_strength_turnover_cost_efficiency(data_root: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    summary_path = price_strength_horizon_sensitivity_summary_path(data_root)
    turnover_path = price_strength_horizon_sensitivity_turnover_path(data_root)
    summary = read_parquet(summary_path)
    turnover = read_parquet(turnover_path)
    efficiency = compute_efficiency_table(summary, turnover)
    focus = filter_focus_variants(efficiency)
    best_variants = select_best_efficiency_variants(efficiency)

    paths = _output_paths(data_root)
    write_parquet(efficiency, paths["efficiency_parquet"])
    write_csv(efficiency, paths["efficiency_csv"])
    write_parquet(focus, paths["focus_parquet"])
    write_csv(focus, paths["focus_csv"])
    write_parquet(best_variants, paths["best_variants_parquet"])
    write_csv(best_variants, paths["best_variants_csv"])
    write_markdown_report(
        paths["markdown_report"],
        efficiency=efficiency,
        focus=focus,
        best_variants=best_variants,
    )
    metadata_path = write_metadata(
        dataset_name=DATASET_NAME,
        output_path=paths["efficiency_parquet"],
        input_paths=[summary_path, turnover_path],
        dataframe=efficiency,
        extra_metadata={
            "dataset_type": "research_turnover_cost_efficiency",
            "research_only": True,
            "output_paths": {key: str(value) for key, value in paths.items()},
            "rebalances_per_year_assumptions": REBALANCES_PER_YEAR,
            "annualized_turnover_formulas": {
                "annualized_one_way_turnover": "mean_one_way_turnover * rebalances_per_year",
                "annualized_two_way_turnover": "2 * mean_one_way_turnover * rebalances_per_year",
            },
            "annualized_cost_drag_formula": "mean_transaction_cost * rebalances_per_year",
            "efficiency_ratio_formulas": {
                "net_excess_return_per_annualized_one_way_turnover": "net_excess_annualized_return / max(annualized_one_way_turnover, 0.01)",
                "net_excess_return_per_annualized_two_way_turnover": "net_excess_annualized_return / max(annualized_two_way_turnover, 0.01)",
                "net_excess_return_per_annualized_cost_drag": "net_excess_annualized_return / max(estimated_annualized_cost_drag, 0.0001)",
            },
            "cost_efficiency_score_formula": "net_excess_annualized_return - estimated_annualized_cost_drag - max(abs(net_max_drawdown) - 0.50, 0) * 0.25",
            "focus_variant_definitions": [
                {
                    "rebalance_frequency": frequency,
                    "holding_period_days": holding_period_days,
                    "portfolio_name": portfolio_name,
                }
                for frequency, holding_period_days, portfolio_name in FOCUS_VARIANTS
            ],
            "best_practical_default_selection_rule": (
                "Among non-baseline rows with net_excess_annualized_return > 0, "
                "net_daily_win_rate_vs_benchmark >= 0.50, net_max_drawdown >= -0.65, "
                "and transaction_cost_bps >= 50, choose highest cost_efficiency_score. "
                "If none qualify, choose highest cost_efficiency_score and flag that constraints were not met."
            ),
            "limitations": [
                "Research diagnostic only, not production trading logic.",
                "Uses horizon sensitivity outputs; does not rerun a backtest.",
                "Cost-efficiency score is a heuristic ranking aid, not an optimized objective.",
                "Annualized turnover assumes monthly has 12 and quarterly has 4 rebalances per year.",
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
        "efficiency_rows": int(len(efficiency)),
        "focus_rows": int(len(focus)),
        "best_variant_rows": int(len(best_variants)),
    }


def compute_efficiency_table(summary: pd.DataFrame, turnover: pd.DataFrame | None = None) -> pd.DataFrame:
    required = [
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
        "median_one_way_turnover",
        "mean_transaction_cost",
    ]
    missing = [column for column in required if column not in summary.columns]
    if missing:
        raise ValidationError(f"horizon sensitivity summary missing required columns: {missing}")
    frame = summary.copy()
    if turnover is not None and not turnover.empty:
        _validate_turnover_input(frame, turnover)
    frame["rebalances_per_year"] = frame["rebalance_frequency"].map(REBALANCES_PER_YEAR)
    if frame["rebalances_per_year"].isna().any():
        unsupported = sorted(frame.loc[frame["rebalances_per_year"].isna(), "rebalance_frequency"].dropna().unique())
        raise ValidationError(f"Unsupported rebalance frequencies for efficiency diagnostics: {unsupported}")
    frame["annualized_one_way_turnover"] = frame["mean_one_way_turnover"] * frame["rebalances_per_year"]
    frame["annualized_two_way_turnover"] = 2 * frame["mean_one_way_turnover"] * frame["rebalances_per_year"]
    frame["estimated_annualized_cost_drag"] = frame["mean_transaction_cost"] * frame["rebalances_per_year"]
    frame["net_excess_return_per_annualized_one_way_turnover"] = frame["net_excess_annualized_return"] / frame[
        "annualized_one_way_turnover"
    ].clip(lower=0.01)
    frame["net_excess_return_per_annualized_two_way_turnover"] = frame["net_excess_annualized_return"] / frame[
        "annualized_two_way_turnover"
    ].clip(lower=0.01)
    frame["net_excess_return_per_annualized_cost_drag"] = frame["net_excess_annualized_return"] / frame[
        "estimated_annualized_cost_drag"
    ].clip(lower=0.0001)
    drawdown_penalty = (frame["net_max_drawdown"].abs() - 0.50).clip(lower=0.0) * 0.25
    frame["cost_efficiency_score"] = (
        frame["net_excess_annualized_return"] - frame["estimated_annualized_cost_drag"] - drawdown_penalty
    )
    return frame[EFFICIENCY_COLUMNS].sort_values(
        ["rebalance_frequency", "holding_period_days", "transaction_cost_bps", "portfolio_name"]
    ).reset_index(drop=True)


def _validate_turnover_input(summary: pd.DataFrame, turnover: pd.DataFrame) -> None:
    keys = ["rebalance_frequency", "holding_period_days", "transaction_cost_bps", "portfolio_name"]
    missing = [column for column in [*keys, "mean_one_way_turnover", "mean_transaction_cost"] if column not in turnover.columns]
    if missing:
        raise ValidationError(f"horizon sensitivity turnover table missing required columns: {missing}")
    joined = summary[keys + ["mean_one_way_turnover", "mean_transaction_cost"]].merge(
        turnover[keys + ["mean_one_way_turnover", "mean_transaction_cost"]],
        on=keys,
        how="left",
        suffixes=("_summary", "_turnover"),
    )
    if joined["mean_one_way_turnover_turnover"].isna().any():
        raise ValidationError("horizon sensitivity turnover table is missing rows for summary combinations")


def filter_focus_variants(efficiency: pd.DataFrame) -> pd.DataFrame:
    masks = []
    for frequency, holding_period_days, portfolio_name in FOCUS_VARIANTS:
        masks.append(
            (efficiency["rebalance_frequency"] == frequency)
            & (efficiency["holding_period_days"] == holding_period_days)
            & (efficiency["portfolio_name"] == portfolio_name)
        )
    if not masks:
        return efficiency.iloc[0:0].copy()
    mask = masks[0]
    for next_mask in masks[1:]:
        mask = mask | next_mask
    return efficiency[mask].sort_values(
        ["rebalance_frequency", "holding_period_days", "transaction_cost_bps", "portfolio_name"]
    ).reset_index(drop=True)


def select_best_efficiency_variants(efficiency: pd.DataFrame) -> pd.DataFrame:
    candidates = efficiency[efficiency["portfolio_name"] != "eligible_universe_baseline"].copy()
    criteria = [
        ("highest_net_excess_annualized_return", "net_excess_annualized_return", False, candidates),
        ("highest_net_sharpe_like", "net_sharpe_like", False, candidates),
        (
            "lowest_annualized_one_way_turnover_among_positive_excess",
            "annualized_one_way_turnover",
            True,
            candidates[candidates["net_excess_annualized_return"] > 0],
        ),
        (
            "highest_excess_per_annualized_turnover",
            "net_excess_return_per_annualized_one_way_turnover",
            False,
            candidates,
        ),
        (
            "highest_excess_per_annualized_cost_drag",
            "net_excess_return_per_annualized_cost_drag",
            False,
            candidates,
        ),
        ("highest_cost_efficiency_score", "cost_efficiency_score", False, candidates),
    ]
    rows = []
    for criterion, column, ascending, frame in criteria:
        if frame.empty:
            continue
        rows.append(_best_row(frame, criterion=criterion, column=column, ascending=ascending, constraints_met=True))
    practical = candidates[
        (candidates["net_excess_annualized_return"] > 0)
        & (candidates["net_daily_win_rate_vs_benchmark"] >= 0.50)
        & (candidates["net_max_drawdown"] >= -0.65)
        & (candidates["transaction_cost_bps"] >= 50)
    ]
    constraints_met = not practical.empty
    frame = practical if constraints_met else candidates
    if not frame.empty:
        rows.append(
            _best_row(
                frame,
                criterion="best_practical_default_candidate",
                column="cost_efficiency_score",
                ascending=False,
                constraints_met=constraints_met,
            )
        )
    columns = [
        "selection_criterion",
        "selection_metric",
        "constraints_met",
        *EFFICIENCY_COLUMNS,
    ]
    return pd.DataFrame(rows)[columns]


def _best_row(
    frame: pd.DataFrame,
    *,
    criterion: str,
    column: str,
    ascending: bool,
    constraints_met: bool,
) -> dict[str, Any]:
    row = frame.sort_values(column, ascending=ascending).iloc[0].to_dict()
    row["selection_criterion"] = criterion
    row["selection_metric"] = column
    row["constraints_met"] = constraints_met
    return row


def write_markdown_report(
    path: Path,
    *,
    efficiency: pd.DataFrame,
    focus: pd.DataFrame,
    best_variants: pd.DataFrame,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    monthly_quarterly = focus[
        [
            "rebalance_frequency",
            "holding_period_days",
            "transaction_cost_bps",
            "portfolio_name",
            "annualized_one_way_turnover",
            "estimated_annualized_cost_drag",
            "net_excess_annualized_return",
            "cost_efficiency_score",
        ]
    ]
    cost_adjusted = efficiency[
        (efficiency["portfolio_name"].str.contains("higher_conviction", regex=False))
        & (efficiency["transaction_cost_bps"].isin([50.0, 100.0]))
    ][
        [
            "rebalance_frequency",
            "holding_period_days",
            "transaction_cost_bps",
            "portfolio_name",
            "net_excess_annualized_return",
            "estimated_annualized_cost_drag",
            "net_excess_return_per_annualized_one_way_turnover",
            "cost_efficiency_score",
        ]
    ]
    lines = [
        "# Equity Price Strength Turnover / Cost Efficiency",
        "",
        "## Purpose",
        "",
        "Compare horizon-sensitivity variants after annualizing turnover and estimated transaction-cost drag.",
        "",
        "## Inputs",
        "",
        "- Horizon sensitivity summary parquet.",
        "- Horizon sensitivity turnover parquet for validation and supplemental turnover context.",
        "",
        "## Method",
        "",
        "- Monthly assumes 12 rebalances per year; quarterly assumes 4.",
        "- Annualized one-way turnover is mean one-way turnover times rebalances per year.",
        "- Estimated annualized cost drag is mean transaction cost times rebalances per year.",
        "- Cost-efficiency score is heuristic: net excess annualized return minus annualized cost drag minus a drawdown penalty above 50%.",
        "",
        "## Executive Summary",
        "",
        _markdown_table(best_variants),
        "",
        "## Focus Variant Comparison",
        "",
        _markdown_table(focus),
        "",
        "## Monthly vs Quarterly Annualized Turnover",
        "",
        _markdown_table(monthly_quarterly),
        "",
        "## Cost-Adjusted Results",
        "",
        _markdown_table(cost_adjusted),
        "",
        "## Best Variants",
        "",
        _markdown_table(best_variants),
        "",
        "## Interpretation",
        "",
        "- Monthly / 63d remains strong only if its excess return offsets more frequent rebalance events.",
        "- Quarterly / 63d and quarterly / 126d are useful conservative comparisons when cost drag and annualized turnover matter.",
        "- The practical default rule requires positive excess return, at least 50% daily win rate versus benchmark, drawdown no worse than -65%, and at least 50 bps transaction cost.",
        "",
        "## Important Caveats",
        "",
        "- Research diagnostic only, not production trading logic or a trading recommendation.",
        "- The composite score is heuristic and for ranking only.",
        "- Annualized turnover assumes fixed monthly or quarterly rebalance counts.",
        "- This command does not rerun the backtest; it depends on the existing horizon sensitivity outputs.",
        "",
        "## Output File Guide",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {"File": "equity_price_strength_turnover_cost_efficiency.parquet/csv", "Purpose": "Full efficiency table."},
                    {"File": "equity_price_strength_turnover_cost_efficiency_focus.parquet/csv", "Purpose": "Focus variant comparison."},
                    {"File": "equity_price_strength_turnover_cost_efficiency_best_variants.parquet/csv", "Purpose": "Best rows by selection criterion."},
                    {"File": "equity_price_strength_turnover_cost_efficiency.metadata.json", "Purpose": "Inputs, formulas, assumptions, and output paths."},
                ]
            )
        ),
        "",
        "## Suggested Decision",
        "",
        "Use the practical default candidate as the scorecard_v1 research default and keep the best quarterly 63d/126d row as a conservative comparison.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _output_paths(data_root: Path) -> dict[str, Path]:
    return {
        "efficiency_parquet": price_strength_turnover_cost_efficiency_path(data_root),
        "efficiency_csv": price_strength_turnover_cost_efficiency_csv_path(data_root),
        "focus_parquet": price_strength_turnover_cost_efficiency_focus_path(data_root),
        "focus_csv": price_strength_turnover_cost_efficiency_focus_csv_path(data_root),
        "best_variants_parquet": price_strength_turnover_cost_efficiency_best_variants_path(data_root),
        "best_variants_csv": price_strength_turnover_cost_efficiency_best_variants_csv_path(data_root),
        "markdown_report": price_strength_turnover_cost_efficiency_report_path(data_root),
        "metadata": price_strength_turnover_cost_efficiency_metadata_path(data_root),
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
            "Price strength turnover/cost efficiency complete.",
            "",
            f"Efficiency rows: {summary['efficiency_rows']}",
            f"Focus rows: {summary['focus_rows']}",
            f"Best variant rows: {summary['best_variant_rows']}",
            "",
            f"- Efficiency parquet: {paths['efficiency_parquet']}",
            f"- Focus parquet: {paths['focus_parquet']}",
            f"- Best variants parquet: {paths['best_variants_parquet']}",
            f"- Markdown report: {paths['markdown_report']}",
            f"- Metadata JSON: {paths['metadata']}",
        ]
    )
