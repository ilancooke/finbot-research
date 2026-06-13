from __future__ import annotations

import typer

from finbot_research.bucket_signal_diagnostics import build_bucket_signal_diagnostics
from finbot_research.bucket_signal_diagnostics import terminal_summary as bucket_terminal_summary
from finbot_research.config import get_data_root
from finbot_research.diagnostic_summary import build_diagnostic_summary, terminal_summary
from finbot_research.feature_label_diagnostics import build_feature_label_diagnostics
from finbot_research.price_strength_scorecard import build_price_strength_scorecard
from finbot_research.price_strength_scorecard import terminal_summary as price_strength_terminal_summary
from finbot_research.price_strength_scorecard_v1 import build_price_strength_scorecard_v1
from finbot_research.price_strength_scorecard_v1 import terminal_summary as price_strength_v1_terminal_summary
from finbot_research.price_strength_rebalance_feasibility import build_price_strength_rebalance_feasibility
from finbot_research.price_strength_rebalance_feasibility import terminal_summary as rebalance_terminal_summary
from finbot_research.price_strength_holding_period_simulation import build_price_strength_holding_period_simulation
from finbot_research.price_strength_holding_period_simulation import terminal_summary as holding_period_terminal_summary
from finbot_research.price_strength_equity_curve_backtest import build_price_strength_equity_curve_backtest
from finbot_research.price_strength_equity_curve_backtest import terminal_summary as equity_curve_terminal_summary
from finbot_research.price_strength_equity_curve_robustness import build_price_strength_equity_curve_robustness
from finbot_research.price_strength_equity_curve_robustness import terminal_summary as equity_curve_robustness_terminal_summary
from finbot_research.price_strength_horizon_sensitivity import build_price_strength_horizon_sensitivity
from finbot_research.price_strength_horizon_sensitivity import terminal_summary as horizon_sensitivity_terminal_summary
from finbot_research.price_strength_portfolio_simulation import build_price_strength_portfolio_simulation
from finbot_research.price_strength_portfolio_simulation import terminal_summary as portfolio_terminal_summary
from finbot_research.price_strength_turnover_cost_efficiency import build_price_strength_turnover_cost_efficiency
from finbot_research.price_strength_turnover_cost_efficiency import terminal_summary as turnover_cost_efficiency_terminal_summary

app = typer.Typer(help="Run Finbot research diagnostics.")


@app.callback()
def root() -> None:
    """Run Finbot research diagnostics."""


@app.command("feature-label-diagnostics")
def feature_label_diagnostics_command() -> None:
    """Evaluate feature signal against forward-return labels."""

    data_root = get_data_root()
    output_path, metadata_path, summary = build_feature_label_diagnostics(data_root)
    typer.echo(
        "Wrote "
        f"{output_path} rows={summary['joined_rows']} "
        f"features={len(summary['selected_feature_columns'])}"
    )
    if summary["skipped_feature_columns"]:
        typer.echo(f"Skipped missing features: {', '.join(summary['skipped_feature_columns'])}")
    typer.echo(f"Wrote {metadata_path}")


@app.command("summarize-diagnostics")
def summarize_diagnostics_command(
    include_partial_current_year_in_stability: bool = typer.Option(
        False,
        "--include-partial-current-year-in-stability",
        help="Include the max calendar year in completed-year stability and lookback calculations.",
    ),
) -> None:
    """Summarize feature diagnostics into ranked research report outputs."""

    data_root = get_data_root()
    paths, stats = build_diagnostic_summary(
        data_root,
        include_partial_current_year_in_stability=include_partial_current_year_in_stability,
    )
    typer.echo(terminal_summary(paths, stats))


@app.command("bucket-signal-diagnostics")
def bucket_signal_diagnostics_command() -> None:
    """Analyze interpretable bucket-level feature interactions."""

    data_root = get_data_root()
    paths, summary = build_bucket_signal_diagnostics(data_root)
    typer.echo(bucket_terminal_summary(paths, summary))


@app.command("price-strength-scorecard-v0")
def price_strength_scorecard_v0_command() -> None:
    """Build the research-only price-strength scorecard v0 prototype."""

    data_root = get_data_root()
    paths, summary = build_price_strength_scorecard(data_root)
    typer.echo(price_strength_terminal_summary(paths, summary))


@app.command("price-strength-scorecard-v1")
def price_strength_scorecard_v1_command() -> None:
    """Build the stable research-only price-strength scorecard v1 artifact."""

    data_root = get_data_root()
    paths, summary = build_price_strength_scorecard_v1(data_root)
    typer.echo(price_strength_v1_terminal_summary(paths, summary))


@app.command("price-strength-rebalance-feasibility")
def price_strength_rebalance_feasibility_command(
    rebalance_frequency: str = typer.Option(
        "monthly",
        "--rebalance-frequency",
        help="Rebalance frequency. Currently only monthly is supported.",
    ),
    start_date: str | None = typer.Option(
        None,
        "--start-date",
        help="Optional inclusive start date, formatted YYYY-MM-DD.",
    ),
    end_date: str | None = typer.Option(
        None,
        "--end-date",
        help="Optional inclusive end date, formatted YYYY-MM-DD.",
    ),
) -> None:
    """Evaluate scorecard bucket feasibility on rebalance dates."""

    data_root = get_data_root()
    paths, summary = build_price_strength_rebalance_feasibility(
        data_root,
        rebalance_frequency=rebalance_frequency,
        start_date=start_date,
        end_date=end_date,
    )
    typer.echo(rebalance_terminal_summary(paths, summary))


@app.command("price-strength-holding-period-simulation")
def price_strength_holding_period_simulation_command(
    rebalance_frequency: str = typer.Option(
        "monthly",
        "--rebalance-frequency",
        help="Rebalance frequency. Currently only monthly is supported.",
    ),
    start_date: str | None = typer.Option(
        None,
        "--start-date",
        help="Optional inclusive start date, formatted YYYY-MM-DD.",
    ),
    end_date: str | None = typer.Option(
        None,
        "--end-date",
        help="Optional inclusive end date, formatted YYYY-MM-DD.",
    ),
) -> None:
    """Evaluate 63-trading-day holding-period outcomes for scorecard baskets."""

    data_root = get_data_root()
    paths, summary = build_price_strength_holding_period_simulation(
        data_root,
        rebalance_frequency=rebalance_frequency,
        start_date=start_date,
        end_date=end_date,
    )
    typer.echo(holding_period_terminal_summary(paths, summary))


@app.command("price-strength-portfolio-simulation")
def price_strength_portfolio_simulation_command(
    rebalance_frequency: str = typer.Option(
        "monthly",
        "--rebalance-frequency",
        help="Rebalance frequency. Currently only monthly is supported.",
    ),
    start_date: str | None = typer.Option(None, "--start-date", help="Optional inclusive start date, formatted YYYY-MM-DD."),
    end_date: str | None = typer.Option(None, "--end-date", help="Optional inclusive end date, formatted YYYY-MM-DD."),
    sector_cap: float = typer.Option(0.30, "--sector-cap", help="Maximum sector weight for capped portfolios."),
    transaction_cost_bps: float = typer.Option(25.0, "--transaction-cost-bps", help="One-way transaction cost in basis points."),
) -> None:
    """Run a research-only portfolio simulation for price-strength scorecard v0."""

    data_root = get_data_root()
    paths, summary = build_price_strength_portfolio_simulation(
        data_root,
        rebalance_frequency=rebalance_frequency,
        start_date=start_date,
        end_date=end_date,
        sector_cap=sector_cap,
        transaction_cost_bps=transaction_cost_bps,
    )
    typer.echo(portfolio_terminal_summary(paths, summary))


@app.command("price-strength-equity-curve-backtest")
def price_strength_equity_curve_backtest_command(
    rebalance_frequency: str = typer.Option(
        "monthly",
        "--rebalance-frequency",
        help="Rebalance frequency. Currently only monthly is supported.",
    ),
    holding_period_days: int = typer.Option(63, "--holding-period-days", help="Trading days each rebalance vintage remains active."),
    start_date: str | None = typer.Option(None, "--start-date", help="Optional inclusive start date, formatted YYYY-MM-DD."),
    end_date: str | None = typer.Option(None, "--end-date", help="Optional inclusive end date, formatted YYYY-MM-DD."),
    sector_cap: float = typer.Option(0.30, "--sector-cap", help="Maximum sector weight for capped portfolios."),
    transaction_cost_bps: float = typer.Option(25.0, "--transaction-cost-bps", help="One-way transaction cost in basis points."),
) -> None:
    """Run a research-only overlapping equity-curve backtest for price-strength scorecard v0."""

    data_root = get_data_root()
    paths, summary = build_price_strength_equity_curve_backtest(
        data_root,
        rebalance_frequency=rebalance_frequency,
        holding_period_days=holding_period_days,
        start_date=start_date,
        end_date=end_date,
        sector_cap=sector_cap,
        transaction_cost_bps=transaction_cost_bps,
    )
    typer.echo(equity_curve_terminal_summary(paths, summary))


@app.command("price-strength-equity-curve-robustness")
def price_strength_equity_curve_robustness_command() -> None:
    """Run research-only robustness diagnostics for the price-strength equity curve."""

    data_root = get_data_root()
    paths, summary = build_price_strength_equity_curve_robustness(data_root)
    typer.echo(equity_curve_robustness_terminal_summary(paths, summary))


@app.command("price-strength-horizon-sensitivity")
def price_strength_horizon_sensitivity_command() -> None:
    """Run research-only holding-period and rebalance-frequency sensitivity diagnostics."""

    data_root = get_data_root()
    paths, summary = build_price_strength_horizon_sensitivity(data_root)
    typer.echo(horizon_sensitivity_terminal_summary(paths, summary))


@app.command("price-strength-turnover-cost-efficiency")
def price_strength_turnover_cost_efficiency_command() -> None:
    """Run research-only turnover and cost-efficiency diagnostics for horizon sensitivity."""

    data_root = get_data_root()
    paths, summary = build_price_strength_turnover_cost_efficiency(data_root)
    typer.echo(turnover_cost_efficiency_terminal_summary(paths, summary))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
