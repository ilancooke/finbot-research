from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DOTENV_FILE = ".env"


def parse_dotenv_value(key: str, dotenv_path: Path | None = None) -> str | None:
    dotenv_path = dotenv_path or Path.cwd() / DEFAULT_DOTENV_FILE
    if not dotenv_path.exists():
        return None

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        env_key, env_value = line.split("=", 1)
        if env_key.strip() == key:
            return env_value.strip().strip('"').strip("'")
    return None


def get_env(key: str, default: str = "", dotenv_path: Path | None = None) -> str:
    value = os.getenv(key)
    if value:
        return value

    dotenv_value = parse_dotenv_value(key, dotenv_path=dotenv_path)
    if dotenv_value:
        return dotenv_value

    return default


def get_data_root() -> Path:
    value = get_env("FINBOT_DATA_ROOT")
    if not value:
        raise RuntimeError("Missing FINBOT_DATA_ROOT. Set it to the shared Finbot data directory.")
    return Path(value).expanduser().resolve()


def price_features_path(data_root: Path | None = None) -> Path:
    root = data_root or get_data_root()
    return root / "features" / "equity_price_features.parquet"


def relative_features_path(data_root: Path | None = None) -> Path:
    root = data_root or get_data_root()
    return root / "features" / "equity_relative_features.parquet"


def labels_path(data_root: Path | None = None) -> Path:
    root = data_root or get_data_root()
    return root / "labels" / "equity_forward_return_labels.parquet"


def feature_label_diagnostics_dir(data_root: Path | None = None) -> Path:
    root = data_root or get_data_root()
    return root / "research" / "feature_label_diagnostics"


def feature_label_diagnostics_path(data_root: Path | None = None) -> Path:
    return feature_label_diagnostics_dir(data_root) / "equity_price_signal_diagnostics.parquet"


def feature_label_summary_path(data_root: Path | None = None) -> Path:
    return feature_label_diagnostics_dir(data_root) / "equity_price_signal_summary.parquet"


def feature_label_summary_csv_path(data_root: Path | None = None) -> Path:
    return feature_label_diagnostics_dir(data_root) / "equity_price_signal_summary.csv"


def feature_label_report_path(data_root: Path | None = None) -> Path:
    return feature_label_diagnostics_dir(data_root) / "equity_price_signal_report.md"


def feature_label_year_spreads_path(data_root: Path | None = None) -> Path:
    return feature_label_diagnostics_dir(data_root) / "equity_price_signal_year_spreads.parquet"


def feature_label_year_spreads_csv_path(data_root: Path | None = None) -> Path:
    return feature_label_diagnostics_dir(data_root) / "equity_price_signal_year_spreads.csv"


def feature_label_lookback_summary_path(data_root: Path | None = None) -> Path:
    return feature_label_diagnostics_dir(data_root) / "equity_price_signal_lookback_summary.parquet"


def feature_label_lookback_summary_csv_path(data_root: Path | None = None) -> Path:
    return feature_label_diagnostics_dir(data_root) / "equity_price_signal_lookback_summary.csv"


def feature_label_current_year_snapshot_path(data_root: Path | None = None) -> Path:
    return feature_label_diagnostics_dir(data_root) / "equity_price_signal_current_year_snapshot.parquet"


def feature_label_current_year_snapshot_csv_path(data_root: Path | None = None) -> Path:
    return feature_label_diagnostics_dir(data_root) / "equity_price_signal_current_year_snapshot.csv"


def bucket_signal_diagnostics_dir(data_root: Path | None = None) -> Path:
    root = data_root or get_data_root()
    return root / "research" / "bucket_signal_diagnostics"


def bucket_signal_summary_path(data_root: Path | None = None) -> Path:
    return bucket_signal_diagnostics_dir(data_root) / "equity_bucket_signal_summary.parquet"


def bucket_signal_summary_csv_path(data_root: Path | None = None) -> Path:
    return bucket_signal_diagnostics_dir(data_root) / "equity_bucket_signal_summary.csv"


def bucket_candidate_rules_path(data_root: Path | None = None) -> Path:
    return bucket_signal_diagnostics_dir(data_root) / "equity_bucket_candidate_rules.parquet"


def bucket_candidate_rules_csv_path(data_root: Path | None = None) -> Path:
    return bucket_signal_diagnostics_dir(data_root) / "equity_bucket_candidate_rules.csv"


def bucket_candidate_rule_years_path(data_root: Path | None = None) -> Path:
    return bucket_signal_diagnostics_dir(data_root) / "equity_bucket_candidate_rule_years.parquet"


def bucket_candidate_rule_years_csv_path(data_root: Path | None = None) -> Path:
    return bucket_signal_diagnostics_dir(data_root) / "equity_bucket_candidate_rule_years.csv"


def bucket_candidate_rule_stability_path(data_root: Path | None = None) -> Path:
    return bucket_signal_diagnostics_dir(data_root) / "equity_bucket_candidate_rule_stability.parquet"


def bucket_candidate_rule_stability_csv_path(data_root: Path | None = None) -> Path:
    return bucket_signal_diagnostics_dir(data_root) / "equity_bucket_candidate_rule_stability.csv"


def bucket_signal_report_path(data_root: Path | None = None) -> Path:
    return bucket_signal_diagnostics_dir(data_root) / "equity_bucket_signal_report.md"


def price_strength_scorecard_dir(data_root: Path | None = None) -> Path:
    root = data_root or get_data_root()
    return root / "research" / "price_strength_scorecard_v0"


def price_strength_scorecard_path(data_root: Path | None = None) -> Path:
    return price_strength_scorecard_dir(data_root) / "equity_price_strength_scorecard_v0.parquet"


def price_strength_scorecard_csv_path(data_root: Path | None = None) -> Path:
    return price_strength_scorecard_dir(data_root) / "equity_price_strength_scorecard_v0.csv"


def price_strength_scorecard_summary_path(data_root: Path | None = None) -> Path:
    return price_strength_scorecard_dir(data_root) / "equity_price_strength_scorecard_v0_summary.parquet"


def price_strength_scorecard_summary_csv_path(data_root: Path | None = None) -> Path:
    return price_strength_scorecard_dir(data_root) / "equity_price_strength_scorecard_v0_summary.csv"


def price_strength_scorecard_years_path(data_root: Path | None = None) -> Path:
    return price_strength_scorecard_dir(data_root) / "equity_price_strength_scorecard_v0_years.parquet"


def price_strength_scorecard_years_csv_path(data_root: Path | None = None) -> Path:
    return price_strength_scorecard_dir(data_root) / "equity_price_strength_scorecard_v0_years.csv"


def price_strength_scorecard_stability_path(data_root: Path | None = None) -> Path:
    return price_strength_scorecard_dir(data_root) / "equity_price_strength_scorecard_v0_stability.parquet"


def price_strength_scorecard_stability_csv_path(data_root: Path | None = None) -> Path:
    return price_strength_scorecard_dir(data_root) / "equity_price_strength_scorecard_v0_stability.csv"


def price_strength_scorecard_report_path(data_root: Path | None = None) -> Path:
    return price_strength_scorecard_dir(data_root) / "equity_price_strength_scorecard_v0_report.md"


def reference_tickers_path(data_root: Path | None = None) -> Path:
    root = data_root or get_data_root()
    return root / "reference" / "tickers.parquet"


def price_strength_rebalance_feasibility_dir(data_root: Path | None = None) -> Path:
    root = data_root or get_data_root()
    return root / "research" / "price_strength_rebalance_feasibility"


def price_strength_rebalance_bucket_counts_path(data_root: Path | None = None) -> Path:
    return price_strength_rebalance_feasibility_dir(data_root) / "equity_price_strength_rebalance_bucket_counts.parquet"


def price_strength_rebalance_bucket_counts_csv_path(data_root: Path | None = None) -> Path:
    return price_strength_rebalance_feasibility_dir(data_root) / "equity_price_strength_rebalance_bucket_counts.csv"


def price_strength_rebalance_bucket_count_summary_path(data_root: Path | None = None) -> Path:
    return (
        price_strength_rebalance_feasibility_dir(data_root)
        / "equity_price_strength_rebalance_bucket_count_summary.parquet"
    )


def price_strength_rebalance_bucket_count_summary_csv_path(data_root: Path | None = None) -> Path:
    return (
        price_strength_rebalance_feasibility_dir(data_root)
        / "equity_price_strength_rebalance_bucket_count_summary.csv"
    )


def price_strength_rebalance_sector_composition_path(data_root: Path | None = None) -> Path:
    return (
        price_strength_rebalance_feasibility_dir(data_root)
        / "equity_price_strength_rebalance_sector_composition.parquet"
    )


def price_strength_rebalance_sector_composition_csv_path(data_root: Path | None = None) -> Path:
    return (
        price_strength_rebalance_feasibility_dir(data_root)
        / "equity_price_strength_rebalance_sector_composition.csv"
    )


def price_strength_rebalance_sector_concentration_path(data_root: Path | None = None) -> Path:
    return (
        price_strength_rebalance_feasibility_dir(data_root)
        / "equity_price_strength_rebalance_sector_concentration.parquet"
    )


def price_strength_rebalance_sector_concentration_csv_path(data_root: Path | None = None) -> Path:
    return (
        price_strength_rebalance_feasibility_dir(data_root)
        / "equity_price_strength_rebalance_sector_concentration.csv"
    )


def price_strength_rebalance_turnover_path(data_root: Path | None = None) -> Path:
    return price_strength_rebalance_feasibility_dir(data_root) / "equity_price_strength_rebalance_turnover.parquet"


def price_strength_rebalance_turnover_csv_path(data_root: Path | None = None) -> Path:
    return price_strength_rebalance_feasibility_dir(data_root) / "equity_price_strength_rebalance_turnover.csv"


def price_strength_rebalance_feasibility_path(data_root: Path | None = None) -> Path:
    return price_strength_rebalance_feasibility_dir(data_root) / "equity_price_strength_rebalance_feasibility.parquet"


def price_strength_rebalance_feasibility_csv_path(data_root: Path | None = None) -> Path:
    return price_strength_rebalance_feasibility_dir(data_root) / "equity_price_strength_rebalance_feasibility.csv"


def price_strength_rebalance_feasibility_report_path(data_root: Path | None = None) -> Path:
    return price_strength_rebalance_feasibility_dir(data_root) / "equity_price_strength_rebalance_feasibility_report.md"


def price_strength_holding_period_dir(data_root: Path | None = None) -> Path:
    root = data_root or get_data_root()
    return root / "research" / "price_strength_holding_period_simulation"


def price_strength_holding_period_rebalance_results_path(data_root: Path | None = None) -> Path:
    return price_strength_holding_period_dir(data_root) / "equity_price_strength_holding_period_rebalance_results.parquet"


def price_strength_holding_period_rebalance_results_csv_path(data_root: Path | None = None) -> Path:
    return price_strength_holding_period_dir(data_root) / "equity_price_strength_holding_period_rebalance_results.csv"


def price_strength_holding_period_summary_path(data_root: Path | None = None) -> Path:
    return price_strength_holding_period_dir(data_root) / "equity_price_strength_holding_period_summary.parquet"


def price_strength_holding_period_summary_csv_path(data_root: Path | None = None) -> Path:
    return price_strength_holding_period_dir(data_root) / "equity_price_strength_holding_period_summary.csv"


def price_strength_holding_period_sector_composition_path(data_root: Path | None = None) -> Path:
    return price_strength_holding_period_dir(data_root) / "equity_price_strength_holding_period_sector_composition.parquet"


def price_strength_holding_period_sector_composition_csv_path(data_root: Path | None = None) -> Path:
    return price_strength_holding_period_dir(data_root) / "equity_price_strength_holding_period_sector_composition.csv"


def price_strength_holding_period_sector_concentration_path(data_root: Path | None = None) -> Path:
    return price_strength_holding_period_dir(data_root) / "equity_price_strength_holding_period_sector_concentration.parquet"


def price_strength_holding_period_sector_concentration_csv_path(data_root: Path | None = None) -> Path:
    return price_strength_holding_period_dir(data_root) / "equity_price_strength_holding_period_sector_concentration.csv"


def price_strength_holding_period_turnover_path(data_root: Path | None = None) -> Path:
    return price_strength_holding_period_dir(data_root) / "equity_price_strength_holding_period_turnover.parquet"


def price_strength_holding_period_turnover_csv_path(data_root: Path | None = None) -> Path:
    return price_strength_holding_period_dir(data_root) / "equity_price_strength_holding_period_turnover.csv"


def price_strength_holding_period_report_path(data_root: Path | None = None) -> Path:
    return price_strength_holding_period_dir(data_root) / "equity_price_strength_holding_period_simulation_report.md"


def price_strength_holding_period_metadata_path(data_root: Path | None = None) -> Path:
    return price_strength_holding_period_dir(data_root) / "equity_price_strength_holding_period_simulation.metadata.json"


def price_strength_portfolio_simulation_dir(data_root: Path | None = None) -> Path:
    root = data_root or get_data_root()
    return root / "research" / "price_strength_portfolio_simulation"


def price_strength_portfolio_rebalance_results_path(data_root: Path | None = None) -> Path:
    return price_strength_portfolio_simulation_dir(data_root) / "equity_price_strength_portfolio_rebalance_results.parquet"


def price_strength_portfolio_rebalance_results_csv_path(data_root: Path | None = None) -> Path:
    return price_strength_portfolio_simulation_dir(data_root) / "equity_price_strength_portfolio_rebalance_results.csv"


def price_strength_portfolio_summary_path(data_root: Path | None = None) -> Path:
    return price_strength_portfolio_simulation_dir(data_root) / "equity_price_strength_portfolio_summary.parquet"


def price_strength_portfolio_summary_csv_path(data_root: Path | None = None) -> Path:
    return price_strength_portfolio_simulation_dir(data_root) / "equity_price_strength_portfolio_summary.csv"


def price_strength_portfolio_constituents_path(data_root: Path | None = None) -> Path:
    return price_strength_portfolio_simulation_dir(data_root) / "equity_price_strength_portfolio_constituents.parquet"


def price_strength_portfolio_sector_weights_path(data_root: Path | None = None) -> Path:
    return price_strength_portfolio_simulation_dir(data_root) / "equity_price_strength_portfolio_sector_weights.parquet"


def price_strength_portfolio_sector_weights_csv_path(data_root: Path | None = None) -> Path:
    return price_strength_portfolio_simulation_dir(data_root) / "equity_price_strength_portfolio_sector_weights.csv"


def price_strength_portfolio_report_path(data_root: Path | None = None) -> Path:
    return price_strength_portfolio_simulation_dir(data_root) / "equity_price_strength_portfolio_simulation_report.md"


def price_strength_portfolio_metadata_path(data_root: Path | None = None) -> Path:
    return price_strength_portfolio_simulation_dir(data_root) / "equity_price_strength_portfolio_simulation.metadata.json"


def daily_bars_path(data_root: Path | None = None) -> Path:
    root = data_root or get_data_root()
    return root / "market" / "daily_bars" / "historical.parquet"


def price_strength_equity_curve_dir(data_root: Path | None = None) -> Path:
    root = data_root or get_data_root()
    return root / "research" / "price_strength_equity_curve_backtest"


def price_strength_equity_curve_daily_path(data_root: Path | None = None) -> Path:
    return price_strength_equity_curve_dir(data_root) / "equity_price_strength_equity_curve_daily.parquet"


def price_strength_equity_curve_daily_csv_path(data_root: Path | None = None) -> Path:
    return price_strength_equity_curve_dir(data_root) / "equity_price_strength_equity_curve_daily.csv"


def price_strength_equity_curve_vintages_path(data_root: Path | None = None) -> Path:
    return price_strength_equity_curve_dir(data_root) / "equity_price_strength_equity_curve_vintages.parquet"


def price_strength_equity_curve_vintages_csv_path(data_root: Path | None = None) -> Path:
    return price_strength_equity_curve_dir(data_root) / "equity_price_strength_equity_curve_vintages.csv"


def price_strength_equity_curve_constituents_path(data_root: Path | None = None) -> Path:
    return price_strength_equity_curve_dir(data_root) / "equity_price_strength_equity_curve_constituents.parquet"


def price_strength_equity_curve_summary_path(data_root: Path | None = None) -> Path:
    return price_strength_equity_curve_dir(data_root) / "equity_price_strength_equity_curve_summary.parquet"


def price_strength_equity_curve_summary_csv_path(data_root: Path | None = None) -> Path:
    return price_strength_equity_curve_dir(data_root) / "equity_price_strength_equity_curve_summary.csv"


def price_strength_equity_curve_sector_exposure_path(data_root: Path | None = None) -> Path:
    return price_strength_equity_curve_dir(data_root) / "equity_price_strength_equity_curve_sector_exposure.parquet"


def price_strength_equity_curve_sector_exposure_csv_path(data_root: Path | None = None) -> Path:
    return price_strength_equity_curve_dir(data_root) / "equity_price_strength_equity_curve_sector_exposure.csv"


def price_strength_equity_curve_report_path(data_root: Path | None = None) -> Path:
    return price_strength_equity_curve_dir(data_root) / "equity_price_strength_equity_curve_backtest_report.md"


def price_strength_equity_curve_metadata_path(data_root: Path | None = None) -> Path:
    return price_strength_equity_curve_dir(data_root) / "equity_price_strength_equity_curve_backtest.metadata.json"


def price_strength_equity_curve_robustness_dir(data_root: Path | None = None) -> Path:
    root = data_root or get_data_root()
    return root / "research" / "price_strength_equity_curve_robustness"


def price_strength_cost_sensitivity_path(data_root: Path | None = None) -> Path:
    return price_strength_equity_curve_robustness_dir(data_root) / "equity_price_strength_cost_sensitivity.parquet"


def price_strength_cost_sensitivity_csv_path(data_root: Path | None = None) -> Path:
    return price_strength_equity_curve_robustness_dir(data_root) / "equity_price_strength_cost_sensitivity.csv"


def price_strength_sector_cap_sensitivity_path(data_root: Path | None = None) -> Path:
    return price_strength_equity_curve_robustness_dir(data_root) / "equity_price_strength_sector_cap_sensitivity.parquet"


def price_strength_sector_cap_sensitivity_csv_path(data_root: Path | None = None) -> Path:
    return price_strength_equity_curve_robustness_dir(data_root) / "equity_price_strength_sector_cap_sensitivity.csv"


def price_strength_regime_performance_path(data_root: Path | None = None) -> Path:
    return price_strength_equity_curve_robustness_dir(data_root) / "equity_price_strength_regime_performance.parquet"


def price_strength_regime_performance_csv_path(data_root: Path | None = None) -> Path:
    return price_strength_equity_curve_robustness_dir(data_root) / "equity_price_strength_regime_performance.csv"


def price_strength_rolling_performance_path(data_root: Path | None = None) -> Path:
    return price_strength_equity_curve_robustness_dir(data_root) / "equity_price_strength_rolling_performance.parquet"


def price_strength_rolling_performance_csv_path(data_root: Path | None = None) -> Path:
    return price_strength_equity_curve_robustness_dir(data_root) / "equity_price_strength_rolling_performance.csv"


def price_strength_sector_contribution_path(data_root: Path | None = None) -> Path:
    return price_strength_equity_curve_robustness_dir(data_root) / "equity_price_strength_sector_contribution.parquet"


def price_strength_sector_contribution_csv_path(data_root: Path | None = None) -> Path:
    return price_strength_equity_curve_robustness_dir(data_root) / "equity_price_strength_sector_contribution.csv"


def price_strength_symbol_contribution_path(data_root: Path | None = None) -> Path:
    return price_strength_equity_curve_robustness_dir(data_root) / "equity_price_strength_symbol_contribution.parquet"


def price_strength_symbol_contribution_csv_path(data_root: Path | None = None) -> Path:
    return price_strength_equity_curve_robustness_dir(data_root) / "equity_price_strength_symbol_contribution.csv"


def price_strength_robustness_summary_path(data_root: Path | None = None) -> Path:
    return (
        price_strength_equity_curve_robustness_dir(data_root)
        / "equity_price_strength_equity_curve_robustness_summary.parquet"
    )


def price_strength_robustness_summary_csv_path(data_root: Path | None = None) -> Path:
    return (
        price_strength_equity_curve_robustness_dir(data_root)
        / "equity_price_strength_equity_curve_robustness_summary.csv"
    )


def price_strength_robustness_report_path(data_root: Path | None = None) -> Path:
    return (
        price_strength_equity_curve_robustness_dir(data_root)
        / "equity_price_strength_equity_curve_robustness_report.md"
    )


def price_strength_robustness_metadata_path(data_root: Path | None = None) -> Path:
    return (
        price_strength_equity_curve_robustness_dir(data_root)
        / "equity_price_strength_equity_curve_robustness.metadata.json"
    )


def price_strength_horizon_sensitivity_dir(data_root: Path | None = None) -> Path:
    root = data_root or get_data_root()
    return root / "research" / "price_strength_horizon_sensitivity"


def price_strength_horizon_sensitivity_summary_path(data_root: Path | None = None) -> Path:
    return (
        price_strength_horizon_sensitivity_dir(data_root)
        / "equity_price_strength_horizon_sensitivity_summary.parquet"
    )


def price_strength_horizon_sensitivity_summary_csv_path(data_root: Path | None = None) -> Path:
    return (
        price_strength_horizon_sensitivity_dir(data_root)
        / "equity_price_strength_horizon_sensitivity_summary.csv"
    )


def price_strength_horizon_sensitivity_best_variants_path(data_root: Path | None = None) -> Path:
    return (
        price_strength_horizon_sensitivity_dir(data_root)
        / "equity_price_strength_horizon_sensitivity_best_variants.parquet"
    )


def price_strength_horizon_sensitivity_best_variants_csv_path(data_root: Path | None = None) -> Path:
    return (
        price_strength_horizon_sensitivity_dir(data_root)
        / "equity_price_strength_horizon_sensitivity_best_variants.csv"
    )


def price_strength_horizon_sensitivity_daily_path(data_root: Path | None = None) -> Path:
    return (
        price_strength_horizon_sensitivity_dir(data_root)
        / "equity_price_strength_horizon_sensitivity_daily.parquet"
    )


def price_strength_horizon_sensitivity_turnover_path(data_root: Path | None = None) -> Path:
    return (
        price_strength_horizon_sensitivity_dir(data_root)
        / "equity_price_strength_horizon_sensitivity_turnover.parquet"
    )


def price_strength_horizon_sensitivity_turnover_csv_path(data_root: Path | None = None) -> Path:
    return (
        price_strength_horizon_sensitivity_dir(data_root)
        / "equity_price_strength_horizon_sensitivity_turnover.csv"
    )


def price_strength_horizon_sensitivity_report_path(data_root: Path | None = None) -> Path:
    return (
        price_strength_horizon_sensitivity_dir(data_root)
        / "equity_price_strength_horizon_sensitivity_report.md"
    )


def price_strength_horizon_sensitivity_metadata_path(data_root: Path | None = None) -> Path:
    return (
        price_strength_horizon_sensitivity_dir(data_root)
        / "equity_price_strength_horizon_sensitivity.metadata.json"
    )


def price_strength_turnover_cost_efficiency_dir(data_root: Path | None = None) -> Path:
    root = data_root or get_data_root()
    return root / "research" / "price_strength_turnover_cost_efficiency"


def price_strength_turnover_cost_efficiency_path(data_root: Path | None = None) -> Path:
    return (
        price_strength_turnover_cost_efficiency_dir(data_root)
        / "equity_price_strength_turnover_cost_efficiency.parquet"
    )


def price_strength_turnover_cost_efficiency_csv_path(data_root: Path | None = None) -> Path:
    return (
        price_strength_turnover_cost_efficiency_dir(data_root)
        / "equity_price_strength_turnover_cost_efficiency.csv"
    )


def price_strength_turnover_cost_efficiency_focus_path(data_root: Path | None = None) -> Path:
    return (
        price_strength_turnover_cost_efficiency_dir(data_root)
        / "equity_price_strength_turnover_cost_efficiency_focus.parquet"
    )


def price_strength_turnover_cost_efficiency_focus_csv_path(data_root: Path | None = None) -> Path:
    return (
        price_strength_turnover_cost_efficiency_dir(data_root)
        / "equity_price_strength_turnover_cost_efficiency_focus.csv"
    )


def price_strength_turnover_cost_efficiency_best_variants_path(data_root: Path | None = None) -> Path:
    return (
        price_strength_turnover_cost_efficiency_dir(data_root)
        / "equity_price_strength_turnover_cost_efficiency_best_variants.parquet"
    )


def price_strength_turnover_cost_efficiency_best_variants_csv_path(data_root: Path | None = None) -> Path:
    return (
        price_strength_turnover_cost_efficiency_dir(data_root)
        / "equity_price_strength_turnover_cost_efficiency_best_variants.csv"
    )


def price_strength_turnover_cost_efficiency_report_path(data_root: Path | None = None) -> Path:
    return (
        price_strength_turnover_cost_efficiency_dir(data_root)
        / "equity_price_strength_turnover_cost_efficiency_report.md"
    )


def price_strength_turnover_cost_efficiency_metadata_path(data_root: Path | None = None) -> Path:
    return (
        price_strength_turnover_cost_efficiency_dir(data_root)
        / "equity_price_strength_turnover_cost_efficiency.metadata.json"
    )


def price_strength_scorecard_v1_dir(data_root: Path | None = None) -> Path:
    root = data_root or get_data_root()
    return root / "research" / "price_strength_scorecard_v1"


def price_strength_scorecard_v1_path(data_root: Path | None = None) -> Path:
    return price_strength_scorecard_v1_dir(data_root) / "equity_price_strength_scorecard_v1.parquet"


def price_strength_scorecard_v1_csv_path(data_root: Path | None = None) -> Path:
    return price_strength_scorecard_v1_dir(data_root) / "equity_price_strength_scorecard_v1.csv"


def price_strength_scorecard_v1_current_path(data_root: Path | None = None) -> Path:
    return price_strength_scorecard_v1_dir(data_root) / "equity_price_strength_scorecard_v1_current.parquet"


def price_strength_scorecard_v1_current_csv_path(data_root: Path | None = None) -> Path:
    return price_strength_scorecard_v1_dir(data_root) / "equity_price_strength_scorecard_v1_current.csv"


def price_strength_scorecard_v1_summary_path(data_root: Path | None = None) -> Path:
    return price_strength_scorecard_v1_dir(data_root) / "equity_price_strength_scorecard_v1_summary.parquet"


def price_strength_scorecard_v1_summary_csv_path(data_root: Path | None = None) -> Path:
    return price_strength_scorecard_v1_dir(data_root) / "equity_price_strength_scorecard_v1_summary.csv"


def price_strength_scorecard_v1_current_summary_path(data_root: Path | None = None) -> Path:
    return price_strength_scorecard_v1_dir(data_root) / "equity_price_strength_scorecard_v1_current_summary.parquet"


def price_strength_scorecard_v1_current_summary_csv_path(data_root: Path | None = None) -> Path:
    return price_strength_scorecard_v1_dir(data_root) / "equity_price_strength_scorecard_v1_current_summary.csv"


def price_strength_scorecard_v1_evidence_summary_path(data_root: Path | None = None) -> Path:
    return price_strength_scorecard_v1_dir(data_root) / "equity_price_strength_scorecard_v1_evidence_summary.parquet"


def price_strength_scorecard_v1_evidence_summary_csv_path(data_root: Path | None = None) -> Path:
    return price_strength_scorecard_v1_dir(data_root) / "equity_price_strength_scorecard_v1_evidence_summary.csv"


def price_strength_scorecard_v1_report_path(data_root: Path | None = None) -> Path:
    return price_strength_scorecard_v1_dir(data_root) / "equity_price_strength_scorecard_v1_report.md"


def price_strength_scorecard_v1_metadata_path(data_root: Path | None = None) -> Path:
    return price_strength_scorecard_v1_dir(data_root) / "equity_price_strength_scorecard_v1.metadata.json"
