# finbot-research

Exploratory research diagnostics for Finbot datasets.

This repository evaluates whether existing Finbot features and labels show useful signal. It reads durable datasets from `FINBOT_DATA_ROOT` and writes research outputs under `FINBOT_DATA_ROOT/research`. It does not download provider data, generate production features, train models, run dashboards, manage production portfolios, or perform live trading.

## Scope

- Analyze feature and label relationships.
- Produce diagnostics and research reports under `data/research`.
- Prototype research ideas that may later move into `finbot-features`, `finbot-models`, or a future strategy/backtesting package.
- Keep exploratory research separate from durable feature generation.

## Inputs

- `features/equity_price_features.parquet`
- `features/equity_relative_features.parquet`
- `labels/equity_forward_return_labels.parquet`

The primary regression label is `forward_63d_sector_relative_return`. The primary classification label is `forward_63d_top_30pct_sector_flag`.

## Outputs

- `research/feature_label_diagnostics/equity_price_signal_diagnostics.parquet`
- `research/feature_label_diagnostics/equity_price_signal_diagnostics.metadata.json`
- `research/feature_label_diagnostics/equity_price_signal_summary.parquet`
- `research/feature_label_diagnostics/equity_price_signal_summary.csv`
- `research/feature_label_diagnostics/equity_price_signal_report.md`
- `research/feature_label_diagnostics/equity_price_signal_summary.metadata.json`
- `research/feature_label_diagnostics/equity_price_signal_year_spreads.parquet`
- `research/feature_label_diagnostics/equity_price_signal_year_spreads.csv`
- `research/feature_label_diagnostics/equity_price_signal_lookback_summary.parquet`
- `research/feature_label_diagnostics/equity_price_signal_lookback_summary.csv`
- `research/feature_label_diagnostics/equity_price_signal_current_year_snapshot.parquet`
- `research/feature_label_diagnostics/equity_price_signal_current_year_snapshot.csv`
- `research/bucket_signal_diagnostics/equity_bucket_signal_summary.parquet`
- `research/bucket_signal_diagnostics/equity_bucket_signal_summary.csv`
- `research/bucket_signal_diagnostics/equity_bucket_candidate_rules.parquet`
- `research/bucket_signal_diagnostics/equity_bucket_candidate_rules.csv`
- `research/bucket_signal_diagnostics/equity_bucket_candidate_rule_years.parquet`
- `research/bucket_signal_diagnostics/equity_bucket_candidate_rule_years.csv`
- `research/bucket_signal_diagnostics/equity_bucket_candidate_rule_stability.parquet`
- `research/bucket_signal_diagnostics/equity_bucket_candidate_rule_stability.csv`
- `research/bucket_signal_diagnostics/equity_bucket_signal_report.md`
- `research/bucket_signal_diagnostics/equity_bucket_signal_summary.metadata.json`
- `research/price_strength_scorecard_v0/equity_price_strength_scorecard_v0.parquet`
- `research/price_strength_scorecard_v0/equity_price_strength_scorecard_v0.csv`
- `research/price_strength_scorecard_v0/equity_price_strength_scorecard_v0_summary.parquet`
- `research/price_strength_scorecard_v0/equity_price_strength_scorecard_v0_summary.csv`
- `research/price_strength_scorecard_v0/equity_price_strength_scorecard_v0_years.parquet`
- `research/price_strength_scorecard_v0/equity_price_strength_scorecard_v0_years.csv`
- `research/price_strength_scorecard_v0/equity_price_strength_scorecard_v0_stability.parquet`
- `research/price_strength_scorecard_v0/equity_price_strength_scorecard_v0_stability.csv`
- `research/price_strength_scorecard_v0/equity_price_strength_scorecard_v0_report.md`
- `research/price_strength_scorecard_v0/equity_price_strength_scorecard_v0.metadata.json`
- `research/price_strength_rebalance_feasibility/equity_price_strength_rebalance_bucket_counts.parquet`
- `research/price_strength_rebalance_feasibility/equity_price_strength_rebalance_bucket_counts.csv`
- `research/price_strength_rebalance_feasibility/equity_price_strength_rebalance_bucket_count_summary.parquet`
- `research/price_strength_rebalance_feasibility/equity_price_strength_rebalance_bucket_count_summary.csv`
- `research/price_strength_rebalance_feasibility/equity_price_strength_rebalance_turnover.parquet`
- `research/price_strength_rebalance_feasibility/equity_price_strength_rebalance_turnover.csv`
- `research/price_strength_rebalance_feasibility/equity_price_strength_rebalance_feasibility.parquet`
- `research/price_strength_rebalance_feasibility/equity_price_strength_rebalance_feasibility.csv`
- `research/price_strength_rebalance_feasibility/equity_price_strength_rebalance_feasibility_report.md`
- `research/price_strength_rebalance_feasibility/equity_price_strength_rebalance_feasibility.metadata.json`
- `research/price_strength_rebalance_feasibility/equity_price_strength_rebalance_sector_composition.parquet`, when sector is available
- `research/price_strength_rebalance_feasibility/equity_price_strength_rebalance_sector_composition.csv`, when sector is available
- `research/price_strength_rebalance_feasibility/equity_price_strength_rebalance_sector_concentration.parquet`, when sector is available
- `research/price_strength_rebalance_feasibility/equity_price_strength_rebalance_sector_concentration.csv`, when sector is available
- `research/price_strength_holding_period_simulation/equity_price_strength_holding_period_rebalance_results.parquet`
- `research/price_strength_holding_period_simulation/equity_price_strength_holding_period_rebalance_results.csv`
- `research/price_strength_holding_period_simulation/equity_price_strength_holding_period_summary.parquet`
- `research/price_strength_holding_period_simulation/equity_price_strength_holding_period_summary.csv`
- `research/price_strength_holding_period_simulation/equity_price_strength_holding_period_turnover.parquet`
- `research/price_strength_holding_period_simulation/equity_price_strength_holding_period_turnover.csv`
- `research/price_strength_holding_period_simulation/equity_price_strength_holding_period_simulation_report.md`
- `research/price_strength_holding_period_simulation/equity_price_strength_holding_period_simulation.metadata.json`
- `research/price_strength_holding_period_simulation/equity_price_strength_holding_period_sector_composition.parquet`, when sector is available
- `research/price_strength_holding_period_simulation/equity_price_strength_holding_period_sector_composition.csv`, when sector is available
- `research/price_strength_holding_period_simulation/equity_price_strength_holding_period_sector_concentration.parquet`, when sector is available
- `research/price_strength_holding_period_simulation/equity_price_strength_holding_period_sector_concentration.csv`, when sector is available
- `research/price_strength_portfolio_simulation/equity_price_strength_portfolio_rebalance_results.parquet`
- `research/price_strength_portfolio_simulation/equity_price_strength_portfolio_rebalance_results.csv`
- `research/price_strength_portfolio_simulation/equity_price_strength_portfolio_summary.parquet`
- `research/price_strength_portfolio_simulation/equity_price_strength_portfolio_summary.csv`
- `research/price_strength_portfolio_simulation/equity_price_strength_portfolio_constituents.parquet`
- `research/price_strength_portfolio_simulation/equity_price_strength_portfolio_sector_weights.parquet`
- `research/price_strength_portfolio_simulation/equity_price_strength_portfolio_sector_weights.csv`
- `research/price_strength_portfolio_simulation/equity_price_strength_portfolio_simulation_report.md`
- `research/price_strength_portfolio_simulation/equity_price_strength_portfolio_simulation.metadata.json`

The diagnostics parquet contains one table with multiple `metric_type` values:

- `coverage`: non-null/null counts, label coverage, Pearson correlation, and Spearman correlation per feature.
- `decile`: cross-sectional decile diagnostics per feature.
- `spread`: top-minus-bottom decile spreads per feature.
- `year_decile`: decile diagnostics by calendar year.

Deciles are computed cross-sectionally by date, so decile 10 means the highest-ranked 10% for that feature on that date. Dates with fewer than two non-null feature values are left without deciles for that feature.

The diagnostics summary converts the machine-readable diagnostics into feature-level candidate categories:

- `bullish_candidate`: higher feature values show positive spread, positive top-30% hit-rate lift, positive Spearman correlation, and positive decile monotonicity.
- `risk_penalty_candidate`: higher feature values show negative spread, negative top-30% hit-rate lift, negative Spearman correlation, and negative decile monotonicity.
- `nonlinear_or_unstable`: top-minus-bottom spread or lift looks meaningful, but correlation or decile monotonicity disagrees.
- `weak_or_noisy`: return spread and classification lift are mixed or weak.
- `coverage_problem`: feature or label coverage is too sparse for reliable interpretation.

The summary also adds mean/median tail-effect diagnostics from decile 1 and decile 10. Tail-effect assessments include `broad_positive`, `broad_negative`, `right_tail_positive`, `left_tail_negative`, `mean_median_disagreement`, `limited_tail_evidence`, and `insufficient_data`. Broad average and median agreement is stronger evidence than a mean-only result.

Suggested scorecard-use labels consider recent-window evidence, tail effects, and liquidity handling. Examples include `positive_component_candidate`, `risk_penalty_candidate`, `upside_optional_component_requires_review`, `liquidity_filter_candidate`, `nonlinear_feature_requires_review`, `nonlinear_or_tail_feature_requires_review`, `ignore_for_v1`, and `coverage_issue`. Liquidity features are explicitly treated as possible tradability filters rather than automatic alpha components.

The summary ranking formula is `abs(top_minus_bottom_avg_forward_return) + abs(top_minus_bottom_top_30pct_flag_rate) + 0.25 * abs(avg_return_decile_spearman_corr) + 0.25 * abs(flag_rate_decile_spearman_corr) + 0.10 * max(pct_years_positive_spread, pct_years_negative_spread) - null_rate`. It is a first-pass research aid, not a final model score.

The Markdown report includes a short executive summary, key findings, compact candidate and recent/regime tables, selected decile curves, selected year-by-year details, an output file guide, and caveats. It is meant to guide future scorecard research; diagnostics do not replace backtesting.

Because the available history starts in the late 1990s, the summary also separates full-history behavior from recent-window behavior. It writes annual top-minus-bottom spreads and lookback summaries for `full_history`, `last_20y`, `last_15y`, `last_10y`, and `last_5y`. By default, the max calendar year is treated as a partial current year: it remains visible in year spreads, the Markdown report, and the current-year snapshot, but it is excluded from completed-year stability and lookback calculations. Use `--include-partial-current-year-in-stability` only when you deliberately want the max year included in those calculations. The main summary includes recent signal assessments such as `persistent_positive`, `persistent_negative`, `recent_positive_only`, `recent_negative_only`, `historical_positive_but_recent_weak`, `historical_negative_but_recent_weak`, `mixed_or_regime_dependent`, and `insufficient_year_data`.

## `summarize-diagnostics` outputs

`finbot-research summarize-diagnostics` creates a human-readable Markdown report plus machine-readable research summaries.

Parquet files are the canonical machine-readable outputs. The Markdown report is the canonical human-readable output. CSV files are convenience exports for manual inspection and may later become optional via `--write-csv`.

| File | Purpose |
| --- | --- |
| `equity_price_signal_report.md` | Human-readable executive research summary for manual review. |
| `equity_price_signal_summary.parquet` | Canonical feature-level summary for downstream research code. |
| `equity_price_signal_summary.csv` | Convenience export for manual inspection in spreadsheet tools. |
| `equity_price_signal_summary.metadata.json` | Documents inputs, outputs, rules, thresholds, and generation metadata. |
| `equity_price_signal_year_spreads.parquet` | Canonical year-by-year feature spread details. |
| `equity_price_signal_year_spreads.csv` | Convenience export for manual inspection of year-by-year spread details. |
| `equity_price_signal_lookback_summary.parquet` | Canonical lookback-window summary by feature and window. |
| `equity_price_signal_lookback_summary.csv` | Convenience export for manual inspection of lookback-window summaries. |
| `equity_price_signal_current_year_snapshot.parquet` | Canonical current partial-year/current-regime feature spread snapshot. |
| `equity_price_signal_current_year_snapshot.csv` | Convenience export for manual inspection of the current partial-year snapshot. |

The Markdown report should stay concise and decision-oriented. Selected decile curves include average, median, and average-minus-median returns so possible tail effects are visible without opening the parquet files. Full detail belongs in the parquet outputs, with CSVs available only as convenience copies.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Set the shared data root:

```bash
export FINBOT_DATA_ROOT=/Users/ilan/workspace/finbot/data
```

You can also place `FINBOT_DATA_ROOT` in a local `.env` file in this repo.

## Usage

Run feature/label diagnostics:

```bash
finbot-research feature-label-diagnostics
```

Summarize the diagnostics into ranked machine-readable outputs and a Markdown report:

```bash
finbot-research summarize-diagnostics
```

Include the max calendar year in completed-year stability and lookback calculations:

```bash
finbot-research summarize-diagnostics --include-partial-current-year-in-stability
```

The Markdown report is intended to guide future price-based scorecard research. PDF output and chart generation are intentionally deferred until the report format stabilizes.

Show CLI help:

```bash
finbot-research --help
```

Rebuild `finbot-catalog` after creating or materially changing research outputs so downstream tools can discover them.

## bucket-signal-diagnostics

`finbot-research bucket-signal-diagnostics` analyzes simple two-way bucket interactions across volatility, momentum, drawdown, and dollar-volume/size-liquidity features. This is research for deciding whether any interpretable interactions deserve later scorecard work; it does not generate production scores, train models, run backtests, build portfolios, or make trading recommendations.

Inputs:

- `features/equity_price_features.parquet`
- `features/equity_relative_features.parquet`
- `labels/equity_forward_return_labels.parquet`

Outputs under `research/bucket_signal_diagnostics/`:

- `equity_bucket_signal_summary.parquet`
- `equity_bucket_signal_summary.csv`
- `equity_bucket_candidate_rules.parquet`
- `equity_bucket_candidate_rules.csv`
- `equity_bucket_candidate_rule_years.parquet`
- `equity_bucket_candidate_rule_years.csv`
- `equity_bucket_candidate_rule_stability.parquet`
- `equity_bucket_candidate_rule_stability.csv`
- `equity_bucket_signal_report.md`
- `equity_bucket_signal_summary.metadata.json`

Bucket dimensions:

- `volatility_63d_bucket`: date-level percentile buckets from `volatility_63d`: `low_volatility`, `medium_volatility`, `high_volatility`.
- `momentum_63d_sector_bucket`: `return_63d_sector_pct_rank` buckets: `weak_momentum`, `middle_momentum`, `strong_momentum`.
- `drawdown_52w_sector_bucket`: `drawdown_from_52w_high_sector_pct_rank` buckets: `sector_relative_deep_drawdown`, `sector_relative_mid_drawdown`, `sector_relative_near_52w_high`.
- `dollar_volume_63d_bucket`: date-level percentile buckets from `average_dollar_volume_63d`: `lower_dollar_volume`, `middle_dollar_volume`, `highest_dollar_volume`.

The first implementation intentionally analyzes only these combinations:

- `volatility_63d_bucket x momentum_63d_sector_bucket`
- `volatility_63d_bucket x drawdown_52w_sector_bucket`
- `momentum_63d_sector_bucket x drawdown_52w_sector_bucket`
- `dollar_volume_63d_bucket x momentum_63d_sector_bucket`
- `dollar_volume_63d_bucket x volatility_63d_bucket`

Because this universe is mid-cap and larger US equities, dollar-volume buckets are interpreted as a size/liquidity/crowding proxy, not as a simple liquid-versus-illiquid split. Drawdown is computed as `adjusted_close / rolling_52w_high - 1`; values closer to `0` indicate stocks closer to their 52-week highs. Sector percentile ranks are ascending, so higher percentile rank means stronger price resilience relative to sector peers.

The command uses the same partial-year policy as `summarize-diagnostics`: the max calendar year is analyzed separately as `current_partial_year`, while completed-history windows exclude it by default.

The report also writes focused candidate-rule diagnostics for a small set of research hypotheses, including high volatility plus sector-relative price resilience, high volatility plus strong momentum, their 3-way combination, and review/avoid candidates such as high volatility plus weak momentum or deep drawdown. These are research rules only, not production scorecard rules or trading rules.

Candidate-rule diagnostics include baseline comparisons. The baseline is the full eligible joined feature/label universe for the same time window or calendar year. Candidate-rule outputs include average return, median return, top-30% flag rate, and bottom-30% flag rate minus that baseline.

Candidate-rule stability is computed over completed years only; the current partial year remains visible in year-by-year diagnostics but is excluded from completed-year stability. Stability assessments are:

- `broadly_consistent`: median and top-rate lift are positive in at least 60% of completed years, with positive mean median and top-rate lift.
- `average_only_tail_driven`: average lift is positive in at least 60% of completed years, but median or top-rate support is weaker.
- `mixed_or_regime_dependent`: completed-year evidence is mixed across average, median, and top-rate lift.
- `weak_or_negative`: median and top-rate support are weak or negative across completed years.
- `insufficient_data`: fewer than three completed years with candidate-rule observations.

## price-strength-scorecard-v0

`finbot-research price-strength-scorecard-v0` builds a daily research-only prototype table that classifies each symbol/date into interpretable price-strength and risk buckets. It is not a production signal, model, formal backtest, dashboard integration, portfolio construction tool, or trading recommendation.

Inputs:

- `features/equity_price_features.parquet`
- `features/equity_relative_features.parquet`
- `labels/equity_forward_return_labels.parquet`

Outputs under `research/price_strength_scorecard_v0/`:

- `equity_price_strength_scorecard_v0.parquet`
- `equity_price_strength_scorecard_v0.csv`
- `equity_price_strength_scorecard_v0_summary.parquet`
- `equity_price_strength_scorecard_v0_summary.csv`
- `equity_price_strength_scorecard_v0_years.parquet`
- `equity_price_strength_scorecard_v0_years.csv`
- `equity_price_strength_scorecard_v0_stability.parquet`
- `equity_price_strength_scorecard_v0_stability.csv`
- `equity_price_strength_scorecard_v0_report.md`
- `equity_price_strength_scorecard_v0.metadata.json`

Scorecard logic:

- `higher_conviction_price_strength`: high volatility, sector-relative near 52-week high, and strong momentum.
- `price_strength_candidate`: high volatility and sector-relative near 52-week high.
- `momentum_resilience_candidate`: sector-relative near 52-week high and strong momentum.
- `high_volatility_trap`: high volatility plus weak momentum or high volatility plus sector-relative deep drawdown.
- `neutral`: all remaining rows.

The integer `price_strength_score_v0` is a research ranking label only: `3`, `2`, `1`, `0`, or `-1` for the buckets above. Summary outputs compare each scorecard bucket to the full eligible universe baseline for the same time window.

The row-level scorecard retains all joined rows and includes `is_scorecard_bucket_eligible`. Rows with missing required bucket inputs remain visible in the row-level output, but summary, yearly, and stability diagnostics exclude them so sparse-start history does not inflate the `neutral` bucket.

Year-by-year scorecard diagnostics compare each scorecard bucket to the same-year full eligible universe baseline. The max calendar year is shown as a partial year in yearly diagnostics and excluded from completed-year stability calculations.

Completed-year stability uses the yearly rule-minus-baseline fields for average return, median return, top-30% hit rate, and bottom-30% hit rate. Stability assessments are:

- `broadly_positive`: average, median, and top-rate support are consistently above baseline without meaningfully worse bottom-rate behavior.
- `positive_but_high_risk`: average/top-rate support is consistently positive, but bottom-rate behavior is also worse.
- `tail_driven`: average/top-rate support is positive, but median support is weak or negative and bottom-rate behavior is worse.
- `neutral_or_defensive`: upside support is limited, but bottom-rate behavior is better than baseline.
- `negative_or_trap`: median return underperforms and bottom-rate behavior is worse.
- `mixed_or_regime_dependent`: completed-year evidence is inconsistent.
- `insufficient_data`: fewer than three completed years are available for that bucket.

This differs from a production signal because it is a transparent diagnostic table with simple handcrafted research flags. It does not train a model, tune a portfolio, simulate execution, size positions, or make buy/sell decisions.

## price-strength-rebalance-feasibility

`finbot-research price-strength-rebalance-feasibility` evaluates whether scorecard v0 buckets are operationally usable on realistic rebalance dates before any holding-period simulation.

Input:

- `research/price_strength_scorecard_v0/equity_price_strength_scorecard_v0.parquet`

Options:

- `--rebalance-frequency monthly`: monthly only for now.
- `--start-date YYYY-MM-DD`: optional inclusive date filter.
- `--end-date YYYY-MM-DD`: optional inclusive date filter.

Monthly rebalance dates are the last available trading date in each calendar month. The command uses only rows where `is_scorecard_bucket_eligible=true`.

Outputs under `research/price_strength_rebalance_feasibility/`:

- `equity_price_strength_rebalance_bucket_counts.parquet`
- `equity_price_strength_rebalance_bucket_counts.csv`
- `equity_price_strength_rebalance_bucket_count_summary.parquet`
- `equity_price_strength_rebalance_bucket_count_summary.csv`
- `equity_price_strength_rebalance_turnover.parquet`
- `equity_price_strength_rebalance_turnover.csv`
- `equity_price_strength_rebalance_feasibility.parquet`
- `equity_price_strength_rebalance_feasibility.csv`
- `equity_price_strength_rebalance_feasibility_report.md`
- `equity_price_strength_rebalance_feasibility.metadata.json`
- `equity_price_strength_rebalance_sector_composition.parquet/csv`, when sector is available
- `equity_price_strength_rebalance_sector_concentration.parquet/csv`, when sector is available

If `sector` is not present in the scorecard, the command tries to join it from `reference/tickers.parquet` using `ticker -> symbol`. If sector cannot be found, sector diagnostics are skipped and the report/metadata record that limitation.

This diagnostic answers count, diversification, sector concentration, and membership-change questions. It does not evaluate returns, P&L, execution, portfolio construction, or trading rules. If feasibility looks acceptable, the next research step is a simple monthly rebalance / 63-trading-day holding-period simulation.

## price-strength-holding-period-simulation

`finbot-research price-strength-holding-period-simulation` evaluates monthly rebalance scorecard baskets using existing 63-trading-day forward outcome labels. It is a holding-period outcome simulation, not a full portfolio backtest.

Input:

- `research/price_strength_scorecard_v0/equity_price_strength_scorecard_v0.parquet`

Options:

- `--rebalance-frequency monthly`: monthly only for now.
- `--start-date YYYY-MM-DD`: optional inclusive date filter.
- `--end-date YYYY-MM-DD`: optional inclusive date filter.

Monthly rebalance dates are the last available trading date in each calendar month. The command uses only rows where `is_scorecard_bucket_eligible=true` and the 63-trading-day forward labels are non-null.

Basket definitions:

- `higher_conviction_price_strength`: only the matching scorecard bucket.
- `positive_combined`: `higher_conviction_price_strength` plus `price_strength_candidate`.
- `momentum_resilience_candidate`: only the matching scorecard bucket.
- `high_volatility_trap`: risk-bucket comparison only, not a long recommendation.
- `eligible_universe_baseline`: all eligible rows on the same rebalance date.

Outputs under `research/price_strength_holding_period_simulation/`:

- `equity_price_strength_holding_period_rebalance_results.parquet`
- `equity_price_strength_holding_period_rebalance_results.csv`
- `equity_price_strength_holding_period_summary.parquet`
- `equity_price_strength_holding_period_summary.csv`
- `equity_price_strength_holding_period_turnover.parquet`
- `equity_price_strength_holding_period_turnover.csv`
- `equity_price_strength_holding_period_simulation_report.md`
- `equity_price_strength_holding_period_simulation.metadata.json`
- `equity_price_strength_holding_period_sector_composition.parquet/csv`, when sector is available
- `equity_price_strength_holding_period_sector_concentration.parquet/csv`, when sector is available

If `sector` is not present in the scorecard, the command tries to join it from `reference/tickers.parquet` using `ticker -> symbol`. If sector cannot be found, sector exposure outputs are skipped and the report/metadata record that limitation.

This differs from a full backtest because it does not model transaction costs, slippage, position sizing, overlapping portfolios, or realistic equity curves. It summarizes forward outcomes for scorecard baskets selected on monthly rebalance dates.

## price-strength-portfolio-simulation

`finbot-research price-strength-portfolio-simulation` runs a research-only portfolio simulation for `price_strength_scorecard_v0` using monthly rebalance dates, equal weighting, optional sector caps, turnover tracking, and a simple transaction-cost assumption.

Input:

- `research/price_strength_scorecard_v0/equity_price_strength_scorecard_v0.parquet`

Options:

- `--rebalance-frequency monthly`: monthly only for now.
- `--start-date YYYY-MM-DD`: optional inclusive date filter.
- `--end-date YYYY-MM-DD`: optional inclusive date filter.
- `--sector-cap 0.30`: sector cap for capped portfolios.
- `--transaction-cost-bps 25`: one-way transaction cost assumption in basis points.

Portfolio definitions:

- `higher_conviction_raw`: equal-weight `higher_conviction_price_strength`.
- `higher_conviction_sector_capped`: same names with sector cap applied.
- `positive_combined_raw`: equal-weight `higher_conviction_price_strength` plus `price_strength_candidate`.
- `positive_combined_sector_capped`: same names with sector cap applied.
- `eligible_universe_baseline`: equal-weight all eligible names.

The sector-cap method caps sector-level weights, redistributes remaining weight across uncapped sectors proportionally, and equal-weights names within each sector. If too few sectors exist to satisfy the cap while summing to 100%, sector weights are equalized across available sectors and the cap may remain binding.

Outputs under `research/price_strength_portfolio_simulation/`:

- `equity_price_strength_portfolio_rebalance_results.parquet`
- `equity_price_strength_portfolio_rebalance_results.csv`
- `equity_price_strength_portfolio_summary.parquet`
- `equity_price_strength_portfolio_summary.csv`
- `equity_price_strength_portfolio_constituents.parquet`
- `equity_price_strength_portfolio_sector_weights.parquet`
- `equity_price_strength_portfolio_sector_weights.csv`
- `equity_price_strength_portfolio_simulation_report.md`
- `equity_price_strength_portfolio_simulation.metadata.json`

This remains a research simulation. It uses 63-trading-day forward labels, does not build an overlapping equity curve, and does not include live trading logic, order generation, broker integration, or production signals.
Summary outputs include simple forward-return volatility and max-drawdown diagnostics computed over the sequence of rebalance-level forward outcomes; these are research diagnostics, not an investable equity curve.

## Tests

```bash
.venv/bin/python -m compileall src tests
.venv/bin/python -m pytest
```

If `.venv` is unavailable, use the active Python environment:

```bash
python -m compileall src tests
pytest
```
