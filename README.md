# finbot-research

Exploratory research diagnostics for Finbot datasets.

This repository evaluates whether existing Finbot features and labels show useful signal. It reads durable datasets from `FINBOT_DATA_ROOT` and writes research outputs under `FINBOT_DATA_ROOT/research`. It does not download provider data, generate production features, train models, run dashboards, build portfolios, or perform live trading.

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
