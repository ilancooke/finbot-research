# finbot-research Agent Guidelines

This package owns exploratory research, feature/label diagnostics, signal analysis, early scorecard research, and later lightweight research backtests for Finbot.

It should:

- Read canonical feature and label inputs from `FINBOT_DATA_ROOT`.
- Write research outputs under `data/research`.
- Write parquet plus sidecar metadata JSON for durable research outputs.
- Keep `summarize-diagnostics` outputs under `data/research/feature_label_diagnostics`, including the feature summary, year spreads, completed-year lookback summary, current-year snapshot, Markdown report, and metadata JSON.
- Keep `bucket-signal-diagnostics` outputs under `data/research/bucket_signal_diagnostics`, including bucket summaries, focused candidate-rule diagnostics, candidate-rule year/stability diagnostics, Markdown report, and metadata JSON; this command is for interpretable bucket interaction research only, not score generation, modeling, backtesting, portfolio construction, or dashboard work.
- Keep `price-strength-scorecard-v0` outputs under `data/research/price_strength_scorecard_v0`, including row-level scorecard outputs, time-window summaries, year-by-year bucket diagnostics, completed-year stability summaries, the Markdown report, and metadata JSON; this is a research-only scorecard prototype and must not be promoted to production scoring, model training, formal backtesting, portfolio construction, dashboard integration, or trading recommendations.
- Keep `price-strength-rebalance-feasibility` outputs under `data/research/price_strength_rebalance_feasibility`; this command may evaluate rebalance-date counts, sector concentration, turnover, and feasibility labels, but must not add return simulation, P&L, formal backtesting, portfolio construction, dashboard integration, production signals, or trading logic.
- Keep `price-strength-holding-period-simulation` outputs under `data/research/price_strength_holding_period_simulation`; this command may summarize existing 63-trading-day forward outcomes for monthly scorecard baskets, but must not add production signals, dashboard integration, model training, transaction-cost modeling, portfolio construction, trading logic, or a formal equity-curve backtest.
- Keep `price-strength-portfolio-simulation` outputs under `data/research/price_strength_portfolio_simulation`; this command may run a simple research portfolio simulation with equal weights, sector caps, turnover, transaction-cost assumptions, and portfolio-level forward-label returns, but must not add production signals, dashboard integration, live trading logic, order generation, broker integration, or production portfolio management.
- Keep `price-strength-equity-curve-backtest` outputs under `data/research/price_strength_equity_curve_backtest`; this command may build research-only overlapping equity curves from monthly rebalance vintages, adjusted-close returns, sector caps, turnover, and transaction-cost assumptions, but must not add production signals, dashboard integration, live trading logic, order generation, broker integration, or production portfolio management.
- Keep `price-strength-equity-curve-robustness` outputs under `data/research/price_strength_equity_curve_robustness`; this command may stress-test existing equity-curve outputs across transaction costs, sector caps, market regimes, rolling windows, and approximate contribution diagnostics, but must not define production strategies, dashboard views, scheduled jobs, or live-trading behavior.
- Keep `price-strength-horizon-sensitivity` outputs under `data/research/price_strength_horizon_sensitivity`; this command may rerun overlapping-vintage mechanics across rebalance frequencies, holding periods, costs, and sector-cap variants, but must not generate production signals, dashboard integration, live trading logic, order generation, broker integration, or production portfolio management.
- Keep `price-strength-turnover-cost-efficiency` outputs under `data/research/price_strength_turnover_cost_efficiency`; this command may post-process horizon sensitivity outputs into annualized turnover, estimated cost-drag, and cost-efficiency diagnostics, but must not add new production strategies, dashboard integration, scheduled jobs, or trading logic.
- Keep `price-strength-scorecard-v1` outputs under `data/research/price_strength_scorecard_v1`; this command may stabilize the research scorecard schema, current research snapshot, evidence summary, and report, but must not own daily feature/signal generation, catalog publishing, dashboard integration, scheduled jobs, or production signal generation.
- Treat promoted daily/current scorecard v1 signal snapshots under `data/signals/price_strength` as `finbot-features` ownership, not `finbot-research` ownership.
- Treat the max calendar year as a visible partial current year by default: show it in year spreads/report/snapshot, but exclude it from completed-year stability and lookback calculations unless the CLI flag explicitly includes it.
- Keep diagnostics reproducible, testable, and separate from production feature generation.
- Avoid provider API calls and raw market/reference/fundamental data ingestion.
- Avoid dashboard, production model-training, scoring, portfolio-construction, and live-trading responsibilities.
- Avoid direct imports from sibling repos; communicate through shared data files.
- Use temporary directories and small fake datasets in tests.
- Keep dependencies light. Do not add sklearn, LightGBM, backtesting frameworks, dashboard dependencies, or provider clients unless the user explicitly expands this repo's scope.

Before finishing code changes, run:

```bash
.venv/bin/python -m compileall src tests
.venv/bin/python -m pytest
```
