# finbot-research Agent Guidelines

This package owns exploratory research, feature/label diagnostics, signal analysis, early scorecard research, and later lightweight research backtests for Finbot.

It should:

- Read canonical feature and label inputs from `FINBOT_DATA_ROOT`.
- Write research outputs under `data/research`.
- Write parquet plus sidecar metadata JSON for durable research outputs.
- Keep `summarize-diagnostics` outputs under `data/research/feature_label_diagnostics`, including the feature summary, year spreads, completed-year lookback summary, current-year snapshot, Markdown report, and metadata JSON.
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
