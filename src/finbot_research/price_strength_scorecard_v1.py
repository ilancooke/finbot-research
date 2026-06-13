from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from finbot_research.config import (
    price_strength_equity_curve_summary_path,
    price_strength_robustness_summary_path,
    price_strength_scorecard_path,
    price_strength_scorecard_v1_csv_path,
    price_strength_scorecard_v1_current_csv_path,
    price_strength_scorecard_v1_current_path,
    price_strength_scorecard_v1_current_summary_csv_path,
    price_strength_scorecard_v1_current_summary_path,
    price_strength_scorecard_v1_evidence_summary_csv_path,
    price_strength_scorecard_v1_evidence_summary_path,
    price_strength_scorecard_v1_metadata_path,
    price_strength_scorecard_v1_path,
    price_strength_scorecard_v1_report_path,
    price_strength_scorecard_v1_summary_csv_path,
    price_strength_scorecard_v1_summary_path,
    price_strength_turnover_cost_efficiency_best_variants_path,
    price_strength_turnover_cost_efficiency_focus_path,
    reference_tickers_path,
)
from finbot_research.io import parquet_columns, read_parquet, write_csv, write_parquet
from finbot_research.metadata import write_metadata
from finbot_research.price_strength_rebalance_feasibility import attach_sector
from finbot_research.validation import ValidationError

DATASET_NAME = "equity_price_strength_scorecard_v1"
DEFAULT_RESEARCH_EXPRESSION = "monthly_rebalance_63d_hold_higher_conviction_raw_25bps_reference"
PRACTICAL_DEFAULT_EXPRESSION = (
    "quarterly_rebalance_126d_hold_higher_conviction_sector_capped_30pct_sector_cap_50bps_reference"
)
REQUIRED_V0_COLUMNS = [
    "symbol",
    "date",
    "price_strength_scorecard_bucket",
    "price_strength_score_v0",
    "is_scorecard_bucket_eligible",
]
OPTIONAL_PRESERVED_COLUMNS = [
    "volatility_63d_bucket",
    "momentum_63d_sector_bucket",
    "drawdown_from_52w_high_sector_bucket",
    "drawdown_52w_sector_bucket",
    "return_63d_sector_pct_rank",
    "drawdown_from_52w_high_sector_pct_rank",
    "volatility_63d",
]
V1_REQUIRED_COLUMNS = [
    "symbol",
    "date",
    "sector",
    "price_strength_score_v1",
    "price_strength_bucket_v1",
    "price_strength_confidence_v1",
    "price_strength_risk_label_v1",
    "price_strength_research_label_v1",
    "is_high_conviction_price_strength_v1",
    "is_moderate_price_strength_v1",
    "is_momentum_resilience_v1",
    "is_high_volatility_trap_v1",
    "is_neutral_v1",
    "default_research_expression_v1",
    "practical_default_expression_v1",
    "evidence_summary_label_v1",
    "source_price_strength_scorecard_bucket_v0",
    "source_price_strength_score_v0",
]
BUCKET_MAPPING: dict[str, dict[str, Any]] = {
    "higher_conviction_price_strength": {
        "price_strength_score_v1": 3,
        "price_strength_bucket_v1": "high_conviction_price_strength",
        "price_strength_confidence_v1": "high",
        "price_strength_risk_label_v1": "aggressive_upside_high_drawdown",
        "price_strength_research_label_v1": "primary_price_strength_signal",
        "evidence_summary_label_v1": (
            "historically strongest price-strength bucket; best monthly/63d expression; "
            "practical quarterly/126d sector-capped expression also supported"
        ),
    },
    "price_strength_candidate": {
        "price_strength_score_v1": 2,
        "price_strength_bucket_v1": "moderate_price_strength",
        "price_strength_confidence_v1": "medium_low",
        "price_strength_risk_label_v1": "positive_but_sparse",
        "price_strength_research_label_v1": "secondary_price_strength_signal",
        "evidence_summary_label_v1": "positive but sparse; diluted combined portfolio; not recommended as standalone default",
    },
    "momentum_resilience_candidate": {
        "price_strength_score_v1": 1,
        "price_strength_bucket_v1": "momentum_resilience",
        "price_strength_confidence_v1": "medium",
        "price_strength_risk_label_v1": "defensive_relative_strength",
        "price_strength_research_label_v1": "comparison_defensive_bucket",
        "evidence_summary_label_v1": "more defensive/resilient than return-seeking; useful comparison bucket",
    },
    "high_volatility_trap": {
        "price_strength_score_v1": -1,
        "price_strength_bucket_v1": "high_volatility_trap",
        "price_strength_confidence_v1": "high",
        "price_strength_risk_label_v1": "tail_risk_trap",
        "price_strength_research_label_v1": "risk_exclusion_bucket",
        "evidence_summary_label_v1": "tail-driven; high upside bursts but poor median/downside profile",
    },
    "neutral": {
        "price_strength_score_v1": 0,
        "price_strength_bucket_v1": "neutral",
        "price_strength_confidence_v1": "medium",
        "price_strength_risk_label_v1": "no_price_strength_edge",
        "price_strength_research_label_v1": "baseline_bucket",
        "evidence_summary_label_v1": "no clear price-strength edge",
    },
}
CONFIDENCE_SORT_ORDER = {"high": 3, "medium": 2, "medium_low": 1}
SUPPORTING_EVIDENCE_FILES = {
    "turnover_cost_efficiency_best_variants": price_strength_turnover_cost_efficiency_best_variants_path,
    "turnover_cost_efficiency_focus": price_strength_turnover_cost_efficiency_focus_path,
    "equity_curve_robustness_summary": price_strength_robustness_summary_path,
    "equity_curve_backtest_summary": price_strength_equity_curve_summary_path,
}


def build_price_strength_scorecard_v1(data_root: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    scorecard_path = price_strength_scorecard_path(data_root)
    scorecard = read_parquet(scorecard_path, columns=_scorecard_columns(scorecard_path))
    scorecard, sector_metadata = attach_sector(scorecard, data_root)
    scorecard_v1 = compute_scorecard_v1(scorecard)
    current = compute_current_snapshot(scorecard_v1)
    summary = compute_scorecard_v1_summary(scorecard_v1, current)
    current_summary = compute_scorecard_v1_current_summary(current)
    evidence, evidence_sources = build_evidence_summary(data_root)

    paths = _output_paths(data_root)
    write_parquet(scorecard_v1, paths["scorecard_parquet"])
    write_csv(scorecard_v1, paths["scorecard_csv"])
    write_parquet(current, paths["current_parquet"])
    write_csv(current, paths["current_csv"])
    write_parquet(summary, paths["summary_parquet"])
    write_csv(summary, paths["summary_csv"])
    write_parquet(current_summary, paths["current_summary_parquet"])
    write_csv(current_summary, paths["current_summary_csv"])
    write_parquet(evidence, paths["evidence_summary_parquet"])
    write_csv(evidence, paths["evidence_summary_csv"])
    write_markdown_report(
        paths["markdown_report"],
        summary=summary,
        current_summary=current_summary,
        evidence=evidence,
        evidence_sources=evidence_sources,
    )
    input_paths = [scorecard_path]
    if sector_metadata.get("source") == "reference/tickers.parquet":
        input_paths.append(reference_tickers_path(data_root))
    input_paths.extend(Path(path) for path in evidence_sources["found_paths"])
    metadata_path = write_metadata(
        dataset_name=DATASET_NAME,
        output_path=paths["scorecard_parquet"],
        input_paths=input_paths,
        dataframe=scorecard_v1,
        extra_metadata={
            "dataset_type": "research_scorecard_artifact",
            "research_only": True,
            "scorecard_version": "v1",
            "output_paths": {key: str(value) for key, value in paths.items()},
            "v1_mapping_rules": BUCKET_MAPPING,
            "default_research_expression": DEFAULT_RESEARCH_EXPRESSION,
            "practical_default_expression": PRACTICAL_DEFAULT_EXPRESSION,
            "evidence_source_files_found": evidence_sources["found_paths"],
            "evidence_source_files_missing": evidence_sources["missing_paths"],
            "eligible_row_filter": "is_scorecard_bucket_eligible == true",
            "current_snapshot_definition": "For each symbol, keep the latest available v1 row and sort by score desc, confidence order desc, symbol.",
            "sector_availability": {"available": True, **sector_metadata},
            "limitations": [
                "Research artifact only, not a production signal or trading recommendation.",
                "Encodes conclusions from prior research outputs and does not rerun backtests.",
                "Default and practical expressions are research defaults, not execution instructions.",
                "Main risks remain drawdown, turnover, sector concentration, and regime sensitivity.",
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
        "scorecard_rows": int(len(scorecard_v1)),
        "current_rows": int(len(current)),
        "summary_rows": int(len(summary)),
        "current_summary_rows": int(len(current_summary)),
        "evidence_rows": int(len(evidence)),
        "evidence_files_found": len(evidence_sources["found_paths"]),
        "evidence_files_missing": len(evidence_sources["missing_paths"]),
    }


def _scorecard_columns(path: Path) -> list[str]:
    columns = parquet_columns(path)
    missing = [column for column in REQUIRED_V0_COLUMNS if column not in columns]
    if missing:
        raise ValidationError(f"scorecard v0 missing required columns for v1: {missing}")
    return [column for column in [*REQUIRED_V0_COLUMNS, "sector", *OPTIONAL_PRESERVED_COLUMNS] if column in columns]


def compute_scorecard_v1(scorecard_v0: pd.DataFrame) -> pd.DataFrame:
    frame = scorecard_v0.copy()
    frame["symbol"] = frame["symbol"].astype("string").str.upper()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.date
    frame = frame[frame["is_scorecard_bucket_eligible"].fillna(False)].copy()
    if frame.empty:
        raise ValidationError("No eligible scorecard v0 rows found for scorecard v1")
    if "sector" not in frame.columns or frame["sector"].isna().any():
        missing_count = int(frame["sector"].isna().sum()) if "sector" in frame.columns else len(frame)
        raise ValidationError(f"price-strength-scorecard-v1 requires sector for all eligible rows; missing: {missing_count}")
    frame["source_price_strength_scorecard_bucket_v0"] = frame["price_strength_scorecard_bucket"].astype("string")
    frame["source_price_strength_score_v0"] = frame["price_strength_score_v0"]
    mapped = frame["source_price_strength_scorecard_bucket_v0"].map(lambda bucket: BUCKET_MAPPING.get(str(bucket), BUCKET_MAPPING["neutral"]))
    for column in [
        "price_strength_score_v1",
        "price_strength_bucket_v1",
        "price_strength_confidence_v1",
        "price_strength_risk_label_v1",
        "price_strength_research_label_v1",
        "evidence_summary_label_v1",
    ]:
        frame[column] = mapped.map(lambda values: values[column])
    frame["is_high_conviction_price_strength_v1"] = frame["price_strength_bucket_v1"] == "high_conviction_price_strength"
    frame["is_moderate_price_strength_v1"] = frame["price_strength_bucket_v1"] == "moderate_price_strength"
    frame["is_momentum_resilience_v1"] = frame["price_strength_bucket_v1"] == "momentum_resilience"
    frame["is_high_volatility_trap_v1"] = frame["price_strength_bucket_v1"] == "high_volatility_trap"
    frame["is_neutral_v1"] = frame["price_strength_bucket_v1"] == "neutral"
    frame["default_research_expression_v1"] = DEFAULT_RESEARCH_EXPRESSION
    frame["practical_default_expression_v1"] = PRACTICAL_DEFAULT_EXPRESSION
    optional = [column for column in OPTIONAL_PRESERVED_COLUMNS if column in frame.columns]
    return frame[V1_REQUIRED_COLUMNS + optional].sort_values(["date", "symbol"]).reset_index(drop=True)


def compute_current_snapshot(scorecard_v1: pd.DataFrame) -> pd.DataFrame:
    frame = scorecard_v1.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    latest = frame.sort_values(["symbol", "date"]).groupby("symbol", sort=True).tail(1).copy()
    latest["_confidence_sort"] = latest["price_strength_confidence_v1"].map(CONFIDENCE_SORT_ORDER).fillna(0)
    latest = latest.sort_values(
        ["price_strength_score_v1", "_confidence_sort", "symbol"],
        ascending=[False, False, True],
    ).drop(columns=["_confidence_sort"])
    latest["date"] = latest["date"].dt.date
    return latest.reset_index(drop=True)


def compute_scorecard_v1_summary(scorecard_v1: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "price_strength_bucket_v1",
        "price_strength_score_v1",
        "price_strength_confidence_v1",
        "price_strength_risk_label_v1",
        "price_strength_research_label_v1",
    ]
    frame = scorecard_v1.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    summary = (
        frame.groupby(group_columns, sort=True)
        .agg(
            row_count=("symbol", "size"),
            symbol_count=("symbol", "nunique"),
            min_date=("date", "min"),
            max_date=("date", "max"),
        )
        .reset_index()
    )
    current_counts = current.groupby(group_columns, sort=True)["symbol"].nunique().rename("current_symbol_count").reset_index()
    summary = summary.merge(current_counts, on=group_columns, how="left")
    summary["current_symbol_count"] = summary["current_symbol_count"].fillna(0).astype(int)
    current_universe = max(int(current["symbol"].nunique()), 1)
    summary["pct_of_current_universe"] = summary["current_symbol_count"] / current_universe
    summary["min_date"] = summary["min_date"].dt.date
    summary["max_date"] = summary["max_date"].dt.date
    return summary.sort_values(["price_strength_score_v1", "price_strength_bucket_v1"], ascending=[False, True]).reset_index(drop=True)


def compute_scorecard_v1_current_summary(current: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "price_strength_bucket_v1",
        "price_strength_score_v1",
        "price_strength_confidence_v1",
        "price_strength_risk_label_v1",
        "price_strength_research_label_v1",
    ]
    frame = current.copy()
    current_universe = max(int(frame["symbol"].nunique()), 1)
    summary = (
        frame.groupby(group_columns, sort=True)
        .agg(current_symbol_count=("symbol", "nunique"), current_row_count=("symbol", "size"))
        .reset_index()
    )
    summary["pct_of_current_universe"] = summary["current_symbol_count"] / current_universe
    return summary.sort_values(["price_strength_score_v1", "price_strength_bucket_v1"], ascending=[False, True]).reset_index(drop=True)


def build_evidence_summary(data_root: Path) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    sources = _read_supporting_evidence(data_root)
    rows = [
        _primary_research_expression_row(sources),
        _risk_adjusted_row(sources),
        _practical_default_row(sources),
        _static_evidence_row(
            "trap_bucket_interpretation",
            "high_volatility_trap_is_risk_exclusion",
            "source_bucket",
            "high_volatility_trap",
            "score_v1",
            -1,
            "confidence",
            "high",
            None,
            "High-volatility trap remains a risk/exclusion bucket, not a positive long candidate.",
        ),
        _sector_cap_row(sources),
        _turnover_cost_row(sources),
    ]
    evidence = pd.DataFrame(rows)
    for column in [
        "supporting_metric_1_value",
        "supporting_metric_2_value",
        "supporting_metric_3_value",
    ]:
        evidence[column] = evidence[column].map(_format_evidence_value)
    return evidence, {
        "found_paths": [str(path) for path in sources["found_paths"]],
        "missing_paths": [str(path) for path in sources["missing_paths"]],
    }


def _format_evidence_value(value: Any) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _read_supporting_evidence(data_root: Path) -> dict[str, Any]:
    sources: dict[str, Any] = {"found_paths": [], "missing_paths": []}
    for name, path_fn in SUPPORTING_EVIDENCE_FILES.items():
        path = path_fn(data_root)
        if path.exists():
            sources[name] = read_parquet(path)
            sources["found_paths"].append(path)
        else:
            sources[name] = pd.DataFrame()
            sources["missing_paths"].append(path)
    return sources


def _primary_research_expression_row(sources: dict[str, Any]) -> dict[str, Any]:
    best = sources["turnover_cost_efficiency_best_variants"]
    row = _select_evidence_row(best, selection_criterion="highest_net_excess_annualized_return")
    return _evidence_row(
        "primary_research_expression",
        "monthly_63d_higher_conviction_raw_strongest_excess_return",
        "net_excess_annualized_return",
        row.get("net_excess_annualized_return"),
        "net_annualized_return",
        row.get("net_annualized_return"),
        "transaction_cost_bps",
        row.get("transaction_cost_bps"),
        row.get("_source_output_path"),
        "Monthly / 63d / higher-conviction raw has the strongest historical excess return in the latest grid.",
    )


def _risk_adjusted_row(sources: dict[str, Any]) -> dict[str, Any]:
    best = sources["turnover_cost_efficiency_best_variants"]
    row = _select_evidence_row(best, selection_criterion="highest_net_sharpe_like")
    return _evidence_row(
        "risk_adjusted_comparison",
        "monthly_63d_sector_capped_best_sharpe_like",
        "net_sharpe_like",
        row.get("net_sharpe_like"),
        "net_excess_annualized_return",
        row.get("net_excess_annualized_return"),
        "net_max_drawdown",
        row.get("net_max_drawdown"),
        row.get("_source_output_path"),
        "Monthly / 63d / sector-capped has the strongest Sharpe-like score in the latest ranking.",
    )


def _practical_default_row(sources: dict[str, Any]) -> dict[str, Any]:
    best = sources["turnover_cost_efficiency_best_variants"]
    row = _select_evidence_row(best, selection_criterion="best_practical_default_candidate")
    return _evidence_row(
        "practical_default_expression",
        "quarterly_126d_sector_capped_50bps_practical_default",
        "cost_efficiency_score",
        row.get("cost_efficiency_score"),
        "net_excess_annualized_return",
        row.get("net_excess_annualized_return"),
        "estimated_annualized_cost_drag",
        row.get("estimated_annualized_cost_drag"),
        row.get("_source_output_path"),
        "Quarterly / 126d / sector-capped at 50 bps is the practical default candidate from turnover/cost efficiency.",
    )


def _sector_cap_row(sources: dict[str, Any]) -> dict[str, Any]:
    focus = sources["turnover_cost_efficiency_focus"]
    capped = focus[
        (focus.get("rebalance_frequency") == "monthly")
        & (focus.get("holding_period_days") == 63)
        & (focus.get("transaction_cost_bps") == 25)
        & (focus.get("portfolio_name") == "higher_conviction_sector_capped")
    ] if not focus.empty else pd.DataFrame()
    row = capped.iloc[0].to_dict() if not capped.empty else {}
    return _evidence_row(
        "sector_cap_interpretation",
        "sector_caps_modestly_help_risk_adjusted_comparison",
        "net_sharpe_like",
        row.get("net_sharpe_like"),
        "net_max_drawdown",
        row.get("net_max_drawdown"),
        "annualized_one_way_turnover",
        row.get("annualized_one_way_turnover"),
        str(price_strength_turnover_cost_efficiency_focus_path_placeholder()) if row else None,
        "Sector caps help risk-adjusted comparison modestly but do not transform the signal.",
    )


def _turnover_cost_row(sources: dict[str, Any]) -> dict[str, Any]:
    best = sources["turnover_cost_efficiency_best_variants"]
    row = _select_evidence_row(best, selection_criterion="best_practical_default_candidate")
    return _evidence_row(
        "turnover_cost_interpretation",
        "turnover_drawdown_and_cost_drag_remain_primary_risks",
        "annualized_one_way_turnover",
        row.get("annualized_one_way_turnover"),
        "estimated_annualized_cost_drag",
        row.get("estimated_annualized_cost_drag"),
        "net_max_drawdown",
        row.get("net_max_drawdown"),
        row.get("_source_output_path"),
        "Turnover, drawdowns, sector concentration, and regime sensitivity remain the primary risks.",
    )


def _select_evidence_row(frame: pd.DataFrame, *, selection_criterion: str) -> dict[str, Any]:
    if frame.empty or "selection_criterion" not in frame.columns:
        return {}
    matched = frame[frame["selection_criterion"] == selection_criterion]
    if matched.empty:
        return {}
    row = matched.iloc[0].to_dict()
    row["_source_output_path"] = "turnover_cost_efficiency_best_variants"
    return row


def _static_evidence_row(
    evidence_topic: str,
    evidence_label: str,
    metric_1_name: str,
    metric_1_value: Any,
    metric_2_name: str,
    metric_2_value: Any,
    metric_3_name: str,
    metric_3_value: Any,
    source_output_path: str | None,
    notes: str,
) -> dict[str, Any]:
    return _evidence_row(
        evidence_topic,
        evidence_label,
        metric_1_name,
        metric_1_value,
        metric_2_name,
        metric_2_value,
        metric_3_name,
        metric_3_value,
        source_output_path,
        notes,
    )


def _evidence_row(
    evidence_topic: str,
    evidence_label: str,
    metric_1_name: str,
    metric_1_value: Any,
    metric_2_name: str,
    metric_2_value: Any,
    metric_3_name: str,
    metric_3_value: Any,
    source_output_path: str | None,
    notes: str,
) -> dict[str, Any]:
    return {
        "evidence_topic": evidence_topic,
        "evidence_label": evidence_label,
        "supporting_metric_1_name": metric_1_name,
        "supporting_metric_1_value": metric_1_value,
        "supporting_metric_2_name": metric_2_name,
        "supporting_metric_2_value": metric_2_value,
        "supporting_metric_3_name": metric_3_name,
        "supporting_metric_3_value": metric_3_value,
        "source_output_path": source_output_path or "derived_from_research_conclusions",
        "notes": notes,
    }


def price_strength_turnover_cost_efficiency_focus_path_placeholder() -> str:
    return "turnover_cost_efficiency_focus"


def write_markdown_report(
    path: Path,
    *,
    summary: pd.DataFrame,
    current_summary: pd.DataFrame,
    evidence: pd.DataFrame,
    evidence_sources: dict[str, list[str]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Equity Price Strength Scorecard v1",
        "",
        "## Purpose",
        "",
        "Create a stable, documented research artifact for price-strength scorecard labels.",
        "",
        "## What Changed from v0",
        "",
        "- v1 filters to scorecard-eligible rows only.",
        "- v1 renames exploratory v0 buckets into stable research labels.",
        "- v1 records default and practical research expressions informed by follow-on backtest diagnostics.",
        "",
        "## Scorecard v1 Fields",
        "",
        _markdown_table(pd.DataFrame({"field": V1_REQUIRED_COLUMNS})),
        "",
        "## Bucket Definitions",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {"source_bucket_v0": source, **mapping}
                    for source, mapping in BUCKET_MAPPING.items()
                ]
            )
        ),
        "",
        "## Current Snapshot Summary",
        "",
        _markdown_table(current_summary),
        "",
        "## Research Evidence Summary",
        "",
        _markdown_table(evidence),
        "",
        "## Default Research Expression",
        "",
        DEFAULT_RESEARCH_EXPRESSION,
        "",
        "## Practical Default Expression",
        "",
        PRACTICAL_DEFAULT_EXPRESSION,
        "",
        "## Risk Labels and Caveats",
        "",
        "- This is a research artifact, not a production signal or trading recommendation.",
        "- `high_conviction_price_strength` is the primary positive signal.",
        "- `moderate_price_strength` is secondary and not preferred as a standalone default.",
        "- `momentum_resilience` is defensive/comparison, not a return engine.",
        "- `high_volatility_trap` is a risk/exclusion bucket.",
        "- Main risks are drawdown, turnover, sector concentration, and regime sensitivity.",
        "",
        "## Output File Guide",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {"File": "equity_price_strength_scorecard_v1.parquet/csv", "Purpose": "Row-level v1 scorecard."},
                    {"File": "equity_price_strength_scorecard_v1_current.parquet/csv", "Purpose": "Latest row per symbol."},
                    {"File": "equity_price_strength_scorecard_v1_summary.parquet/csv", "Purpose": "Full-history v1 bucket summary."},
                    {"File": "equity_price_strength_scorecard_v1_current_summary.parquet/csv", "Purpose": "Current snapshot bucket summary."},
                    {"File": "equity_price_strength_scorecard_v1_evidence_summary.parquet/csv", "Purpose": "Supporting evidence summary."},
                    {"File": "equity_price_strength_scorecard_v1.metadata.json", "Purpose": "Inputs, mapping rules, assumptions, and outputs."},
                ]
            )
        ),
        "",
        "## Suggested Next Step",
        "",
        "Use scorecard_v1 as the stable research input for scorecard_v1 diagnostics; keep production promotion out of finbot-research until explicitly scoped.",
        "",
        f"Evidence files found: {len(evidence_sources['found_paths'])}; missing: {len(evidence_sources['missing_paths'])}.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _output_paths(data_root: Path) -> dict[str, Path]:
    return {
        "scorecard_parquet": price_strength_scorecard_v1_path(data_root),
        "scorecard_csv": price_strength_scorecard_v1_csv_path(data_root),
        "current_parquet": price_strength_scorecard_v1_current_path(data_root),
        "current_csv": price_strength_scorecard_v1_current_csv_path(data_root),
        "summary_parquet": price_strength_scorecard_v1_summary_path(data_root),
        "summary_csv": price_strength_scorecard_v1_summary_csv_path(data_root),
        "current_summary_parquet": price_strength_scorecard_v1_current_summary_path(data_root),
        "current_summary_csv": price_strength_scorecard_v1_current_summary_csv_path(data_root),
        "evidence_summary_parquet": price_strength_scorecard_v1_evidence_summary_path(data_root),
        "evidence_summary_csv": price_strength_scorecard_v1_evidence_summary_csv_path(data_root),
        "markdown_report": price_strength_scorecard_v1_report_path(data_root),
        "metadata": price_strength_scorecard_v1_metadata_path(data_root),
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
            "Price strength scorecard v1 complete.",
            "",
            f"Scorecard rows: {summary['scorecard_rows']}",
            f"Current rows: {summary['current_rows']}",
            f"Summary rows: {summary['summary_rows']}",
            f"Current summary rows: {summary['current_summary_rows']}",
            f"Evidence rows: {summary['evidence_rows']}",
            f"Evidence files found: {summary['evidence_files_found']}",
            f"Evidence files missing: {summary['evidence_files_missing']}",
            "",
            f"- Scorecard parquet: {paths['scorecard_parquet']}",
            f"- Current snapshot parquet: {paths['current_parquet']}",
            f"- Summary parquet: {paths['summary_parquet']}",
            f"- Current summary parquet: {paths['current_summary_parquet']}",
            f"- Evidence summary parquet: {paths['evidence_summary_parquet']}",
            f"- Markdown report: {paths['markdown_report']}",
            f"- Metadata JSON: {paths['metadata']}",
        ]
    )
