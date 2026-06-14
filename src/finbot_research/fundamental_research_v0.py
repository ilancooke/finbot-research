from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from finbot_research.config import (
    fundamental_features_path,
    fundamental_research_v0_dir,
    labels_path,
    reference_tickers_path,
)
from finbot_research.io import parquet_columns, read_parquet, write_parquet

FEATURE_COLUMNS = [
    "gross_margin_ttm",
    "ebitda_margin_ttm",
    "net_margin_ttm",
    "fcf_margin_ttm",
    "roa_ttm",
    "roe_ttm",
    "roic_ttm",
    "asset_turnover_ttm",
    "revenue_yoy_ttm",
    "ebitda_yoy_ttm",
    "netinc_yoy_ttm",
    "fcf_yoy_ttm",
    "revenue_qoq",
    "revenue_yoy_quarter",
    "gross_margin_ttm_change_yoy",
    "net_margin_ttm_change_yoy",
    "fcf_to_netinc_ttm",
    "ncfo_to_netinc_ttm",
    "capex_to_revenue_ttm",
    "accruals_to_assets",
    "debt_to_assets",
    "debt_to_equity",
    "cash_to_assets",
    "net_debt_to_ebitda",
    "equity_to_assets",
    "liabilities_to_assets",
    "workingcapital_to_assets",
    "eps_ttm",
    "bvps",
    "fcfps_ttm",
    "shareswa_yoy_change",
    "fundamentals_age_days",
]

HIGHER_IS_WORSE_FEATURES = {
    "accruals_to_assets",
    "capex_to_revenue_ttm",
    "debt_to_assets",
    "debt_to_equity",
    "liabilities_to_assets",
    "net_debt_to_ebitda",
    "shareswa_yoy_change",
    "fundamentals_age_days",
}

HORIZONS = (63, 126, 252)
RETURN_LABELS = {horizon: f"forward_{horizon}d_sector_relative_return" for horizon in HORIZONS}
TOP_LABELS = {horizon: f"forward_{horizon}d_top_30pct_sector_flag" for horizon in HORIZONS}
BOTTOM_LABELS = {horizon: f"forward_{horizon}d_bottom_30pct_sector_flag" for horizon in HORIZONS}

BUCKET_COLUMNS = [
    "quality_bucket",
    "growth_bucket",
    "valuation_proxy_bucket",
    "cashflow_quality_bucket",
    "balance_sheet_bucket",
    "staleness_bucket",
]

SCORECARD_RELABELING_GUIDANCE = {
    "speculative_growth": "speculative_upside_high_downside",
    "fundamental_deterioration": "deterioration_rebound_risk",
    "high_quality_growth": "defensive_quality_growth",
    "cashflow_supported_growth": "defensive_cashflow_quality_or_weak_return_quality",
    "quality_cashflow_compounder": "defensive_quality_cashflow",
    "fundamental_trap": "trap_or_rebound_mixed",
    "levered_growth_risk": "levered_growth_mixed_risk",
    "insufficient_data": "data_quality_flag_only",
    "neutral": "neutral",
}

SCORECARD_V0_SCORE_MAP = {
    "high_quality_growth": 3,
    "quality_cashflow_compounder": 2,
    "cashflow_supported_growth": 1,
    "speculative_growth": 0,
    "neutral": 0,
    "levered_growth_risk": -1,
    "fundamental_deterioration": -2,
    "fundamental_trap": -3,
    "insufficient_data": np.nan,
}

SCORECARD_V0_1_DIMENSION_POLICY = {
    "defensive_quality_growth": {
        "fundamental_quality_score_v0_1": 2,
        "fundamental_opportunity_score_v0_1": 0,
        "fundamental_risk_label_v0_1": "lower_downside",
        "fundamental_data_quality_flag_v0_1": False,
        "score_policy_notes": "Defensive quality dimension; not a return-seeking composite score.",
    },
    "defensive_cashflow_quality_or_weak_return_quality": {
        "fundamental_quality_score_v0_1": 2,
        "fundamental_opportunity_score_v0_1": 0,
        "fundamental_risk_label_v0_1": "lower_downside",
        "fundamental_data_quality_flag_v0_1": False,
        "score_policy_notes": "Cash-flow quality dimension; return evidence remains weak or defensive.",
    },
    "defensive_quality_cashflow": {
        "fundamental_quality_score_v0_1": 2,
        "fundamental_opportunity_score_v0_1": 0,
        "fundamental_risk_label_v0_1": "lower_downside",
        "fundamental_data_quality_flag_v0_1": False,
        "score_policy_notes": "Quality/cash-flow dimension; do not treat as a standalone alpha score.",
    },
    "speculative_upside_high_downside": {
        "fundamental_quality_score_v0_1": -1,
        "fundamental_opportunity_score_v0_1": 2,
        "fundamental_risk_label_v0_1": "high_downside_tail_risk",
        "fundamental_data_quality_flag_v0_1": False,
        "score_policy_notes": "Opportunity dimension is positive, but quality is weak and downside tail risk is high.",
    },
    "deterioration_rebound_risk": {
        "fundamental_quality_score_v0_1": -2,
        "fundamental_opportunity_score_v0_1": 1,
        "fundamental_risk_label_v0_1": "high_dispersion_rebound",
        "fundamental_data_quality_flag_v0_1": False,
        "score_policy_notes": "Rebound/opportunity dimension is separate from poor quality.",
    },
    "trap_or_rebound_mixed": {
        "fundamental_quality_score_v0_1": -2,
        "fundamental_opportunity_score_v0_1": 0,
        "fundamental_risk_label_v0_1": "mixed_trap_rebound",
        "fundamental_data_quality_flag_v0_1": False,
        "score_policy_notes": "Mixed trap/rebound risk; do not force into a clean negative alpha score.",
    },
    "levered_growth_mixed_risk": {
        "fundamental_quality_score_v0_1": -1,
        "fundamental_opportunity_score_v0_1": 0,
        "fundamental_risk_label_v0_1": "leverage_risk_mixed",
        "fundamental_data_quality_flag_v0_1": False,
        "score_policy_notes": "Leverage risk dimension is separate from opportunity.",
    },
    "insufficient_data": {
        "fundamental_quality_score_v0_1": np.nan,
        "fundamental_opportunity_score_v0_1": np.nan,
        "fundamental_risk_label_v0_1": "insufficient_data",
        "fundamental_data_quality_flag_v0_1": True,
        "score_policy_notes": "Data-quality flag only; not an alpha score.",
    },
    "neutral": {
        "fundamental_quality_score_v0_1": 0,
        "fundamental_opportunity_score_v0_1": 0,
        "fundamental_risk_label_v0_1": "neutral",
        "fundamental_data_quality_flag_v0_1": False,
        "score_policy_notes": "Baseline/comparison bucket.",
    },
}

OUTPUT_FILENAMES = [
    "01_fundamental_feature_coverage.parquet",
    "02_fundamental_feature_label_diagnostics.parquet",
    "02b_fundamental_feature_direction_audit.parquet",
    "fundamental_feature_diagnostics_report.md",
    "03_fundamental_bucket_summary.parquet",
    "04_fundamental_bucket_years.parquet",
    "05_fundamental_bucket_stability.parquet",
    "fundamental_bucket_diagnostics_report.md",
    "06_fundamental_candidate_rules.parquet",
    "07_fundamental_candidate_rule_years.parquet",
    "08_fundamental_candidate_rule_stability.parquet",
    "fundamental_candidate_rules_report.md",
    "09_fundamental_scorecard_v0.parquet",
    "10_fundamental_scorecard_v0_current.parquet",
    "11_fundamental_scorecard_v0_summary.parquet",
    "12_fundamental_scorecard_v0_stability.parquet",
    "12b_fundamental_scorecard_behavior_summary.parquet",
    "12c_fundamental_scorecard_relabeling_recommendation.parquet",
    "fundamental_scorecard_v0_report.md",
    "13_fundamental_rebalance_feasibility.parquet",
    "14_fundamental_holding_period_summary.parquet",
    "15_fundamental_holding_period_turnover.parquet",
    "fundamental_feasibility_and_holding_period_report.md",
    "16_fundamental_robustness_summary.parquet",
    "17_fundamental_scorecard_v1_recommendation.parquet",
    "18_fundamental_scorecard_v0_1_current.parquet",
    "fundamental_scorecard_v1_recommendation_report.md",
]

PARQUET_OUTPUT_FILENAMES = [name for name in OUTPUT_FILENAMES if name.endswith(".parquet")]
STALE_OUTPUT_FILENAMES = ["18_fundamental_scorecard_v1_current.parquet"]
REPORT_FILE_GUIDES = {
    "feature": [
        ("01_fundamental_feature_coverage.parquet", "Feature coverage, yearly availability, and staleness diagnostics."),
        ("02_fundamental_feature_label_diagnostics.parquet", "Feature-level forward outcome spreads and correlations."),
        ("02b_fundamental_feature_direction_audit.parquet", "Assumed direction versus empirical strong-minus-weak behavior by horizon."),
    ],
    "bucket": [
        ("03_fundamental_bucket_summary.parquet", "Bucket-level forward outcome comparisons versus the eligible universe baseline, including behavior labels."),
        ("04_fundamental_bucket_years.parquet", "Year-by-year bucket performance diagnostics."),
        ("05_fundamental_bucket_stability.parquet", "Completed-year stability summary by bucket and horizon."),
    ],
    "candidate": [
        ("06_fundamental_candidate_rules.parquet", "Candidate-rule forward outcome comparisons versus the eligible universe baseline."),
        ("07_fundamental_candidate_rule_years.parquet", "Year-by-year candidate-rule diagnostics."),
        ("08_fundamental_candidate_rule_stability.parquet", "Completed-year stability summary by candidate rule and horizon."),
    ],
    "scorecard": [
        ("09_fundamental_scorecard_v0.parquet", "Filing-event scorecard v0 rows and original labels."),
        ("10_fundamental_scorecard_v0_current.parquet", "Latest scorecard v0 row per symbol."),
        ("11_fundamental_scorecard_v0_summary.parquet", "Scorecard label forward outcome comparisons versus the eligible universe baseline."),
        ("12_fundamental_scorecard_v0_stability.parquet", "Completed-year scorecard label stability summary."),
        ("12b_fundamental_scorecard_behavior_summary.parquet", "Scorecard behavior summary with downside-aware labels and role guidance."),
        ("12c_fundamental_scorecard_relabeling_recommendation.parquet", "Recommended v0.1 relabeling with separate quality, opportunity, risk, and data-quality dimensions."),
    ],
    "feasibility": [
        ("13_fundamental_rebalance_feasibility.parquet", "Monthly and quarterly scorecard label counts and concentration diagnostics."),
        ("14_fundamental_holding_period_summary.parquet", "Rebalance-date holding-period outcome comparisons versus same-date baseline."),
        ("15_fundamental_holding_period_turnover.parquet", "Scorecard label membership turnover by rebalance date."),
    ],
    "v1": [
        ("16_fundamental_robustness_summary.parquet", "Robustness summary combining scorecard behavior and stability evidence."),
        ("17_fundamental_scorecard_v1_recommendation.parquet", "Conservative v1 readiness decision and required next stage."),
        ("18_fundamental_scorecard_v0_1_current.parquet", "Latest per-symbol v0.1 current snapshot with separate quality, opportunity, risk, and data-quality dimensions."),
        ("12c_fundamental_scorecard_relabeling_recommendation.parquet", "Relabeling recommendation that must be reviewed before any v1 freeze."),
    ],
}
ProgressCallback = Callable[[str], None]


def build_fundamental_research_v0(
    data_root: Path,
    *,
    progress: ProgressCallback | None = None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    _progress(progress, "Preparing output directory")
    output_dir = fundamental_research_v0_dir(data_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_outputs(output_dir)

    _progress(progress, "Loading fundamental features, labels, and reference data")
    fundamentals = _load_fundamentals(data_root)
    labels = _load_labels(data_root)
    reference = _load_reference(data_root)

    _progress(progress, "Attaching reference sectors and assigning fundamental buckets")
    fundamentals = _attach_reference(fundamentals, reference)
    bucketed = assign_fundamental_buckets(fundamentals)
    _progress(progress, "Building filing-effective event panel")
    panel = _build_event_panel(bucketed, labels)

    _progress(progress, "Computing feature coverage and feature-label diagnostics")
    coverage = compute_feature_coverage(bucketed, panel)
    diagnostics = compute_feature_label_diagnostics(panel)
    direction_audit = compute_feature_direction_audit(diagnostics)
    _progress(progress, "Computing bucket diagnostics")
    bucket_summary = summarize_bucket_outcomes(panel)
    bucket_years = summarize_bucket_years(panel)
    bucket_stability = summarize_stability(bucket_years, "bucket_name", "bucket_value")

    _progress(progress, "Computing candidate rule diagnostics")
    rule_panel = assign_candidate_rules(panel)
    candidate_rules = summarize_rule_outcomes(rule_panel)
    candidate_years = summarize_rule_years(rule_panel)
    candidate_stability = summarize_stability(candidate_years, "rule_name", None)

    _progress(progress, "Computing scorecard v0 outputs")
    scorecard = compute_scorecard_v0(bucketed)
    scorecard_current = current_snapshot(scorecard, "effective_date")
    scorecard_panel = _build_event_panel(scorecard, labels)
    scorecard_summary = summarize_scorecard_outcomes(scorecard_panel)
    scorecard_stability = summarize_stability(summarize_scorecard_years(scorecard_panel), "scorecard_label_v0", None)
    scorecard_behavior = compute_scorecard_behavior_summary(scorecard_summary, scorecard_stability)
    relabeling_recommendation = compute_scorecard_relabeling_recommendation(scorecard_behavior)

    _progress(progress, "Computing feasibility, holding-period, and v1-readiness outputs")
    feasibility, holding_summary, turnover = compute_rebalance_and_holding_outputs(scorecard_panel)
    robustness, v1_recommendation, v0_1_current = compute_v1_recommendation(
        scorecard_behavior,
        scorecard_stability,
        scorecard_current,
        relabeling_recommendation,
    )

    outputs: dict[str, pd.DataFrame] = {
        "01_fundamental_feature_coverage.parquet": coverage,
        "02_fundamental_feature_label_diagnostics.parquet": diagnostics,
        "02b_fundamental_feature_direction_audit.parquet": direction_audit,
        "03_fundamental_bucket_summary.parquet": bucket_summary,
        "04_fundamental_bucket_years.parquet": bucket_years,
        "05_fundamental_bucket_stability.parquet": bucket_stability,
        "06_fundamental_candidate_rules.parquet": candidate_rules,
        "07_fundamental_candidate_rule_years.parquet": candidate_years,
        "08_fundamental_candidate_rule_stability.parquet": candidate_stability,
        "09_fundamental_scorecard_v0.parquet": scorecard,
        "10_fundamental_scorecard_v0_current.parquet": scorecard_current,
        "11_fundamental_scorecard_v0_summary.parquet": scorecard_summary,
        "12_fundamental_scorecard_v0_stability.parquet": scorecard_stability,
        "12b_fundamental_scorecard_behavior_summary.parquet": scorecard_behavior,
        "12c_fundamental_scorecard_relabeling_recommendation.parquet": relabeling_recommendation,
        "13_fundamental_rebalance_feasibility.parquet": feasibility,
        "14_fundamental_holding_period_summary.parquet": holding_summary,
        "15_fundamental_holding_period_turnover.parquet": turnover,
        "16_fundamental_robustness_summary.parquet": robustness,
        "17_fundamental_scorecard_v1_recommendation.parquet": v1_recommendation,
        "18_fundamental_scorecard_v0_1_current.parquet": v0_1_current,
    }

    _progress(progress, "Writing numbered parquet outputs")
    paths: dict[str, Path] = {}
    for filename in PARQUET_OUTPUT_FILENAMES:
        paths[filename] = write_parquet(outputs[filename], output_dir / filename)

    _progress(progress, "Writing Markdown research reports")
    reports = {
        "fundamental_feature_diagnostics_report.md": render_feature_report(coverage, diagnostics, direction_audit),
        "fundamental_bucket_diagnostics_report.md": render_bucket_report(bucket_summary, bucket_stability),
        "fundamental_candidate_rules_report.md": render_candidate_report(candidate_rules, candidate_stability),
        "fundamental_scorecard_v0_report.md": render_scorecard_report(
            scorecard_behavior,
            scorecard_stability,
            relabeling_recommendation,
        ),
        "fundamental_feasibility_and_holding_period_report.md": render_feasibility_report(feasibility, holding_summary),
        "fundamental_scorecard_v1_recommendation_report.md": render_v1_report(
            v1_recommendation,
            robustness,
            relabeling_recommendation,
        ),
    }
    for filename, contents in reports.items():
        path = output_dir / filename
        path.write_text(contents, encoding="utf-8")
        paths[filename] = path

    _assert_output_convention(output_dir, paths)
    label_match_columns = [column for column in RETURN_LABELS.values() if column in panel.columns]
    summary = {
        "output_dir": str(output_dir),
        "fundamental_rows": int(len(fundamentals)),
        "daily_label_rows": int(len(labels)),
        "event_panel_rows": int(len(panel)),
        "event_panel_label_match_rows": int(panel[label_match_columns].notna().any(axis=1).sum()) if label_match_columns else 0,
        "research_panel_grain": "symbol,effective_date",
        "panel_rows": int(len(panel)),
        "symbols": int(panel["symbol"].nunique()) if "symbol" in panel else 0,
        "date_min": _date_str(panel["date"].min()) if not panel.empty else None,
        "date_max": _date_str(panel["date"].max()) if not panel.empty else None,
        "output_count": len(paths),
    }
    return paths, summary


def _load_fundamentals(data_root: Path) -> pd.DataFrame:
    path = fundamental_features_path(data_root)
    columns = parquet_columns(path)
    required = ["symbol", "effective_date", *[col for col in FEATURE_COLUMNS if col in columns]]
    data = read_parquet(path, columns=required)
    missing = {"symbol", "effective_date"} - set(data.columns)
    if missing:
        raise ValueError(f"Fundamental features missing required columns: {sorted(missing)}")
    data = data.copy()
    data["symbol"] = data["symbol"].astype("string").str.upper()
    data["effective_date"] = pd.to_datetime(data["effective_date"])
    return data.sort_values(["symbol", "effective_date"]).reset_index(drop=True)


def _load_labels(data_root: Path) -> pd.DataFrame:
    path = labels_path(data_root)
    available = set(parquet_columns(path))
    required = ["symbol", "date"]
    label_columns = [
        column
        for column in [*RETURN_LABELS.values(), *TOP_LABELS.values(), *BOTTOM_LABELS.values(), "sector"]
        if column in available
    ]
    data = read_parquet(path, columns=[*required, *label_columns])
    data = data.copy()
    data["symbol"] = data["symbol"].astype("string").str.upper()
    data["date"] = pd.to_datetime(data["date"])
    return data.sort_values(["symbol", "date"]).reset_index(drop=True)


def _build_event_panel(fundamentals: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    left = fundamentals.copy()
    right = labels.copy()
    left["symbol"] = left["symbol"].astype("string").str.upper()
    right["symbol"] = right["symbol"].astype("string").str.upper()
    left["effective_date"] = pd.to_datetime(left["effective_date"])
    right["date"] = pd.to_datetime(right["date"])

    if "sector" in left.columns and "sector" in right.columns:
        right = right.drop(columns=["sector"])
    panel = left.merge(right, left_on=["symbol", "effective_date"], right_on=["symbol", "date"], how="left")
    panel["date"] = panel["date"].fillna(panel["effective_date"])
    panel["fundamental_effective_date"] = panel["effective_date"]
    panel["fundamental_data_age_days"] = 0
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)


def _load_reference(data_root: Path) -> pd.DataFrame:
    path = reference_tickers_path(data_root)
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "sector", "industry"])
    available = set(parquet_columns(path))
    symbol_column = "symbol" if "symbol" in available else "ticker" if "ticker" in available else None
    if symbol_column is None:
        return pd.DataFrame(columns=["symbol", "sector", "industry"])
    columns = [symbol_column, *[column for column in ("sector", "industry") if column in available]]
    reference = read_parquet(path, columns=columns).rename(columns={symbol_column: "symbol"})
    reference["symbol"] = reference["symbol"].astype("string").str.upper()
    return reference.drop_duplicates("symbol")


def _attach_reference(fundamentals: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    data = fundamentals.copy()
    if "sector" not in data.columns:
        data["sector"] = pd.NA
    if not reference.empty:
        reference_columns = [column for column in ["symbol", "sector", "industry"] if column in reference.columns]
        data = data.drop(columns=[column for column in ["sector", "industry"] if column in data.columns], errors="ignore")
        data = data.merge(reference[reference_columns], on="symbol", how="left")
    data["sector"] = data.get("sector", pd.Series(index=data.index, dtype="string")).fillna("Unknown")
    return data


def _asof_merge(daily_panel: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    left = daily_panel.copy()
    right = fundamentals.copy()
    left["symbol"] = left["symbol"].astype("string").str.upper()
    right["symbol"] = right["symbol"].astype("string").str.upper()
    left["date"] = pd.to_datetime(left["date"])
    right["effective_date"] = pd.to_datetime(right["effective_date"])

    pieces = []
    for symbol, symbol_daily in left.sort_values(["symbol", "date"]).groupby("symbol", sort=False):
        symbol_fundamentals = right[right["symbol"] == symbol].sort_values("effective_date")
        if symbol_fundamentals.empty:
            missing = symbol_daily.copy()
            for column in right.columns:
                if column not in missing.columns and column != "symbol":
                    missing[column] = pd.NA
            pieces.append(missing)
            continue
        merged = pd.merge_asof(
            symbol_daily.sort_values("date"),
            symbol_fundamentals.drop(columns=["symbol"]).sort_values("effective_date"),
            left_on="date",
            right_on="effective_date",
            direction="backward",
        )
        merged["symbol"] = symbol
        if "sector_x" in merged.columns or "sector_y" in merged.columns:
            merged["sector"] = merged.get("sector_x", pd.Series(index=merged.index, dtype="object")).combine_first(
                merged.get("sector_y", pd.Series(index=merged.index, dtype="object"))
            )
            merged = merged.drop(columns=["sector_x", "sector_y"], errors="ignore")
        pieces.append(merged)
    if not pieces:
        return left
    panel = pd.concat(pieces, ignore_index=True)
    if "effective_date" in panel.columns:
        panel["fundamental_effective_date"] = pd.to_datetime(panel["effective_date"])
        panel["fundamental_data_age_days"] = (panel["date"] - panel["fundamental_effective_date"]).dt.days
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)


def assign_fundamental_buckets(fundamentals: pd.DataFrame) -> pd.DataFrame:
    data = fundamentals.copy()
    group_columns = ["effective_date", "sector"] if "sector" in data.columns else ["effective_date"]
    rank_columns = [column for column in FEATURE_COLUMNS if column in data.columns]
    for column in rank_columns:
        data[f"{column}_sector_pct_rank"] = data.groupby(group_columns, dropna=False)[column].rank(pct=True)

    quality_metrics = [
        "gross_margin_ttm_sector_pct_rank",
        "ebitda_margin_ttm_sector_pct_rank",
        "net_margin_ttm_sector_pct_rank",
        "fcf_margin_ttm_sector_pct_rank",
        "roa_ttm_sector_pct_rank",
        "roe_ttm_sector_pct_rank",
        "roic_ttm_sector_pct_rank",
    ]
    growth_metrics = [
        "revenue_yoy_ttm_sector_pct_rank",
        "ebitda_yoy_ttm_sector_pct_rank",
        "netinc_yoy_ttm_sector_pct_rank",
        "fcf_yoy_ttm_sector_pct_rank",
        "revenue_yoy_quarter_sector_pct_rank",
    ]
    cashflow_metrics = [
        "fcf_to_netinc_ttm_sector_pct_rank",
        "ncfo_to_netinc_ttm_sector_pct_rank",
    ]
    balance_good_metrics = [
        "cash_to_assets_sector_pct_rank",
        "equity_to_assets_sector_pct_rank",
        "workingcapital_to_assets_sector_pct_rank",
    ]
    balance_bad_metrics = [
        "debt_to_assets_sector_pct_rank",
        "debt_to_equity_sector_pct_rank",
        "net_debt_to_ebitda_sector_pct_rank",
        "liabilities_to_assets_sector_pct_rank",
    ]

    data["quality_score"] = _mean_with_min_count(data, quality_metrics, 3)
    data["growth_score"] = _mean_with_min_count(data, growth_metrics, 2)
    data["cashflow_quality_score"] = _mean_with_min_count(data, cashflow_metrics, 1)
    data["balance_sheet_score"] = _mean_with_min_count(data, balance_good_metrics, 1) - _mean_with_min_count(
        data, balance_bad_metrics, 1
    ).fillna(0)

    data["quality_bucket"] = np.select(
        [data["quality_score"].isna(), data["quality_score"] >= 0.70, data["quality_score"] <= 0.30],
        ["insufficient_quality_data", "high_quality", "low_quality"],
        default="average_quality",
    )
    data["growth_bucket"] = np.select(
        [
            data["growth_score"].isna(),
            data["growth_score"] >= 0.75,
            data["growth_score"] >= 0.55,
            data["growth_score"] <= 0.30,
        ],
        ["insufficient_growth_data", "high_growth", "positive_growth", "contracting"],
        default="flat_or_mixed_growth",
    )
    positive_per_share_count = data[[col for col in ["eps_ttm", "fcfps_ttm", "bvps"] if col in data.columns]].gt(0).sum(axis=1)
    per_share_non_null = data[[col for col in ["eps_ttm", "fcfps_ttm", "bvps"] if col in data.columns]].notna().sum(axis=1)
    data["valuation_proxy_bucket"] = np.select(
        [per_share_non_null < 2, positive_per_share_count >= 2],
        ["insufficient_per_share_data", "positive_eps_fcf_book"],
        default="weak_or_negative_eps_fcf",
    )
    accrual_rank = data.get("accruals_to_assets_sector_pct_rank", pd.Series(np.nan, index=data.index))
    data["cashflow_quality_bucket"] = np.select(
        [
            data["cashflow_quality_score"].isna() & accrual_rank.isna(),
            accrual_rank >= 0.80,
            data["cashflow_quality_score"] >= 0.65,
            data["cashflow_quality_score"] <= 0.35,
        ],
        [
            "insufficient_cashflow_quality_data",
            "accruals_risk",
            "high_cashflow_quality",
            "low_cashflow_quality",
        ],
        default="low_cashflow_quality",
    )
    debt_rank = data.get("debt_to_assets_sector_pct_rank", pd.Series(np.nan, index=data.index))
    equity_rank = data.get("equity_to_assets_sector_pct_rank", pd.Series(np.nan, index=data.index))
    data["balance_sheet_bucket"] = np.select(
        [
            data["balance_sheet_score"].isna() & debt_rank.isna(),
            (debt_rank >= 0.85) & (equity_rank <= 0.25),
            data["balance_sheet_score"] >= 0.45,
            debt_rank >= 0.70,
        ],
        [
            "insufficient_balance_sheet_data",
            "distressed_or_high_risk",
            "strong_balance_sheet",
            "levered_balance_sheet",
        ],
        default="normal_balance_sheet",
    )
    age = data.get("fundamentals_age_days", pd.Series(np.nan, index=data.index))
    data["staleness_bucket"] = np.select(
        [age.isna(), age <= 180, age <= 365],
        ["very_stale_fundamentals", "fresh_fundamentals", "stale_fundamentals"],
        default="very_stale_fundamentals",
    )
    return data


def compute_feature_coverage(fundamentals: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in [column for column in FEATURE_COLUMNS if column in fundamentals.columns]:
        series = fundamentals[feature]
        rows.append(
            {
                "scope": "overall",
                "feature": feature,
                "calendar_year": pd.NA,
                "row_count": len(fundamentals),
                "symbol_count": fundamentals.loc[series.notna(), "symbol"].nunique(),
                "non_null_count": int(series.notna().sum()),
                "non_null_rate": float(series.notna().mean()) if len(series) else np.nan,
                "date_min": _date_str(fundamentals["effective_date"].min()),
                "date_max": _date_str(fundamentals["effective_date"].max()),
                "warning": _coverage_warning(series.notna().mean() if len(series) else 0, feature),
            }
        )
        yearly = fundamentals.assign(calendar_year=fundamentals["effective_date"].dt.year).groupby("calendar_year")
        for year, group in yearly:
            year_series = group[feature]
            rows.append(
                {
                    "scope": "year",
                    "feature": feature,
                    "calendar_year": int(year),
                    "row_count": len(group),
                    "symbol_count": group.loc[year_series.notna(), "symbol"].nunique(),
                    "non_null_count": int(year_series.notna().sum()),
                    "non_null_rate": float(year_series.notna().mean()) if len(year_series) else np.nan,
                    "date_min": _date_str(group["effective_date"].min()),
                    "date_max": _date_str(group["effective_date"].max()),
                    "warning": _coverage_warning(year_series.notna().mean() if len(year_series) else 0, feature),
                }
            )
    age_column = "fundamental_data_age_days" if "fundamental_data_age_days" in panel.columns else "fundamentals_age_days"
    if age_column in panel.columns:
        age = pd.to_numeric(panel[age_column], errors="coerce")
        rows.append(
            {
                "scope": "staleness",
                "feature": age_column,
                "calendar_year": pd.NA,
                "row_count": len(panel),
                "symbol_count": panel.loc[age.notna(), "symbol"].nunique(),
                "non_null_count": int(age.notna().sum()),
                "non_null_rate": float(age.notna().mean()) if len(age) else np.nan,
                "date_min": _date_str(panel["date"].min()) if "date" in panel else None,
                "date_max": _date_str(panel["date"].max()) if "date" in panel else None,
                "age_p50_days": float(age.quantile(0.50)) if age.notna().any() else np.nan,
                "age_p90_days": float(age.quantile(0.90)) if age.notna().any() else np.nan,
                "warning": "stale" if age.dropna().gt(365).mean() > 0.20 else "",
            }
        )
    return pd.DataFrame(rows)


def compute_feature_label_diagnostics(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in [column for column in FEATURE_COLUMNS if column in panel.columns]:
        for horizon, return_label in RETURN_LABELS.items():
            if return_label not in panel.columns:
                continue
            selected_columns = ["symbol", "date", feature, return_label]
            if "sector" in panel.columns:
                selected_columns.append("sector")
            if TOP_LABELS[horizon] in panel.columns:
                selected_columns.append(TOP_LABELS[horizon])
            if BOTTOM_LABELS[horizon] in panel.columns:
                selected_columns.append(BOTTOM_LABELS[horizon])
            subset = panel[selected_columns]
            subset = subset.rename(columns={return_label: "forward_return"}).dropna(subset=[feature, "forward_return"])
            if subset.empty:
                rows.append(_empty_outcome_row("feature", feature, horizon))
                continue
            buckets = _tertile_labels(subset[feature], higher_is_worse=feature in HIGHER_IS_WORSE_FEATURES)
            subset = subset.assign(research_bucket=buckets)
            high = subset[subset["research_bucket"] == "strong"]
            weak = subset[subset["research_bucket"] == "weak"]
            rows.append(
                {
                    "feature": feature,
                    "horizon_days": horizon,
                    "direction": "higher_is_worse" if feature in HIGHER_IS_WORSE_FEATURES else "higher_is_better_or_contextual",
                    "row_count": len(subset),
                    "symbol_count": subset["symbol"].nunique(),
                    "pearson_corr": _safe_corr(subset[feature], subset["forward_return"], "pearson"),
                    "spearman_corr": _safe_corr(subset[feature], subset["forward_return"], "spearman"),
                    "strong_bucket_avg_return": _mean(high["forward_return"]),
                    "weak_bucket_avg_return": _mean(weak["forward_return"]),
                    "strong_minus_weak_avg_return": _mean(high["forward_return"]) - _mean(weak["forward_return"]),
                    "strong_bucket_median_return": _median(high["forward_return"]),
                    "weak_bucket_median_return": _median(weak["forward_return"]),
                    "strong_minus_weak_median_return": _median(high["forward_return"]) - _median(weak["forward_return"]),
                    "strong_top_30_rate": _mean(subset.loc[high.index, TOP_LABELS[horizon]]) if TOP_LABELS[horizon] in subset else np.nan,
                    "weak_top_30_rate": _mean(subset.loc[weak.index, TOP_LABELS[horizon]]) if TOP_LABELS[horizon] in subset else np.nan,
                    "strong_minus_weak_top_30pct_rate": (
                        _mean(subset.loc[high.index, TOP_LABELS[horizon]]) - _mean(subset.loc[weak.index, TOP_LABELS[horizon]])
                        if TOP_LABELS[horizon] in subset
                        else np.nan
                    ),
                    "strong_bottom_30_rate": _mean(subset.loc[high.index, BOTTOM_LABELS[horizon]]) if BOTTOM_LABELS[horizon] in subset else np.nan,
                    "weak_bottom_30_rate": _mean(subset.loc[weak.index, BOTTOM_LABELS[horizon]]) if BOTTOM_LABELS[horizon] in subset else np.nan,
                    "strong_minus_weak_bottom_30pct_rate": (
                        _mean(subset.loc[high.index, BOTTOM_LABELS[horizon]]) - _mean(subset.loc[weak.index, BOTTOM_LABELS[horizon]])
                        if BOTTOM_LABELS[horizon] in subset
                        else np.nan
                    ),
                    "sector_count": subset["sector"].nunique() if "sector" in subset else 0,
                }
            )
    return pd.DataFrame(rows)


def compute_feature_direction_audit(diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in diagnostics.to_dict(orient="records"):
        feature = row.get("feature")
        assumed_direction = _assumed_feature_direction(str(feature))
        avg_spread = row.get("strong_minus_weak_avg_return", np.nan)
        median_spread = row.get("strong_minus_weak_median_return", np.nan)
        top_spread = row.get("strong_minus_weak_top_30pct_rate", np.nan)
        bottom_spread = row.get("strong_minus_weak_bottom_30pct_rate", np.nan)
        empirical = _empirical_direction_label(avg_spread, median_spread, top_spread, bottom_spread)
        rows.append(
            {
                "feature": feature,
                "feature_family": _feature_family(str(feature)),
                "assumed_direction": assumed_direction,
                "horizon_days": row.get("horizon_days"),
                "strong_minus_weak_avg_return": avg_spread,
                "strong_minus_weak_median_return": median_spread,
                "strong_minus_weak_top_30pct_rate": top_spread,
                "strong_minus_weak_bottom_30pct_rate": bottom_spread,
                "empirical_direction_label": empirical,
                "direction_consistency_status": _direction_consistency_status(assumed_direction, empirical, row.get("row_count", 0)),
                "notes": _direction_notes(assumed_direction, empirical),
            }
        )
    return pd.DataFrame(rows)


def summarize_bucket_outcomes(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bucket_column in BUCKET_COLUMNS:
        if bucket_column not in panel.columns:
            continue
        for bucket_value, group in panel.dropna(subset=[bucket_column]).groupby(bucket_column):
            for horizon in HORIZONS:
                rows.append(_outcome_row(group, "bucket_name", bucket_column, "bucket_value", bucket_value, horizon, baseline=panel))
    return pd.DataFrame(rows)


def summarize_bucket_years(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    data = panel.assign(calendar_year=panel["date"].dt.year)
    for bucket_column in BUCKET_COLUMNS:
        if bucket_column not in data.columns:
            continue
        for (year, bucket_value), group in data.dropna(subset=[bucket_column]).groupby(["calendar_year", bucket_column]):
            for horizon in HORIZONS:
                row = _outcome_row(
                    group,
                    "bucket_name",
                    bucket_column,
                    "bucket_value",
                    bucket_value,
                    horizon,
                    baseline=data[data["calendar_year"] == year],
                )
                row["calendar_year"] = int(year)
                rows.append(row)
    return pd.DataFrame(rows)


def assign_candidate_rules(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.copy()
    data["high_quality_growth"] = (data["quality_bucket"] == "high_quality") & data["growth_bucket"].isin(
        ["high_growth", "positive_growth"]
    )
    data["quality_at_reasonable_risk"] = (data["quality_bucket"] == "high_quality") & data["balance_sheet_bucket"].isin(
        ["strong_balance_sheet", "normal_balance_sheet"]
    )
    data["cashflow_supported_growth"] = data["growth_bucket"].isin(["high_growth", "positive_growth"]) & (
        data["cashflow_quality_bucket"] == "high_cashflow_quality"
    )
    data["expensive_or_speculative_growth_proxy"] = data["growth_bucket"].isin(["high_growth", "positive_growth"]) & (
        data["valuation_proxy_bucket"] == "weak_or_negative_eps_fcf"
    )
    data["cheap_or_positive_per_share_quality"] = (data["valuation_proxy_bucket"] == "positive_eps_fcf_book") & (
        data["quality_bucket"] != "low_quality"
    )
    data["levered_growth_risk"] = data["growth_bucket"].isin(["high_growth", "positive_growth"]) & data[
        "balance_sheet_bucket"
    ].isin(["levered_balance_sheet", "distressed_or_high_risk"])
    data["fundamental_deterioration"] = data["quality_bucket"].isin(["low_quality"]) | (
        data["growth_bucket"] == "contracting"
    )
    data["fundamental_trap"] = (data["valuation_proxy_bucket"] == "positive_eps_fcf_book") & (
        data["quality_bucket"] == "low_quality"
    )
    data["stale_or_insufficient_data"] = (data["staleness_bucket"] == "very_stale_fundamentals") | data[
        BUCKET_COLUMNS
    ].apply(lambda row: any(str(value).startswith("insufficient") for value in row), axis=1)
    return data


def summarize_rule_outcomes(rule_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rule in _rule_columns():
        group = rule_panel[rule_panel[rule].fillna(False)]
        for horizon in HORIZONS:
            rows.append(_outcome_row(group, "rule_name", rule, None, None, horizon, baseline=rule_panel))
    return pd.DataFrame(rows)


def summarize_rule_years(rule_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    data = rule_panel.assign(calendar_year=rule_panel["date"].dt.year)
    for rule in _rule_columns():
        for year, year_panel in data.groupby("calendar_year"):
            group = year_panel[year_panel[rule].fillna(False)]
            for horizon in HORIZONS:
                row = _outcome_row(group, "rule_name", rule, None, None, horizon, baseline=year_panel)
                row["calendar_year"] = int(year)
                rows.append(row)
    return pd.DataFrame(rows)


def compute_scorecard_v0(bucketed: pd.DataFrame) -> pd.DataFrame:
    data = assign_candidate_rules(bucketed)
    conditions = [
        data["stale_or_insufficient_data"],
        data["fundamental_trap"],
        data["fundamental_deterioration"],
        data["levered_growth_risk"],
        data["expensive_or_speculative_growth_proxy"],
        data["high_quality_growth"],
        data["quality_at_reasonable_risk"],
        data["cashflow_supported_growth"],
    ]
    labels = [
        "insufficient_data",
        "fundamental_trap",
        "fundamental_deterioration",
        "levered_growth_risk",
        "speculative_growth",
        "high_quality_growth",
        "quality_cashflow_compounder",
        "cashflow_supported_growth",
    ]
    data["scorecard_label_v0"] = np.select(conditions, labels, default="neutral")
    score_map = {
        "high_quality_growth": 3,
        "quality_cashflow_compounder": 2,
        "cashflow_supported_growth": 1,
        "speculative_growth": 0,
        "neutral": 0,
        "levered_growth_risk": -1,
        "fundamental_deterioration": -2,
        "fundamental_trap": -3,
    }
    data["fundamental_score_v0"] = data["scorecard_label_v0"].map(score_map)
    keep = [
        "symbol",
        "effective_date",
        "sector",
        *[column for column in BUCKET_COLUMNS if column in data.columns],
        "scorecard_label_v0",
        "fundamental_score_v0",
        *[rule for rule in _rule_columns() if rule in data.columns],
    ]
    return data[keep].sort_values(["symbol", "effective_date"]).reset_index(drop=True)


def summarize_scorecard_outcomes(scorecard_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in scorecard_panel.dropna(subset=["scorecard_label_v0"]).groupby("scorecard_label_v0"):
        for horizon in HORIZONS:
            rows.append(_outcome_row(group, "scorecard_label_v0", label, None, None, horizon, baseline=scorecard_panel))
    return pd.DataFrame(rows)


def summarize_scorecard_years(scorecard_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    data = scorecard_panel.assign(calendar_year=scorecard_panel["date"].dt.year)
    for (year, label), group in data.dropna(subset=["scorecard_label_v0"]).groupby(["calendar_year", "scorecard_label_v0"]):
        for horizon in HORIZONS:
            row = _outcome_row(group, "scorecard_label_v0", label, None, None, horizon, baseline=data[data["calendar_year"] == year])
            row["calendar_year"] = int(year)
            rows.append(row)
    return pd.DataFrame(rows)


def compute_rebalance_and_holding_outputs(scorecard_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rebalance_panels = []
    for frequency, period in [("monthly", "M"), ("quarterly", "Q")]:
        dates = scorecard_panel[["date"]].drop_duplicates().sort_values("date")
        dates["period"] = dates["date"].dt.to_period(period)
        rebalance_dates = dates.groupby("period")["date"].max().reset_index(drop=True)
        subset = scorecard_panel[scorecard_panel["date"].isin(rebalance_dates)].copy()
        subset["rebalance_frequency"] = frequency
        rebalance_panels.append(subset)
    rebalances = pd.concat(rebalance_panels, ignore_index=True) if rebalance_panels else pd.DataFrame()

    feasibility_rows = []
    turnover_rows = []
    holding_rows = []
    for (frequency, date, label), group in rebalances.dropna(subset=["scorecard_label_v0"]).groupby(
        ["rebalance_frequency", "date", "scorecard_label_v0"]
    ):
        sector_counts = group["sector"].value_counts(normalize=True) if "sector" in group else pd.Series(dtype=float)
        feasibility_rows.append(
            {
                "rebalance_frequency": frequency,
                "date": _date_str(date),
                "scorecard_label_v0": label,
                "symbol_count": group["symbol"].nunique(),
                "max_sector_weight": float(sector_counts.max()) if not sector_counts.empty else np.nan,
                "dominant_sector": sector_counts.index[0] if not sector_counts.empty else None,
                "feasibility_label": _feasibility_label(group["symbol"].nunique(), float(sector_counts.max()) if not sector_counts.empty else np.nan),
            }
        )
    feasibility = pd.DataFrame(feasibility_rows)

    for (frequency, date, label), group in rebalances.dropna(subset=["scorecard_label_v0"]).groupby(
        ["rebalance_frequency", "date", "scorecard_label_v0"]
    ):
        same_date_baseline = rebalances[(rebalances["rebalance_frequency"] == frequency) & (rebalances["date"] == date)]
        for horizon in HORIZONS:
            row = _outcome_row(group, "scorecard_label_v0", label, None, None, horizon, baseline=same_date_baseline)
            row["rebalance_frequency"] = frequency
            row["date"] = _date_str(date)
            holding_rows.append(row)
    holding_summary = pd.DataFrame(holding_rows)

    for (frequency, label), group in rebalances.dropna(subset=["scorecard_label_v0"]).groupby(
        ["rebalance_frequency", "scorecard_label_v0"]
    ):
        previous: set[str] | None = None
        for date, date_group in group.groupby("date"):
            symbols = set(date_group["symbol"])
            if previous is None:
                turnover = np.nan
            elif previous or symbols:
                turnover = 1 - (len(previous & symbols) / max(len(previous | symbols), 1))
            else:
                turnover = np.nan
            turnover_rows.append(
                {
                    "rebalance_frequency": frequency,
                    "date": _date_str(date),
                    "scorecard_label_v0": label,
                    "symbol_count": len(symbols),
                    "turnover_rate": turnover,
                }
            )
            previous = symbols
    return feasibility, holding_summary, pd.DataFrame(turnover_rows)


def compute_v1_recommendation(
    scorecard_summary: pd.DataFrame,
    scorecard_stability: pd.DataFrame,
    scorecard_current: pd.DataFrame,
    relabeling_recommendation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    robustness = scorecard_summary.merge(scorecard_stability, on=["scorecard_label_v0", "horizon_days"], how="left")
    if not robustness.empty:
        robustness["keep_drop_sparse"] = np.select(
            [
                robustness["row_count"] < 50,
                robustness["scorecard_label_v0"].eq("insufficient_data"),
                robustness["behavior_label"].isin(["average_positive_median_positive", "defensive_low_downside"]),
                robustness["behavior_label"].isin(["tail_driven_upside", "downside_risk", "stable_negative"]),
            ],
            ["sparse", "data_quality_flag", "review", "risk_or_relabel"],
            default="review_or_drop",
        )
        robustness["observed_role"] = robustness["scorecard_label_v0"].map(SCORECARD_RELABELING_GUIDANCE).fillna("review")
    ready = False
    default_horizon = 63
    recommendation = pd.DataFrame(
        [
            {
                "v1_ready": False,
                "default_horizon_days": default_horizon,
                "recommended_stable_labels": "",
                "recommended_next_stage": "fundamental_scorecard_v0_1_review",
                "relabeling_output": "12c_fundamental_scorecard_relabeling_recommendation.parquet",
                "current_snapshot_output": "18_fundamental_scorecard_v0_1_current.parquet",
                "risk_caveats": (
                    "A single fundamentals score is misleading because defensive quality, speculative upside, rebound risk, "
                    "and data quality are different concepts."
                ),
                "why_not_ready": (
                    "V0.1 separates quality, opportunity, risk, and data-quality dimensions; these dimensions need review "
                    "before any v1 freeze."
                ),
                "ready_for_price_strength_cross_diagnostics": False,
            }
        ]
    )
    current = scorecard_current.copy()
    current["ready_for_v1"] = False
    current["default_horizon_days"] = default_horizon
    current["recommended_next_stage"] = "fundamental_scorecard_v0_1_review"
    current["observed_role_guidance"] = current["scorecard_label_v0"].map(SCORECARD_RELABELING_GUIDANCE).fillna("review")
    if not relabeling_recommendation.empty:
        current = current.merge(
            relabeling_recommendation[
                [
                    "scorecard_label_v0",
                    "recommended_label_v0_1",
                    "recommended_score_v0_1",
                    "recommended_role_v0_1",
                    "fundamental_quality_score_v0_1",
                    "fundamental_opportunity_score_v0_1",
                    "fundamental_risk_label_v0_1",
                    "fundamental_data_quality_flag_v0_1",
                    "recommended_keep_drop",
                ]
            ],
            on="scorecard_label_v0",
            how="left",
        )
    current["scorecard_label_v1_recommendation"] = "not_ready"
    columns = [
        "symbol",
        "effective_date",
        "sector",
        "scorecard_label_v0",
        "recommended_label_v0_1",
        "fundamental_quality_score_v0_1",
        "fundamental_opportunity_score_v0_1",
        "fundamental_risk_label_v0_1",
        "fundamental_data_quality_flag_v0_1",
        "ready_for_v1",
    ]
    passthrough_columns = [column for column in columns if column in current.columns]
    extra_columns = [column for column in current.columns if column not in passthrough_columns]
    return robustness, recommendation, current[passthrough_columns + extra_columns]


def compute_scorecard_relabeling_recommendation(scorecard_behavior: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in SCORECARD_RELABELING_GUIDANCE:
        label_rows = scorecard_behavior[scorecard_behavior["scorecard_label_v0"] == label].copy()
        selected = _select_relabeling_support_row(label_rows)
        behavior = selected.get("behavior_label", "insufficient_data")
        role_guidance = SCORECARD_RELABELING_GUIDANCE[label]
        recommendation = _relabeling_recommendation(label, behavior, selected)
        rows.append(
            {
                "scorecard_label_v0": label,
                "current_score_v0": SCORECARD_V0_SCORE_MAP[label],
                "observed_behavior_label": behavior,
                "observed_role_guidance": role_guidance,
                "recommended_label_v0_1": recommendation["recommended_label_v0_1"],
                "recommended_score_v0_1": recommendation["recommended_score_v0_1"],
                "recommended_role_v0_1": recommendation["recommended_role_v0_1"],
                "fundamental_quality_score_v0_1": recommendation["fundamental_quality_score_v0_1"],
                "fundamental_opportunity_score_v0_1": recommendation["fundamental_opportunity_score_v0_1"],
                "fundamental_risk_label_v0_1": recommendation["fundamental_risk_label_v0_1"],
                "fundamental_data_quality_flag_v0_1": recommendation["fundamental_data_quality_flag_v0_1"],
                "recommended_keep_drop": recommendation["recommended_keep_drop"],
                "primary_reason": recommendation["primary_reason"],
                "supporting_horizon_days": selected.get("horizon_days", pd.NA),
                "avg_forward_return_vs_baseline": selected.get("avg_forward_return_vs_baseline", np.nan),
                "median_forward_return_vs_baseline": selected.get("median_forward_return_vs_baseline", np.nan),
                "top_30pct_rate_vs_baseline": selected.get("top_30pct_rate_vs_baseline", np.nan),
                "bottom_30pct_rate_vs_baseline": selected.get("bottom_30pct_rate_vs_baseline", np.nan),
                "row_count": selected.get("row_count", 0),
                "score_policy_notes": recommendation["score_policy_notes"],
                "notes": recommendation["notes"],
            }
        )
    return pd.DataFrame(rows)


def compute_scorecard_behavior_summary(scorecard_summary: pd.DataFrame, scorecard_stability: pd.DataFrame) -> pd.DataFrame:
    if scorecard_summary.empty:
        return scorecard_summary.copy()
    data = scorecard_summary.copy()
    data["intended_label"] = data["scorecard_label_v0"]
    data["observed_role_guidance"] = data["scorecard_label_v0"].map(SCORECARD_RELABELING_GUIDANCE).fillna("review")
    if not scorecard_stability.empty:
        stability = scorecard_stability[
            ["scorecard_label_v0", "horizon_days", "completed_years_count", "stability_assessment"]
        ].drop_duplicates()
        data = data.merge(stability, on=["scorecard_label_v0", "horizon_days"], how="left", suffixes=("", "_stability"))
    return data


def _select_relabeling_support_row(label_rows: pd.DataFrame) -> dict[str, Any]:
    if label_rows.empty:
        return {}
    rows = label_rows.copy()
    rows["horizon_preference"] = rows["horizon_days"].map({63: 0, 126: 1, 252: 2}).fillna(9)
    rows["evidence_strength"] = (
        rows["avg_forward_return_vs_baseline"].abs().fillna(0)
        + rows["median_forward_return_vs_baseline"].abs().fillna(0)
        + rows["bottom_30pct_rate_vs_baseline"].abs().fillna(0)
    )
    preferred = rows.sort_values(["horizon_preference", "evidence_strength"], ascending=[True, False]).iloc[0]
    return preferred.to_dict()


def _relabeling_recommendation(label: str, behavior: str, row: dict[str, Any]) -> dict[str, Any]:
    if label == "insufficient_data":
        return _dimension_relabeling_row(
            "insufficient_data",
            "data_quality_flag_only",
            "keep_as_data_quality_flag",
            "Insufficient data is not an alpha signal.",
            "Use only to exclude or review sparse observations.",
        )
    if label == "neutral":
        return _dimension_relabeling_row(
            "neutral",
            "baseline_comparison_bucket",
            "keep_as_baseline",
            "Neutral is the comparison bucket.",
            "Quality and opportunity scores both remain 0.",
        )
    if label in {"high_quality_growth", "quality_cashflow_compounder", "cashflow_supported_growth"}:
        return _dimension_relabeling_row(
            SCORECARD_RELABELING_GUIDANCE[label],
            "defensive_quality",
            "relabel_review_dimensions",
            "Observed evidence is more defensive/downside-oriented than return-seeking.",
            "Use the quality dimension, not a composite alpha score.",
        )
    if label == "speculative_growth":
        return _dimension_relabeling_row(
            "speculative_upside_high_downside",
            "return_seeking_high_risk",
            "relabel_review_dimensions",
            "Average upside appears tail-driven and downside risk is material.",
            "Opportunity score is separate from weak quality and high downside-tail risk.",
        )
    if label == "fundamental_deterioration":
        return _dimension_relabeling_row(
            "deterioration_rebound_risk",
            "rebound_high_dispersion",
            "relabel_review_dimensions",
            "Observed behavior is not a clean negative-alpha bucket.",
            "Poor quality and rebound opportunity are separate dimensions.",
        )
    if label == "fundamental_trap":
        return _dimension_relabeling_row(
            "trap_or_rebound_mixed",
            "mixed_or_downside_risk",
            "relabel_review_dimensions",
            "Trap label is mixed and may include rebound behavior.",
            "Keep risk label separate from quality and opportunity dimensions.",
        )
    if label == "levered_growth_risk":
        return _dimension_relabeling_row(
            "levered_growth_mixed_risk",
            "risk_bucket_mixed_evidence",
            "relabel_review_dimensions",
            "Leverage risk evidence is mixed and should not be treated as clean short/avoid signal.",
            "Keep leverage risk separate from opportunity.",
        )
    return _dimension_relabeling_row(
        SCORECARD_RELABELING_GUIDANCE.get(label, "neutral"),
        "review",
        "review",
        "Evidence is mixed.",
        "Manual review required before freezing v0.1.",
    )


def _dimension_relabeling_row(
    recommended_label: str,
    recommended_role: str,
    keep_drop: str,
    primary_reason: str,
    notes: str,
) -> dict[str, Any]:
    policy = SCORECARD_V0_1_DIMENSION_POLICY[recommended_label]
    return {
        "recommended_label_v0_1": recommended_label,
        "recommended_score_v0_1": np.nan,
        "recommended_role_v0_1": recommended_role,
        "recommended_keep_drop": keep_drop,
        "primary_reason": primary_reason,
        "notes": notes,
        **policy,
    }


def current_snapshot(data: pd.DataFrame, date_column: str) -> pd.DataFrame:
    if data.empty:
        return data.copy()
    return (
        data.sort_values(["symbol", date_column])
        .groupby("symbol", as_index=False, sort=False)
        .tail(1)
        .sort_values("symbol")
        .reset_index(drop=True)
    )


def summarize_stability(years: pd.DataFrame, name_column: str, value_column: str | None) -> pd.DataFrame:
    if years.empty:
        columns = [name_column, "horizon_days", "completed_years_count", "stability_assessment"]
        if value_column:
            columns.insert(1, value_column)
        return pd.DataFrame(columns=columns)
    group_columns = [name_column, "horizon_days"] if value_column is None else [name_column, value_column, "horizon_days"]
    rows = []
    for keys, group in years.groupby(group_columns, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, key_values, strict=False))
        row.update(
            {
                "completed_years_count": int(group["calendar_year"].nunique()),
                "years_avg_above_baseline": int((group["avg_forward_return_vs_baseline"] > 0).sum()),
                "pct_years_avg_above_baseline": _mean(group["avg_forward_return_vs_baseline"] > 0),
                "mean_avg_return_vs_baseline": _mean(group["avg_forward_return_vs_baseline"]),
                "mean_top_rate_vs_baseline": _mean(group["top_30pct_rate_vs_baseline"]),
                "mean_bottom_rate_vs_baseline": _mean(group["bottom_30pct_rate_vs_baseline"]),
            }
        )
        row["stability_assessment"] = _stability_label(row)
        rows.append(row)
    return pd.DataFrame(rows)


def _outcome_row(
    group: pd.DataFrame,
    name_column: str,
    name_value: str,
    value_column: str | None,
    value_value: str | None,
    horizon: int,
    *,
    baseline: pd.DataFrame | None = None,
) -> dict[str, Any]:
    return_label = RETURN_LABELS[horizon]
    top_label = TOP_LABELS[horizon]
    bottom_label = BOTTOM_LABELS[horizon]
    group_data = _eligible_for_horizon(group, horizon)
    baseline_data = _eligible_for_horizon(baseline if baseline is not None else group, horizon)
    result = {
        name_column: name_value,
        "horizon_days": horizon,
        "row_count": int(len(group_data)),
        "symbol_count": int(group_data["symbol"].nunique()) if "symbol" in group_data else 0,
        "sector_count": int(group_data["sector"].nunique()) if "sector" in group_data else 0,
        "sector_concentration": _sector_concentration(group_data),
        "baseline_row_count": int(len(baseline_data)),
        "avg_forward_sector_relative_return": _mean(group_data[return_label]) if return_label in group_data else np.nan,
        "median_forward_sector_relative_return": _median(group_data[return_label]) if return_label in group_data else np.nan,
        "top_30pct_rate": _mean(group_data[top_label]) if top_label in group_data else np.nan,
        "bottom_30pct_rate": _mean(group_data[bottom_label]) if bottom_label in group_data else np.nan,
        "baseline_avg_forward_sector_relative_return": _mean(baseline_data[return_label]) if return_label in baseline_data else np.nan,
        "baseline_median_forward_sector_relative_return": _median(baseline_data[return_label]) if return_label in baseline_data else np.nan,
        "baseline_top_30pct_rate": _mean(baseline_data[top_label]) if top_label in baseline_data else np.nan,
        "baseline_bottom_30pct_rate": _mean(baseline_data[bottom_label]) if bottom_label in baseline_data else np.nan,
    }
    if value_column is not None:
        result[value_column] = value_value
    result["avg_forward_return_vs_baseline"] = (
        result["avg_forward_sector_relative_return"] - result["baseline_avg_forward_sector_relative_return"]
    )
    result["median_forward_return_vs_baseline"] = (
        result["median_forward_sector_relative_return"] - result["baseline_median_forward_sector_relative_return"]
    )
    result["top_30pct_rate_vs_baseline"] = result["top_30pct_rate"] - result["baseline_top_30pct_rate"]
    result["bottom_30pct_rate_vs_baseline"] = result["bottom_30pct_rate"] - result["baseline_bottom_30pct_rate"]
    behavior_label, confidence, notes = _behavior_assessment(result)
    result["behavior_label"] = behavior_label
    result["behavior_confidence"] = confidence
    result["interpretation_notes"] = notes
    return result


def _eligible_for_horizon(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return_label = RETURN_LABELS[horizon]
    if return_label not in data.columns:
        return data.iloc[0:0].copy()
    return data.dropna(subset=[return_label])


def _behavior_assessment(row: dict[str, Any]) -> tuple[str, str, str]:
    label_values = [str(row.get(key, "")) for key in ("scorecard_label_v0", "rule_name", "bucket_value")]
    if "insufficient_data" in label_values or any(value.startswith("insufficient_") for value in label_values):
        return "data_quality_flag", "high", "Data availability flag only; do not treat as alpha."
    if row.get("row_count", 0) < 50:
        return "sparse_unreliable", "low", "Too few eligible rows for stable interpretation."

    avg = row.get("avg_forward_return_vs_baseline", np.nan)
    median = row.get("median_forward_return_vs_baseline", np.nan)
    top = row.get("top_30pct_rate_vs_baseline", np.nan)
    bottom = row.get("bottom_30pct_rate_vs_baseline", np.nan)
    if any(pd.isna(value) for value in [avg, median, top, bottom]):
        return "mixed_or_regime_dependent", "low", "Missing one or more downside-aware comparison fields."

    if avg > 0 and median > 0 and top > 0 and bottom <= 0.02:
        return "average_positive_median_positive", "medium", "Average, median, and top-rate are above baseline."
    if avg > 0 and median <= 0 and top > 0 and bottom > 0.02:
        return "tail_driven_upside", "medium", "Average upside is positive, but median is weak and downside rate is worse."
    if avg > 0 and median < 0:
        return "average_positive_median_negative", "medium", "Average return is positive but typical outcome is not."
    if avg <= 0.01 and bottom < -0.02:
        return "defensive_low_downside", "medium", "Return lift is muted, but downside rate is lower than baseline."
    if bottom > 0.02 and avg <= 0.01:
        return "downside_risk", "medium", "Downside rate is worse than baseline without enough excess return."
    if avg < 0 and median < 0 and top < 0:
        return "stable_negative", "medium", "Average, median, and top-rate are below baseline."
    if abs(avg) <= 0.002 and abs(median) <= 0.002 and abs(top) <= 0.01 and abs(bottom) <= 0.01:
        return "neutral_like", "medium", "Outcome is close to eligible-universe baseline."
    return "mixed_or_regime_dependent", "medium", "Evidence is mixed across average, median, top-rate, and downside behavior."


def _assumed_feature_direction(feature: str) -> str:
    if feature in HIGHER_IS_WORSE_FEATURES:
        return "higher_generally_worse"
    if feature in {
        "gross_margin_ttm",
        "ebitda_margin_ttm",
        "net_margin_ttm",
        "fcf_margin_ttm",
        "roa_ttm",
        "roe_ttm",
        "roic_ttm",
        "asset_turnover_ttm",
        "revenue_yoy_ttm",
        "ebitda_yoy_ttm",
        "netinc_yoy_ttm",
        "fcf_yoy_ttm",
        "revenue_qoq",
        "revenue_yoy_quarter",
        "gross_margin_ttm_change_yoy",
        "net_margin_ttm_change_yoy",
        "fcf_to_netinc_ttm",
        "ncfo_to_netinc_ttm",
        "cash_to_assets",
        "equity_to_assets",
        "workingcapital_to_assets",
        "eps_ttm",
        "bvps",
        "fcfps_ttm",
    }:
        return "higher_generally_better"
    return "context_dependent"


def _feature_family(feature: str) -> str:
    if "margin" in feature or feature in {"roa_ttm", "roe_ttm", "roic_ttm", "asset_turnover_ttm"}:
        return "quality_profitability"
    if "yoy" in feature or "qoq" in feature:
        return "growth_change"
    if feature in {"fcf_to_netinc_ttm", "ncfo_to_netinc_ttm", "accruals_to_assets", "capex_to_revenue_ttm"}:
        return "cashflow_quality"
    if "debt" in feature or "liabilities" in feature or feature.endswith("_to_assets"):
        return "balance_sheet"
    if feature in {"eps_ttm", "bvps", "fcfps_ttm"}:
        return "per_share"
    if "age" in feature:
        return "staleness"
    return "other"


def _empirical_direction_label(avg: float, median: float, top: float, bottom: float) -> str:
    if any(pd.isna(value) for value in [avg, median, top, bottom]):
        return "insufficient_data"
    if avg > 0 and median > 0 and top > 0 and bottom <= 0:
        return "empirically_positive"
    if avg < 0 and median < 0 and top < 0:
        return "empirically_negative"
    if avg > 0 and median <= 0:
        return "tail_positive_median_weak"
    return "mixed"


def _direction_consistency_status(assumed: str, empirical: str, row_count: Any) -> str:
    if pd.isna(row_count) or int(row_count) < 50 or empirical == "insufficient_data":
        return "insufficient_data"
    if assumed == "context_dependent":
        return "context_dependent"
    if empirical == "mixed" or empirical == "tail_positive_median_weak":
        return "mixed"
    if assumed == "higher_generally_better" and empirical == "empirically_positive":
        return "consistent"
    if assumed == "higher_generally_worse" and empirical == "empirically_positive":
        return "consistent"
    return "opposite"


def _direction_notes(assumed: str, empirical: str) -> str:
    if assumed == "context_dependent":
        return "Interpret with feature context rather than a fixed monotonic direction."
    if empirical == "tail_positive_median_weak":
        return "Average spread is positive but median behavior is weak; inspect tails."
    return ""


def _mean_with_min_count(data: pd.DataFrame, columns: list[str], min_count: int) -> pd.Series:
    available = [column for column in columns if column in data.columns]
    if not available:
        return pd.Series(np.nan, index=data.index)
    values = data[available]
    return values.mean(axis=1).where(values.notna().sum(axis=1) >= min_count)


def _tertile_labels(series: pd.Series, *, higher_is_worse: bool) -> pd.Series:
    rank = series.rank(pct=True)
    strong = rank <= 0.33 if higher_is_worse else rank >= 0.67
    weak = rank >= 0.67 if higher_is_worse else rank <= 0.33
    return pd.Series(np.select([strong, weak], ["strong", "weak"], default="middle"), index=series.index)


def _rule_columns() -> list[str]:
    return [
        "high_quality_growth",
        "quality_at_reasonable_risk",
        "cashflow_supported_growth",
        "expensive_or_speculative_growth_proxy",
        "cheap_or_positive_per_share_quality",
        "levered_growth_risk",
        "fundamental_deterioration",
        "fundamental_trap",
        "stale_or_insufficient_data",
    ]


def _empty_outcome_row(name_column: str, name_value: str, horizon: int) -> dict[str, Any]:
    return {name_column: name_value, "horizon_days": horizon, "row_count": 0, "symbol_count": 0}


def _safe_corr(left: pd.Series, right: pd.Series, method: str) -> float:
    data = pd.concat([left, right], axis=1).dropna()
    if len(data) < 3 or data.iloc[:, 0].nunique() < 2 or data.iloc[:, 1].nunique() < 2:
        return np.nan
    if method == "spearman":
        ranked = data.rank()
        return float(ranked.iloc[:, 0].corr(ranked.iloc[:, 1], method="pearson"))
    return float(data.iloc[:, 0].corr(data.iloc[:, 1], method=method))


def _mean(series: pd.Series | np.ndarray | list[Any]) -> float:
    values = pd.Series(series).dropna()
    return float(values.mean()) if not values.empty else np.nan


def _median(series: pd.Series | np.ndarray | list[Any]) -> float:
    values = pd.Series(series).dropna()
    return float(values.median()) if not values.empty else np.nan


def _sector_concentration(group: pd.DataFrame) -> float:
    if group.empty or "sector" not in group:
        return np.nan
    shares = group["sector"].value_counts(normalize=True)
    return float(shares.max()) if not shares.empty else np.nan


def _coverage_warning(non_null_rate: float, feature: str) -> str:
    if feature in HIGHER_IS_WORSE_FEATURES:
        direction = "higher_is_worse"
    else:
        direction = "higher_is_better_or_contextual"
    if non_null_rate < 0.50:
        return f"sparse; {direction}"
    return direction


def _stability_label(row: dict[str, Any]) -> str:
    if row["completed_years_count"] < 3:
        return "insufficient_year_data"
    if row["pct_years_avg_above_baseline"] >= 0.60 and row["mean_avg_return_vs_baseline"] > 0:
        return "stable_positive"
    if row["pct_years_avg_above_baseline"] <= 0.40 and row["mean_avg_return_vs_baseline"] < 0:
        return "stable_negative_or_risk"
    return "mixed_or_regime_dependent"


def _feasibility_label(symbol_count: int, sector_concentration: float) -> str:
    if symbol_count < 10:
        return "too_sparse"
    if pd.notna(sector_concentration) and sector_concentration > 0.40:
        return "sector_concentrated"
    return "research_feasible"


def _date_str(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return str(pd.Timestamp(value).date())


def _assert_output_convention(output_dir: Path, paths: dict[str, Path]) -> None:
    unexpected_suffixes = {path.suffix for path in paths.values()} - {".parquet", ".md"}
    if unexpected_suffixes:
        raise RuntimeError(f"Unexpected fundamental research output suffixes: {sorted(unexpected_suffixes)}")
    for path in output_dir.glob("*"):
        if path.suffix in {".csv", ".json"}:
            raise RuntimeError(f"Unexpected CSV/JSON output in fundamental research directory: {path}")


def _remove_stale_outputs(output_dir: Path) -> None:
    for filename in STALE_OUTPUT_FILENAMES:
        path = output_dir / filename
        if path.exists():
            path.unlink()


def render_feature_report(coverage: pd.DataFrame, diagnostics: pd.DataFrame, direction_audit: pd.DataFrame) -> str:
    sparse = coverage[(coverage["scope"] == "overall") & (coverage["non_null_rate"] < 0.50)]
    strongest = diagnostics.sort_values("strong_minus_weak_avg_return", ascending=False).head(8)
    direction_sample = direction_audit.sort_values("strong_minus_weak_avg_return", ascending=False).head(12)
    return "\n".join(
        [
            "# Fundamental Feature Diagnostics",
            "",
            "Diagnostic report. Research-only; not a production signal.",
            "",
            f"Overall features reviewed: {coverage[coverage['scope'] == 'overall']['feature'].nunique()}",
            f"Sparse features flagged: {len(sparse)}",
            "",
            "## Strongest Initial Spreads",
            _markdown_table(
                strongest[
                    [
                        "feature",
                        "horizon_days",
                        "strong_minus_weak_avg_return",
                        "strong_minus_weak_median_return",
                        "strong_minus_weak_top_30pct_rate",
                        "strong_minus_weak_bottom_30pct_rate",
                        "row_count",
                    ]
                ]
            ),
            "",
            "## Feature Direction Audit",
            _markdown_table(
                direction_sample[
                    [
                        "feature",
                        "feature_family",
                        "assumed_direction",
                        "horizon_days",
                        "empirical_direction_label",
                        "direction_consistency_status",
                    ]
                ]
            ),
            "",
            "Use these diagnostics as research evidence only; they are not a production signal.",
            "",
            _output_file_guide("feature"),
        ]
    )


def render_bucket_report(summary: pd.DataFrame, stability: pd.DataFrame) -> str:
    best = summary.sort_values("avg_forward_return_vs_baseline", ascending=False).head(10) if not summary.empty else summary
    return "\n".join(
        [
            "# Fundamental Bucket Diagnostics",
            "",
            "Diagnostic report. Research-only bucket evidence, not a scorecard recommendation.",
            "",
            "Buckets use sector-relative ranks where appropriate. Leverage, accruals, dilution, and staleness are treated as higher-is-worse.",
            "",
            "## Best Bucket Outcomes",
            _markdown_table(
                best[
                    [
                        "bucket_name",
                        "bucket_value",
                        "horizon_days",
                        "avg_forward_return_vs_baseline",
                        "median_forward_return_vs_baseline",
                        "bottom_30pct_rate_vs_baseline",
                        "behavior_label",
                        "row_count",
                    ]
                ]
            ),
            "",
            f"Stability rows: {len(stability)}",
            "",
            _output_file_guide("bucket"),
        ]
    )


def render_candidate_report(summary: pd.DataFrame, stability: pd.DataFrame) -> str:
    best = summary.sort_values("avg_forward_return_vs_baseline", ascending=False).head(10) if not summary.empty else summary
    return "\n".join(
        [
            "# Fundamental Candidate Rules",
            "",
            "Diagnostic report. Research-only candidate-rule evidence.",
            "",
            "Candidate rules compare forward sector-relative outcomes against the eligible baseline.",
            "",
            _markdown_table(
                best[
                    [
                        "rule_name",
                        "horizon_days",
                        "avg_forward_return_vs_baseline",
                        "median_forward_return_vs_baseline",
                        "top_30pct_rate_vs_baseline",
                        "bottom_30pct_rate_vs_baseline",
                        "behavior_label",
                        "row_count",
                    ]
                ]
            ),
            "",
            f"Rule stability rows: {len(stability)}",
            "",
            _output_file_guide("candidate"),
        ]
    )


def render_scorecard_report(
    summary: pd.DataFrame,
    stability: pd.DataFrame,
    relabeling_recommendation: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "# Fundamental Scorecard v0",
            "",
            "Scorecard-level report. Research-only; labels are not coherent enough for v1.",
            "",
            _markdown_table(
                summary[
                    [
                        "scorecard_label_v0",
                        "observed_role_guidance",
                        "horizon_days",
                        "avg_forward_return_vs_baseline",
                        "median_forward_return_vs_baseline",
                        "bottom_30pct_rate_vs_baseline",
                        "behavior_label",
                        "row_count",
                    ]
                ].head(20)
            ),
            "",
            f"Stability rows: {len(stability)}",
            "",
            "## Relabeling Recommendation",
            "",
            "Inspect `12c_fundamental_scorecard_relabeling_recommendation.parquet` before freezing any scorecard labels. A single fundamentals score is misleading because defensive quality, speculative upside, rebound risk, and data quality are different concepts. V0.1 therefore separates quality score, opportunity score, risk label, and data-quality flag.",
            "",
            _markdown_table(
                relabeling_recommendation[
                    [
                        "scorecard_label_v0",
                        "recommended_label_v0_1",
                        "fundamental_quality_score_v0_1",
                        "fundamental_opportunity_score_v0_1",
                        "fundamental_risk_label_v0_1",
                        "fundamental_data_quality_flag_v0_1",
                        "recommended_keep_drop",
                        "primary_reason",
                    ]
                ]
            ),
            "",
            "Next recommended step: review the separate v0.1 dimensions before any v1 freeze.",
            "",
            _output_file_guide("scorecard"),
        ]
    )


def render_feasibility_report(feasibility: pd.DataFrame, holding_summary: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Fundamental Rebalance Feasibility and Holding Periods",
            "",
            "Feasibility report. Monthly and quarterly rebalance snapshots are research-only checks.",
            "",
            _markdown_table(
                holding_summary[
                    [
                        "rebalance_frequency",
                        "date",
                        "scorecard_label_v0",
                        "horizon_days",
                        "avg_forward_return_vs_baseline",
                        "median_forward_return_vs_baseline",
                        "behavior_label",
                        "row_count",
                    ]
                ].head(20)
            ),
            "",
            f"Feasibility rows: {len(feasibility)}",
            "",
            "Next recommended step: use this only after relabeling review to decide whether any bucket is operationally usable.",
            "",
            _output_file_guide("feasibility"),
        ]
    )


def render_v1_report(
    recommendation: pd.DataFrame,
    robustness: pd.DataFrame,
    relabeling_recommendation: pd.DataFrame,
) -> str:
    row = recommendation.iloc[0].to_dict() if not recommendation.empty else {}
    return "\n".join(
        [
            "# Fundamental Scorecard v1 Recommendation",
            "",
            "Recommendation-level report. Research-only; no production signal should be derived from this output.",
            "",
            f"Ready for v1: {row.get('v1_ready', False)}",
            f"Recommended next stage: {row.get('recommended_next_stage', 'fundamental_scorecard_v0_1_review')}",
            f"Default horizon days: {row.get('default_horizon_days', 63)}",
            f"Recommended labels: {row.get('recommended_stable_labels', '') or 'none yet'}",
            "",
            "## Why v1 Is Not Ready",
            "",
            "- A single fundamentals score is misleading because defensive quality, speculative upside, rebound risk, and data quality are different concepts.",
            "- V0.1 now separates quality score, opportunity score, risk label, and data-quality flag.",
            "- Some good-sounding labels behave more like defensive quality than return-seeking alpha.",
            "- Some bad-sounding labels look tail-driven, rebound-like, or mixed rather than clean negative alpha.",
            "- `insufficient_data` is a data-quality flag only.",
            "- V0.1 dimension review in `12c_fundamental_scorecard_relabeling_recommendation.parquet` is required before v1.",
            "",
            _markdown_table(
                robustness[
                    [
                        "scorecard_label_v0",
                        "observed_role",
                        "horizon_days",
                        "keep_drop_sparse",
                        "behavior_label",
                        "avg_forward_return_vs_baseline",
                        "median_forward_return_vs_baseline",
                        "bottom_30pct_rate_vs_baseline",
                    ]
                ].head(20)
            ),
            "",
            "## Relabeling Output Snapshot",
            "",
            _markdown_table(
                relabeling_recommendation[
                    [
                        "scorecard_label_v0",
                        "recommended_label_v0_1",
                        "fundamental_quality_score_v0_1",
                        "fundamental_opportunity_score_v0_1",
                        "fundamental_risk_label_v0_1",
                        "fundamental_data_quality_flag_v0_1",
                        "recommended_keep_drop",
                        "primary_reason",
                    ]
                ]
            ),
            "",
            _output_file_guide("v1"),
        ]
    )


def _markdown_table(data: pd.DataFrame) -> str:
    if data.empty:
        return "_No rows._"
    columns = list(data.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for record in data.to_dict(orient="records"):
        values = []
        for column in columns:
            value = record[column]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _output_file_guide(report_key: str) -> str:
    rows = ["## Output File Guide", "", "| File | Purpose |", "| --- | --- |"]
    for filename, purpose in REPORT_FILE_GUIDES[report_key]:
        rows.append(f"| {filename} | {purpose} |")
    return "\n".join(rows)


def _progress(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


def terminal_summary(paths: dict[str, Path], summary: dict[str, Any]) -> str:
    return (
        f"Wrote fundamental research v0 to {summary['output_dir']} "
        f"outputs={summary['output_count']} rows={summary['panel_rows']} symbols={summary['symbols']} "
        f"grain={summary['research_panel_grain']} "
        f"label_matches={summary['event_panel_label_match_rows']}/{summary['event_panel_rows']} "
        f"date_range={summary['date_min']}..{summary['date_max']}"
    )
