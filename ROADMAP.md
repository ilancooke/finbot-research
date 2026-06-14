# Finbot Fundamentals Research Roadmap

## Current Strategic Direction

The fundamentals research phase should proceed as a **context layer for price-strength interpretation**, not as a standalone fundamentals ranking scorecard yet.

The original goal remains valid:

> Use fundamentals to help explain whether price strength is supported by business quality, growth, valuation, balance-sheet strength, risk, or data-quality concerns.

The key update is that fundamentals should not currently be collapsed into one composite score.

The latest research suggests fundamentals are better represented as separate dimensions:

```text
fundamental_quality_score
fundamental_opportunity_score
fundamental_risk_label
fundamental_data_quality_flag
```

This allows Finbot to say things like:

```text
This stock has strong price action and defensive fundamental quality.

This stock has strong price action, but the fundamental profile looks speculative and high-risk.

This stock has price strength, but fundamental data is insufficient, so confidence should be lower.
```

---

## Why Fundamentals Should Be Context First

Price strength already produced a usable research artifact.

The fundamentals workflow produced useful evidence, but not a clean standalone return-ranking signal.

The first-pass fundamentals results showed:

```text
Quality-style buckets often behave defensively rather than return-seeking.
Speculative or deteriorating buckets can show upside, but often with tail risk.
Insufficient data behaves like a data-quality issue, not an alpha signal.
A single fundamentals score would hide important distinctions.
```

Therefore, the next research phase should ask:

> Among price-strength candidates, do fundamentals help distinguish cleaner strength from speculative strength?

This is more useful than asking:

> Do fundamentals alone predict returns?

---

# Current State

## Completed Price-Strength Work

Price-strength research has progressed through:

```text
feature-label diagnostics
bucket diagnostics
candidate rules
scorecard v0
feasibility checks
holding-period simulation
portfolio simulation
equity-curve backtest
robustness checks
horizon sensitivity
turnover/cost efficiency
scorecard v1
```

Current status:

```text
Price-strength scorecard v1 exists.
It is usable as a research artifact.
It should remain research-only.
```

---

## Completed Fundamentals Work

The fundamentals workflow has been implemented in `finbot-research`.

Important design choice:

```text
Fundamentals are processed at filing-event grain:
one row per symbol,effective_date
joined to forward labels on that same effective_date.
```

This avoids expanding quarterly/annual fundamentals into daily rows, which caused memory blowups and was conceptually unnecessary.

Current fundamentals output directory:

```text
data/research/fundamental_research_v0/
```

Current output convention:

```text
parquet tables only
Markdown reports allowed
no CSV duplicates
no metadata sidecar JSON files
sequentially numbered parquet filenames
```

---

## Current Fundamentals Reports

```text
fundamental_feature_diagnostics_report.md
fundamental_bucket_diagnostics_report.md
fundamental_candidate_rules_report.md
fundamental_scorecard_v0_report.md
fundamental_feasibility_and_holding_period_report.md
fundamental_scorecard_v1_recommendation_report.md
```

Each report should include an `Output File Guide` mapping the report to its parquet outputs.

---

## Current Important Fundamentals Outputs

```text
01_fundamental_feature_coverage.parquet
02_fundamental_feature_label_diagnostics.parquet
02b_fundamental_feature_direction_audit.parquet

03_fundamental_bucket_summary.parquet
04_fundamental_bucket_years.parquet
05_fundamental_bucket_stability.parquet

06_fundamental_candidate_rules.parquet
07_fundamental_candidate_rule_years.parquet
08_fundamental_candidate_rule_stability.parquet

09_fundamental_scorecard_v0.parquet
10_fundamental_scorecard_v0_current.parquet
11_fundamental_scorecard_v0_summary.parquet
12_fundamental_scorecard_v0_stability.parquet
12b_fundamental_scorecard_behavior_summary.parquet
12c_fundamental_scorecard_relabeling_recommendation.parquet

13_fundamental_rebalance_feasibility.parquet
14_fundamental_holding_period_summary.parquet
15_fundamental_holding_period_turnover.parquet

16_fundamental_robustness_summary.parquet
17_fundamental_scorecard_v1_recommendation.parquet
18_fundamental_scorecard_v0_1_current.parquet
```

---

# Current Fundamentals Interpretation

## Fundamentals v1 Is Not Ready

Current recommendation:

```text
Ready for v1: False
Recommended next stage: fundamental_scorecard_v0_1_review
Recommended labels: none yet
```

This is the correct conclusion.

Fundamentals should not be frozen as v1 until:
- label meanings are stable,
- the v0.1 interpretation layer has been tested against price strength,
- insufficient data remains a data-quality flag only,
- speculative and rebound buckets carry explicit risk labels,
- quality buckets are not mistaken for return-seeking alpha.

---

## Current v0.1 Dimensional Mapping

The current proposed v0.1 structure separates quality, opportunity, risk, and data quality.

```text
speculative_growth
→ speculative_upside_high_downside
quality_score = -1
opportunity_score = +2
risk_label = high_downside_tail_risk
data_quality_flag = false

fundamental_deterioration
→ deterioration_rebound_risk
quality_score = -2
opportunity_score = +1
risk_label = high_dispersion_rebound
data_quality_flag = false

high_quality_growth
→ defensive_quality_growth
quality_score = +2
opportunity_score = 0
risk_label = lower_downside
data_quality_flag = false

cashflow_supported_growth
→ defensive_cashflow_quality_or_weak_return_quality
quality_score = +2
opportunity_score = 0
risk_label = lower_downside
data_quality_flag = false

quality_cashflow_compounder
→ defensive_quality_cashflow
quality_score = +2
opportunity_score = 0
risk_label = lower_downside
data_quality_flag = false

fundamental_trap
→ trap_or_rebound_mixed
quality_score = -2
opportunity_score = 0
risk_label = mixed_trap_rebound
data_quality_flag = false

levered_growth_risk
→ levered_growth_mixed_risk
quality_score = -1
opportunity_score = 0
risk_label = leverage_risk_mixed
data_quality_flag = false

insufficient_data
→ insufficient_data
quality_score = null
opportunity_score = null
risk_label = insufficient_data
data_quality_flag = true

neutral
→ neutral
quality_score = 0
opportunity_score = 0
risk_label = neutral
data_quality_flag = false
```

This mapping should be treated as **provisional**.

Use it for cross-diagnostics, but do not freeze it as v1.

---

# Reconciled Roadmap

## Phase 1: Fundamental Data Inventory and Coverage

### Status

Completed enough for current research.

### Key Decision

Fundamentals should remain at filing-event grain.

Do not expand fundamentals into daily carried-forward rows.

### Current Outputs

```text
01_fundamental_feature_coverage.parquet
fundamental_feature_diagnostics_report.md
```

---

## Phase 2: Fundamental Feature Diagnostics

### Status

Completed for v0.

### Current Outputs

```text
02_fundamental_feature_label_diagnostics.parquet
02b_fundamental_feature_direction_audit.parquet
fundamental_feature_diagnostics_report.md
```

### Current Finding

The strongest feature-level evidence comes from balance-sheet, liquidity, leverage, and accrual features.

However, many effects are mixed or tail-driven.

---

## Phase 3: Fundamental Bucket Diagnostics

### Status

Completed for v0.

### Current Outputs

```text
03_fundamental_bucket_summary.parquet
04_fundamental_bucket_years.parquet
05_fundamental_bucket_stability.parquet
fundamental_bucket_diagnostics_report.md
```

### Current Finding

The best average-return buckets are often not clean quality buckets.

Many have:
- positive average excess return,
- weak or negative median excess return,
- elevated bottom-tail risk.

These are better interpreted as speculative or rebound-like behavior.

---

## Phase 4: Candidate Rule Diagnostics

### Status

Completed for v0.

### Current Outputs

```text
06_fundamental_candidate_rules.parquet
07_fundamental_candidate_rule_years.parquet
08_fundamental_candidate_rule_stability.parquet
fundamental_candidate_rules_report.md
```

### Current Finding

The strongest candidate rules are often speculative or tail-driven rather than cleanly quality-based.

---

## Phase 5: Fundamental Scorecard v0

### Status

Completed as a diagnostic scorecard.

### Current Outputs

```text
09_fundamental_scorecard_v0.parquet
10_fundamental_scorecard_v0_current.parquet
11_fundamental_scorecard_v0_summary.parquet
12_fundamental_scorecard_v0_stability.parquet
12b_fundamental_scorecard_behavior_summary.parquet
fundamental_scorecard_v0_report.md
```

### Current Finding

The original v0 labels are useful as raw diagnostic groupings, but not coherent enough for v1.

---

## Phase 6: Fundamental Scorecard v0.1 Interpretation Layer

### Status

Partially completed.

### Current Outputs

```text
12c_fundamental_scorecard_relabeling_recommendation.parquet
18_fundamental_scorecard_v0_1_current.parquet
```

### Current Finding

The v0.1 interpretation layer should use separate dimensions:

```text
fundamental_quality_score_v0_1
fundamental_opportunity_score_v0_1
fundamental_risk_label_v0_1
fundamental_data_quality_flag_v0_1
```

### Recommendation

Accept this provisionally for cross-diagnostics.

Do not freeze as v1.

---

## Phase 7: Feasibility and Holding-Period Checks

### Status

Completed for v0.

### Current Outputs

```text
13_fundamental_rebalance_feasibility.parquet
14_fundamental_holding_period_summary.parquet
15_fundamental_holding_period_turnover.parquet
fundamental_feasibility_and_holding_period_report.md
```

### Current Finding

Use these as diagnostics only.

Do not treat fundamentals as operationally usable by themselves until cross-diagnostics are complete.

---

## Phase 8: Fundamental Scorecard v1 Recommendation

### Status

Completed.

### Current Outputs

```text
16_fundamental_robustness_summary.parquet
17_fundamental_scorecard_v1_recommendation.parquet
fundamental_scorecard_v1_recommendation_report.md
```

### Current Finding

Fundamentals v1 is not ready.

```text
Ready for v1: False
Recommended next stage: fundamental_scorecard_v0_1_review
Recommended labels: none yet
```

---

# Next Recommended Research Step

## Phase 9: Price Strength × Fundamentals Cross-Diagnostics

This is the next step.

Do not create a standalone fundamentals v1 yet.

Use the current fundamentals v0.1 interpretation layer as context for price-strength v1.

## Core Research Question

```text
Among price-strength candidates, do fundamentals help distinguish cleaner strength from speculative strength?
```

## Secondary Questions

```text
Do high-conviction price-strength names with high fundamental quality have better downside behavior?

Do high-conviction price-strength names with speculative fundamentals have higher upside but worse drawdown/tail risk?

Do high-volatility price-strength traps with weak fundamentals perform worse?

Do momentum-resilience names with defensive fundamentals behave differently?

Does insufficient fundamental data reduce confidence in price-strength candidates?

Does fundamental opportunity score add useful information beyond price strength?

Does fundamental quality score help identify lower-risk price-strength candidates?
```

## Suggested Inputs

Verify exact paths in repo.

```text
data/research/price_strength_scorecard_v1/equity_price_strength_scorecard_v1.parquet
data/research/price_strength_scorecard_v1/equity_price_strength_scorecard_v1_current.parquet

data/research/fundamental_research_v0/09_fundamental_scorecard_v0.parquet
data/research/fundamental_research_v0/18_fundamental_scorecard_v0_1_current.parquet
```

For historical cross-diagnostics, use historical scorecard rows where possible.

For current inspection, use current snapshots.

## Suggested Output Directory

```text
data/research/price_strength_fundamental_cross_diagnostics/
```

## Suggested Outputs

Use reduced output convention:

```text
01_price_strength_fundamental_cross_summary.parquet
02_price_strength_fundamental_cross_years.parquet
03_price_strength_fundamental_cross_stability.parquet
04_price_strength_fundamental_current_candidates.parquet
price_strength_fundamental_cross_diagnostics_report.md
```

No CSV files. No metadata JSON sidecars.

## Candidate Cross-Diagnostic Groupings

Evaluate price-strength buckets crossed with:

```text
fundamental_quality_score_v0_1
fundamental_opportunity_score_v0_1
fundamental_risk_label_v0_1
fundamental_data_quality_flag_v0_1
recommended_label_v0_1
```

Example combinations:

```text
high_conviction_price_strength + quality_score >= 2
high_conviction_price_strength + opportunity_score >= 2
high_conviction_price_strength + risk_label = high_downside_tail_risk
high_conviction_price_strength + data_quality_flag = true

high_volatility_trap + quality_score < 0
high_volatility_trap + risk_label in high-risk categories
momentum_resilience + quality_score >= 2
momentum_resilience + lower_downside
```

## Candidate Combined Labels

```text
quality_confirmed_price_strength
defensive_quality_strength
speculative_price_strength
deterioration_rebound_strength
price_strength_with_fundamental_risk
price_strength_with_insufficient_fundamental_data
trap_confirmed_by_fundamentals
quality_without_price_strength
neutral_or_unconfirmed
```

## Success Criteria

```text
We know whether fundamentals improve interpretation of price-strength candidates.
We can distinguish cleaner price strength from speculative price strength.
We can identify where fundamental risk reinforces price-strength trap risk.
We know whether the fundamentals v0.1 dimensions are useful enough to keep.
We still do not freeze fundamentals v1 unless cross-diagnostics support it.
```

---

# Future Phase: Combined Research Candidate v0

Only proceed here if cross-diagnostics are useful.

## Goal

Create a research-only combined candidate output that joins:

```text
price_strength_scorecard_v1
fundamental_scorecard_v0_1
combined_research_labels
```

## Suggested Output Directory

```text
data/research/combined_equity_research_candidates_v0/
```

## Suggested Outputs

```text
01_combined_research_candidates_current.parquet
02_combined_research_candidate_summary.parquet
combined_research_candidates_v0_report.md
```

## Suggested Fields

```text
symbol
date
sector

price_strength_bucket_v1
price_strength_score_v1
price_strength_confidence_v1
price_strength_risk_label_v1

fundamental_label_v0_1
fundamental_quality_score_v0_1
fundamental_opportunity_score_v0_1
fundamental_risk_label_v0_1
fundamental_data_quality_flag_v0_1

combined_research_label_v0
combined_candidate_priority_v0
bullish_evidence
risk_evidence
data_quality_notes
```

---

# Future Phase: Standalone Fundamentals Scorecard

Do not prioritize this yet.

A standalone fundamentals scorecard may make sense later if fundamentals demonstrate stable standalone usefulness across:

```text
horizons
sectors
regimes
rebalance dates
downside metrics
current snapshots
```

It should only be revisited after price-strength cross-diagnostics.

## Conditions for Standalone Fundamentals v1

Consider a standalone fundamentals v1 only if:

```text
labels are stable and coherent,
quality/opportunity/risk dimensions are empirically useful,
missing data behavior is well understood,
standalone performance is robust enough to matter,
and the scorecard adds value beyond price-strength context.
```

---

# What Not To Do Yet

Do not:

```text
freeze fundamentals v1
create a reusable research framework
create a production signal
create trading recommendations
over-optimize score weights
collapse fundamentals into one composite score
expand fundamentals to daily carried-forward rows
promote insufficient_data to an alpha signal
```

---

# Resume Point

Resume later from:

```text
price_strength_fundamental_cross_diagnostics
```

The next useful coding-agent task should be to implement cross-diagnostics between:

```text
price_strength_scorecard_v1
fundamental_scorecard_v0_1
```

---

# Handoff Summary

Current state:

```text
Price-strength scorecard v1 exists and is usable as a research artifact.
Fundamentals v0 diagnostics exist and are useful.
Fundamentals v0.1 interpretation layer exists but is not v1.
Fundamentals should remain separate quality/opportunity/risk/data-quality dimensions.
The next recommended research step is price-strength × fundamentals cross-diagnostics.
```

Main principle:

> Fundamentals should first help explain, qualify, and risk-label price-strength candidates. A standalone fundamentals scorecard can be revisited later if the evidence supports it.
