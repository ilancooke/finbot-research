# Finbot Fundamentals Research Roadmap

## Purpose

The next research phase is to add **fundamental context** to Finbot.

The goal is not to replace the price-strength signal. The goal is to answer:

> Is a stock’s price strength supported by business quality, growth, valuation, and balance-sheet evidence?

Price strength tells us that the market is rewarding a stock. Fundamentals help us understand whether that move appears supported, speculative, expensive, cheap, improving, or risky.

---

## End Goal

Create a fundamentals research module that can classify each stock into interpretable buckets such as:

```text
high_quality_growth
quality_at_reasonable_price
expensive_growth
cheap_but_weak
fundamental_turnaround
fundamental_deterioration
levered_risk
insufficient_fundamental_data
```

Eventually, this should support combined explanations like:

```text
This stock has strong price action and improving fundamentals.

This stock has strong price action but weak profitability and expensive valuation.

This stock is a price-strength candidate, but fundamentals do not yet support the move.
```

---

## Guiding Principles

```text
Start with interpretable rules and diagnostics.
Avoid black-box modeling too early.
Use sector-relative comparisons wherever possible.
Preserve raw values and percentile ranks.
Make missingness explicit.
Avoid lookahead bias.
Keep outputs research-only until the schema and evidence are stable.
```

---

# Phase 1: Fundamental Data Inventory and Coverage

## Goal

Understand what fundamental data is available, how complete it is, and how far back it goes.

## Questions to Answer

```text
Which fundamental tables are available?
Which symbols have usable data?
How many years/quarters of history are available?
Which metrics are sparse or stale?
Do fundamentals align cleanly to symbol/date observations?
Which fields are point-in-time safe?
Which fields require careful reporting-date or filing-date handling?
```

## Candidate Inputs

Use existing shared data if available:

```text
data/ratios/
data/fundamentals/
data/reference/
data/catalog/
```

Exact paths should follow existing Finbot repo conventions.

## Suggested Output Directory

```text
data/research/fundamental_data_diagnostics/
```

## Suggested Outputs

```text
equity_fundamental_coverage_summary.parquet
equity_fundamental_metric_coverage.parquet
equity_fundamental_symbol_coverage.parquet
equity_fundamental_data_diagnostics_report.md
equity_fundamental_data_diagnostics.metadata.json
```

## Success Criteria

```text
We know which fundamental data is reliable enough to use.
We know which metrics have good coverage.
We know which symbols and dates have missing or stale data.
We know whether the data can be safely joined to historical signal dates.
```

---

# Phase 2: Fundamental Feature v0

## Goal

Create a first set of reusable, interpretable fundamental features.

## Feature Families

### Valuation

```text
P/E
forward P/E, if available
EV/EBITDA
P/S
P/B
FCF yield
earnings yield
sales yield
```

### Growth

```text
revenue growth
EPS growth
EBITDA growth
free cash flow growth
gross profit growth
operating income growth
```

### Quality

```text
gross margin
operating margin
net margin
ROE
ROA
ROIC
free cash flow margin
asset turnover
```

### Balance Sheet

```text
debt/equity
net debt/EBITDA
interest coverage
cash ratio
current ratio
debt/assets
```

### Stability

```text
revenue volatility
margin volatility
earnings consistency
free cash flow consistency
drawdown in fundamentals
frequency of negative earnings
frequency of negative free cash flow
```

## Design Principles

```text
Use point-in-time safe joins where possible.
Avoid lookahead bias.
Prefer sector-relative percentile ranks.
Preserve raw values and transformed values.
Clearly distinguish annual vs quarterly data.
Clearly distinguish trailing, current, and forward-looking metrics.
Track missingness and staleness.
```

## Suggested Output Directory

Research-first option:

```text
data/research/fundamental_features_v0/
```

Potential later durable feature output:

```text
data/features/equity_fundamental_features_v0.parquet
```

## Suggested Outputs

```text
equity_fundamental_features_v0.parquet
equity_fundamental_features_v0_summary.parquet
equity_fundamental_features_v0_report.md
equity_fundamental_features_v0.metadata.json
```

## Success Criteria

```text
Each symbol/date has a usable set of fundamental features when data exists.
Raw values and sector-relative ranks are available.
Missing and stale values are explicitly flagged.
Feature definitions are documented.
```

---

# Phase 3: Fundamental Bucket Diagnostics

## Goal

Test whether simple fundamental buckets have useful forward-return behavior.

## Example Buckets

### Valuation Buckets

```text
cheap
middle
expensive
extreme_expensive
insufficient_valuation_data
```

### Quality Buckets

```text
low_quality
average_quality
high_quality
exceptional_quality
insufficient_quality_data
```

### Growth Buckets

```text
contracting
stable
growing
high_growth
insufficient_growth_data
```

### Balance Sheet Buckets

```text
levered
normal
strong_balance_sheet
distressed_or_high_risk
insufficient_balance_sheet_data
```

## Useful Combinations

```text
high_quality + reasonable_valuation
high_growth + expensive
cheap + low_quality
cheap + improving_quality
strong_growth + improving_margin
high_debt + deteriorating_margin
high_quality + high_price_strength
weak_fundamentals + high_price_strength
```

## Labels to Use

Reuse existing forward labels where appropriate:

```text
forward_63d_sector_relative_return
forward_126d_sector_relative_return
forward_252d_sector_relative_return
forward_*_top_30pct_sector_flag
forward_*_bottom_30pct_sector_flag
```

## Suggested Output Directory

```text
data/research/fundamental_bucket_diagnostics/
```

## Suggested Outputs

```text
equity_fundamental_bucket_summary.parquet
equity_fundamental_bucket_years.parquet
equity_fundamental_bucket_stability.parquet
equity_fundamental_bucket_diagnostics_report.md
equity_fundamental_bucket_diagnostics.metadata.json
```

## Success Criteria

```text
We know which fundamental buckets have useful historical evidence.
We know which buckets are defensive, return-seeking, trap-like, or neutral.
We know whether effects are stable by year and regime.
We avoid over-interpreting sparse buckets.
```

---

# Phase 4: Fundamental Scorecard v0

## Goal

Create an exploratory fundamental scorecard.

This should remain research-only.

## Possible Score Components

```text
quality_score_v0
growth_score_v0
valuation_score_v0
balance_sheet_score_v0
fundamental_risk_score_v0
fundamental_composite_score_v0
```

## Example Interpretation Labels

```text
high_quality_growth
quality_at_reasonable_price
expensive_growth
cheap_but_weak
cheap_and_improving
levered_risk
deteriorating_fundamentals
insufficient_data
```

## Suggested Output Directory

```text
data/research/fundamental_scorecard_v0/
```

## Suggested Outputs

```text
equity_fundamental_scorecard_v0.parquet
equity_fundamental_scorecard_v0_current.parquet
equity_fundamental_scorecard_v0_summary.parquet
equity_fundamental_scorecard_v0_report.md
equity_fundamental_scorecard_v0.metadata.json
```

## Success Criteria

```text
The scorecard is interpretable.
The scorecard captures meaningful fundamental states.
The scorecard does not rely on fragile overfit weights.
The scorecard can be joined to price-strength scorecard outputs by symbol/date.
```

---

# Phase 5: Price Strength + Fundamentals Cross-Diagnostics

## Goal

Cross the price-strength scorecard v1 with fundamental scorecard v0.

## Main Research Question

> Which combinations of price strength and fundamentals are most attractive?

## Example Combinations

```text
high_conviction_price_strength + high_quality_growth
high_conviction_price_strength + quality_at_reasonable_price
high_conviction_price_strength + expensive_growth
high_conviction_price_strength + weak_fundamentals
momentum_resilience + high_quality
high_volatility_trap + weak_fundamentals
high_volatility_trap + improving_fundamentals
neutral_price_strength + high_quality_value
```

## Possible Combined Labels

```text
confirmed_strength
quality_breakout
speculative_strength
price_strength_without_fundamental_support
possible_turnaround
defensive_quality
avoid_trap
fundamental_value_watchlist
```

## Suggested Output Directory

```text
data/research/price_strength_fundamental_cross_diagnostics/
```

## Suggested Outputs

```text
equity_price_strength_fundamental_cross_summary.parquet
equity_price_strength_fundamental_cross_years.parquet
equity_price_strength_fundamental_cross_stability.parquet
equity_price_strength_fundamental_cross_report.md
equity_price_strength_fundamental_cross.metadata.json
```

## Success Criteria

```text
We know whether fundamentals improve price-strength candidate selection.
We know whether weak fundamentals identify price-strength traps.
We know which combined labels are useful and which are noisy.
```

---

# Phase 6: Combined Research Candidate v0

## Goal

Create a research-only combined candidate output that joins:

```text
price_strength_scorecard_v1
fundamental_scorecard_v0
combined_research_labels
```

## Example Fields

```text
symbol
date
sector

price_strength_bucket_v1
price_strength_score_v1
price_strength_risk_label_v1

fundamental_bucket_v0
fundamental_score_v0
fundamental_risk_label_v0

combined_research_label_v0
combined_research_score_v0
bullish_evidence
risk_evidence
candidate_priority
```

## Suggested Output Directory

```text
data/research/combined_equity_research_candidates_v0/
```

## Suggested Outputs

```text
equity_combined_research_candidates_v0.parquet
equity_combined_research_candidates_v0_current.parquet
equity_combined_research_candidates_v0_summary.parquet
equity_combined_research_candidates_v0_report.md
equity_combined_research_candidates_v0.metadata.json
```

## Success Criteria

```text
Each current symbol has a combined research interpretation when data exists.
The output explains both bullish evidence and risk evidence.
The output remains research-only.
The output can later be consumed by other Finbot components.
```

---

# Recommended Implementation Order

```text
1. Fundamental data inventory and coverage diagnostics.
2. Fundamental features v0.
3. Fundamental bucket diagnostics.
4. Fundamental scorecard v0.
5. Price strength + fundamentals cross-diagnostics.
6. Combined research candidate v0.
```

---

# Immediate Next Step

Start with:

```text
fundamental data inventory and coverage diagnostics
```

Do **not** start with a composite model.

First understand:

```text
data quality
coverage
history depth
point-in-time safety
missingness
symbol/date alignment
```

---

# Definition of Done for the Fundamentals Research Phase

The fundamentals research phase is successful when Finbot can answer:

```text
What do the fundamentals say about this stock?
Are the fundamentals supportive of price strength?
Is the stock high quality, cheap, expensive, improving, deteriorating, or risky?
Is the price-strength signal confirmed or contradicted by fundamentals?
```

The final output should not merely rank stocks. It should explain the evidence:

```text
This stock has strong price action, improving revenue growth, high margins, and reasonable sector-relative valuation.

This stock has strong price action, but valuation is extreme and free cash flow quality is weak.

This stock is cheap, but fundamentals are deteriorating and price strength is absent.
```

---

# Guiding Principle

Finbot should not just say:

```text
This stock ranks highly.
```

It should say:

```text
This stock ranks highly because price strength is strong, fundamentals are supportive, valuation is reasonable, and the main risks are sector concentration, drawdown sensitivity, or weak balance-sheet quality.
```
