from __future__ import annotations

JOIN_COLUMNS = ("symbol", "date")

PRIMARY_REGRESSION_LABEL = "forward_63d_sector_relative_return"
PRIMARY_CLASSIFICATION_LABEL = "forward_63d_top_30pct_sector_flag"

PRICE_FEATURE_COLUMNS = (
    "return_21d",
    "return_63d",
    "return_126d",
    "return_252d",
    "volatility_21d",
    "volatility_63d",
    "average_dollar_volume_21d",
    "average_dollar_volume_63d",
    "drawdown_from_52w_high",
    "distance_from_50dma",
    "distance_from_200dma",
)

RELATIVE_FEATURE_COLUMNS = (
    "return_21d_market_pct_rank",
    "return_63d_market_pct_rank",
    "return_126d_market_pct_rank",
    "return_252d_market_pct_rank",
    "return_21d_sector_pct_rank",
    "return_63d_sector_pct_rank",
    "return_126d_sector_pct_rank",
    "return_252d_sector_pct_rank",
    "volatility_63d_market_pct_rank",
    "volatility_63d_sector_pct_rank",
    "drawdown_from_52w_high_market_pct_rank",
    "drawdown_from_52w_high_sector_pct_rank",
)

SELECTED_FEATURE_COLUMNS = (*PRICE_FEATURE_COLUMNS, *RELATIVE_FEATURE_COLUMNS)
