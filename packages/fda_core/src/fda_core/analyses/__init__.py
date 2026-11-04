"""Importing this package registers every analysis in `fda_core.registry.REGISTRY`."""

import warnings

from fda_core.analyses import (
    alpha_ml,
    backtesting,
    carver_engine,
    comparison,
    derivatives,
    diagnostics,
    filings_reader,
    invest,
    library,
    markets,
    news_sentiment,
    optimiser,
    option_strategy,
    paper_trading,
    peers,
    plans,
    portfolio,
    price,
    quant,
    ratios,
    risk,
    screener,
    statements,
    stats,
    technical,
    ts_breaks,
    ts_info,
    ts_memory,
    ts_models,
    ts_relations,
    valuation,
)

# statsmodels 0.15 announces result objects for adfuller/kpss/acorr_lm (0.16): we unpack the tuple
# layout on purpose until then; the notice would otherwise flood worker logs.
warnings.filterwarnings(
    "ignore", message=r"(adfuller|kpss|acorr_lm) currently returns", category=FutureWarning
)

__all__ = [
    "alpha_ml",
    "backtesting",
    "carver_engine",
    "comparison",
    "derivatives",
    "diagnostics",
    "filings_reader",
    "invest",
    "library",
    "markets",
    "news_sentiment",
    "optimiser",
    "option_strategy",
    "paper_trading",
    "peers",
    "plans",
    "portfolio",
    "price",
    "quant",
    "ratios",
    "risk",
    "screener",
    "statements",
    "stats",
    "technical",
    "ts_breaks",
    "ts_info",
    "ts_memory",
    "ts_models",
    "ts_relations",
    "valuation",
]
