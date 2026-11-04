"""Engine diagnostics: analyses that exercise the pipeline without market data."""

import math

import numpy as np
from pydantic import BaseModel, Field

from fda_core.registry import AnalysisInputs, AnalysisOutput, analysis

TRADING_DAYS = 252


class SyntheticStatisticsParams(BaseModel):
    n: int = Field(252, ge=30, le=100_000, description="Number of daily returns to simulate")
    seed: int = Field(7, ge=0, le=2**32 - 1, description="Random seed (same seed → same sample)")
    mu: float = Field(0.08, ge=-1.0, le=1.0, description="Annualised drift")
    sigma: float = Field(0.20, gt=0.0, le=3.0, description="Annualised volatility")


@analysis(
    id="DEV-001",
    version="1.0.0",
    title="Synthetic statistics",
    description=(
        "Simulates n daily Gaussian returns with the given annualised drift and volatility, "
        "then reports annualised mean, volatility and Sharpe ratio (rf = 0). Used to check the "
        "engine, cache and streaming pipeline; not a market analysis."
    ),
    applies_to={"asset", "portfolio"},
    params=SyntheticStatisticsParams,
    result_kind="table",
    diagnostic=True,
)
def synthetic_statistics(
    inputs: AnalysisInputs, params: SyntheticStatisticsParams
) -> AnalysisOutput:
    rng = np.random.default_rng(params.seed)
    returns = rng.normal(params.mu / TRADING_DAYS, params.sigma / math.sqrt(TRADING_DAYS), params.n)
    mean_annualised = float(returns.mean() * TRADING_DAYS)
    vol_annualised = float(returns.std(ddof=1) * math.sqrt(TRADING_DAYS))
    return AnalysisOutput(
        kind="table",
        payload={
            "n": params.n,
            "mean_annualised": mean_annualised,
            "vol_annualised": vol_annualised,
            "sharpe": mean_annualised / vol_annualised,
            "min_daily": float(returns.min()),
            "max_daily": float(returns.max()),
        },
        warnings=("Synthetic data — engine diagnostic only.",),
    )
