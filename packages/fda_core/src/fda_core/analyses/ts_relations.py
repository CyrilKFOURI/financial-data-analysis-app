"""Two-series dynamics, null simulation and derived series (TS-018 pairs, TS-020 lead–lag,
TS-022 placebo, TS-024 transformations studio)."""

import math
import warnings
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, coint, grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from fda_core.analyses.price import PRICE_SUBJECTS, _annualisation, _field, _prices, _windowed
from fda_core.analyses.stats import stationary_bootstrap
from fda_core.analyses.ts_common import (
    SeriesParams,
    clean,
    iso,
    make_test,
    require,
    sample_info,
    select_series,
)
from fda_core.analyses.ts_memory import hurst_rs
from fda_core.analyses.ts_models import dynamic_regression, half_life, ou_fit
from fda_core.registry import AnalysisInputs, AnalysisOutput, InsufficientDataError, analysis


def _pair(inputs: AnalysisInputs) -> pd.DataFrame:
    """Aligned log prices of the subject (a) and the second asset (b) over the range."""
    other = inputs.data.get("other_prices")
    if not isinstance(other, pd.DataFrame) or other.empty:
        raise InsufficientDataError("Choose a second asset with stored prices (param `other`)")
    field = _field(inputs)
    a = _windowed(inputs, _prices(inputs))[field]
    both = pd.concat([np.log(a), np.log(other[field])], axis=1, join="inner").dropna()
    both.columns = ["a", "b"]
    require(both["a"], 60, "Pair analysis")
    return both


# ── TS-018 pair dynamics ─────────────────────────────────────────────────────


class PairParams(SeriesParams):
    zscore_window: int = Field(63, ge=20, le=252)
    kalman_delta: float = Field(1e-4, gt=0, le=0.1)


@analysis(
    id="TS-018",
    version="1.0.0",
    title="Pair dynamics",
    description=(
        "Engle–Granger (both directions) and Johansen trace cointegration tests on log prices; "
        "hedge ratios by OLS, total least squares and a Kalman filter; spread z-score over a "
        "rolling window and OU half-life; return correlation shown against cointegration "
        "(correlated ≠ cointegrated)."
    ),
    applies_to=PRICE_SUBJECTS,
    params=PairParams,
    result_kind="test",
    requires={"prices", "other_prices"},
    min_obs=60,
    budget_ms=8_000,
)
def pair_dynamics(inputs: AnalysisInputs, params: PairParams) -> AnalysisOutput:
    both = _pair(inputs)
    a, b = both["a"].to_numpy(), both["b"].to_numpy()
    tests = []
    for name, y, x in (("engle_granger_a_on_b", a, b), ("engle_granger_b_on_a", b, a)):
        stat, p, crit = coint(y, x)
        tests.append(
            make_test(
                name,
                "no cointegration",
                float(stat),
                float(p),
                {"1%": float(crit[0]), "5%": float(crit[1]), "10%": float(crit[2])},
                reading_reject="cointegrated",
                reading_keep="not cointegrated",
            )
        )
    jo = coint_johansen(both.to_numpy(), det_order=0, k_ar_diff=1)
    for rank in range(2):
        tests.append(
            make_test(
                f"johansen_trace_r{rank}",
                f"cointegration rank ≤ {rank}",
                float(jo.lr1[rank]),
                None,
                {
                    "90%": float(jo.cvt[rank, 0]),
                    "95%": float(jo.cvt[rank, 1]),
                    "99%": float(jo.cvt[rank, 2]),
                },
                reject=bool(jo.lr1[rank] > jo.cvt[rank, 1]),
                reading_reject=f"rank > {rank}",
                reading_keep=f"rank ≤ {rank}",
            )
        )
    ols_beta, ols_alpha = np.polyfit(b, a, 1)
    centred = np.column_stack([b - b.mean(), a - a.mean()])
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    tls_beta = float(-vt[-1, 0] / vt[-1, 1])
    kalman = dynamic_regression(a, b, params.kalman_delta)
    spread = pd.Series(a - ols_beta * b, index=both.index)
    z = (spread - spread.rolling(params.zscore_window).mean()) / spread.rolling(
        params.zscore_window
    ).std()
    fit = ou_fit(spread.to_numpy())
    returns = both.diff().dropna()
    return AnalysisOutput(
        "test",
        {
            "tests": tests,
            "hedge_ratio": {
                "ols": float(ols_beta),
                "tls": tls_beta,
                "kalman_last": float(kalman["beta"][-1]),
            },
            "intercept": float(ols_alpha),
            "spread": {
                "dates": iso(both.index),
                "values": clean(spread),
                "zscore": clean(z),
                "kalman_beta": clean(kalman["beta"]),
            },
            "spread_half_life": half_life(fit["phi"]),
            "spread_phi": fit["phi"],
            "spread_adf_p": float(adfuller(spread.to_numpy(), autolag="AIC")[1]),
            "return_correlation": float(returns.corr().iloc[0, 1]),
            "other": inputs.conventions.get("other_symbol"),
            "n": len(both),
        },
        ("Descriptive: trading the spread (entry/exit rules, costs) is a V5 strategy.",),
    )


# ── TS-020 lead–lag & causality ──────────────────────────────────────────────


class LeadLagParams(SeriesParams):
    max_lag: int = Field(10, ge=1, le=20)


@analysis(
    id="TS-020",
    version="1.0.0",
    title="Lead–lag & causality",
    description=(
        "Cross-correlation of daily returns at ±20 lags (positive lag: the second asset leads), "
        "and Granger causality in both directions with the VAR lag chosen by AIC (≤ max lag)."
    ),
    applies_to=PRICE_SUBJECTS,
    params=LeadLagParams,
    result_kind="test",
    requires={"prices", "other_prices"},
    min_obs=60,
    budget_ms=8_000,
)
def lead_lag(inputs: AnalysisInputs, params: LeadLagParams) -> AnalysisOutput:
    returns = _pair(inputs).diff().dropna()
    a, b = returns["a"], returns["b"]
    ccf = {str(k): float(a.corr(b.shift(k))) for k in range(-20, 21)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lag = max(int(VAR(returns.to_numpy()).select_order(params.max_lag).aic), 1)
        tests = []
        for name, data in (
            ("b_granger_causes_a", returns[["a", "b"]]),
            ("a_granger_causes_b", returns[["b", "a"]]),
        ):
            result = grangercausalitytests(data.to_numpy(), maxlag=[lag])
            f_stat, p_value, *_ = result[lag][0]["ssr_ftest"]
            tests.append(
                make_test(
                    name,
                    "no Granger causality",
                    float(f_stat),
                    float(p_value),
                    reading_reject="past values help predict",
                    reading_keep="no predictive content",
                )
            )
    return AnalysisOutput(
        "test",
        {
            "ccf": ccf,
            "band": 1.96 / math.sqrt(len(returns)),
            "var_lag": lag,
            "tests": tests,
            "other": inputs.conventions.get("other_symbol"),
        },
    )


# ── TS-022 placebo & null simulation ─────────────────────────────────────────

Statistic = Literal[
    "lag1_autocorr", "variance_ratio_5", "hurst", "sharpe", "max_drawdown", "adf_price"
]


def _statistic(name: str, factor: int) -> Callable[[np.ndarray], float]:
    def lag1(r: np.ndarray) -> float:
        return float(np.corrcoef(r[:-1], r[1:])[0, 1])

    def vr5(r: np.ndarray) -> float:
        five = np.convolve(r, np.ones(5), "valid")
        return float(np.var(five, ddof=1) / (5 * np.var(r, ddof=1)))

    def sharpe(r: np.ndarray) -> float:
        return float(r.mean() / r.std(ddof=1) * math.sqrt(factor))

    def mdd(r: np.ndarray) -> float:
        wealth = np.cumprod(1 + r)
        return float((wealth / np.maximum.accumulate(wealth) - 1).min())

    def adf_price(r: np.ndarray) -> float:
        return float(adfuller(np.cumsum(np.log1p(r)), maxlag=1, autolag=None)[0])

    def hurst(r: np.ndarray) -> float:
        return float(hurst_rs(r)["anis_lloyd"])

    return {
        "lag1_autocorr": lag1,
        "variance_ratio_5": vr5,
        "hurst": hurst,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "adf_price": adf_price,
    }[name]


class PlaceboParams(SeriesParams):
    statistic: Statistic = "lag1_autocorr"
    n_sims: int = Field(500, ge=100, le=5_000)
    seed: int = Field(23, ge=0, le=2**32 - 1)


@analysis(
    id="TS-022",
    version="1.0.0",
    title="Placebo & null simulation",
    description=(
        "Observed statistic against its distribution under three nulls: i.i.d. bootstrap of "
        "returns (no dependence), stationary block bootstrap (dependence kept, mean "
        "recentred to 0), and GBM with matched μ and σ; two-sided empirical p-values."
    ),
    applies_to=PRICE_SUBJECTS,
    params=PlaceboParams,
    result_kind="test",
    requires={"prices"},
    min_obs=100,
    budget_ms=20_000,
)
def placebo(inputs: AnalysisInputs, params: PlaceboParams) -> AnalysisOutput:
    factor = _annualisation(inputs)
    returns = _windowed(inputs, _prices(inputs))[_field(inputs)].pct_change().dropna()
    r = require(returns, 100, "Placebo simulation")
    stat = _statistic(params.statistic, factor)
    observed = stat(r)
    rng = np.random.default_rng(params.seed)
    nulls: dict[str, np.ndarray] = {
        "iid_bootstrap": np.array([stat(rng.choice(r, len(r))) for _ in range(params.n_sims)]),
        "block_bootstrap": stationary_bootstrap(
            r - r.mean() if params.statistic == "sharpe" else r,
            stat,
            params.n_sims,
            max(len(r) ** (1 / 3), 1.0),
            params.seed + 1,
        ),
        "gbm": np.array(
            [
                stat(np.expm1(rng.normal(np.log1p(r).mean(), np.log1p(r).std(ddof=1), len(r))))
                for _ in range(params.n_sims)
            ]
        ),
    }
    tests = []
    for name, values in nulls.items():
        values = values[np.isfinite(values)]
        centre = float(np.median(values))
        p = float(np.mean(np.abs(values - centre) >= abs(observed - centre)))
        tests.append(
            make_test(
                name,
                "statistic is typical of the null",
                observed,
                max(p, 1 / (len(values) + 1)),
                reading_reject="unusual under this null",
                reading_keep="typical of the null",
                extra={
                    "null_quantiles": np.quantile(values, [0.025, 0.5, 0.975]).tolist(),
                    "histogram": np.histogram(values, bins=30)[0].tolist(),
                    "edges": np.histogram(values, bins=30)[1].tolist(),
                },
            )
        )
    return AnalysisOutput(
        "test",
        {
            "statistic": params.statistic,
            "observed": observed,
            "tests": tests,
            "n_sims": params.n_sims,
        },
        ("A significant result under the block bootstrap null is the hardest to explain away.",),
    )


# ── TS-024 transformations studio ────────────────────────────────────────────


@analysis(
    id="TS-024",
    version="1.0.0",
    title="Series transformations studio",
    description=(
        "Derived series used by every TS- analysis: price, log price, (log/abs/squared) returns, "
        "volume, z-score, linearly detrended log price, FFD frac-diff at d, A/B log ratio and "
        "OLS spread with a second asset; returned with summary statistics."
    ),
    applies_to=PRICE_SUBJECTS,
    params=SeriesParams,
    result_kind="series",
    requires={"prices", "other_prices"},
    min_obs=2,
)
def transformations(inputs: AnalysisInputs, params: SeriesParams) -> AnalysisOutput:
    series = select_series(inputs, params)
    x = require(series, 2, "A derived series")
    summary: dict[str, Any] = {
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)),
        "min": float(x.min()),
        "max": float(x.max()),
    }
    return AnalysisOutput(
        "series",
        {
            "dates": iso(series.index),
            "values": clean(series),
            "summary": summary,
            "sample": sample_info(series, params),
        },
    )
