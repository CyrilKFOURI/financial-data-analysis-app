"""Risk analysis (ANALYTICS_CATALOG R-001…R-023; module M07).

Returns are simple returns of the performance price over the context range. The risk-free
rate, MAR and confidence levels are explicit annual parameters (never globals); losses are
reported as positive numbers (VaR 2 % = "lose 2 % or more with probability 1 − α").
Benchmark-relative analyses read `data["other_prices"]` (param `other`, default: the type's
benchmark).
"""

import math
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from typing import Any, Literal

import numpy as np
import pandas as pd
from arch import arch_model
from pydantic import BaseModel, Field
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from fda_core.analyses.price import (
    PRICE_SUBJECTS,
    _annualisation,
    _field,
    _prices,
    _windowed,
    cagr,
)
from fda_core.analyses.stats import stationary_bootstrap
from fda_core.registry import AnalysisInputs, AnalysisOutput, InsufficientDataError, analysis

# ── helpers ──────────────────────────────────────────────────────────────────


class RiskParams(BaseModel):
    rf: float = Field(0.0, ge=-0.05, le=0.5, description="Annual risk-free rate")
    other: str | None = Field(None, max_length=20, pattern=r"^[\w.^=-]+$", description="Benchmark")


def _series(inputs: AnalysisInputs) -> tuple[pd.Series, pd.Series, int]:
    """(prices, returns, annualisation) over the context range."""
    factor = _annualisation(inputs)
    prices = _windowed(inputs, _prices(inputs))[_field(inputs)].astype(float)
    returns = prices.pct_change().dropna()
    if len(returns) < 20:
        raise InsufficientDataError(f"Risk analytics need ≥ 20 returns, got {len(returns)}")
    return prices, returns, factor


def _iso(day: Any) -> str | None:
    return None if day is None or pd.isna(day) else pd.Timestamp(day).date().isoformat()


def _f(value: Any) -> float | None:
    return None if value is None or not np.isfinite(value) else float(value)


def drawdown(wealth: pd.Series) -> pd.Series:
    """R-001: DD_t = V_t / max_{s≤t} V_s − 1."""
    return wealth / wealth.cummax() - 1


def episodes(dd: pd.Series) -> list[dict[str, Any]]:
    """Peak → trough → recovery episodes (recovery = first return to the previous peak)."""
    out: list[dict[str, Any]] = []
    in_dd, peak_i = False, 0
    values = dd.to_numpy()
    index = dd.index
    for i, v in enumerate(values):
        if v < 0 and not in_dd:
            in_dd, peak_i = True, i - 1 if i > 0 else 0
        elif v == 0 and in_dd:
            segment = values[peak_i : i + 1]
            trough_i = peak_i + int(np.argmin(segment))
            out.append(_episode(index, values, peak_i, trough_i, i))
            in_dd = False
    if in_dd:
        trough_i = peak_i + int(np.argmin(values[peak_i:]))
        out.append(_episode(index, values, peak_i, trough_i, None))
    return out


def _episode(
    index: pd.Index, values: np.ndarray, peak: int, trough: int, end: int | None
) -> dict[str, Any]:
    return {
        "peak": _iso(index[peak]),
        "trough": _iso(index[trough]),
        "recovery": _iso(index[end]) if end is not None else None,
        "depth": float(values[trough]),
        "decline_sessions": trough - peak,
        "recovery_sessions": (end - trough) if end is not None else None,
        "length_sessions": ((end if end is not None else len(values) - 1) - peak),
        "recovered": end is not None,
    }


@dataclass(frozen=True, slots=True)
class _Moments:
    """Sample statistics and the Student-t fit, estimated once per return sample."""

    mu: float
    sigma: float
    skew: float
    kurt: float
    nu: float
    loc: float
    scale: float

    @classmethod
    def of(cls, r: np.ndarray) -> "_Moments":
        nu, loc, scale = stats.t.fit(r)
        return cls(
            float(r.mean()),
            float(r.std(ddof=1)),
            float(stats.skew(r)),
            float(stats.kurtosis(r)),
            max(float(nu), 2.01),
            float(loc),
            float(scale),
        )


def _cf_z(z: np.ndarray | float, s: float, k: float) -> np.ndarray | float:
    """Cornish–Fisher expansion of the Normal quantile with skewness s, excess kurtosis k."""
    return z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36


def _var_quantile(r: np.ndarray, alpha: float, method: str, m: "_Moments | None" = None) -> float:
    """Return quantile at 1 − α (a negative number for a loss) under each model."""
    p = 1 - alpha
    if method == "historical":
        return float(np.quantile(r, p))
    m = m or _Moments.of(r)
    z = stats.norm.ppf(p)
    if method == "gaussian":
        return float(m.mu + m.sigma * z)
    if method == "cornish_fisher":
        return float(m.mu + m.sigma * _cf_z(z, m.skew, m.kurt))
    return float(stats.t.ppf(p, m.nu, m.loc, m.scale))


def _es(r: np.ndarray, alpha: float, method: str, m: "_Moments | None" = None) -> float:
    """Expected shortfall as a (negative) mean return beyond the VaR quantile."""
    p = 1 - alpha
    if method == "historical":
        q = np.quantile(r, p)
        return float(r[r <= q].mean())
    m = m or _Moments.of(r)
    if method == "gaussian":
        return float(m.mu - m.sigma * stats.norm.pdf(stats.norm.ppf(p)) / p)
    if method == "student_t":
        t_q = stats.t.ppf(p, m.nu)
        return float(m.loc - m.scale * stats.t.pdf(t_q, m.nu) / p * (m.nu + t_q**2) / (m.nu - 1))
    grid = stats.norm.ppf(np.linspace(p / 200, p, 200))  # Cornish–Fisher: mean of tail quantiles
    return float(np.mean(m.mu + m.sigma * _cf_z(grid, m.skew, m.kurt)))


# ── R-001 / R-002 / R-003 / R-004 drawdowns ──────────────────────────────────


class DrawdownParams(RiskParams):
    top: int = Field(10, ge=1, le=50)


@analysis(
    id="R-001",
    version="1.0.0",
    title="Drawdown series",
    description="DD_t = V_t / max_{s≤t} V_s − 1 on the performance price over the range.",
    applies_to=PRICE_SUBJECTS,
    params=RiskParams,
    result_kind="series",
    requires={"prices"},
    min_obs=20,
)
def drawdown_series(inputs: AnalysisInputs, params: RiskParams) -> AnalysisOutput:
    prices, _, _ = _series(inputs)
    dd = drawdown(prices)
    return AnalysisOutput(
        "series",
        {
            "dates": [_iso(d) for d in dd.index],
            "values": [float(v) for v in dd],
            "max_drawdown": float(dd.min()),
            "current": float(dd.iloc[-1]),
        },
    )


@analysis(
    id="R-002",
    version="1.0.0",
    title="Drawdown episodes",
    description=(
        "Maximum drawdown and the top-N episodes by depth: peak, trough, recovery, depth, "
        "decline, recovery and total length in sessions (unrecovered episodes flagged)."
    ),
    applies_to=PRICE_SUBJECTS,
    params=DrawdownParams,
    result_kind="table",
    requires={"prices"},
    min_obs=20,
)
def drawdown_episodes(inputs: AnalysisInputs, params: DrawdownParams) -> AnalysisOutput:
    prices, _, _ = _series(inputs)
    found = sorted(episodes(drawdown(prices)), key=lambda e: e["depth"])
    return AnalysisOutput(
        "table",
        {
            "max_drawdown": found[0]["depth"] if found else 0.0,
            "count": len(found),
            "episodes": found[: params.top],
        },
    )


@analysis(
    id="R-003",
    version="1.0.0",
    title="Time under water",
    description="Longest drawdown duration (peak to recovery or to today) and current duration.",
    applies_to=PRICE_SUBJECTS,
    params=RiskParams,
    result_kind="table",
    requires={"prices"},
    min_obs=20,
)
def time_under_water(inputs: AnalysisInputs, params: RiskParams) -> AnalysisOutput:
    prices, _, _ = _series(inputs)
    dd = drawdown(prices)
    found = episodes(dd)
    longest = max(found, key=lambda e: e["length_sessions"], default=None)
    current = found[-1] if found and not found[-1]["recovered"] else None
    share = float((dd < 0).mean())
    return AnalysisOutput(
        "table",
        {"longest": longest, "current": current, "share_of_time_under_water": share},
    )


@analysis(
    id="R-004",
    version="1.0.0",
    title="Ulcer & Pain",
    description=(
        "Ulcer index √(mean DD²), Pain index mean |DD|, Martin ratio = (CAGR − rf) / Ulcer and "
        "Pain ratio = (CAGR − rf) / Pain."
    ),
    applies_to=PRICE_SUBJECTS,
    params=RiskParams,
    result_kind="table",
    requires={"prices"},
    min_obs=20,
)
def ulcer_pain(inputs: AnalysisInputs, params: RiskParams) -> AnalysisOutput:
    prices, _, _ = _series(inputs)
    dd = drawdown(prices).to_numpy()
    ulcer = float(np.sqrt(np.mean(dd**2)))
    pain = float(np.mean(np.abs(dd)))
    days = (prices.index[-1] - prices.index[0]).days
    growth = cagr(float(prices.iloc[0]), float(prices.iloc[-1]), max(days, 1)) - params.rf
    return AnalysisOutput(
        "table",
        {
            "ulcer_index": ulcer,
            "pain_index": pain,
            "martin_ratio": growth / ulcer if ulcer else None,
            "pain_ratio": growth / pain if pain else None,
        },
    )


# ── R-005 … R-008 ratios ─────────────────────────────────────────────────────


def sharpe_ratio(r: np.ndarray, rf: float, factor: int) -> float:
    excess = r - rf / factor
    return float(excess.mean() / excess.std(ddof=1) * math.sqrt(factor))


def sortino_ratio(r: np.ndarray, mar: float, factor: int) -> float | None:
    target = mar / factor
    downside = math.sqrt(float(np.mean(np.minimum(r - target, 0) ** 2)))
    return (
        float((r.mean() - target) * factor / (downside * math.sqrt(factor))) if downside else None
    )


class SortinoParams(RiskParams):
    mar: float = Field(0.0, ge=-0.5, le=0.5, description="Annual minimum acceptable return")


@analysis(
    id="R-005",
    version="1.0.0",
    title="Sharpe",
    description="mean(r − rf) / std(r − rf) · √A with the annual rf converted per period.",
    applies_to=PRICE_SUBJECTS,
    params=RiskParams,
    result_kind="value",
    requires={"prices"},
    min_obs=20,
)
def sharpe(inputs: AnalysisInputs, params: RiskParams) -> AnalysisOutput:
    _, returns, factor = _series(inputs)
    r = returns.to_numpy()
    return AnalysisOutput(
        "value",
        {
            "value": sharpe_ratio(r, params.rf, factor),
            "rf": params.rf,
            "n": len(r),
            "annualisation": factor,
        },
    )


@analysis(
    id="R-006",
    version="1.0.0",
    title="Sortino",
    description=(
        "(mean r − MAR)·A / (DD_MAR·√A) with the downside deviation √(mean(min(r − MAR, 0)²)) "
        "taken over all observations."
    ),
    applies_to=PRICE_SUBJECTS,
    params=SortinoParams,
    result_kind="value",
    requires={"prices"},
    min_obs=20,
)
def sortino(inputs: AnalysisInputs, params: SortinoParams) -> AnalysisOutput:
    _, returns, factor = _series(inputs)
    return AnalysisOutput(
        "value", {"value": sortino_ratio(returns.to_numpy(), params.mar, factor), "mar": params.mar}
    )


@analysis(
    id="R-007",
    version="1.0.0",
    title="Calmar / Sterling / Burke",
    description=(
        "Calmar = CAGR / |MDD|; Sterling = CAGR / (|mean of yearly MDDs| + 10 %); Burke = "
        "(CAGR − rf) / √Σ DD_i² over drawdown episode depths."
    ),
    applies_to=PRICE_SUBJECTS,
    params=RiskParams,
    result_kind="table",
    requires={"prices"},
    min_obs=20,
)
def calmar_family(inputs: AnalysisInputs, params: RiskParams) -> AnalysisOutput:
    prices, _, _ = _series(inputs)
    days = max((prices.index[-1] - prices.index[0]).days, 1)
    growth = cagr(float(prices.iloc[0]), float(prices.iloc[-1]), days)
    dd = drawdown(prices)
    mdd = float(dd.min())
    yearly = [
        float(drawdown(chunk).min())
        for _, chunk in prices.groupby(prices.index.year)
        if len(chunk) > 1
    ]
    depths = np.array([e["depth"] for e in episodes(dd)])
    burke_denominator = math.sqrt(float(np.sum(depths**2))) if len(depths) else 0.0
    return AnalysisOutput(
        "table",
        {
            "cagr": growth,
            "max_drawdown": mdd,
            "calmar": growth / abs(mdd) if mdd else None,
            "sterling": growth / (abs(float(np.mean(yearly))) + 0.10) if yearly else None,
            "burke": (growth - params.rf) / burke_denominator if burke_denominator else None,
        },
    )


class DistributionRatioParams(RiskParams):
    threshold: float = Field(0.0, ge=-0.1, le=0.1, description="Omega threshold per period")
    tail: float = Field(0.05, gt=0.0, le=0.25)


@analysis(
    id="R-008",
    version="1.0.0",
    title="Distribution-based ratios",
    description=(
        "Omega(θ) = Σ(r − θ)⁺ / Σ(θ − r)⁺; tail ratio |q95| / |q05|; gain-to-pain Σr / Σ|r⁻| on "
        "monthly returns; profit factor; recovery factor = total return / |MDD|; common-sense "
        "ratio = tail ratio × profit factor; Rachev = ES of gains / ES of losses at the tail level."
    ),
    applies_to=PRICE_SUBJECTS,
    params=DistributionRatioParams,
    result_kind="table",
    requires={"prices"},
    min_obs=20,
)
def distribution_ratios(inputs: AnalysisInputs, params: DistributionRatioParams) -> AnalysisOutput:
    prices, returns, _ = _series(inputs)
    r = returns.to_numpy()
    gains, losses = (
        np.maximum(r - params.threshold, 0).sum(),
        np.maximum(params.threshold - r, 0).sum(),
    )
    q95, q05 = np.quantile(r, 0.95), np.quantile(r, 0.05)
    profit_factor = r[r > 0].sum() / abs(r[r < 0].sum()) if (r < 0).any() else None
    monthly = prices.resample("ME").last().pct_change().dropna().to_numpy()
    monthly_losses = abs(monthly[monthly < 0].sum())
    total = float(prices.iloc[-1] / prices.iloc[0] - 1)
    mdd = float(drawdown(prices).min())
    lower, upper = np.quantile(r, params.tail), np.quantile(r, 1 - params.tail)
    rachev = r[r >= upper].mean() / abs(r[r <= lower].mean()) if (r <= lower).any() else None
    tail_ratio = abs(q95) / abs(q05) if q05 else None
    return AnalysisOutput(
        "table",
        {
            "omega": float(gains / losses) if losses else None,
            "tail_ratio": _f(tail_ratio),
            "gain_to_pain": float(monthly.sum() / monthly_losses) if monthly_losses else None,
            "profit_factor": _f(profit_factor),
            "recovery_factor": total / abs(mdd) if mdd else None,
            "common_sense_ratio": _f(tail_ratio * profit_factor)
            if tail_ratio and profit_factor
            else None,
            "rachev": _f(rachev),
        },
    )


# ── R-009 probabilistic Sharpe ───────────────────────────────────────────────


class PsrParams(RiskParams):
    benchmark_sharpe: float = Field(0.0, ge=-2, le=5, description="Annual SR* to beat")
    confidence: float = Field(0.95, ge=0.5, lt=1.0)


def probabilistic_sharpe(
    r: np.ndarray, sr_star: float, rf_period: float = 0.0
) -> dict[str, float | None]:
    """PSR and MinTRL on per-period Sharpe ratios (Bailey & López de Prado, AFML-14)."""
    excess = r - rf_period
    sr = float(excess.mean() / excess.std(ddof=1))
    g3 = float(stats.skew(excess))
    g4 = float(stats.kurtosis(excess, fisher=False))
    denominator = math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr**2, 1e-12))
    psr = float(stats.norm.cdf((sr - sr_star) * math.sqrt(len(r) - 1) / denominator))
    return {"sr": sr, "skew": g3, "kurtosis": g4, "psr": psr, "denominator": denominator}


@analysis(
    id="R-009",
    version="1.0.0",
    title="Probabilistic Sharpe",
    description=(
        "PSR = Φ((SR − SR*)√(T − 1) / √(1 − γ̂₃SR + (γ̂₄ − 1)/4·SR²)) on per-period Sharpe "
        "ratios (non-excess kurtosis γ̂₄), and the Minimum Track Record Length "
        "1 + (1 − γ̂₃SR + (γ̂₄ − 1)/4·SR²)·(z_c / (SR − SR*))²."
    ),
    applies_to=PRICE_SUBJECTS,
    params=PsrParams,
    result_kind="table",
    requires={"prices"},
    min_obs=30,
)
def psr(inputs: AnalysisInputs, params: PsrParams) -> AnalysisOutput:
    _, returns, factor = _series(inputs)
    r = returns.to_numpy()
    sr_star = params.benchmark_sharpe / math.sqrt(factor)
    out = probabilistic_sharpe(r, sr_star, params.rf / factor)
    sr = out["sr"] or 0.0
    z = stats.norm.ppf(params.confidence)
    den = (out["denominator"] or 1.0) ** 2
    min_trl = 1 + den * (z / (sr - sr_star)) ** 2 if sr > sr_star else None
    return AnalysisOutput(
        "table",
        {
            **out,
            "sharpe_annualised": sr * math.sqrt(factor),
            "benchmark_sharpe": params.benchmark_sharpe,
            "n": len(r),
            "min_track_record": min_trl,
            "min_track_record_years": min_trl / factor if min_trl else None,
            "confidence": params.confidence,
        },
    )


# ── R-010 / R-011 VaR & ES ───────────────────────────────────────────────────

VarMethod = Literal["historical", "gaussian", "cornish_fisher", "student_t"]


class VarParams(RiskParams):
    confidences: list[float] = Field(default=[0.95, 0.99], min_length=1, max_length=4)
    horizon: int = Field(1, ge=1, le=20)
    scaling: Literal["sqrt", "overlapping"] = "sqrt"
    notional: float = Field(10_000, gt=0, le=1e12)


def _horizon_returns(returns: pd.Series, horizon: int, scaling: str) -> tuple[np.ndarray, float]:
    """Overlapping h-day compounded returns, or daily returns with a √h scale factor."""
    if horizon == 1 or scaling == "sqrt":
        return returns.to_numpy(), math.sqrt(horizon)
    wealth = (1 + returns).cumprod()
    return (wealth / wealth.shift(horizon) - 1).dropna().to_numpy(), 1.0


def _risk_table(inputs: AnalysisInputs, params: VarParams, measure: str) -> AnalysisOutput:
    _, returns, _ = _series(inputs)
    for c in params.confidences:
        if not 0.8 <= c <= 0.999:
            raise InsufficientDataError("Confidence levels must lie in [80 %, 99.9 %]")
    r, scale = _horizon_returns(returns, params.horizon, params.scaling)
    moments = _Moments.of(r)
    rows = []
    for method in ("historical", "gaussian", "cornish_fisher", "student_t"):
        for c in params.confidences:
            q = (
                _var_quantile(r, c, method, moments)
                if measure == "var"
                else _es(r, c, method, moments)
            )
            loss = -q * scale
            rows.append(
                {
                    "method": method,
                    "confidence": c,
                    "loss": float(loss),
                    "amount": float(loss * params.notional),
                }
            )
    warnings = []
    if params.horizon > 1 and params.scaling == "sqrt":
        warnings.append("√h scaling assumes i.i.d. returns; use overlapping returns to check.")
    if min(len(r) * (1 - c) for c in params.confidences) < 5:
        warnings.append(
            "Fewer than 5 observations beyond the highest confidence: historical estimate unstable."
        )
    return AnalysisOutput(
        "table",
        {
            "measure": measure,
            "horizon": params.horizon,
            "scaling": params.scaling,
            "notional": params.notional,
            "currency": inputs.conventions.get("currency"),
            "rows": rows,
            "n": len(r),
            "histogram": np.histogram(r, bins=60)[0].tolist(),
            "edges": np.histogram(r, bins=60)[1].tolist(),
        },
        tuple(warnings),
    )


@analysis(
    id="R-010",
    version="1.0.0",
    title="VaR",
    description=(
        "Value-at-Risk as a positive loss: historical quantile, Gaussian μ + σz, Cornish–Fisher "
        "(skew/kurtosis-adjusted z) and Student-t (MLE); horizons 1–20 days by √h scaling or "
        "overlapping h-day returns; amounts per notional in the asset currency."
    ),
    applies_to=PRICE_SUBJECTS,
    params=VarParams,
    result_kind="table",
    requires={"prices"},
    min_obs=60,
)
def value_at_risk(inputs: AnalysisInputs, params: VarParams) -> AnalysisOutput:
    return _risk_table(inputs, params, "var")


@analysis(
    id="R-011",
    version="1.0.0",
    title="CVaR / ES",
    description=(
        "Expected shortfall as a positive loss: historical −E[r | r ≤ q], Gaussian "
        "σφ(z)/(1 − α) − μ, Student-t closed form and Cornish–Fisher by averaging tail quantiles."
    ),
    applies_to=PRICE_SUBJECTS,
    params=VarParams,
    result_kind="table",
    requires={"prices"},
    min_obs=60,
)
def expected_shortfall(inputs: AnalysisInputs, params: VarParams) -> AnalysisOutput:
    return _risk_table(inputs, params, "es")


# ── R-012 VaR backtesting ────────────────────────────────────────────────────


class BacktestParams(RiskParams):
    confidence: float = Field(0.99, ge=0.9, le=0.999)
    window: int = Field(250, ge=100, le=1_000)
    method: VarMethod = "historical"


def kupiec_pof(exceptions: int, n: int, p: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures LR statistic and χ²(1) p-value."""
    x = exceptions
    if n == 0:
        return float("nan"), float("nan")
    phat = x / n

    def ll(q: float) -> float:
        return (n - x) * math.log(1 - q) + (x * math.log(q) if x else 0.0)

    lr = -2 * (ll(p) - (ll(phat) if 0 < phat < 1 else 0.0))
    return lr, float(stats.chi2.sf(lr, 1))


def christoffersen(hits: np.ndarray) -> tuple[float, float]:
    """Christoffersen independence LR (first-order Markov) and χ²(1) p-value."""
    h = hits.astype(int)
    pairs = list(pairwise(h))
    n00 = sum(1 for a, b in pairs if a == 0 and b == 0)
    n01 = sum(1 for a, b in pairs if a == 0 and b == 1)
    n10 = sum(1 for a, b in pairs if a == 1 and b == 0)
    n11 = sum(1 for a, b in pairs if a == 1 and b == 1)
    pi0 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi1 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / max(len(pairs), 1)

    def term(count: int, prob: float) -> float:
        return count * math.log(prob) if count and prob > 0 else 0.0

    ll_null = term(n00 + n10, 1 - pi) + term(n01 + n11, pi)
    ll_alt = term(n00, 1 - pi0) + term(n01, pi0) + term(n10, 1 - pi1) + term(n11, pi1)
    lr = -2 * (ll_null - ll_alt)
    return lr, float(stats.chi2.sf(lr, 1))


@analysis(
    id="R-012",
    version="1.0.0",
    title="VaR backtesting",
    description=(
        "Rolling VaR from the previous `window` returns (no look-ahead) against the realised "
        "return; exceptions, Kupiec POF, Christoffersen independence and conditional coverage "
        "(LR_POF + LR_IND, χ²(2)); Basel traffic light on the last 250 days (99 %: green ≤ 4, "
        "yellow 5–9, red ≥ 10, scaled for other levels)."
    ),
    applies_to=PRICE_SUBJECTS,
    params=BacktestParams,
    result_kind="test",
    requires={"prices"},
    min_obs=150,
    budget_ms=10_000,
)
def var_backtest(inputs: AnalysisInputs, params: BacktestParams) -> AnalysisOutput:
    _, returns, _ = _series(inputs)
    r = returns.to_numpy()
    if len(r) < params.window + 50:
        raise InsufficientDataError(f"Backtesting needs ≥ {params.window + 50} returns")
    var = np.array(
        [
            -_var_quantile(r[t - params.window : t], params.confidence, params.method)
            for t in range(params.window, len(r))
        ]
    )
    realised = r[params.window :]
    hits = realised < -var
    n, x = len(hits), int(hits.sum())
    p = 1 - params.confidence
    lr_pof, p_pof = kupiec_pof(x, n, p)
    lr_ind, p_ind = christoffersen(hits)
    lr_cc = lr_pof + lr_ind
    last = hits[-250:]
    expected_250 = 250 * p
    green, red = expected_250 * 1.6 + 2.4, expected_250 * 4 + 0.0  # 99 % → 4 / 10
    count_250 = int(last.sum())
    zone = "green" if count_250 <= round(green) else "red" if count_250 >= round(red) else "yellow"
    dates = returns.index[params.window :]
    return AnalysisOutput(
        "test",
        {
            "tests": [
                {
                    "test": "kupiec_pof",
                    "h0": "exception rate = 1 − α",
                    "statistic": lr_pof,
                    "p_value": p_pof,
                    "critical_values": {},
                    "alpha": 0.05,
                    "reject": p_pof < 0.05,
                    "reading": "wrong exception rate"
                    if p_pof < 0.05
                    else "exception rate consistent",
                },
                {
                    "test": "christoffersen_ind",
                    "h0": "exceptions independent",
                    "statistic": lr_ind,
                    "p_value": p_ind,
                    "critical_values": {},
                    "alpha": 0.05,
                    "reject": p_ind < 0.05,
                    "reading": "exceptions cluster" if p_ind < 0.05 else "no clustering",
                },
                {
                    "test": "conditional_coverage",
                    "h0": "correct rate and independence",
                    "statistic": lr_cc,
                    "p_value": float(stats.chi2.sf(lr_cc, 2)),
                    "critical_values": {},
                    "alpha": 0.05,
                    "reject": bool(stats.chi2.sf(lr_cc, 2) < 0.05),
                    "reading": "",
                },
            ],
            "exceptions": x,
            "observations": n,
            "expected": n * p,
            "exception_rate": x / n,
            "traffic_light": {"zone": zone, "exceptions_250": count_250},
            "dates": [_iso(d) for d in dates],
            "var": var.tolist(),
            "realised": realised.tolist(),
            "exception_dates": [_iso(d) for d, h in zip(dates, hits, strict=True) if h],
        },
    )


# ── R-013 / R-014 / R-015 / R-018 benchmark-relative ─────────────────────────


def _pair_returns(inputs: AnalysisInputs) -> pd.DataFrame:
    other = inputs.data.get("other_prices")
    if not isinstance(other, pd.DataFrame) or other.empty:
        bench = inputs.conventions.get("other_symbol") or "the benchmark"
        raise InsufficientDataError(f"Ingest {bench} (or choose another benchmark) to compare")
    field = _field(inputs)
    a = _windowed(inputs, _prices(inputs))[field]
    both = pd.concat([a, other[field]], axis=1, join="inner").pct_change().dropna()
    both.columns = ["asset", "bench"]
    if len(both) < 30:
        raise InsufficientDataError("Fewer than 30 common sessions with the benchmark")
    return both


@analysis(
    id="R-013",
    version="1.0.0",
    title="CAPM beta & alpha",
    description=(
        "OLS of excess asset returns on excess benchmark returns: β, Jensen α (annualised), R², "
        "Newey–West (5 lags) t-statistics; Blume-adjusted β = 0.67β + 0.33; Dimson β = sum of "
        "slopes on lagged, current and leading benchmark returns."
    ),
    applies_to=PRICE_SUBJECTS,
    params=RiskParams,
    result_kind="table",
    requires={"prices", "other_prices"},
    min_obs=30,
)
def capm(inputs: AnalysisInputs, params: RiskParams) -> AnalysisOutput:
    factor = _annualisation(inputs)
    both = _pair_returns(inputs) - params.rf / factor
    fit = OLS(both["asset"].to_numpy(), add_constant(both["bench"].to_numpy())).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5}
    )
    alpha, beta = fit.params
    lagged = pd.concat(
        [both["bench"].shift(1), both["bench"], both["bench"].shift(-1)], axis=1
    ).dropna()
    dimson = OLS(both["asset"].loc[lagged.index].to_numpy(), add_constant(lagged.to_numpy())).fit()
    points = both.iloc[:: max(len(both) // 500, 1)]
    return AnalysisOutput(
        "table",
        {
            "beta": float(beta),
            "alpha_annualised": float(alpha * factor),
            "r_squared": float(fit.rsquared),
            "t_beta": float(fit.tvalues[1]),
            "t_alpha": float(fit.tvalues[0]),
            "blume_beta": 0.67 * float(beta) + 0.33,
            "dimson_beta": float(np.sum(dimson.params[1:])),
            "n": len(both),
            "benchmark": inputs.conventions.get("other_symbol"),
            "scatter": {"bench": points["bench"].tolist(), "asset": points["asset"].tolist()},
        },
    )


class RollingBetaParams(RiskParams):
    window: int = Field(126, ge=21, le=756)


@analysis(
    id="R-014",
    version="1.0.0",
    title="Rolling beta / alpha / correlation",
    description=(
        "Trailing-window OLS β, annualised α and correlation of daily returns vs the benchmark."
    ),
    applies_to=PRICE_SUBJECTS,
    params=RollingBetaParams,
    result_kind="series",
    requires={"prices", "other_prices"},
    min_obs=30,
)
def rolling_beta(inputs: AnalysisInputs, params: RollingBetaParams) -> AnalysisOutput:
    factor = _annualisation(inputs)
    both = _pair_returns(inputs)
    if len(both) <= params.window:
        raise InsufficientDataError("History shorter than the rolling window")
    roll = both.rolling(params.window)
    cov = roll.cov().xs("bench", level=1)["asset"]
    var = both["bench"].rolling(params.window).var()
    beta = cov / var
    alpha = (
        both["asset"].rolling(params.window).mean()
        - beta * both["bench"].rolling(params.window).mean()
    ) * factor
    corr = both["asset"].rolling(params.window).corr(both["bench"])
    return AnalysisOutput(
        "series",
        {
            "dates": [_iso(d) for d in both.index],
            "beta": [_f(v) for v in beta],
            "alpha": [_f(v) for v in alpha],
            "correlation": [_f(v) for v in corr],
            "window": params.window,
            "benchmark": inputs.conventions.get("other_symbol"),
        },
    )


@analysis(
    id="R-015",
    version="1.0.0",
    title="Benchmark-relative set",
    description=(
        "Tracking error (annualised std of active returns), information ratio, Treynor "
        "(excess CAGR / β), M² = SR·σ_b + rf, up/down capture (mean asset / mean benchmark "
        "return on up/down benchmark periods), monthly batting average, max relative drawdown."
    ),
    applies_to=PRICE_SUBJECTS,
    params=RiskParams,
    result_kind="table",
    requires={"prices", "other_prices"},
    min_obs=30,
)
def relative_set(inputs: AnalysisInputs, params: RiskParams) -> AnalysisOutput:
    factor = _annualisation(inputs)
    both = _pair_returns(inputs)
    active = both["asset"] - both["bench"]
    te = float(active.std(ddof=1) * math.sqrt(factor))
    beta = float(np.cov(both["asset"], both["bench"], ddof=1)[0, 1] / both["bench"].var(ddof=1))
    wealth = (1 + both).cumprod()
    days = max((both.index[-1] - both.index[0]).days, 1)
    growth = {k: cagr(1.0, float(wealth[k].iloc[-1]), days) for k in ("asset", "bench")}
    sr = sharpe_ratio(both["asset"].to_numpy(), params.rf, factor)
    up, down = both[both["bench"] > 0], both[both["bench"] < 0]
    monthly = (1 + both).resample("ME").prod() - 1
    relative = drawdown(wealth["asset"] / wealth["bench"])
    return AnalysisOutput(
        "table",
        {
            "tracking_error": te,
            "information_ratio": (growth["asset"] - growth["bench"]) / te if te else None,
            "treynor": (growth["asset"] - params.rf) / beta if beta else None,
            "m_squared": sr * float(both["bench"].std(ddof=1) * math.sqrt(factor)) + params.rf,
            "up_capture": float(up["asset"].mean() / up["bench"].mean()) if len(up) else None,
            "down_capture": float(down["asset"].mean() / down["bench"].mean())
            if len(down)
            else None,
            "batting_average": float((monthly["asset"] > monthly["bench"]).mean()),
            "max_relative_drawdown": float(relative.min()),
            "beta": beta,
            "cagr_asset": growth["asset"],
            "cagr_benchmark": growth["bench"],
            "benchmark": inputs.conventions.get("other_symbol"),
        },
    )


@analysis(
    id="R-018",
    version="1.0.0",
    title="Downside capture & beta",
    description=(
        "Downside β (OLS on benchmark-down days), upside β, standardised co-skewness "
        "E[(a−ā)(b−b̄)²]/(σ_a σ_b²) and co-kurtosis E[(a−ā)(b−b̄)³]/(σ_a σ_b³)."
    ),
    applies_to=PRICE_SUBJECTS,
    params=RiskParams,
    result_kind="table",
    requires={"prices", "other_prices"},
    min_obs=30,
)
def downside_beta(inputs: AnalysisInputs, params: RiskParams) -> AnalysisOutput:
    both = _pair_returns(inputs)
    a, b = both["asset"].to_numpy(), both["bench"].to_numpy()

    def slope(mask: np.ndarray) -> float | None:
        return float(np.polyfit(b[mask], a[mask], 1)[0]) if mask.sum() > 10 else None

    da, db = a - a.mean(), b - b.mean()
    return AnalysisOutput(
        "table",
        {
            "downside_beta": slope(b < 0),
            "upside_beta": slope(b > 0),
            "coskewness": float(np.mean(da * db**2) / (a.std() * b.std() ** 2)),
            "cokurtosis": float(np.mean(da * db**3) / (a.std() * b.std() ** 3)),
        },
    )


# ── R-016 stress windows ─────────────────────────────────────────────────────

STRESS_WINDOWS: list[tuple[str, date, date]] = [
    ("GFC", date(2007, 10, 9), date(2009, 3, 9)),
    ("Lehman month", date(2008, 9, 12), date(2008, 10, 10)),
    ("Euro crisis 2011", date(2011, 7, 1), date(2011, 10, 3)),
    ("China deval. Aug-2015", date(2015, 8, 10), date(2015, 8, 25)),
    ("Volmageddon 2018", date(2018, 1, 26), date(2018, 2, 8)),
    ("Q4-2018", date(2018, 9, 20), date(2018, 12, 24)),
    ("COVID crash", date(2020, 2, 19), date(2020, 3, 23)),
    ("2022 rates", date(2022, 1, 3), date(2022, 10, 12)),
    ("SVB Mar-2023", date(2023, 3, 8), date(2023, 3, 17)),
    ("Carry unwind Aug-2024", date(2024, 7, 31), date(2024, 8, 5)),
    ("Tariffs Apr-2025", date(2025, 4, 2), date(2025, 4, 8)),
]


@analysis(
    id="R-016",
    version="1.0.0",
    title="Historical stress windows",
    description=(
        "Return and max drawdown of the asset (and the benchmark when stored) inside named "
        "market stress windows covered by its full history."
    ),
    applies_to=PRICE_SUBJECTS,
    params=RiskParams,
    result_kind="table",
    requires={"prices", "other_prices"},
    min_obs=2,
)
def stress_windows(inputs: AnalysisInputs, params: RiskParams) -> AnalysisOutput:
    _annualisation(inputs)
    prices = _prices(inputs)[_field(inputs)]
    other = inputs.data.get("other_prices")
    bench = other[_field(inputs)] if isinstance(other, pd.DataFrame) and not other.empty else None
    rows = []
    for name, start, end in STRESS_WINDOWS:

        def window(series: pd.Series, s: date = start, e: date = end) -> pd.Series:
            before = series.loc[series.index < pd.Timestamp(s)]
            inside = series.loc[
                (series.index >= pd.Timestamp(s)) & (series.index <= pd.Timestamp(e))
            ]
            return pd.concat([before.iloc[-1:], inside]) if len(before) else inside

        chunk = window(prices)
        if len(chunk) < 2 or prices.index[0] > pd.Timestamp(start):
            rows.append(
                {
                    "window": name,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "covered": False,
                }
            )
            continue
        row: dict[str, Any] = {
            "window": name,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "covered": True,
            "return": float(chunk.iloc[-1] / chunk.iloc[0] - 1),
            "max_drawdown": float(drawdown(chunk).min()),
        }
        if bench is not None:
            b = window(bench)
            row["benchmark_return"] = float(b.iloc[-1] / b.iloc[0] - 1) if len(b) > 1 else None
        rows.append(row)
    return AnalysisOutput(
        "table", {"rows": rows, "benchmark": inputs.conventions.get("other_symbol")}
    )


# ── R-017 leveraged-product decay ────────────────────────────────────────────


class DecayParams(RiskParams):
    leverage: float | None = Field(
        None, ge=-5, le=5, description="k (default: from the asset type)"
    )


@analysis(
    id="R-017",
    version="1.0.0",
    title="Leveraged-product decay",
    description=(
        "Realised path of the product vs k × the daily benchmark return compounded and vs k × "
        "the benchmark's total return; volatility decay in log terms, "
        "log(1 + R_k) − k·log(1 + R_b) ≈ −k(k − 1)/2·σ²·T, realised vs approximation."
    ),
    applies_to=PRICE_SUBJECTS,
    params=DecayParams,
    result_kind="series",
    requires={"prices", "other_prices"},
    min_obs=30,
)
def leverage_decay(inputs: AnalysisInputs, params: DecayParams) -> AnalysisOutput:
    factor = _annualisation(inputs)
    k = (
        params.leverage
        if params.leverage is not None
        else inputs.conventions.get("leverage_factor")
    )
    if k is None:
        raise InsufficientDataError("Not a leveraged product: set the leverage k explicitly")
    k = float(k)
    both = _pair_returns(inputs)
    product = (1 + both["asset"]).cumprod()
    daily_reset = (1 + k * both["bench"]).cumprod()
    bench_total = float((1 + both["bench"]).prod() - 1)
    years = len(both) / factor
    sigma2 = float(both["bench"].var(ddof=1) * factor)
    return AnalysisOutput(
        "series",
        {
            "dates": [_iso(d) for d in both.index],
            "product": product.tolist(),
            "daily_reset": daily_reset.tolist(),
            "k": k,
            "product_return": float(product.iloc[-1] - 1),
            "k_times_benchmark": k * bench_total,
            "decay_realised": float(np.log(product.iloc[-1]) - k * math.log1p(bench_total)),
            "decay_approximation": -k * (k - 1) / 2 * sigma2 * years,
            "benchmark": inputs.conventions.get("other_symbol"),
        },
        (
            "Daily-reset products compound k × daily returns: over long horizons they do "
            "not deliver k × the period return.",
        ),
    )


# ── R-019 / R-023 drawdown risk ──────────────────────────────────────────────


class CdarParams(RiskParams):
    confidence: float = Field(0.95, ge=0.5, lt=1.0)


@analysis(
    id="R-019",
    version="1.0.0",
    title="Conditional Drawdown-at-Risk",
    description=(
        "Drawdown-at-Risk = α-quantile of the drawdown depths observed each session; CDaR = "
        "mean of drawdowns at or beyond it (Chekhlov–Uryasev–Zabarankin)."
    ),
    applies_to=PRICE_SUBJECTS,
    params=CdarParams,
    result_kind="table",
    requires={"prices"},
    min_obs=20,
)
def cdar(inputs: AnalysisInputs, params: CdarParams) -> AnalysisOutput:
    prices, _, _ = _series(inputs)
    depth = -drawdown(prices).to_numpy()
    dar = float(np.quantile(depth, params.confidence))
    tail = depth[depth >= dar]
    return AnalysisOutput(
        "table",
        {
            "drawdown_at_risk": dar,
            "cdar": float(tail.mean()),
            "confidence": params.confidence,
            "max_drawdown": float(depth.max()),
        },
    )


def magdon_ismail_qp(x: float) -> float:
    """Q_p(x) of Magdon-Ismail et al. (2004) for μ > 0: small-x and large-x asymptotes,
    blended linearly between them (an approximation, flagged in the output)."""
    small = math.sqrt(math.pi / 8) * math.sqrt(2 * x)
    large = 0.25 * math.log(x) + 0.49088 if x > 0 else 0.0
    if x <= 0.5:
        return small
    if x >= 3.0:
        return large
    w = (x - 0.5) / 2.5
    return (1 - w) * small + w * large


class DrawdownDistributionParams(RiskParams):
    horizon_years: float = Field(1.0, gt=0.05, le=10)
    n_sims: int = Field(500, ge=100, le=5_000)
    seed: int = Field(29, ge=0, le=2**32 - 1)


@analysis(
    id="R-023",
    version="1.0.0",
    title="Drawdown distribution",
    description=(
        "Distribution of episode depths and durations; expected maximum drawdown over a "
        "horizon by the Magdon-Ismail Brownian approximation (μ > 0) and by stationary block "
        "bootstrap of daily returns (median and 95th percentile)."
    ),
    applies_to=PRICE_SUBJECTS,
    params=DrawdownDistributionParams,
    result_kind="table",
    requires={"prices"},
    min_obs=60,
    budget_ms=10_000,
)
def drawdown_distribution(
    inputs: AnalysisInputs, params: DrawdownDistributionParams
) -> AnalysisOutput:
    prices, returns, factor = _series(inputs)
    found = episodes(drawdown(prices))
    depths = np.array([e["depth"] for e in found]) if found else np.array([0.0])
    durations = np.array([e["length_sessions"] for e in found]) if found else np.array([0])
    log_r = np.log1p(returns.to_numpy())
    mu, sigma = float(log_r.mean() * factor), float(log_r.std(ddof=1) * math.sqrt(factor))
    analytic = None
    if mu > 0 and sigma > 0:
        x = params.horizon_years * mu**2 / (2 * sigma**2)
        analytic = 1 - math.exp(-(2 * sigma**2 / mu) * magdon_ismail_qp(x))
    horizon = max(int(params.horizon_years * factor), 5)
    r = returns.to_numpy()

    def mdd(path: np.ndarray) -> float:
        wealth = np.cumprod(1 + path[:horizon])
        return float(-(wealth / np.maximum.accumulate(wealth) - 1).min())

    if len(r) < horizon:
        r = np.resize(r, horizon)
    boot = stationary_bootstrap(r, mdd, params.n_sims, max(len(r) ** (1 / 3), 1.0), params.seed)
    return AnalysisOutput(
        "table",
        {
            "episodes": len(found),
            "depth_quantiles": np.quantile(depths, [0.5, 0.9, 1.0]).tolist(),
            "duration_quantiles": np.quantile(durations, [0.5, 0.9, 1.0]).tolist(),
            "depth_histogram": np.histogram(depths, bins=20)[0].tolist(),
            "expected_mdd_analytic": analytic,
            "bootstrap_mdd_median": float(np.median(boot)),
            "bootstrap_mdd_p95": float(np.quantile(boot, 0.95)),
            "horizon_years": params.horizon_years,
        },
        (
            "Analytic expectation uses a Brownian approximation of Q_p; "
            "trust the bootstrap for fat tails.",
        ),
    )


# ── R-020 risk card ──────────────────────────────────────────────────────────


@analysis(
    id="R-020",
    version="1.0.0",
    title="Risk summary card",
    description=(
        "Plain-language sheet: annualised volatility bucket (< 10 % low, < 20 % moderate, "
        "< 35 % high, else very high), worst day and month, maximum drawdown, and the 1-day "
        "95 % historical VaR per 10 000 invested in the asset currency."
    ),
    applies_to=PRICE_SUBJECTS,
    params=RiskParams,
    result_kind="table",
    requires={"prices"},
    min_obs=60,
)
def risk_card(inputs: AnalysisInputs, params: RiskParams) -> AnalysisOutput:
    prices, returns, factor = _series(inputs)
    r = returns.to_numpy()
    vol = float(r.std(ddof=1) * math.sqrt(factor))
    bucket = (
        "low" if vol < 0.10 else "moderate" if vol < 0.20 else "high" if vol < 0.35 else "very high"
    )
    monthly = prices.resample("ME").last().pct_change().dropna()
    var95 = -float(np.quantile(r, 0.05))
    currency = inputs.conventions.get("currency") or ""
    worst_day = returns.idxmin()
    return AnalysisOutput(
        "table",
        {
            "volatility": vol,
            "bucket": bucket,
            "worst_day": {"date": _iso(worst_day), "return": float(returns.min())},
            "worst_month": {"date": _iso(monthly.idxmin()), "return": float(monthly.min())}
            if len(monthly)
            else None,
            "max_drawdown": float(drawdown(prices).min()),
            "var95_1d": var95,
            "var95_per_10000": var95 * 10_000,
            "currency": currency,
            "sentence": (
                f"On 1 day in 20, a 10 000 {currency} position lost about "
                f"{var95 * 10_000:,.0f} {currency} or more over this period."
            ).replace(",", " "),
        },
    )


# ── R-021 model-based VaR/ES ─────────────────────────────────────────────────


class ModelVarParams(RiskParams):
    confidence: float = Field(0.99, ge=0.9, le=0.999)
    horizon: int = Field(1, ge=1, le=20)
    n_sims: int = Field(5_000, ge=1_000, le=50_000)
    seed: int = Field(31, ge=0, le=2**32 - 1)


@analysis(
    id="R-021",
    version="1.0.0",
    title="Model-based VaR/ES",
    description=(
        "GARCH(1,1)-t filtered historical simulation (standardised residuals rescaled by the "
        "next-day σ; h-day paths by simulating the GARCH recursion), bootstrap and Student-t "
        "Monte Carlo of h-day returns, compared with historical VaR/ES."
    ),
    applies_to=PRICE_SUBJECTS,
    params=ModelVarParams,
    result_kind="table",
    requires={"prices"},
    min_obs=250,
    budget_ms=15_000,
)
def model_var(inputs: AnalysisInputs, params: ModelVarParams) -> AnalysisOutput:
    _, returns, _ = _series(inputs)
    r = returns.to_numpy()
    if len(r) < 250:
        raise InsufficientDataError("Model-based VaR needs ≥ 250 returns")
    rng = np.random.default_rng(params.seed)
    h, p = params.horizon, 1 - params.confidence
    fit = arch_model(r * 100, p=1, q=1, dist="t").fit(disp="off", show_warning=False)
    omega, alpha, beta = (float(fit.params[k]) for k in ("omega", "alpha[1]", "beta[1]"))
    mu = float(fit.params["mu"])
    z = np.asarray(fit.std_resid, float)
    z = z[np.isfinite(z)]
    sigma2 = float(fit.forecast(horizon=1, reindex=False).variance.to_numpy()[-1, 0])
    paths = np.zeros(params.n_sims)
    s2 = np.full(params.n_sims, sigma2)
    for _ in range(h):
        eps = np.sqrt(s2) * rng.choice(z, params.n_sims)
        paths += (mu + eps) / 100
        s2 = omega + alpha * eps**2 + beta * s2
    fhs = paths
    boot = np.array([r[rng.integers(0, len(r), h)].sum() for _ in range(params.n_sims)])
    nu, loc, scale = stats.t.fit(r)
    student = stats.t.rvs(max(nu, 2.01), loc, scale, size=(params.n_sims, h), random_state=rng).sum(
        axis=1
    )
    hist = np.convolve(r, np.ones(h), "valid") if h > 1 else r

    def row(name: str, sample: np.ndarray) -> dict[str, Any]:
        q = np.quantile(sample, p)
        return {"method": name, "var": float(-q), "es": float(-sample[sample <= q].mean())}

    return AnalysisOutput(
        "table",
        {
            "confidence": params.confidence,
            "horizon": h,
            "rows": [
                row("historical", hist),
                row("garch_fhs", fhs),
                row("bootstrap_mc", boot),
                row("student_t_mc", student),
            ],
            "garch": {
                "omega": omega,
                "alpha": alpha,
                "beta": beta,
                "next_day_vol": math.sqrt(sigma2) / 100,
            },
        },
        ("FHS reacts to the current volatility regime; unconditional methods do not.",),
    )


# ── R-022 extreme value theory ───────────────────────────────────────────────


class EvtParams(RiskParams):
    threshold_quantile: float = Field(0.95, ge=0.8, le=0.99)
    confidences: list[float] = Field(default=[0.99, 0.995, 0.999], min_length=1, max_length=5)


@analysis(
    id="R-022",
    version="1.0.0",
    title="Extreme-value tail risk",
    description=(
        "Peaks-over-threshold on daily losses: GPD (ξ, β) fitted by MLE to exceedances over the "
        "u = q-quantile, EVT VaR_p = u + β/ξ·[(n/N_u·(1 − p))^(−ξ) − 1] and ES_p = (VaR_p + β − ξu)"
        "/(1 − ξ); mean-excess plot; Hill index; return levels (worst daily loss expected once "
        "in N years)."
    ),
    applies_to=PRICE_SUBJECTS,
    params=EvtParams,
    result_kind="table",
    requires={"prices"},
    min_obs=250,
)
def extreme_value(inputs: AnalysisInputs, params: EvtParams) -> AnalysisOutput:
    _, returns, factor = _series(inputs)
    losses = -returns.to_numpy()
    if len(losses) < 250:
        raise InsufficientDataError("EVT needs ≥ 250 observations")
    u = float(np.quantile(losses, params.threshold_quantile))
    excess = losses[losses > u] - u
    n, n_u = len(losses), len(excess)
    if n_u < 20:
        raise InsufficientDataError("Fewer than 20 exceedances: lower the threshold quantile")
    xi, _, beta = stats.genpareto.fit(excess, floc=0)

    def evt_var(p: float) -> float:
        tail = n / n_u * (1 - p)
        if abs(xi) > 1e-9:
            return float(u + beta / xi * (tail ** (-xi) - 1))
        return float(u - beta * math.log(tail))

    rows = []
    for p in params.confidences:
        v = evt_var(p)
        es = (v + beta - xi * u) / (1 - xi) if xi < 1 else None
        rows.append(
            {
                "confidence": p,
                "var": float(v),
                "es": _f(es),
                "historical_var": float(np.quantile(losses, p)),
            }
        )
    sorted_losses = np.sort(losses)
    thresholds = np.quantile(losses, np.linspace(0.80, 0.995, 30))
    mean_excess = [
        {"u": float(t), "mean_excess": float((losses[losses > t] - t).mean())} for t in thresholds
    ]
    k = n_u
    hill = (
        float(1 / np.mean(np.log(sorted_losses[-k:] / sorted_losses[-k - 1])))
        if sorted_losses[-k - 1] > 0
        else None
    )
    return_levels = [
        {"years": y, "loss": float(evt_var(1 - 1 / (y * factor)))} for y in (1, 5, 10, 25)
    ]
    return AnalysisOutput(
        "table",
        {
            "threshold": u,
            "exceedances": n_u,
            "xi": float(xi),
            "beta": float(beta),
            "rows": rows,
            "mean_excess": mean_excess,
            "hill_alpha": hill,
            "return_levels": return_levels,
        },
        () if xi < 1 else ("ξ ≥ 1: infinite mean tail, ES undefined.",),
    )
