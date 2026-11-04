"""Stationarity, dependence, memory, spectrum and distribution dynamics (TS-001, TS-002,
TS-004, TS-005, TS-006, TS-008, TS-017, TS-021, TS-023)."""

import math
import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
from arch.unitroot import PhillipsPerron, VarianceRatio
from pydantic import Field
from scipy import signal, stats
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, bds, kpss, pacf, zivot_andrews

from fda_core.analyses.price import PRICE_SUBJECTS, _annualisation, _field, _prices, _windowed
from fda_core.analyses.ts_common import (
    SeriesParams,
    clean,
    ffd_weights,
    frac_diff,
    iso,
    make_test,
    require,
    sample_info,
    select_series,
)
from fda_core.registry import AnalysisInputs, AnalysisOutput, InsufficientDataError, analysis

# ── TS-001 stationarity battery ──────────────────────────────────────────────


class StationarityParams(SeriesParams):
    series: Literal[
        "log_price",
        "price",
        "returns",
        "log_returns",
        "frac_diff",
        "ratio",
        "spread",
        "zscore",
        "detrended",
        "log_volume",
    ] = "log_price"
    trend: Literal["c", "ct"] = "c"
    autolag: Literal["AIC", "BIC"] = "AIC"
    alpha: float = Field(0.05, gt=0, lt=0.5)
    rolling_window: int = Field(252, ge=60, le=1_260)


def adf_row(x: np.ndarray, trend: str, autolag: str, alpha: float) -> dict[str, Any]:
    stat, p, lags, nobs, crit, _ = adfuller(x, regression=trend, autolag=autolag)
    return make_test(
        "adf",
        "unit root (non-stationary)",
        stat,
        p,
        {k: float(v) for k, v in crit.items()},
        alpha,
        reading_reject="stationary at α",
        reading_keep="unit root not rejected",
        extra={"lags": int(lags), "nobs": int(nobs)},
    )


@analysis(
    id="TS-001",
    version="1.0.0",
    title="Stationarity battery",
    description=(
        "ADF (lags by AIC/BIC), KPSS (H0 stationary), Phillips–Perron and Zivot–Andrews (one "
        "endogenous break) on the chosen series; order of integration from ADF on levels and "
        "differences; rolling ADF statistic."
    ),
    applies_to=PRICE_SUBJECTS,
    params=StationarityParams,
    result_kind="test",
    requires={"prices"},
    min_obs=50,
    budget_ms=8_000,
)
def stationarity(inputs: AnalysisInputs, params: StationarityParams) -> AnalysisOutput:
    series = select_series(inputs, params)
    x = require(series, 50, "Stationarity tests")
    trend = params.trend
    tests = [adf_row(x, trend, params.autolag, params.alpha)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        k_stat, k_p, k_lags, k_crit = kpss(x, regression=trend, nlags="auto")
    tests.append(
        make_test(
            "kpss",
            "stationary",
            k_stat,
            k_p,
            {k: float(v) for k, v in k_crit.items()},
            params.alpha,
            reading_reject="non-stationary at α",
            reading_keep="stationarity not rejected",
            extra={"lags": int(k_lags), "p_value_bounded": bool(k_p in (0.01, 0.1))},
        )
    )
    pp = PhillipsPerron(x, trend=trend)
    tests.append(
        make_test(
            "phillips_perron",
            "unit root",
            float(pp.stat),
            float(pp.pvalue),
            {k: float(v) for k, v in pp.critical_values.items()},
            params.alpha,
            reading_reject="stationary at α",
            reading_keep="unit root not rejected",
        )
    )
    if len(x) >= 100:
        # Zivot–Andrews re-estimates ADF at every candidate break: cap to the last 1 000 points.
        za_x = x[-1_000:]
        za_offset = len(x) - len(za_x)
        za_stat, za_p, za_crit, _, za_break = zivot_andrews(
            za_x, regression="c" if trend == "c" else "ct", autolag=params.autolag
        )
        za_break = int(za_break) + za_offset
        tests.append(
            make_test(
                "zivot_andrews",
                "unit root without break",
                za_stat,
                za_p,
                {k: float(v) for k, v in za_crit.items()},
                params.alpha,
                reading_reject="stationary around one break",
                reading_keep="unit root not rejected",
                extra={"break_date": iso(series.index[[int(za_break)]])[0]},
            )
        )
    order = 0
    level = x
    while (
        order < 2 and adfuller(level, regression=trend, autolag=params.autolag)[1] >= params.alpha
    ):
        level = np.diff(level)
        order += 1
    rolling: dict[str, list[Any]] = {"dates": [], "adf": []}
    w = params.rolling_window
    if len(x) >= w + 21:
        for end in range(w, len(x) + 1, 21):
            rolling["dates"].append(iso(series.index[[end - 1]])[0])
            rolling["adf"].append(
                float(adfuller(x[end - w : end], regression=trend, maxlag=1, autolag=None)[0])
            )
    return AnalysisOutput(
        "test",
        {
            "tests": tests,
            "order_of_integration": order if order < 2 else ">=2",
            "rolling_adf": rolling,
            "sample": sample_info(series, params),
        },
    )


# ── TS-002 serial dependence ─────────────────────────────────────────────────


class DependenceParams(SeriesParams):
    nlags: int = Field(20, ge=5, le=60)
    alpha: float = Field(0.05, gt=0, lt=0.5)


@analysis(
    id="TS-002",
    version="1.0.0",
    title="Serial dependence",
    description=(
        "ACF/PACF of r, |r| and r² with ±1.96/√n bands; Ljung–Box Q at 5/10/20 lags; "
        "Lo–MacKinlay variance ratios (heteroskedasticity-robust) for q = 2, 5, 10, 20; runs "
        "test on signs; BDS test for non-linear dependence (dimensions 2–3)."
    ),
    applies_to=PRICE_SUBJECTS,
    params=DependenceParams,
    result_kind="test",
    requires={"prices"},
    min_obs=60,
    budget_ms=8_000,
)
def serial_dependence(inputs: AnalysisInputs, params: DependenceParams) -> AnalysisOutput:
    series = select_series(inputs, params)
    r = require(series, 60, "Serial dependence")
    nlags = min(params.nlags, len(r) // 4)
    band = 1.96 / math.sqrt(len(r))
    correlograms = {}
    for name, values in (("r", r), ("abs", np.abs(r)), ("squared", r**2)):
        correlograms[name] = {
            "acf": acf(values, nlags=nlags, fft=True)[1:].tolist(),
            "pacf": pacf(values, nlags=nlags, method="ywm")[1:].tolist(),
        }
    lb = acorr_ljungbox(r, lags=[k for k in (5, 10, 20) if k < len(r) // 2])
    tests = [
        make_test(
            f"ljung_box_{int(lag)}",
            "no autocorrelation up to lag",
            float(row.lb_stat),
            float(row.lb_pvalue),
            alpha=params.alpha,
            reading_reject="serial correlation",
            reading_keep="no evidence of autocorrelation",
        )
        for lag, row in lb.iterrows()
    ]
    prices = np.exp(np.cumsum(np.log1p(r))) if params.series == "returns" else None
    if prices is not None:
        for q in (2, 5, 10, 20):
            if len(r) > 4 * q:
                vr = VarianceRatio(np.log(prices), lags=q, robust=True)
                tests.append(
                    make_test(
                        f"variance_ratio_{q}",
                        "random walk (VR = 1)",
                        float(vr.stat),
                        float(vr.pvalue),
                        alpha=params.alpha,
                        reading_reject="mean reversion" if vr.vr < 1 else "momentum",
                        reading_keep="consistent with a random walk",
                        extra={"vr": float(vr.vr)},
                    )
                )
    runs_z, runs_p = runstest_1samp(r, cutoff=0.0, correction=True)
    tests.append(
        make_test(
            "runs",
            "signs are random",
            runs_z,
            runs_p,
            alpha=params.alpha,
            reading_reject="non-random sign sequence",
            reading_keep="random sign sequence",
        )
    )
    if len(r) <= 5_000:
        bds_stat, bds_p = bds(r, max_dim=3)
        for dim, (s, p) in enumerate(
            zip(np.atleast_1d(bds_stat), np.atleast_1d(bds_p), strict=True), 2
        ):
            tests.append(
                make_test(
                    f"bds_m{dim}",
                    "i.i.d.",
                    float(s),
                    float(p),
                    alpha=params.alpha,
                    reading_reject="non-linear dependence",
                    reading_keep="no evidence against i.i.d.",
                )
            )
    return AnalysisOutput(
        "test",
        {
            "correlograms": correlograms,
            "band": band,
            "tests": tests,
            "sample": sample_info(series, params),
        },
    )


# ── TS-004 leverage effect & volatility dynamics ─────────────────────────────


@analysis(
    id="TS-004",
    version="1.0.0",
    title="Leverage effect & vol dynamics",
    description=(
        "corr(r_t, |r|_{t+k}) for k = 1…20 (negative = leverage effect), vol-of-vol of 21-day "
        "realised volatility, realised-vol terciles and the half-life of log realised vol "
        "from an AR(1) on monthly samples."
    ),
    applies_to=PRICE_SUBJECTS,
    params=SeriesParams,
    result_kind="table",
    requires={"prices"},
    min_obs=150,
)
def vol_dynamics(inputs: AnalysisInputs, params: SeriesParams) -> AnalysisOutput:
    factor = _annualisation(inputs)
    returns = _windowed(inputs, _prices(inputs))[_field(inputs)].pct_change().dropna()
    r = require(returns, 150, "Volatility dynamics")
    leverage = [float(np.corrcoef(r[:-k], np.abs(r[k:]))[0, 1]) for k in range(1, 21)]
    rv = returns.rolling(21).std().dropna() * math.sqrt(factor)
    log_rv = np.log(rv[rv > 0])
    monthly = log_rv.iloc[::21].to_numpy()
    phi = float(np.corrcoef(monthly[:-1], monthly[1:])[0, 1]) if len(monthly) > 5 else float("nan")
    half_life = -math.log(2) / math.log(phi) * 21 if 0 < phi < 1 else None
    terciles = np.quantile(rv, [1 / 3, 2 / 3])
    return AnalysisOutput(
        "table",
        {
            "leverage_corr": leverage,
            "vol_of_vol": float(log_rv.std(ddof=1)),
            "rv_terciles": terciles.tolist(),
            "rv_last": float(rv.iloc[-1]),
            "rv_regime": int(np.searchsorted(terciles, rv.iloc[-1])),
            "rv_half_life_days": half_life,
            "rv": {"dates": iso(rv.index), "values": clean(rv)},
        },
    )


# ── TS-005 long memory ───────────────────────────────────────────────────────


def hurst_rs(x: np.ndarray) -> dict[str, Any]:
    """R/S Hurst with the Anis–Lloyd–Peters expected-R/S correction (H = 0.5 + slope)."""
    n = len(x)
    sizes = np.unique(np.floor(np.logspace(math.log10(10), math.log10(n // 2), 20)).astype(int))
    rs, expected = [], []
    for m in sizes:
        chunks = x[: (n // m) * m].reshape(-1, m)
        dev = np.cumsum(chunks - chunks.mean(axis=1, keepdims=True), axis=1)
        r_range = dev.max(axis=1) - dev.min(axis=1)
        s = chunks.std(axis=1, ddof=0)
        valid = s > 0
        rs.append(float(np.mean(r_range[valid] / s[valid])))
        i = np.arange(1, m)
        factor = (m - 0.5) / m if m <= 340 else 1.0
        gamma_term = (
            math.gamma((m - 1) / 2) / (math.sqrt(math.pi) * math.gamma(m / 2))
            if m <= 340
            else 1 / math.sqrt(m * math.pi / 2)
        )
        expected.append(factor * gamma_term * float(np.sum(np.sqrt((m - i) / i))))
    log_n = np.log(sizes)
    raw = float(np.polyfit(log_n, np.log(rs), 1)[0])
    corrected = 0.5 + float(np.polyfit(log_n, np.log(rs) - np.log(expected), 1)[0])
    return {"raw": raw, "anis_lloyd": corrected, "sizes": sizes.tolist(), "rs": rs}


def hurst_dfa(x: np.ndarray) -> float:
    """Detrended fluctuation analysis (linear detrending) on the integrated series."""
    profile = np.cumsum(x - x.mean())
    n = len(x)
    sizes = np.unique(np.floor(np.logspace(1, math.log10(n // 4), 15)).astype(int))
    fluct = []
    for m in sizes:
        segments = profile[: (n // m) * m].reshape(-1, m)
        t = np.arange(m)
        coef = np.polyfit(t, segments.T, 1)
        trend = np.outer(coef[0], t) + coef[1][:, None]
        fluct.append(math.sqrt(float(np.mean((segments - trend) ** 2))))
    return float(np.polyfit(np.log(sizes), np.log(fluct), 1)[0])


def gph(x: np.ndarray, power: float = 0.5) -> tuple[float, float]:
    """Geweke–Porter-Hudak log-periodogram estimate of d and its standard error."""
    n = len(x)
    m = int(n**power)
    freqs = 2 * np.pi * np.arange(1, m + 1) / n
    periodogram = np.abs(np.fft.fft(x - x.mean())[1 : m + 1]) ** 2 / (2 * np.pi * n)
    regressor = np.log(4 * np.sin(freqs / 2) ** 2)
    slope = np.polyfit(regressor, np.log(periodogram), 1)[0]
    se = math.pi / math.sqrt(24 * m)
    return float(-slope), se


@analysis(
    id="TS-005",
    version="1.0.0",
    title="Long memory",
    description=(
        "Hurst exponent by rescaled range (raw and Anis–Lloyd corrected) and by DFA; GPH "
        "log-periodogram d with SE π/√(24m), m = √n; variance-time plot."
    ),
    applies_to=PRICE_SUBJECTS,
    params=SeriesParams,
    result_kind="table",
    requires={"prices"},
    min_obs=200,
)
def long_memory(inputs: AnalysisInputs, params: SeriesParams) -> AnalysisOutput:
    series = select_series(inputs, params)
    x = require(series, 200, "Long-memory estimators")
    rs = hurst_rs(x)
    d, d_se = gph(x)
    lags = [1, 2, 5, 10, 20, 50, 100]
    cumulative = np.cumsum(x)
    variance_time = [
        {
            "m": m,
            "variance_ratio": float(np.var(cumulative[m:] - cumulative[:-m]) / (m * np.var(x))),
        }
        for m in lags
        if m < len(x) // 4
    ]
    return AnalysisOutput(
        "table",
        {
            "hurst_rs": rs["raw"],
            "hurst_rs_corrected": rs["anis_lloyd"],
            "hurst_dfa": hurst_dfa(x),
            "gph_d": d,
            "gph_se": d_se,
            "rs_curve": {"sizes": rs["sizes"], "rs": rs["rs"]},
            "variance_time": variance_time,
            "sample": sample_info(series, params),
        },
        ("Hurst on returns ≈ 0.5 means no memory; on prices it is ≈ 1 by construction.",),
    )


# ── TS-006 fractional differentiation ────────────────────────────────────────


class FracDiffParams(SeriesParams):
    series: Literal["log_price", "price", "ratio", "spread", "log_volume"] = "log_price"
    alpha: float = Field(0.05, gt=0, lt=0.5)
    threshold: float = Field(1e-4, ge=1e-6, le=1e-2)


@analysis(
    id="TS-006",
    version="1.0.0",
    title="Fractional differentiation",
    description=(
        "Fixed-width-window FFD (weights w_k = −w_{k−1}(d − k + 1)/k, cut at τ) for d = 0…1 by "
        "0.05: ADF statistic (constant, 1 lag) and correlation with the original series; the "
        "minimum d* whose ADF rejects a unit root at α keeps the most memory (AFML-5)."
    ),
    applies_to=PRICE_SUBJECTS,
    params=FracDiffParams,
    result_kind="table",
    requires={"prices"},
    min_obs=250,
    budget_ms=10_000,
)
def fractional_differentiation(inputs: AnalysisInputs, params: FracDiffParams) -> AnalysisOutput:
    level = select_series(inputs, params, full=True)
    require(level, 250, "Fractional differentiation")
    rows = []
    for d in np.round(np.arange(0, 1.0001, 0.05), 2):
        try:
            diffed = frac_diff(level, float(d), params.threshold)
        except InsufficientDataError:
            rows.append(
                {
                    "d": float(d),
                    "adf": None,
                    "p_value": None,
                    "corr": None,
                    "width": len(ffd_weights(float(d), params.threshold)),
                }
            )
            continue
        stat, p, *_ = adfuller(diffed.to_numpy(), maxlag=1, regression="c", autolag=None)
        corr = float(np.corrcoef(level.loc[diffed.index], diffed)[0, 1])
        rows.append(
            {
                "d": float(d),
                "adf": float(stat),
                "p_value": float(p),
                "corr": corr,
                "width": len(ffd_weights(float(d), params.threshold)),
            }
        )
    passing = [row for row in rows if row["p_value"] is not None and row["p_value"] < params.alpha]
    d_star = passing[0] if passing else None
    critical = adfuller(level.to_numpy(), maxlag=1, regression="c", autolag=None)[4]["5%"]
    return AnalysisOutput(
        "table",
        {
            "rows": rows,
            "d_star": d_star["d"] if d_star else None,
            "corr_at_d_star": d_star["corr"] if d_star else None,
            "adf_critical_5pct": float(critical),
            "threshold": params.threshold,
        },
        () if d_star else ("No d ≤ 1 makes the series stationary at α.",),
    )


# ── TS-008 decomposition & spectrum ──────────────────────────────────────────


class SpectrumParams(SeriesParams):
    series: Literal["log_price", "returns", "abs_returns", "log_volume"] = "log_price"
    period: int = Field(21, ge=2, le=260, description="STL seasonal period in observations")


@analysis(
    id="TS-008",
    version="1.0.0",
    title="Decomposition & spectrum",
    description=(
        "STL (robust) trend/seasonal/residual with the chosen period; periodogram and Welch "
        "spectral density of the (differenced if level) series; five dominant cycles."
    ),
    applies_to=PRICE_SUBJECTS,
    params=SpectrumParams,
    result_kind="series",
    requires={"prices"},
    min_obs=100,
    budget_ms=8_000,
)
def decomposition(inputs: AnalysisInputs, params: SpectrumParams) -> AnalysisOutput:
    series = select_series(inputs, params)
    require(series, max(100, 2 * params.period + 1), "Decomposition")
    stl = STL(series.to_numpy(float), period=params.period, robust=True).fit()
    x = series.to_numpy(float)
    stationary = np.diff(x) if params.series in {"log_price", "log_volume"} else x
    freqs, power = signal.periodogram(stationary, detrend="constant")
    w_freqs, w_power = signal.welch(stationary, nperseg=min(256, len(stationary)))
    order = np.argsort(power[1:])[::-1][:5] + 1
    cycles = [{"period": float(1 / freqs[i]), "power": float(power[i])} for i in order]
    return AnalysisOutput(
        "series",
        {
            "dates": iso(series.index),
            "observed": clean(x),
            "trend": clean(stl.trend),
            "seasonal": clean(stl.seasonal),
            "resid": clean(stl.resid),
            "welch": {"freq": w_freqs.tolist(), "density": w_power.tolist()},
            "dominant_cycles": cycles,
            "sample": sample_info(series, params),
        },
    )


# ── TS-017 return concentration & runs ───────────────────────────────────────


def hhi_concentration(weights: np.ndarray) -> float | None:
    """Normalised HHI (AFML 14.5): (Σw² − 1/n)/(1 − 1/n); 0 = even, 1 = one observation."""
    n = len(weights)
    if n <= 1 or weights.sum() == 0:
        return None
    w = weights / weights.sum()
    return float((np.sum(w**2) - 1 / n) / (1 - 1 / n))


@analysis(
    id="TS-017",
    version="1.0.0",
    title="Return concentration & runs",
    description=(
        "HHI concentration of positive returns, of negative returns and of observations across "
        "months (AFML-14); up/down run-length frequencies against the geometric null of "
        "independent signs."
    ),
    applies_to=PRICE_SUBJECTS,
    params=SeriesParams,
    result_kind="table",
    requires={"prices"},
    min_obs=60,
)
def concentration(inputs: AnalysisInputs, params: SeriesParams) -> AnalysisOutput:
    returns = _windowed(inputs, _prices(inputs))[_field(inputs)].pct_change().dropna()
    r = require(returns, 60, "Concentration")
    by_month = returns.groupby(pd.DatetimeIndex(returns.index).to_period("M")).size()
    signs = np.sign(r[r != 0])
    p_up = float((signs > 0).mean())
    runs: dict[str, dict[int, int]] = {"up": {}, "down": {}}
    current, length = signs[0], 1
    for s in signs[1:]:
        if s == current:
            length += 1
            continue
        key = "up" if current > 0 else "down"
        runs[key][length] = runs[key].get(length, 0) + 1
        current, length = s, 1
    key = "up" if current > 0 else "down"
    runs[key][length] = runs[key].get(length, 0) + 1
    table = []
    for side, p in (("up", p_up), ("down", 1 - p_up)):
        total = sum(runs[side].values())
        for k in range(1, 9):
            table.append(
                {
                    "side": side,
                    "length": k,
                    "observed": runs[side].get(k, 0) / total if total else 0,
                    "geometric": p ** (k - 1) * (1 - p),
                }
            )
    return AnalysisOutput(
        "table",
        {
            "hhi_positive": hhi_concentration(r[r > 0]),
            "hhi_negative": hhi_concentration(-r[r < 0]),
            "hhi_time": hhi_concentration(by_month.to_numpy(float)),
            "p_up": p_up,
            "runs": table,
        },
    )


# ── TS-021 distribution dynamics ─────────────────────────────────────────────


class DynamicsParams(SeriesParams):
    window: int = Field(252, ge=63, le=1_260)
    tail_fraction: float = Field(0.05, gt=0.01, le=0.2)


def hill(losses: np.ndarray, k: int) -> float | None:
    """Hill tail index α̂ = 1 / mean(log(x_(i) / x_(k+1))) over the k largest losses."""
    x = np.sort(losses[losses > 0])[::-1]
    if len(x) <= k or k < 2:
        return None
    return float(1 / np.mean(np.log(x[:k] / x[k])))


@analysis(
    id="TS-021",
    version="1.0.0",
    title="Distribution dynamics",
    description=(
        "Rolling skewness, excess kurtosis, Hill left-tail index (k = tail fraction × window) and "
        "Jarque–Bera p-value on trailing windows, sampled every 5 sessions."
    ),
    applies_to=PRICE_SUBJECTS,
    params=DynamicsParams,
    result_kind="series",
    requires={"prices"},
    min_obs=63,
)
def distribution_dynamics(inputs: AnalysisInputs, params: DynamicsParams) -> AnalysisOutput:
    returns = _prices(inputs)[_field(inputs)].pct_change().dropna()
    visible = _windowed(inputs, _prices(inputs)).index
    r = require(returns, params.window + 5, "Distribution dynamics")
    k = max(int(params.tail_fraction * params.window), 2)
    dates, skew, kurt, tail, jb = [], [], [], [], []
    first_visible = int(np.searchsorted(returns.index, visible[0]))
    for end in range(max(params.window, first_visible), len(r) + 1, 5):
        chunk = r[end - params.window : end]
        dates.append(iso(returns.index[[end - 1]])[0])
        skew.append(float(stats.skew(chunk, bias=False)))
        kurt.append(float(stats.kurtosis(chunk, bias=False)))
        tail.append(hill(-chunk, k))
        jb.append(float(stats.jarque_bera(chunk).pvalue))
    return AnalysisOutput(
        "series",
        {
            "dates": dates,
            "skew": skew,
            "kurtosis": kurt,
            "hill_left": tail,
            "jb_p_value": jb,
            "window": params.window,
            "k": k,
        },
    )


# ── TS-023 volume & activity ─────────────────────────────────────────────────


@analysis(
    id="TS-023",
    version="1.0.0",
    title="Volume & activity",
    description=(
        "Volume trend (21 vs 252-session mean), correlation of log volume with |r|, abnormal "
        "volume z-score of log volume against the trailing 63 sessions (shifted, no "
        "look-ahead), and activity regimes by tercile."
    ),
    applies_to=PRICE_SUBJECTS,
    params=SeriesParams,
    result_kind="series",
    requires={"prices"},
    min_obs=100,
)
def volume_activity(inputs: AnalysisInputs, params: SeriesParams) -> AnalysisOutput:
    if not inputs.conventions.get("has_volume", True):
        raise InsufficientDataError("No meaningful volume for this asset type")
    full = _prices(inputs)
    volume = full["volume"].astype(float)
    volume = volume[volume > 0]
    require(volume, 100, "Volume analytics")
    log_v = np.log(volume)
    abs_r = full[_field(inputs)].pct_change().abs().reindex(volume.index)
    mean63 = log_v.rolling(63).mean().shift(1)
    std63 = log_v.rolling(63).std().shift(1)
    z = (log_v - mean63) / std63
    visible = _windowed(inputs, full).index.intersection(volume.index)
    terciles = np.quantile(log_v.loc[visible], [1 / 3, 2 / 3])
    valid = abs_r.notna()
    return AnalysisOutput(
        "series",
        {
            "dates": iso(visible),
            "volume": clean(volume.loc[visible]),
            "abnormal_z": clean(z.loc[visible]),
            "trend_ratio": float(volume.iloc[-21:].mean() / volume.iloc[-252:].mean()),
            "corr_volume_abs_return": float(np.corrcoef(log_v[valid], abs_r[valid])[0, 1]),
            "terciles": np.exp(terciles).tolist(),
            "regime": int(np.searchsorted(terciles, log_v.iloc[-1])),
            "abnormal_days": int((z.loc[visible] > 3).sum()),
        },
    )
