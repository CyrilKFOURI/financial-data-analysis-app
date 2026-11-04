"""Structural breaks, explosiveness and event sampling (TS-010, TS-011, TS-014; AFML 2.5, 17).

Explosiveness statistics run on log prices resampled to weekly closes when the range is long
(T ≤ 600 keeps the O(T²) window search fast). Every window's Dickey–Fuller regression
Δy_t = α + βy_{t−1} + ε (no augmentation lags, as in AFML 17.4) is computed from prefix sums.
"""

import math
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from pydantic import Field
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import breaks_cusumolsresid, recursive_olsresiduals

from fda_core.analyses.price import PRICE_SUBJECTS, _annualisation, _field, _prices, _windowed
from fda_core.analyses.ts_common import SeriesParams, clean, iso, make_test, require
from fda_core.registry import AnalysisInputs, AnalysisOutput, analysis

MAX_EXPLOSIVE_T = 600


# ── prefix-sum Dickey–Fuller ─────────────────────────────────────────────────


class _Sums:
    """Prefix sums of x = y_{t−1}, d = Δy_t (and of time) for O(1) window regressions."""

    def __init__(self, y: np.ndarray) -> None:
        x, d = y[:-1], np.diff(y)
        t = np.arange(len(y), dtype=float)
        self.cx = np.r_[0, np.cumsum(x)]
        self.cxx = np.r_[0, np.cumsum(x * x)]
        self.cd = np.r_[0, np.cumsum(d)]
        self.cxd = np.r_[0, np.cumsum(x * d)]
        self.cdd = np.r_[0, np.cumsum(d * d)]
        self.ct = np.r_[0, np.cumsum(t)]
        self.ctt = np.r_[0, np.cumsum(t * t)]
        self.cy = np.r_[0, np.cumsum(y)]
        self.cty = np.r_[0, np.cumsum(t * y)]
        self.cyy = np.r_[0, np.cumsum(y * y)]

    def df_t(self, starts: np.ndarray, end: int) -> np.ndarray:
        """t-statistics of β for windows of differences [start, end) (indices into Δy)."""
        n = (end - starts).astype(float)
        sx = self.cx[end] - self.cx[starts]
        sxx = self.cxx[end] - self.cxx[starts]
        sd = self.cd[end] - self.cd[starts]
        sxd = self.cxd[end] - self.cxd[starts]
        sdd = self.cdd[end] - self.cdd[starts]
        denom = n * sxx - sx * sx
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = (n * sxd - sx * sd) / denom
            alpha = (sd - beta * sx) / n
            rss = sdd - alpha * sd - beta * sxd
            se = np.sqrt(np.maximum(rss, 0) / (n - 2) * n / denom)
            t_stat: np.ndarray = beta / se
            return t_stat

    def trend_t(self, starts: np.ndarray, end: int) -> np.ndarray:
        """|t| of the slope of y on time over levels [start, end] (SMT, linear trend)."""
        e = end + 1
        n = (e - starts).astype(float)
        st, stt = self.ct[e] - self.ct[starts], self.ctt[e] - self.ctt[starts]
        sy, sty, syy = (
            self.cy[e] - self.cy[starts],
            self.cty[e] - self.cty[starts],
            self.cyy[e] - self.cyy[starts],
        )
        denom = n * stt - st * st
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = (n * sty - st * sy) / denom
            alpha = (sy - beta * st) / n
            rss = syy - alpha * sy - beta * sty
            se = np.sqrt(np.maximum(rss, 0) / (n - 2) * n / denom)
            t_abs: np.ndarray = np.abs(beta / se)
            return t_abs


def explosiveness_paths(
    y: np.ndarray, min_window: int, q: float = 0.95, full: bool = True
) -> dict[str, np.ndarray]:
    """For every end t: SADF (start 0), BSADF (sup over starts), QADF (q-quantile), CADF
    (mean above the quantile), SMT (sup |trend t|)."""
    sums = _Sums(y)
    n_diff = len(y) - 1
    ends = np.arange(min_window, n_diff + 1)
    out = {k: np.full(len(ends), np.nan) for k in ("sadf", "bsadf", "qadf", "cadf", "smt")}
    for i, end_ in enumerate(ends):
        end = int(end_)
        starts = np.arange(0, end - min_window + 1)
        stats_ = sums.df_t(starts, end)
        stats_ = stats_[np.isfinite(stats_)]
        if not len(stats_):
            continue
        out["sadf"][i] = sums.df_t(np.array([0]), end)[0]
        out["bsadf"][i] = stats_.max()
        if not full:  # null simulations only need SADF/BSADF
            continue
        quantile = np.quantile(stats_, q)
        out["qadf"][i] = quantile
        out["cadf"][i] = stats_[stats_ >= quantile].mean()
        trend = sums.trend_t(starts, end)
        out["smt"][i] = np.nanmax(trend) if np.isfinite(trend).any() else np.nan
    out["ends"] = ends
    return out


def simulated_critical_values(
    t: int, min_window: int, n_sims: int, seed: int, level: float = 0.95
) -> dict[str, Any]:
    """Random-walk nulls of the same length: per-end BSADF and overall SADF/GSADF quantiles."""
    rng = np.random.default_rng(seed)
    bsadf, sadf_sup, gsadf = [], [], []
    for _ in range(n_sims):
        walk = np.cumsum(rng.normal(0, 1, t))
        paths = explosiveness_paths(walk, min_window, full=False)
        bsadf.append(paths["bsadf"])
        sadf_sup.append(np.nanmax(paths["sadf"]))
        gsadf.append(np.nanmax(paths["bsadf"]))
    return {
        "bsadf": np.nanquantile(np.vstack(bsadf), level, axis=0),
        "sadf": float(np.quantile(sadf_sup, level)),
        "gsadf": float(np.quantile(gsadf, level)),
    }


def _explosive_series(inputs: AnalysisInputs) -> pd.Series:
    prices = _windowed(inputs, _prices(inputs))[_field(inputs)]
    log_price = np.log(prices)
    if len(log_price) > MAX_EXPLOSIVE_T:
        log_price = log_price.resample("W-FRI").last().dropna()
    return log_price.iloc[-MAX_EXPLOSIVE_T:]


class ExplosiveParams(SeriesParams):
    n_sims: int = Field(49, ge=19, le=499)
    seed: int = Field(17, ge=0, le=2**32 - 1)
    quantile: float = Field(0.95, ge=0.5, lt=1.0)


@analysis(
    id="TS-011",
    version="1.0.0",
    title="Explosiveness / bubbles",
    description=(
        "SADF, GSADF (Phillips–Shi–Yu), QADF and CADF (AFML-17) from Dickey–Fuller statistics of "
        "every window of log prices (minimum window 0.01 + 1.8/√T), SDFC Chow-type DF with a "
        "break in the AR root, and the SMT linear-trend statistic; critical values simulated "
        "from random walks of the same length; bubble periods where BSADF exceeds its "
        "95 % critical value."
    ),
    applies_to=PRICE_SUBJECTS,
    params=ExplosiveParams,
    result_kind="test",
    requires={"prices"},
    min_obs=60,
    budget_ms=20_000,
)
def explosiveness(inputs: AnalysisInputs, params: ExplosiveParams) -> AnalysisOutput:
    _annualisation(inputs)
    series = _explosive_series(inputs)
    y = require(series, 60, "Explosiveness tests")
    t = len(y)
    min_window = max(int((0.01 + 1.8 / math.sqrt(t)) * t), 10)
    paths = explosiveness_paths(y, min_window, params.quantile)
    crit = simulated_critical_values(t, min_window, params.n_sims, params.seed)
    end_dates = series.index[paths["ends"]]
    above = paths["bsadf"] > crit["bsadf"]
    episodes, start = [], None
    for i, flag in enumerate(above):
        if flag and start is None:
            start = i
        if (not flag or i == len(above) - 1) and start is not None:
            last = i if flag else i - 1
            if last - start + 1 >= 2:  # one-period exceedances are noise
                episodes.append(
                    {"start": iso(end_dates[[start]])[0], "end": iso(end_dates[[last]])[0]}
                )
            start = None
    # SDFC: Δy_t = δ·y_{t−1}·1[t ≥ τ] + ε, sup over τ in the middle 70 %.
    x, d = y[:-1], np.diff(y)
    sdfc = []
    for tau in range(int(0.15 * len(d)), int(0.85 * len(d))):
        xs, ds = x[tau:], d[tau:]
        delta = float(xs @ ds / (xs @ xs))
        resid = d.copy()
        resid[tau:] -= delta * xs
        sigma2 = float(resid @ resid) / (len(d) - 1)
        sdfc.append(delta / math.sqrt(sigma2 / float(xs @ xs)))
    sadf, gsadf = float(np.nanmax(paths["sadf"])), float(np.nanmax(paths["bsadf"]))
    return AnalysisOutput(
        "test",
        {
            "frequency": "W" if len(_windowed(inputs, _prices(inputs))) > MAX_EXPLOSIVE_T else "D",
            "min_window": min_window,
            "tests": [
                make_test(
                    "sadf",
                    "no explosive episode",
                    sadf,
                    None,
                    {"95%": crit["sadf"]},
                    reject=sadf > crit["sadf"],
                    reading_reject="explosive behaviour",
                    reading_keep="no explosiveness detected",
                ),
                make_test(
                    "gsadf",
                    "no explosive episode",
                    gsadf,
                    None,
                    {"95%": crit["gsadf"]},
                    reject=gsadf > crit["gsadf"],
                    reading_reject="explosive episode(s)",
                    reading_keep="no explosiveness detected",
                ),
                make_test("sdfc", "no change to an explosive root", max(sdfc) if sdfc else None),
            ],
            "dates": iso(end_dates),
            "bsadf": clean(paths["bsadf"]),
            "bsadf_critical": clean(crit["bsadf"]),
            "qadf": clean(paths["qadf"]),
            "cadf": clean(paths["cadf"]),
            "smt": clean(paths["smt"]),
            "episodes": episodes,
            "n_sims": params.n_sims,
        },
        ("Critical values from simulated random walks (finite sample); weekly data when long.",),
    )


# ── TS-010 structural breaks ─────────────────────────────────────────────────


def bai_perron(y: np.ndarray, max_breaks: int, min_size: int) -> dict[str, Any]:
    """Mean-shift breaks by dynamic programming on segment SSR; BIC picks the number."""
    n = len(y)
    c1, c2 = np.r_[0, np.cumsum(y)], np.r_[0, np.cumsum(y * y)]

    def ssr(i: int, j: int) -> float:  # segment [i, j)
        s = c1[j] - c1[i]
        return float(c2[j] - c2[i] - s * s / (j - i))

    inf = float("inf")
    cost = [[inf] * (n + 1) for _ in range(max_breaks + 1)]
    back = [[-1] * (n + 1) for _ in range(max_breaks + 1)]
    for j in range(min_size, n + 1):
        cost[0][j] = ssr(0, j)
    for m in range(1, max_breaks + 1):
        for j in range((m + 1) * min_size, n + 1):
            best, arg = inf, -1
            for i in range(m * min_size, j - min_size + 1):
                c = cost[m - 1][i] + ssr(i, j)
                if c < best:
                    best, arg = c, i
            cost[m][j], back[m][j] = best, arg
    results = []
    for m in range(max_breaks + 1):
        if not math.isfinite(cost[m][n]):
            continue
        bic = n * math.log(cost[m][n] / n) + (2 * m + 1) * math.log(n)
        cuts, j = [], n
        for k in range(m, 0, -1):
            j = back[k][j]
            cuts.append(j)
        results.append({"breaks": m, "bic": bic, "cuts": sorted(cuts)})
    return min(results, key=lambda r: r["bic"])


class BreakParams(SeriesParams):
    chow_date: date | None = None
    max_breaks: int = Field(3, ge=1, le=5)


@analysis(
    id="TS-010",
    version="1.0.0",
    title="Structural breaks — CUSUM family",
    description=(
        "Brown–Durbin–Evans CUSUM on recursive residuals of returns (sup statistic, p-value, 5 % "
        "bounds); Chu–Stinchcombe–White CUSUM on log levels with c = √(4.6 + ln(t − n)); Chow "
        "mean-shift F-test at a chosen date; Bai–Perron breaks in the mean of weekly returns and "
        "of weekly |returns| (volatility level), number chosen by BIC."
    ),
    applies_to=PRICE_SUBJECTS,
    params=BreakParams,
    result_kind="test",
    requires={"prices"},
    min_obs=100,
    budget_ms=15_000,
)
def structural_breaks(inputs: AnalysisInputs, params: BreakParams) -> AnalysisOutput:
    _annualisation(inputs)
    prices = _windowed(inputs, _prices(inputs))[_field(inputs)]
    returns = prices.pct_change().dropna()
    r = require(returns, 100, "Structural break tests")
    tests = []
    stat, p_value, crit = breaks_cusumolsresid((r - r.mean()), ddof=1)
    tests.append(
        make_test(
            "bde_cusum",
            "parameter stability",
            float(stat),
            float(p_value),
            {k: float(v) for k, v in crit},
            reading_reject="unstable mean",
            reading_keep="stable mean",
        )
    )
    rec = recursive_olsresiduals(OLS(r, np.ones((len(r), 1))).fit(), skip=10)
    cusum = np.asarray(rec[5])
    bounds = np.asarray(rec[6])
    # CSW on log levels, reference point = start of the range.
    y = np.log(prices.to_numpy(float))
    dy = np.diff(y)
    t_idx = np.arange(1, len(y))
    sigma_t = np.sqrt(np.cumsum(dy**2) / np.maximum(t_idx, 1))
    csw = (y[1:] - y[0]) / (sigma_t * np.sqrt(t_idx))
    csw_crit = np.sqrt(4.6 + np.log(t_idx))
    if params.chow_date is not None:
        split = int(np.searchsorted(returns.index, pd.Timestamp(params.chow_date)))
        if 10 <= split <= len(r) - 10:
            a, b = r[:split], r[split:]
            ssr_full = float(((r - r.mean()) ** 2).sum())
            ssr_split = float(((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum())
            f = (ssr_full - ssr_split) / (ssr_split / (len(r) - 2))
            tests.append(
                make_test(
                    "chow",
                    "no mean shift at the date",
                    f,
                    float(stats.f.sf(f, 1, len(r) - 2)),
                    reading_reject="mean shift at the date",
                    reading_keep="no shift",
                )
            )
    weekly = prices.resample("W-FRI").last().pct_change().dropna()
    w = weekly.to_numpy(float)
    min_size = max(int(0.1 * len(w)), 8)
    mean_breaks = bai_perron(w, params.max_breaks, min_size) if len(w) >= 3 * min_size else None
    vol_breaks = (
        bai_perron(np.abs(w - w.mean()), params.max_breaks, min_size)
        if len(w) >= 3 * min_size
        else None
    )

    def dates_of(result: dict[str, Any] | None) -> list[str]:
        return [] if result is None else iso(weekly.index[result["cuts"]])

    return AnalysisOutput(
        "test",
        {
            "tests": tests,
            "bde": {
                "dates": iso(returns.index[-len(cusum) :]),
                "cusum": clean(cusum),
                "bound_upper": clean(bounds[1]) if bounds.ndim == 2 else [],
                "bound_lower": clean(bounds[0]) if bounds.ndim == 2 else [],
            },
            "csw": {
                "dates": iso(prices.index[1:]),
                "stat": clean(csw),
                "critical": clean(csw_crit),
                "exceedances": int((np.abs(csw) > csw_crit).sum()),
            },
            "bai_perron": {
                "mean_breaks": dates_of(mean_breaks),
                "volatility_breaks": dates_of(vol_breaks),
                "min_segment_weeks": min_size,
            },
        },
    )


# ── TS-014 event-based sampling ──────────────────────────────────────────────


class CusumFilterParams(SeriesParams):
    k: float = Field(2.0, gt=0.1, le=10, description="Threshold in rolling σ of log returns")
    window: int = Field(50, ge=10, le=252)


def cusum_filter(log_price: pd.Series, threshold: pd.Series) -> list[pd.Timestamp]:
    """Symmetric CUSUM filter (AFML 2.5.2.1): reset after each event; h is known at t−1."""
    events: list[pd.Timestamp] = []
    s_pos = s_neg = 0.0
    diff = log_price.diff().dropna()
    for day, change in diff.items():
        h = threshold.get(day)
        if h is None or not np.isfinite(h):
            continue
        s_pos, s_neg = max(0.0, s_pos + change), min(0.0, s_neg + change)
        if s_neg < -h:
            s_neg = 0.0
            events.append(day)
        elif s_pos > h:
            s_pos = 0.0
            events.append(day)
    return events


@analysis(
    id="TS-014",
    version="1.0.0",
    title="Event-based sampling",
    description=(
        "Symmetric CUSUM filter on log prices with threshold h = k·σ (σ = std of log returns "
        "over the previous `window` sessions, shifted one bar: no look-ahead); events per "
        "year, inter-event duration and its coefficient of variation (clustering), compared "
        "with closes outside 20-day Bollinger bands."
    ),
    applies_to=PRICE_SUBJECTS,
    params=CusumFilterParams,
    result_kind="table",
    requires={"prices"},
    min_obs=60,
)
def event_sampling(inputs: AnalysisInputs, params: CusumFilterParams) -> AnalysisOutput:
    _annualisation(inputs)
    full = _prices(inputs)
    log_price = np.log(full[_field(inputs)])
    sigma = log_price.diff().rolling(params.window).std().shift(1)
    events = cusum_filter(log_price, params.k * sigma)
    visible = _windowed(inputs, full).index
    events = [e for e in events if e >= visible[0]]
    require(log_price.loc[visible], 60, "CUSUM filter")
    years = max((visible[-1] - visible[0]).days / 365.25, 1 / 12)
    gaps = np.diff([visible.get_loc(e) for e in events]) if len(events) > 1 else np.array([])
    close = full["close"]
    mid, sd = close.rolling(20).mean(), close.rolling(20).std(ddof=0)
    outside = ((close > mid + 2 * sd) | (close < mid - 2 * sd)).loc[visible]
    return AnalysisOutput(
        "table",
        {
            "events": iso(pd.DatetimeIndex(events)),
            "count": len(events),
            "per_year": len(events) / years,
            "mean_gap_sessions": float(gaps.mean()) if len(gaps) else None,
            "gap_cv": float(gaps.std() / gaps.mean()) if len(gaps) > 1 and gaps.mean() else None,
            "bollinger_events": int(outside.sum()),
            "k": params.k,
            "window": params.window,
        },
    )
