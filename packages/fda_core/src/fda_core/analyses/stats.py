"""Returns, statistics & distributions (ANALYTICS_CATALOG T-004…T-024; module M05).

All statistics use simple returns of the performance price (`conventions.price_field`) over
the context range, resampled by compounding for weekly/monthly frequencies. Annualisation
comes from the asset type for daily data (252/260/365) and is 52/12 for W/M.
"""

import math
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from fda_core.analyses.price import PRICE_SUBJECTS, _annualisation, _field, _prices, _windowed
from fda_core.registry import AnalysisInputs, AnalysisOutput, InsufficientDataError, analysis

Frequency = Literal["D", "W", "M"]
PERIODS_PER_YEAR = {"W": 52, "M": 12, "Q": 4}


class FreqParams(BaseModel):
    freq: Frequency = "D"
    log_returns: bool = False


# ── helpers ──────────────────────────────────────────────────────────────────


def resample_returns(prices: pd.Series, freq: str, calendar: str = "exchange") -> pd.Series:
    """Compounded returns per period: weekly ends Friday (Sunday for 24/7 assets), monthly ME."""
    if freq == "D":
        return prices.pct_change().dropna()
    rule = {"W": "W-SUN" if calendar == "24/7" else "W-FRI", "M": "ME", "Q": "QE", "Y": "YE"}[freq]
    closes = prices.resample(rule).last().dropna()
    return closes.pct_change().dropna()


def _returns(inputs: AnalysisInputs, freq: str = "D", log: bool = False) -> tuple[pd.Series, int]:
    prices = _windowed(inputs, _prices(inputs))[_field(inputs)]
    factor = _annualisation(inputs) if freq == "D" else PERIODS_PER_YEAR[freq]
    returns = resample_returns(prices, freq, str(inputs.conventions.get("calendar", "exchange")))
    if log:
        returns = np.log1p(returns)
    return returns, factor


def _need(returns: pd.Series, minimum: int, what: str) -> np.ndarray:
    if len(returns) < minimum:
        raise InsufficientDataError(f"{what} needs ≥ {minimum} returns, got {len(returns)}")
    values: np.ndarray = returns.to_numpy(float)
    return values


def fit_student_t(r: np.ndarray) -> tuple[float, float, float]:
    """MLE on standardised data (the optimiser struggles at daily-return scales), ν > 2."""
    centre, spread = float(r.mean()), float(r.std(ddof=1))
    z = (r - centre) / spread
    nu, loc, scale = stats.t.fit(z)
    if nu <= 2:
        nu, loc, scale = stats.t.fit(z, f0=2.01)
    return float(nu), centre + spread * float(loc), spread * float(scale)


def _f(value: float | np.floating[Any] | None) -> float | None:
    return None if value is None or not np.isfinite(value) else float(value)


def _iso(index: pd.Index) -> list[str]:
    return [pd.Timestamp(d).date().isoformat() for d in index]


def _sample(returns: pd.Series, freq: str) -> dict[str, Any]:
    return {
        "n": len(returns),
        "freq": freq,
        "from": _iso(returns.index[:1])[0],
        "to": _iso(returns.index[-1:])[0],
    }


# ── T-004 means, T-006 volatility drag ───────────────────────────────────────


def means(r: np.ndarray, trim: float = 0.05) -> dict[str, float | None]:
    gross = 1 + r
    return {
        "arithmetic": float(r.mean()),
        "geometric": float(np.exp(np.log(gross).mean()) - 1) if (gross > 0).all() else None,
        "harmonic": float(stats.hmean(gross) - 1) if (gross > 0).all() else None,
        "median": float(np.median(r)),
        "trimmed": float(stats.trim_mean(r, trim)),
    }


@analysis(
    id="T-004",
    version="1.0.0",
    title="Means",
    description=(
        "Arithmetic, geometric (Π(1+r))^(1/N) − 1, harmonic, median and 5 % trimmed mean of "
        "periodic returns, with arithmetic and geometric annualised."
    ),
    applies_to=PRICE_SUBJECTS,
    params=FreqParams,
    result_kind="table",
    requires={"prices"},
    min_obs=10,
)
def central_tendency(inputs: AnalysisInputs, params: FreqParams) -> AnalysisOutput:
    returns, factor = _returns(inputs, params.freq)
    r = _need(returns, 10, "Means")
    m = means(r)
    geo = m["geometric"]
    return AnalysisOutput(
        "table",
        {
            **m,
            "arithmetic_annualised": m["arithmetic"] * factor
            if m["arithmetic"] is not None
            else None,
            "geometric_annualised": (1 + geo) ** factor - 1 if geo is not None else None,
            "sample": _sample(returns, params.freq),
        },
    )


@analysis(
    id="T-006",
    version="1.0.0",
    title="Volatility drag",
    description=(
        "Annualised arithmetic minus geometric mean, compared with the ½σ² approximation; the "
        "gap between them is the approximation error."
    ),
    applies_to=PRICE_SUBJECTS,
    params=FreqParams,
    result_kind="table",
    requires={"prices"},
    min_obs=20,
)
def volatility_drag(inputs: AnalysisInputs, params: FreqParams) -> AnalysisOutput:
    returns, factor = _returns(inputs, params.freq)
    r = _need(returns, 20, "Volatility drag")
    m = means(r)
    if m["geometric"] is None:
        raise InsufficientDataError("A −100 % return makes the geometric mean undefined")
    arith = m["arithmetic"] * factor  # type: ignore[operator]
    geo = factor * math.log1p(m["geometric"])  # continuously compounded annual growth
    drag = arith - geo
    approx = 0.5 * float(np.var(r, ddof=1)) * factor
    return AnalysisOutput(
        "table",
        {
            "arithmetic_annualised": arith,
            "log_growth_annualised": geo,
            "drag": drag,
            "half_sigma_squared": approx,
            "approximation_error": drag - approx,
            "sample": _sample(returns, params.freq),
        },
    )


# ── T-008 dispersion ─────────────────────────────────────────────────────────


class DispersionParams(FreqParams):
    mar: float = Field(0.0, ge=-1, le=1, description="Minimum acceptable return per period")


@analysis(
    id="T-008",
    version="1.0.0",
    title="Dispersion",
    description=(
        "Std, variance, mean and median absolute deviation, IQR, range, semi-variance "
        "(1/N)Σmin(r − r̄, 0)², semi-deviation, downside deviation vs MAR and the "
        "downside/upside deviation ratio; std annualised by √A."
    ),
    applies_to=PRICE_SUBJECTS,
    params=DispersionParams,
    result_kind="table",
    requires={"prices"},
    min_obs=10,
)
def dispersion(inputs: AnalysisInputs, params: DispersionParams) -> AnalysisOutput:
    returns, factor = _returns(inputs, params.freq)
    r = _need(returns, 10, "Dispersion")
    mean = r.mean()
    below = np.minimum(r - mean, 0)
    above = np.maximum(r - mean, 0)
    downside = np.minimum(r - params.mar, 0)
    q1, q3 = np.percentile(r, [25, 75])
    semi_var = float((below**2).mean())
    up_dev = float(np.sqrt((above**2).mean()))
    return AnalysisOutput(
        "table",
        {
            "std": float(r.std(ddof=1)),
            "std_annualised": float(r.std(ddof=1) * math.sqrt(factor)),
            "variance": float(r.var(ddof=1)),
            "mean_abs_deviation": float(np.abs(r - mean).mean()),
            "median_abs_deviation": float(stats.median_abs_deviation(r)),
            "iqr": float(q3 - q1),
            "range": float(r.max() - r.min()),
            "min": float(r.min()),
            "max": float(r.max()),
            "semi_variance": semi_var,
            "semi_deviation": math.sqrt(semi_var),
            "semi_deviation_annualised": math.sqrt(semi_var * factor),
            "downside_deviation_mar": float(np.sqrt((downside**2).mean())),
            "downside_upside_ratio": math.sqrt(semi_var) / up_dev if up_dev else None,
            "mar": params.mar,
            "sample": _sample(returns, params.freq),
        },
    )


# ── T-023 up/down statistics ─────────────────────────────────────────────────


def _streaks(signs: np.ndarray, value: int) -> list[int]:
    runs, current = [], 0
    for s in signs:
        if s == value:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


@analysis(
    id="T-023",
    version="1.0.0",
    title="Up/down statistics",
    description=(
        "Share of positive periods, average up and down move, gain/loss ratio, longest and "
        "average up/down streaks (zero returns break streaks)."
    ),
    applies_to=PRICE_SUBJECTS,
    params=FreqParams,
    result_kind="table",
    requires={"prices"},
    min_obs=10,
)
def up_down(inputs: AnalysisInputs, params: FreqParams) -> AnalysisOutput:
    returns, _ = _returns(inputs, params.freq)
    r = _need(returns, 10, "Up/down statistics")
    ups, downs = r[r > 0], r[r < 0]
    signs = np.sign(r).astype(int)
    up_runs, down_runs = _streaks(signs, 1), _streaks(signs, -1)
    avg_up = float(ups.mean()) if len(ups) else None
    avg_down = float(downs.mean()) if len(downs) else None
    return AnalysisOutput(
        "table",
        {
            "pct_up": len(ups) / len(r),
            "pct_down": len(downs) / len(r),
            "avg_up": avg_up,
            "avg_down": avg_down,
            "gain_loss_ratio": abs(avg_up / avg_down) if avg_up and avg_down else None,
            "longest_up_streak": max(up_runs, default=0),
            "longest_down_streak": max(down_runs, default=0),
            "avg_up_streak": float(np.mean(up_runs)) if up_runs else None,
            "avg_down_streak": float(np.mean(down_runs)) if down_runs else None,
            "sample": _sample(returns, params.freq),
        },
    )


# ── T-014 distribution, T-015 moments, T-016 normality, T-017 tails ──────────


@analysis(
    id="T-014",
    version="1.0.0",
    title="Return distribution",
    description=(
        "Histogram with Freedman–Diaconis bins, Normal (MLE) and Student-t (MLE, ν > 2) fits, "
        "and a Gaussian KDE with Scott bandwidth, evaluated on a common grid."
    ),
    applies_to=PRICE_SUBJECTS,
    params=FreqParams,
    result_kind="series",
    requires={"prices"},
    min_obs=30,
    budget_ms=4_000,
)
def distribution(inputs: AnalysisInputs, params: FreqParams) -> AnalysisOutput:
    returns, _ = _returns(inputs, params.freq, params.log_returns)
    r = _need(returns, 30, "A distribution")
    edges = np.histogram_bin_edges(r, bins="fd")
    if len(edges) > 201:  # FD explodes on fat tails: cap the bin count
        edges = np.histogram_bin_edges(r, bins=200)
    density, edges = np.histogram(r, bins=edges, density=True)
    grid = np.linspace(r.min(), r.max(), 200)
    mu, sigma = stats.norm.fit(r)
    nu, loc, scale = fit_student_t(r)
    kde = stats.gaussian_kde(r, bw_method="scott")
    return AnalysisOutput(
        "series",
        {
            "bins": {"edges": edges.tolist(), "density": density.tolist()},
            "grid": grid.tolist(),
            "normal": {"mu": mu, "sigma": sigma, "pdf": stats.norm.pdf(grid, mu, sigma).tolist()},
            "student_t": {
                "nu": nu,
                "loc": loc,
                "scale": scale,
                "implied_excess_kurtosis": 6 / (nu - 4) if nu > 4 else None,
                "pdf": stats.t.pdf(grid, nu, loc, scale).tolist(),
            },
            "kde": kde(grid).tolist(),
            "log_returns": params.log_returns,
            "sample": _sample(returns, params.freq),
        },
    )


@analysis(
    id="T-015",
    version="1.0.0",
    title="Moments",
    description=(
        "Bias-corrected skewness G1 and excess kurtosis G2 with standard errors √(6/n) and "
        "√(24/n); standardised 5th and 6th moments."
    ),
    applies_to=PRICE_SUBJECTS,
    params=FreqParams,
    result_kind="table",
    requires={"prices"},
    min_obs=30,
)
def moments(inputs: AnalysisInputs, params: FreqParams) -> AnalysisOutput:
    returns, _ = _returns(inputs, params.freq, params.log_returns)
    r = _need(returns, 30, "Moments")
    n = len(r)
    z = (r - r.mean()) / r.std(ddof=0)
    return AnalysisOutput(
        "table",
        {
            "skewness": float(stats.skew(r, bias=False)),
            "skewness_se": math.sqrt(6 / n),
            "excess_kurtosis": float(stats.kurtosis(r, fisher=True, bias=False)),
            "excess_kurtosis_se": math.sqrt(24 / n),
            "moment_5": float((z**5).mean()),
            "moment_6": float((z**6).mean()),
            "sample": _sample(returns, params.freq),
        },
    )


class NormalityParams(FreqParams):
    alpha: float = Field(0.05, gt=0, lt=0.5)


@analysis(
    id="T-016",
    version="1.0.0",
    title="Normality tests",
    description=(
        "Jarque–Bera, Shapiro–Wilk (N ≤ 5 000), Anderson–Darling and D'Agostino K² with "
        "decisions at α, plus QQ points against the Normal and the fitted Student-t."
    ),
    applies_to=PRICE_SUBJECTS,
    params=NormalityParams,
    result_kind="test",
    requires={"prices"},
    min_obs=30,
    budget_ms=4_000,
)
def normality(inputs: AnalysisInputs, params: NormalityParams) -> AnalysisOutput:
    returns, _ = _returns(inputs, params.freq, params.log_returns)
    r = _need(returns, 30, "Normality tests")
    tests: list[dict[str, Any]] = []
    jb = stats.jarque_bera(r)
    tests.append(
        {"test": "jarque_bera", "statistic": float(jb.statistic), "p_value": float(jb.pvalue)}
    )
    if len(r) <= 5_000:
        sw = stats.shapiro(r)
        tests.append(
            {"test": "shapiro_wilk", "statistic": float(sw.statistic), "p_value": float(sw.pvalue)}
        )
    ad = stats.anderson(r, dist="norm")
    level = int(params.alpha * 100)
    levels = list(ad.significance_level)
    critical = float(ad.critical_values[levels.index(level)]) if level in levels else None
    tests.append(
        {
            "test": "anderson_darling",
            "statistic": float(ad.statistic),
            "p_value": None,
            "critical_value": critical,
        }
    )
    k2 = stats.normaltest(r)
    tests.append(
        {"test": "dagostino_k2", "statistic": float(k2.statistic), "p_value": float(k2.pvalue)}
    )
    for test in tests:
        if test["p_value"] is not None:
            test["reject_normality"] = test["p_value"] < params.alpha
        else:
            test["reject_normality"] = (
                test["statistic"] > test["critical_value"] if test["critical_value"] else None
            )
    ordered = np.sort((r - r.mean()) / r.std(ddof=1))
    probs = (np.arange(1, len(r) + 1) - 0.5) / len(r)
    nu, _, _ = fit_student_t(r)
    t_q = stats.t.ppf(probs, nu) / math.sqrt(nu / (nu - 2))  # unit-variance t
    step = max(len(r) // 400, 1)  # ≤ 400 points on the wire
    warnings = (
        ("Large sample: tiny deviations are 'significant'; read effect sizes (kurtosis, tails).",)
        if len(r) > 2_000
        else ()
    )
    return AnalysisOutput(
        "test",
        {
            "alpha": params.alpha,
            "tests": tests,
            "qq": {
                "empirical": ordered[::step].tolist(),
                "normal": stats.norm.ppf(probs)[::step].tolist(),
                "student_t": t_q[::step].tolist(),
                "nu": nu,
            },
            "sample": _sample(returns, params.freq),
        },
        warnings,
    )


@analysis(
    id="T-017",
    version="1.0.0",
    title="Tail frequency",
    description=(
        "Observed share of |z| beyond 2, 3, 4, 5 σ (full-sample mean and σ) against the "
        "Normal-expected share; the ratio is the fat-tail multiplier. Largest moves listed."
    ),
    applies_to=PRICE_SUBJECTS,
    params=FreqParams,
    result_kind="table",
    requires={"prices"},
    min_obs=60,
)
def tail_frequency(inputs: AnalysisInputs, params: FreqParams) -> AnalysisOutput:
    returns, _ = _returns(inputs, params.freq)
    r = _need(returns, 60, "Tail frequency")
    z = (r - r.mean()) / r.std(ddof=1)
    rows = []
    for k in (2, 3, 4, 5):
        observed = float((np.abs(z) > k).mean())
        expected = float(2 * stats.norm.sf(k))
        rows.append(
            {
                "k": k,
                "count": int((np.abs(z) > k).sum()),
                "observed": observed,
                "expected": expected,
                "multiplier": observed / expected,
            }
        )
    order = np.argsort(-np.abs(z))[:10]
    largest = [
        {"date": _iso(returns.index[i : i + 1])[0], "return": float(r[i]), "z": float(z[i])}
        for i in order
    ]
    return AnalysisOutput(
        "table", {"rows": rows, "largest": largest, "sample": _sample(returns, params.freq)}
    )


# ── T-018 calendar, T-019 quantiles ──────────────────────────────────────────


@analysis(
    id="T-018",
    version="1.0.0",
    title="Calendar returns",
    description=(
        "Compounded monthly returns as a year × month grid, annual returns (partial years "
        "flagged), and best/worst day, week, month, quarter and year over full history."
    ),
    applies_to=PRICE_SUBJECTS,
    params=FreqParams,
    result_kind="matrix",
    requires={"prices"},
    min_obs=2,
)
def calendar_returns(inputs: AnalysisInputs, params: FreqParams) -> AnalysisOutput:
    _annualisation(inputs)
    prices = _prices(inputs)[_field(inputs)]
    calendar = str(inputs.conventions.get("calendar", "exchange"))
    monthly = resample_returns(prices, "M", calendar)
    if monthly.empty:
        raise InsufficientDataError("Calendar returns need at least two month-ends")
    years = sorted({d.year for d in monthly.index})
    grid: dict[str, list[float | None]] = {str(y): [None] * 12 for y in years}
    for day, value in monthly.items():
        grid[str(day.year)][day.month - 1] = float(value)
    first, last = pd.Timestamp(prices.index[0]), pd.Timestamp(prices.index[-1])
    annual = []
    for y in years:
        chunk = prices.loc[(prices.index.year == y)]
        previous = prices.loc[prices.index.year < y]
        start = float(previous.iloc[-1]) if len(previous) else float(chunk.iloc[0])
        annual.append(
            {
                "year": y,
                "return": float(chunk.iloc[-1]) / start - 1,
                "partial": (y == first.year and len(previous) == 0) or y == last.year,
            }
        )
    extremes = {}
    for label, freq in (
        ("day", "D"),
        ("week", "W"),
        ("month", "M"),
        ("quarter", "Q"),
        ("year", "Y"),
    ):
        series = resample_returns(prices, freq, calendar)
        if series.empty:
            continue
        extremes[label] = {
            "best": {"date": _iso(pd.Index([series.idxmax()]))[0], "return": float(series.max())},
            "worst": {"date": _iso(pd.Index([series.idxmin()]))[0], "return": float(series.min())},
        }
    return AnalysisOutput(
        "matrix",
        {"years": [str(y) for y in years], "months": grid, "annual": annual, "extremes": extremes},
    )


@analysis(
    id="T-019",
    version="1.0.0",
    title="Return quantiles",
    description="Quantiles 1/5/25/50/75/95/99 % of compounded returns per horizon D, W, M, Q.",
    applies_to=PRICE_SUBJECTS,
    params=FreqParams,
    result_kind="table",
    requires={"prices"},
    min_obs=30,
)
def return_quantiles(inputs: AnalysisInputs, params: FreqParams) -> AnalysisOutput:
    _annualisation(inputs)
    prices = _windowed(inputs, _prices(inputs))[_field(inputs)]
    calendar = str(inputs.conventions.get("calendar", "exchange"))
    levels = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    rows: list[dict[str, Any]] = []
    for horizon in ("D", "W", "M", "Q"):
        r = resample_returns(prices, horizon, calendar)
        if len(r) < 8:
            rows.append(
                {
                    "horizon": horizon,
                    "n": len(r),
                    "quantiles": None,
                    "reason": "fewer than 8 periods",
                }
            )
            continue
        rows.append(
            {
                "horizon": horizon,
                "n": len(r),
                "quantiles": dict(
                    zip([str(q) for q in levels], np.quantile(r, levels).tolist(), strict=True)
                ),
            }
        )
    return AnalysisOutput("table", {"rows": rows})


# ── T-009 rolling statistics, T-020 estimate stability ───────────────────────


class RollingParams(BaseModel):
    windows: list[int] = Field(default=[21, 63, 126, 252], min_length=1, max_length=4)


@analysis(
    id="T-009",
    version="1.0.0",
    title="Rolling statistics",
    description=(
        "Trailing-window annualised mean and volatility, skewness and excess kurtosis of "
        "daily returns; values are aligned to the window's last day (no look-ahead)."
    ),
    applies_to=PRICE_SUBJECTS,
    params=RollingParams,
    result_kind="series",
    requires={"prices"},
    min_obs=22,
)
def rolling_statistics(inputs: AnalysisInputs, params: RollingParams) -> AnalysisOutput:
    factor = _annualisation(inputs)
    full = _prices(inputs)[_field(inputs)].pct_change().dropna()
    visible = _windowed(inputs, _prices(inputs)).index
    visible = visible[visible.isin(full.index)]
    out: dict[str, Any] = {"dates": _iso(visible), "windows": {}}
    for window in params.windows:
        if not 5 <= window <= 1_000 or len(full) < window:
            continue
        roll = full.rolling(window)
        out["windows"][str(window)] = {
            "mean": [_f(v) for v in (roll.mean() * factor).loc[visible]],
            "vol": [_f(v) for v in (roll.std() * math.sqrt(factor)).loc[visible]],
            "skew": [_f(v) for v in roll.skew().loc[visible]],
            "kurtosis": [_f(v) for v in roll.kurt().loc[visible]],
        }
    if not out["windows"]:
        raise InsufficientDataError("History shorter than every rolling window")
    return AnalysisOutput("series", out)


def stationary_bootstrap(
    r: np.ndarray, statistic: Any, n_boot: int, block: float, seed: int
) -> np.ndarray:
    """Politis–Romano stationary bootstrap with mean block length `block` (vectorised).

    Each resample starts a new block with probability 1/block at every step; inside a block
    indices advance by one, wrapping around the sample.
    """
    rng = np.random.default_rng(seed)
    n = len(r)
    jumps = rng.random((n_boot, n)) < 1 / block
    jumps[:, 0] = True
    starts = rng.integers(n, size=(n_boot, n))
    positions = np.broadcast_to(np.arange(n), (n_boot, n))
    last_jump = np.maximum.accumulate(np.where(jumps, positions, 0), axis=1)
    idx = (np.take_along_axis(starts, last_jump, axis=1) + positions - last_jump) % n
    samples = r[idx]
    return np.array([statistic(row) for row in samples])


class StabilityParams(BaseModel):
    window: int = Field(252, ge=21, le=1_260)
    n_boot: int = Field(1_000, ge=200, le=5_000)
    rf: float = Field(0.0, ge=-0.05, le=0.5, description="Annual risk-free rate")
    seed: int = Field(11, ge=0, le=2**32 - 1)


@analysis(
    id="T-020",
    version="1.0.0",
    title="Estimate instability",
    description=(
        "Rolling annualised Sharpe with the Lo (2002) iid standard-error band "
        "SE = √((1 + ½SR²)/n) · √A, and stationary-block-bootstrap 95 % intervals of the "
        "annualised mean and Sharpe (mean block length n^(1/3))."
    ),
    applies_to=PRICE_SUBJECTS,
    params=StabilityParams,
    result_kind="series",
    requires={"prices"},
    min_obs=60,
    budget_ms=8_000,
)
def estimate_instability(inputs: AnalysisInputs, params: StabilityParams) -> AnalysisOutput:
    factor = _annualisation(inputs)
    returns = _windowed(inputs, _prices(inputs))[_field(inputs)].pct_change().dropna()
    r = _need(returns, 60, "Estimate stability")
    excess = r - params.rf / factor

    def sharpe(x: np.ndarray) -> float:
        sd = x.std(ddof=1)
        return float(x.mean() / sd * math.sqrt(factor)) if sd else float("nan")

    full = _prices(inputs)[_field(inputs)].pct_change().dropna() - params.rf / factor
    roll = full.rolling(params.window)
    rolling_sr = roll.mean() / roll.std() * math.sqrt(factor)
    per_period = rolling_sr / math.sqrt(factor)
    se = np.sqrt((1 + 0.5 * per_period**2) / params.window) * math.sqrt(factor)
    visible = returns.index
    block = max(len(r) ** (1 / 3), 1.0)
    boot_mean = stationary_bootstrap(
        excess, lambda x: x.mean() * factor, params.n_boot, block, params.seed
    )
    boot_sr = stationary_bootstrap(excess, sharpe, params.n_boot, block, params.seed + 1)
    return AnalysisOutput(
        "series",
        {
            "dates": _iso(visible),
            "rolling_sharpe": [_f(v) for v in rolling_sr.reindex(visible)],
            "rolling_se": [_f(v) for v in se.reindex(visible)],
            "sharpe": sharpe(excess),
            "sharpe_ci": np.nanpercentile(boot_sr, [2.5, 97.5]).tolist(),
            "mean_annualised": float(excess.mean() * factor),
            "mean_ci": np.percentile(boot_mean, [2.5, 97.5]).tolist(),
            "block_length": block,
            "n_boot": params.n_boot,
            "sample": _sample(returns, "D"),
        },
    )


# ── T-021 seasonality ────────────────────────────────────────────────────────


def benjamini_hochberg(p_values: list[float], q: float) -> list[tuple[float, bool]]:
    """Adjusted p-values (step-up) and rejection at FDR level q."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, p_values[i] * m / rank)
        adjusted[i] = running
    return [(float(a), bool(a <= q)) for a in adjusted]


class SeasonalityParams(BaseModel):
    kind: Literal["month", "weekday", "turn_of_month"] = "month"
    fdr: float = Field(0.10, gt=0, lt=0.5)


@analysis(
    id="T-021",
    version="1.0.0",
    title="Seasonality",
    description=(
        "Mean and median daily return per bucket (month, weekday, turn-of-month −1…+3), "
        "Welch t-test of the bucket against all other days, Benjamini–Hochberg adjusted "
        "p-values at the chosen FDR, and a first-half/second-half sign-stability check."
    ),
    applies_to=PRICE_SUBJECTS,
    params=SeasonalityParams,
    result_kind="test",
    requires={"prices"},
    min_obs=250,
)
def seasonality(inputs: AnalysisInputs, params: SeasonalityParams) -> AnalysisOutput:
    _annualisation(inputs)
    returns = _windowed(inputs, _prices(inputs))[_field(inputs)].pct_change().dropna()
    _need(returns, 250, "Seasonality")
    index = pd.DatetimeIndex(returns.index)
    if params.kind == "month":
        labels = pd.Series(index.month, index=index).map(lambda m: f"M{m:02d}")
    elif params.kind == "weekday":
        labels = pd.Series(index.dayofweek, index=index).map(
            lambda d: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d]
        )
    else:
        month = index.to_period("M")
        position = pd.Series(range(len(index)), index=index).groupby(month).cumcount()
        size = pd.Series(1, index=index).groupby(month).transform("size")
        from_end = size - position - 1
        labels = pd.Series("other", index=index)
        labels[from_end == 0] = "TOM-1"
        for k in range(4):
            labels[position == k] = f"TOM+{k}"
    half = index < index[len(index) // 2]
    rows, p_values = [], []
    for label in sorted(set(labels)):
        inside = returns[labels == label].to_numpy(float)
        outside = returns[labels != label].to_numpy(float)
        if len(inside) < 5 or len(outside) < 5:
            continue
        test = stats.ttest_ind(inside, outside, equal_var=False)
        first = returns[(labels == label) & half].mean()
        second = returns[(labels == label) & ~half].mean()
        rows.append(
            {
                "bucket": label,
                "n": len(inside),
                "mean": float(inside.mean()),
                "median": float(np.median(inside)),
                "t_stat": float(test.statistic),
                "p_value": float(test.pvalue),
                "stable_sign": bool(np.sign(first) == np.sign(second)),
            }
        )
        p_values.append(float(test.pvalue))
    for row, (adjusted, significant) in zip(
        rows, benjamini_hochberg(p_values, params.fdr), strict=True
    ):
        row["p_adjusted"] = adjusted
        row["significant"] = significant
    return AnalysisOutput(
        "test",
        {
            "kind": params.kind,
            "fdr": params.fdr,
            "hypotheses": len(rows),
            "rows": rows,
            "sample": _sample(returns, "D"),
        },
        (
            "Seasonality is a classic data-snooping trap: trust only BH-significant, "
            "sign-stable buckets.",
        ),
    )


# ── T-024 best/worst days ────────────────────────────────────────────────────


@analysis(
    id="T-024",
    version="1.0.0",
    title="Return contribution by period",
    description=(
        "Total return of the range versus the same path without its best (or worst) 5, 10 "
        "and 20 days, and the share of log growth earned on the top days."
    ),
    applies_to=PRICE_SUBJECTS,
    params=FreqParams,
    result_kind="table",
    requires={"prices"},
    min_obs=60,
)
def best_days(inputs: AnalysisInputs, params: FreqParams) -> AnalysisOutput:
    returns, _ = _returns(inputs, "D")
    r = _need(returns, 60, "Best/worst days")
    total = float(np.prod(1 + r) - 1)
    log_total = float(np.log1p(r).sum())
    ranked = np.sort(r)
    rows = []
    for k in (5, 10, 20):
        best, worst = ranked[-k:], ranked[:k]
        rows.append(
            {
                "days": k,
                "without_best": float(np.prod(1 + ranked[:-k]) - 1),
                "without_worst": float(np.prod(1 + ranked[k:]) - 1),
                "share_of_growth_best": float(np.log1p(best).sum() / log_total)
                if log_total
                else None,
                "share_of_growth_worst": float(np.log1p(worst).sum() / log_total)
                if log_total
                else None,
            }
        )
    return AnalysisOutput("table", {"total": total, "rows": rows, "sample": _sample(returns, "D")})


# ── T-010 OHLC volatility estimators ─────────────────────────────────────────


class OhlcVolParams(BaseModel):
    window: int = Field(21, ge=5, le=252)


def ohlc_estimators(frame: pd.DataFrame, window: int, factor: int) -> dict[str, pd.Series]:
    """Annualised rolling estimators (log prices). YZ uses k = 0.34 / (1.34 + (n+1)/(n−1))."""
    o, h, low, c = (np.log(frame[k]) for k in ("open", "high", "low", "close"))
    prev_c = c.shift(1)
    hl, co = h - low, c - o
    parkinson = (hl**2 / (4 * math.log(2))).rolling(window).mean()
    garman = (0.5 * hl**2 - (2 * math.log(2) - 1) * co**2).rolling(window).mean()
    rs_term = (h - c) * (h - o) + (low - c) * (low - o)
    rogers = rs_term.rolling(window).mean()
    overnight = o - prev_c
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    yang = overnight.rolling(window).var() + k * co.rolling(window).var() + (1 - k) * rogers
    close = (c - prev_c).rolling(window).var()
    return {
        name: np.sqrt(series * factor)
        for name, series in {
            "close_to_close": close,
            "parkinson": parkinson,
            "garman_klass": garman,
            "rogers_satchell": rogers,
            "yang_zhang": yang,
        }.items()
    }


@analysis(
    id="T-010",
    version="1.0.0",
    title="OHLC volatility estimators",
    description=(
        "Rolling annualised Parkinson, Garman–Klass, Rogers–Satchell and Yang–Zhang "
        "estimators against close-to-close, with each estimator's mean over the range."
    ),
    applies_to=PRICE_SUBJECTS,
    params=OhlcVolParams,
    result_kind="series",
    requires={"prices"},
    min_obs=30,
)
def ohlc_volatility(inputs: AnalysisInputs, params: OhlcVolParams) -> AnalysisOutput:
    factor = _annualisation(inputs)
    full = _prices(inputs)
    if len(full) < params.window + 2:
        raise InsufficientDataError(f"OHLC estimators need ≥ {params.window + 2} sessions")
    estimators = ohlc_estimators(full, params.window, factor)
    visible = _windowed(inputs, full).index
    return AnalysisOutput(
        "series",
        {
            "dates": _iso(visible),
            "lines": {name: [_f(v) for v in s.loc[visible]] for name, s in estimators.items()},
            "means": {name: _f(s.loc[visible].mean()) for name, s in estimators.items()},
            "window": params.window,
        },
        ("Open/high/low of adjusted providers are split-adjusted, not dividend-adjusted.",),
    )
