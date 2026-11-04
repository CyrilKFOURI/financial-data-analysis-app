"""Volatility and dynamic models (TS-003 GARCH family, TS-007 ARIMA, TS-009 OU / mean
reversion, TS-012 regimes, TS-019 Kalman filtering). In-sample description only: forward
forecasts used for decisions belong to V5."""

import math
import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
from arch import arch_model
from pydantic import Field
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tools import add_constant
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import arma2ma
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.statespace.structural import UnobservedComponents

from fda_core.analyses.price import PRICE_SUBJECTS, _annualisation, _field, _prices, _windowed
from fda_core.analyses.ts_common import (
    SeriesParams,
    clean,
    iso,
    make_test,
    require,
    sample_info,
    select_series,
)
from fda_core.registry import AnalysisInputs, AnalysisOutput, InsufficientDataError, analysis

# ── TS-003 volatility clustering & GARCH family ──────────────────────────────


class GarchParams(SeriesParams):
    model: Literal["garch", "gjr", "egarch"] = "garch"
    dist: Literal["normal", "t", "skewt"] = "t"
    horizon: int = Field(60, ge=5, le=252)


def news_impact(params: dict[str, float], model: str, sigma2_bar: float) -> dict[str, list[float]]:
    """Next-period variance as a function of today's shock ε (lagged variance at its mean)."""
    eps = np.linspace(-4, 4, 81) * math.sqrt(sigma2_bar)
    omega, alpha, beta = params["omega"], params["alpha[1]"], params.get("beta[1]", 0.0)
    gamma = params.get("gamma[1]", 0.0)
    if model == "egarch":
        z = eps / math.sqrt(sigma2_bar)
        log_s2 = (
            omega
            + alpha * (np.abs(z) - math.sqrt(2 / math.pi))
            + gamma * z
            + beta * math.log(sigma2_bar)
        )
        variance = np.exp(log_s2)
    else:
        variance = omega + (alpha + gamma * (eps < 0)) * eps**2 + beta * sigma2_bar
    return {"shock": (eps / 100).tolist(), "variance": (variance / 1e4).tolist()}


@analysis(
    id="TS-003",
    version="1.0.0",
    title="Volatility clustering & models",
    description=(
        "ARCH-LM test; GARCH(1,1), GJR-GARCH(1,1,1) or EGARCH(1,1) with Normal / t / skew-t "
        "innovations fitted by MLE on percent returns; persistence α + β (+ γ/2), shock "
        "half-life ln 0.5 / ln(persistence), unconditional and conditional annualised vol, "
        "forecast term structure, news-impact curve, standardised-residual diagnostics."
    ),
    applies_to=PRICE_SUBJECTS,
    params=GarchParams,
    result_kind="series",
    requires={"prices"},
    min_obs=250,
    budget_ms=10_000,
)
def garch(inputs: AnalysisInputs, params: GarchParams) -> AnalysisOutput:
    factor = _annualisation(inputs)
    returns = _windowed(inputs, _prices(inputs))[_field(inputs)].pct_change().dropna() * 100
    r = require(returns, 250, "GARCH")
    lm, lm_p, _, _ = het_arch(r - r.mean(), nlags=5)
    spec = {
        "garch": {"vol": "GARCH", "o": 0},
        "gjr": {"vol": "GARCH", "o": 1},
        "egarch": {"vol": "EGARCH", "o": 1},
    }[params.model]
    model = arch_model(r, mean="Constant", p=1, q=1, dist=params.dist, **spec)  # type: ignore[arg-type]
    fit = model.fit(disp="off", show_warning=False)
    coef = {k: float(v) for k, v in fit.params.items()}
    se = {k: float(v) for k, v in fit.std_err.items()}
    alpha, beta, gamma = (
        coef.get("alpha[1]", 0.0),
        coef.get("beta[1]", 0.0),
        coef.get("gamma[1]", 0.0),
    )
    if params.model == "egarch":
        persistence = beta
        sigma2_bar = math.exp(coef["omega"] / (1 - beta)) if beta < 1 else float("nan")
    else:
        persistence = alpha + beta + gamma / 2
        sigma2_bar = coef["omega"] / (1 - persistence) if persistence < 1 else float("nan")
    half_life = math.log(0.5) / math.log(persistence) if 0 < persistence < 1 else None
    method: Literal["analytic", "simulation"] = (
        "simulation" if params.model == "egarch" else "analytic"
    )
    forecast = fit.forecast(
        horizon=params.horizon,
        method=method,
        reindex=False,
        simulations=500,
        random_state=np.random.RandomState(1),
    )
    term = np.sqrt(forecast.variance.to_numpy()[-1] * factor) / 100
    z = np.asarray(fit.std_resid, float)
    z = z[np.isfinite(z)]
    lb_z = acorr_ljungbox(z, lags=[10]).iloc[0]
    lb_z2 = acorr_ljungbox(z**2, lags=[10]).iloc[0]
    cond_vol = fit.conditional_volatility * math.sqrt(factor) / 100
    return AnalysisOutput(
        "series",
        {
            "model": params.model,
            "dist": params.dist,
            "params": coef,
            "std_err": se,
            "loglik": float(fit.loglikelihood),
            "aic": float(fit.aic),
            "bic": float(fit.bic),
            "persistence": persistence,
            "half_life_days": half_life,
            "unconditional_vol": math.sqrt(sigma2_bar * factor) / 100
            if math.isfinite(sigma2_bar)
            else None,
            "conditional_vol": {"dates": iso(returns.index), "values": clean(cond_vol)},
            "term_structure": {"horizon": list(range(1, params.horizon + 1)), "vol": term.tolist()},
            "news_impact": news_impact(
                coef, params.model, sigma2_bar if math.isfinite(sigma2_bar) else float(np.var(r))
            ),
            "diagnostics": [
                make_test(
                    "arch_lm_5",
                    "no ARCH effects",
                    float(lm),
                    float(lm_p),
                    reading_reject="volatility clustering",
                    reading_keep="no clustering",
                ),
                make_test(
                    "ljung_box_z_10",
                    "no autocorrelation in z",
                    float(lb_z.lb_stat),
                    float(lb_z.lb_pvalue),
                    reading_reject="mean dynamics left",
                    reading_keep="mean well specified",
                ),
                make_test(
                    "ljung_box_z2_10",
                    "no autocorrelation in z²",
                    float(lb_z2.lb_stat),
                    float(lb_z2.lb_pvalue),
                    reading_reject="variance dynamics left",
                    reading_keep="variance well specified",
                ),
            ],
        },
        ()
        if persistence < 1
        else ("Persistence ≥ 1: integrated variance, no finite long-run vol.",),
    )


# ── TS-007 ARIMA dynamics ────────────────────────────────────────────────────


class ArimaParams(SeriesParams):
    max_p: int = Field(3, ge=0, le=5)
    max_q: int = Field(3, ge=0, le=5)
    d: int = Field(0, ge=0, le=2)
    criterion: Literal["aic", "bic"] = "bic"


@analysis(
    id="TS-007",
    version="1.0.0",
    title="ARIMA dynamics",
    description=(
        "ARIMA(p,d,q) order chosen on a p, q ≤ max grid by AIC/BIC; coefficients ± SE, "
        "Ljung–Box on residuals, impulse response (ψ weights) to lag 20. In-sample only."
    ),
    applies_to=PRICE_SUBJECTS,
    params=ArimaParams,
    result_kind="table",
    requires={"prices"},
    min_obs=100,
    budget_ms=12_000,
)
def arima(inputs: AnalysisInputs, params: ArimaParams) -> AnalysisOutput:
    series = select_series(inputs, params)
    x = require(series, 100, "ARIMA")[-2_000:]  # cap the grid cost; recent dynamics matter most
    grid = []
    best: Any = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in range(params.max_p + 1):
            for q in range(params.max_q + 1):
                try:
                    fit = ARIMA(
                        x, order=(p, params.d, q), trend="c" if params.d == 0 else "n"
                    ).fit()
                except (ValueError, np.linalg.LinAlgError):
                    continue
                score = fit.aic if params.criterion == "aic" else fit.bic
                grid.append({"p": p, "q": q, "aic": float(fit.aic), "bic": float(fit.bic)})
                if best is None or score < best[0]:
                    best = (score, p, q, fit)
    if best is None:
        raise InsufficientDataError("No ARIMA model could be estimated")
    _, p, q, fit = best
    names = list(fit.param_names)
    coefficients = [
        {"name": n, "value": float(v), "se": float(s)}
        for n, v, s in zip(names, fit.params, fit.bse, strict=True)
    ]
    ar = np.r_[1, -np.asarray([fit.params[names.index(f"ar.L{i}")] for i in range(1, p + 1)])]
    ma = np.r_[1, np.asarray([fit.params[names.index(f"ma.L{i}")] for i in range(1, q + 1)])]
    lb = acorr_ljungbox(fit.resid, lags=[10]).iloc[0]
    return AnalysisOutput(
        "table",
        {
            "order": [p, params.d, q],
            "criterion": params.criterion,
            "grid": grid,
            "coefficients": coefficients,
            "impulse_response": arma2ma(ar, ma, lags=21).tolist(),
            "residual_test": make_test(
                "ljung_box_10",
                "white-noise residuals",
                float(lb.lb_stat),
                float(lb.lb_pvalue),
                reading_reject="dynamics left in residuals",
                reading_keep="residuals look white",
            ),
            "sample": sample_info(series, params),
        },
        ("In-sample description; ARIMA forecasts are not used for decisions before V5.",),
    )


# ── TS-009 mean reversion / OU ───────────────────────────────────────────────


class OuParams(SeriesParams):
    series: Literal["log_price", "price", "spread", "ratio", "zscore", "detrended", "frac_diff"] = (
        "log_price"
    )
    n_boot: int = Field(500, ge=100, le=5_000)
    seed: int = Field(3, ge=0, le=2**32 - 1)


def ou_fit(x: np.ndarray) -> dict[str, float]:
    """AR(1) by OLS x_t = a + φx_{t−1} + ε with Kendall bias correction φ + (1 + 3φ)/n."""
    y, lag = x[1:], x[:-1]
    design = np.column_stack([np.ones_like(lag), lag])
    (a, phi), *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ np.array([a, phi])
    sigma = float(resid.std(ddof=2))
    phi_c = min(phi + (1 + 3 * phi) / len(y), 0.9999)
    return {"a": float(a), "phi": float(phi), "phi_corrected": float(phi_c), "sigma": sigma}


def half_life(phi: float) -> float | None:
    return -math.log(2) / math.log(phi) if 0 < phi < 1 else None


@analysis(
    id="TS-009",
    version="1.0.0",
    title="Mean-reversion / OU fit",
    description=(
        "Discrete OU as AR(1): φ (OLS and Kendall-corrected), long-run mean μ = a/(1 − φ), "
        "stationary σ/√(1 − φ²), half-life −ln 2 / ln φ with a residual-bootstrap 95 % interval; "
        "warns when φ̂ → 1 (AFML-13)."
    ),
    applies_to=PRICE_SUBJECTS,
    params=OuParams,
    result_kind="table",
    requires={"prices"},
    min_obs=60,
    budget_ms=6_000,
)
def ou_mean_reversion(inputs: AnalysisInputs, params: OuParams) -> AnalysisOutput:
    series = select_series(inputs, params)
    x = require(series, 60, "OU fit")
    fit = ou_fit(x)
    phi, a = fit["phi"], fit["a"]
    rng = np.random.default_rng(params.seed)
    resid = x[1:] - (a + phi * x[:-1])
    lives = []
    for _ in range(params.n_boot):
        shocks = rng.choice(resid, size=len(resid), replace=True)
        path = np.empty_like(x)
        path[0] = x[0]
        for t in range(1, len(x)):
            path[t] = a + phi * path[t - 1] + shocks[t - 1]
        hl = half_life(ou_fit(path)["phi_corrected"])
        lives.append(hl if hl is not None else np.inf)
    finite = np.array(lives)
    ci = np.percentile(finite, [2.5, 97.5])
    warnings_out = []
    if phi >= 0.98:
        warnings_out.append("φ̂ ≥ 0.98: close to a unit root, the half-life is unreliable.")
    return AnalysisOutput(
        "table",
        {
            **fit,
            "mu": a / (1 - phi) if phi < 1 else None,
            "sigma_stationary": fit["sigma"] / math.sqrt(1 - phi**2) if abs(phi) < 1 else None,
            "half_life": half_life(phi),
            "half_life_corrected": half_life(fit["phi_corrected"]),
            "half_life_ci": [float(c) if np.isfinite(c) else None for c in ci],
            "share_without_reversion": float(np.mean(~np.isfinite(finite))),
            "last_zscore": float((x[-1] - a / (1 - phi)) / (fit["sigma"] / math.sqrt(1 - phi**2)))
            if abs(phi) < 1
            else None,
            "sample": sample_info(series, params),
        },
        tuple(warnings_out),
    )


# ── TS-012 regimes ───────────────────────────────────────────────────────────


class RegimeParams(SeriesParams):
    k_regimes: int = Field(2, ge=2, le=3)


@analysis(
    id="TS-012",
    version="1.0.0",
    title="Regimes",
    description=(
        "Hamilton Markov-switching model with regime-specific mean and variance on daily "
        "returns: smoothed probabilities, transition matrix, expected durations 1/(1 − p_ii), "
        "annualised mean and vol per regime (regimes ordered by volatility)."
    ),
    applies_to=PRICE_SUBJECTS,
    params=RegimeParams,
    result_kind="series",
    requires={"prices"},
    min_obs=250,
    budget_ms=15_000,
)
def regimes(inputs: AnalysisInputs, params: RegimeParams) -> AnalysisOutput:
    factor = _annualisation(inputs)
    returns = _windowed(inputs, _prices(inputs))[_field(inputs)].pct_change().dropna()
    r = require(returns, 250, "Regime model") * 100
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MarkovRegression(r, k_regimes=params.k_regimes, trend="c", switching_variance=True)
        fit = model.fit(search_reps=10, disp=False)
    k = params.k_regimes
    named = dict(zip(model.param_names, np.asarray(fit.params, float), strict=True))
    sigmas = np.sqrt([named[f"sigma2[{i}]"] for i in range(k)])
    means = np.array([named[f"const[{i}]"] for i in range(k)])
    order = np.argsort(sigmas)
    transition = np.asarray(fit.regime_transition)[:, :, 0]  # [to, from]
    probs = np.asarray(fit.smoothed_marginal_probabilities)
    regimes_out = []
    for rank, i in enumerate(order):
        p_stay = float(transition[i, i])
        regimes_out.append(
            {
                "regime": rank,
                "mean_annualised": float(means[i] / 100 * factor),
                "vol_annualised": float(sigmas[i] / 100 * math.sqrt(factor)),
                "expected_duration_days": 1 / (1 - p_stay) if p_stay < 1 else None,
                "share_of_time": float(probs[:, i].mean()),
            }
        )
    return AnalysisOutput(
        "series",
        {
            "regimes": regimes_out,
            "transition": [[float(transition[j, i]) for j in order] for i in order],
            "dates": iso(returns.index),
            "probabilities": [clean(probs[:, i]) for i in order],
            "current_regime": int(np.argmax(probs[-1, order])),
            "loglik": float(fit.llf),
        },
        (
            "Regimes are fitted on the whole range: shading is descriptive "
            "(smoothed probabilities use future data).",
        ),
    )


# ── TS-019 Kalman filtering ──────────────────────────────────────────────────


class KalmanParams(SeriesParams):
    model: Literal["local_level", "local_linear_trend", "dynamic_beta"] = "local_level"
    delta: float = Field(1e-4, gt=0, le=1e-1, description="State noise for the dynamic beta")


def dynamic_regression(y: np.ndarray, x: np.ndarray, delta: float) -> dict[str, np.ndarray]:
    """Random-walk alpha/beta θ_t = θ_{t−1} + w, y_t = [1, x_t]θ_t + v (filtered: no look-ahead)."""
    n = len(y)
    theta = np.zeros(2)
    p = np.eye(2)
    q = delta / (1 - delta) * np.eye(2)
    beta_ols = OLS(y, add_constant(x)).fit()
    r = float(beta_ols.mse_resid)
    out = np.empty((n, 2))
    var = np.empty((n, 2))
    for t in range(n):
        p = p + q
        h = np.array([1.0, x[t]])
        s = h @ p @ h + r
        k = p @ h / s
        theta = theta + k * (y[t] - h @ theta)
        p = p - np.outer(k, h) @ p
        out[t] = theta
        var[t] = np.diag(p)
    return {"alpha": out[:, 0], "beta": out[:, 1], "beta_se": np.sqrt(var[:, 1])}


@analysis(
    id="TS-019",
    version="1.0.0",
    title="Kalman filtering",
    description=(
        "Local-level or local-linear-trend model on log price (MLE): signal-to-noise ratio and "
        "smoothed state ± 2 SE; or dynamic alpha/beta of daily returns against the benchmark "
        "(random-walk coefficients, filtered estimates only)."
    ),
    applies_to=PRICE_SUBJECTS,
    params=KalmanParams,
    result_kind="series",
    requires={"prices", "other_prices"},
    min_obs=60,
    budget_ms=10_000,
)
def kalman(inputs: AnalysisInputs, params: KalmanParams) -> AnalysisOutput:
    prices = _windowed(inputs, _prices(inputs))[_field(inputs)]
    if params.model == "dynamic_beta":
        other = inputs.data.get("other_prices")
        if not isinstance(other, pd.DataFrame) or other.empty:
            raise InsufficientDataError(
                "Dynamic beta needs benchmark prices (ingest the benchmark)"
            )
        both = pd.concat(
            [prices.pct_change(), other[_field(inputs)].pct_change()], axis=1, join="inner"
        ).dropna()
        require(both.iloc[:, 0], 60, "Dynamic beta")
        est = dynamic_regression(
            both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy(), params.delta
        )
        return AnalysisOutput(
            "series",
            {
                "model": params.model,
                "dates": iso(both.index),
                "beta": clean(est["beta"]),
                "beta_upper": clean(est["beta"] + 2 * est["beta_se"]),
                "beta_lower": clean(est["beta"] - 2 * est["beta_se"]),
                "alpha": clean(est["alpha"]),
                "benchmark": inputs.conventions.get("other_symbol"),
            },
        )
    y = np.log(prices)
    require(y, 60, "Kalman filter")
    level = "llevel" if params.model == "local_level" else "lltrend"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = UnobservedComponents(y.to_numpy(), level=level).fit(disp=False)
    smoothed = fit.smoothed_state[0]
    se = np.sqrt(fit.smoothed_state_cov[0, 0])
    names = list(fit.model.param_names)
    variances = dict(zip(names, np.asarray(fit.params, float), strict=True))
    snr = variances.get("sigma2.level", 0.0) / variances.get("sigma2.irregular", np.nan)
    return AnalysisOutput(
        "series",
        {
            "model": params.model,
            "dates": iso(y.index),
            "observed": clean(y),
            "smoothed": clean(smoothed),
            "upper": clean(smoothed + 2 * se),
            "lower": clean(smoothed - 2 * se),
            "variances": {k: float(v) for k, v in variances.items()},
            "signal_to_noise": float(snr) if np.isfinite(snr) else None,
        },
    )
