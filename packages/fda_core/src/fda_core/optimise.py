"""Portfolio construction engine (M23, IN-010…IN-017): pure functions on a returns matrix
(rows = periods, columns = assets) and on (μ, Σ) estimates. SciPy only (SLSQP, HiGHS LP,
hierarchical clustering); no extra optimisation dependency.

Weights are long-only and fully invested unless bounds say otherwise; every solver returns a
weight vector summing to 1 (tolerance 1e-9 after cleaning) or raises InsufficientDataError.
"""

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import optimize, stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from fda_core.portfolio import covariance
from fda_core.registry import InsufficientDataError

Array = NDArray[np.float64]
MeanEstimator = Literal["historical", "james_stein", "equilibrium"]
CovEstimator = Literal["sample", "ledoit_wolf", "oas", "ewma", "denoised"]
EULER_GAMMA = 0.5772156649015329


# ── estimators (IN-012) ──────────────────────────────────────────────────────


def mean_estimate(
    returns: pd.DataFrame,
    method: MeanEstimator,
    cov: Array | None = None,
    risk_aversion: float = 2.5,
) -> Array:
    """Per-period means: historical; James–Stein shrinkage towards the grand mean (Jorion
    1986 intensity); CAPM equilibrium π = δΣw with equal market weights (no cap data)."""
    x = returns.to_numpy(dtype=float)
    mu = x.mean(axis=0)
    n, p = x.shape
    if method == "historical" or p < 3:
        return np.asarray(mu, dtype=float)
    sigma = cov if cov is not None else np.cov(x, rowvar=False)
    if method == "equilibrium":
        return np.asarray(risk_aversion * sigma @ np.full(p, 1 / p), dtype=float)
    inv = np.linalg.pinv(sigma)
    ones = np.ones(p)
    target = float(ones @ inv @ mu / (ones @ inv @ ones))
    gap = mu - target
    denom = float(gap @ inv @ gap) * n
    intensity = min(1.0, (p + 2) / (p + 2 + denom)) if denom > 0 else 1.0
    return np.asarray((1 - intensity) * mu + intensity * target, dtype=float)


def denoise(cov: Array, n_obs: int) -> Array:
    """Marčenko–Pastur clipping: eigenvalues of the correlation below λ+ = (1 + √(p/n))² are
    replaced by their mean (trace preserved), then rescaled to the original variances."""
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    vals, vecs = np.linalg.eigh(corr)
    q = corr.shape[0] / n_obs
    lam_plus = (1 + math.sqrt(q)) ** 2
    noise = vals < lam_plus
    if noise.any() and (~noise).any():
        vals = vals.copy()
        vals[noise] = vals[noise].mean()
    cleaned = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(cleaned))
    cleaned = cleaned / np.outer(d, d)
    return np.asarray(cleaned * np.outer(sd, sd))


def cov_estimate(returns: pd.DataFrame, method: CovEstimator) -> Array:
    if method == "denoised":
        return denoise(np.cov(returns.to_numpy(dtype=float), rowvar=False), len(returns))
    return np.asarray(covariance(returns, method))


def black_litterman(
    cov: Array, prior: Array, views: list[tuple[int, float, float]], tau: float = 0.05
) -> Array:
    """Absolute views (asset index, expected per-period return, confidence 0–1) on the prior;
    Ω_k = (1/c − 1)·τ·σ²_k (He–Litterman-style confidence scaling)."""
    if not views:
        return prior
    p = len(prior)
    pick = np.zeros((len(views), p))
    q = np.zeros(len(views))
    omega = np.zeros((len(views), len(views)))
    for i, (asset, value, confidence) in enumerate(views):
        pick[i, asset] = 1.0
        q[i] = value
        c = min(max(confidence, 1e-3), 0.999)
        omega[i, i] = (1 / c - 1) * tau * cov[asset, asset]
    tau_sigma = tau * cov
    middle = np.linalg.inv(pick @ tau_sigma @ pick.T + omega)
    return np.asarray(prior + tau_sigma @ pick.T @ middle @ (q - pick @ prior))


# ── solvers (IN-010, IN-011, IN-013, IN-015, IN-016) ─────────────────────────


def _clean(w: Array) -> Array:
    w = np.where(np.abs(w) < 1e-10, 0.0, w)
    total = w.sum()
    if not math.isfinite(total) or abs(total) < 1e-12:
        raise InsufficientDataError("The optimiser returned degenerate weights")
    return np.asarray(w / total)


def _slsqp(
    objective: Callable[[Array], float],
    p: int,
    lower: float,
    upper: float,
    extra: list[dict[str, object]] | None = None,
    start: Array | None = None,
) -> Array:
    if lower * p > 1 + 1e-12 or upper * p < 1 - 1e-12:
        raise InsufficientDataError(f"Bounds [{lower}, {upper}] cannot sum to 1 with {p} assets")
    x0 = start if start is not None else np.full(p, 1 / p)
    result = optimize.minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[(lower, upper)] * p,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}, *(extra or [])],
        options={"maxiter": 1000, "ftol": 1e-14},
    )
    if not result.success and result.status not in (8, 9):
        raise InsufficientDataError(f"Optimiser did not converge: {result.message}")
    return _clean(np.asarray(result.x))


def equal_weight(p: int) -> Array:
    return np.full(p, 1 / p)


def inverse_vol(cov: Array) -> Array:
    inv = 1 / np.sqrt(np.diag(cov))
    return np.asarray(inv / inv.sum())


def gmv(cov: Array, lower: float = 0.0, upper: float = 1.0) -> Array:
    return _slsqp(lambda w: float(w @ cov @ w), len(cov), lower, upper)


def max_sharpe(
    mu: Array, cov: Array, rf: float = 0.0, lower: float = 0.0, upper: float = 1.0
) -> Array:
    if (mu - rf).max() <= 0:
        raise InsufficientDataError("No asset has an expected return above the risk-free rate")
    return _slsqp(
        lambda w: -float((w @ mu - rf) / math.sqrt(max(float(w @ cov @ w), 1e-18))),
        len(cov),
        lower,
        upper,
    )


def target_volatility(
    mu: Array, cov: Array, target: float, lower: float = 0.0, upper: float = 1.0
) -> Array:
    """Max expected return subject to σ_p ≤ target (per-period units)."""
    con = [{"type": "ineq", "fun": lambda w: target**2 - float(w @ cov @ w)}]
    return _slsqp(lambda w: -float(w @ mu), len(cov), lower, upper, con, gmv(cov, lower, upper))


def target_return(
    mu: Array, cov: Array, target: float, lower: float = 0.0, upper: float = 1.0
) -> Array:
    con = [{"type": "ineq", "fun": lambda w: float(w @ mu) - target}]
    return _slsqp(lambda w: float(w @ cov @ w), len(cov), lower, upper, con)


def max_utility(
    mu: Array, cov: Array, risk_aversion: float, lower: float = 0.0, upper: float = 1.0
) -> Array:
    return _slsqp(
        lambda w: -float(w @ mu - risk_aversion / 2 * w @ cov @ w), len(cov), lower, upper
    )


def risk_budget(cov: Array, budgets: Array | None = None) -> Array:
    """Spinu (2013) convex form: min ½ yᵀΣy − Σ b_i ln y_i (y > 0), w = y / Σy; the risk
    contributions w_i(Σw)_i are then proportional to b (ERC for equal budgets)."""
    p = len(cov)
    b = np.full(p, 1 / p) if budgets is None else budgets / budgets.sum()
    result = optimize.minimize(
        lambda y: 0.5 * y @ cov @ y - b @ np.log(y),
        np.full(p, 1 / math.sqrt(np.trace(cov))),
        jac=lambda y: cov @ y - b / y,
        method="L-BFGS-B",
        bounds=[(1e-12, None)] * p,
        options={"maxiter": 5000, "gtol": 1e-14, "ftol": 1e-16},
    )
    return _clean(np.asarray(result.x))


def max_diversification(cov: Array, lower: float = 0.0, upper: float = 1.0) -> Array:
    """Max wᵀσ / √(wᵀΣw) (Choueifaty)."""
    sd = np.sqrt(np.diag(cov))
    return _slsqp(
        lambda w: -float(w @ sd / math.sqrt(max(float(w @ cov @ w), 1e-18))), len(cov), lower, upper
    )


def min_cvar(returns: Array, alpha: float = 0.95, upper: float = 1.0) -> Array:
    """Rockafellar–Uryasev LP: min ζ + 1/((1−α)T) Σ u_t, u_t ≥ −r_tᵀw − ζ, u ≥ 0."""
    t, p = returns.shape
    c = np.concatenate([np.zeros(p), [1.0], np.full(t, 1 / ((1 - alpha) * t))])
    a_ub = np.hstack([-returns, -np.ones((t, 1)), -np.eye(t)])
    a_eq = np.concatenate([np.ones(p), [0.0], np.zeros(t)])[None, :]
    bounds = [(0.0, upper)] * p + [(None, None)] + [(0.0, None)] * t
    res = optimize.linprog(
        c, A_ub=a_ub, b_ub=np.zeros(t), A_eq=a_eq, b_eq=[1.0], bounds=bounds, method="highs"
    )
    if not res.success:
        raise InsufficientDataError(f"CVaR LP failed: {res.message}")
    return _clean(np.asarray(res.x[:p]))


def min_cdar(returns: Array, alpha: float = 0.95, upper: float = 1.0) -> Array:
    """Chekhlov–Uryasev–Zabarankin LP on cumulative (uncompounded) returns: drawdown
    d_t = u_t − c_tᵀw with u_t ≥ c_tᵀw, u_t ≥ u_{t−1}, u_0 = 0; min ζ + 1/((1−α)T) Σ z_t,
    z_t ≥ d_t − ζ, z ≥ 0."""
    t, p = returns.shape
    cum = np.cumsum(returns, axis=0)
    # variables: w (p), u (t), z (t), zeta (1)
    n = p + 2 * t + 1
    c = np.zeros(n)
    c[p + t : p + 2 * t] = 1 / ((1 - alpha) * t)
    c[-1] = 1.0
    rows, rhs = [], []
    for i in range(t):
        r = np.zeros(n)  # c_tᵀw − u_t ≤ 0
        r[:p], r[p + i] = cum[i], -1.0
        rows.append(r)
        rhs.append(0.0)
        if i > 0:  # u_{t−1} − u_t ≤ 0
            r = np.zeros(n)
            r[p + i - 1], r[p + i] = 1.0, -1.0
            rows.append(r)
            rhs.append(0.0)
        r = np.zeros(n)  # u_t − c_tᵀw − ζ − z_t ≤ 0
        r[:p], r[p + i], r[p + t + i], r[-1] = -cum[i], 1.0, -1.0, -1.0
        rows.append(r)
        rhs.append(0.0)
    a_eq = np.zeros((1, n))
    a_eq[0, :p] = 1.0
    bounds = [(0.0, upper)] * p + [(0.0, None)] * t + [(0.0, None)] * t + [(None, None)]
    res = optimize.linprog(
        c,
        A_ub=np.array(rows),
        b_ub=np.array(rhs),
        A_eq=a_eq,
        b_eq=[1.0],
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise InsufficientDataError(f"CDaR LP failed: {res.message}")
    return _clean(np.asarray(res.x[:p]))


def hrp(
    returns: pd.DataFrame, cov: Array | None = None, linkage: str = "single"
) -> tuple[Array, Array]:
    """López de Prado (AFML-16): distance √(½(1 − corr)), hierarchical clustering, quasi-
    diagonal order, recursive bisection with inverse-variance cluster weights. Returns the
    weights and the linkage matrix (dendrogram)."""
    sigma = cov if cov is not None else np.cov(returns.to_numpy(dtype=float), rowvar=False)
    sd = np.sqrt(np.diag(sigma))
    corr = np.clip(sigma / np.outer(sd, sd), -1, 1)
    dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, None))
    np.fill_diagonal(dist, 0)
    link = hierarchy.linkage(squareform(dist, checks=False), method=linkage)
    order = list(hierarchy.leaves_list(link))
    w = np.ones(len(sd))

    def cluster_var(items: list[int]) -> float:
        sub = sigma[np.ix_(items, items)]
        ivp = 1 / np.diag(sub)
        ivp /= ivp.sum()
        return float(ivp @ sub @ ivp)

    clusters = [order]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) < 2:
                continue
            left, right = c[: len(c) // 2], c[len(c) // 2 :]
            v_left, v_right = cluster_var(left), cluster_var(right)
            alpha = 1 - v_left / (v_left + v_right)
            w[left] *= alpha
            w[right] *= 1 - alpha
            nxt += [left, right]
        clusters = nxt
    return _clean(w), np.asarray(link)


def resampled(
    returns: pd.DataFrame, solver: Callable[[Array, Array], Array], draws: int = 200, seed: int = 11
) -> Array:
    """Michaud resampling: weights solved on bootstrap samples of the returns, averaged."""
    rng = np.random.default_rng(seed)
    x = returns.to_numpy(dtype=float)
    acc = np.zeros(x.shape[1])
    ok = 0
    for _ in range(draws):
        sample = x[rng.integers(0, len(x), len(x))]
        try:
            acc += solver(sample.mean(axis=0), np.cov(sample, rowvar=False))
            ok += 1
        except InsufficientDataError:
            continue
    if ok == 0:
        raise InsufficientDataError("Every resample failed")
    return _clean(acc / ok)


def robust_mean(mu: Array, cov: Array, n_obs: int, kappa: float = 1.0) -> Array:
    """Box-uncertainty worst case: μ_i − κ·σ_i/√T (penalises noisy means)."""
    return np.asarray(mu - kappa * np.sqrt(np.diag(cov) / n_obs))


def index_tracking(returns: Array, benchmark: Array, names: int) -> Array:
    """Greedy forward selection of `names` assets minimising tracking-error variance, then
    long-only least squares of the benchmark on the selection (weights renormalised)."""
    p = returns.shape[1]
    chosen: list[int] = []

    def fit(cols: list[int]) -> tuple[Array, float]:
        sub = returns[:, cols]
        w, _ = optimize.nnls(sub, benchmark)
        w = w / w.sum() if w.sum() > 0 else np.full(len(cols), 1 / len(cols))
        te = float(np.var(sub @ w - benchmark, ddof=1))
        return w, te

    while len(chosen) < min(names, p):
        best = min((c for c in range(p) if c not in chosen), key=lambda c: fit([*chosen, c])[1])
        chosen.append(best)
    w_sel, _ = fit(chosen)
    out = np.zeros(p)
    out[chosen] = w_sel
    return _clean(out)


def frontier(
    mu: Array, cov: Array, points: int = 20, lower: float = 0.0, upper: float = 1.0
) -> list[tuple[float, float]]:
    """(volatility, return) per period along target returns from the GMV return to the max."""
    w0 = gmv(cov, lower, upper)
    lo, hi = (
        float(w0 @ mu),
        float(mu.max() if upper >= 1 else np.sort(mu)[::-1][: max(1, int(1 / upper))].mean()),
    )
    out = []
    for target in np.linspace(lo, hi, points):
        try:
            w = target_return(mu, cov, float(target), lower, upper)
        except InsufficientDataError:
            continue
        out.append((float(math.sqrt(w @ cov @ w)), float(w @ mu)))
    return out


# ── validation statistics (IN-018) ───────────────────────────────────────────


def deflated_sharpe(
    sr: float, sr_trials: list[float], n_obs: int, skew: float, kurt: float
) -> dict[str, float]:
    """Bailey & López de Prado (2014): SR₀ = √V[SR_trials]·((1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(Ne))),
    DSR = PSR(SR₀) with per-period Sharpe ratios and non-excess kurtosis."""
    n = max(len(sr_trials), 1)
    var = float(np.var(sr_trials, ddof=1)) if len(sr_trials) > 1 else 0.0
    sr0 = (
        math.sqrt(var)
        * (
            (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n)
            + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n * math.e))
        )
        if n > 1
        else 0.0
    )
    denominator = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12))
    dsr = float(stats.norm.cdf((sr - sr0) * math.sqrt(n_obs - 1) / denominator))
    return {"sr0": sr0, "dsr": dsr, "trials": n}
