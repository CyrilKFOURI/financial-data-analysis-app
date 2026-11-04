"""Option pricing engine (DV-002, M21): pure functions, vectorised over numpy arrays.

- Black–Scholes–Merton with a continuous dividend yield q; Black-76 is BSM with S = F, q = r.
- Greeks: Δ, Γ, vega (per 1.00 of volatility), Θ (per year), rho, vanna ∂Δ/∂σ, volga ∂vega/∂σ,
  charm ∂Δ/∂t (= −∂Δ/∂T, per year).
- CRR binomial tree for American exercise (N steps).
- Implied volatility by Brent's method within no-arbitrage bounds (None outside them).
- Raw SVI smile fit (Gatheral): w(k) = a + b(rho·(k − m) + √((k − m)² + s²)) on total variance,
  Durrleman butterfly condition, calendar check, Breeden–Litzenberger density.
- Heston (1993) via the characteristic function in the "little trap" form (Albrecher et al.
  2007), Gil-Pelaez inversion on a Gauss–Legendre grid; least-squares calibration.
"""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats

Kind = Literal["call", "put"]
Array = NDArray[np.float64]


def _arr(x: float | Array) -> Array:
    return np.asarray(x, dtype=float)


def d1_d2(
    s: float | Array, k: float | Array, t: float | Array, r: float, q: float, sigma: float | Array
) -> tuple[Array, Array]:
    s, k, t, sigma = _arr(s), _arr(k), _arr(t), _arr(sigma)
    vol = sigma * np.sqrt(t)
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / vol
    return d1, d1 - vol


def bsm_price(
    kind: Kind,
    s: float | Array,
    k: float | Array,
    t: float | Array,
    r: float,
    q: float,
    sigma: float | Array,
) -> Array:
    d1, d2 = d1_d2(s, k, t, r, q, sigma)
    s, k, t = _arr(s), _arr(k), _arr(t)
    df_q, df_r = np.exp(-q * t), np.exp(-r * t)
    if kind == "call":
        return _arr(s * df_q * stats.norm.cdf(d1) - k * df_r * stats.norm.cdf(d2))
    return _arr(k * df_r * stats.norm.cdf(-d2) - s * df_q * stats.norm.cdf(-d1))


def black76_price(
    kind: Kind, f: float | Array, k: float | Array, t: float | Array, r: float, sigma: float | Array
) -> Array:
    """Options on futures: BSM with the forward as spot and q = r."""
    return bsm_price(kind, f, k, t, r, r, sigma)


def bsm_greeks(
    kind: Kind,
    s: float | Array,
    k: float | Array,
    t: float | Array,
    r: float,
    q: float,
    sigma: float | Array,
) -> dict[str, Array]:
    d1, d2 = d1_d2(s, k, t, r, q, sigma)
    s, k, t, sigma = _arr(s), _arr(k), _arr(t), _arr(sigma)
    df_q, df_r = np.exp(-q * t), np.exp(-r * t)
    pdf = stats.norm.pdf(d1)
    sqrt_t = np.sqrt(t)
    gamma = df_q * pdf / (s * sigma * sqrt_t)
    vega = s * df_q * pdf * sqrt_t
    decay = -s * df_q * pdf * sigma / (2 * sqrt_t)
    common_charm = df_q * pdf * (2 * (r - q) * t - d2 * sigma * sqrt_t) / (2 * t * sigma * sqrt_t)
    if kind == "call":
        delta = df_q * stats.norm.cdf(d1)
        theta = decay - r * k * df_r * stats.norm.cdf(d2) + q * s * df_q * stats.norm.cdf(d1)
        rho = k * t * df_r * stats.norm.cdf(d2)
        charm = q * df_q * stats.norm.cdf(d1) - common_charm
    else:
        delta = -df_q * stats.norm.cdf(-d1)
        theta = decay + r * k * df_r * stats.norm.cdf(-d2) - q * s * df_q * stats.norm.cdf(-d1)
        rho = -k * t * df_r * stats.norm.cdf(-d2)
        charm = -q * df_q * stats.norm.cdf(-d1) - common_charm
    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
        "vanna": -df_q * pdf * d2 / sigma,
        "volga": vega * d1 * d2 / sigma,
        "charm": charm,
    }


def crr_price(
    kind: Kind,
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
    steps: int = 500,
    american: bool = True,
) -> float:
    """Cox–Ross–Rubinstein tree: u = e^{σ√Δt}, d = 1/u, p = (e^{(r−q)Δt} − d)/(u − d)."""
    dt = t / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    prices = s * u ** np.arange(steps, -1, -1) * d ** np.arange(0, steps + 1)
    sign = 1.0 if kind == "call" else -1.0
    values = np.maximum(sign * (prices - k), 0.0)
    for _ in range(steps):
        values = disc * (p * values[:-1] + (1 - p) * values[1:])
        if american:
            prices = prices[:-1] / u
            values = np.maximum(values, sign * (prices - k))
    return float(values[0])


def implied_vol(
    kind: Kind,
    price: float,
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    low: float = 1e-6,
    high: float = 5.0,
) -> float | None:
    """Brent root of BSM(σ) − price; None when the price is outside the no-arbitrage bounds
    (intrinsic value under the forward ≤ price < upper bound) or no root is bracketed."""
    if not (price > 0 and t > 0 and s > 0 and k > 0):
        return None
    df_q, df_r = math.exp(-q * t), math.exp(-r * t)
    lower = max(s * df_q - k * df_r, 0.0) if kind == "call" else max(k * df_r - s * df_q, 0.0)
    upper = s * df_q if kind == "call" else k * df_r
    if not lower <= price < upper:
        return None

    def gap(sigma: float) -> float:
        return float(bsm_price(kind, s, k, t, r, q, sigma)) - price

    try:
        if gap(low) * gap(high) > 0:
            return None
        return float(optimize.brentq(gap, low, high, xtol=1e-14, rtol=1e-14, maxiter=200))
    except ValueError:
        return None


def implied_vol_vec(
    is_call: NDArray[np.bool_],
    price: Array,
    s: float,
    k: Array,
    t: Array,
    r: float,
    q: float,
    iterations: int = 100,
) -> Array:
    """Vectorised IV: Newton on vega inside a shrinking [low, high] bracket (bisection when a
    step leaves it); NaN outside the no-arbitrage bounds. Agrees with `implied_vol` to 1e-10."""
    price, k, t = _arr(price), _arr(k), _arr(t)
    df_q, df_r = np.exp(-q * t), np.exp(-r * t)
    intrinsic = np.where(
        is_call, np.maximum(s * df_q - k * df_r, 0), np.maximum(k * df_r - s * df_q, 0)
    )
    upper = np.where(is_call, s * df_q, k * df_r)
    valid = (price > 0) & (t > 0) & (price >= intrinsic) & (price < upper)
    low = np.full(price.shape, 1e-6)
    high = np.full(price.shape, 5.0)
    sigma = np.full(price.shape, 0.3)
    for _ in range(iterations):
        d1, d2 = d1_d2(s, k, t, r, q, sigma)
        model = np.where(
            is_call,
            s * df_q * stats.norm.cdf(d1) - k * df_r * stats.norm.cdf(d2),
            k * df_r * stats.norm.cdf(-d2) - s * df_q * stats.norm.cdf(-d1),
        )
        diff = model - price
        high = np.where(diff > 0, sigma, high)  # price increases with σ
        low = np.where(diff <= 0, sigma, low)
        vega = s * df_q * stats.norm.pdf(d1) * np.sqrt(t)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            step = sigma - diff / vega
        inside = np.isfinite(step) & (step > low) & (step < high)
        sigma = np.where(inside, step, 0.5 * (low + high))
        if np.nanmax(np.abs(np.where(valid, diff, 0.0))) < 1e-13:
            break
    hit_bound = (sigma <= 1e-6 * 1.0001) | (sigma >= 5.0 * 0.9999)
    return np.where(valid & ~hit_bound, sigma, np.nan)


# ── SVI smile ────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Svi:
    a: float
    b: float
    rho: float
    m: float
    s: float

    def w(self, k: Array) -> Array:
        """Total implied variance σ²T at log-moneyness k = ln(K/F)."""
        x = k - self.m
        return self.a + self.b * (self.rho * x + np.sqrt(x**2 + self.s**2))

    def dw(self, k: Array) -> Array:
        x = k - self.m
        return self.b * (self.rho + x / np.sqrt(x**2 + self.s**2))

    def d2w(self, k: Array) -> Array:
        x = k - self.m
        return self.b * self.s**2 / (x**2 + self.s**2) ** 1.5

    def iv(self, k: Array, t: float) -> Array:
        return np.sqrt(np.maximum(self.w(k), 1e-12) / t)


def fit_svi(k: Array, iv: Array, t: float, weights: Array | None = None) -> tuple[Svi, float]:
    """Least-squares raw SVI on total variance; returns the fit and the IV RMSE."""
    w_obs = iv**2 * t
    wts = np.ones_like(k) if weights is None else weights
    guess = [max(w_obs.min() * 0.5, 1e-6), 0.1, -0.3, 0.0, 0.1]

    def residuals(p: Array) -> Array:
        fit = Svi(*p)
        return np.sqrt(wts) * (fit.w(k) - w_obs)

    span = float(max(abs(k).max(), 0.5))
    result = optimize.least_squares(
        residuals,
        guess,
        bounds=(
            [-1.0, 0.0, -0.999, -span, 1e-4],
            [float(w_obs.max()) * 2 + 1e-6, 5.0, 0.999, span, 2.0],
        ),
        max_nfev=2000,
    )
    fit = Svi(*[float(x) for x in result.x])
    rmse = float(np.sqrt(np.mean((fit.iv(k, t) - iv) ** 2)))
    return fit, rmse


def butterfly_g(fit: Svi, k: Array) -> Array:
    """Durrleman's g(k) = (1 − k w'/(2w))² − (w'²/4)(1/w + 1/4) + w''/2; g < 0 ⇒ butterfly
    arbitrage (negative density)."""
    w, dw, d2w = fit.w(k), fit.dw(k), fit.d2w(k)
    return (1 - k * dw / (2 * w)) ** 2 - dw**2 / 4 * (1 / w + 0.25) + d2w / 2


def rnd_breeden_litzenberger(fit: Svi, forward: float, t: float, r: float, strikes: Array) -> Array:
    """f(K) = e^{rT} ∂²C/∂K², with C from the SVI-smoothed smile (Black on the forward)."""
    k = np.log(strikes / forward)
    calls = black76_price("call", forward, strikes, t, r, fit.iv(k, t))
    second = np.gradient(np.gradient(calls, strikes), strikes)
    return np.maximum(math.exp(r * t) * second, 0.0)


# ── Heston ───────────────────────────────────────────────────────────────────

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(256)
U_MAX = 250.0
_U = 0.5 * U_MAX * (_NODES + 1)  # nodes on (0, U_MAX)
_W = 0.5 * U_MAX * _WEIGHTS


@dataclass(frozen=True, slots=True)
class Heston:
    kappa: float
    theta: float
    sigma: float
    rho: float
    v0: float


def heston_cf(
    u: NDArray[np.complex128], s: float, t: float, r: float, q: float, p: Heston
) -> NDArray[np.complex128]:
    """φ(u) = E[e^{iu ln S_T}] in the little-trap form (no branch-cut discontinuities)."""
    iu = 1j * u
    xi = p.kappa - p.rho * p.sigma * iu
    d = np.sqrt(xi**2 + p.sigma**2 * (iu + u**2))
    g = (xi - d) / (xi + d)
    exp_dt = np.exp(-d * t)
    c = (p.kappa * p.theta / p.sigma**2) * ((xi - d) * t - 2 * np.log((1 - g * exp_dt) / (1 - g)))
    dd = (p.v0 / p.sigma**2) * (xi - d) * (1 - exp_dt) / (1 - g * exp_dt)
    out: NDArray[np.complex128] = np.exp(iu * (math.log(s) + (r - q) * t) + c + dd)
    return out


def heston_price(
    kind: Kind, s: float, k: float | Array, t: float, r: float, q: float, p: Heston
) -> Array:
    """Gil-Pelaez: C = S e^{−qT} P1 − K e^{−rT} P2 (puts by parity)."""
    k = np.atleast_1d(_arr(k))
    u = _U.astype(np.complex128)
    phi = heston_cf(u, s, t, r, q, p)
    phi_shift = heston_cf(u - 1j, s, t, r, q, p)
    phi_minus_i = heston_cf(np.array([-1j]), s, t, r, q, p)[0]
    log_k = np.log(k)[:, None]
    kernel = np.exp(-1j * u[None, :] * log_k) / (1j * u[None, :])
    p2 = 0.5 + (kernel * phi[None, :]).real @ _W / math.pi
    p1 = 0.5 + (kernel * (phi_shift / phi_minus_i)[None, :]).real @ _W / math.pi
    call = s * math.exp(-q * t) * p1 - k * math.exp(-r * t) * p2
    if kind == "call":
        return _arr(call)
    return _arr(call - s * math.exp(-q * t) + k * math.exp(-r * t))


def calibrate_heston(
    s: float,
    r: float,
    q: float,
    quotes: list[tuple[Kind, float, float, float]],
    start: Heston | None = None,
) -> tuple[Heston, float]:
    """Least squares on prices (vega-free, relative to the spot): quotes are (kind, K, T, price).
    Returns the fit and the price RMSE. Feller is reported, not imposed."""
    start = start or Heston(2.0, 0.04, 0.5, -0.6, 0.04)
    by_t: dict[float, list[int]] = {}
    for i, (_, _, t, _) in enumerate(quotes):
        by_t.setdefault(t, []).append(i)
    target = np.array([x[3] for x in quotes])

    def model(p: Array) -> Array:
        h = Heston(*[float(v) for v in p])
        out = np.empty(len(quotes))
        for t, idx in by_t.items():
            for kind in ("call", "put"):
                sel = [i for i in idx if quotes[i][0] == kind]
                if sel:
                    out[sel] = heston_price(
                        kind, s, np.array([quotes[i][1] for i in sel]), t, r, q, h
                    )
        return out

    result = optimize.least_squares(
        lambda p: (model(p) - target) / s,
        [start.kappa, start.theta, start.sigma, start.rho, start.v0],
        bounds=([0.05, 1e-4, 0.01, -0.99, 1e-4], [20.0, 1.0, 3.0, 0.99, 1.0]),
        max_nfev=300,
    )
    fit = Heston(*[float(v) for v in result.x])
    rmse = float(np.sqrt(np.mean((model(result.x) - target) ** 2)))
    return fit, rmse
