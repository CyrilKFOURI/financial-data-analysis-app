"""Golden tests for the option engine (DV-002): reference prices, Greeks by finite differences,
IV round trips, CRR, SVI, Breeden–Litzenberger and Heston."""

import math

import numpy as np
import pytest

from fda_core.options import (
    Heston,
    Svi,
    black76_price,
    bsm_greeks,
    bsm_price,
    butterfly_g,
    calibrate_heston,
    crr_price,
    fit_svi,
    heston_price,
    implied_vol,
    rnd_breeden_litzenberger,
)


def ncdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def reference_bsm(kind: str, s: float, k: float, t: float, r: float, q: float, v: float) -> float:
    """Independent textbook implementation (math.erf), the reference for 1e-8 agreement."""
    d1 = (math.log(s / k) + (r - q + v * v / 2) * t) / (v * math.sqrt(t))
    d2 = d1 - v * math.sqrt(t)
    if kind == "call":
        return s * math.exp(-q * t) * ncdf(d1) - k * math.exp(-r * t) * ncdf(d2)
    return k * math.exp(-r * t) * ncdf(-d2) - s * math.exp(-q * t) * ncdf(-d1)


CASES = [
    ("call", 42.0, 40.0, 0.5, 0.10, 0.0, 0.20),
    ("put", 42.0, 40.0, 0.5, 0.10, 0.0, 0.20),
    ("call", 100.0, 120.0, 2.0, 0.03, 0.02, 0.35),
    ("put", 100.0, 80.0, 0.05, 0.01, 0.015, 0.6),
]


@pytest.mark.parametrize(("kind", "s", "k", "t", "r", "q", "v"), CASES)
def test_bsm_matches_the_reference_and_hull(
    kind: str, s: float, k: float, t: float, r: float, q: float, v: float
) -> None:
    price = float(bsm_price(kind, s, k, t, r, q, v))  # type: ignore[arg-type]
    assert price == pytest.approx(reference_bsm(kind, s, k, t, r, q, v), abs=1e-8)
    iv = implied_vol(kind, price, s, k, t, r, q)  # type: ignore[arg-type]
    assert iv is not None
    assert float(bsm_price(kind, s, k, t, r, q, iv)) == pytest.approx(price, abs=1e-10)  # type: ignore[arg-type]


def test_hull_example_and_parity() -> None:
    call = float(bsm_price("call", 42, 40, 0.5, 0.1, 0.0, 0.2))
    put = float(bsm_price("put", 42, 40, 0.5, 0.1, 0.0, 0.2))
    assert round(call, 2) == 4.76  # Hull, Options, Futures and Other Derivatives, ex. 15.6
    assert round(put, 2) == 0.81
    assert call - put == pytest.approx(42 - 40 * math.exp(-0.05), abs=1e-12)
    assert float(black76_price("call", 100, 100, 1, 0.05, 0.2)) == pytest.approx(
        math.exp(-0.05) * (100 * ncdf(0.1) - 100 * ncdf(-0.1)), abs=1e-10
    )


@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_match_finite_differences(kind: str) -> None:
    s, k, t, r, q, v = 105.0, 100.0, 0.75, 0.04, 0.02, 0.28
    g = {n: float(x) for n, x in bsm_greeks(kind, s, k, t, r, q, v).items()}  # type: ignore[arg-type]

    def p(**kw: float) -> float:
        args = {"s": s, "k": k, "t": t, "r": r, "q": q, "sigma": v} | kw
        return float(bsm_price(kind, **args))  # type: ignore[arg-type]

    def delta(**kw: float) -> float:
        args = {"s": s, "k": k, "t": t, "r": r, "q": q, "sigma": v} | kw
        return float(bsm_greeks(kind, **args)["delta"])  # type: ignore[arg-type]

    h = 1e-4
    assert g["delta"] == pytest.approx((p(s=s + h) - p(s=s - h)) / (2 * h), abs=1e-6)
    assert g["gamma"] == pytest.approx((p(s=s + h) - 2 * p() + p(s=s - h)) / h**2, abs=1e-5)
    assert g["vega"] == pytest.approx((p(sigma=v + h) - p(sigma=v - h)) / (2 * h), abs=1e-6)
    assert g["theta"] == pytest.approx(-(p(t=t + h) - p(t=t - h)) / (2 * h), abs=1e-6)
    assert g["rho"] == pytest.approx((p(r=r + h) - p(r=r - h)) / (2 * h), abs=1e-6)
    assert g["vanna"] == pytest.approx(
        (delta(sigma=v + h) - delta(sigma=v - h)) / (2 * h), abs=1e-6
    )
    vega = lambda sig: float(bsm_greeks(kind, s, k, t, r, q, sig)["vega"])  # type: ignore[arg-type]  # noqa: E731
    assert g["volga"] == pytest.approx((vega(v + h) - vega(v - h)) / (2 * h), abs=1e-5)
    assert g["charm"] == pytest.approx(-(delta(t=t + h) - delta(t=t - h)) / (2 * h), abs=1e-6)


def test_iv_outside_bounds_is_none() -> None:
    assert implied_vol("call", 0.5, 100, 50, 1, 0.0, 0.0) is None  # below intrinsic
    assert implied_vol("call", 101.0, 100, 50, 1, 0.0, 0.0) is None  # above the spot
    assert implied_vol("put", 0.0, 100, 100, 1, 0.0, 0.0) is None


def test_crr_converges_and_values_early_exercise() -> None:
    european = float(bsm_price("put", 50, 52, 2, 0.05, 0.0, 0.3))
    tree_eu = crr_price("put", 50, 52, 2, 0.05, 0.0, 0.3, steps=2000, american=False)
    assert tree_eu == pytest.approx(european, abs=2e-3)
    american = crr_price("put", 50, 52, 2, 0.05, 0.0, 0.3, steps=2000)
    assert american > european + 0.1  # early exercise premium of a deep-ish put
    call_am = crr_price("call", 50, 52, 2, 0.05, 0.0, 0.3, steps=2000)
    assert call_am == pytest.approx(float(bsm_price("call", 50, 52, 2, 0.05, 0.0, 0.3)), abs=2e-3)


def test_svi_fit_density_and_butterfly_check() -> None:
    true = Svi(a=0.02, b=0.15, rho=-0.5, m=0.02, s=0.15)
    t, forward, r = 0.5, 100.0, 0.02
    k = np.linspace(-0.5, 0.4, 40)
    fit, rmse = fit_svi(k, true.iv(k, t), t)
    assert rmse < 1e-5
    assert fit.rho == pytest.approx(-0.5, abs=1e-2)
    assert (butterfly_g(fit, np.linspace(-1, 1, 200)) > 0).all()
    strikes = np.linspace(1, 400, 20000)  # wide: negative skew puts mass far left
    density = rnd_breeden_litzenberger(fit, forward, t, r, strikes)
    mass = np.trapezoid(density, strikes)
    assert mass == pytest.approx(1.0, abs=1e-3)
    mean = np.trapezoid(density * strikes, strikes) / mass
    assert mean == pytest.approx(forward, rel=1e-3)
    bad = Svi(a=-0.05, b=0.9, rho=0.95, m=0.0, s=0.01)  # violates Durrleman around the money
    assert (butterfly_g(bad, np.linspace(-0.3, 0.3, 200)) < 0).any()


def test_heston_limits_parity_and_calibration() -> None:
    s, r, q, t = 100.0, 0.03, 0.01, 1.0
    near_bs = Heston(kappa=1.5, theta=0.04, sigma=1e-3, rho=0.0, v0=0.04)
    strikes = np.array([80.0, 100.0, 120.0])
    hp = heston_price("call", s, strikes, t, r, q, near_bs)
    assert hp == pytest.approx(bsm_price("call", s, strikes, t, r, q, 0.2), abs=2e-4)
    p = Heston(kappa=2.0, theta=0.05, sigma=0.6, rho=-0.7, v0=0.03)
    calls = heston_price("call", s, strikes, t, r, q, p)
    puts = heston_price("put", s, strikes, t, r, q, p)
    assert calls - puts == pytest.approx(
        s * math.exp(-q * t) - strikes * math.exp(-r * t), abs=1e-10
    )
    quotes = []
    for tt in (0.25, 0.5, 1.0):
        ks = np.array([85.0, 95.0, 100.0, 105.0, 115.0])
        for kk, price in zip(ks, heston_price("call", s, ks, tt, r, q, p), strict=True):
            quotes.append(("call", float(kk), tt, float(price)))
    fit, rmse = calibrate_heston(s, r, q, quotes)  # type: ignore[arg-type]
    assert rmse < 1e-3
    assert fit.rho == pytest.approx(-0.7, abs=0.1)
    assert fit.v0 == pytest.approx(0.03, abs=0.005)


def test_vectorised_iv_matches_the_scalar_solver() -> None:
    from fda_core.options import implied_vol_vec

    rng = np.random.default_rng(0)
    n = 400
    is_call = rng.random(n) < 0.5
    k = rng.uniform(50, 150, n)
    t = rng.uniform(1 / 365, 2, n)
    true = rng.uniform(0.05, 1.5, n)
    price = np.where(
        is_call,
        bsm_price("call", 100, k, t, 0.03, 0.01, true),
        bsm_price("put", 100, k, t, 0.03, 0.01, true),
    )
    vec = implied_vol_vec(is_call, price, 100, k, t, 0.03, 0.01)
    for i in range(n):
        kind = "call" if is_call[i] else "put"
        args = (100, float(k[i]), float(t[i]), 0.03, 0.01)
        scalar = implied_vol(kind, float(price[i]), *args)
        if scalar is None:  # tiny far-OTM prices: both solvers refuse
            assert np.isnan(vec[i]) or abs(price[i]) < 1e-12
            continue
        vega = float(bsm_greeks(kind, 100, k[i], t[i], 0.03, 0.01, scalar)["vega"])
        if vega > 1e-3:  # identifiable: the volatilities agree
            assert vec[i] == pytest.approx(scalar, abs=1e-8)
        else:  # flat price in σ (deep OTM): both reproduce the price
            assert float(bsm_price(kind, *args[:2], args[2], 0.03, 0.01, vec[i])) == pytest.approx(
                float(price[i]), abs=1e-9
            )
    bad = implied_vol_vec(
        np.array([True]), np.array([0.5]), 100, np.array([50.0]), np.array([1.0]), 0, 0
    )
    assert np.isnan(bad[0])  # below intrinsic
