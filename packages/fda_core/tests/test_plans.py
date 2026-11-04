"""Golden tests for M24: DCA rule tables, cohort accounting, rebalancing, Kelly."""

import math

import numpy as np
import pandas as pd
import pytest

from fda_core.analyses.plans import (
    POLICIES,
    KellyParams,
    kelly,
    monthly_irr,
    rebalance_sim,
    simulate_plan,
    vix_multiplier,
)
from fda_core.registry import AnalysisInputs

FLAT = np.full(12, 100.0)
ABOVE = np.full(12, 90.0)  # 200-DMA below the price: trend "up"
CALM = np.full(12, 14.0)


def test_vix_rule_table() -> None:
    assert [vix_multiplier(v) for v in (10, 14.99, 15, 20, 20.01, 30, 30.01, 80)] == [
        0.5,
        0.5,
        1.0,
        1.0,
        1.5,
        1.5,
        2.0,
        2.0,
    ]
    assert vix_multiplier(None) == 1.0
    assert vix_multiplier(float("nan")) == 1.0


def test_flat_market_accounting() -> None:
    for variant in ("lump_sum", "dca", "trend_filtered", "vix_scaled"):
        out = simulate_plan(variant, FLAT, ABOVE, CALM, 100.0, 0.0)  # type: ignore[arg-type]
        assert out["wealth"] == pytest.approx(out["contributed"]), variant
    dca = simulate_plan("dca", FLAT, ABOVE, CALM, 100.0, 0.0)
    assert dca["contributed"] == 1200
    assert dca["irr"] == pytest.approx(0.0, abs=1e-9)
    vix = simulate_plan("vix_scaled", FLAT, ABOVE, CALM, 100.0, 0.0)
    assert vix["contributed"] == 600  # VIX 14 → 0.5×


def test_trend_filter_pauses_then_doubles() -> None:
    prices = np.full(6, 100.0)
    sma = np.array([110, 110, 90, 90, 90, 90.0])  # below the average for 2 months
    out = simulate_plan("trend_filtered", prices, sma, np.full(6, 18.0), 100.0, 0.0)
    # months 0–1 paused (200 in cash); month 2 invests 2× (100 new + 100 from cash);
    # months 3–5 invest 100 each
    assert out["cash_left"] == pytest.approx(100.0)
    assert out["wealth"] == pytest.approx(600.0)
    assert out["contributed"] == 600


def test_hybrid_pause_and_boost() -> None:
    prices = np.full(4, 100.0)
    below = np.full(4, 110.0)
    calm_below = simulate_plan("hybrid", prices, below, np.full(4, 15.0), 100.0, 0.0)
    assert calm_below["cash_left"] == pytest.approx(400.0)  # paused all along
    stressed = simulate_plan("hybrid", prices, np.full(4, 90.0), np.full(4, 35.0), 100.0, 0.0)
    assert stressed["contributed"] > 400  # +50 % boost funded by new money


def test_value_averaging_invests_less_in_rising_markets() -> None:
    rising = 100 * 1.03 ** np.arange(12)
    va = simulate_plan("value_averaging", rising, rising * 0.9, CALM, 100.0, 0.0, va_growth=0.0)
    dca = simulate_plan("dca", rising, rising * 0.9, CALM, 100.0, 0.0)
    assert va["cash_left"] > 0
    assert va["wealth"] < dca["wealth"]


def test_monthly_irr_round_trip() -> None:
    r = 0.07
    m = (1 + r) ** (1 / 12) - 1
    flows = [100.0] * 24
    terminal = sum(100 * (1 + m) ** (24 - k) for k in range(24))
    assert monthly_irr(flows, terminal) == pytest.approx(r, abs=1e-9)


def test_rebalancing_policies() -> None:
    days = pd.bdate_range("2020-01-01", periods=504)
    rng = np.random.default_rng(3)
    returns = pd.DataFrame(rng.normal(0.0003, 0.01, (504, 3)), index=days, columns=list("ABC"))
    target = np.array([0.5, 0.3, 0.2])
    monthly = rebalance_sim(returns, target, POLICIES["monthly"], 10)
    none = rebalance_sim(returns, target, POLICIES["none"], 10)
    assert none["turnover_per_year"] == 0
    assert monthly["tracking_error"] < none["tracking_error"]
    assert monthly["rebalances_per_year"] == pytest.approx(23 / 2, abs=1)
    same = pd.DataFrame(
        np.tile(rng.normal(0, 0.01, (504, 1)), (1, 3)), index=days, columns=list("ABC")
    )
    flat = rebalance_sim(same, target, POLICIES["band_5pt"], 10)
    assert flat["tracking_error"] == pytest.approx(0, abs=1e-12)  # identical assets never drift


def test_kelly_matches_the_moments() -> None:
    days = pd.bdate_range("2010-01-01", periods=252 * 10)
    rng = np.random.default_rng(5)
    r = rng.normal(0.0004, 0.01, len(days))
    close = pd.Series(100 * np.exp(np.cumsum(r)), index=days)
    inputs = AnalysisInputs(
        {"prices": pd.DataFrame({"adj_close": close, "close": close})},
        {"annualisation": 252, "price_field": "adj_close", "rf": 0.0},
    )
    out = kelly(inputs, KellyParams()).payload
    lr = np.diff(np.log(close.to_numpy()))
    mu = lr.mean() * 252 + lr.var(ddof=1) * 252 / 2
    assert out["kelly_full"] == pytest.approx(mu / (lr.var(ddof=1) * 252), rel=1e-6)
    assert out["kelly_low"] < out["kelly_full"] < out["kelly_high"]
    assert out["kelly_fractional"] == pytest.approx(out["kelly_full"] / 2)
    assert math.isfinite(out["years"])
