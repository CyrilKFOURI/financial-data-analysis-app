"""Option strategy engine tests, incl. M28 acceptance: payoffs match the options notebook's P&L
functions for the four implemented presets, Greeks aggregate as the sum of legs, and the iron
condor max loss equals wing width − credit."""

import numpy as np
import pandas as pd
import pytest

from fda_core.option_strategies import (
    Leg,
    aggregate_greeks,
    breakevens,
    build_preset,
    expiry_pnl,
    extremes,
    margin_estimate,
    net_premium,
    option_rule_backtest,
    position_pnl,
    probability_of_profit,
    strike_for_delta,
)
from fda_core.options import bsm_greeks, bsm_price

S = np.linspace(50, 150, 201)
S0 = 100.0


# Notebook (Hull · Option strategies/options_strat.ipynb) P&L functions, verbatim formulas.
def covered_call_pnl(s_t, s_init, k, premium):  # type: ignore[no-untyped-def]
    return (s_t - s_init) + premium - np.maximum(s_t - k, 0)


def protective_put_pnl(s_t, s_init, k, premium):  # type: ignore[no-untyped-def]
    return (s_t - s_init) + np.maximum(k - s_t, 0) - premium


def covered_put_pnl(s_t, s_init, k, premium):  # type: ignore[no-untyped-def]
    return (s_init - s_t) + premium - np.maximum(k - s_t, 0)


def cash_secured_put_pnl(s_t, k, premium):  # type: ignore[no-untyped-def]
    return premium - np.maximum(k - s_t, 0)


def test_payoffs_match_the_notebook() -> None:
    legs = build_preset("covered_call", S0, 0.2, width=0.05)
    call = legs[1]
    assert np.allclose(expiry_pnl(legs, S), covered_call_pnl(S, S0, call.strike, call.premium))
    legs = build_preset("protective_put", S0, 0.2, width=0.05)
    put = legs[1]
    assert np.allclose(expiry_pnl(legs, S), protective_put_pnl(S, S0, put.strike, put.premium))
    legs = build_preset("covered_put", S0, 0.2, width=0.05)
    put = legs[1]
    assert np.allclose(expiry_pnl(legs, S), covered_put_pnl(S, S0, put.strike, put.premium))
    legs = build_preset("cash_secured_put", S0, 0.2, width=0.05)
    put = legs[0]
    assert np.allclose(expiry_pnl(legs, S), cash_secured_put_pnl(S, put.strike, put.premium))


def test_greeks_are_the_sum_of_legs() -> None:
    legs = build_preset("iron_condor", S0, 0.25, days=45, r=0.04)
    total = aggregate_greeks(legs, S0, 0.04, 0.0)
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        by_leg = sum(
            leg.side
            * leg.quantity
            * float(bsm_greeks(leg.kind, S0, leg.strike, 45 / 365, 0.04, 0.0, 0.25)[name])  # type: ignore[arg-type]
            for leg in legs
        )
        assert total[name] == pytest.approx(by_leg)
    covered = aggregate_greeks(build_preset("covered_call", S0, 0.2), S0, 0.0, 0.0)
    call_delta = float(bsm_greeks("call", S0, S0 * 1.05, 30 / 365, 0.0, 0.0, 0.2)["delta"])
    assert covered["delta"] == pytest.approx(1 - call_delta)


def test_iron_condor_max_loss_is_wing_width_minus_credit() -> None:
    legs = build_preset("iron_condor", S0, 0.25, width=0.05, wing=0.05)
    credit = net_premium(legs)
    assert credit > 0
    ext = extremes(legs, S0)
    assert ext["max_loss"] == pytest.approx(-(5.0 - credit), abs=1e-9)
    assert ext["max_profit"] == pytest.approx(credit, abs=1e-9)
    be = breakevens(S, expiry_pnl(legs, S))
    assert be == pytest.approx([95 - credit, 105 + credit], abs=1e-6)
    assert margin_estimate(legs, S0)["requirement"] == pytest.approx(5.0 - credit, abs=1e-9)


def test_unbounded_and_zero_cost_collar() -> None:
    assert extremes(build_preset("short_strangle", S0, 0.2), S0)["max_loss"] is None
    assert extremes(build_preset("long_straddle", S0, 0.2), S0)["max_profit"] is None
    collar = build_preset("zero_cost_collar", S0, 0.2, width=0.05)
    assert net_premium(collar) == pytest.approx(0.0, abs=1e-8)
    assert collar[2].strike > S0
    naked = margin_estimate(build_preset("short_strangle", S0, 0.2), S0)
    assert naked["requirement"] > 10  # ≥ 10 % of spot per short leg


def test_calendar_is_valued_at_the_first_expiry() -> None:
    legs = build_preset("calendar_call", S0, 0.2, days=30)
    pnl = expiry_pnl(legs, np.array([S0]))[0]
    long_left = float(bsm_price("call", S0, S0, 30 / 365, 0.0, 0.0, 0.2))
    assert pnl == pytest.approx(long_left - legs[1].premium + legs[0].premium)
    assert pnl > 0  # a calendar profits when the spot pins the strike
    assert extremes(legs, S0)["max_loss"] is not None


def test_pnl_today_and_probability_of_profit() -> None:
    legs = build_preset("long_call_butterfly", S0, 0.2)
    assert position_pnl(legs, np.array([S0]), 0.0)[0] == pytest.approx(0.0, abs=1e-9)
    long_call = [build_preset("bull_call_spread", S0, 0.2)[0]]
    pop = probability_of_profit(long_call, S0, 0.2, drift=0.0)
    be = S0 + long_call[0].premium
    assert 0.2 < pop < 0.5
    from scipy import stats

    t = 30 / 365
    exact = 1 - stats.norm.cdf((np.log(be / S0) + 0.02 * t) / (0.2 * np.sqrt(t)))
    assert pop == pytest.approx(exact, abs=2e-3)
    short_put = [Leg("put", -1, 90.0, 30, 1.0, 1.0, 0.2)]
    assert probability_of_profit(short_put, S0, 0.2, 0.0) > 0.8


def test_option_rule_backtest_accounts_and_manages() -> None:
    days = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(4)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days)))), index=days)
    iv = pd.Series(0.2, index=days)
    put = option_rule_backtest(close, iv, "put_write_atm", days=30)
    assert put["event"].str.contains("expiry").sum() >= 10
    # per cycle, the summed daily P&L = credit − intrinsic at expiry (marks telescope)
    opens = list(put.index[put["event"].str.contains("open")])
    first, expiry = opens[0], put.index[put["event"].str.contains("expiry")][0]
    cycle = put.loc[first:expiry, "pnl"].sum()
    credit = float(bsm_price("put", close[first], close[first], 30 / 365, 0.0, 0.0, 0.2))
    assert cycle == pytest.approx(credit - max(close[first] - close[expiry], 0.0), abs=1e-9)
    strangle = option_rule_backtest(
        close, iv, "short_strangle_16d", days=45, profit_take=0.5, close_dte=21
    )
    assert strangle["event"].str.contains("managed").sum() > 0
    assert strangle["event"].str.contains("expiry").sum() == 0
    covered = option_rule_backtest(close, iv, "covered_call_30d", cost_per_option=0.05)
    gross = option_rule_backtest(close, iv, "covered_call_30d")
    assert covered["pnl"].sum() < gross["pnl"].sum()
    assert strike_for_delta("call", S0, 0.3, 30, 0.0, 0.0, 0.2) > S0


def test_covered_call_cycle_matches_the_closed_form() -> None:
    days = pd.bdate_range("2010-01-01", periods=300)
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(days)))), index=days)
    iv = pd.Series(0.16, index=days)
    frame = option_rule_backtest(close, iv, "covered_call_30d", days=30)
    start = frame.index[frame["event"].str.contains("open")][0]
    end = frame.index[frame["event"].str.contains("expiry")][0]
    call_k = strike_for_delta("call", close[start], 0.30, 30, 0.0, 0.0, 0.16)
    premium = float(bsm_price("call", close[start], call_k, 30 / 365, 0.0, 0.0, 0.16))
    expected = (close[end] - close[start]) + premium - max(close[end] - call_k, 0.0)
    assert frame.loc[start:end, "pnl"].sum() == pytest.approx(expected, abs=1e-9)
