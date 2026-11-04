"""Golden tests for M22: scoring table, required return, CMA, allocations, glide path."""

import math

import numpy as np
import pandas as pd
import pytest

from fda_core.analyses.invest import (
    CLASSES,
    MODEL_ALLOCATIONS,
    VOL_BANDS,
    AllocationParams,
    CmaParams,
    ProfileParams,
    capital_market_assumptions,
    classify_holding,
    glide_score,
    investor_profile,
    monte_carlo_drawdown,
    required_return,
    score_profile,
    strategic_allocation,
)
from fda_core.registry import AnalysisInputs, InsufficientDataError

BALANCED = {
    "horizon": "10-20y",
    "income_stability": "stable",
    "emergency_fund": "3-6m",
    "savings_rate": "5-15%",
    "debt": "low",
    "liquidity_needs": "none",
    "drawdown_reaction": "hold",
    "max_loss": "20%",
    "preference": "balanced",
    "knowledge": "good",
    "experience": "3-10",
}


def test_scoring_is_deterministic_and_takes_the_lower_dimension() -> None:
    a = score_profile(BALANCED)
    assert a == score_profile(dict(BALANCED))  # same answers → same score
    assert a["capacity"] == round(np.mean([6, 5, 5, 4, 6, 6]))  # 5.33 → 5
    assert a["willingness"] == round(np.mean([5, 4, 4, 5, 5]))  # 4.6 → 5
    assert a["risk_score"] == 5
    assert a["max_drawdown_tolerance"] == 0.20  # stated 20 % < table 28 %
    timid = score_profile(
        BALANCED | {"drawdown_reaction": "sell_all", "max_loss": "5%", "preference": "stability"}
    )
    assert timid["binding"] == "willingness"
    assert timid["risk_score"] == timid["willingness"] < timid["capacity"]
    assert len(a["contributions"]) == 11
    with pytest.raises(InsufficientDataError, match="Unanswered"):
        score_profile({"horizon": "<2y"})
    with pytest.raises(InsufficientDataError, match="unknown answer"):
        score_profile(BALANCED | {"debt": "huge"})


def test_required_return_solves_the_savings_equation() -> None:
    r = required_return(goal=200_000, years=10, current=50_000, monthly_saving=500)
    assert r is not None
    m = (1 + r) ** (1 / 12) - 1
    value = 50_000 * (1 + r) ** 10 + 500 * ((1 + m) ** 120 - 1) / m
    assert value == pytest.approx(200_000, rel=1e-8)
    assert required_return(100, 10, 1_000, 0) == pytest.approx(0.1 ** (1 / 10) - 1)  # can shrink
    assert required_return(10**9, 1, 10, 0) is None  # unreachable


def proxies(seed: int = 0, years: int = 20) -> dict[str, pd.Series]:
    rng = np.random.default_rng(seed)
    months = pd.date_range("2005-01-31", periods=years * 12, freq="ME")
    spec = {
        "IEF": (0.003, 0.02),
        "LQD": (0.004, 0.025),
        "HYG": (0.005, 0.03),
        "SPY": (0.008, 0.045),
        "EFA": (0.006, 0.05),
        "EEM": (0.007, 0.06),
        "VNQ": (0.006, 0.06),
        "GLD": (0.004, 0.045),
        "DBC": (0.002, 0.055),
        "BTC-USD": (0.03, 0.2),
    }
    common = rng.normal(0, 1, len(months))
    out = {}
    for symbol, (mu, vol) in spec.items():
        beta = 0.8 if symbol in ("SPY", "EFA", "EEM", "VNQ") else 0.1
        r = mu + vol * (beta * common + math.sqrt(1 - beta**2) * rng.normal(0, 1, len(months)))
        out[symbol] = pd.Series(100 * np.cumprod(1 + r), index=months)
    return out


SERIES = {
    c: pd.Series([v], index=pd.to_datetime(["2026-10-22"]))
    for c, v in {"DGS10": 4.3, "BAMLC0A0CM": 0.9, "BAMLH0A0HYM2": 3.0, "T10YIE": 2.3}.items()
} | {"DTB3": pd.Series(4.0, index=pd.date_range("2004-12-01", "2026-10-22", freq="D"))}


def test_cma_methods_and_ranges() -> None:
    inputs = AnalysisInputs(
        {"cma_prices": proxies(), "cma_series": SERIES, "cma_yields": {"SPY": 0.013}}, {}
    )
    out = capital_market_assumptions(inputs, CmaParams(method="building_block"))
    rows = {r["class"]: r for r in out["rows"]}
    assert rows["cash"]["expected_return"] == pytest.approx(0.04)
    assert rows["govt_bonds"]["expected_return"] == pytest.approx(0.043)
    assert rows["ig_credit"]["expected_return"] == pytest.approx(0.043 + 0.009 - 0.002)
    assert rows["hy_credit"]["expected_return"] == pytest.approx(0.043 + 0.03 - 0.02)
    assert rows["us_equity"]["expected_return"] == pytest.approx(0.013 + 0.015 + 0.023)
    hist = capital_market_assumptions(inputs, CmaParams(method="historical"))
    spy = next(r for r in hist["rows"] if r["class"] == "us_equity")
    monthly = proxies()["SPY"].pct_change().dropna()
    assert spy["historical"] == pytest.approx(math.expm1(12 * np.log1p(monthly).mean()))
    assert spy["range_low"] < spy["expected_return"] < spy["range_high"]
    assert out["corr"].loc["us_equity", "devx_equity"] > 0.4  # common factor
    override = capital_market_assumptions(inputs, CmaParams(overrides={"gold": 0.01}))
    assert next(r for r in override["rows"] if r["class"] == "gold")["source"] == "override"


def test_allocations_sum_to_one_and_hit_their_vol_bands() -> None:
    for score, weights in MODEL_ALLOCATIONS.items():
        assert sum(weights.values()) == pytest.approx(1.0)
        assert set(weights) <= set(CLASSES)
        assert score in VOL_BANDS
    inputs = AnalysisInputs(
        {
            "profile": {
                "name": "Test",
                "answers": BALANCED
                | {
                    "goal": {"amount": 300000, "years": 25, "current": 50000, "monthly_saving": 800}
                },
            },
            "cma_prices": proxies(),
            "cma_series": SERIES,
            "cma_yields": {},
        },
        {},
    )
    out = strategic_allocation(inputs, AllocationParams()).payload
    assert out["risk_score"] == 5
    assert out["label"].startswith("Model output")
    assert out["volatility"] is not None
    assert out["monte_carlo"]["mdd_p5"] <= out["monte_carlo"]["mdd_p50"] <= 0
    assert any(r["implementation"].startswith("SXR8.DE") for r in out["rows"])  # UCITS for EUR
    evidence = out["historical_evidence"]
    assert evidence is not None
    assert evidence["months"] >= 36
    assert set(evidence) >= {"model", "equal_weight", "beats_equal_weight_sharpe"}
    prof = investor_profile(inputs, ProfileParams()).payload
    assert prof["required_return"] is not None


def test_glide_path_and_classification() -> None:
    assert glide_score(6, None) == 6
    assert glide_score(6, 12) == 6
    assert glide_score(6, 9) == 5
    assert glide_score(6, 1) == 3
    assert glide_score(2, 0) == 1
    assert classify_holding("IWDA.AS", "etf", "NL") == "devx_equity"
    assert classify_holding("AAPL", "equity", "US") == "us_equity"
    assert classify_holding("MC.PA", "equity", "FR") == "devx_equity"
    assert classify_holding("BTC-USD", "crypto", None) == "crypto"
    assert classify_holding("XYZ", "etf", "US") == "unclassified"


def test_monte_carlo_is_seeded() -> None:
    inputs = AnalysisInputs({"cma_prices": proxies(), "cma_series": SERIES, "cma_yields": {}}, {})
    cma_out = capital_market_assumptions(inputs, CmaParams())
    a = monte_carlo_drawdown(MODEL_ALLOCATIONS[4], cma_out, 10)
    b = monte_carlo_drawdown(MODEL_ALLOCATIONS[4], cma_out, 10)
    assert a == b
    assert a is not None
    assert a["terminal_p5"] < a["terminal_p50"] < a["terminal_p95"]
