"""J4 risk: empyrical oracles, textbook backtests, EVT and drawdown goldens."""

import math

import empyrical
import numpy as np
import pandas as pd
import pytest

from fda_core.analyses.risk import (
    BacktestParams,
    CdarParams,
    DecayParams,
    DistributionRatioParams,
    DrawdownDistributionParams,
    DrawdownParams,
    EvtParams,
    ModelVarParams,
    PsrParams,
    RiskParams,
    RollingBetaParams,
    SortinoParams,
    VarParams,
    calmar_family,
    capm,
    cdar,
    christoffersen,
    distribution_ratios,
    downside_beta,
    drawdown_distribution,
    drawdown_episodes,
    drawdown_series,
    expected_shortfall,
    extreme_value,
    kupiec_pof,
    leverage_decay,
    model_var,
    psr,
    relative_set,
    risk_card,
    rolling_beta,
    sharpe,
    sortino,
    stress_windows,
    time_under_water,
    ulcer_pain,
    value_at_risk,
    var_backtest,
)
from fda_core.registry import AnalysisInputs, InsufficientDataError

CONV = {
    "annualisation": 252,
    "price_field": "adj_close",
    "has_volume": True,
    "range": "MAX",
    "currency": "USD",
}


def frame(returns: np.ndarray, start: str = "2015-01-02") -> pd.DataFrame:
    close = 100 * np.cumprod(np.r_[1.0, 1 + returns])
    index = pd.bdate_range(start, periods=len(close))
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "adj_close": close,
            "volume": 1.0,
        },
        index=index,
    )


def inputs(r: np.ndarray, bench: np.ndarray | None = None, **conv: object) -> AnalysisInputs:
    data = {"prices": frame(r)} | ({"other_prices": frame(bench)} if bench is not None else {})
    return AnalysisInputs(data=data, conventions=CONV | {"other_symbol": "BENCH"} | conv)


def series(r: np.ndarray) -> pd.Series:
    return pd.Series(r, index=pd.bdate_range("2015-01-05", periods=len(r)))


def gaussian(n: int = 1_500, seed: int = 1, mu: float = 0.0004, sigma: float = 0.012) -> np.ndarray:
    return np.random.default_rng(seed).normal(mu, sigma, n)


# ── drawdowns ────────────────────────────────────────────────────────────────


def test_drawdown_golden_and_episodes() -> None:
    # 20 flat sessions, then 110, 88, 92.4, 110.88, 99.79, 99.79
    r = np.r_[np.zeros(20), [0.10, -0.20, 0.05, 0.20, -0.10, 0.0]]
    out = drawdown_series(inputs(r), RiskParams()).payload
    assert out["max_drawdown"] == pytest.approx(88 / 110 - 1)
    eps = drawdown_episodes(inputs(r), DrawdownParams()).payload["episodes"]
    first = next(e for e in eps if e["recovered"])
    assert (first["decline_sessions"], first["recovery_sessions"]) == (1, 2)
    assert any(not e["recovered"] for e in eps)
    tuw = time_under_water(inputs(r), RiskParams()).payload
    assert tuw["current"] is not None


def test_max_drawdown_and_ratios_match_empyrical() -> None:
    r = gaussian()
    s = series(r)
    assert drawdown_series(inputs(r), RiskParams()).payload["max_drawdown"] == pytest.approx(
        empyrical.max_drawdown(s), rel=1e-9
    )
    assert sharpe(inputs(r), RiskParams()).payload["value"] == pytest.approx(
        empyrical.sharpe_ratio(s), rel=1e-9
    )
    assert sharpe(inputs(r), RiskParams(rf=0.02)).payload["value"] == pytest.approx(
        empyrical.sharpe_ratio(s, risk_free=0.02 / 252), rel=1e-9
    )
    assert sortino(inputs(r), SortinoParams(mar=0.0)).payload["value"] == pytest.approx(
        empyrical.sortino_ratio(s), rel=1e-9
    )
    ratios = distribution_ratios(inputs(r), DistributionRatioParams()).payload
    assert ratios["omega"] == pytest.approx(empyrical.omega_ratio(s), rel=1e-9)
    assert ratios["tail_ratio"] == pytest.approx(empyrical.tail_ratio(s), rel=1e-6)


def test_calmar_uses_cagr_over_max_drawdown() -> None:
    r = gaussian(seed=2)
    out = calmar_family(inputs(r), RiskParams()).payload
    assert out["calmar"] == pytest.approx(out["cagr"] / abs(out["max_drawdown"]))
    assert out["sterling"] is not None
    assert out["burke"] is not None
    up = ulcer_pain(inputs(r), RiskParams()).payload
    assert 0 < up["pain_index"] < up["ulcer_index"]


# ── VaR / ES ─────────────────────────────────────────────────────────────────


def test_historical_var_and_es_match_empyrical() -> None:
    r = gaussian(2_000, seed=3)
    var = {
        (row["method"], row["confidence"]): row
        for row in value_at_risk(inputs(r), VarParams()).payload["rows"]
    }
    es = {
        (row["method"], row["confidence"]): row
        for row in expected_shortfall(inputs(r), VarParams()).payload["rows"]
    }
    assert var[("historical", 0.95)]["loss"] == pytest.approx(
        -empyrical.value_at_risk(series(r), cutoff=0.05), rel=1e-9
    )
    assert es[("historical", 0.95)]["loss"] == pytest.approx(
        -empyrical.conditional_value_at_risk(series(r), cutoff=0.05), rel=1e-9
    )
    # On Gaussian data all models agree within sampling error, and ES > VaR.
    assert var[("gaussian", 0.99)]["loss"] == pytest.approx(
        var[("cornish_fisher", 0.99)]["loss"], rel=0.1
    )
    assert es[("gaussian", 0.99)]["loss"] > var[("gaussian", 0.99)]["loss"]
    assert var[("historical", 0.95)]["amount"] == pytest.approx(
        var[("historical", 0.95)]["loss"] * 10_000
    )


def test_gaussian_es_closed_form() -> None:
    r = gaussian(5_000, seed=4, mu=0.0)
    es = {
        (row["method"], row["confidence"]): row
        for row in expected_shortfall(inputs(r), VarParams(confidences=[0.99])).payload["rows"]
    }
    sigma = r.std(ddof=1)
    expected = sigma * math.exp(-(2.326348**2) / 2) / math.sqrt(2 * math.pi) / 0.01 - r.mean()
    assert es[("gaussian", 0.99)]["loss"] == pytest.approx(expected, rel=1e-4)


def test_kupiec_matches_the_basel_non_rejection_region() -> None:
    # Jorion: T = 250, p = 1 %, 95 % test → do not reject for N ≤ 6, reject for N ≥ 7.
    assert kupiec_pof(6, 250, 0.01)[1] > 0.05
    assert kupiec_pof(7, 250, 0.01)[1] < 0.05
    # Zero exceptions is also rejected by the LR test (VaR too conservative): LR = −2·250·ln 0.99.
    assert kupiec_pof(0, 250, 0.01)[0] == pytest.approx(-2 * 250 * math.log(0.99))


def test_christoffersen_flags_clustered_exceptions() -> None:
    clustered = np.zeros(500, dtype=bool)
    clustered[100:105] = True
    spread = np.zeros(500, dtype=bool)
    spread[::100] = True
    assert christoffersen(clustered)[1] < 0.05
    assert christoffersen(spread)[1] > 0.05


def test_var_backtest_on_gaussian_returns_is_green() -> None:
    out = var_backtest(inputs(gaussian(1_200, seed=5)), BacktestParams(method="gaussian")).payload
    assert out["traffic_light"]["zone"] in {"green", "yellow"}
    assert out["exception_rate"] == pytest.approx(0.01, abs=0.01)
    assert len(out["var"]) == len(out["realised"])


def test_model_var_and_evt() -> None:
    rng = np.random.default_rng(6)
    r = rng.standard_t(4, 3_000) * 0.01 / math.sqrt(2)
    rows = {
        row["method"]: row
        for row in model_var(inputs(r), ModelVarParams(n_sims=2_000)).payload["rows"]
    }
    assert set(rows) == {"historical", "garch_fhs", "bootstrap_mc", "student_t_mc"}
    assert rows["student_t_mc"]["var"] == pytest.approx(rows["historical"]["var"], rel=0.25)
    evt = extreme_value(inputs(r), EvtParams()).payload
    assert evt["xi"] == pytest.approx(0.25, abs=0.15)  # ξ = 1/ν for Student-t tails
    hist = evt["rows"][0]["historical_var"]
    assert evt["rows"][0]["var"] == pytest.approx(hist, rel=0.15)
    assert evt["return_levels"][-1]["loss"] > evt["return_levels"][0]["loss"]


# ── PSR, CDaR, drawdown distribution, card ───────────────────────────────────


def test_probabilistic_sharpe() -> None:
    r = gaussian(2_000, seed=7, mu=0.0006)
    out = psr(inputs(r), PsrParams()).payload
    assert 0.5 < out["psr"] <= 1
    tough = psr(inputs(r), PsrParams(benchmark_sharpe=out["sharpe_annualised"])).payload
    assert tough["psr"] == pytest.approx(0.5, abs=1e-6)
    assert out["min_track_record"] is not None


def test_cdar_drawdown_distribution_and_card() -> None:
    r = gaussian(seed=8)
    c = cdar(inputs(r), CdarParams()).payload
    assert c["drawdown_at_risk"] <= c["cdar"] <= c["max_drawdown"]
    d = drawdown_distribution(inputs(r), DrawdownDistributionParams(n_sims=200)).payload
    assert 0 < d["bootstrap_mdd_median"] < d["bootstrap_mdd_p95"]
    card = risk_card(inputs(r), RiskParams()).payload
    assert card["bucket"] == "moderate"  # 1.2 % daily ≈ 19 % annualised
    assert "10 000 USD" in card["sentence"]


# ── benchmark-relative ───────────────────────────────────────────────────────


def test_capm_and_relative_set_match_empyrical() -> None:
    rng = np.random.default_rng(9)
    b = rng.normal(0.0004, 0.01, 1_500)
    a = 0.0002 + 1.3 * b + rng.normal(0, 0.006, 1_500)
    out = capm(inputs(a, b), RiskParams()).payload
    _alpha, beta = empyrical.alpha_beta(series(a), series(b))
    assert out["beta"] == pytest.approx(beta, rel=1e-6)
    assert out["alpha_annualised"] == pytest.approx(0.0002 * 252, abs=0.03)
    rel = relative_set(inputs(a, b), RiskParams()).payload
    assert rel["up_capture"] == pytest.approx(a[b > 0].mean() / b[b > 0].mean())
    assert rel["tracking_error"] == pytest.approx((a - b).std(ddof=1) * math.sqrt(252))
    down = downside_beta(inputs(a, b), RiskParams()).payload
    assert down["downside_beta"] == pytest.approx(1.3, abs=0.1)
    roll = rolling_beta(inputs(a, b), RollingBetaParams(window=252)).payload
    assert roll["beta"][-1] == pytest.approx(1.3, abs=0.15)


def test_benchmark_missing_is_explained() -> None:
    with pytest.raises(InsufficientDataError, match="BENCH"):
        capm(inputs(gaussian()), RiskParams())


def test_leverage_decay_matches_the_approximation() -> None:
    b = gaussian(756, seed=10, mu=0.0, sigma=0.015)
    product = 3 * b - 0.0001  # daily-reset 3× with a small fee
    out = leverage_decay(inputs(product, b), DecayParams(leverage=3)).payload
    assert out["decay_approximation"] == pytest.approx(-3 * 0.015**2 * 252 * 3, rel=0.25)
    # In log terms: log(1 + R_k) − k·log(1 + R_b) ≈ −k(k − 1)/2·σ²·T (+ fees).
    assert out["decay_realised"] == pytest.approx(
        out["decay_approximation"] - 0.0001 * 756, abs=0.05
    )


def test_stress_windows_cover_only_available_history() -> None:
    r = gaussian(2_500, seed=11)  # 2015 → 2024
    rows = {row["window"]: row for row in stress_windows(inputs(r), RiskParams()).payload["rows"]}
    assert rows["GFC"]["covered"] is False
    assert rows["COVID crash"]["covered"] is True
    assert rows["COVID crash"]["max_drawdown"] <= 0
