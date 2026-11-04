"""Time-Series Lab validation on processes with known properties (J3 "done when")."""

import math

import numpy as np
import pandas as pd
import pytest
from arch import arch_model
from arch.univariate import Normal
from scipy.signal import fftconvolve

from fda_core.analyses.ts_breaks import (
    BreakParams,
    CusumFilterParams,
    ExplosiveParams,
    bai_perron,
    cusum_filter,
    event_sampling,
    explosiveness,
    structural_breaks,
)
from fda_core.analyses.ts_common import SeriesParams, ffd_weights, frac_diff, select_series
from fda_core.analyses.ts_info import (
    EntropyParams,
    MicrostructureParams,
    entropy,
    kontoyiannis_entropy,
    lempel_ziv_entropy,
    microstructure,
    plug_in_entropy,
    roll_spread,
)
from fda_core.analyses.ts_memory import (
    DependenceParams,
    DynamicsParams,
    FracDiffParams,
    SpectrumParams,
    StationarityParams,
    concentration,
    decomposition,
    distribution_dynamics,
    fractional_differentiation,
    gph,
    hhi_concentration,
    hurst_dfa,
    hurst_rs,
    long_memory,
    serial_dependence,
    stationarity,
    vol_dynamics,
    volume_activity,
)
from fda_core.analyses.ts_models import (
    ArimaParams,
    GarchParams,
    KalmanParams,
    OuParams,
    RegimeParams,
    arima,
    garch,
    kalman,
    ou_mean_reversion,
    regimes,
)
from fda_core.analyses.ts_relations import (
    LeadLagParams,
    PairParams,
    PlaceboParams,
    lead_lag,
    pair_dynamics,
    placebo,
    transformations,
)
from fda_core.registry import AnalysisInputs, InsufficientDataError

EQUITY = {
    "annualisation": 252,
    "price_field": "adj_close",
    "has_volume": True,
    "range": "MAX",
    "calendar": "exchange",
}


def frame(
    log_price: np.ndarray,
    start: str = "2010-01-04",
    spread: float = 0.002,
    volume: np.ndarray | None = None,
) -> pd.DataFrame:
    close = np.exp(log_price)
    index = pd.bdate_range(start, periods=len(close))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * (1 + spread),
            "low": close * (1 - spread),
            "close": close,
            "adj_close": close,
            "volume": volume if volume is not None else np.full(len(close), 1e6),
        },
        index=index,
    )


def inputs(f: pd.DataFrame, other: pd.DataFrame | None = None, **conv: object) -> AnalysisInputs:
    data = {"prices": f} | ({"other_prices": other} if other is not None else {})
    return AnalysisInputs(data=data, conventions=EQUITY | conv)


def random_walk(n: int = 1_500, seed: int = 0, sigma: float = 0.01) -> np.ndarray:
    return np.log(100) + np.cumsum(np.random.default_rng(seed).normal(0, sigma, n))


# ── foundations ──────────────────────────────────────────────────────────────


def test_ffd_weights_golden() -> None:
    w = ffd_weights(0.5, threshold=1e-3)
    assert w[:3] == pytest.approx([1.0, -0.5, -0.125])
    assert ffd_weights(1.0).tolist() == [1.0, -1.0]  # d = 1 is the first difference


def test_frac_diff_with_d_one_is_the_first_difference() -> None:
    s = pd.Series(np.arange(10.0) ** 2, index=pd.bdate_range("2020-01-01", periods=10))
    assert frac_diff(s, 1.0).to_numpy() == pytest.approx(np.diff(s.to_numpy()))


def test_series_transformations() -> None:
    f = frame(random_walk(300))
    other = frame(random_walk(300, seed=1))
    assert len(select_series(inputs(f), SeriesParams(series="returns"))) == 299
    ratio = select_series(inputs(f, other), SeriesParams(series="ratio"))
    assert ratio.iloc[0] == pytest.approx(
        np.log(f["adj_close"].iloc[0] / other["adj_close"].iloc[0])
    )
    with pytest.raises(InsufficientDataError):
        select_series(inputs(f), SeriesParams(series="spread"))
    out = transformations(inputs(f), SeriesParams(series="zscore")).payload
    assert out["summary"]["mean"] == pytest.approx(0, abs=1e-9)


# ── TS-001 / TS-002 / TS-022: behaviour on random walks ──────────────────────


def test_random_walk_has_a_unit_root_and_returns_are_stationary() -> None:
    f = frame(random_walk())
    level = {t["test"]: t for t in stationarity(inputs(f), StationarityParams()).payload["tests"]}
    assert level["adf"]["reject"] is False
    assert level["kpss"]["reject"] is True
    out = stationarity(inputs(f), StationarityParams(series="returns"))
    assert {t["test"]: t["reject"] for t in out.payload["tests"]}["adf"] is True
    assert stationarity(inputs(f), StationarityParams()).payload["order_of_integration"] == 1


def test_serial_dependence_on_iid_and_on_ar1() -> None:
    iid = serial_dependence(inputs(frame(random_walk())), DependenceParams()).payload
    lb = {t["test"]: t for t in iid["tests"]}
    assert lb["ljung_box_10"]["reject"] is False
    rng = np.random.default_rng(4)
    r = np.zeros(1_500)
    for t in range(1, 1_500):
        r[t] = 0.3 * r[t - 1] + rng.normal(0, 0.01)
    ar = {
        t["test"]: t
        for t in serial_dependence(inputs(frame(np.cumsum(r))), DependenceParams()).payload["tests"]
    }
    assert ar["ljung_box_10"]["reject"] is True
    assert ar["variance_ratio_5"]["vr"] > 1  # positive autocorrelation → momentum


def test_placebo_rejects_about_five_percent_of_random_walks() -> None:
    rejections = 0
    for seed in range(40):
        out = placebo(
            inputs(frame(random_walk(500, seed=seed))), PlaceboParams(n_sims=200, seed=seed)
        ).payload
        rejections += {t["test"]: t for t in out["tests"]}["iid_bootstrap"]["reject"]
    assert rejections <= 6  # nominal 2/40; generous for 40 draws


# ── TS-005 / TS-006 memory ───────────────────────────────────────────────────


def test_hurst_and_gph_on_white_noise_and_long_memory() -> None:
    noise = np.random.default_rng(2).normal(size=4_000)
    assert hurst_rs(noise)["anis_lloyd"] == pytest.approx(0.5, abs=0.07)
    assert hurst_dfa(noise) == pytest.approx(0.5, abs=0.07)
    assert gph(noise)[0] == pytest.approx(0.0, abs=0.15)
    # ARFIMA(0, 0.3, 0): full (untruncated) MA(∞) filter of noise with FFD weights of d = −0.3.
    n = 30_000
    shocks = np.random.default_rng(3).normal(size=n)
    long = fftconvolve(shocks, ffd_weights(-0.3, 1e-9, max_len=n))[:n][5_000:]
    assert gph(long, power=0.6)[0] == pytest.approx(0.3, abs=0.1)
    out = long_memory(inputs(frame(random_walk())), SeriesParams()).payload
    assert 0.4 < out["hurst_rs_corrected"] < 0.6


def test_minimum_d_star_makes_the_series_stationary() -> None:
    out = fractional_differentiation(inputs(frame(random_walk(1_500))), FracDiffParams()).payload
    rows = {row["d"]: row for row in out["rows"]}
    assert rows[0.0]["p_value"] > 0.05
    assert out["d_star"] is not None
    assert 0 < out["d_star"] <= 1
    assert rows[out["d_star"]]["corr"] > rows[1.0]["corr"]  # less differencing keeps more memory


# ── TS-003 / TS-009 / TS-012 / TS-019 models ─────────────────────────────────


def test_garch_recovers_persistence() -> None:
    model = arch_model(None, p=1, q=1)
    model.distribution = Normal(seed=np.random.default_rng(11))  # reproducible draw
    sim = model.simulate([0.0, 0.02, 0.08, 0.9], 3_000, burn=500)
    r = sim["data"].to_numpy() / 100
    out = garch(inputs(frame(np.log(100) + np.cumsum(r))), GarchParams(dist="normal")).payload
    assert out["persistence"] == pytest.approx(0.98, abs=0.03)
    assert out["half_life_days"] > 10
    assert len(out["term_structure"]["vol"]) == 60
    gjr = garch(inputs(frame(np.log(100) + np.cumsum(r))), GarchParams(model="gjr")).payload
    assert "gamma[1]" in gjr["params"]


def test_ou_half_life_recovered_with_interval() -> None:
    rng = np.random.default_rng(6)
    x = np.zeros(3_000)
    for t in range(1, 3_000):
        x[t] = 0.95 * x[t - 1] + rng.normal(0, 0.01)
    out = ou_mean_reversion(inputs(frame(x)), OuParams(n_boot=200)).payload
    truth = -math.log(2) / math.log(0.95)
    assert out["half_life"] == pytest.approx(truth, rel=0.25)
    low, high = out["half_life_ci"]
    assert low < truth < high


def test_regimes_separate_calm_and_turbulent_periods() -> None:
    rng = np.random.default_rng(8)
    r = np.concatenate(
        [rng.normal(0, 0.005, 500), rng.normal(0, 0.025, 300), rng.normal(0, 0.005, 500)]
    )
    out = regimes(inputs(frame(np.cumsum(r))), RegimeParams()).payload
    calm, turbulent = out["regimes"]
    assert calm["vol_annualised"] == pytest.approx(0.005 * math.sqrt(252), rel=0.2)
    assert turbulent["vol_annualised"] == pytest.approx(0.025 * math.sqrt(252), rel=0.2)


def test_kalman_dynamic_beta_follows_a_step_change() -> None:
    rng = np.random.default_rng(9)
    m = rng.normal(0, 0.01, 1_000)
    beta = np.r_[np.full(500, 0.5), np.full(500, 1.5)]
    a = beta * m + rng.normal(0, 0.003, 1_000)
    f, bench = frame(np.cumsum(np.r_[0, a])), frame(np.cumsum(np.r_[0, m]))
    out = kalman(inputs(f, bench), KalmanParams(model="dynamic_beta", delta=1e-3)).payload
    assert out["beta"][450] == pytest.approx(0.5, abs=0.15)
    assert out["beta"][-1] == pytest.approx(1.5, abs=0.15)
    level = kalman(inputs(f), KalmanParams()).payload
    assert level["signal_to_noise"] is None or level["signal_to_noise"] >= 0


def test_arima_picks_an_ar_model_for_ar1_returns() -> None:
    rng = np.random.default_rng(10)
    r = np.zeros(1_200)
    for t in range(1, 1_200):
        r[t] = 0.4 * r[t - 1] + rng.normal(0, 0.01)
    out = arima(
        inputs(frame(np.cumsum(r))), ArimaParams(series="log_returns", max_p=2, max_q=2)
    ).payload
    assert out["order"][0] >= 1
    ar1 = next(c for c in out["coefficients"] if c["name"] == "ar.L1")
    assert ar1["value"] == pytest.approx(0.4, abs=0.1)


# ── TS-010 / TS-011 / TS-014 breaks & events ─────────────────────────────────


def test_bai_perron_finds_a_planted_mean_shift() -> None:
    rng = np.random.default_rng(11)
    y = np.r_[rng.normal(0, 1, 200), rng.normal(2, 1, 200)]
    result = bai_perron(y, max_breaks=3, min_size=20)
    assert result["breaks"] == 1
    assert result["cuts"][0] == pytest.approx(200, abs=5)


def test_structural_break_suite_runs() -> None:
    f = frame(random_walk(800))
    out = structural_breaks(inputs(f), BreakParams(chow_date=f.index[400].date())).payload
    assert {t["test"] for t in out["tests"]} >= {"bde_cusum", "chow"}
    assert len(out["csw"]["stat"]) == 799


def test_explosiveness_detects_a_planted_bubble_but_not_a_random_walk() -> None:
    rng = np.random.default_rng(12)
    calm = np.cumsum(rng.normal(0, 0.01, 300))
    bubble = calm[-1] + np.cumsum(0.02 + rng.normal(0, 0.005, 40)) * np.linspace(1, 3, 40)
    crash = bubble[-1] - np.linspace(0, 0.8, 30)
    y = np.log(100) + np.r_[calm, bubble, crash]
    out = explosiveness(inputs(frame(y)), ExplosiveParams(n_sims=49)).payload
    tests = {t["test"]: t for t in out["tests"]}
    assert tests["gsadf"]["reject"] is True
    assert out["episodes"]
    walk = explosiveness(
        inputs(frame(random_walk(370, seed=5))), ExplosiveParams(n_sims=49)
    ).payload
    assert {t["test"]: t for t in walk["tests"]}["sadf"]["reject"] is False


def test_cusum_filter_golden() -> None:
    log_price = pd.Series(
        [0.0, 0.01, 0.02, 0.03, 0.02, 0.01, -0.01], index=pd.bdate_range("2024-01-01", periods=7)
    )
    h = pd.Series(0.025, index=log_price.index)
    events = cusum_filter(log_price, h)
    assert [e.day for e in events] == [4, 9]  # +0.03 cumulated on Thu 4, −0.03 on Tue 9
    out = event_sampling(inputs(frame(random_walk(600))), CusumFilterParams()).payload
    assert out["count"] > 0


# ── TS-013 / TS-016 information & microstructure ─────────────────────────────


def test_entropy_estimators_on_known_sequences() -> None:
    bits = np.random.default_rng(13).integers(0, 2, 4_000)
    assert plug_in_entropy(bits, 2) == pytest.approx(1.0, abs=0.02)
    assert lempel_ziv_entropy(bits) == pytest.approx(1.0, abs=0.15)
    assert kontoyiannis_entropy(bits[:1_000]) == pytest.approx(1.0, abs=0.25)
    assert plug_in_entropy(np.zeros(500, dtype=int)) == pytest.approx(0.0)
    out = entropy(inputs(frame(random_walk(800))), EntropyParams()).payload
    assert out["normalised"]["plug_in"] > 0.9


def test_roll_spread_recovers_a_bid_ask_bounce() -> None:
    rng = np.random.default_rng(14)
    mid = 100 + np.cumsum(rng.normal(0, 0.05, 5_000))
    traded = mid + 0.1 / 2 * rng.choice([-1, 1], 5_000)  # spread 0.10
    estimate = roll_spread(pd.Series(traded), window=4_000).iloc[-1]
    assert estimate == pytest.approx(0.1, rel=0.15)
    out = microstructure(inputs(frame(random_walk(400))), MicrostructureParams()).payload
    assert set(out["current"]) == {
        "roll_spread",
        "corwin_schultz",
        "amihud_per_million",
        "kyle_lambda",
        "hasbrouck_lambda",
    }


# ── TS-018 / TS-020 pairs & lead–lag ─────────────────────────────────────────


def test_cointegrated_pair_is_detected_and_hedge_ratio_recovered() -> None:
    rng = np.random.default_rng(15)
    b = random_walk(1_000, seed=16)
    spread = np.zeros(1_000)
    for t in range(1, 1_000):
        spread[t] = 0.9 * spread[t - 1] + rng.normal(0, 0.005)
    a = 0.1 + 1.5 * b + spread
    out = pair_dynamics(inputs(frame(a), frame(b)), PairParams()).payload
    tests = {t["test"]: t for t in out["tests"]}
    assert tests["engle_granger_a_on_b"]["reject"] is True
    assert out["hedge_ratio"]["ols"] == pytest.approx(1.5, abs=0.05)
    assert out["spread_half_life"] == pytest.approx(-math.log(2) / math.log(0.9), rel=0.3)


def test_lead_lag_finds_the_leader() -> None:
    rng = np.random.default_rng(17)
    lead = rng.normal(0, 0.01, 1_001)
    follow = 0.6 * np.r_[0, lead[:-1]] + rng.normal(0, 0.005, 1_001)
    out = lead_lag(
        inputs(frame(np.cumsum(follow)), frame(np.cumsum(lead))), LeadLagParams()
    ).payload
    assert out["ccf"]["1"] > 0.5
    tests = {t["test"]: t for t in out["tests"]}
    assert tests["b_granger_causes_a"]["reject"] is True
    assert tests["a_granger_causes_b"]["reject"] is False


# ── smoke: remaining TS analyses ─────────────────────────────────────────────


def test_remaining_analyses_produce_payloads() -> None:
    f = frame(random_walk(1_000), volume=np.random.default_rng(1).lognormal(13, 0.3, 1_000))
    assert decomposition(inputs(f), SpectrumParams()).payload["dominant_cycles"]
    assert hhi_concentration(np.ones(10)) == pytest.approx(0.0)
    assert concentration(inputs(f), SeriesParams()).payload["runs"]
    assert distribution_dynamics(inputs(f), DynamicsParams()).payload["skew"]
    assert len(vol_dynamics(inputs(f), SeriesParams()).payload["leverage_corr"]) == 20
    assert volume_activity(inputs(f), SeriesParams()).payload["abnormal_days"] >= 0
