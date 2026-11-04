"""J2 statistics: golden values, statistical recovery and empyrical oracles."""

import math

import empyrical
import numpy as np
import pandas as pd
import pytest

from fda_core.analyses.stats import (
    DispersionParams,
    FreqParams,
    NormalityParams,
    OhlcVolParams,
    RollingParams,
    SeasonalityParams,
    StabilityParams,
    benjamini_hochberg,
    best_days,
    calendar_returns,
    central_tendency,
    dispersion,
    distribution,
    estimate_instability,
    means,
    moments,
    normality,
    ohlc_volatility,
    resample_returns,
    return_quantiles,
    rolling_statistics,
    seasonality,
    tail_frequency,
    up_down,
    volatility_drag,
)
from fda_core.registry import AnalysisInputs, InsufficientDataError

EQUITY = {"annualisation": 252, "price_field": "adj_close", "has_volume": True, "range": "MAX"}


def frame_from_returns(
    r: np.ndarray, start: str = "2015-01-02", ohlc_sigma: float = 0.0
) -> pd.DataFrame:
    """Prices whose daily returns are exactly `r` (a base close precedes the first return)."""
    close = 100 * np.cumprod(np.concatenate([[1.0], 1 + r]))
    index = pd.bdate_range(start, periods=len(close))
    rng = np.random.default_rng(99)
    spread = np.abs(rng.normal(0, ohlc_sigma, len(close))) if ohlc_sigma else np.zeros(len(close))
    return pd.DataFrame(
        {"open": close, "high": close * (1 + spread), "low": close * (1 - spread), "close": close,
         "adj_close": close, "volume": 1.0},
        index=index,
    )  # fmt: skip


def inputs(frame: pd.DataFrame, **conventions: object) -> AnalysisInputs:
    return AnalysisInputs(data={"prices": frame}, conventions=EQUITY | conventions)


def gaussian(n: int = 2_000, mu: float = 0.08, sigma: float = 0.20, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(mu / 252, sigma / math.sqrt(252), n)


# ── golden ───────────────────────────────────────────────────────────────────


def test_means_golden() -> None:
    m = means(np.array([0.10, -0.10]))
    assert m["arithmetic"] == pytest.approx(0.0)
    assert m["geometric"] == pytest.approx(math.sqrt(1.1 * 0.9) - 1)
    assert m["harmonic"] == pytest.approx(2 / (1 / 1.1 + 1 / 0.9) - 1)


def test_weekly_resampling_compounds_daily_returns() -> None:
    frame = frame_from_returns(np.full(10, 0.01), start="2023-12-29")  # Fri base → 2 full weeks
    weekly = resample_returns(frame["adj_close"], "W")
    assert weekly.iloc[0] == pytest.approx(1.01**5 - 1)


def test_dispersion_semi_variance_golden() -> None:
    r = np.array([0.02, -0.02, 0.01, -0.01, 0.0, 0.03, -0.03, 0.02, -0.02, 0.0, 0.01])
    out = dispersion(inputs(frame_from_returns(r)), DispersionParams()).payload
    mean = r.mean()
    assert out["semi_variance"] == pytest.approx((np.minimum(r - mean, 0) ** 2).mean())
    assert out["iqr"] == pytest.approx(np.percentile(r, 75) - np.percentile(r, 25))


def test_up_down_streaks() -> None:
    r = np.array([0.01, 0.01, 0.01, -0.01, -0.01, 0.0, 0.01, -0.01, 0.02, 0.02, 0.02, 0.02])
    out = up_down(inputs(frame_from_returns(r)), FreqParams()).payload
    assert out["longest_up_streak"] == 4
    assert out["longest_down_streak"] == 2
    assert out["pct_up"] == pytest.approx(8 / 12)


def test_volatility_drag_matches_half_sigma_squared_for_gaussian_returns() -> None:
    out = volatility_drag(inputs(frame_from_returns(gaussian(5_000))), FreqParams()).payload
    assert out["drag"] == pytest.approx(out["half_sigma_squared"], rel=0.05)


def test_benjamini_hochberg_known_example() -> None:
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.20], q=0.05)
    assert [round(a, 4) for a, _ in adjusted] == [0.04, 0.0533, 0.0533, 0.2]
    assert [s for _, s in adjusted] == [True, False, False, False]


def test_tail_multiplier_is_one_for_gaussian_and_large_for_student_t() -> None:
    gauss = tail_frequency(inputs(frame_from_returns(gaussian(20_000))), FreqParams()).payload
    rng = np.random.default_rng(5)
    fat = tail_frequency(
        inputs(frame_from_returns(rng.standard_t(3, 20_000) * 0.005)), FreqParams()
    ).payload
    assert gauss["rows"][0]["multiplier"] == pytest.approx(1.0, abs=0.15)
    assert fat["rows"][1]["multiplier"] > 2  # |z| > 3


def test_student_t_fit_recovers_degrees_of_freedom() -> None:
    rng = np.random.default_rng(8)
    out = distribution(
        inputs(frame_from_returns(rng.standard_t(4, 8_000) * 0.006)), FreqParams()
    ).payload
    assert 3.2 < out["student_t"]["nu"] < 5.0
    assert sum(np.diff(out["bins"]["edges"]) * np.array(out["bins"]["density"])) == pytest.approx(1)


def test_moments_and_normality_on_gaussian_data() -> None:
    frame = frame_from_returns(gaussian(3_000, seed=4))
    m = moments(inputs(frame), FreqParams()).payload
    assert abs(m["skewness"]) < 3 * m["skewness_se"]
    tests = normality(inputs(frame), NormalityParams()).payload["tests"]
    assert {t["test"] for t in tests} == {
        "jarque_bera",
        "shapiro_wilk",
        "anderson_darling",
        "dagostino_k2",
    }
    jb = next(t for t in tests if t["test"] == "jarque_bera")
    assert jb["reject_normality"] is False


def test_calendar_grid_and_partial_years() -> None:
    frame = frame_from_returns(gaussian(600), start="2023-03-01")
    out = calendar_returns(inputs(frame), FreqParams()).payload
    assert out["years"] == ["2023", "2024", "2025"]
    assert out["months"]["2023"][0] is None  # January 2023 before the first price
    assert out["annual"][0]["partial"]
    assert not out["annual"][1]["partial"]
    assert set(out["extremes"]) >= {"day", "week", "month"}


def test_quantiles_rolling_and_best_days() -> None:
    frame = frame_from_returns(gaussian(800))
    q = return_quantiles(inputs(frame), FreqParams()).payload["rows"]
    assert q[0]["horizon"] == "D"
    assert q[0]["quantiles"]["0.5"] is not None
    roll = rolling_statistics(inputs(frame), RollingParams(windows=[21, 63])).payload
    assert set(roll["windows"]) == {"21", "63"}
    assert roll["windows"]["21"]["vol"][-1] == pytest.approx(
        frame["adj_close"].pct_change().iloc[-21:].std() * math.sqrt(252)
    )
    days = best_days(inputs(frame), FreqParams()).payload
    assert days["rows"][0]["without_best"] < days["total"] < days["rows"][0]["without_worst"]


def test_seasonality_finds_a_planted_january_effect_and_controls_fdr() -> None:
    r = gaussian(252 * 12, mu=0.0, sigma=0.15, seed=2)
    frame = frame_from_returns(r)
    january = frame.index[1:].month == 1  # month of each return (index[0] is the base close)
    r_planted = np.where(january, r + 0.004, r)
    out = seasonality(inputs(frame_from_returns(r_planted)), SeasonalityParams()).payload
    flagged = {row["bucket"] for row in out["rows"] if row["significant"]}
    assert flagged == {"M01"}
    assert out["hypotheses"] == 12


def test_ohlc_estimators_agree_with_true_volatility() -> None:
    frame = frame_from_returns(gaussian(1_500, sigma=0.25), ohlc_sigma=0.004)
    out = ohlc_volatility(inputs(frame), OhlcVolParams(window=21)).payload
    assert out["means"]["close_to_close"] == pytest.approx(0.25, rel=0.1)
    assert all(v is not None and v > 0 for v in out["means"].values())


# ── recovery (M05 AC-3, reduced to keep CI fast) ─────────────────────────────


def test_bootstrap_interval_covers_the_true_mean_most_of_the_time() -> None:
    covered = 0
    for seed in range(40):
        frame = frame_from_returns(gaussian(756, mu=0.08, sigma=0.20, seed=seed))
        out = estimate_instability(inputs(frame), StabilityParams(n_boot=300, seed=seed)).payload
        low, high = out["mean_ci"]
        covered += low <= 0.08 <= high
    assert covered >= 34  # nominal 95 %; ≥ 85 % on 40 draws


# ── oracle (empyrical-reloaded) ──────────────────────────────────────────────


def test_downside_deviation_matches_empyrical() -> None:
    r = gaussian(1_000, seed=3)
    out = dispersion(inputs(frame_from_returns(r)), DispersionParams(mar=0.0)).payload
    returns = pd.Series(r, index=pd.bdate_range("2015-01-05", periods=len(r)))
    oracle = empyrical.downside_risk(returns, required_return=0.0, annualization=1)
    assert out["downside_deviation_mar"] == pytest.approx(oracle, rel=1e-9)


def test_geometric_mean_matches_empyrical_annual_return() -> None:
    r = gaussian(1_260, seed=6)
    out = central_tendency(inputs(frame_from_returns(r)), FreqParams()).payload
    returns = pd.Series(r, index=pd.bdate_range("2015-01-05", periods=len(r)))
    assert out["geometric_annualised"] == pytest.approx(empyrical.annual_return(returns), rel=1e-9)


def test_sample_guards() -> None:
    with pytest.raises(InsufficientDataError):
        moments(inputs(frame_from_returns(gaussian(10))), FreqParams())
