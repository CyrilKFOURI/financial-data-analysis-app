"""Carver forecast engine (TR-018) and prop-firm challenge (TR-019): invariants, golden values."""

import numpy as np
import pandas as pd
import pytest

from fda_core import carver
from fda_core.backtest import Costs, run_vectorised, signal, trades_of

DAYS = pd.bdate_range("2005-01-03", periods=2600)


def walk(seed: int = 3, drift: float = 0.0003) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, 0.012, len(DAYS)))), index=DAYS)


def test_fdm_from_correlations() -> None:
    w = np.full(4, 0.25)
    assert carver.fdm_from_corr(w, np.eye(4)) == pytest.approx(2.0)
    assert carver.fdm_from_corr(w, np.ones((4, 4))) == pytest.approx(1.0)
    # negative correlations are floored at 0, so they never push the FDM above the identity case
    neg = np.full((4, 4), -0.5)
    np.fill_diagonal(neg, 1.0)
    assert carver.fdm_from_corr(w, neg) == pytest.approx(2.0)
    assert carver.fdm_from_corr(np.full(9, 1 / 9), np.eye(9)) == carver.FDM_MAX


def test_weights_sum_to_one_and_warmup() -> None:
    assert carver.weights().sum() == pytest.approx(1.0)
    f, _ = carver.forecasts(walk())
    assert f.iloc[: carver.WARMUP].isna().all().all()
    assert f.iloc[carver.WARMUP + 300 :].notna().all().all()
    assert (f.abs().max() <= carver.FC_CAP).all()


def test_no_look_ahead() -> None:
    """Changing the future never changes a past position (the FDM is point-in-time too)."""
    close = walk()
    cut = 1500
    altered = close.copy()
    altered.iloc[cut:] = altered.iloc[cut:] * np.linspace(1, 3, len(altered) - cut)
    a, b = carver.position(close), carver.position(altered)
    pd.testing.assert_series_equal(a.iloc[:cut], b.iloc[:cut])


def test_position_bounds_buffer_and_short() -> None:
    close = walk(seed=5)
    long = carver.position(close, 0.2, 0.0)
    assert long.between(0.0, 1.0).all()
    buffered = carver.position(close, 0.2, 0.1)
    assert buffered.diff().abs().sum() < long.diff().abs().sum()
    both = carver.position(close, 0.2, 0.0, long_only=False)
    assert both.min() < 0 <= both.max() <= 1.0


def test_template_runs_through_the_engine_with_whole_trades() -> None:
    """Resizing a continuous position is not a new trade: trades only end on a sign change."""
    close = walk(seed=7, drift=0.0006)
    target = signal("carver_trend", close, {"vol_target": 0.2, "buffer": 0.1})
    frame = run_vectorised(close, target, Costs(), carver.WARMUP)
    trades = trades_of(frame)
    flips = int((np.sign(frame["position"]).diff().fillna(0) != 0).sum())
    assert len(trades) <= flips + 1
    assert len(trades) < (frame["position"].diff().abs() > 0).sum()


def test_trades_resized_once() -> None:
    idx = pd.bdate_range("2020-01-01", periods=6)
    frame = pd.DataFrame(
        {
            "position": [0.0, 0.5, 0.7, 0.7, 0.0, 0.0],
            "net": [0.0, 0.01, 0.01, -0.005, 0.0, 0.0],
            "cost": [0.0] * 6,
        },
        index=idx,
    )
    trades = trades_of(frame)
    assert len(trades) == 1
    assert trades.iloc[0]["bars"] == 3


def test_vol_targeted_hold_is_always_long() -> None:
    w = carver.vol_targeted_hold(walk(), 0.2)
    assert (w.iloc[carver.WARMUP + 10 :] > 0).all()
    assert w.max() <= 1.0


def test_bars_per_year() -> None:
    assert 250 <= carver.bars_per_year(pd.bdate_range("2010-01-01", "2015-01-01")) <= 262
    assert carver.bars_per_year(pd.date_range("2018-01-01", "2021-01-01")) == pytest.approx(
        365, abs=1
    )


def test_prop_challenge() -> None:
    rng = np.random.default_rng(0)
    good = rng.normal(0.0008, 0.007, 2000)  # ≈ Sharpe 1.8
    rows = carver.prop_challenge(good, (0.08, 0.40), n_paths=2000, seed=1)
    low, high = rows
    assert 0 <= low["p_funded"] <= 1
    assert 0 <= high["p_funded"] <= 1
    assert low["median_days_if_funded"] > high["median_days_if_funded"]  # faster when riskier
    assert high["p_fail_phase1"] > low["p_fail_phase1"]  # but fails more often
    again = carver.prop_challenge(good, (0.08, 0.40), n_paths=2000, seed=1)
    assert again == rows  # deterministic with a seed
    with pytest.raises(ValueError, match="Not enough"):
        carver.prop_challenge(good[:10], (0.1,))


def test_golden_values() -> None:
    """Frozen output of the engine on a seeded random walk (formulas of the Quantrick course)."""
    close = walk(seed=11, drift=0.0004)
    f, v = carver.forecasts(close)
    fc, fdm = carver.combined(f)
    pos = carver.position(close, 0.2, 0.1)
    got = {
        "ewmac_mean_abs": float(f["ewmac"].abs().mean()),
        "breakout_last": float(f["breakout"].iloc[-1]),
        "accel_last": float(f["accel"].iloc[-1]),
        "skew_last": float(f["skew"].iloc[-1]),
        "vol_last": float(v["vol"].iloc[-1]),
        "fdm_last": float(fdm.iloc[-1]),
        "combined_last": float(fc.iloc[-1]),
        "position_mean": float(pos.mean()),
    }
    assert got == pytest.approx(GOLDEN, rel=1e-9)


GOLDEN: dict[str, float] = {
    "ewmac_mean_abs": 7.998519822134844,
    "breakout_last": 0.7322901359440038,
    "accel_last": -9.404960639115865,
    "skew_last": -3.5959492570227445,
    "vol_last": 0.19337099154285495,
    "fdm_last": 1.3115925579418422,
    "combined_last": -4.2611724190169395,
    "position_mean": 0.3841564191483288,
}
