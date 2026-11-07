"""Kalman trend template (the Community's "Kalman vs MA200" challenge): calibration, causality."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fda_core import backtest as bt
from fda_core.registry import InsufficientDataError

DATA = Path(__file__).resolve().parents[3] / "tests/data/tqqq_yf_adjusted_2018-01-02_2026-03-30.csv"


def test_memory_calibration_and_its_scaling_law() -> None:
    for memory in (20.0, 50.0, 100.0):
        assert bt.kalman_memory(bt.kalman_q(memory)) == pytest.approx(memory, rel=1e-6)
    # an integrated random walk filter's bandwidth scales as q^(−1/4)
    ratio = bt.kalman_memory(1e-8) / bt.kalman_memory(1e-6)
    assert ratio == pytest.approx(100**0.25, rel=0.03)
    with pytest.raises(InsufficientDataError):
        bt.kalman_q(5000)


def test_slope_estimate_is_unbiased() -> None:
    """The steady-state slope weights sum to 0 and recover a straight line's slope exactly."""
    f, h, k = bt._kalman_system(bt.kalman_q(50))
    a = (np.eye(2) - np.outer(k, h.ravel())) @ f
    w, v = [], k.copy()
    for _ in range(3000):
        w.append(v[1])
        v = a @ v
    weights = np.array(w)
    assert weights.sum() == pytest.approx(0.0, abs=1e-9)
    assert (-np.arange(len(weights)) * weights).sum() == pytest.approx(1.0, abs=1e-6)


def test_no_look_ahead_and_direction() -> None:
    days = pd.bdate_range("2015-01-02", periods=900)
    rng = np.random.default_rng(2)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 900))), index=days)
    a = bt.kalman_trend(close, 50)
    altered = close.copy()
    altered.iloc[600:] *= 0.5
    pd.testing.assert_series_equal(a.iloc[:600], bt.kalman_trend(altered, 50).iloc[:600])
    rising = pd.Series(100 * 1.001 ** np.arange(400), index=days[:400])
    assert bt.kalman_trend(rising, 100).iloc[-1] == 1.0


@pytest.mark.skipif(not DATA.exists(), reason="TQQQ reference series not present (private data)")
def test_challenge_on_tqqq_trades_less_than_close_above_sma200() -> None:
    """The forum's finding: at equal memory the Kalman rule trades far less than close > SMA200;
    it is not a Sharpe improvement (here it is lower)."""
    close = pd.read_csv(DATA, index_col=0, parse_dates=True)["close"].astype(float)
    sma = (close > close.rolling(200).mean()).astype(float)
    kal = bt.kalman_trend(close, 50)
    run_k = bt.run_vectorised(close, kal, bt.Costs(5, 5), 220)
    run_s = bt.run_vectorised(close, sma, bt.Costs(5, 5), 220)
    assert len(bt.trades_of(run_k)) < len(bt.trades_of(run_s)) / 2
    assert bt.sharpe(run_k["net"].to_numpy()) < bt.sharpe(run_s["net"].to_numpy())
