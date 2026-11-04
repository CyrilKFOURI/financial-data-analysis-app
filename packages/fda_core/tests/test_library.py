"""Tests for M26: library templates, overlays, execution cost models."""

import math

import numpy as np
import pandas as pd
import pytest

from fda_core.analyses.library import almgren_chriss, corwin_schultz
from fda_core.backtest import DEFAULT_GRIDS, TEMPLATE_DOCS, apply_overlays, param_grid, signal

DAYS = pd.bdate_range("2015-01-02", periods=1200)


def walk(seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, len(DAYS)))), index=DAYS)


def test_every_template_is_documented_and_produces_bounded_targets() -> None:
    close = walk()
    assert set(DEFAULT_GRIDS) == set(TEMPLATE_DOCS)
    for template, ranges in DEFAULT_GRIDS.items():
        params = param_grid(ranges)[0]
        target = signal(template, close, params)  # type: ignore[arg-type]
        assert target.index.equals(close.index)
        assert target.between(0, 1).all(), template  # long-only by default
        assert {"family", "rules", "fails"} <= set(TEMPLATE_DOCS[template])


def test_tema_reacts_faster_than_ema() -> None:
    up_then_down = pd.Series(
        np.r_[np.linspace(100, 150, 300), np.linspace(150, 100, 300)], index=DAYS[:600]
    )
    tema = signal("tema_cross", up_then_down, {"fast": 10, "slow": 50})
    ema = signal("ema_cross", up_then_down, {"fast": 10, "slow": 50})

    def first_exit(target: pd.Series) -> pd.Timestamp:
        after_top = target.loc[target.index > DAYS[300]]
        return after_top.index[after_top == 0][0]

    assert first_exit(tema) <= first_exit(ema)


def test_overlays() -> None:
    close = walk(1)
    ones = pd.Series(1.0, index=close.index)
    trend = apply_overlays(ones, close, ["trend_filter"])
    sma = close.rolling(200, min_periods=200).mean()
    assert (trend[close < sma] == 0).all()
    assert (trend[(close >= sma) & sma.notna()] == 1).all()
    vol = apply_overlays(ones, close, ["vol_target"], vol_target=0.10, max_leverage=1.5)
    realised = close.pct_change().rolling(20).std() * math.sqrt(252)
    t = close.index[100]
    assert vol.loc[t] == pytest.approx(min(0.10 / realised.loc[t], 1.5))
    dd = apply_overlays(ones, close, ["drawdown_control"], drawdown_limit=0.1)
    assert (dd[close / close.cummax() - 1 <= -0.1] == 0).all()


def test_corwin_schultz_recovers_a_known_spread() -> None:
    rng = np.random.default_rng(3)
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    spread = 0.01
    high = pd.Series(mid * (1 + spread / 2) * np.exp(np.abs(rng.normal(0, 0.003, 400))))
    low = pd.Series(mid * (1 - spread / 2) * np.exp(-np.abs(rng.normal(0, 0.003, 400))))
    est = corwin_schultz(high, low).median()
    assert 0.0 <= est < 0.05


def test_almgren_chriss_schedule() -> None:
    flat = almgren_chriss(10_000, 10, 0.5, 0.01, 0.0)
    assert flat == pytest.approx(np.full(10, 1000.0))
    urgent = almgren_chriss(10_000, 10, 0.5, 0.01, 1e-3)
    assert urgent.sum() == pytest.approx(10_000)
    assert urgent[0] > urgent[-1]  # risk aversion front-loads the trading
