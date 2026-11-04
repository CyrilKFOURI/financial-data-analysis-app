"""Tests for the backtesting engine against M25 acceptance criteria (2–5) and invariants."""

import math

import numpy as np
import pandas as pd
import pytest

from fda_core.backtest import (
    Costs,
    binary_sharpe,
    cpcv_paths,
    cpcv_splits,
    dsr,
    grid_returns,
    implied_precision,
    inject_lookahead,
    lookahead_check,
    ordered_constraint,
    param_grid,
    pbo,
    psr,
    run_event_driven,
    run_vectorised,
    sharpe,
    signal,
    trade_metrics,
    trades_of,
    walk_forward,
)

DAYS = pd.bdate_range("2012-01-02", periods=2000)


def walk(seed: int = 0, drift: float = 0.0003) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, 0.012, len(DAYS)))), index=DAYS)


def test_one_bar_shift_warmup_and_costs() -> None:
    close = walk()
    target = signal("ema_cross", close, {"fast": 10, "slow": 50})
    frame = run_vectorised(close, target, Costs(5, 5), warmup=220)
    assert frame.index[0] == close.index[220]
    r = close.pct_change()
    t = frame.index[100]
    assert frame.loc[t, "position"] == target.shift(1).loc[t]  # decided at t−1's close
    trades = frame["position"].diff().abs().fillna(frame["position"].abs())
    assert frame["cost"].sum() == pytest.approx(trades.sum() * 10 / 10_000)
    assert frame.loc[t, "gross"] == pytest.approx(frame.loc[t, "position"] * r.loc[t])
    assert (frame["position"] >= 0).all()  # long-only by default


def test_injected_lookahead_is_caught() -> None:
    close = walk(1)
    honest = signal("ema_cross", close, {"fast": 10, "slow": 50})
    ok = lookahead_check(close, honest, Costs(5, 5), 220)
    assert ok["suspicious"] is False
    peeking = pd.Series(
        np.sign(close.pct_change().shift(-1)).fillna(0).clip(lower=0), index=close.index
    )
    bad = lookahead_check(close, peeking, Costs(5, 5), 220)
    assert bad["suspicious"] is True
    assert bad["sharpe_lag1"] > 5  # the notebook shows "Sharpe ≈ 15 with look-ahead" on TQQQ
    assert inject_lookahead(honest).iloc[:-1].equals(honest.shift(-1).iloc[:-1])


def test_skill_less_strategies_pass_psr_more_often_than_dsr() -> None:
    passes_psr = passes_dsr = 0
    for rep in range(12):
        rng = np.random.default_rng(100 + rep)
        r = rng.normal(0.0, 0.01, (750, 50))  # 50 skill-less strategies, zero true Sharpe
        srs = list(r.mean(0) / r.std(0, ddof=1))
        best = int(np.argmax(srs))
        passes_psr += psr(r[:, best]) > 0.95
        passes_dsr += dsr(r[:, best], srs) > 0.95
    assert passes_psr >= 6
    assert passes_dsr <= 2


def test_pbo_on_noise_is_about_one_half() -> None:
    rng = np.random.default_rng(7)
    noise = pd.DataFrame(rng.normal(0, 0.01, (1600, 40)))
    out = pbo(noise, s=10)
    assert out["combinations"] == math.comb(10, 5)
    assert 0.3 <= out["pbo"] <= 0.7
    skilled = noise.copy()
    skilled[0] += 0.002  # one genuinely better strategy
    assert pbo(skilled, s=10)["pbo"] < 0.1


def test_cpcv_splits_paths_and_purging() -> None:
    splits = cpcv_splits(600, 6, 2, embargo=5, label_span=10)
    assert len(splits) == 15
    for train, test, combo in splits:
        assert not set(train) & set(test)
        for g in combo:
            lo = g * 100
            assert not set(range(lo - 10, lo)) & set(train)  # purged before each tested group
            assert not set(range(lo + 100, min(600, lo + 105))) & set(train)  # embargo after
    rng = np.random.default_rng(2)
    out = cpcv_paths(pd.DataFrame(rng.normal(0, 0.01, (600, 8))), 6, 2)
    assert out["splits"] == 15
    assert out["paths"] == 5
    assert len(out["path_sharpes"]) == 5


def test_grid_walk_forward_and_trades() -> None:
    close = walk(3, 0.0005)
    grid = param_grid({"fast": [5, 10, 20], "slow": [20, 50, 100]}, ordered_constraint("ema_cross"))
    assert {"fast": 20, "slow": 20} not in grid
    trials = grid_returns("ema_cross", close, grid, Costs(5, 5), 220)
    assert trials.shape[1] == len(grid)
    fwd = walk_forward(trials, 4)
    back = walk_forward(trials, 4, backward=True)
    assert len(fwd["chosen"]) == 4
    assert math.isfinite(back["sharpe"])
    frame = run_vectorised(close, signal("ema_cross", close, grid[0]), Costs(5, 5), 220)
    trades = trades_of(frame)
    m = trade_metrics(trades)
    assert m["trades"] == len(trades) > 3
    assert 0 <= m["win_rate"] <= 1
    assert np.prod(1 + trades["return"]) == pytest.approx(np.prod(1 + frame["net"]), rel=1e-9)


def test_strategy_risk_inverse() -> None:
    sr = binary_sharpe(0.55, 0.02, 0.02, 252)
    assert implied_precision(sr, 0.02, 0.02, 252) == pytest.approx(0.55, abs=1e-9)
    assert sharpe(np.array([0.01, -0.01, 0.02])) == pytest.approx(
        np.mean([0.01, -0.01, 0.02]) / np.std([0.01, -0.01, 0.02], ddof=1) * math.sqrt(252)
    )


def test_event_driven_reconciles_with_vectorised_when_open_equals_prior_close() -> None:
    close = walk(4)
    bars = pd.DataFrame(
        {"open": close.shift(1).fillna(close.iloc[0]), "close": close, "volume": np.inf}, index=DAYS
    )
    target = signal("ema_cross", close, {"fast": 10, "slow": 50})
    vec = run_vectorised(close, target, Costs(0, 0), 220)
    evt = run_event_driven(bars, target, Costs(0, 0), warmup=220)
    assert (1 + evt["net"]).prod() == pytest.approx((1 + vec["net"]).prod(), rel=1e-6)
    capped = run_event_driven(bars.assign(volume=1.0), target, Costs(0, 0), warmup=220)
    assert capped["shares"].abs().max() <= 1.0 * len(capped)  # partial fills under the cap
