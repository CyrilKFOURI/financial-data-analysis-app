"""AFML toolkit tests, incl. M27 acceptance 1 (purged K-fold removes the fake accuracy gain on
a random walk) and 2 (the event day is excluded from sample-weight attribution)."""

import math

import numpy as np
import pandas as pd
import pytest

from fda_core.afml import (
    Logistic,
    NearestNeighbours,
    average_active_bets,
    average_uniqueness,
    bet_size,
    concurrency,
    cusum_events,
    cv_scores,
    daily_vol,
    indicator_matrix,
    mda,
    platt,
    purged_kfold,
    sequential_bootstrap,
    triple_barrier,
)

DAYS = pd.bdate_range("2015-01-02", periods=1500)


def walk(seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(DAYS)))), index=DAYS)


def test_event_day_is_excluded_from_uniqueness() -> None:
    index = pd.bdate_range("2024-01-01", periods=6)
    t1 = pd.Series({index[0]: index[2], index[1]: index[3]})
    c = concurrency(index, t1)
    assert c.iloc[0] == 0  # label 0 starts at bar 0: not counted on its event day
    assert list(c.iloc[1:4]) == [1, 2, 1]
    u = average_uniqueness(index, t1)
    assert u[index[0]] == pytest.approx((1 / 1 + 1 / 2) / 2)  # bars 1–2, not 0–2
    assert u[index[1]] == pytest.approx((1 / 2 + 1 / 1) / 2)


def test_triple_barrier_and_cusum() -> None:
    close = walk(1)
    vol = daily_vol(close)
    events = cusum_events(close, vol * 2)
    assert len(events) > 20
    labels = triple_barrier(close, events, vol, pt=1.0, sl=1.0, horizon=10)
    assert set(labels["label"].unique()) <= {-1, 0, 1}
    assert (labels["t1"] > labels.index).all()
    pt = labels[labels["barrier"] == "pt"]
    assert (pt["label"] == 1).all()
    ind = indicator_matrix(close.index, labels["t1"])
    draws = sequential_bootstrap(ind[:, :30], 30, seed=1)
    assert len(draws) == 30
    assert len(set(draws)) < 30  # sampling with replacement


def test_purged_kfold_removes_the_fake_gain_on_a_random_walk() -> None:
    close = walk(2)
    r = np.log(close).diff()
    horizon = 20
    fwd = (np.log(close).shift(-horizon) - np.log(close)).dropna()  # overlapping 20-day labels
    features = (
        pd.concat({f"ret_{k}": r.rolling(k).sum() for k in (5, 20, 60)}, axis=1)
        .loc[fwd.index]
        .dropna()
    )
    y = (fwd.loc[features.index] > 0).astype(float).to_numpy()
    x = features.to_numpy()
    t0 = features.index
    t1 = pd.Series(
        [close.index[min(close.index.get_loc(t) + horizon, len(close) - 1)] for t in t0], index=t0
    )
    shuffled = cv_scores(NearestNeighbours, x, y, purged_kfold(t0, t1, 5, shuffle=True))
    purged = cv_scores(NearestNeighbours, x, y, purged_kfold(t0, t1, 5, embargo=0.02))
    assert shuffled["accuracy"] > 0.6  # leakage through overlapping labels
    assert abs(purged["accuracy"] - 0.5) < 0.08  # no skill on a random walk
    for train, test in purged_kfold(t0, t1, 5, embargo=0.02):
        test_start, test_end = t0[test[0]], t1.iloc[test].max()
        overlapping = (t0[train] <= test_end) & (pd.DatetimeIndex(t1.iloc[train]) >= test_start)
        assert not overlapping.any()


def test_logistic_recovers_a_linear_signal_and_mda_finds_it() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, (2000, 3))
    p = 1 / (1 + np.exp(-(1.5 * x[:, 0])))
    y = (rng.uniform(size=2000) < p).astype(float)
    model = Logistic(c=10).fit(x, y)
    assert model.coef is not None
    assert model.coef[1] == pytest.approx(1.5, abs=0.2)  # standardised x ≈ x
    t0 = pd.bdate_range("2010-01-01", periods=2000)
    splits = purged_kfold(t0, pd.Series(t0, index=t0), 5)
    imp = {r["feature"]: r["mda"] for r in mda(x, y, splits, ["signal", "noise1", "noise2"])}
    assert imp["signal"] > 0.05
    assert abs(imp["noise1"]) < 0.01
    a, b = platt(model.proba(x), y)
    assert math.isfinite(a)
    assert math.isfinite(b)


def test_bet_sizing() -> None:
    assert bet_size(np.array([0.5]))[0] == pytest.approx(0.0)
    sizes = bet_size(np.array([0.55, 0.7, 0.95]))
    assert (np.diff(sizes) > 0).all()
    assert bet_size(np.array([0.7]), step=0.25)[0] in (0.25, 0.5, 0.75, 1.0)
    index = pd.bdate_range("2024-01-01", periods=5)
    t1 = pd.Series({index[0]: index[3], index[1]: index[4]})
    pos = average_active_bets(t1, pd.Series({index[0]: 1.0, index[1]: 0.0}), index)
    assert pos.iloc[1] == 1.0  # only the first bet is alive
    assert pos.iloc[2] == 0.5  # both alive: average
