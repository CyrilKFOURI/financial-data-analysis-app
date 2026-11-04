"""Oracle tests for the optimisation engine: closed forms, brute-force grids, invariants."""

import math

import numpy as np
import pandas as pd
import pytest

from fda_core.optimise import (
    black_litterman,
    cov_estimate,
    deflated_sharpe,
    denoise,
    frontier,
    gmv,
    hrp,
    index_tracking,
    inverse_vol,
    max_diversification,
    max_sharpe,
    mean_estimate,
    min_cdar,
    min_cvar,
    resampled,
    risk_budget,
    target_volatility,
)
from fda_core.portfolio import risk_contributions

VOLS = np.array([0.10, 0.15, 0.20, 0.25])
CORR = np.array(
    [[1.0, 0.3, 0.2, 0.1], [0.3, 1.0, 0.4, 0.2], [0.2, 0.4, 1.0, 0.5], [0.1, 0.2, 0.5, 1.0]]
)
COV = CORR * np.outer(VOLS, VOLS)
MU = np.array([0.04, 0.06, 0.08, 0.09])


def returns(seed: int = 0, n: int = 750) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.multivariate_normal(MU / 252, COV / 252, n)
    return pd.DataFrame(x, columns=list("ABCD"))


def test_unconstrained_gmv_and_max_sharpe_match_closed_forms() -> None:
    inv = np.linalg.inv(COV)
    ones = np.ones(4)
    expected = inv @ ones / (ones @ inv @ ones)
    assert gmv(COV, lower=-2, upper=3) == pytest.approx(expected, abs=1e-6)
    tangent = inv @ (MU - 0.02)
    tangent /= tangent.sum()
    assert max_sharpe(MU, COV, 0.02, lower=-2, upper=3) == pytest.approx(tangent, abs=1e-5)
    long_only = gmv(COV)
    assert long_only.min() >= -1e-12
    assert long_only.sum() == pytest.approx(1)


def test_risk_budgets_equalise_contributions() -> None:
    w = risk_budget(COV)
    _, _, rc = risk_contributions(w, COV)
    assert rc / rc.sum() == pytest.approx(np.full(4, 0.25), abs=1e-8)
    budgets = np.array([0.4, 0.3, 0.2, 0.1])
    w = risk_budget(COV, budgets)
    _, _, rc = risk_contributions(w, COV)
    assert rc / rc.sum() == pytest.approx(budgets, abs=1e-8)
    two = np.array([[0.04, 0.01], [0.01, 0.09]])
    assert risk_budget(two) == pytest.approx(np.array([3, 2]) / 5, abs=1e-8)  # ∝ 1/σ for 2 assets


def test_hrp_equals_inverse_variance_when_uncorrelated() -> None:
    diag = np.diag(VOLS**2)
    w, link = hrp(returns(), diag)
    ivp = (1 / VOLS**2) / (1 / VOLS**2).sum()
    assert w == pytest.approx(ivp, abs=1e-12)
    assert link.shape == (3, 4)
    assert inverse_vol(COV) == pytest.approx((1 / VOLS) / (1 / VOLS).sum())


def test_max_diversification_is_inverse_vol_under_constant_correlation() -> None:
    corr = np.full((4, 4), 0.3)
    np.fill_diagonal(corr, 1.0)
    cov = corr * np.outer(VOLS, VOLS)
    assert max_diversification(cov) == pytest.approx((1 / VOLS) / (1 / VOLS).sum(), abs=1e-6)


def _cvar(r: np.ndarray, alpha: float) -> float:
    losses = -r
    var = np.quantile(losses, alpha)
    return float(var + np.maximum(losses - var, 0).mean() / (1 - alpha))


def test_tail_optimisers_beat_a_brute_force_grid() -> None:
    x = returns(1, 500)[["A", "D"]].to_numpy()
    grid = np.linspace(0, 1, 201)
    best_cvar = min(_cvar(x @ np.array([g, 1 - g]), 0.95) for g in grid)
    w = min_cvar(x, 0.95)
    assert _cvar(x @ w, 0.95) <= best_cvar + 1e-9

    def cdar(r: np.ndarray, alpha: float) -> float:
        c = np.cumsum(r)
        dd = np.maximum.accumulate(np.maximum(c, 0)) - c
        q = np.quantile(dd, alpha)
        return float(q + np.maximum(dd - q, 0).mean() / (1 - alpha))

    best_cdar = min(cdar(x @ np.array([g, 1 - g]), 0.95) for g in grid)
    w = min_cdar(x, 0.95)
    assert cdar(x @ w, 0.95) <= best_cdar + 1e-6


def test_estimators() -> None:
    r = returns()
    mu = mean_estimate(r, "historical")
    js = mean_estimate(r, "james_stein")
    grand = mean_estimate(r, "equilibrium", COV / 252)
    assert np.all(np.abs(js - js.mean()) <= np.abs(mu - js.mean()) + 1e-15)  # shrunk inwards
    assert grand == pytest.approx(2.5 * (COV / 252) @ np.full(4, 0.25))
    clean = denoise(np.cov(r.to_numpy(), rowvar=False), len(r))
    assert np.diag(clean) == pytest.approx(np.diag(np.cov(r.to_numpy(), rowvar=False)))
    lw = cov_estimate(r, "ledoit_wolf")
    assert lw.shape == (4, 4)
    prior = MU / 252
    sure = black_litterman(COV / 252, prior, [(0, 0.001, 0.999)])
    assert sure[0] == pytest.approx(0.001, rel=0.02)
    unsure = black_litterman(COV / 252, prior, [(0, 0.001, 0.001)])
    assert unsure[0] == pytest.approx(prior[0], rel=0.02)


def test_target_vol_frontier_resampling_and_tracking() -> None:
    w = target_volatility(MU, COV, 0.15)
    assert math.sqrt(w @ COV @ w) <= 0.15 + 1e-6
    points = frontier(MU, COV, 8)
    vols = [p[0] for p in points]
    rets = [p[1] for p in points]
    assert rets == sorted(rets)
    assert vols[0] == pytest.approx(math.sqrt(gmv(COV) @ COV @ gmv(COV)), rel=1e-4)
    r = returns()
    avg = resampled(r, lambda m, c: gmv(c), draws=20)
    assert avg.sum() == pytest.approx(1)
    x = r.to_numpy()
    benchmark = 0.6 * x[:, 1] + 0.4 * x[:, 3]
    tracked = index_tracking(x, benchmark, 2)
    assert tracked == pytest.approx(np.array([0, 0.6, 0, 0.4]), abs=1e-6)


def test_deflated_sharpe() -> None:
    one = deflated_sharpe(0.1, [0.1], 500, 0.0, 3.0)
    assert one["sr0"] == 0
    many = deflated_sharpe(0.1, [0.02, 0.05, 0.1, 0.08, 0.03], 500, 0.0, 3.0)
    assert many["sr0"] > 0
    assert many["dsr"] < one["dsr"]
