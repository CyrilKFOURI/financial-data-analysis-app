"""Robustness protocol (TR-024): plateau selection, execution delay, epochs, ablation, golden."""

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from fda_core import backtest as bt
from fda_core import carver
from fda_core import robustness as rb
from fda_core.analyses.carver_engine import ChallengeParams, prop_challenge
from fda_core.analyses.robustness import RobustnessParams, robustness_protocol
from fda_core.registry import AnalysisInputs, InsufficientDataError

DAYS = pd.bdate_range("2008-01-02", periods=2400)


def walk(seed: int = 11, drift: float = 0.0004) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, 0.011, len(DAYS)))), index=DAYS)


def prices(close: pd.Series) -> pd.DataFrame:
    open_ = close.shift(1).fillna(close.iloc[0]) * 1.0005
    return pd.DataFrame(
        {"open": open_, "high": close, "low": close, "close": close, "adj_close": close}
    )


def strategy(
    template: str, grid: dict[str, list[float]], params: dict[str, float]
) -> dict[str, Any]:
    return {
        "id": 1,
        "name": "test",
        "hypothesis": "Trend persistence in a drifting series, tested before reading results.",
        "trials": 0,
        "spec": {
            "symbol": "WALK",
            "template": template,
            "grid": grid,
            "params": params,
            "short": False,
            "overlays": [],
            "costs": {"fee_bps": 5, "slippage_bps": 5},
            "warmup": 220,
            "is_fraction": 0.6,
        },
    }


def run(strat: dict[str, Any], close: pd.Series, **params: str) -> dict[str, Any]:
    inputs = AnalysisInputs(data={"strategy": strat, "prices": prices(close)})
    return dict(robustness_protocol(inputs, RobustnessParams(**params)).payload)


# ── plateau selection ────────────────────────────────────────────────────────

GRID_1D = [{"n": float(n)} for n in (1, 2, 3, 4, 5, 6, 7)]
RANGES_1D = {"n": [1, 2, 3, 4, 5, 6, 7]}


def test_plateau_beats_a_lone_peak() -> None:
    # cell 1 is a lone in-sample peak; cells 4-6 form a plateau (6 is on the edge)
    sharpe = [0.1, 2.0, 0.0, 0.0, 0.8, 1.0, 0.9]
    out = rb.plateau_choice(GRID_1D, sharpe, [5.0] * 7, RANGES_1D)
    assert out["is_best"] == 1
    assert out["pick"] == 5
    assert out["scores"][5] == pytest.approx(0.9)
    assert out["scores"][1] == pytest.approx(0.1)  # median of 0.1, 2.0, 0.0


def test_plateau_excludes_edges_and_rare_traders() -> None:
    sharpe = [3.0, 3.0, 0.2, 0.2, 0.2, 2.5, 2.5]
    trades = [5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 5.0]  # cell 5 trades once a year
    out = rb.plateau_choice(GRID_1D, sharpe, trades, RANGES_1D)
    assert out["edge"] == [True, False, False, False, False, False, True]
    assert out["eligible"][0] is False
    assert out["eligible"][5] is False
    assert out["pick"] == 1  # the best inner, active cell


def test_neighbours_on_two_axes() -> None:
    grid = [{"a": float(a), "b": float(b)} for a in (1, 2, 3) for b in (10, 20, 30)]
    ranges = {"a": [1, 2, 3], "b": [10, 20, 30]}
    centre = grid.index({"a": 2.0, "b": 20.0})
    assert len(rb.neighbours(grid, centre, ranges)) == 8
    corner = grid.index({"a": 1.0, "b": 10.0})
    assert len(rb.neighbours(grid, corner, ranges)) == 3


# ── execution, epochs, benchmarks ────────────────────────────────────────────


def test_lag_one_is_the_engine_default_and_open_execution_runs() -> None:
    close = walk()
    target = bt.signal("ema_cross", close, {"fast": 20, "slow": 100})
    periods = rb.bars_per_year(close.index)
    rows = rb.lag_sweep(close, target, bt.Costs(), 220, periods, prices(close)["open"])
    base = bt.run_vectorised(close, target, bt.Costs(), 220)
    assert rows[0]["sharpe"] == pytest.approx(bt.sharpe(base["net"].to_numpy(), round(periods)))
    assert [r["execution"] for r in rows] == ["close t+1", "close t+2", "close t+3", "open t+1"]


def test_short_epoch_gives_null_and_reason() -> None:
    close = walk()
    frame = bt.run_vectorised(close, pd.Series(1.0, index=close.index), bt.Costs(), 220)
    rows = rb.epochs({"x": frame}, frame.index[100], rb.bars_per_year(close.index))
    assert rows[0]["sharpe"] is None
    assert "under the one-year minimum" in rows[0]["reason"]
    assert rows[1]["sharpe"] is not None


def test_sharpe_standard_error() -> None:
    assert rb.sharpe_se(1.0, 6.0) == pytest.approx(math.sqrt(1.5 / 6))
    assert rb.sharpe_se(0.5, 0.0) is None


def test_benchmarks_are_decided_at_the_close() -> None:
    close = walk()
    b = rb.benchmark_targets(close, 0.2)
    assert (b["buy_and_hold"] == 1.0).all()
    assert b["ma200"].iloc[:199].eq(0).all()
    assert b["vol_targeted_hold"].between(0, 1).all()


# ── Carver ablation ──────────────────────────────────────────────────────────


def test_carver_with_all_rules_is_unchanged() -> None:
    close = walk(seed=3)
    pd.testing.assert_series_equal(
        carver.position(close), carver.position(close, rules=tuple(carver.RULE_WEIGHTS))
    )
    one = carver.position(close, rules=("ewmac",))
    assert one.between(0, 1).all()
    assert one.abs().sum() > 0


# ── the analysis ─────────────────────────────────────────────────────────────


def test_protocol_golden_and_checks() -> None:
    close = walk()
    strat = strategy(
        "ema_cross", {"fast": [10, 20, 30], "slow": [100, 150, 200]}, {"fast": 20, "slow": 150}
    )
    out = run(strat, close)
    assert [c["check"] for c in out["checks"]] == [
        "Plateau",
        "Trades a year",
        "Execution delay",
        "Both epochs",
        "Layers",
    ]
    assert out["checks"][4]["status"] == "n/a"  # a single-rule template has no layer to remove
    assert {r["series"] for r in out["epochs"]} == {
        "strategy",
        "buy_and_hold",
        "vol_targeted_hold",
        "ma200",
    }
    strat_row = next(r for r in out["plateau"] if "strategy" in r["role"])
    # golden values on the seeded walk (seed 11)
    assert strat_row["is_sharpe"] == pytest.approx(GOLDEN["is_sharpe"], abs=1e-6)
    assert out["execution"][0]["sharpe"] == pytest.approx(GOLDEN["lag1"], abs=1e-6)
    assert out["epoch_split"] == GOLDEN["epoch_split"]


def test_protocol_has_no_look_ahead_before_the_split() -> None:
    close = walk()
    strat = strategy(
        "ema_cross", {"fast": [10, 20, 30], "slow": [100, 150, 200]}, {"fast": 20, "slow": 150}
    )
    a = run(strat, close)
    altered = close.copy()
    cut = len(close) - 300  # after the IS/OOS split and the epoch split
    altered.iloc[cut:] *= np.linspace(1, 0.4, len(close) - cut)
    b = run(strat, altered)
    for key in ("is_sharpe", "neighbourhood_median", "trades_per_year"):
        assert [r[key] for r in a["plateau"]] == [r[key] for r in b["plateau"]]
    first = [r for r in a["epochs"] if r["epoch"] == "epoch 1"]
    assert first == [r for r in b["epochs"] if r["epoch"] == "epoch 1"]


def test_carver_protocol_ablates_rules_and_buffer() -> None:
    close = walk(seed=5)
    strat = strategy(
        "carver_trend",
        {"vol_target": [0.12, 0.2], "buffer": [0.0, 0.1]},
        {"vol_target": 0.2, "buffer": 0.1},
    )
    out = run(strat, close)
    layers = [r["layer"] for r in out["ablation"]]
    assert layers == ["rule ewmac", "rule breakout", "rule accel", "rule skew", "buffer"]
    # continuous sizing: the on/off trades rule does not apply, turnover is reported instead
    assert out["checks"][1]["status"] == "n/a"
    assert "turnover" in out["checks"][1]["detail"]
    assert out["checks"][4]["status"] in {"pass", "warn"}


def test_epoch_split_outside_the_sample_is_refused() -> None:
    close = walk()
    strat = strategy("ema_cross", {"fast": [10, 20], "slow": [100, 150]}, {"fast": 20, "slow": 150})
    with pytest.raises(InsufficientDataError):
        run(strat, close, epoch_split="1990-01-01")


# ── leveraged ETFs, the LETF baseline, CFD costs (311c) ──────────────────────


def test_leveraged_simulation_on_constant_returns() -> None:
    """r = 1 % a day: ×3 gives 3 % minus (0.88 % fee + 2 × 4 % financing) / 252 a day."""
    close = pd.Series(100 * 1.01 ** np.arange(11), index=DAYS[:11])
    lev = rb.leveraged(close, 3.0, 0.0088, 0.04, 252.0)
    daily = 0.03 - (0.0088 + 2 * 0.04) / 252
    assert lev.iloc[-1] == pytest.approx(100 * (1 + daily) ** 10)
    # a fall of more than a third wipes a ×3 product out, never below zero
    crash = pd.Series([100.0, 60.0, 61.0], index=DAYS[:3])
    assert rb.leveraged(crash, 3.0, 0.0, 0.0, 252.0).iloc[-1] == 0.0


def test_letf_baseline_is_half_hold_half_trend() -> None:
    close = walk()
    costs, periods = bt.Costs(), rb.bars_per_year(close.index)
    base = rb.letf_baseline(close, costs, 220, periods)
    hold = bt.run_vectorised(close, pd.Series(1.0, index=close.index), costs, 220)
    assert base.index.equals(hold.index)
    # the hold sleeve is always half the book; the ×3 sleeve adds 1.5 when above its average
    assert set(np.round(base["position"].unique(), 6)) <= {0.5, 2.0}


def test_cfd_costs_pay_spread_on_changes_and_swap_per_night() -> None:
    days = pd.DatetimeIndex(["2026-01-08", "2026-01-09", "2026-01-12", "2026-01-13"])  # Thu..Tue
    frame = pd.DataFrame(
        {"position": [1.0, 1.0, 1.0, 0.0], "asset": [0.0, 0.0, 0.0, 0.0]}, index=days
    )
    net = rb.cfd_net(frame, spread_bps=2.0, swap_long=0.0365, swap_short=0.0)
    assert net.iloc[0] == pytest.approx(-(2.0 / 10_000) - 0.0365 / 365)  # entry + one night
    assert net.iloc[2] == pytest.approx(-3 * 0.0365 / 365)  # Friday to Monday: three nights
    assert net.iloc[3] == pytest.approx(-2.0 / 10_000)  # exit, nothing held overnight


def test_protocol_with_the_letf_baseline() -> None:
    close = walk()
    strat = strategy(
        "ema_cross", {"fast": [10, 20, 30], "slow": [100, 150, 200]}, {"fast": 20, "slow": 150}
    )
    out = run(strat, close, letf_baseline="on")
    assert "letf_baseline" in {r["series"] for r in out["epochs"]}
    assert out["checks"][-1]["check"] == "Leveraged baseline"
    assert "Calmar" in out["checks"][-1]["detail"]


def test_prop_challenge_with_cfd_costs() -> None:
    """A heavy swap (30 %/yr, crypto CFDs) can only lower the chance of passing."""
    close = walk()
    strat = strategy("ema_cross", {"fast": [10, 20], "slow": [100, 150]}, {"fast": 20, "slow": 150})
    inputs = AnalysisInputs(data={"strategy": strat, "prices": prices(close)})
    base = prop_challenge(inputs, ChallengeParams(vol_targets=[0.16], paths=600)).payload
    cfd = prop_challenge(
        inputs,
        ChallengeParams(vol_targets=[0.16], paths=600, cfd_spread_bps=2, cfd_swap_long=0.30),
    ).payload
    assert base["costs_model"] == "the strategy's fees and slippage"
    assert cfd["costs_model"].startswith("CFD: spread 2 bp")
    assert cfd["rows"][0]["p_funded"] <= base["rows"][0]["p_funded"]


# seeded walk (seed 11), ema_cross 20/150, 5 + 5 bp costs, warm-up 220, IS 60 %
GOLDEN: dict[str, Any] = {
    "is_sharpe": 0.44063798993329845,
    "lag1": 0.613379341431327,
    "epoch_split": "2013-01-09",
}
