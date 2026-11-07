"""Strategy builder engine (feature 321): schemas, validation, execution, simulation, templates."""

import numpy as np
import pandas as pd
import pytest

from fda_core import blocks as bk

DAYS = pd.bdate_range("2020-01-02", periods=600)


def bars(close: np.ndarray | list[float], low: list[float] | None = None) -> pd.DataFrame:
    c = np.asarray(close, dtype=float)
    idx = DAYS[: len(c)]
    lo = np.asarray(low, dtype=float) if low is not None else c
    return pd.DataFrame(
        {"open": c, "high": c, "low": lo, "close": c, "adj_close": c, "volume": 1e6}, index=idx
    )


def walk(seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return bars(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, len(DAYS)))))


# ── schemas ──────────────────────────────────────────────────────────────────


def test_ports_keep_the_clone_order() -> None:
    sim = bk.BLOCKS["PortfolioSimulation"]
    assert [p.name for p in sim.inputs] == [
        "EntrySignals",
        "ExitSignals",
        "CurrentClosePrice",
        "StopLossPrice",
        "InitialCash",
        "Leverage",
        "RiskRewardRatio",
        "MaxRiskDistance",
        "StopGap",
        "Slippage",
        "Size",
        "SizeType",
        "FeesPerUnit",
        "MinEquityThreshold",
        "Ticker",
    ]
    assert [p.name for p in sim.outputs] == ["Equity", "Cash", "DeltaPositions", "DeltaTypes"]
    candle = bk.BLOCKS["Candlestick"]
    assert [p.name for p in candle.outputs] == ["Open", "Close", "High", "Low", "Volume", "Closure"]
    assert {b["type"] for b in bk.catalogue()} == set(bk.BLOCKS)


# ── the simulation, checked by hand ──────────────────────────────────────────


def signals(n: int, on: list[int]) -> pd.Series:
    s = pd.Series(False, index=DAYS[:n])
    s.iloc[on] = True
    return s


def test_fills_fees_and_slippage_by_hand() -> None:
    close = pd.Series([10.0, 10.0, 11.0, 12.0, 12.0], index=DAYS[:5])
    sim = bk.simulate(
        signals(5, [1]),
        signals(5, [3]),
        close,
        initial_cash=1000,
        size=10,
        size_type="Units",
        fees_per_unit=0.1,
        slippage=0.05,
    )
    # buy 10 at 10.05 (fee 1) on day 1, sell 10 at 11.95 (fee 1) on day 3
    assert sim.trades[0]["entry_price"] == pytest.approx(10.05)
    assert sim.trades[0]["exit_price"] == pytest.approx(11.95)
    assert sim.trades[0]["pnl"] == pytest.approx(10 * (11.95 - 10.05) - 2)
    assert sim.equity.iloc[-1] == pytest.approx(1000 + 10 * (11.95 - 10.05) - 2)
    assert sim.fees == pytest.approx(2.0)
    assert sim.delta_types.tolist() == [0, bk.DELTA_ENTRY, 0, bk.DELTA_EXIT, 0]


def test_stop_take_profit_risk_skip_and_equity_floor() -> None:
    close = pd.Series([10.0, 10.0, 9.0, 10.0, 10.0, 13.5], index=DAYS[:6])
    stop = pd.Series(9.5, index=close.index)
    sim = bk.simulate(signals(6, [1]), signals(6, []), close, stop, size=1, size_type="Units")
    assert sim.trades[0]["exit"] == "stop"  # 9.0 ≤ the previous bar's level 9.5
    # take profit at entry + 2 × risk (10 − 9.5 = 0.5) = 11
    sim = bk.simulate(
        signals(6, [3]), signals(6, []), close, stop, risk_reward=2, size=1, size_type="Units"
    )
    assert sim.trades[0]["exit"] == "target"
    # a risk distance of 0.5 above the 0.4 limit: no trade
    sim = bk.simulate(
        signals(6, [1]),
        signals(6, []),
        close,
        stop,
        max_risk_distance=0.4,
        size=1,
        size_type="Units",
    )
    assert sim.trades == []
    # below the equity floor the account stops trading
    sim = bk.simulate(
        signals(6, [1, 4]),
        signals(6, [2]),
        close,
        initial_cash=100,
        size=50,
        size_type="Units",
        min_equity=60,
    )
    assert len(sim.trades) == 1  # lost 50 on day 2 → equity 50 < 60, the day-4 entry is skipped


def test_percent_of_equity_sizing_with_leverage() -> None:
    close = pd.Series([50.0, 50.0, 50.0], index=DAYS[:3])
    sim = bk.simulate(
        signals(3, [1]), signals(3, []), close, initial_cash=10_000, leverage=4, size=0.5
    )
    assert sim.delta_positions.iloc[1] == pytest.approx(0.5 * 10_000 * 4 / 50)


# ── graphs ───────────────────────────────────────────────────────────────────


def test_validation_names_each_problem() -> None:
    g = bk.template("ema_cross", "SPY")
    assert bk.validate(g) == []
    broken = {"blocks": g["blocks"], "wires": [w for w in g["wires"] if w["to"][0] != "ema_fast"]}
    problems = bk.validate(broken)
    assert any("ema_fast.Closure needs a wire" in p for p in problems)
    typed = {
        "blocks": g["blocks"],
        "wires": [*g["wires"][:-1], bk._wire("candle1", "Close", "portfolio1", "InitialCash")],
    }
    assert any("Timeseries into Real" in p for p in bk.validate(typed))
    cyclic = {
        "blocks": [
            {"id": "a", "type": "PreviousK", "params": {}},
            {"id": "b", "type": "PreviousK", "params": {}},
            *[b for b in g["blocks"] if b["type"] == "PortfolioSimulation"],
        ],
        "wires": [
            bk._wire("a", "Closure", "b", "Closure"),
            bk._wire("b", "Closure", "a", "Closure"),
        ],
    }
    assert any("cycle" in p for p in bk.validate(cyclic))


def test_ema_cross_template_runs_and_trades_on_crosses() -> None:
    data = walk()
    result = bk.run(bk.template("ema_cross", "SPY"), data)
    entries = result.outputs["entry_signal"]["Cross"]
    fast = data["adj_close"].ewm(span=12, adjust=False, min_periods=12).mean()
    slow = data["adj_close"].ewm(span=26, adjust=False, min_periods=26).mean()
    crossed = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    pd.testing.assert_series_equal(entries, crossed.fillna(False), check_names=False)
    sim = result.simulation
    assert sim.trades
    assert {t["exit"] for t in sim.trades} <= {"stop", "target", "signal"}
    perf = bk.performance(sim, 100_000, 252)
    assert perf["winning_trades"] + perf["losing_trades"] == perf["total_trades"]
    assert perf["final_equity"] == pytest.approx(sim.equity.iloc[-1])


def test_rsi_bounce_template_follows_the_notebook_conditions() -> None:
    data = walk(seed=3)
    out = bk.run(bk.template("rsi_bounce", "ES=F"), data).outputs
    close = data["adj_close"]
    delta = close.diff()
    rsi = 100 - 100 / (
        1 + delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0)).rolling(14).mean()
    )
    entry = (rsi.shift(1) < 20) & (rsi > 25)
    pd.testing.assert_series_equal(
        out["entrySignal"]["Closure"], entry.fillna(False), check_names=False
    )


def test_no_look_ahead_in_a_graph_run() -> None:
    data = walk(seed=5)
    a = bk.run(bk.template("ema_cross", "SPY"), data).simulation.equity
    altered = data.copy()
    altered.iloc[400:, :5] *= 0.5
    b = bk.run(bk.template("ema_cross", "SPY"), altered).simulation.equity
    pd.testing.assert_series_equal(a.iloc[:400], b.iloc[:400])


def test_templates_reproduce_the_clone_wiring() -> None:
    ema = bk.template("ema_cross", "XCME:ES")
    assert [b["id"] for b in ema["blocks"]] == [
        "candle1",
        "ema_fast",
        "ema_slow",
        "stop_loss",
        "entry_signal",
        "exit_signal",
        "portfolio1",
    ]
    assert len(ema["wires"]) == 14
    # the exit is the same block type wired the other way round
    assert bk._wire("ema_slow", "Closure", "exit_signal", "BelowSeriesClosure") in ema["wires"]
    rsi = bk.template("rsi_bounce", "XCME:ES")
    assert len(rsi["blocks"]) == 9
    assert len(rsi["wires"]) == 14
    with pytest.raises(bk.GraphError):
        bk.template("nope", "X")
