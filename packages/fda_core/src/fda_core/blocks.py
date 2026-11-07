"""Strategy builder engine (feature 321): typed blocks, graph validation, execution, simulation.

After the owner's Edgebot clone (`edgebot-clone/App-clone`, docs/ARCHITECTURE.md §4): a strategy is
a DAG of blocks joined by wires. Ports keep the clone's names, types and order. Every Timeseries
port carries a value series (`Closure`) and the clone's time axis (`T`); on daily bars T is the
bar dates, kept so graphs read exactly like the clone's.

Pure: no I/O. Data stays point-in-time — every block output at t uses bars ≤ t — and the
simulation trades at the close of the bar where a decision is known, holding from the next bar
(the engine's convention, `backtest.run_vectorised` lag 1).
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from fda_core.registry import InsufficientDataError

PortKind = Literal["Timeseries", "Real", "Integer", "Category"]


@dataclass(frozen=True, slots=True)
class Port:
    name: str
    kind: PortKind
    default: Any = None
    choices: tuple[str, ...] = ()
    required: bool = True
    help: str = ""


@dataclass(frozen=True, slots=True)
class BlockSpec:
    type: str
    title: str
    inputs: tuple[Port, ...]
    outputs: tuple[Port, ...]
    help: str
    fn: Callable[[dict[str, Any], pd.DataFrame], dict[str, Any]] = field(repr=False)


class GraphError(InsufficientDataError):
    """A graph that cannot run: the message says which block and port."""


def _ts(name: str, help: str = "", required: bool = True) -> Port:
    return Port(name, "Timeseries", required=required, help=help)


T_IN = _ts("T", "Time axis of the input series", required=False)


def _closure(values: pd.Series) -> dict[str, Any]:
    return {"T": pd.Series(values.index, index=values.index), "Closure": values}


# ── block functions (inputs: wired series by port name, plus scalar values) ──


def _candlestick(i: dict[str, Any], bars: pd.DataFrame) -> dict[str, Any]:
    close = bars["adj_close"].astype(float)
    ratio = close / bars["close"].astype(float) if "close" in bars else 1.0
    out = {
        "Open": bars["open"].astype(float) * ratio if "open" in bars else close,
        "Close": close,
        "High": bars["high"].astype(float) * ratio if "high" in bars else close,
        "Low": bars["low"].astype(float) * ratio if "low" in bars else close,
        "Volume": bars["volume"].astype(float) if "volume" in bars else close * np.nan,
        "Closure": pd.Series(True, index=bars.index),
    }
    return out


def _ema(i: dict[str, Any], _: pd.DataFrame) -> dict[str, Any]:
    n = int(i["Period"])
    if n < 1:
        raise GraphError("Exponential Moving Average needs Period ≥ 1")
    return _closure(i["Closure"].ewm(span=n, adjust=False, min_periods=n).mean())


def _rsi(i: dict[str, Any], _: pd.DataFrame) -> dict[str, Any]:
    """RSI as in the owner's notebook: 100 − 100 / (1 + mean gain / mean loss), simple averages
    over `Period` bars."""
    n = int(i["Period"])
    delta = i["Closure"].diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    return _closure(100 - 100 / (1 + gain / loss))


AGGREGATES = ("Min", "Max", "Mean", "Sum")


def _rolling(i: dict[str, Any], _: pd.DataFrame) -> dict[str, Any]:
    k, op = int(i["K"]), str(i["Operation"])
    if op not in AGGREGATES:
        raise GraphError(f"Rolling Aggregate: unknown operation {op!r}")
    window = i["Closure"].rolling(k)
    return _closure(getattr(window, op.lower())())


def _previous(i: dict[str, Any], _: pd.DataFrame) -> dict[str, Any]:
    return _closure(i["Closure"].shift(int(i["K"])))


OPERATORS: dict[str, Callable[[pd.Series, float], pd.Series]] = {
    "<": lambda s, c: s < c,
    "<=": lambda s, c: s <= c,
    ">": lambda s, c: s > c,
    ">=": lambda s, c: s >= c,
    "==": lambda s, c: s == c,
}


def _binary(i: dict[str, Any], _: pd.DataFrame) -> dict[str, Any]:
    op = str(i["Operator"])
    if op not in OPERATORS:
        raise GraphError(f"Binary Operation With Constant: unknown operator {op!r}")
    s = i["Closure"]
    return _closure(OPERATORS[op](s, float(i["Constant"])) & s.notna())


def _and(i: dict[str, Any], _: pd.DataFrame) -> dict[str, Any]:
    a, b = i["InputA"].astype(bool), i["InputB"].astype(bool)
    return {"Closure": a & b}


def _cross(i: dict[str, Any], _: pd.DataFrame) -> dict[str, Any]:
    """True on the bar where the Below series goes from ≤ the Above series to above it."""
    below, above = i["BelowSeriesClosure"], i["AboveSeriesClosure"]
    now = (below > above).fillna(False)
    before = (below.shift(1) <= above.shift(1)).fillna(False)
    return {"Cross": now & before}


SIZE_TYPES = ("PercentOfEquity", "Units")


def _portfolio(i: dict[str, Any], bars: pd.DataFrame) -> dict[str, Any]:
    sim = simulate(
        entries=i["EntrySignals"],
        exits=i["ExitSignals"],
        close=i["CurrentClosePrice"],
        stop=i.get("StopLossPrice"),
        initial_cash=float(i["InitialCash"]),
        leverage=float(i["Leverage"]),
        risk_reward=float(i["RiskRewardRatio"]),
        max_risk_distance=float(i["MaxRiskDistance"]),
        stop_gap=float(i["StopGap"]),
        slippage=float(i["Slippage"]),
        size=float(i["Size"]),
        size_type=str(i["SizeType"]),
        fees_per_unit=float(i["FeesPerUnit"]),
        min_equity=float(i["MinEquityThreshold"]),
    )
    return {
        "Equity": sim.equity,
        "Cash": sim.cash,
        "DeltaPositions": sim.delta_positions,
        "DeltaTypes": sim.delta_types,
        "_simulation": sim,
    }


# ── the catalogue (ports in the clone's order) ───────────────────────────────

BLOCKS: dict[str, BlockSpec] = {
    spec.type: spec
    for spec in (
        BlockSpec(
            "Candlestick",
            "Candlestick",
            (
                _ts("PriceSeries", "Price feed (the strategy's stored daily bars)", required=False),
                _ts("VolumeSeries", "Volume feed (the stored daily bars)", required=False),
                Port("Interval", "Category", "1d", ("1d",), help="Bar length (daily bars only)"),
                Port("SnapshotPeriod", "Category", "1d", ("1d",)),
                Port("Ticker", "Category", "", help="The strategy's symbol"),
                Port("TradesToInclude", "Category", "AllTrades", ("AllTrades",)),
            ),
            (
                _ts("Open"),
                _ts("Close"),
                _ts("High"),
                _ts("Low"),
                _ts("Volume"),
                _ts("Closure", "Completed-bar marker"),
            ),
            "Daily OHLCV bars of the strategy's symbol (adjusted prices).",
            _candlestick,
        ),
        BlockSpec(
            "ExponentialMovingAverage",
            "Exponential Moving Average",
            (T_IN, _ts("Closure"), Port("Period", "Integer", 12)),
            (_ts("T"), _ts("Closure")),
            "EMA, α = 2/(n+1), started on the first value, empty for the first n − 1 bars.",
            _ema,
        ),
        BlockSpec(
            "RSI",
            "RSI",
            (T_IN, _ts("Closure"), Port("Period", "Integer", 14)),
            (_ts("T"), _ts("Closure")),
            "Relative Strength Index (0–100) on simple averages of gains and losses.",
            _rsi,
        ),
        BlockSpec(
            "RollingAggregate",
            "Rolling Aggregate",
            (
                T_IN,
                _ts("Closure"),
                Port("K", "Integer", 2),
                Port("Operation", "Category", "Min", AGGREGATES),
            ),
            (_ts("T"), _ts("Closure")),
            "Min, max, mean or sum over the last K bars (this bar included).",
            _rolling,
        ),
        BlockSpec(
            "PreviousK",
            "Previous K",
            (T_IN, _ts("Closure"), Port("K", "Integer", 1)),
            (_ts("T"), _ts("Closure")),
            "The value K bars back.",
            _previous,
        ),
        BlockSpec(
            "BinaryOperationWithConstant",
            "Binary Operation With Constant",
            (
                T_IN,
                _ts("Closure"),
                Port("Operator", "Category", "<", tuple(OPERATORS)),
                Port("Constant", "Real", 0.0),
            ),
            (_ts("T"), _ts("Closure")),
            "Compares a series with a constant; false where the series is empty.",
            _binary,
        ),
        BlockSpec(
            "LogicalAnd",
            "Logical And",
            (_ts("InputA"), _ts("InputB")),
            (_ts("Closure"),),
            "Both conditions true on the bar.",
            _and,
        ),
        BlockSpec(
            "BelowCrossesAbove",
            "Below Crosses Above",
            (
                _ts("BelowSeries", required=False),
                _ts("BelowSeriesClosure"),
                _ts("AboveSeries", required=False),
                _ts("AboveSeriesClosure"),
            ),
            (_ts("Cross"),),
            "True on the bar where the first series crosses above the second.",
            _cross,
        ),
        BlockSpec(
            "PortfolioSimulation",
            "Portfolio Simulation",
            (
                _ts("EntrySignals"),
                _ts("ExitSignals"),
                _ts("CurrentClosePrice"),
                _ts("StopLossPrice", "Stop level, recomputed every bar", required=False),
                Port("InitialCash", "Real", 100_000.0),
                Port("Leverage", "Real", 1.0),
                Port(
                    "RiskRewardRatio", "Real", 0.0, help="Take profit at this multiple of the risk"
                ),
                Port(
                    "MaxRiskDistance", "Real", 0.0, help="Skip entries riskier than this (0 = off)"
                ),
                Port("StopGap", "Real", 0.0, help="Stop placed this far below the stop level"),
                Port("Slippage", "Real", 0.0, help="Price units lost on each fill"),
                Port("Size", "Real", 1.0),
                Port("SizeType", "Category", "PercentOfEquity", SIZE_TYPES),
                Port("FeesPerUnit", "Real", 0.0),
                Port("MinEquityThreshold", "Real", 0.0, help="Stop trading below this equity"),
                Port("Ticker", "Category", "", help="The strategy's symbol"),
            ),
            (_ts("Equity"), _ts("Cash"), _ts("DeltaPositions"), _ts("DeltaTypes")),
            "Long-only simulation: entries and exits at the close of the signal bar, stop and "
            "take-profit checked on later closes, fees per unit and slippage on every fill.",
            _portfolio,
        ),
    )
}


def catalogue() -> list[dict[str, Any]]:
    """The block schemas as data, for the API and the UI (never redefined in the frontend)."""

    def port(p: Port) -> dict[str, Any]:
        return {
            "name": p.name,
            "kind": p.kind,
            "default": p.default,
            "choices": list(p.choices),
            "required": p.required,
            "help": p.help,
        }

    return [
        {
            "type": b.type,
            "title": b.title,
            "help": b.help,
            "inputs": [port(p) for p in b.inputs],
            "outputs": [port(p) for p in b.outputs],
        }
        for b in BLOCKS.values()
    ]


# ── the simulation ───────────────────────────────────────────────────────────

DELTA_ENTRY, DELTA_EXIT, DELTA_STOP, DELTA_TARGET = 1, -1, -2, 2


@dataclass
class Simulation:
    equity: pd.Series
    cash: pd.Series
    delta_positions: pd.Series
    delta_types: pd.Series
    trades: list[dict[str, Any]]
    fees: float


def simulate(
    entries: pd.Series,
    exits: pd.Series,
    close: pd.Series,
    stop: pd.Series | None = None,
    initial_cash: float = 100_000.0,
    leverage: float = 1.0,
    risk_reward: float = 0.0,
    max_risk_distance: float = 0.0,
    stop_gap: float = 0.0,
    slippage: float = 0.0,
    size: float = 1.0,
    size_type: str = "PercentOfEquity",
    fees_per_unit: float = 0.0,
    min_equity: float = 0.0,
) -> Simulation:
    """Long-only, one position at a time, decided and filled at the close of each bar:

    - flat and an entry (without an exit) on bar t → buy at close_t + slippage. Size: Units, or
      PercentOfEquity × equity × leverage / price. With a stop level, the entry is skipped when
      close − stop exceeds MaxRiskDistance (> 0);
    - long → on a later bar, the first that applies: close ≤ the previous bar's stop level −
      StopGap (stop), close ≥ entry + RiskRewardRatio × initial risk (take profit, > 0), or an
      exit signal; sell at close − slippage;
    - fees = FeesPerUnit × units on every fill; equity marked to the close every bar; once
      equity is below MinEquityThreshold (> 0) the account stops trading.
    The stop level used on bar t is the one known at close t−1: no look-ahead."""
    if size_type not in SIZE_TYPES:
        raise GraphError(f"Portfolio Simulation: unknown SizeType {size_type!r}")
    idx = close.index
    c = close.to_numpy(dtype=float)
    ent = entries.reindex(idx).fillna(False).astype(bool).to_numpy()
    ext = exits.reindex(idx).fillna(False).astype(bool).to_numpy()
    stp = None if stop is None else stop.reindex(idx).to_numpy(dtype=float)
    n = len(c)
    cash, units = initial_cash, 0.0
    eq = np.empty(n)
    cash_path = np.empty(n)
    d_pos = np.zeros(n)
    d_type = np.zeros(n)
    trades: list[dict[str, Any]] = []
    fees_total = 0.0
    entry_price = risk = entry_fee = 0.0
    entry_i = -1
    halted = False
    for t in range(n):
        price = c[t]
        if units > 0 and math.isfinite(price):
            level = stp[t - 1] - stop_gap if stp is not None and t > 0 else math.nan
            target = entry_price + risk_reward * risk if risk_reward > 0 and risk > 0 else math.inf
            kind = 0
            if math.isfinite(level) and price <= level:
                kind = DELTA_STOP
            elif price >= target:
                kind = DELTA_TARGET
            elif ext[t]:
                kind = DELTA_EXIT
            if kind:
                fill = price - slippage
                fee = fees_per_unit * units
                cash += units * fill - fee
                fees_total += fee
                trades.append(
                    {
                        "entry_date": idx[entry_i].date().isoformat(),
                        "exit_date": idx[t].date().isoformat(),
                        "entry_price": entry_price,
                        "exit_price": fill,
                        "units": units,
                        "pnl": units * (fill - entry_price) - fee - entry_fee,
                        "exit": {DELTA_STOP: "stop", DELTA_TARGET: "target", DELTA_EXIT: "signal"}[
                            kind
                        ],
                    }
                )
                d_pos[t], d_type[t] = -units, kind
                units = 0.0
        equity = cash + units * (price if math.isfinite(price) else 0.0)
        if min_equity > 0 and equity < min_equity:
            halted = True
        if units == 0 and not halted and ent[t] and not ext[t] and math.isfinite(price):
            fill = price + slippage
            stop_now = stp[t] if stp is not None else math.nan
            distance = fill - stop_now if math.isfinite(stop_now) else math.nan
            too_risky = (
                max_risk_distance > 0 and math.isfinite(distance) and distance > max_risk_distance
            )
            qty = size if size_type == "Units" else size * equity * leverage / fill
            if qty > 0 and not too_risky:
                fee = fees_per_unit * qty
                cash -= qty * fill + fee
                fees_total += fee
                units, entry_price, entry_i = qty, fill, t
                risk = distance if math.isfinite(distance) and distance > 0 else 0.0
                entry_fee = fee
                d_pos[t], d_type[t] = qty, DELTA_ENTRY
                equity = cash + units * price
        eq[t], cash_path[t] = equity, cash
    return Simulation(
        pd.Series(eq, index=idx),
        pd.Series(cash_path, index=idx),
        pd.Series(d_pos, index=idx),
        pd.Series(d_type, index=idx),
        trades,
        fees_total,
    )


# ── graph validation and execution ───────────────────────────────────────────


def _inputs_of(graph: Mapping[str, Any]) -> dict[tuple[str, str], tuple[str, str]]:
    wired: dict[tuple[str, str], tuple[str, str]] = {}
    for w in graph.get("wires", []):
        (src, out), (dst, inp) = w["from"], w["to"]
        if (dst, inp) in wired:
            raise GraphError(f"{dst}.{inp} has two wires: an input takes one")
        wired[(dst, inp)] = (src, out)
    return wired


def validate(graph: Mapping[str, Any]) -> list[str]:
    """Every problem of the graph, in words (empty = it can run)."""
    problems: list[str] = []
    blocks = {b["id"]: b for b in graph.get("blocks", [])}
    if len(blocks) != len(graph.get("blocks", [])):
        problems.append("Two blocks share a name")
    try:
        wired = _inputs_of(graph)
    except GraphError as error:
        return [str(error)]
    for (dst, inp), (src, out) in wired.items():
        if src not in blocks or dst not in blocks:
            problems.append(f"Wire {src}.{out} → {dst}.{inp} joins a missing block")
            continue
        s_spec, d_spec = BLOCKS.get(blocks[src]["type"]), BLOCKS.get(blocks[dst]["type"])
        if s_spec is None or d_spec is None:
            continue
        s_port = next((p for p in s_spec.outputs if p.name == out), None)
        d_port = next((p for p in d_spec.inputs if p.name == inp), None)
        if s_port is None or d_port is None:
            problems.append(f"Wire {src}.{out} → {dst}.{inp}: no such port")
        elif s_port.kind != d_port.kind:
            problems.append(f"Wire {src}.{out} → {dst}.{inp}: {s_port.kind} into {d_port.kind}")
    for b in blocks.values():
        spec = BLOCKS.get(b["type"])
        if spec is None:
            problems.append(f"{b['id']}: unknown block type {b['type']!r}")
            continue
        for p in spec.inputs:
            if p.kind == "Timeseries" and p.required and (b["id"], p.name) not in wired:
                problems.append(f"{b['id']}.{p.name} needs a wire")
    sims = [b for b in blocks.values() if b["type"] == "PortfolioSimulation"]
    if len(sims) != 1:
        problems.append("A strategy needs exactly one Portfolio Simulation block")
    dangling = any(src not in blocks or dst not in blocks for (dst, _), (src, _) in wired.items())
    if not dangling:
        try:
            _order(graph)
        except GraphError as error:
            problems.append(str(error))
    return problems


def _order(graph: Mapping[str, Any]) -> list[str]:
    """Topological order (Kahn); a cycle is an error."""
    ids = [b["id"] for b in graph["blocks"]]
    deps: dict[str, set[str]] = {i: set() for i in ids}
    for (dst, _), (src, _) in _inputs_of(graph).items():
        deps[dst].add(src)
    order: list[str] = []
    ready = [i for i in ids if not deps[i]]
    while ready:
        node = ready.pop(0)
        order.append(node)
        for other in ids:
            if node in deps[other]:
                deps[other].discard(node)
                if not deps[other] and other not in order and other not in ready:
                    ready.append(other)
    if len(order) != len(ids):
        raise GraphError("The graph has a cycle: wires must flow one way")
    return order


@dataclass
class GraphRun:
    outputs: dict[str, dict[str, Any]]
    simulation: Simulation


def run(graph: Mapping[str, Any], bars: pd.DataFrame) -> GraphRun:
    """Execute every block in topological order on the strategy's daily bars."""
    problems = validate(graph)
    if problems:
        raise GraphError("; ".join(problems))
    blocks = {b["id"]: b for b in graph["blocks"]}
    wired = _inputs_of(graph)
    outputs: dict[str, dict[str, Any]] = {}
    for block_id in _order(graph):
        b = blocks[block_id]
        spec = BLOCKS[b["type"]]
        values: dict[str, Any] = {}
        for p in spec.inputs:
            source = wired.get((block_id, p.name))
            if source is not None:
                values[p.name] = outputs[source[0]][source[1]]
            elif p.kind != "Timeseries":
                values[p.name] = (b.get("params") or {}).get(p.name, p.default)
        outputs[block_id] = spec.fn(values, bars)
    sim_id = next(i for i, b in blocks.items() if b["type"] == "PortfolioSimulation")
    return GraphRun(outputs, outputs[sim_id]["_simulation"])


# ── performance summary (the clone's Results grid) ───────────────────────────


def performance(sim: Simulation, initial_cash: float, periods: float) -> dict[str, Any]:
    """Total / winning / losing trades, win rate, realized P&L, fees, max drawdown (% and bars),
    profit factor, max consecutive losses, Sharpe, skewness, excess kurtosis, final equity."""
    pnl = np.array([t["pnl"] for t in sim.trades], dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    eq = sim.equity
    r = eq.pct_change().dropna()
    dd = eq / eq.cummax() - 1
    under = (dd < 0).astype(int)
    longest = int(under.groupby((under != under.shift()).cumsum()).sum().max()) if len(under) else 0
    streak = best = 0
    for x in pnl:
        streak = streak + 1 if x <= 0 else 0
        best = max(best, streak)
    sd = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    return {
        "total_trades": len(pnl),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": float(len(wins) / len(pnl)) if len(pnl) else None,
        "total_realized_pnl": float(pnl.sum()),
        "total_fees": sim.fees,
        "max_drawdown": float(-dd.min()) if len(dd) else 0.0,
        "max_drawdown_duration_bars": longest,
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else None,
        "max_consecutive_losses": best,
        "sharpe_ratio": float(r.mean() / sd * math.sqrt(periods)) if sd > 0 else None,
        "skewness": float(r.skew()) if len(r) > 2 else None,
        "excess_kurtosis": float(r.kurt()) if len(r) > 3 else None,
        "final_equity": float(eq.iloc[-1]) if len(eq) else initial_cash,
    }


# ── templates: the clone's two graphs, wire for wire ─────────────────────────


def _wire(src: str, out: str, dst: str, inp: str) -> dict[str, list[str]]:
    return {"from": [src, out], "to": [dst, inp]}


def _portfolio_block(x: float, y: float, symbol: str) -> dict[str, Any]:
    return {
        "id": "portfolio1",
        "type": "PortfolioSimulation",
        "x": x,
        "y": y,
        "params": {
            "InitialCash": 100000,
            "Leverage": 4,
            "RiskRewardRatio": 3,
            "MaxRiskDistance": 10,
            "StopGap": 0.25,
            "Slippage": 0,
            "Size": 0.01,
            "SizeType": "PercentOfEquity",
            "FeesPerUnit": 3,
            "MinEquityThreshold": 20000,
            "Ticker": symbol,
        },
    }


def _candle(x: float, y: float, symbol: str) -> dict[str, Any]:
    return {
        "id": "candle1",
        "type": "Candlestick",
        "x": x,
        "y": y,
        "params": {
            "Interval": "1d",
            "SnapshotPeriod": "1d",
            "Ticker": symbol,
            "TradesToInclude": "AllTrades",
        },
    }


def template(name: str, symbol: str) -> dict[str, Any]:
    """`ema_cross`, `rsi_bounce` (the clone's graphs) or `blank` (a candle and a portfolio)."""
    if name == "ema_cross":
        blocks = [
            _candle(0, 400, symbol),
            {
                "id": "ema_fast",
                "type": "ExponentialMovingAverage",
                "x": 460,
                "y": 120,
                "params": {"Period": 12},
                "tunable": {"Period": [8, 12, 16]},
            },
            {
                "id": "ema_slow",
                "type": "ExponentialMovingAverage",
                "x": 460,
                "y": 488,
                "params": {"Period": 26},
                "tunable": {"Period": [21, 26, 34]},
            },
            {
                "id": "stop_loss",
                "type": "RollingAggregate",
                "x": 460,
                "y": 856,
                "params": {"K": 2, "Operation": "Min"},
            },
            {"id": "entry_signal", "type": "BelowCrossesAbove", "x": 920, "y": 280, "params": {}},
            {"id": "exit_signal", "type": "BelowCrossesAbove", "x": 920, "y": 724, "params": {}},
            _portfolio_block(1400, 400, symbol),
        ]
        wires = [
            _wire("candle1", "Close", "ema_fast", "Closure"),
            _wire("candle1", "Closure", "ema_fast", "T"),
            _wire("candle1", "Close", "ema_slow", "Closure"),
            _wire("candle1", "Closure", "ema_slow", "T"),
            _wire("candle1", "Low", "stop_loss", "Closure"),
            _wire("candle1", "Closure", "stop_loss", "T"),
            _wire("ema_fast", "Closure", "entry_signal", "BelowSeriesClosure"),
            _wire("ema_slow", "Closure", "entry_signal", "AboveSeriesClosure"),
            _wire("ema_slow", "Closure", "exit_signal", "BelowSeriesClosure"),
            _wire("ema_fast", "Closure", "exit_signal", "AboveSeriesClosure"),
            _wire("entry_signal", "Cross", "portfolio1", "EntrySignals"),
            _wire("exit_signal", "Cross", "portfolio1", "ExitSignals"),
            _wire("candle1", "Close", "portfolio1", "CurrentClosePrice"),
            _wire("stop_loss", "Closure", "portfolio1", "StopLossPrice"),
        ]
    elif name == "rsi_bounce":
        blocks = [
            _candle(0, 400, symbol),
            {
                "id": "rsi1",
                "type": "RSI",
                "x": 460,
                "y": 400,
                "params": {"Period": 14},
                "tunable": {"Period": [10, 14, 21]},
            },
            {
                "id": "stopLoss",
                "type": "RollingAggregate",
                "x": 460,
                "y": 768,
                "params": {"K": 2, "Operation": "Min"},
            },
            {"id": "prevRsi", "type": "PreviousK", "x": 920, "y": 140, "params": {"K": 1}},
            {
                "id": "prevBelow20",
                "type": "BinaryOperationWithConstant",
                "x": 1380,
                "y": 140,
                "params": {"Operator": "<", "Constant": 20},
            },
            {
                "id": "currAbove25",
                "type": "BinaryOperationWithConstant",
                "x": 1380,
                "y": 584,
                "params": {"Operator": ">", "Constant": 25},
            },
            {
                "id": "exitSignal",
                "type": "BinaryOperationWithConstant",
                "x": 1380,
                "y": 1028,
                "params": {"Operator": ">", "Constant": 50},
            },
            {"id": "entrySignal", "type": "LogicalAnd", "x": 1840, "y": 340, "params": {}},
            _portfolio_block(2300, 500, symbol),
        ]
        wires = [
            _wire("candle1", "Close", "rsi1", "Closure"),
            _wire("candle1", "Closure", "rsi1", "T"),
            _wire("rsi1", "Closure", "prevRsi", "Closure"),
            _wire("prevRsi", "Closure", "prevBelow20", "Closure"),
            _wire("rsi1", "Closure", "currAbove25", "Closure"),
            _wire("prevBelow20", "Closure", "entrySignal", "InputA"),
            _wire("currAbove25", "Closure", "entrySignal", "InputB"),
            _wire("rsi1", "Closure", "exitSignal", "Closure"),
            _wire("candle1", "Low", "stopLoss", "Closure"),
            _wire("candle1", "Closure", "stopLoss", "T"),
            _wire("entrySignal", "Closure", "portfolio1", "EntrySignals"),
            _wire("exitSignal", "Closure", "portfolio1", "ExitSignals"),
            _wire("candle1", "Close", "portfolio1", "CurrentClosePrice"),
            _wire("stopLoss", "Closure", "portfolio1", "StopLossPrice"),
        ]
    elif name == "blank":
        blocks = [_candle(0, 300, symbol), _portfolio_block(900, 300, symbol)]
        wires = [_wire("candle1", "Close", "portfolio1", "CurrentClosePrice")]
    else:
        raise GraphError(f"Unknown template {name!r} (ema_cross, rsi_bounce, blank)")
    return {"blocks": blocks, "wires": wires}


TEMPLATES = ("ema_cross", "rsi_bounce", "blank")


# ── bridge to the backtest lab (template `graph`) ────────────────────────────


def tunable_grid(graph: Mapping[str, Any]) -> dict[str, list[float]]:
    """The grid of a graph strategy: every block port marked tunable, keyed "block.Port"."""
    out: dict[str, list[float]] = {}
    for b in graph.get("blocks", []):
        for port, values in (b.get("tunable") or {}).items():
            out[f"{b['id']}.{port}"] = [float(v) for v in values]
    return out


def tunable_params(graph: Mapping[str, Any]) -> dict[str, float]:
    """The current value of every tunable port (the strategy's own parameters)."""
    out: dict[str, float] = {}
    for b in graph.get("blocks", []):
        for port in b.get("tunable") or {}:
            out[f"{b['id']}.{port}"] = float((b.get("params") or {}).get(port, 0))
    return out


def apply_params(graph: Mapping[str, Any], params: Mapping[str, float]) -> dict[str, Any]:
    """A copy of the graph with "block.Port" values set (integers stay integers)."""
    blocks_out = []
    for b in graph["blocks"]:
        p = dict(b.get("params") or {})
        for key, value in params.items():
            block_id, _, port = key.partition(".")
            if block_id == b["id"]:
                p[port] = int(value) if float(value).is_integer() else value
        blocks_out.append({**b, "params": p})
    return {**graph, "blocks": blocks_out}


def bars_from_close(close: pd.Series) -> pd.DataFrame:
    """Bars from closes only (the lab passes closes): open = high = low = close."""
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "adj_close": close},
        index=close.index,
    )


def graph_position(graph: Mapping[str, Any], close: pd.Series) -> pd.Series:
    """1 while the graph's Portfolio Simulation holds a position, decided at each close — the
    target the lab's engine shifts and costs like any template. Stops read closes here."""
    sim = run(graph, bars_from_close(close)).simulation
    held = sim.delta_positions.cumsum().round(12) > 0
    return held.astype(float)
