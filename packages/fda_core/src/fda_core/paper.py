"""Paper trading simulator (M29, TR-040) and execution analytics (TR-042): pure functions.

Stored state is only the account and its orders; fills, positions, cash and statements are
replayed deterministically from daily bars, so the P&L reconciles to the fills by construction.

Fill rules (daily bars, unadjusted OHLC): an order decided on session d is eligible from the next
session d+1 (no look-ahead). Market: next open. Limit: when the bar reaches the limit, at the
better of open and limit. Stop: when the bar trades through the stop, at the worse of open and
stop. Stop-limit: triggered as a stop, then fills only if the trigger price respects the limit,
else it rests as a limit. Trailing stop (sell): stop = highest high since eligibility × (1 − trail);
(buy): lowest low × (1 + trail). DAY orders live for one eligible session, GTC until filled or
cancelled. Every fill pays a half-spread (bps) and a square-root impact η·σ·√(Q/ADV₂₀) on its
price, and a fee in bps of the notional (at least the minimum fee). Cash is kept in integer cents.
No margin and no shorting: a buy above the cash is rejected, a sell is capped at the position.
"""

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd

OrderType = Literal["market", "limit", "stop", "stop_limit", "trailing_stop"]
Side = Literal["buy", "sell"]
TIF = Literal["day", "gtc"]


@dataclass(frozen=True, slots=True)
class CostModel:
    fee_bps: float = 1.0
    min_fee: float = 1.0
    half_spread_bps: float = 2.0
    impact_eta: float = 0.7

    def impact(self, shares: float, adv: float, sigma: float) -> float:
        """Square-root temporary impact as a fraction of price (TR-013 model)."""
        if adv <= 0 or not math.isfinite(sigma):
            return 0.0
        return self.impact_eta * sigma * math.sqrt(shares / adv)


@dataclass(frozen=True, slots=True)
class Order:
    id: int
    symbol: str
    side: Side
    quantity: float
    order_type: OrderType = "market"
    limit_price: float | None = None
    stop_price: float | None = None
    trail_pct: float | None = None
    tif: TIF = "gtc"
    decision_date: date = date(1970, 1, 1)
    decision_price: float = math.nan
    cancelled_on: date | None = None
    source: str = "manual"


@dataclass(slots=True)
class Fill:
    order_id: int
    date: date
    symbol: str
    side: Side
    quantity: float
    price: float
    arrival: float  # open of the fill session
    reference: float  # trigger / limit / open the fill is measured from, before costs
    fee: float
    model_cost_bps: float
    decision_price: float
    typical_price: float  # (H + L + C) / 3, daily VWAP proxy

    @property
    def notional(self) -> float:
        return round(self.quantity * self.price, 2)


@dataclass(slots=True)
class OrderState:
    order: Order
    status: str = "working"
    filled: float = 0.0
    reason: str = ""
    extreme: float | None = None  # trailing high / low watermark
    triggered: bool = False


@dataclass(slots=True)
class Simulation:
    fills: list[Fill] = field(default_factory=list)
    orders: list[OrderState] = field(default_factory=list)
    statement: pd.DataFrame = field(default_factory=pd.DataFrame)
    positions: dict[str, dict[str, float]] = field(default_factory=dict)
    cash_cents: int = 0
    realised: float = 0.0
    fees: float = 0.0


def _cents(x: float) -> int:
    return round(x * 100)


def _limit_ok(side: Side, price: float, limit: float) -> bool:
    return price <= limit if side == "buy" else price >= limit


def _trigger(state: OrderState, bar: pd.Series) -> float | None:
    """Reference execution price on this bar before costs, or None if the order does not fill."""
    o = state.order
    op, hi, lo = float(bar["open"]), float(bar["high"]), float(bar["low"])
    buy = o.side == "buy"
    if o.order_type == "market":
        return op
    if o.order_type == "limit":
        assert o.limit_price is not None
        if buy and lo <= o.limit_price:
            return min(op, o.limit_price)
        if not buy and hi >= o.limit_price:
            return max(op, o.limit_price)
        return None
    if o.order_type == "trailing_stop":
        assert o.trail_pct is not None
        # watermark from the previous bars (the stop is set before this bar trades)
        mark = state.extreme if state.extreme is not None else op
        stop = mark * (1 - o.trail_pct) if not buy else mark * (1 + o.trail_pct)
        hit = (lo <= stop) if not buy else (hi >= stop)
        state.extreme = max(mark, hi) if not buy else min(mark, lo)
        if hit:
            return min(op, stop) if not buy else max(op, stop)
        return None
    assert o.stop_price is not None
    if state.triggered:  # stop-limit resting as a limit
        assert o.limit_price is not None
        if buy and lo <= o.limit_price:
            return min(op, o.limit_price)
        if not buy and hi >= o.limit_price:
            return max(op, o.limit_price)
        return None
    hit = (hi >= o.stop_price) if buy else (lo <= o.stop_price)
    if not hit:
        return None
    level = max(op, o.stop_price) if buy else min(op, o.stop_price)
    if o.order_type == "stop":
        return level
    state.triggered = True
    assert o.limit_price is not None
    return level if _limit_ok(o.side, level, o.limit_price) else None


def simulate(
    starting_cash: float,
    orders: list[Order],
    bars: dict[str, pd.DataFrame],
    costs: CostModel,
    until: date | None = None,
) -> Simulation:
    """Replays every order over the stored bars (columns open, high, low, close, volume, index =
    session dates) and returns fills, order states, positions and a daily statement."""
    sim = Simulation(cash_cents=_cents(starting_cash))
    states = [OrderState(o) for o in sorted(orders, key=lambda o: (o.decision_date, o.id))]
    sim.orders = states
    frames = {s: b.sort_index() for s, b in bars.items()}
    sessions = sorted({d.date() for b in frames.values() for d in pd.DatetimeIndex(b.index)})
    if until is not None:
        sessions = [d for d in sessions if d <= until]
    stats: dict[str, pd.DataFrame] = {}
    for s, b in frames.items():
        r = np.log(b["close"].astype(float)).diff()
        stats[s] = pd.DataFrame(
            {
                "sigma": r.rolling(20, min_periods=5).std().shift(1),
                "adv": b["volume"].astype(float).rolling(20, min_periods=5).mean().shift(1),
            },
            index=b.index,
        )
    qty: dict[str, float] = {}
    cost_basis: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    first = min((o.decision_date for o in orders), default=None)
    for day in sessions:
        ts = pd.Timestamp(day)
        for st in states:
            o = st.order
            if st.status != "working" or day <= o.decision_date:
                continue
            if o.cancelled_on is not None and day > o.cancelled_on:
                st.status, st.reason = "cancelled", f"cancelled on {o.cancelled_on.isoformat()}"
                continue
            frame = frames.get(o.symbol)
            if frame is None or ts not in frame.index:
                continue  # no session for this symbol today
            bar = frame.loc[ts]
            ref = _trigger(st, bar)
            if ref is None:
                if o.tif == "day":
                    st.status, st.reason = "expired", "day order not filled"
                continue
            shares = o.quantity
            if o.side == "sell":
                shares = min(shares, qty.get(o.symbol, 0.0))
                if shares <= 0:
                    st.status, st.reason = "rejected", "no position to sell (no shorting)"
                    continue
            s_stats = stats[o.symbol].loc[ts]
            impact = costs.impact(shares, float(s_stats["adv"]), float(s_stats["sigma"]))
            slip = costs.half_spread_bps / 10_000 + impact
            price = round(ref * (1 + slip) if o.side == "buy" else ref * (1 - slip), 4)
            notional = round(shares * price, 2)
            fee = round(max(notional * costs.fee_bps / 10_000, costs.min_fee), 2)
            if o.side == "buy":
                if _cents(notional + fee) > sim.cash_cents:
                    st.status, st.reason = "rejected", "insufficient cash (no margin)"
                    continue
                sim.cash_cents -= _cents(notional) + _cents(fee)
                cost_basis[o.symbol] = cost_basis.get(o.symbol, 0.0) + notional
                qty[o.symbol] = qty.get(o.symbol, 0.0) + shares
            else:
                held = qty[o.symbol]
                basis = cost_basis[o.symbol] * shares / held
                sim.realised += notional - basis
                cost_basis[o.symbol] -= basis
                qty[o.symbol] = held - shares
                sim.cash_cents += _cents(notional) - _cents(fee)
                if qty[o.symbol] <= 1e-12:
                    qty.pop(o.symbol)
                    cost_basis.pop(o.symbol)
            sim.fees += fee
            st.filled = shares
            st.status = "filled" if shares >= o.quantity - 1e-12 else "partially filled"
            if st.status != "filled":
                st.reason = "capped at the position"
            sim.fills.append(
                Fill(
                    o.id,
                    day,
                    o.symbol,
                    o.side,
                    shares,
                    price,
                    float(bar["open"]),
                    ref,
                    fee,
                    slip * 10_000,
                    o.decision_price,
                    float((bar["high"] + bar["low"] + bar["close"]) / 3),
                )
            )
        if first is None or day < first:
            continue
        value = 0.0
        for s, n in qty.items():
            close = frames[s]["close"].loc[:ts]
            value += n * float(close.iloc[-1])
        rows.append(
            {
                "date": day,
                "cash": sim.cash_cents / 100,
                "positions": round(value, 2),
                "equity": round(sim.cash_cents / 100 + value, 2),
            }
        )
    sim.statement = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    for s, n in qty.items():
        last = (
            float(frames[s]["close"].iloc[-1])
            if until is None
            else float(frames[s]["close"].loc[: pd.Timestamp(until)].iloc[-1])
        )
        sim.positions[s] = {
            "quantity": n,
            "average_cost": cost_basis[s] / n,
            "last": last,
            "market_value": round(n * last, 2),
            "unrealised": round(n * last - cost_basis[s], 2),
        }
    return sim


def reconcile(starting_cash: float, sim: Simulation) -> dict[str, float]:
    """Equity from cash + marks vs equity from the fills (start + realised + unrealised − fees).
    Both must agree to the cent (acceptance M29 §5). Sums use the unrounded marks: adding
    per-position values already rounded to the cent can leave a one-cent gap."""
    marks = [p["quantity"] * p["last"] for p in sim.positions.values()]
    costs = [p["quantity"] * p["average_cost"] for p in sim.positions.values()]
    unrealised = sum(marks) - sum(costs)
    from_fills = starting_cash + sim.realised + unrealised - sim.fees
    from_book = sim.cash_cents / 100 + sum(marks)
    cash_from_fills = starting_cash + sum(
        (f.notional if f.side == "sell" else -f.notional) - f.fee for f in sim.fills
    )
    return {
        "equity_book": round(from_book, 2),
        "equity_from_fills": round(from_fills, 2),
        "difference": round(from_book - from_fills, 2),
        "cash_book": sim.cash_cents / 100,
        "cash_from_fills": round(cash_from_fills, 2),
        "cash_difference": round(sim.cash_cents / 100 - cash_from_fills, 2),
    }


def execution_report(fills: list[Fill], states: list[OrderState]) -> dict[str, Any]:
    """TR-042: implementation shortfall vs the decision price (delay + execution + fees, bps,
    signed so that > 0 is a cost), slippage vs the cost model, fill rate, fill vs the daily
    typical price (VWAP proxy), cost by weekday."""
    rows = []
    for f in fills:
        sign = 1 if f.side == "buy" else -1
        dec = (
            f.decision_price
            if math.isfinite(f.decision_price) and f.decision_price > 0
            else f.arrival
        )
        delay = sign * (f.arrival - dec) / dec * 10_000
        execution = sign * (f.price - f.arrival) / dec * 10_000
        fees = f.fee / max(f.notional, 1e-9) * 10_000
        realised_slip = sign * (f.price - f.reference) / f.reference * 10_000
        rows.append(
            {
                "order_id": f.order_id,
                "date": f.date.isoformat(),
                "symbol": f.symbol,
                "side": f.side,
                "quantity": f.quantity,
                "decision_price": dec,
                "arrival": f.arrival,
                "fill": f.price,
                "delay_bps": delay,
                "execution_bps": execution,
                "fee_bps": fees,
                "shortfall_bps": delay + execution + fees,
                "slippage_bps": realised_slip,
                "model_bps": f.model_cost_bps,
                "vs_typical_bps": sign * (f.price - f.typical_price) / f.typical_price * 10_000,
                "weekday": f.date.strftime("%a"),
            }
        )
    frame = pd.DataFrame(rows)
    ordered = sum(s.order.quantity for s in states if s.status != "working")
    filled = sum(s.filled for s in states)
    out: dict[str, Any] = {
        "fills": rows,
        "orders_done": sum(1 for s in states if s.status != "working"),
        "orders_working": sum(1 for s in states if s.status == "working"),
        "fill_rate": filled / ordered if ordered > 0 else None,
        "status_counts": pd.Series([s.status for s in states]).value_counts().to_dict()
        if states
        else {},
    }
    if frame.empty:
        return out | {"summary": None, "by_weekday": []}
    weights = frame["quantity"] * frame["fill"]
    w = weights / weights.sum()
    out["summary"] = {
        k: float((frame[k] * w).sum())
        for k in (
            "shortfall_bps",
            "delay_bps",
            "execution_bps",
            "fee_bps",
            "slippage_bps",
            "model_bps",
            "vs_typical_bps",
        )
    }
    out["summary"]["slippage_minus_model_bps"] = (
        out["summary"]["slippage_bps"] - out["summary"]["model_bps"]
    )
    by_day = frame.groupby("weekday").agg(
        fills=("order_id", "count"), shortfall_bps=("shortfall_bps", "mean")
    )
    out["by_weekday"] = [{"weekday": k, **v} for k, v in by_day.to_dict("index").items()]
    return out
