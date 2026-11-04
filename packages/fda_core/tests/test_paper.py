"""Paper trading simulator: fill rules per order type, no look-ahead, cash/position limits, and
M29 acceptance — paper P&L reconciles to the fills to the cent."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from fda_core.paper import CostModel, Order, execution_report, reconcile, simulate

DAYS = pd.bdate_range("2024-01-01", periods=6)
FREE = CostModel(fee_bps=0, min_fee=0, half_spread_bps=0, impact_eta=0)


def bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    index = DAYS[: len(rows)]
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)
    return frame.assign(volume=1e6)


BARS = bars(
    [
        (100, 101, 99, 100),
        (101, 103, 100, 102),
        (102, 102, 95, 96),
        (96, 99, 94, 98),
        (98, 105, 97, 104),
        (104, 106, 103, 105),
    ]
)
D0 = DAYS[0].date()


def run(*orders: Order, cash: float = 10_000, costs: CostModel = FREE):  # type: ignore[no-untyped-def]
    return simulate(cash, list(orders), {"X": BARS}, costs)


def test_market_fills_at_the_next_open_not_the_decision_bar() -> None:
    sim = run(Order(1, "X", "buy", 10, decision_date=D0, decision_price=100))
    assert len(sim.fills) == 1
    assert sim.fills[0].date == DAYS[1].date()
    assert sim.fills[0].price == 101
    assert sim.statement["equity"].iloc[-1] == pytest.approx(10_000 + 10 * (105 - 101))


def test_limit_stop_trailing_and_day_orders() -> None:
    limit = Order(1, "X", "buy", 1, "limit", limit_price=97, decision_date=D0)
    day_limit = Order(2, "X", "buy", 1, "limit", limit_price=90, tif="day", decision_date=D0)
    stop_buy = Order(3, "X", "buy", 1, "stop", stop_price=104, decision_date=D0)
    sim = run(limit, day_limit, stop_buy)
    by_id = {f.order_id: f for f in sim.fills}
    assert by_id[1].date == DAYS[2].date()  # low 95 reached the limit
    assert by_id[1].price == 97
    assert by_id[3].date == DAYS[4].date()  # high 105 ≥ stop, open 98
    assert by_id[3].price == 104
    states = {s.order.id: s.status for s in sim.orders}
    assert states[2] == "expired"
    # gap through a sell stop fills at the open, not the stop
    hold = Order(4, "X", "buy", 5, decision_date=D0)
    stop_sell = Order(5, "X", "sell", 5, "stop", stop_price=97, decision_date=DAYS[1].date())
    sim = run(hold, stop_sell)
    sell = next(f for f in sim.fills if f.side == "sell")
    assert sell.date == DAYS[2].date()
    assert sell.price == 97  # open 102, low 95 → stop level
    trail = Order(6, "X", "sell", 5, "trailing_stop", trail_pct=0.05, decision_date=DAYS[1].date())
    sim = run(hold, trail)
    sell = next(f for f in sim.fills if f.side == "sell")
    assert sell.date == DAYS[2].date()
    assert sell.price == pytest.approx(102 * 0.95)  # watermark starts at the eligible open
    assert sim.positions == {}


def test_stop_limit_rests_as_a_limit_and_cancellation() -> None:
    hold = Order(1, "X", "buy", 5, decision_date=D0)
    stop_limit = Order(
        2, "X", "sell", 5, "stop_limit", limit_price=98, stop_price=97, decision_date=DAYS[1].date()
    )
    sim = run(hold, stop_limit)
    sell = next(f for f in sim.fills if f.side == "sell")
    assert sell.date == DAYS[3].date()  # triggered day 2 at 97 < limit 98: rests
    assert sell.price == 98
    cancelled = Order(
        3, "X", "buy", 1, "limit", limit_price=50, decision_date=D0, cancelled_on=DAYS[2].date()
    )
    sim = run(cancelled)
    assert sim.orders[0].status == "cancelled"


def test_cash_and_position_limits() -> None:
    sim = run(Order(1, "X", "buy", 1_000, decision_date=D0), cash=1_000)
    assert sim.orders[0].status == "rejected"
    sim = run(Order(1, "X", "sell", 1, decision_date=D0))
    assert sim.orders[0].status == "rejected"
    sim = run(
        Order(1, "X", "buy", 2, decision_date=D0),
        Order(2, "X", "sell", 5, decision_date=DAYS[2].date()),
    )
    assert sim.orders[1].status == "partially filled"
    assert sim.positions == {}


def test_pnl_reconciles_to_the_fills_to_the_cent() -> None:
    rng = np.random.default_rng(7)
    days = pd.bdate_range("2023-01-02", periods=260)
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.015, len(days))))
    frame = pd.DataFrame(
        {"open": close * (1 + rng.normal(0, 0.003, len(days))), "close": close}, index=days
    )
    frame["high"] = frame[["open", "close"]].max(axis=1) * 1.01
    frame["low"] = frame[["open", "close"]].min(axis=1) * 0.99
    frame["volume"] = 2e5
    costs = CostModel(fee_bps=1.5, min_fee=1.0, half_spread_bps=3.0, impact_eta=0.7)
    orders = []
    for i, d in enumerate(days[20:240:7]):
        side = "buy" if i % 3 != 2 else "sell"
        kind = ["market", "limit", "stop", "trailing_stop"][i % 4]
        px = float(frame.loc[d, "close"])
        orders.append(
            Order(
                i,
                "Y",
                side,
                3 + i % 5,
                kind,  # type: ignore[arg-type]
                limit_price=px * (0.99 if side == "buy" else 1.01),
                stop_price=px * (1.01 if side == "buy" else 0.99),
                trail_pct=0.03,
                decision_date=d.date(),
                decision_price=px,
            )
        )
    sim = simulate(25_000, orders, {"Y": frame}, costs)
    assert len(sim.fills) > 15
    rec = reconcile(25_000, sim)
    assert rec["difference"] == 0.0
    assert rec["cash_difference"] == 0.0
    assert sim.statement["equity"].iloc[-1] == rec["equity_book"]
    report = execution_report(sim.fills, sim.orders)
    assert report["summary"]["fee_bps"] > 0
    assert report["summary"]["slippage_minus_model_bps"] == pytest.approx(0, abs=0.05)  # simulated
    assert 0 < report["fill_rate"] <= 1


def test_execution_report_signs() -> None:
    sim = run(
        Order(1, "X", "buy", 10, decision_date=D0, decision_price=100),
        costs=CostModel(fee_bps=10, min_fee=0, half_spread_bps=5, impact_eta=0),
    )
    report = execution_report(sim.fills, sim.orders)
    row = report["fills"][0]
    assert row["delay_bps"] == pytest.approx(100)  # open 101 vs decision 100
    assert row["execution_bps"] == pytest.approx(5 * 1.01, rel=1e-3)
    assert row["fee_bps"] == pytest.approx(10, rel=1e-2)
    assert report["fill_rate"] == 1.0
    assert date.fromisoformat(row["date"]) == DAYS[1].date()
