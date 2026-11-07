"""M29 endpoints: paper accounts and their orders (simulated; nothing reaches a broker).

Orders are decided at the close of the last stored session of their symbol (decision price =
that close) and can only fill from the next stored session. Strategy-driven orders turn the
latest signal of a saved strategy (M25) into a market order towards a target weight.
"""

import math
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import Engine

from fda_api.resolve import _frame, _instrument
from fda_core import backtest as bt
from fda_core.paper import simulate
from fda_core.registry import InsufficientDataError
from fda_data.lake import PriceLake
from fda_data.paper import (
    PaperError,
    cancel_order,
    cost_model,
    create_account,
    delete_account,
    list_accounts,
    load_account,
    place_order,
    to_orders,
    validate_order,
)
from fda_data.strategies import load_strategy

router = APIRouter(prefix="/v1/paper", tags=["paper"])


def _engine(request: Request) -> Engine:
    engine: Engine = request.app.state.engine
    return engine


def _lake(request: Request) -> PriceLake:
    lake: PriceLake = request.app.state.lake
    return lake


def _account(request: Request, account_id: int) -> dict[str, Any]:
    account = load_account(_engine(request), account_id)
    if account is None:
        raise HTTPException(404, f"Unknown paper account {account_id}")
    return account


def _last_bar(request: Request, symbol: str) -> tuple[Any, float]:
    try:
        row = _instrument(_engine(request), symbol)
    except InsufficientDataError as exc:
        raise HTTPException(422, str(exc)) from exc
    frame = _frame(_lake(request), row)
    if frame.empty:
        raise HTTPException(422, f"{symbol} has no stored prices: ingest it first")
    return frame.index[-1].date(), float(frame["close"].iloc[-1])


class AccountBody(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    currency: str = Field("USD", min_length=3, max_length=3)
    starting_cash: float = Field(100_000, gt=0, le=1e12)
    costs: dict[str, float] | None = None


class OrderBody(BaseModel):
    symbol: str = Field(min_length=1, max_length=32)
    side: str
    quantity: float = Field(gt=0, le=1e9)
    order_type: str = "market"
    limit_price: float | None = Field(None, gt=0)
    stop_price: float | None = Field(None, gt=0)
    trail_pct: float | None = Field(None, gt=0, lt=1)
    tif: str = "gtc"
    note: str | None = Field(None, max_length=500)


class StrategyOrderBody(BaseModel):
    strategy_id: int
    weight: float = Field(1.0, gt=0, le=1, description="Fraction of equity when the signal is long")


@router.get("")
def get_accounts(request: Request) -> list[dict[str, Any]]:
    return list_accounts(_engine(request))


@router.post("", status_code=201)
def post_account(request: Request, body: AccountBody) -> dict[str, int]:
    try:
        aid = create_account(
            _engine(request), body.name, body.currency, body.starting_cash, body.costs
        )
    except PaperError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"id": aid}


@router.get("/{account_id}")
def get_account(request: Request, account_id: int) -> dict[str, Any]:
    return _account(request, account_id)


@router.delete("/{account_id}", status_code=204)
def remove_account(request: Request, account_id: int) -> None:
    delete_account(_engine(request), account_id)


@router.post("/{account_id}/orders", status_code=201)
def post_order(request: Request, account_id: int, body: OrderBody) -> dict[str, Any]:
    _account(request, account_id)
    try:
        order = validate_order(body.model_dump())
    except PaperError as exc:
        raise HTTPException(422, str(exc)) from exc
    decided, price = _last_bar(request, order["symbol"])
    oid = place_order(_engine(request), account_id, order, decided, price)
    return {
        "id": oid,
        "decision_date": decided.isoformat(),
        "decision_price": price,
        "fills_from": "the next stored session after the decision date",
    }


@router.post("/{account_id}/orders/{order_id}/cancel")
def post_cancel(request: Request, account_id: int, order_id: int) -> dict[str, Any]:
    account = _account(request, account_id)
    row = next((o for o in account["orders"] if o["id"] == order_id), None)
    if row is None:
        raise HTTPException(404, f"Unknown order {order_id}")
    decided, _ = _last_bar(request, row["symbol"])
    if not cancel_order(_engine(request), account_id, order_id, decided):
        raise HTTPException(409, "Order already cancelled")
    return {"id": order_id, "cancelled_on": decided.isoformat()}


@router.post("/{account_id}/orders/from-strategy", status_code=201)
def post_strategy_order(
    request: Request, account_id: int, body: StrategyOrderBody
) -> dict[str, Any]:
    """Latest signal of the strategy (long-only on paper) → market order to the target weight of
    the current simulated equity; no order when already at target."""
    account = _account(request, account_id)
    strat = load_strategy(_engine(request), body.strategy_id)
    if strat is None:
        raise HTTPException(404, f"Unknown strategy {body.strategy_id}")
    spec = strat["spec"]
    symbol = spec["symbol"]
    symbols = {o["symbol"] for o in account["orders"]} | {symbol}
    bars = {}
    for s in symbols:
        try:
            bars[s] = _frame(_lake(request), _instrument(_engine(request), s))
        except InsufficientDataError as exc:
            raise HTTPException(422, str(exc)) from exc
    sim = simulate(
        float(account["starting_cash"]),
        to_orders(account["orders"]),
        bars,
        cost_model(account["costs"]),
    )
    close = bars[symbol]["adj_close"].astype(float)
    target = float(
        bt.signal(spec["template"], close, spec["params"], False, spec.get("graph")).iloc[-1]
    )
    target = min(max(target, 0.0), 1.0)
    last = float(bars[symbol]["close"].iloc[-1])
    equity = sim.cash_cents / 100 + sum(p["market_value"] for p in sim.positions.values())
    pending = sum(
        (1 if s.order.side == "buy" else -1) * s.order.quantity
        for s in sim.orders
        if s.status == "working" and s.order.symbol == symbol
    )
    held = sim.positions.get(symbol, {}).get("quantity", 0.0) + pending
    wanted = math.floor(target * body.weight * equity / last)
    delta = wanted - held
    if abs(delta) < 1:
        return {
            "id": None,
            "signal": target,
            "target_shares": wanted,
            "held": held,
            "reason": "already at target (incl. working orders)",
        }
    order = validate_order(
        {
            "symbol": symbol,
            "side": "buy" if delta > 0 else "sell",
            "quantity": abs(delta),
            "order_type": "market",
            "tif": "day",
            "note": f"strategy {strat['name']}: signal {target:+.0f}",
        }
    )
    decided = bars[symbol].index[-1].date()
    oid = place_order(_engine(request), account_id, order, decided, last, f"strategy:{strat['id']}")
    return {"id": oid, "signal": target, "target_shares": wanted, "held": held, "order": order}
