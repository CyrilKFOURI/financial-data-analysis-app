"""Paper accounts (M29, TR-040): accounts and orders. Orders are immutable once placed except
for cancellation; fills are derived by `fda_core.paper.simulate` from the stored bars."""

from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import Engine, delete, insert, select, update

from fda_core.paper import CostModel, Order
from fda_data.schema import paper_accounts, paper_orders

ORDER_TYPES = ("market", "limit", "stop", "stop_limit", "trailing_stop")
DEFAULT_COSTS = {"fee_bps": 1.0, "min_fee": 1.0, "half_spread_bps": 2.0, "impact_eta": 0.7}


class PaperError(ValueError):
    pass


def clean_costs(costs: dict[str, Any] | None) -> dict[str, float]:
    merged = DEFAULT_COSTS | {k: float(v) for k, v in (costs or {}).items() if k in DEFAULT_COSTS}
    if any(v < 0 for v in merged.values()):
        raise PaperError("Costs cannot be negative")
    if merged["fee_bps"] + merged["half_spread_bps"] + merged["min_fee"] <= 0:
        raise PaperError("Costs are always on: fees or spread must be > 0")
    return merged


def create_account(
    engine: Engine,
    name: str,
    currency: str,
    starting_cash: float,
    costs: dict[str, Any] | None = None,
) -> int:
    if starting_cash <= 0:
        raise PaperError("Starting cash must be > 0")
    with engine.begin() as conn:
        exists = conn.execute(select(paper_accounts.c.id).where(paper_accounts.c.name == name))
        if exists.scalar() is not None:
            raise PaperError(f"A paper account named {name!r} exists")
        aid = conn.execute(
            insert(paper_accounts).values(
                name=name,
                currency=currency.upper(),
                starting_cash=starting_cash,
                costs=clean_costs(costs),
            )
        ).inserted_primary_key[0]  # type: ignore[index]
    return int(aid)


def list_accounts(engine: Engine) -> list[dict[str, Any]]:
    with engine.connect() as conn:
        rows = conn.execute(select(paper_accounts).order_by(paper_accounts.c.name)).mappings()
        return [dict(r) for r in rows]


def load_account(engine: Engine, account_id: int) -> dict[str, Any] | None:
    with engine.connect() as conn:
        row = (
            conn.execute(select(paper_accounts).where(paper_accounts.c.id == account_id))
            .mappings()
            .one_or_none()
        )
        if row is None:
            return None
        orders = (
            conn.execute(
                select(paper_orders)
                .where(paper_orders.c.account_id == account_id)
                .order_by(paper_orders.c.id)
            )
            .mappings()
            .all()
        )
    return dict(row) | {"orders": [dict(o) for o in orders]}


def delete_account(engine: Engine, account_id: int) -> None:
    with engine.begin() as conn:
        conn.execute(delete(paper_orders).where(paper_orders.c.account_id == account_id))
        conn.execute(delete(paper_accounts).where(paper_accounts.c.id == account_id))


def validate_order(body: dict[str, Any]) -> dict[str, Any]:
    side = body.get("side")
    if side not in ("buy", "sell"):
        raise PaperError("side must be buy or sell")
    kind = body.get("order_type", "market")
    if kind not in ORDER_TYPES:
        raise PaperError(f"order_type must be one of {', '.join(ORDER_TYPES)}")
    quantity = float(body.get("quantity") or 0)
    if quantity <= 0:
        raise PaperError("quantity must be > 0")
    limit, stop, trail = body.get("limit_price"), body.get("stop_price"), body.get("trail_pct")
    if kind in ("limit", "stop_limit") and not (limit and float(limit) > 0):
        raise PaperError(f"{kind} orders need a limit price > 0")
    if kind in ("stop", "stop_limit") and not (stop and float(stop) > 0):
        raise PaperError(f"{kind} orders need a stop price > 0")
    if kind == "trailing_stop" and not (trail and 0 < float(trail) < 1):
        raise PaperError("trailing stops need trail_pct in (0, 1), e.g. 0.05")
    tif = body.get("tif", "gtc")
    if tif not in ("day", "gtc"):
        raise PaperError("tif must be day or gtc")
    return {
        "symbol": str(body.get("symbol") or "").strip().upper(),
        "side": side,
        "quantity": quantity,
        "order_type": kind,
        "limit_price": float(limit) if limit else None,
        "stop_price": float(stop) if stop else None,
        "trail_pct": float(trail) if trail else None,
        "tif": tif,
        "note": (body.get("note") or None),
    }


def place_order(
    engine: Engine,
    account_id: int,
    order: dict[str, Any],
    decision_date: date,
    decision_price: float,
    source: str = "manual",
) -> int:
    """Records an order decided at the close of `decision_date` (the last stored session): it can
    only fill from the next stored session."""
    with engine.begin() as conn:
        oid = conn.execute(
            insert(paper_orders).values(
                account_id=account_id,
                decision_date=decision_date,
                decision_price=decision_price,
                source=source,
                **order,
            )
        ).inserted_primary_key[0]  # type: ignore[index]
        conn.execute(
            update(paper_accounts)
            .where(paper_accounts.c.id == account_id)
            .values(updated_at=datetime.now(UTC))
        )
    return int(oid)


def cancel_order(engine: Engine, account_id: int, order_id: int, on: date) -> bool:
    """Cancels a working order as of the session `on` (it can no longer fill after it)."""
    with engine.begin() as conn:
        result = conn.execute(
            update(paper_orders)
            .where(
                (paper_orders.c.id == order_id)
                & (paper_orders.c.account_id == account_id)
                & paper_orders.c.cancelled_on.is_(None)
            )
            .values(cancelled_on=on)
        )
    return bool(result.rowcount)


def to_orders(rows: list[dict[str, Any]]) -> list[Order]:
    return [
        Order(
            int(r["id"]),
            str(r["symbol"]),
            r["side"],
            float(r["quantity"]),
            r["order_type"],
            r["limit_price"],
            r["stop_price"],
            r["trail_pct"],
            r["tif"],
            r["decision_date"],
            float(r["decision_price"]),
            r["cancelled_on"],
            str(r["source"]),
        )
        for r in rows
    ]


def cost_model(costs: dict[str, Any]) -> CostModel:
    return CostModel(**clean_costs(costs))
