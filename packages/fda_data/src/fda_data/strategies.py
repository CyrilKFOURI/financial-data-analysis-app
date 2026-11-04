"""Strategies (M25): hypothesis-first research objects and the append-only trial ledger."""

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Engine, delete, func, insert, select, update
from sqlalchemy.exc import IntegrityError

from fda_core.backtest import DEFAULT_GRIDS
from fda_data.schema import strategies, strategy_trials

TEMPLATES = DEFAULT_GRIDS


class StrategyError(ValueError):
    pass


def validate_spec(spec: dict[str, Any]) -> dict[str, Any]:
    template = spec.get("template")
    if template not in TEMPLATES:
        raise StrategyError(f"Unknown template {template!r} (options: {', '.join(TEMPLATES)})")
    if not str(spec.get("symbol") or "").strip():
        raise StrategyError("A strategy needs a symbol")
    grid = spec.get("grid") or TEMPLATES[template]
    params = spec.get("params") or {k: v[len(v) // 2] for k, v in grid.items()}
    costs = spec.get("costs") or {"fee_bps": 5.0, "slippage_bps": 5.0}
    if float(costs.get("fee_bps", 0)) + float(costs.get("slippage_bps", 0)) <= 0:
        raise StrategyError("Costs are always on: fees + slippage must be > 0 bp")
    overlays = [
        o
        for o in spec.get("overlays") or []
        if o in ("trend_filter", "vol_target", "drawdown_control")
    ]
    return {
        "overlays": overlays,
        "symbol": str(spec["symbol"]).upper(),
        "template": template,
        "params": params,
        "grid": grid,
        "short": bool(spec.get("short", False)),
        "costs": costs,
        "warmup": int(spec.get("warmup", 220)),
        "is_fraction": float(spec.get("is_fraction", 0.6)),
    }


def create_strategy(engine: Engine, name: str, hypothesis: str, spec: dict[str, Any]) -> int:
    if len(hypothesis.strip()) < 20:
        raise StrategyError(
            "Write the economic hypothesis first (≥ 20 characters): storytelling defence"
        )
    clean = validate_spec(spec)
    with engine.begin() as conn:
        if (
            conn.execute(select(strategies.c.id).where(strategies.c.name == name)).scalar()
            is not None
        ):
            raise StrategyError(f"A strategy named {name!r} exists")
        sid = conn.execute(
            insert(strategies).values(name=name, hypothesis=hypothesis.strip(), spec=clean)
        ).inserted_primary_key[0]  # type: ignore[index]
    return int(sid)


def update_spec(engine: Engine, strategy_id: int, spec: dict[str, Any]) -> None:
    clean = validate_spec(spec)
    with engine.begin() as conn:
        conn.execute(
            update(strategies)
            .where(strategies.c.id == strategy_id)
            .values(spec=clean, updated_at=datetime.now(UTC))
        )


def load_strategy(engine: Engine, strategy_id: int) -> dict[str, Any] | None:
    with engine.connect() as conn:
        row = (
            conn.execute(select(strategies).where(strategies.c.id == strategy_id))
            .mappings()
            .one_or_none()
        )
        if row is None:
            return None
        total = conn.execute(
            select(func.coalesce(func.sum(strategy_trials.c.configurations), 0)).where(
                strategy_trials.c.strategy_id == strategy_id
            )
        ).scalar()
    return dict(row) | {"trials": int(total or 0)}


def list_strategies(engine: Engine) -> list[dict[str, Any]]:
    with engine.connect() as conn:
        rows = conn.execute(select(strategies).order_by(strategies.c.name)).mappings().all()
    return [dict(r) for r in rows]


def delete_strategy(engine: Engine, strategy_id: int) -> None:
    with engine.begin() as conn:
        conn.execute(delete(strategy_trials).where(strategy_trials.c.strategy_id == strategy_id))
        conn.execute(delete(strategies).where(strategies.c.id == strategy_id))


def record_trials(
    engine: Engine, strategy_id: int, config: dict[str, Any], configurations: int
) -> int:
    """Counts a distinct configuration once (idempotent, also under concurrent panels: a
    duplicate insert loses the race and is ignored); returns the strategy's total trials."""
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()
    try:
        with engine.begin() as conn:
            conn.execute(
                insert(strategy_trials).values(
                    strategy_id=strategy_id, config_hash=digest, configurations=configurations
                )
            )
    except IntegrityError:
        pass  # already counted
    with engine.connect() as conn:
        total = conn.execute(
            select(func.coalesce(func.sum(strategy_trials.c.configurations), 0)).where(
                strategy_trials.c.strategy_id == strategy_id
            )
        ).scalar()
    return int(total or 0)
