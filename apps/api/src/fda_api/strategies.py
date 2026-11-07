"""M25 endpoints: strategies (hypothesis-first) and their trial ledger."""

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import Engine

from fda_core import blocks
from fda_data.strategies import (
    TEMPLATES,
    StrategyError,
    create_strategy,
    delete_strategy,
    list_strategies,
    load_strategy,
    update_spec,
)

router = APIRouter(prefix="/v1/strategies", tags=["strategies"])


def _engine(request: Request) -> Engine:
    engine: Engine = request.app.state.engine
    return engine


class StrategyBody(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    hypothesis: str = Field(max_length=2000)
    spec: dict[str, Any]


@router.get("/templates")
def get_templates() -> dict[str, dict[str, list[float]]]:
    return TEMPLATES


@router.get("/blocks")
def get_blocks() -> list[dict[str, Any]]:
    """The strategy builder's block catalogue (feature 321): ports, types, defaults."""
    return blocks.catalogue()


@router.get("/blocks/templates/{name}")
def get_graph_template(name: str, symbol: str = "SPY") -> dict[str, Any]:
    """A starting graph: `ema_cross`, `rsi_bounce` (the Edgebot clone's) or `blank`."""
    try:
        return blocks.template(name, symbol.upper())
    except blocks.GraphError as exc:
        raise HTTPException(404, str(exc)) from exc


class GraphBody(BaseModel):
    graph: dict[str, Any]


@router.post("/blocks/validate")
def post_validate(body: GraphBody) -> dict[str, list[str]]:
    """Every problem of a graph in words; empty when it can run."""
    return {"problems": blocks.validate(body.graph)}


@router.get("")
def get_strategies(request: Request) -> list[dict[str, Any]]:
    return list_strategies(_engine(request))


@router.post("", status_code=201)
def post_strategy(request: Request, body: StrategyBody) -> dict[str, int]:
    try:
        return {"id": create_strategy(_engine(request), body.name, body.hypothesis, body.spec)}
    except StrategyError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/{strategy_id}")
def get_strategy(request: Request, strategy_id: int) -> dict[str, Any]:
    row = load_strategy(_engine(request), strategy_id)
    if row is None:
        raise HTTPException(404, f"Unknown strategy {strategy_id}")
    return row


@router.put("/{strategy_id}")
def put_strategy(request: Request, strategy_id: int, spec: dict[str, Any]) -> dict[str, Any]:
    """Changes the spec; the trial ledger keeps counting (a new grid adds its configurations)."""
    if load_strategy(_engine(request), strategy_id) is None:
        raise HTTPException(404, f"Unknown strategy {strategy_id}")
    try:
        update_spec(_engine(request), strategy_id, spec)
    except StrategyError as exc:
        raise HTTPException(422, str(exc)) from exc
    return get_strategy(request, strategy_id)


@router.delete("/{strategy_id}", status_code=204)
def remove_strategy(request: Request, strategy_id: int) -> None:
    delete_strategy(_engine(request), strategy_id)
