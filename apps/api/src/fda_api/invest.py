"""M22 endpoints: questionnaire, investor profiles (IN-001), CMA inputs ingestion, IPS (IN-004)."""

from typing import Any, Literal

import anyio
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import Engine, insert, select

from fda_api.data import request_ingestion, request_series_ingestion
from fda_api.reports import build_report
from fda_api.resolve import load_profile
from fda_core.analyses.invest import (
    CLASSES,
    CMA_SERIES,
    MODEL_LABEL,
    QUESTIONS,
    SCORING_VERSION,
    score_profile,
)
from fda_core.registry import InsufficientDataError
from fda_data.lake import PriceLake
from fda_data.schema import investor_profiles

router = APIRouter(prefix="/v1/invest", tags=["invest"])


def _engine(request: Request) -> Engine:
    engine: Engine = request.app.state.engine
    return engine


@router.get("/questionnaire")
def get_questionnaire() -> dict[str, Any]:
    return {
        "scoring_version": SCORING_VERSION,
        "label": MODEL_LABEL,
        "questions": [
            {
                "key": k,
                "dimension": "capacity" if d == "C" else "willingness",
                "text": text,
                "options": list(opts),
            }
            for k, (d, text, opts) in QUESTIONS.items()
        ],
    }


class ProfileBody(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    answers: dict[str, Any]


@router.post("/profiles", status_code=201)
def post_profile(request: Request, body: ProfileBody) -> dict[str, Any]:
    """Validates and scores the answers, then stores them (each save is a new version)."""
    try:
        scored = score_profile(body.answers)
    except InsufficientDataError as exc:
        raise HTTPException(422, str(exc)) from exc
    with _engine(request).begin() as conn:
        pid = conn.execute(
            insert(investor_profiles).values(
                name=body.name, answers=body.answers, scoring_version=SCORING_VERSION
            )
        ).inserted_primary_key[0]  # type: ignore[index]
    return {"id": int(pid), "risk_score": scored["risk_score"]}


@router.get("/profiles")
def get_profiles(request: Request) -> list[dict[str, Any]]:
    with _engine(request).connect() as conn:
        rows = (
            conn.execute(select(investor_profiles).order_by(investor_profiles.c.id.desc()))
            .mappings()
            .all()
        )
    latest: dict[str, dict[str, Any]] = {}
    for r in rows:  # newest version per name
        latest.setdefault(r["name"], dict(r))
    return [r | {"risk_score": score_profile(r["answers"])["risk_score"]} for r in latest.values()]


@router.get("/profiles/{profile_id}")
def get_profile(request: Request, profile_id: int) -> dict[str, Any]:
    row = load_profile(_engine(request), profile_id)
    if row is None:
        raise HTTPException(404, f"Unknown profile {profile_id}")
    return row


@router.post("/ingest", status_code=202)
async def ingest_cma(request: Request, background: BackgroundTasks) -> dict[str, int]:
    """Queues the asset-class proxies and the rate/spread/breakeven series used by IN-002."""
    proxies = [p for _, p, _ in CLASSES.values() if p]
    for symbol in proxies:
        await request_ingestion(request, background, symbol)
    for code in CMA_SERIES:
        await request_series_ingestion(request, background, code)
    return {"prices": len(proxies), "series": len(CMA_SERIES)}


@router.get("/ips/{profile_id}", response_class=HTMLResponse)
async def get_ips(
    request: Request,
    profile_id: int,
    currency: Literal["EUR", "USD"] = "EUR",
    compare: int | None = None,
) -> HTMLResponse:
    """IN-004: one-page IPS (policy, profile, allocation, assumptions with dates and versions)."""
    engine: Engine = request.app.state.engine
    lake: PriceLake = request.app.state.lake
    if load_profile(engine, profile_id) is None:
        raise HTTPException(404, f"Unknown profile {profile_id}")
    overrides = {
        "IN-003": {"currency": currency} | ({"compare": compare} if compare is not None else {})
    }
    page = await anyio.to_thread.run_sync(
        lambda: build_report(engine, lake, "ips", str(profile_id), {"range": "MAX"}, overrides)
    )
    return HTMLResponse(page)
