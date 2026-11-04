"""M31 §3 schedules: cron-like recurring jobs run by the API process.

Kinds: `data_refresh` (re-ingest symbols), `alerts` (evaluate every alert rule), `snapshot`
(capture analyses of a subject), `report` (render a report template to `data/reports`, and
e-mail it when SMTP is configured), `backup` (database + lake archive in `data/backups`).
Every run is recorded in `schedule_runs` with its status and result. The scheduler ticks once
a minute when `FDA_SCHEDULER=1` (laptop / home-server profile); `POST /v1/schedules/run-due`
runs whatever is due now (tests, external cron).
"""

import asyncio
import contextlib
import json
import logging
import os
import re
import smtplib
from datetime import UTC, datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import anyio
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import Engine, delete, insert, select, update
from sqlalchemy.exc import IntegrityError

from fda_api.data import SYMBOL_PATTERN
from fda_api.llm import LLMError
from fda_api.narrate import narrate_brief
from fda_api.reports import TEMPLATES, build_report
from fda_api.workbench import SnapshotIn, evaluate_alerts, take_snapshot
from fda_api.workflows import WORKFLOWS, render_html, run_workflow
from fda_core.cron import CronError, parse
from fda_data import backup
from fda_data.actor import Actor, acting_as, current_actor
from fda_data.ingest import ingest_daily_prices
from fda_data.lake import PriceLake
from fda_data.schema import schedule_runs, schedules, users

router = APIRouter(prefix="/v1/schedules", tags=["schedules"])
log = logging.getLogger(__name__)

Kind = Literal["data_refresh", "alerts", "snapshot", "report", "backup", "workflow"]
SYMBOL_MAX = 50


class ScheduleIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    kind: Kind
    cron: str = Field(
        max_length=64, description="minute hour day month weekday, e.g. 30 18 * * 1-5"
    )
    timezone: str = Field("UTC", max_length=64)
    spec: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


def _zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise HTTPException(422, f"Unknown time zone {name!r}") from exc


def validate(body: ScheduleIn) -> dict[str, Any]:
    try:
        parse(body.cron)
    except CronError as exc:
        raise HTTPException(422, str(exc)) from exc
    _zone(body.timezone)
    spec = body.spec
    if body.kind == "data_refresh":
        symbols = [str(s).strip().upper() for s in spec.get("symbols", []) if str(s).strip()]
        invalid = [s for s in symbols if not re.match(SYMBOL_PATTERN, s)]
        if invalid:
            raise HTTPException(422, f"Invalid symbols: {', '.join(invalid[:5])}")
        if not symbols or len(symbols) > SYMBOL_MAX:
            raise HTTPException(422, f"data_refresh needs 1–{SYMBOL_MAX} symbols")
        spec = {"symbols": symbols}
    elif body.kind == "snapshot":
        try:
            spec = SnapshotIn.model_validate(spec).model_dump()
        except ValueError as exc:
            raise HTTPException(422, f"snapshot spec: {exc}") from exc
    elif body.kind == "report":
        if spec.get("template") not in TEMPLATES or not str(spec.get("subject") or "").strip():
            raise HTTPException(
                422, f"report needs a template ({', '.join(TEMPLATES)}) and a subject"
            )
        spec = {k: spec[k] for k in ("template", "subject", "range", "email") if spec.get(k)}
    elif body.kind == "workflow":
        if spec.get("workflow") not in WORKFLOWS or not str(spec.get("subject") or "").strip():
            raise HTTPException(
                422, f"workflow needs a workflow ({', '.join(WORKFLOWS)}) and a subject"
            )
        keys = ("workflow", "subject", "range", "email", "narrate", "provider", "model")
        spec = {k: spec[k] for k in keys if spec.get(k)}
    elif body.kind == "backup":
        actor = current_actor()
        if actor is not None and actor.role != "admin":
            raise HTTPException(403, "Only admins schedule backups")
        keep = spec.get("keep", 14)
        if not isinstance(keep, int) or not 1 <= keep <= 365:
            raise HTTPException(422, "backup keep must be a whole number of archives, 1–365")
        spec = {"keep": keep}
    else:
        spec = {}
    return spec


def next_run(cron: str, zone: str, after: datetime) -> datetime:
    local = after.astimezone(_zone(zone))
    return parse(cron).next_after(local).astimezone(UTC)


def _aware(t: datetime | None) -> datetime | None:
    return None if t is None else t if t.tzinfo else t.replace(tzinfo=UTC)


# ── execution ────────────────────────────────────────────────────────────────


def reports_dir(lake: PriceLake) -> Path:
    path = lake.root.parent / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def send_email(to: str, subject: str, html: str) -> str:
    """SMTP delivery (STARTTLS) when SMTP_HOST is set; returns a status line."""
    host = os.getenv("SMTP_HOST")
    if not host:
        return "e-mail skipped: SMTP_HOST is not configured"
    msg = EmailMessage()
    msg["From"] = os.getenv("SMTP_FROM") or os.getenv("SMTP_USER", "")
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content("This report is attached as HTML.")
    msg.add_alternative(html, subtype="html")
    with smtplib.SMTP(host, int(os.getenv("SMTP_PORT", "587")), timeout=30) as smtp:
        smtp.starttls()
        if os.getenv("SMTP_USER"):
            smtp.login(os.environ["SMTP_USER"], os.getenv("SMTP_PASSWORD", ""))
        smtp.send_message(msg)
    return f"e-mailed to {to}"


def execute(app_state: Any, row: dict[str, Any]) -> dict[str, Any]:
    engine: Engine = app_state.engine
    lake: PriceLake = app_state.lake
    spec = row["spec"]
    kind = row["kind"]
    if kind == "data_refresh":
        out = {}
        for symbol in spec["symbols"]:
            result = ingest_daily_prices(engine, lake, app_state.chain, symbol)
            out[symbol] = {"status": result.status, "rows": result.rows_written}
        return {"symbols": out}
    if kind == "alerts":
        fired = evaluate_alerts(engine, lake)
        return {"evaluated": len(fired), "fired": [f for f in fired if f.get("fired")][:20]}
    if kind == "snapshot":
        return take_snapshot(engine, lake, SnapshotIn.model_validate(spec))
    if kind == "report":
        context = {"range": spec.get("range", "1Y"), "ccy": "listing"}
        html = build_report(engine, lake, spec["template"], spec["subject"], context)
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in row["name"])[:60]
        path = reports_dir(lake) / f"{safe}-{stamp}.html"
        path.write_text(html, "utf-8")
        delivery = (
            send_email(spec["email"], f"{row['name']} — {stamp}", html)
            if spec.get("email")
            else "saved"
        )
        return {"file": str(path), "bytes": len(html), "delivery": delivery}
    if kind == "workflow":
        brief = run_workflow(
            engine, lake, spec["workflow"], str(spec["subject"]), spec.get("range", "1Y")
        )
        if spec.get("narrate"):  # a failed narration never blocks the report itself
            try:
                brief["narrative"] = narrate_brief(
                    engine, brief, spec.get("provider"), spec.get("model")
                )
            except LLMError as exc:
                brief["narrative"] = {"error": str(exc)}
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in row["name"])[:60]
        base = reports_dir(lake) / f"{safe}-{stamp}"
        base.with_suffix(".json").write_text(json.dumps(brief, indent=1, default=str), "utf-8")
        page = render_html(brief)
        base.with_suffix(".html").write_text(page, "utf-8")
        delivery = (
            send_email(spec["email"], f"{brief['title']} — {stamp}", page)
            if spec.get("email")
            else "saved"
        )
        return {
            "file": str(base.with_suffix(".html")),
            "highlights": brief["highlights"][:10],
            "delivery": delivery,
        }
    if kind == "backup":
        # checked at run time too: an owner demoted since creating the schedule loses it
        actor = current_actor()
        if actor is not None and actor.role != "admin":
            raise PermissionError("Backups run for admins only; the owner is no longer an admin")
        url = engine.url.render_as_string(hide_password=False)
        root = lake.root.parent
        return backup.create(root / "backups", int(spec.get("keep", 14)), url, root)
    raise ValueError(f"Unknown kind {kind}")


def run_schedule(app_state: Any, schedule_id: int, trigger: str) -> dict[str, Any]:
    engine: Engine = app_state.engine
    with engine.connect() as conn:
        row = (
            conn.execute(select(schedules).where(schedules.c.id == schedule_id))
            .mappings()
            .one_or_none()
        )
    if row is None:
        raise HTTPException(404, f"Unknown schedule {schedule_id}")
    with acting_as(_owner(engine, row["owner_id"])):
        return _run(app_state, dict(row), trigger)


def _owner(engine: Engine, owner_id: int | None) -> Actor | None:
    """A schedule runs as its owner (their rows, their sharing); system schedules as system."""
    if owner_id is None:
        return None
    with acting_as(None), engine.connect() as conn:
        role = conn.execute(select(users.c.role).where(users.c.id == owner_id)).scalar()
    return Actor(int(owner_id), role or "viewer", "write")


def _run(app_state: Any, row: dict[str, Any], trigger: str) -> dict[str, Any]:
    engine: Engine = app_state.engine
    schedule_id = int(row["id"])
    started = datetime.now(UTC)
    with engine.begin() as conn:
        run_id = conn.execute(
            insert(schedule_runs).values(
                schedule_id=schedule_id, started_at=started, status="running", trigger=trigger
            )
        ).inserted_primary_key[0]  # type: ignore[index]
    try:
        result, status = execute(app_state, row), "succeeded"
    except Exception as exc:
        log.exception("schedule %s failed", schedule_id)
        result, status = {"error": f"{type(exc).__name__}: {exc}"[:500]}, "failed"
    finished = datetime.now(UTC)
    with engine.begin() as conn:
        conn.execute(
            update(schedule_runs)
            .where(schedule_runs.c.id == run_id)
            .values(finished_at=finished, status=status, result=result)
        )
        conn.execute(
            update(schedules)
            .where(schedules.c.id == schedule_id)
            .values(
                last_run_at=started,
                last_status=status,
                next_run_at=next_run(row["cron"], row["timezone"], finished),
                updated_at=finished,
            )
        )
    return {"run_id": int(run_id), "status": status, "result": result}


def run_due(app_state: Any, now: datetime | None = None) -> list[dict[str, Any]]:
    now = now or datetime.now(UTC)
    with app_state.engine.connect() as conn:
        rows = conn.execute(select(schedules).where(schedules.c.enabled.is_(True))).mappings().all()
    due = [r for r in rows if (_aware(r["next_run_at"]) or now) <= now]
    return [{"schedule_id": r["id"], **run_schedule(app_state, int(r["id"]), "cron")} for r in due]


async def scheduler_loop(app_state: Any) -> None:
    """Minute tick for the single-process profiles (FDA_SCHEDULER=1)."""
    while True:
        try:
            await anyio.to_thread.run_sync(run_due, app_state)
        except Exception:
            log.exception("scheduler tick failed")
        await asyncio.sleep(60 - datetime.now(UTC).second)


@contextlib.asynccontextmanager
async def maybe_scheduler(app_state: Any) -> Any:
    task = (
        asyncio.create_task(scheduler_loop(app_state))
        if os.getenv("FDA_SCHEDULER") == "1"
        else None
    )
    try:
        yield
    finally:
        if task is not None:
            task.cancel()


# ── endpoints ────────────────────────────────────────────────────────────────


def _row(r: Any) -> dict[str, Any]:
    out = dict(r)
    for k, v in out.items():
        if isinstance(v, datetime):
            out[k] = _aware(v).isoformat()  # type: ignore[union-attr]
    return out


@router.get("")
def list_schedules(request: Request) -> list[dict[str, Any]]:
    with request.app.state.engine.connect() as conn:
        return [
            _row(r) for r in conn.execute(select(schedules).order_by(schedules.c.name)).mappings()
        ]


@router.post("", status_code=201)
def create_schedule(request: Request, body: ScheduleIn) -> dict[str, Any]:
    spec = validate(body)
    first = next_run(body.cron, body.timezone, datetime.now(UTC))
    try:
        with request.app.state.engine.begin() as conn:
            sid = conn.execute(
                insert(schedules).values(
                    name=body.name,
                    kind=body.kind,
                    cron=body.cron,
                    timezone=body.timezone,
                    spec=spec,
                    enabled=body.enabled,
                    next_run_at=first,
                )
            ).inserted_primary_key[0]
    except IntegrityError as exc:
        raise HTTPException(422, f"A schedule named {body.name!r} exists") from exc
    return {"id": int(sid), "next_run_at": first.isoformat()}


@router.put("/{schedule_id}")
def update_schedule(request: Request, schedule_id: int, body: ScheduleIn) -> dict[str, Any]:
    spec = validate(body)
    first = next_run(body.cron, body.timezone, datetime.now(UTC))
    with request.app.state.engine.begin() as conn:
        done = conn.execute(
            update(schedules)
            .where(schedules.c.id == schedule_id)
            .values(
                name=body.name,
                kind=body.kind,
                cron=body.cron,
                timezone=body.timezone,
                spec=spec,
                enabled=body.enabled,
                next_run_at=first,
                updated_at=datetime.now(UTC),
            )
        ).rowcount
    if not done:
        raise HTTPException(404, f"Unknown schedule {schedule_id}")
    return {"id": schedule_id, "next_run_at": first.isoformat()}


@router.delete("/{schedule_id}", status_code=204)
def delete_schedule(request: Request, schedule_id: int) -> None:
    with request.app.state.engine.begin() as conn:
        conn.execute(delete(schedule_runs).where(schedule_runs.c.schedule_id == schedule_id))
        conn.execute(delete(schedules).where(schedules.c.id == schedule_id))


@router.post("/{schedule_id}/run")
async def run_now(request: Request, schedule_id: int) -> dict[str, Any]:
    return await anyio.to_thread.run_sync(run_schedule, request.app.state, schedule_id, "manual")


@router.post("/run-due")
async def post_run_due(request: Request) -> list[dict[str, Any]]:
    return await anyio.to_thread.run_sync(run_due, request.app.state)


@router.get("/{schedule_id}/runs")
def list_runs(request: Request, schedule_id: int) -> list[dict[str, Any]]:
    query = (
        select(schedule_runs)
        .where(schedule_runs.c.schedule_id == schedule_id)
        .order_by(schedule_runs.c.id.desc())
        .limit(50)
    )
    with request.app.state.engine.connect() as conn:
        return [_row(r) for r in conn.execute(query).mappings()]
