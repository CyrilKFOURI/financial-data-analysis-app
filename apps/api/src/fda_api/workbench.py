"""J11 endpoints: comparisons (M16, CP-001/003), benchmark presets (CP-002), reports (CP-004),
analysis alerts & notifications (CP-005) and snapshots (CP-006)."""

import http.client
import ipaddress
import operator
import socket
import ssl
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import urlparse

import anyio
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import Engine, delete, insert, select, update
from sqlalchemy.exc import IntegrityError

from fda_api.reports import TEMPLATES, build_report
from fda_api.resolve import parse_members
from fda_api.runner import run_analysis, value_at
from fda_core.registry import REGISTRY, InsufficientDataError, UnknownAnalysisError
from fda_data.lake import PriceLake
from fda_data.portfolios import PortfolioError, create_portfolio, load_portfolio, save_version
from fda_data.schema import alert_rules, comparisons, notifications, portfolios, snapshots

router = APIRouter(prefix="/v1", tags=["workbench"])


def _fail(exc: Exception, status: int = 422) -> HTTPException:
    return HTTPException(status_code=404 if "not found" in str(exc) else status, detail=str(exc))


# ── Comparisons ──────────────────────────────────────────────────────────────


class ComparisonIn(BaseModel):
    name: str = Field(max_length=80)
    members: str = Field(max_length=600, description="AAPL, portfolio:3, SPY …")
    note: str | None = Field(None, max_length=500)


def _row(r: Any) -> dict[str, Any]:
    return {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in dict(r).items()}


@router.get("/comparisons")
async def get_comparisons(request: Request) -> list[dict[str, Any]]:
    def load() -> list[dict[str, Any]]:
        with request.app.state.engine.connect() as conn:
            rows = conn.execute(select(comparisons).order_by(comparisons.c.name)).mappings().all()
        return [_row(r) for r in rows]

    return await anyio.to_thread.run_sync(load)


@router.post("/comparisons", status_code=201)
async def post_comparison(body: ComparisonIn, request: Request) -> dict[str, Any]:
    try:
        members = parse_members(body.members)
    except InsufficientDataError as exc:
        raise _fail(exc) from exc

    def save() -> int:
        with request.app.state.engine.begin() as conn:
            return int(
                conn.execute(
                    insert(comparisons).values(
                        name=body.name.strip(), note=body.note, members=members
                    )
                ).inserted_primary_key[0]
            )

    try:
        cid = await anyio.to_thread.run_sync(save)
    except IntegrityError as exc:
        raise HTTPException(
            status_code=422, detail=f"A comparison named “{body.name}” exists"
        ) from exc
    return {"id": cid, "members": members}


@router.put("/comparisons/{cid}")
async def put_comparison(cid: int, body: ComparisonIn, request: Request) -> dict[str, Any]:
    try:
        members = parse_members(body.members)
    except InsufficientDataError as exc:
        raise _fail(exc) from exc
    with request.app.state.engine.begin() as conn:
        done = conn.execute(
            update(comparisons)
            .where(comparisons.c.id == cid)
            .values(name=body.name.strip(), note=body.note, members=members)
        ).rowcount
    if not done:
        raise HTTPException(status_code=404, detail="Comparison not found")
    return {"id": cid, "members": members}


@router.get("/comparisons/{cid}")
async def get_comparison(cid: int, request: Request) -> dict[str, Any]:
    with request.app.state.engine.connect() as conn:
        row = (
            conn.execute(select(comparisons).where(comparisons.c.id == cid))
            .mappings()
            .one_or_none()
        )
    if row is None:
        raise HTTPException(status_code=404, detail="Comparison not found")
    return _row(row)


@router.delete("/comparisons/{cid}", status_code=204)
async def delete_comparison(cid: int, request: Request) -> None:
    with request.app.state.engine.begin() as conn:
        conn.execute(delete(comparisons).where(comparisons.c.id == cid))


# ── Benchmark presets (CP-002) ───────────────────────────────────────────────

PRESETS: dict[str, dict[str, Any]] = {
    "sp500": {"label": "S&P 500", "symbol": "SPY"},
    "nasdaq100": {"label": "Nasdaq-100", "symbol": "QQQ"},
    "msci_world": {"label": "MSCI World", "symbol": "URTH"},
    "msci_acwi": {"label": "MSCI ACWI", "symbol": "ACWI"},
    "stoxx600": {"label": "STOXX 600", "symbol": "EXSA.DE"},
    "cac40": {"label": "CAC 40", "symbol": "^FCHI"},
    "us_agg": {"label": "US Aggregate", "symbol": "AGG"},
    "gold": {"label": "Gold", "symbol": "GLD"},
    "bitcoin": {"label": "Bitcoin", "symbol": "BTC-USD"},
    "6040_usd": {"label": "60/40 USD", "ccy": "USD", "components": {"SPY": 0.6, "AGG": 0.4}},
    "6040_eur": {
        "label": "60/40 EUR",
        "ccy": "EUR",
        "components": {"IWDA.AS": 0.6, "IEAG.AS": 0.4},
    },
    "8020_eur": {
        "label": "80/20 EUR",
        "ccy": "EUR",
        "components": {"IWDA.AS": 0.8, "IEAG.AS": 0.2},
    },
}


@router.get("/benchmarks/presets")
async def get_presets() -> list[dict[str, Any]]:
    return [{"key": k} | v for k, v in PRESETS.items()]


@router.post("/benchmarks/presets/{key}")
async def post_preset(key: str, request: Request) -> dict[str, Any]:
    """Single-instrument presets are instruments; blends become benchmark portfolios (monthly
    rebalance) usable as comparison members and portfolio benchmarks. Idempotent."""
    preset = PRESETS.get(key)
    if preset is None:
        raise HTTPException(status_code=404, detail=f"Unknown preset {key}")
    if "symbol" in preset:
        return {"member": f"asset:{preset['symbol']}", "ingest": [preset["symbol"]]}
    engine: Engine = request.app.state.engine
    name = f"Benchmark – {preset['label']}"

    def ensure() -> int:
        with engine.connect() as conn:
            found = conn.execute(
                select(portfolios.c.id).where(portfolios.c.name == name)
            ).scalar_one_or_none()
        holdings = [{"symbol": s, "weight": w} for s, w in preset["components"].items()]
        if found is not None:
            current = {
                h["symbol"]: h["weight"] for h in load_portfolio(engine, int(found))["holdings"]
            }
            if current != preset["components"]:  # preset definition changed: new version
                save_version(engine, int(found), {"holdings": holdings})
            return int(found)
        return create_portfolio(
            engine,
            {
                "name": name,
                "mode": "weights",
                "base_currency": preset["ccy"],
                "holdings": holdings,
                "is_benchmark": True,
                "rebalance": {"type": "calendar", "freq": "M"},
            },
        )

    try:
        pid = await anyio.to_thread.run_sync(ensure)
    except PortfolioError as exc:
        raise _fail(exc) from exc
    return {"member": f"portfolio:{pid}", "portfolio_id": pid, "ingest": list(preset["components"])}


# ── Reports (CP-004) ─────────────────────────────────────────────────────────


@router.get("/reports/templates")
async def get_templates() -> list[dict[str, Any]]:
    return [
        {
            "key": k,
            "title": v["title"],
            "subject": v["subject"],
            "analyses": [a for a, _ in v["analyses"]],
        }
        for k, v in TEMPLATES.items()
    ]


@router.get("/reports/{template}", response_class=HTMLResponse)
async def get_report(
    template: str, subject: str, request: Request, range: str = "1Y", ccy: str = "listing"
) -> HTMLResponse:
    if template not in TEMPLATES:
        raise HTTPException(status_code=404, detail=f"Unknown template {template}")
    context = {"range": range, "ccy": ccy}
    try:
        page = await anyio.to_thread.run_sync(
            build_report,
            request.app.state.engine,
            request.app.state.lake,
            template,
            subject,
            context,
        )
    except ValueError as exc:  # invalid context (RunContext validation)
        raise _fail(exc) from exc
    return HTMLResponse(page)


# ── Alerts (CP-005) ──────────────────────────────────────────────────────────

OPS: dict[str, Callable[[float, float], bool]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}


class AlertIn(BaseModel):
    name: str = Field(max_length=120)
    subject_type: Literal["asset", "portfolio", "comparison", "strategy"]
    subject_id: str = Field(max_length=64)
    analysis_id: str = Field(max_length=16)
    params: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    path: str = Field(max_length=200, description="e.g. max_drawdown, rows[symbol=AAPL].weight")
    op: Literal[">", ">=", "<", "<="]
    threshold: float
    webhook_url: str | None = Field(
        None, max_length=500, pattern=r"^https://", description="Optional outgoing JSON (TR-012)"
    )


class WebhookPayload(BaseModel):
    """TR-012 outgoing message (15-TV pattern). Informational only: never an order."""

    ticker: str
    action: str
    size: float | None
    price: float | None
    timestamp: str
    rule: str
    value: float | None


def _resolve_public(url: str) -> tuple[str, int, str] | str:
    """(host, port, pinned public IP) or the reason the target is refused: https only, and
    every address the host resolves to must be public (no loopback, private, link-local,
    multicast or reserved ranges)."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return "webhooks must use https:// with a host name"
    port = parsed.port or 443
    try:
        infos = socket.getaddrinfo(parsed.hostname, port, proto=socket.IPPROTO_TCP)
    except OSError:
        return "the webhook host does not resolve"
    addresses = [ipaddress.ip_address(info[4][0]) for info in infos]
    if not addresses or any(not a.is_global or a.is_multicast for a in addresses):
        return "webhook hosts must resolve to public internet addresses"
    return parsed.hostname, port, str(addresses[0])


def unsafe_webhook(url: str) -> str | None:
    """Why a webhook target is refused (None when allowed); checked when the rule is saved."""
    resolved = _resolve_public(url)
    return resolved if isinstance(resolved, str) else None


class _PinnedHTTPS(http.client.HTTPSConnection):
    """Connects to a pre-validated IP while TLS (SNI + certificate) uses the host name: the
    address checked is the address used (no DNS-rebinding window between check and send)."""

    def __init__(self, host: str, port: int, ip: str) -> None:
        super().__init__(host, port, timeout=5.0, context=ssl.create_default_context())
        self._ip = ip

    def connect(self) -> None:
        raw = socket.create_connection((self._ip, self.port), self.timeout)
        self.sock = self._context.wrap_socket(raw, server_hostname=self.host)  # type: ignore[attr-defined]


def send_webhook(url: str, payload: WebhookPayload) -> str:
    resolved = _resolve_public(url)
    if isinstance(resolved, str):
        return "webhook refused"
    host, port, ip = resolved
    parsed = urlparse(url)
    path = (parsed.path or "/") + (f"?{parsed.query}" if parsed.query else "")
    conn = _PinnedHTTPS(host, port, ip)
    try:
        conn.request(
            "POST",
            path,
            body=payload.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        status = conn.getresponse().status  # redirects are not followed
    except (OSError, http.client.HTTPException):
        return "webhook failed"
    finally:
        conn.close()
    return "webhook delivered" if 200 <= status < 300 else "webhook failed"


@router.get("/alerts")
async def get_alerts(request: Request) -> list[dict[str, Any]]:
    with request.app.state.engine.connect() as conn:
        rows = conn.execute(select(alert_rules).order_by(alert_rules.c.id)).mappings().all()
    return [_row(r) for r in rows]


@router.post("/alerts", status_code=201)
async def post_alert(body: AlertIn, request: Request) -> dict[str, Any]:
    try:
        REGISTRY.get(body.analysis_id)
    except UnknownAnalysisError as exc:
        raise _fail(exc) from exc
    if body.webhook_url:
        reason = await anyio.to_thread.run_sync(unsafe_webhook, body.webhook_url)
        if reason:
            raise HTTPException(status_code=422, detail=reason)
    with request.app.state.engine.begin() as conn:
        rid = conn.execute(insert(alert_rules).values(**body.model_dump())).inserted_primary_key[0]
    return {"id": int(rid)}


@router.delete("/alerts/{rid}", status_code=204)
async def delete_alert(rid: int, request: Request) -> None:
    with request.app.state.engine.begin() as conn:
        conn.execute(delete(alert_rules).where(alert_rules.c.id == rid))


def evaluate_alerts(engine: Engine, lake: PriceLake) -> list[dict[str, Any]]:
    """Evaluate every active rule; a notification is created when a rule turns true (not while
    it stays true). Returns one line per rule."""
    with engine.connect() as conn:
        rules = (
            conn.execute(select(alert_rules).where(alert_rules.c.active.is_(True))).mappings().all()
        )
    out = []
    memo: dict[Any, Any] = {}
    for rule in rules:
        run = run_analysis(
            engine,
            lake,
            rule["subject_type"],
            rule["subject_id"],
            rule["analysis_id"],
            rule["params"],
            rule["context"] or None,
            memo,
        )
        value = value_at(run.payload, rule["path"]) if run.payload is not None else None
        state = None if value is None else OPS[rule["op"]](value, float(rule["threshold"]))
        fired = bool(state) and not bool(rule["last_state"])
        with engine.begin() as conn:
            conn.execute(
                update(alert_rules)
                .where(alert_rules.c.id == rule["id"])
                .values(last_value=value, last_state=state, last_evaluated_at=datetime.now(UTC))
            )
            if fired:
                condition = f"{rule['path']} = {value:.6g} {rule['op']} {rule['threshold']:g}"
                if rule.get("webhook_url"):
                    body = run.payload or {}
                    hook = WebhookPayload(
                        ticker=str(body.get("symbol") or rule["subject_id"]),
                        action=str(body.get("action") or "alert"),
                        size=body.get("position"),
                        price=body.get("close"),
                        timestamp=datetime.now(UTC).isoformat(),
                        rule=rule["name"],
                        value=value,
                    )
                    condition += f" · {send_webhook(rule['webhook_url'], hook)}"
                route = {"asset": "a", "portfolio": "p", "comparison": "c", "strategy": "t"}[
                    rule["subject_type"]
                ]
                conn.execute(
                    insert(notifications).values(
                        rule_id=rule["id"],
                        value=value,
                        message=f"{rule['name']}: {condition}",
                        link=f"/{route}/{rule['subject_id']}/"
                        + ("validation" if route == "t" else "overview"),
                    )
                )
        out.append(
            {
                "id": rule["id"],
                "name": rule["name"],
                "value": value,
                "state": state,
                "fired": fired,
                "problem": run.problem,
            }
        )
    return out


@router.post("/alerts/evaluate")
async def post_evaluate(request: Request) -> list[dict[str, Any]]:
    return await anyio.to_thread.run_sync(
        evaluate_alerts, request.app.state.engine, request.app.state.lake
    )


@router.get("/notifications")
async def get_notifications(request: Request, unread: bool = False) -> list[dict[str, Any]]:
    query = select(notifications).order_by(notifications.c.id.desc()).limit(200)
    if unread:
        query = query.where(notifications.c.read_at.is_(None))
    with request.app.state.engine.connect() as conn:
        rows = conn.execute(query).mappings().all()
    return [_row(r) for r in rows]


@router.post("/notifications/read")
async def post_read(request: Request) -> dict[str, int]:
    with request.app.state.engine.begin() as conn:
        n = conn.execute(
            update(notifications)
            .where(notifications.c.read_at.is_(None))
            .values(read_at=datetime.now(UTC))
        ).rowcount
    return {"read": int(n)}


# ── Snapshots (CP-006) ───────────────────────────────────────────────────────


class SnapshotIn(BaseModel):
    subject_type: Literal["asset", "portfolio", "comparison"]
    subject_id: str = Field(max_length=64)
    label: str = Field(max_length=120)
    analyses: list[str] = Field(min_length=1, max_length=20)
    context: dict[str, Any] = Field(default_factory=dict)


def _flat(payload: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Numeric leaves of a payload, depth ≤ 2, lists of records skipped (kept small)."""
    out: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}{key}"
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float) and value == value:
            out[name] = float(value)
        elif isinstance(value, dict) and not prefix:
            out |= _flat(value, f"{name}.")
    return out


def take_snapshot(engine: Engine, lake: PriceLake, body: SnapshotIn) -> dict[str, Any]:
    """Runs the analyses and stores their numeric leaves (also used by M31 schedules)."""
    memo: dict[Any, Any] = {}
    metrics, versions, problems = {}, {}, {}
    for aid in body.analyses:
        run = run_analysis(
            engine, lake, body.subject_type, body.subject_id, aid, {}, body.context or None, memo
        )
        if run.payload is None:
            problems[aid] = run.problem
            continue
        metrics[aid] = _flat(run.payload)
        versions[aid] = f"{run.version}|{run.data_version}"
    with engine.begin() as conn:
        sid = conn.execute(
            insert(snapshots).values(
                subject_type=body.subject_type,
                subject_id=body.subject_id,
                label=body.label,
                metrics=metrics,
                versions=versions,
            )
        ).inserted_primary_key[0]
    return {"id": int(sid), "analyses": list(metrics), "problems": problems}


@router.post("/snapshots", status_code=201)
async def post_snapshot(body: SnapshotIn, request: Request) -> dict[str, Any]:
    engine, lake = request.app.state.engine, request.app.state.lake
    try:
        return await anyio.to_thread.run_sync(take_snapshot, engine, lake, body)
    except UnknownAnalysisError as exc:
        raise _fail(exc) from exc


@router.get("/snapshots")
async def get_snapshots(
    request: Request, subject_type: str, subject_id: str
) -> list[dict[str, Any]]:
    with request.app.state.engine.connect() as conn:
        rows = (
            conn.execute(
                select(snapshots.c.id, snapshots.c.label, snapshots.c.created_at)
                .where(
                    (snapshots.c.subject_type == subject_type)
                    & (snapshots.c.subject_id == subject_id)
                )
                .order_by(snapshots.c.id.desc())
            )
            .mappings()
            .all()
        )
    return [_row(r) for r in rows]


@router.get("/snapshots/diff")
async def get_diff(request: Request, a: int, b: int) -> dict[str, Any]:
    """Metric-by-metric change from snapshot a to snapshot b."""
    with request.app.state.engine.connect() as conn:
        rows = {
            r["id"]: r
            for r in conn.execute(select(snapshots).where(snapshots.c.id.in_([a, b]))).mappings()
        }
    if a not in rows or b not in rows:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    old, new = rows[a]["metrics"], rows[b]["metrics"]
    changes = []
    for aid in sorted(set(old) | set(new)):
        for key in sorted(set(old.get(aid, {})) | set(new.get(aid, {}))):
            x, y = old.get(aid, {}).get(key), new.get(aid, {}).get(key)
            changes.append(
                {
                    "analysis": aid,
                    "metric": key,
                    "before": x,
                    "after": y,
                    "change": None if x is None or y is None else y - x,
                }
            )
    return {
        "a": _row({"id": a, "label": rows[a]["label"], "created_at": rows[a]["created_at"]}),
        "b": _row({"id": b, "label": rows[b]["label"], "created_at": rows[b]["created_at"]}),
        "changes": changes,
    }
