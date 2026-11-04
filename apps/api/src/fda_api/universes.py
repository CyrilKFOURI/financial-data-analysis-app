"""M20 endpoints: universes (SC-001), member ingestion, metric snapshots (SC-002), screens
(SC-003, SC-008) and saved screens with entrants/leavers (SC-007)."""

import math
from datetime import UTC, date, datetime
from typing import Any

import anyio
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import Engine, delete, insert, select, update

from fda_api.data import request_fundamentals_ingestion, request_ingestion
from fda_api.resolve import _catalog_row, _frame, type_conventions
from fda_api.runner import run_analysis
from fda_contracts.data import JobAccepted
from fda_core.analyses.screener import (
    FACTORS,
    METRICS,
    PRESETS,
    ScreenParams,
    price_metrics,
    run_screen,
    screen_diff,
)
from fda_core.registry import InsufficientDataError
from fda_data.fundamentals import FUNDAMENTALS_DATASET
from fda_data.ingest import PRICES_DATASET, _update_job, create_job
from fda_data.lake import PriceLake
from fda_data.portfolios import PortfolioError, load_portfolio
from fda_data.schema import saved_screens
from fda_data.universes import (
    CRYPTO_SOURCE,
    SOURCES,
    Member,
    UniverseError,
    create_universe,
    delete_universe,
    fetch_members,
    list_universes,
    load_universe,
    parse_pasted,
    read_snapshot,
    save_members,
    snapshot_dates,
    write_snapshot,
)

router = APIRouter(prefix="/v1", tags=["screener"])

LABELS = {k: v.label for k, v in SOURCES.items()} | {CRYPTO_SOURCE: "Crypto top 50"}


def _engine(request: Request) -> Engine:
    engine: Engine = request.app.state.engine
    return engine


def _lake(request: Request) -> PriceLake:
    lake: PriceLake = request.app.state.lake
    return lake


class UniverseCreate(BaseModel):
    name: str | None = Field(None, max_length=80)
    source: str | None = Field(None, description="sp500, nasdaq100, dow30, cac40, crypto50")
    symbols: str | None = Field(None, max_length=20_000, description="Pasted list")
    portfolio: int | None = None


def _members_for(engine: Engine, body: UniverseCreate) -> tuple[str, str, str | None, list[Member]]:
    if body.symbols:
        return body.name or "Custom universe", "custom", None, parse_pasted(body.symbols)
    if body.portfolio is not None:
        try:
            d = load_portfolio(engine, body.portfolio, None)
        except PortfolioError as exc:
            raise UniverseError(str(exc)) from exc
        symbols = {h["symbol"] for h in d["holdings"] if not h["is_cash"]} | set(
            d.get("symbols") or []
        )
        members = [Member(s) for s in sorted(symbols)]
        return (
            body.name or f"Holdings – {d['name']}",
            "portfolio",
            f"portfolio:{body.portfolio}",
            members,
        )
    if body.source:
        if body.source not in LABELS:
            raise UniverseError(f"Unknown source {body.source!r}")
        kind = "crypto" if body.source == CRYPTO_SOURCE else "index"
        return body.name or LABELS[body.source], kind, body.source, fetch_members(body.source)
    raise UniverseError("Give a source, a pasted list or a portfolio")


@router.get("/universes")
def get_universes(request: Request) -> list[dict[str, Any]]:
    rows = list_universes(_engine(request))
    lake = _lake(request)
    return [
        r | {"snapshots": [d.isoformat() for d in snapshot_dates(lake, r["id"])][-5:]} for r in rows
    ]


@router.get("/universes/sources")
def get_sources() -> list[dict[str, str]]:
    return [{"key": k, "label": v} for k, v in LABELS.items()]


@router.post("/universes", status_code=201)
async def post_universe(request: Request, body: UniverseCreate) -> dict[str, Any]:
    engine = _engine(request)
    try:
        name, kind, source, members = await anyio.to_thread.run_sync(_members_for, engine, body)
        uid = await anyio.to_thread.run_sync(
            create_universe, engine, name, kind, source, members, date.today()
        )
    except UniverseError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:  # provider failures surface as a reason, not a 500
        raise HTTPException(502, f"Could not load constituents: {exc}") from exc
    return {"id": uid, "members": len(members)}


@router.get("/universes/{uid}")
def get_universe(request: Request, uid: int) -> dict[str, Any]:
    try:
        u = load_universe(_engine(request), uid)
    except UniverseError as exc:
        raise HTTPException(404, str(exc)) from exc
    return u | {"snapshots": [d.isoformat() for d in snapshot_dates(_lake(request), uid)]}


@router.delete("/universes/{uid}", status_code=204)
def remove_universe(request: Request, uid: int) -> None:
    delete_universe(_engine(request), uid)


@router.post("/universes/{uid}/refresh")
async def refresh_universe(request: Request, uid: int) -> dict[str, list[str]]:
    """Re-reads the source list: entrants open a membership, leavers get valid_to today."""
    engine = _engine(request)
    try:
        u = load_universe(engine, uid)
    except UniverseError as exc:
        raise HTTPException(404, str(exc)) from exc
    source = u["source"]
    if not source or source.startswith("portfolio:"):
        raise HTTPException(422, "Only index and crypto universes have a source to refresh")
    members = await anyio.to_thread.run_sync(fetch_members, source)
    return save_members(engine, uid, members, date.today())


@router.post("/universes/{uid}/ingest", status_code=202)
async def ingest_universe(
    request: Request,
    background: BackgroundTasks,
    uid: int,
    fundamentals: bool = False,
    snapshot: bool = True,
) -> dict[str, Any]:
    """Queues price ingestion for every member (fundamentals for equities on request), then a
    snapshot: inline, background tasks run in order, so the snapshot sees the new data; with
    the worker, rebuild the snapshot once the jobs are done."""
    engine = _engine(request)
    try:
        u = load_universe(engine, uid)
    except UniverseError as exc:
        raise HTTPException(404, str(exc)) from exc
    symbols = [m["symbol"] for m in u["members"]]
    prices = [await request_ingestion(request, background, s) for s in symbols]
    funds: list[JobAccepted] = []
    if fundamentals and u["kind"] != "crypto":
        funds = [await request_fundamentals_ingestion(request, background, s) for s in symbols]
    snapshot_job = None
    if snapshot and request.app.state.arq is None:
        snapshot_job = await anyio.to_thread.run_sync(
            create_job, engine, "snapshot", {"universe": uid}
        )
        background.add_task(build_snapshot, engine, _lake(request), uid, snapshot_job, date.today())
    return {"prices": len(prices), "fundamentals": len(funds), "snapshot_job": snapshot_job}


# ── SC-002 snapshot builder ──────────────────────────────────────────────────

SNAPSHOT_ANALYSES = ("V-001", "F-010", "F-011", "F-012", "F-015", "F-017", "S-001", "S-002")
LAST_ROW = {
    "F-011": {
        "gross_margin": "gross_margin",
        "ebit_margin": "ebit_margin",
        "net_margin": "net_margin",
        "fcf_margin": "fcf_margin",
    },
    "F-012": {"roe": "roe", "roic": "roic"},
    "F-015": {"nd_ebitda": "nd_ebitda", "debt_to_equity": "debt_to_equity"},
    "F-017": {"fcf_to_ni": "fcf_to_ni"},
    "F-010": {"revenue_yoy": "revenue_growth", "eps_diluted_yoy": "eps_growth"},
}


def _num(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _last(values: list[Any]) -> float | None:
    for v in reversed(values or []):
        if (x := _num(v)) is not None:
            return x
    return None


def fundamental_metrics(engine: Engine, lake: PriceLake, symbol: str) -> dict[str, Any]:
    """Snapshot fields from the catalog analyses (same formulas as the company pages)."""
    out: dict[str, Any] = {}
    memo: dict[Any, Any] = {}
    for aid in SNAPSHOT_ANALYSES:
        run = run_analysis(engine, lake, "asset", symbol, aid, {}, {"range": "1Y"}, memo)
        p = run.payload
        if p is None:
            continue
        if aid == "V-001":
            m, y = p.get("multiples") or {}, p.get("yields") or {}
            out |= {
                "market_cap": _num(p.get("market_cap")),
                "enterprise_value": _num(p.get("enterprise_value")),
                "pe": _num(m.get("pe")),
                "ev_ebitda": _num(m.get("ev_ebitda")),
                "ev_ebit": _num(m.get("ev_ebit")),
                "ev_sales": _num(m.get("ev_sales")),
                "p_b": _num(m.get("p_b")),
                "earnings_yield": _num(y.get("earnings")),
                "fcf_yield": _num(y.get("fcf")),
                "dividend_yield": _num(y.get("dividend")),
            }
            net_debt = _num(p.get("net_debt"))
            out["net_cash"] = None if net_debt is None else float(net_debt < 0)
        elif aid == "S-001":
            out["health_score"] = _num(p.get("overall"))
        elif aid == "S-002":
            out["piotroski"] = _num(p.get("score"))
        if aid in LAST_ROW:
            rows = {r.get("code"): r for r in p.get("rows") or []}
            for code, key in LAST_ROW[aid].items():
                if code in rows:
                    out[key] = _last(rows[code].get("values") or [])
        if aid == "F-010":
            out["revenue_cagr_3y"] = _num(
                ((p.get("summary") or {}).get("revenue") or {}).get("cagr_3y")
            )
    return out


def dividend_years(actions: pd.DataFrame, today: date) -> float | None:
    """Consecutive completed calendar years with a higher total dividend than the year before."""
    if actions.empty:
        return None
    div = actions.loc[actions["action"] == "dividend"]
    if div.empty:
        return 0.0
    totals = div.groupby(pd.to_datetime(div["ex_date"]).dt.year)["amount"].sum()
    years = 0
    year = today.year - 1
    while year in totals.index and year - 1 in totals.index and totals[year] > totals[year - 1]:
        years += 1
        year -= 1
    return float(years)


def build_snapshot(engine: Engine, lake: PriceLake, uid: int, job_id: int, today: date) -> None:
    u = load_universe(engine, uid)
    members = u["members"]
    _update_job(
        engine,
        job_id,
        status="running",
        started_at=datetime.now(UTC),
        message=f"{len(members)} members",
    )
    benches: dict[str, pd.Series | None] = {}
    rows = []
    for i, m in enumerate(members):
        symbol = m["symbol"]
        row: dict[str, Any] = {"symbol": symbol, "name": m["name"], "sector": m["sector"]}
        price_row = _catalog_row(engine, symbol, PRICES_DATASET, {})
        if price_row is not None and price_row["rows"] is not None:
            conv = type_conventions(price_row)
            bench_symbol = str(conv.get("benchmark") or "SPY")
            if bench_symbol not in benches:
                b_row = _catalog_row(engine, bench_symbol, PRICES_DATASET, {})
                benches[bench_symbol] = (
                    _frame(lake, b_row)["adj_close"] if b_row and b_row["rows"] else None
                )
            frame = _frame(lake, price_row)
            row |= price_metrics(
                frame, benches[bench_symbol], int(conv.get("annualisation") or 252)
            )
            row["name"] = row["name"] or price_row.get("name")
            row["sector"] = row["sector"] or price_row.get("sector")
            row["dividend_years"] = dividend_years(lake.read_actions(int(price_row["id"])), today)
        fund_row = _catalog_row(engine, symbol, FUNDAMENTALS_DATASET, {})
        if fund_row is not None and fund_row["rows"] is not None:
            row |= fundamental_metrics(engine, lake, symbol)
        rows.append(row)
        if i % 10 == 0:
            _update_job(
                engine,
                job_id,
                progress=0.05 + 0.9 * i / max(len(members), 1),
                message=f"{i}/{len(members)} {symbol}",
            )
    frame = pd.DataFrame(rows)
    for metric in METRICS:
        if metric not in frame:
            frame[metric] = None
    frame[list(METRICS)] = frame[list(METRICS)].apply(pd.to_numeric, errors="coerce")
    write_snapshot(lake, uid, today, frame)
    with_prices = int(frame["last_price"].notna().sum())
    _update_job(
        engine,
        job_id,
        status="succeeded",
        progress=1.0,
        finished_at=datetime.now(UTC),
        rows_written=len(frame),
        message=f"{len(frame)} members, {with_prices} with prices",
    )


@router.post("/universes/{uid}/snapshot", status_code=202)
async def post_snapshot(request: Request, background: BackgroundTasks, uid: int) -> JobAccepted:
    engine = _engine(request)
    try:
        load_universe(engine, uid)
    except UniverseError as exc:
        raise HTTPException(404, str(exc)) from exc
    job_id = await anyio.to_thread.run_sync(create_job, engine, "snapshot", {"universe": uid})
    background.add_task(build_snapshot, engine, _lake(request), uid, job_id, date.today())
    return JobAccepted(job_id=job_id, runner="inline")


# ── screens ──────────────────────────────────────────────────────────────────


@router.get("/screener/metrics")
def get_metrics() -> dict[str, Any]:
    return {
        "metrics": [
            {"key": k, "label": v[0], "unit": v[1], "direction": v[2], "group": v[3]}
            for k, v in METRICS.items()
        ],
        "factors": {k: [m for m, _ in v] for k, v in FACTORS.items()},
        "presets": [
            {"key": k, "label": v["label"], "filters": v["filters"]} for k, v in PRESETS.items()
        ],
    }


def _screen(lake: PriceLake, uid: int, params: ScreenParams) -> dict[str, Any]:
    snap = read_snapshot(lake, uid)
    if snap is None:
        raise HTTPException(409, "No snapshot yet: build one (POST /v1/universes/{id}/snapshot)")
    as_of, frame = snap
    try:
        return {"as_of": as_of.isoformat(), **run_screen(frame, params)}
    except InsufficientDataError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/universes/{uid}/screen")
def post_screen(request: Request, uid: int, body: ScreenParams) -> dict[str, Any]:
    return _screen(_lake(request), uid, body)


class SavedScreen(BaseModel):
    name: str = Field(max_length=80)
    universe_id: int
    filters: dict[str, Any] = Field(default_factory=dict)
    preset: str | None = None
    columns: list[str] = Field(default_factory=lambda: ["market_cap", "pe", "roic", "return_12m"])


@router.get("/screens")
def get_screens(request: Request) -> list[dict[str, Any]]:
    with _engine(request).connect() as conn:
        rows = conn.execute(select(saved_screens).order_by(saved_screens.c.name)).mappings().all()
    return [
        dict(r) | {"last_count": len((r["last_result"] or {}).get("symbols", []))} for r in rows
    ]


@router.post("/screens", status_code=201)
def post_saved_screen(request: Request, body: SavedScreen) -> dict[str, int]:
    filters = PRESETS[body.preset]["filters"] if body.preset in PRESETS else body.filters
    with _engine(request).begin() as conn:
        sid = conn.execute(
            insert(saved_screens).values(
                name=body.name, universe_id=body.universe_id, filters=filters, columns=body.columns
            )
        ).inserted_primary_key[0]  # type: ignore[index]
    return {"id": int(sid)}


@router.post("/screens/{sid}/run")
def run_saved_screen(request: Request, sid: int) -> dict[str, Any]:
    """Runs on the latest snapshot; entrants/leavers vs the previous run (SC-007)."""
    engine = _engine(request)
    with engine.connect() as conn:
        row = (
            conn.execute(select(saved_screens).where(saved_screens.c.id == sid))
            .mappings()
            .one_or_none()
        )
    if row is None:
        raise HTTPException(404, f"Unknown screen {sid}")
    result = _screen(
        _lake(request),
        row["universe_id"],
        ScreenParams(filters=row["filters"], columns=row["columns"]),
    )
    previous = (row["last_result"] or {}).get("symbols")
    diff = screen_diff(previous, result["symbols"]) if previous is not None else None
    with engine.begin() as conn:
        conn.execute(
            update(saved_screens)
            .where(saved_screens.c.id == sid)
            .values(
                last_run_at=datetime.now(UTC),
                last_result={"as_of": result["as_of"], "symbols": result["symbols"]},
            )
        )
    return result | {
        "name": row["name"],
        "diff": diff,
        "previous_as_of": (row["last_result"] or {}).get("as_of"),
    }


@router.delete("/screens/{sid}", status_code=204)
def delete_saved_screen(request: Request, sid: int) -> None:
    with _engine(request).begin() as conn:
        conn.execute(delete(saved_screens).where(saved_screens.c.id == sid))
