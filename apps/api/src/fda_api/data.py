"""Data endpoints (M02): instruments, prices, corporate actions, reference series, data
status, ingestion jobs with SSE progress."""

import asyncio
import json
import re
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, date, datetime
from typing import Any, Literal

import anyio
from fastapi import APIRouter, BackgroundTasks, HTTPException, Path, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import Engine, delete, insert, or_, select

from fda_contracts.data import (
    AssetType,
    CorporateAction,
    DatasetStatus,
    DataStatus,
    InstrumentSummary,
    JobAccepted,
    JobInfo,
    OverrideRequest,
    PriceSeries,
    SeriesData,
    SeriesStatus,
    TypeConventions,
)
from fda_core.analyses.markets import all_market_inputs
from fda_core.asset_types import Classification, conventions
from fda_data.chain import ProviderChain, SeriesChain
from fda_data.company import (
    LiveYahoo,
    build_peer_group,
    edit_peer_group,
    ingest_company_data,
    load_peer_group,
)
from fda_data.fundamentals import ingest_fundamentals
from fda_data.ingest import (
    PRICES_DATASET,
    SERIES_DATASET,
    TERMINAL_STATUSES,
    _update_job,
    create_job,
    ingest_daily_prices,
    ingest_series,
    network_enabled,
    reclassify,
)
from fda_data.lake import PriceLake
from fda_data.options import chain_dates, ingest_options, read_iv_history
from fda_data.providers.base import ProviderError
from fda_data.providers.kenfrench import codes as kf_codes
from fda_data.schema import (
    ingestion_jobs,
    instrument_overrides,
    instruments,
    series_catalog,
)

router = APIRouter(prefix="/v1", tags=["data"])

SYMBOL_PATTERN = r"^[A-Za-z0-9.^=\-]{1,20}$"
SERIES_PATTERN = r"^[A-Za-z0-9_]{2,30}$"
_SYMBOL_RE = re.compile(SYMBOL_PATTERN)
JOB_EVENTS_TIMEOUT_S = 120.0


def _engine(request: Request) -> Engine:
    engine: Engine = request.app.state.engine
    return engine


def _chain(request: Request) -> ProviderChain:
    chain: ProviderChain = request.app.state.chain
    return chain


def _lake(request: Request) -> PriceLake:
    lake: PriceLake = request.app.state.lake
    return lake


def _series_chain(request: Request) -> SeriesChain:
    chain: SeriesChain = request.app.state.series_chain
    return chain


def _dataset(row: Mapping[str, Any]) -> DatasetStatus:
    return DatasetStatus(
        dataset=row["dataset"],
        first_date=row["first_date"],
        last_date=row["last_date"],
        rows=row["rows"],
        source=row["source"],
        last_ingested_at=row["last_ingested_at"],
    )


def _or_none(value: Any) -> Any:
    """Parquet nulls come back as NaN in float columns."""
    return None if value is None or value != value else value


def _summary(row: Mapping[str, Any], ingested: bool, source: str | None) -> InstrumentSummary:
    return InstrumentSummary(
        symbol=row["symbol"],
        name=row["name"],
        asset_class=row["asset_class"],
        asset_subclass=row["asset_subclass"],
        currency=row["currency"],
        ingested=ingested,
        source=source,
        profile=row.get("profile") or {},
    )


def _job(row: Mapping[str, Any]) -> JobInfo:
    return JobInfo(
        id=row["id"],
        kind=row["kind"],
        status=row["status"],
        progress=float(row["progress"]),
        message=row["message"],
        provider=row["provider"],
        rows_written=row["rows_written"],
        qc_summary=row["qc_summary"],
        error=row["error"],
        created_at=row["created_at"],
        finished_at=row["finished_at"],
    )


@router.get("/instruments")
def search_instruments(
    request: Request, q: str = Query(min_length=1, max_length=40)
) -> list[InstrumentSummary]:
    pattern = f"%{q}%"
    with _engine(request).connect() as conn:
        rows = (
            conn.execute(
                select(instruments, series_catalog.c.source)
                .outerjoin(
                    series_catalog,
                    (series_catalog.c.instrument_id == instruments.c.id)
                    & (series_catalog.c.dataset == PRICES_DATASET),
                )
                .where(or_(instruments.c.symbol.ilike(pattern), instruments.c.name.ilike(pattern)))
                .order_by(instruments.c.symbol)
                .limit(20)
            )
            .mappings()
            .all()
        )
    results = [
        _summary(row, ingested=row["source"] is not None, source=row["source"]) for row in rows
    ]
    known = {r.symbol for r in results}
    if _SYMBOL_RE.match(q):
        for found in _chain(request).search(q.upper()):
            if found.value.symbol not in known:
                profile = found.value
                results.append(
                    InstrumentSummary(
                        symbol=profile.symbol,
                        name=profile.name,
                        asset_class=profile.asset_class,
                        asset_subclass=profile.asset_subclass,
                        currency=profile.currency,
                        ingested=False,
                        source=found.source,
                    )
                )
    return results


@router.get("/instruments/{symbol}/prices")
def get_prices(
    request: Request,
    symbol: str = Path(pattern=SYMBOL_PATTERN),
    start: date | None = None,
    end: date | None = None,
) -> PriceSeries:
    with _engine(request).connect() as conn:
        row = (
            conn.execute(select(instruments).where(instruments.c.symbol == symbol.upper()))
            .mappings()
            .one_or_none()
        )
    if row is None:
        raise HTTPException(404, f"{symbol.upper()} has not been ingested")
    bars = _lake(request).read_prices(row["asset_class"], row["id"], start, end)
    if bars.empty:
        raise HTTPException(404, f"No price data stored for {symbol.upper()}")
    return PriceSeries(
        symbol=row["symbol"],
        currency=row["currency"],
        source=str(bars["source"].iloc[-1]),
        dates=list(bars["session_date"]),
        open=bars["open"].tolist(),
        high=bars["high"].tolist(),
        low=bars["low"].tolist(),
        close=bars["close"].tolist(),
        adj_close=bars["adj_close"].tolist(),
        volume=bars["volume"].tolist(),
    )


@router.get("/data-status/{symbol}")
def data_status(request: Request, symbol: str = Path(pattern=SYMBOL_PATTERN)) -> DataStatus:
    symbol = symbol.upper()
    chain = _chain(request)
    with _engine(request).connect() as conn:
        row = (
            conn.execute(select(instruments).where(instruments.c.symbol == symbol))
            .mappings()
            .one_or_none()
        )
        datasets = (
            conn.execute(select(series_catalog).where(series_catalog.c.instrument_id == row["id"]))
            .mappings()
            .all()
            if row
            else []
        )
        jobs = (
            conn.execute(
                select(ingestion_jobs)
                .where(ingestion_jobs.c.target["symbol"].as_string() == symbol)
                .order_by(ingestion_jobs.c.id.desc())
                .limit(5)
            )
            .mappings()
            .all()
        )
    can_ingest = chain.can_serve(symbol)
    return DataStatus(
        symbol=symbol,
        instrument=_summary(row, ingested=bool(datasets), source=None) if row else None,
        datasets=[_dataset(d) for d in datasets],
        jobs=[_job(j) for j in jobs],
        can_ingest=can_ingest,
        reason=None if can_ingest else f"No configured provider can serve {symbol}.",
    )


@router.post("/ingest/{symbol}", status_code=202)
async def request_ingestion(
    request: Request, background: BackgroundTasks, symbol: str = Path(pattern=SYMBOL_PATTERN)
) -> JobAccepted:
    symbol = symbol.upper()
    engine = _engine(request)
    job_id = await anyio.to_thread.run_sync(create_job, engine, "prices", {"symbol": symbol})
    pool = request.app.state.arq
    if pool is not None:
        await pool.enqueue_job("ingest_prices", job_id, symbol)
        return JobAccepted(job_id=job_id, runner="worker")
    background.add_task(
        ingest_daily_prices,
        engine,
        _lake(request),
        _chain(request),
        symbol,
        job_id,
        coingecko=request.app.state.coingecko,
    )
    return JobAccepted(job_id=job_id, runner="inline")


@router.get("/instruments/{symbol}/actions")
def get_actions(
    request: Request, symbol: str = Path(pattern=SYMBOL_PATTERN)
) -> list[CorporateAction]:
    with _engine(request).connect() as conn:
        instrument_id = conn.execute(
            select(instruments.c.id).where(instruments.c.symbol == symbol.upper())
        ).scalar_one_or_none()
    if instrument_id is None:
        raise HTTPException(404, f"{symbol.upper()} has not been ingested")
    frame = _lake(request).read_actions(int(instrument_id))
    return [
        CorporateAction(
            ex_date=row["ex_date"],
            action=row["action"],
            ratio=_or_none(row["ratio"]),
            amount=_or_none(row["amount"]),
            currency=_or_none(row["currency"]),
            source=_or_none(row["source"]),
        )
        for row in frame.to_dict("records")
    ]


@router.get("/series/{code}")
def get_series(
    request: Request,
    code: str = Path(pattern=SERIES_PATTERN),
    start: date | None = None,
    end: date | None = None,
) -> SeriesData:
    code = code.upper()
    frame = _lake(request).read_series(code, start, end)
    if frame.empty:
        raise HTTPException(404, f"Series {code} has not been ingested")
    chain = _series_chain(request)
    title, unit, frequency = code, "unknown", "unknown"
    if chain.can_serve(code):
        info = chain.info(code).value
        title, unit, frequency = info.title, info.unit, info.frequency
    return SeriesData(
        code=code,
        title=title,
        unit=unit,
        frequency=frequency,
        source=str(frame["source"].iloc[-1]),
        dates=list(frame["date"]),
        values=frame["value"].tolist(),
    )


@router.get("/series-status/{code}")
def series_status(request: Request, code: str = Path(pattern=SERIES_PATTERN)) -> SeriesStatus:
    code = code.upper()
    chain = _series_chain(request)
    with _engine(request).connect() as conn:
        dataset = (
            conn.execute(
                select(series_catalog).where(
                    series_catalog.c.dataset == SERIES_DATASET, series_catalog.c.key == code
                )
            )
            .mappings()
            .one_or_none()
        )
        jobs = (
            conn.execute(
                select(ingestion_jobs)
                .where(ingestion_jobs.c.target["code"].as_string() == code)
                .order_by(ingestion_jobs.c.id.desc())
                .limit(5)
            )
            .mappings()
            .all()
        )
    try:
        info = chain.info(code)
    except ProviderError:
        info = None
    return SeriesStatus(
        code=code,
        title=info.value.title if info else None,
        unit=info.value.unit if info else None,
        provider=info.source if info else None,
        dataset=_dataset(dataset) if dataset else None,
        jobs=[_job(j) for j in jobs],
        can_ingest=info is not None,
        reason=None if info else f"No configured provider knows series {code}.",
    )


@router.post("/series/{code}/ingest", status_code=202)
async def request_series_ingestion(
    request: Request, background: BackgroundTasks, code: str = Path(pattern=SERIES_PATTERN)
) -> JobAccepted:
    code = code.upper()
    engine = _engine(request)
    job_id = await anyio.to_thread.run_sync(create_job, engine, "series", {"code": code})
    pool = request.app.state.arq
    if pool is not None:
        await pool.enqueue_job("ingest_reference_series", job_id, code)
        return JobAccepted(job_id=job_id, runner="worker")
    background.add_task(ingest_series, engine, _lake(request), _series_chain(request), code, job_id)
    return JobAccepted(job_id=job_id, runner="inline")


@router.post("/factors/{region}/ingest", status_code=202)
async def request_factor_ingestion(
    request: Request, background: BackgroundTasks, region: Literal["US", "EU"]
) -> list[JobAccepted]:
    """Queues the seven Ken French series of a region (M18): KF_<REGION>_<FACTOR>."""
    return [await request_series_ingestion(request, background, code) for code in kf_codes(region)]


@router.post("/markets/ingest", status_code=202)
async def request_market_ingestion(
    request: Request, background: BackgroundTasks
) -> dict[str, list[JobAccepted]]:
    """Queues every instrument and series of the Markets pages (M19); incremental, so a
    re-run only fetches what is new."""
    symbols, codes = all_market_inputs()
    return {
        "prices": [await request_ingestion(request, background, s) for s in symbols],
        "series": [await request_series_ingestion(request, background, c) for c in codes],
    }


def _ingest_options_job(engine: Engine, lake: PriceLake, symbol: str, job_id: int) -> None:
    _update_job(
        engine, job_id, status="running", started_at=datetime.now(UTC), message="Fetching chain"
    )
    try:
        row = _catalog_lookup(engine, symbol)
        if row is None:
            raise ProviderError(f"{symbol} is unknown: ingest its prices first")
        prices = lake.read_prices(row["asset_class"], row["id"])
        last = float(prices["close"].iloc[-1]) if not prices.empty else None
        out = ingest_options(lake, row, last, date.today())
        _update_job(
            engine,
            job_id,
            status="succeeded",
            progress=1.0,
            finished_at=datetime.now(UTC),
            rows_written=out["contracts"],
            message=f"{out['contracts']} contracts, {out['expiries']} expiries",
        )
    except Exception as exc:  # provider or data errors end the job with the reason
        _update_job(
            engine,
            job_id,
            status="failed",
            progress=1.0,
            finished_at=datetime.now(UTC),
            message="Options ingestion failed",
            error=str(exc)[:500],
        )


def _catalog_lookup(engine: Engine, symbol: str) -> dict[str, Any] | None:
    with engine.connect() as conn:
        row = (
            conn.execute(select(instruments).where(instruments.c.symbol == symbol))
            .mappings()
            .one_or_none()
        )
    return dict(row) if row is not None else None


@router.post("/options/{symbol}/ingest", status_code=202)
async def request_options_ingestion(
    request: Request, background: BackgroundTasks, symbol: str = Path(pattern=SYMBOL_PATTERN)
) -> JobAccepted:
    """Today's option chain snapshot (yfinance; synthetic for FAKE*) and its IV summary (M21)."""
    symbol = symbol.upper()
    engine = _engine(request)
    job_id = await anyio.to_thread.run_sync(create_job, engine, "options", {"symbol": symbol})
    background.add_task(_ingest_options_job, engine, _lake(request), symbol, job_id)
    return JobAccepted(job_id=job_id, runner="inline")


@router.get("/options/{symbol}/snapshots")
def option_snapshots(
    request: Request, symbol: str = Path(pattern=SYMBOL_PATTERN)
) -> dict[str, Any]:
    lake = _lake(request)
    history = read_iv_history(lake, symbol.upper())
    return {
        "dates": [d.isoformat() for d in chain_dates(lake, symbol.upper())],
        "iv_history": history.assign(date=history["date"].astype(str)).to_dict("records"),
    }


@router.get(
    "/jobs/{job_id}/events",
    response_model=JobInfo,
    responses={200: {"content": {"text/event-stream": {}}, "description": "SSE of JobInfo"}},
)
async def job_events(request: Request, job_id: int) -> StreamingResponse:
    engine = _engine(request)

    def load() -> Mapping[str, Any] | None:
        with engine.connect() as conn:
            return (
                conn.execute(select(ingestion_jobs).where(ingestion_jobs.c.id == job_id))
                .mappings()
                .one_or_none()
            )

    if await anyio.to_thread.run_sync(load) is None:
        raise HTTPException(404, f"Unknown job {job_id}")

    async def events() -> AsyncIterator[str]:
        last_payload = ""
        with anyio.move_on_after(JOB_EVENTS_TIMEOUT_S):
            while True:
                row = await anyio.to_thread.run_sync(load)
                if row is None:
                    return
                payload = _job(row).model_dump_json()
                if payload != last_payload:
                    yield f"data: {payload}\n\n"
                    last_payload = payload
                if row["status"] in TERMINAL_STATUSES or await request.is_disconnected():
                    return
                await asyncio.sleep(0.3)
        yield f"event: timeout\ndata: {json.dumps({'job_id': job_id})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


def _instrument_row(conn: Any, symbol: str) -> Mapping[str, Any]:
    row = (
        conn.execute(select(instruments).where(instruments.c.symbol == symbol.upper()))
        .mappings()
        .one_or_none()
    )
    if row is None:
        raise HTTPException(404, f"{symbol.upper()} has not been ingested")
    return dict(row)


def _asset_type(engine: Engine, symbol: str) -> AssetType:
    with engine.connect() as conn:
        row = _instrument_row(conn, symbol)
        overrides = dict(
            conn.execute(
                select(instrument_overrides.c.field, instrument_overrides.c.value).where(
                    instrument_overrides.c.instrument_id == row["id"]
                )
            ).all()
        )
    profile = row["profile"] or {}
    leverage = float(row["leverage_factor"]) if row["leverage_factor"] is not None else None
    classification = Classification(
        row["asset_class"],
        row["asset_subclass"] or "",
        row["classification_source"] or "low",
        row["classification_reason"] or "",
        leverage,
        tuple(profile.get("flags") or ()),
    )
    conv = conventions(classification, row["currency"])
    return AssetType(
        symbol=row["symbol"],
        asset_class=row["asset_class"],
        asset_subclass=row["asset_subclass"],
        leverage_factor=leverage,
        confidence=row["classification_confidence"],
        source=row["classification_source"],
        reason=row["classification_reason"],
        flags=list(classification.flags),
        overrides=overrides,
        conventions=TypeConventions(
            annualisation=conv.annualisation,
            calendar=conv.calendar,
            price_field=conv.price_field,
            has_volume=conv.has_volume,
            return_kind=conv.return_kind,
            benchmark=conv.benchmark,
            caveats=list(conv.caveats),
        ),
    )


@router.get("/instruments/{symbol}/type")
def get_asset_type(request: Request, symbol: str = Path(pattern=SYMBOL_PATTERN)) -> AssetType:
    return _asset_type(_engine(request), symbol)


@router.put("/instruments/{symbol}/overrides")
def put_override(
    request: Request, body: OverrideRequest, symbol: str = Path(pattern=SYMBOL_PATTERN)
) -> AssetType:
    """One-click fix of a wrong classification ("this is not a REIT"); rules re-run on top."""
    engine = _engine(request)
    with engine.begin() as conn:
        instrument_id = _instrument_row(conn, symbol)["id"]
        conn.execute(
            delete(instrument_overrides).where(
                instrument_overrides.c.instrument_id == instrument_id,
                instrument_overrides.c.field == body.field,
            )
        )
        conn.execute(
            insert(instrument_overrides).values(
                instrument_id=instrument_id, field=body.field, value=body.value, reason=body.reason
            )
        )
        try:
            reclassify(conn, instrument_id)
        except ValueError as exc:  # not in the taxonomy → roll back the override
            raise HTTPException(422, str(exc)) from exc
    return _asset_type(engine, symbol)


@router.delete("/instruments/{symbol}/overrides/{field}")
def delete_override(
    request: Request, field: str, symbol: str = Path(pattern=SYMBOL_PATTERN)
) -> AssetType:
    engine = _engine(request)
    with engine.begin() as conn:
        instrument_id = _instrument_row(conn, symbol)["id"]
        conn.execute(
            delete(instrument_overrides).where(
                instrument_overrides.c.instrument_id == instrument_id,
                instrument_overrides.c.field == field,
            )
        )
        reclassify(conn, instrument_id)
    return _asset_type(engine, symbol)


@router.post("/fundamentals/{symbol}/ingest", status_code=202)
async def request_fundamentals_ingestion(
    request: Request, background: BackgroundTasks, symbol: str = Path(pattern=SYMBOL_PATTERN)
) -> JobAccepted:
    """SEC companyfacts for US filers, yfinance statements otherwise (job kind `fundamentals`)."""
    symbol = symbol.upper()
    engine = _engine(request)
    job_id = await anyio.to_thread.run_sync(create_job, engine, "fundamentals", {"symbol": symbol})
    pool = request.app.state.arq
    if pool is not None:
        await pool.enqueue_job("ingest_company_fundamentals", job_id, symbol)
        return JobAccepted(job_id=job_id, runner="worker")
    background.add_task(ingest_fundamentals, engine, symbol, job_id)
    return JobAccepted(job_id=job_id, runner="inline")


async def _enqueue(
    request: Request, background: BackgroundTasks, kind: str, task: str, symbol: str, fn: Any
) -> JobAccepted:
    engine = _engine(request)
    job_id = await anyio.to_thread.run_sync(create_job, engine, kind, {"symbol": symbol})
    pool = request.app.state.arq
    if pool is not None:
        await pool.enqueue_job(task, job_id, symbol)
        return JobAccepted(job_id=job_id, runner="worker")
    background.add_task(fn, engine, symbol, job_id)
    return JobAccepted(job_id=job_id, runner="inline")


@router.post("/company-data/{symbol}/ingest", status_code=202)
async def request_company_data(
    request: Request, background: BackgroundTasks, symbol: str = Path(pattern=SYMBOL_PATTERN)
) -> JobAccepted:
    """Earnings dates, estimates, analyst views, holders, insiders, short interest."""
    return await _enqueue(
        request,
        background,
        "company_data",
        "ingest_company_snapshots",
        symbol.upper(),
        ingest_company_data,
    )


@router.post("/peers/{symbol}/build", status_code=202)
async def request_peer_build(
    request: Request, background: BackgroundTasks, symbol: str = Path(pattern=SYMBOL_PATTERN)
) -> JobAccepted:
    """Auto peers (D-PE-2); a manual group is kept unless reset first."""
    return await _enqueue(
        request, background, "peers", "build_peers", symbol.upper(), build_peer_group
    )


class PeerEdit(BaseModel):
    add: list[str] = Field(default_factory=list, max_length=20)
    remove: list[str] = Field(default_factory=list, max_length=20)
    pin: list[str] = Field(default_factory=list, max_length=20)


@router.get("/peers/{symbol}")
def get_peers(request: Request, symbol: str = Path(pattern=SYMBOL_PATTERN)) -> dict[str, Any]:
    with _engine(request).connect() as conn:
        instrument_id = conn.execute(
            select(instruments.c.id).where(instruments.c.symbol == symbol.upper())
        ).scalar_one_or_none()
    if instrument_id is None:
        raise HTTPException(404, f"{symbol.upper()} has not been ingested")
    group = load_peer_group(_engine(request), int(instrument_id))
    return group or {"mode": None, "members": []}


@router.put("/peers/{symbol}")
def put_peers(
    request: Request, body: PeerEdit, symbol: str = Path(pattern=SYMBOL_PATTERN)
) -> dict[str, Any]:
    """Manual edit (PE-002): the group becomes manual and frozen until reset."""
    for ticker in body.add + body.remove + body.pin:
        if not _SYMBOL_RE.match(ticker):
            raise HTTPException(422, f"Invalid symbol {ticker!r}")
    lookup = LiveYahoo().info if network_enabled() else None  # offline: peers added without metrics
    try:
        edit_peer_group(_engine(request), symbol, body.add, body.remove, body.pin, lookup)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return get_peers(request, symbol)


@router.delete("/peers/{symbol}", status_code=202)
async def reset_peers(
    request: Request, background: BackgroundTasks, symbol: str = Path(pattern=SYMBOL_PATTERN)
) -> JobAccepted:
    """Reset to auto: rebuild with the algorithm, discarding manual edits."""
    engine = _engine(request)
    job_id = await anyio.to_thread.run_sync(create_job, engine, "peers", {"symbol": symbol.upper()})
    background.add_task(build_peer_group, engine, symbol.upper(), job_id, None, True)
    return JobAccepted(job_id=job_id, runner="inline")
