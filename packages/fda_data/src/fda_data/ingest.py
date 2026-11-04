"""Incremental ingestion of daily prices, corporate actions and reference series
(TRD §5, BACKEND_SCHEMA §5/§7)."""

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy import Connection, Engine, insert, select, update
from sqlalchemy.exc import IntegrityError

from fda_core.asset_types import InstrumentFacts, apply_overrides, classify
from fda_data.chain import NoProviderError, ProviderChain, SeriesChain
from fda_data.lake import PriceLake
from fda_data.providers.base import (
    InstrumentProfile,
    PriceProvider,
    ProviderError,
    SeriesProvider,
)
from fda_data.providers.coingecko import CoinGeckoProvider
from fda_data.providers.ecb import EcbProvider
from fda_data.providers.fake import FakeProvider, FakeSeriesProvider
from fda_data.providers.fred import FredProvider
from fda_data.providers.kenfrench import KenFrenchProvider
from fda_data.providers.yahoo import YahooProvider
from fda_data.qc import check_daily_bars
from fda_data.ratelimit import TokenBucket
from fda_data.schema import (
    ingestion_jobs,
    instrument_overrides,
    instruments,
    series_catalog,
)

logger = logging.getLogger(__name__)

PRICES_DATASET = "prices_daily"
ACTIONS_DATASET = "actions"
SERIES_DATASET = "series"
DEFAULT_HISTORY_YEARS = 10
SERIES_HISTORY_YEARS = 25
# Long-lived indices used as long-history proxies (M19 stock–bond correlation since the 1960s).
LONG_HISTORY_YEARS = {"^GSPC": 60, "^DJI": 60, "^IXIC": 55}
LONG_SERIES_YEARS = {"DGS10": 65, "USREC": 70, "T10Y2Y": 50}
TERMINAL_STATUSES = frozenset({"succeeded", "failed"})
DEFAULT_PROVIDERS = "fake,yfinance"
# Each refresh re-reads the last days so provider corrections overwrite stored values.
OVERLAP_DAYS = 5


def _price_provider(name: str) -> tuple[PriceProvider, TokenBucket]:
    if name == "fake":
        return FakeProvider(), TokenBucket(rate_per_second=50, burst=50)
    if name == "yfinance":
        # TRD §5.3: conservative default, Yahoo throttles bursts aggressively.
        return YahooProvider(), TokenBucket(rate_per_second=1, burst=5)
    raise ValueError(f"Unknown price provider {name!r}")


def default_chain() -> ProviderChain:
    """Ordered chain from `FDA_PROVIDERS` (tests and e2e use `fake` only: no network)."""
    built = [_price_provider(name) for name in _provider_names()]
    return ProviderChain(
        [provider for provider, _ in built], {p.name: limiter for p, limiter in built}
    )


def _provider_names() -> list[str]:
    return [
        n.strip() for n in os.getenv("FDA_PROVIDERS", DEFAULT_PROVIDERS).split(",") if n.strip()
    ]


def network_enabled() -> bool:
    """Any real provider configured (offline runs use `FDA_PROVIDERS=fake`)."""
    return any(name != "fake" for name in _provider_names())


def default_series_chain() -> SeriesChain:
    """ECB first (it owns ESTR and ECB_EUR* codes), Ken French (KF_*), then FRED."""
    providers: list[SeriesProvider] = []
    if "fake" in _provider_names():
        providers.append(FakeSeriesProvider())
    if network_enabled():
        providers += [EcbProvider(), KenFrenchProvider(), FredProvider()]
    return SeriesChain(
        providers,
        {
            "ecb": TokenBucket(rate_per_second=2, burst=4),
            "fred": TokenBucket(rate_per_second=2, burst=4),
            "kenfrench": TokenBucket(rate_per_second=1, burst=8),
        },
    )


@dataclass(slots=True)
class IngestResult:
    job_id: int
    symbol: str
    status: str
    instrument_id: int | None = None
    rows_written: int = 0
    source: str | None = None
    message: str = ""
    retryable: bool = False
    qc: dict[str, Any] = field(default_factory=dict)


def create_job(engine: Engine, kind: str, target: Mapping[str, Any]) -> int:
    with engine.begin() as conn:
        job_id: int = conn.execute(
            insert(ingestion_jobs).values(kind=kind, target=dict(target), status="queued")
        ).inserted_primary_key[0]
    return job_id


def _update_job(engine: Engine, job_id: int, **values: Any) -> None:
    with engine.begin() as conn:
        conn.execute(update(ingestion_jobs).where(ingestion_jobs.c.id == job_id).values(**values))


def _start_job(engine: Engine, job_id: int, message: str) -> None:
    _update_job(
        engine,
        job_id,
        status="running",
        started_at=datetime.now(UTC),
        progress=0.05,
        message=message,
        attempts=ingestion_jobs.c.attempts + 1,
    )


def reclassify(conn: Connection, instrument_id: int) -> None:
    """Re-run the detection pipeline (rules + overrides) on the stored profile."""
    row = (
        conn.execute(select(instruments).where(instruments.c.id == instrument_id)).mappings().one()
    )
    profile = dict(row["profile"] or {})
    overrides = dict(
        conn.execute(
            select(instrument_overrides.c.field, instrument_overrides.c.value).where(
                instrument_overrides.c.instrument_id == instrument_id
            )
        ).all()
    )
    base = classify(
        InstrumentFacts(
            symbol=row["symbol"],
            name=row["name"],
            provider_class=profile.get("provider_class", row["asset_class"]),
            provider_subclass=profile.get("provider_subclass", row["asset_subclass"]),
            sector=row["sector"],
            industry=row["industry"],
            country=row["country"],
            profile=profile,
        )
    )
    result = apply_overrides(base, overrides)
    conn.execute(
        update(instruments)
        .where(instruments.c.id == instrument_id)
        .values(
            asset_class=result.asset_class,
            asset_subclass=result.asset_subclass,
            leverage_factor=result.leverage_factor,
            classification_source=result.confidence,
            classification_reason=result.reason,
            classification_confidence="high"
            if result.confidence in {"override", "rule"}
            else "medium"
            if result.confidence == "provider"
            else "low",
            profile=profile | {"flags": list(result.flags)},
        )
    )


def _upsert_instrument(conn: Connection, profile: InstrumentProfile, source: str) -> int:
    instrument_id = _store_profile(conn, profile, source)
    reclassify(conn, instrument_id)
    return instrument_id


def _store_profile(conn: Connection, profile: InstrumentProfile, source: str) -> int:
    # The provider's coarse type is kept in the profile; the columns hold the refined type.
    extras = profile.extras | {
        "provider_class": profile.asset_class,
        "provider_subclass": profile.asset_subclass,
    }
    values = {
        "name": profile.name,
        "asset_class": profile.asset_class,
        "asset_subclass": profile.asset_subclass,
        "currency": profile.currency,
        "country": profile.country,
        "sector": profile.sector,
        "industry": profile.industry,
        "timezone": profile.timezone,
        "provider_symbols": profile.provider_symbols,
        "classification_source": "provider",
        "classification_reason": f"{source} profile",
        "classification_confidence": "medium",
    }
    existing = (
        conn.execute(
            select(instruments.c.id, instruments.c.profile).where(
                instruments.c.symbol == profile.symbol
            )
        )
        .mappings()
        .one_or_none()
    )
    if existing is not None:
        # Keep extras from other sources (e.g. CoinGecko supply) and refresh the provider's own.
        merged = {**(existing["profile"] or {}), **extras}
        conn.execute(
            update(instruments)
            .where(instruments.c.id == existing["id"])
            .values(profile=merged, **values)
        )
        return int(existing["id"])
    try:
        with conn.begin_nested():  # two jobs may race on a new symbol: the loser updates
            inserted = conn.execute(
                insert(instruments).values(symbol=profile.symbol, profile=extras, **values)
            )
        return int(inserted.inserted_primary_key[0])
    except IntegrityError:
        return _store_profile(conn, profile, source)


def _catalog_row(conn: Connection, dataset: str, key: str) -> Mapping[str, Any] | None:
    return (
        conn.execute(
            select(series_catalog).where(
                series_catalog.c.dataset == dataset, series_catalog.c.key == key
            )
        )
        .mappings()
        .one_or_none()
    )


def _fetch_window(
    catalog: Mapping[str, Any] | None, today: date, history_years: int
) -> tuple[date, date] | None:
    """[start, end] to request, or None when already up to date.

    Only completed days are stored (end = yesterday): a bar dated today may still be moving.
    Slice 012 refines this with exchange calendars and closing times.
    """
    end = today - timedelta(days=1)
    last_date: date | None = catalog["last_date"] if catalog else None
    if last_date is None:
        return today - timedelta(days=365 * history_years), end
    if last_date >= end:
        return None
    return last_date - timedelta(days=OVERLAP_DAYS), end


def price_window(
    symbol: str,
    catalog: Mapping[str, Any] | None,
    today: date,
    default: int = DEFAULT_HISTORY_YEARS,
    long: Mapping[str, int] = LONG_HISTORY_YEARS,
) -> tuple[date, date] | None:
    """Incremental window, except that a long-history key stored with a shorter history is
    backfilled once from its full start (the lake merge keeps existing rows)."""
    years = long.get(symbol.upper(), default)
    window = _fetch_window(catalog, today, years)
    first = catalog["first_date"] if catalog else None
    wanted = today - timedelta(days=365 * years)
    if first is not None and years > default and first > wanted + timedelta(days=400):
        return wanted, today - timedelta(days=1)
    return window


def _record_catalog(
    conn: Connection,
    dataset: str,
    key: str,
    coverage: tuple[date, date, int] | None,
    source: str,
    instrument_id: int | None = None,
) -> None:
    """Store the coverage measured on the lake (first date, last date, rows)."""
    if coverage is None:
        return
    first_date, last_date, rows = coverage
    values = {
        "first_date": first_date,
        "last_date": last_date,
        "rows": rows,
        "source": source,
        "last_ingested_at": datetime.now(UTC),
    }
    catalog = _catalog_row(conn, dataset, key)
    if catalog:
        conn.execute(
            update(series_catalog).where(series_catalog.c.id == catalog["id"]).values(**values)
        )
    else:
        conn.execute(
            insert(series_catalog).values(
                dataset=dataset, key=key, instrument_id=instrument_id, **values
            )
        )


def _attach_crypto_metadata(
    engine: Engine, instrument_id: int, symbol: str, coingecko: CoinGeckoProvider | None
) -> None:
    """Best effort: CoinGecko failures never fail a price ingestion."""
    if coingecko is None:
        return
    with engine.connect() as conn:
        profile = conn.execute(
            select(instruments.c.profile).where(instruments.c.id == instrument_id)
        ).scalar_one()
    if (profile or {}).get("coingecko_id"):
        return
    try:
        metadata = coingecko.crypto_metadata(symbol)
    except ProviderError as exc:
        logger.warning("CoinGecko metadata unavailable for %s: %s", symbol, exc)
        return
    with engine.begin() as conn:
        conn.execute(
            update(instruments)
            .where(instruments.c.id == instrument_id)
            .values(profile={**(profile or {}), **metadata})
        )
        reclassify(conn, instrument_id)  # CoinGecko categories can reveal a stablecoin


def ingest_daily_prices(
    engine: Engine,
    lake: PriceLake,
    chain: ProviderChain,
    symbol: str,
    job_id: int | None = None,
    today: date | None = None,
    coingecko: CoinGeckoProvider | None = None,
    requeue: bool = False,
) -> IngestResult:
    """Fetch only missing sessions (+ their corporate actions), QC, write the lake and catalog."""
    symbol = symbol.upper()
    today = today or date.today()
    if job_id is None:
        job_id = create_job(engine, "prices", {"symbol": symbol})
    _start_job(engine, job_id, "Resolving instrument")
    result = IngestResult(job_id=job_id, symbol=symbol, status="running")

    try:
        profile = chain.profile(symbol)
        with engine.begin() as conn:
            instrument_id = _upsert_instrument(conn, profile.value, profile.source)
            catalog = _catalog_row(conn, PRICES_DATASET, str(instrument_id))
        result.instrument_id = instrument_id
        if profile.value.asset_class == "crypto":
            _attach_crypto_metadata(engine, instrument_id, symbol, coingecko)

        window = price_window(symbol, catalog, today)
        if window is None:
            return _finish(engine, result, "succeeded", "Already up to date")
        start, end = window

        _update_job(engine, job_id, progress=0.3, message=f"Fetching {start} → {end}")
        bars = chain.daily_bars(symbol, start, end)
        result.source = bars.source
        actions = chain.actions(symbol, start, end, bars.source).assign(source=bars.source)
        actions = actions.loc[actions["ex_date"] <= end]
        split_dates = set(actions.loc[actions["action"] == "split", "ex_date"])

        _update_job(engine, job_id, progress=0.6, message="Quality control")
        completed = bars.value.loc[bars.value["session_date"] <= end]
        clean, report = check_daily_bars(completed, split_dates)
        clean = clean.assign(source=bars.source)
        result.qc = report.as_dict() | {"actions": len(actions)}

        _update_job(engine, job_id, progress=0.8, message="Writing lake")
        lake.write_prices(profile.value.asset_class, instrument_id, clean)
        lake.write_actions(instrument_id, actions)
        result.rows_written = len(clean)

        asset_class = profile.value.asset_class
        with engine.begin() as conn:
            _record_catalog(
                conn,
                PRICES_DATASET,
                str(instrument_id),
                lake.price_coverage(asset_class, instrument_id),
                bars.source,
                instrument_id,
            )
            _record_catalog(
                conn,
                ACTIONS_DATASET,
                str(instrument_id),
                lake.actions_coverage(instrument_id),
                bars.source,
                instrument_id,
            )
        return _finish(engine, result, "succeeded", f"{len(clean)} sessions written")
    except NoProviderError as exc:
        return _provider_unavailable(engine, result, exc, requeue)
    except ProviderError as exc:
        return _finish(engine, result, "failed", "Provider error", error=str(exc))


def ingest_series(
    engine: Engine,
    lake: PriceLake,
    chain: SeriesChain,
    code: str,
    job_id: int | None = None,
    today: date | None = None,
    requeue: bool = False,
) -> IngestResult:
    """Rates, FX reference rates and macro series: levels by date, incremental."""
    code = code.upper()
    today = today or date.today()
    if job_id is None:
        job_id = create_job(engine, "series", {"code": code})
    _start_job(engine, job_id, "Resolving series")
    result = IngestResult(job_id=job_id, symbol=code, status="running")

    try:
        with engine.connect() as conn:
            catalog = _catalog_row(conn, SERIES_DATASET, code)
        window = price_window(code, catalog, today, SERIES_HISTORY_YEARS, LONG_SERIES_YEARS)
        if window is None:
            return _finish(engine, result, "succeeded", "Already up to date")
        start, end = window

        _update_job(engine, job_id, progress=0.3, message=f"Fetching {start} → {end}")
        observations = chain.series(code, start, end)
        result.source = observations.source
        frame = observations.value.loc[observations.value["date"] <= end]
        frame = frame.assign(source=observations.source)

        _update_job(engine, job_id, progress=0.8, message="Writing lake")
        lake.write_series(code, frame)
        result.rows_written = len(frame)
        with engine.begin() as conn:
            _record_catalog(
                conn, SERIES_DATASET, code, lake.series_coverage(code), observations.source
            )
        return _finish(engine, result, "succeeded", f"{len(frame)} observations written")
    except NoProviderError as exc:
        return _provider_unavailable(engine, result, exc, requeue)
    except ProviderError as exc:
        return _finish(engine, result, "failed", "Provider error", error=str(exc))


def _provider_unavailable(
    engine: Engine, result: IngestResult, exc: NoProviderError, requeue: bool
) -> IngestResult:
    """Throttling only → `queued` when the caller will retry (worker), so the job stays live."""
    result.retryable = exc.rate_limited
    if result.retryable and requeue:
        result.status = "queued"
        result.message = f"Rate limited, retry scheduled ({exc})"
        _update_job(engine, result.job_id, status="queued", message=result.message)
        return result
    return _finish(engine, result, "failed", str(exc), error=str(exc))


def _finish(
    engine: Engine, result: IngestResult, status: str, message: str, error: str | None = None
) -> IngestResult:
    result.status = status
    result.message = message
    _update_job(
        engine,
        result.job_id,
        status=status,
        progress=1.0,
        message=message,
        provider=result.source,
        rows_written=result.rows_written,
        qc_summary=result.qc or None,
        error=error,
        finished_at=datetime.now(UTC),
    )
    return result
