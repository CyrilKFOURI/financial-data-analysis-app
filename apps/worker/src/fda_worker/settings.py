"""arq worker entry point: `uv run arq fda_worker.settings.WorkerSettings`."""

import asyncio
import os
from typing import Any, ClassVar

from arq import Retry
from arq.connections import RedisSettings

from fda_data.company import build_peer_group, ingest_company_data
from fda_data.db import lake_root, make_engine
from fda_data.fundamentals import ingest_fundamentals
from fda_data.ingest import (
    IngestResult,
    default_chain,
    default_series_chain,
    ingest_daily_prices,
    ingest_series,
    network_enabled,
)
from fda_data.lake import PriceLake
from fda_data.providers.coingecko import CoinGeckoProvider

MAX_TRIES = 3
RETRY_BASE_S = 60


async def ping(ctx: dict[str, Any]) -> str:
    return "pong"


def _outcome(ctx: dict[str, Any], result: IngestResult) -> dict[str, Any]:
    """Re-queue throttled jobs with linear back-off (M02 story 2); the last try fails for good."""
    if result.retryable and result.status == "queued":
        raise Retry(defer=RETRY_BASE_S * ctx.get("job_try", 1))
    return {"status": result.status, "rows_written": result.rows_written, "source": result.source}


async def ingest_prices(ctx: dict[str, Any], job_id: int, symbol: str) -> dict[str, Any]:
    """Runs the blocking ingestion in a thread; progress is written to `ingestion_jobs`."""
    result = await asyncio.to_thread(
        ingest_daily_prices,
        ctx["engine"],
        ctx["lake"],
        ctx["chain"],
        symbol,
        job_id,
        coingecko=ctx["coingecko"],
        requeue=ctx.get("job_try", 1) < MAX_TRIES,
    )
    return _outcome(ctx, result)


async def ingest_reference_series(ctx: dict[str, Any], job_id: int, code: str) -> dict[str, Any]:
    result = await asyncio.to_thread(
        ingest_series,
        ctx["engine"],
        ctx["lake"],
        ctx["series_chain"],
        code,
        job_id,
        requeue=ctx.get("job_try", 1) < MAX_TRIES,
    )
    return _outcome(ctx, result)


async def ingest_company_fundamentals(
    ctx: dict[str, Any], job_id: int, symbol: str
) -> dict[str, Any]:
    result = await asyncio.to_thread(ingest_fundamentals, ctx["engine"], symbol, job_id)
    return {"status": result.status, "rows_written": result.rows_written, "source": result.source}


async def ingest_company_snapshots(ctx: dict[str, Any], job_id: int, symbol: str) -> dict[str, Any]:
    result = await asyncio.to_thread(ingest_company_data, ctx["engine"], symbol, job_id)
    return {"status": result.status, "rows_written": result.rows_written}


async def build_peers(ctx: dict[str, Any], job_id: int, symbol: str) -> dict[str, Any]:
    result = await asyncio.to_thread(build_peer_group, ctx["engine"], symbol, job_id)
    return {"status": result.status, "rows_written": result.rows_written}


async def startup(ctx: dict[str, Any]) -> None:
    ctx["engine"] = make_engine()
    ctx["lake"] = PriceLake(lake_root())
    ctx["chain"] = default_chain()
    ctx["series_chain"] = default_series_chain()
    ctx["coingecko"] = CoinGeckoProvider() if network_enabled() else None


async def shutdown(ctx: dict[str, Any]) -> None:
    ctx["engine"].dispose()


class WorkerSettings:
    functions: ClassVar[list[Any]] = [
        ping,
        ingest_prices,
        ingest_reference_series,
        ingest_company_fundamentals,
        ingest_company_snapshots,
        build_peers,
    ]
    on_startup = startup
    on_shutdown = shutdown
    max_tries = MAX_TRIES
    redis_settings = RedisSettings.from_dsn(os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"))
