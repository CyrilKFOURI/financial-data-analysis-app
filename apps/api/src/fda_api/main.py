import logging
import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from redis.exceptions import RedisError

from fda_api import __version__
from fda_api.agent import router as agent_router
from fda_api.analyses import router as analyses_router
from fda_api.auth import authenticate, ensure_local_user, sharing_router
from fda_api.auth import router as auth_router
from fda_api.cache import ResultCache, connect_cache
from fda_api.data import router as data_router
from fda_api.filings import router as filings_router
from fda_api.invest import router as invest_router
from fda_api.mcp import router as mcp_router
from fda_api.news import router as news_router
from fda_api.paper import router as paper_router
from fda_api.portfolios import router as portfolios_router
from fda_api.schedules import maybe_scheduler
from fda_api.schedules import router as schedules_router
from fda_api.security import require_client_header
from fda_api.strategies import router as strategies_router
from fda_api.tools import router as tools_router
from fda_api.universes import router as universes_router
from fda_api.workbench import router as workbench_router
from fda_api.workflows import router as workflows_router
from fda_data import scope
from fda_data.db import lake_root, make_engine
from fda_data.ingest import default_chain, default_series_chain, network_enabled
from fda_data.lake import PriceLake
from fda_data.providers.coingecko import CoinGeckoProvider

logger = logging.getLogger(__name__)


async def connect_job_queue(redis_url: str) -> ArqRedis | None:
    """arq pool when Redis is reachable; otherwise jobs run inline in the API process."""
    if not redis_url or os.getenv("FDA_JOB_RUNNER") == "inline":
        return None
    try:
        settings = RedisSettings.from_dsn(redis_url)
        settings.conn_timeout = 1
        settings.conn_retries = 0
        return await create_pool(settings)
    except (RedisError, OSError):
        logger.warning("Job queue unavailable; ingestion jobs will run inline")
        return None


def _warm_providers(chain: Any) -> None:
    try:
        chain.profile("SPY")
    except Exception:  # warming is best effort; real requests report their own errors
        logger.info("Provider warm-up failed", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    cache: ResultCache = await connect_cache(redis_url)
    app.state.cache = cache
    app.state.engine = make_engine()
    ensure_local_user(app.state.engine)
    app.state.lake = PriceLake(lake_root())
    app.state.chain = default_chain()
    app.state.series_chain = default_series_chain()
    app.state.coingecko = CoinGeckoProvider() if network_enabled() else None
    app.state.arq = await connect_job_queue(redis_url)
    if network_enabled():
        # yfinance opens its session (cookie + crumb) on first use, ~10 s: do it now, off-path.
        threading.Thread(target=_warm_providers, args=(app.state.chain,), daemon=True).start()
    async with maybe_scheduler(app.state):
        yield
    if app.state.arq is not None:
        await app.state.arq.aclose()
    app.state.engine.dispose()
    await cache.close()


scope.install()  # every query of a signed-in user is restricted to their rows (M31 §2)
app = FastAPI(title="Financial Data Analysis Workbench API", version=__version__, lifespan=lifespan)
# the last registered middleware runs first: cross-site guard, then authentication
app.middleware("http")(authenticate)
app.middleware("http")(require_client_header)


@app.exception_handler(scope.ScopeError)
def _scope_error(request: Request, exc: scope.ScopeError) -> JSONResponse:
    """Someone else's row is reported as missing, never as forbidden (no existence leak)."""
    return JSONResponse(
        status_code=404,
        media_type="application/problem+json",
        content={"type": "about:blank", "title": "Not found", "status": 404, "detail": str(exc)},
    )


app.include_router(auth_router)
app.include_router(sharing_router())
app.include_router(analyses_router)
app.include_router(data_router)
app.include_router(portfolios_router)
app.include_router(workbench_router)
app.include_router(universes_router)
app.include_router(invest_router)
app.include_router(strategies_router)
app.include_router(paper_router)
app.include_router(tools_router)
app.include_router(mcp_router)
app.include_router(filings_router)
app.include_router(news_router)
app.include_router(schedules_router)
app.include_router(workflows_router)
app.include_router(agent_router)


class Health(BaseModel):
    status: str
    version: str
    cache: str
    jobs: str


@app.get("/v1/health")
def health(request: Request) -> Health:
    return Health(
        status="ok",
        version=__version__,
        cache=request.app.state.cache.backend,
        jobs="worker" if request.app.state.arq is not None else "inline",
    )
