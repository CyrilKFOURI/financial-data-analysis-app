"""M30 §5 endpoints: SEC filings index and cached documents (AI-004 reads them)."""

from typing import Any

import anyio
from fastapi import APIRouter, HTTPException, Path, Query, Request

from fda_api.data import SYMBOL_PATTERN
from fda_data.filings import ingest_filings, read_filings
from fda_data.lake import PriceLake
from fda_data.providers.base import ProviderError, SymbolNotFoundError

router = APIRouter(prefix="/v1/filings", tags=["filings"])


def _lake(request: Request) -> PriceLake:
    lake: PriceLake = request.app.state.lake
    return lake


@router.get("/{symbol}")
def get_filings(request: Request, symbol: str = Path(pattern=SYMBOL_PATTERN)) -> dict[str, Any]:
    data = read_filings(_lake(request), symbol)
    if data is None:
        raise HTTPException(404, f"No filings stored for {symbol.upper()}: POST …/ingest first")
    return data | {
        "documents": [{k: v for k, v in d.items() if k != "text"} for d in data["documents"]]
    }


@router.post("/{symbol}/ingest")
async def post_ingest(
    request: Request,
    symbol: str = Path(pattern=SYMBOL_PATTERN),
    count: int = Query(2, ge=1, le=6, description="Latest annual reports to download"),
) -> dict[str, Any]:
    """EDGAR index + the latest annual reports (cache-first; SEC ≤ 8 req/s, owner User-Agent)."""
    try:
        return await anyio.to_thread.run_sync(
            lambda: ingest_filings(_lake(request), symbol, count=count)
        )
    except SymbolNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(502, str(exc)) from exc
