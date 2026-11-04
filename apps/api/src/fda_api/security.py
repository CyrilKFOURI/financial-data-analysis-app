"""Cross-site request guard for state-changing endpoints.

A browser can send a cross-site form POST without asking the API first. Requiring a custom
header on every unsafe method turns any cross-site attempt into a CORS preflight, which this
API never approves, so only our own client (same origin, via the Vite proxy) can trigger work
such as ingestion jobs that spend provider quotas. Reads that may call providers (search) are
covered too: browsers mark cross-site requests with `Sec-Fetch-Site`, which we refuse.
"""

from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse

CLIENT_HEADER = "X-FDA-Client"
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
MCP_PATH = "/mcp"


async def require_client_header(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    if request.headers.get("Sec-Fetch-Site") == "cross-site":
        return _forbidden("Cross-site requests are not accepted.")
    if request.url.path == MCP_PATH:
        return await call_next(request)  # the MCP endpoint enforces its own bearer token
    if request.method in SAFE_METHODS or request.headers.get(CLIENT_HEADER):
        return await call_next(request)
    return _forbidden(f"State-changing requests must carry the {CLIENT_HEADER} header.")


def _forbidden(detail: str) -> Response:
    return JSONResponse(
        status_code=403,
        media_type="application/problem+json",
        content={
            "type": "https://fda.local/problems/missing-client-header",
            "title": "Request refused by the cross-site guard",
            "status": 403,
            "detail": detail,
        },
    )
