"""MCP server (M30 §2.2, AI-002): the tool layer over the Model Context Protocol.

Transport: JSON-RPC 2.0 over HTTP POST at `/mcp` (streamable-HTTP servers may answer a POST with
plain `application/json`, which is all this server does: no server-initiated messages). Bound to
the API's localhost address and protected by a bearer token (`FDA_MCP_TOKEN`; the endpoint is off
without it). Claude Desktop / Code use the stdio bridge `python -m fda_api.mcp_stdio`, which
forwards each line to this endpoint. Read-only mode (`FDA_MCP_READ_ONLY`, default on) hides the
write tools; when they are enabled they still only propose actions the user approves in the app.
"""

import hmac
import json
import os
from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from fda_api import __version__
from fda_api.tools import ToolError, call_tool, tools_by_name

router = APIRouter(tags=["mcp"])

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
INSTRUCTIONS = (
    "Financial Data Analysis Workbench tools. Numbers must come from tool results: cite each "
    "figure as [r:result_id]. Start with search_tools to find an analysis, explain_analysis for "
    "its parameters, get_data_status / request_ingestion for data. Analyses V1–V3 describe and "
    "measure (no buy/sell verdicts); V4–V5 outputs are labelled model outputs with their "
    "validation. Write tools only propose actions that the user approves in the workbench; there "
    "is no live-order tool."
)


def read_only() -> bool:
    return os.environ.get("FDA_MCP_READ_ONLY", "1").lower() not in ("0", "false", "no")


def _token_ok(request: Request) -> bool:
    expected = os.environ.get("FDA_MCP_TOKEN", "")
    given = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    return bool(expected) and hmac.compare_digest(given.encode(), expected.encode())


def _visible() -> list[Any]:
    return [t for t in tools_by_name().values() if not (read_only() and t.safety == "write")]


def tool_list() -> list[dict[str, Any]]:
    return [
        {
            "name": t.name,
            "title": t.title,
            "description": t.description,
            "inputSchema": t.input_schema,
            "annotations": {
                "title": t.title,
                "readOnlyHint": t.safety == "read",
                "destructiveHint": False,
                "idempotentHint": t.safety == "read",
                "openWorldHint": t.name == "request_ingestion",
            },
        }
        for t in _visible()
    ]


def _error(rid: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}


async def handle(request: Request, message: dict[str, Any]) -> dict[str, Any] | None:
    """One JSON-RPC message → response (None for notifications)."""
    rid = message.get("id")
    method = message.get("method")
    params = message.get("params") or {}
    if rid is None:
        return None  # notifications (initialized, cancelled) need no answer
    if method == "initialize":
        asked = params.get("protocolVersion")
        version = asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "result": {
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {
                    "name": "fda-workbench",
                    "title": "FDA Workbench",
                    "version": __version__,
                },
                "instructions": INSTRUCTIONS,
            },
        }
    if method == "ping":
        return {"jsonrpc": "2.0", "id": rid, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": tool_list()}}
    if method == "tools/call":
        name = str(params.get("name", ""))
        tool = tools_by_name().get(name)
        if tool is None or (read_only() and tool.safety == "write"):
            return _error(rid, -32602, f"Unknown tool {name!r}")
        try:
            out = await call_tool(request, name, params.get("arguments") or {}, "mcp")
        except ToolError as exc:
            out = {"content": {"error": str(exc)}, "is_error": True}
        text = json.dumps(out["content"], ensure_ascii=False, default=str)
        result: dict[str, Any] = {
            "content": [{"type": "text", "text": text}],
            "isError": out["is_error"],
        }
        if isinstance(out["content"], dict):
            result["structuredContent"] = json.loads(text)
        return {"jsonrpc": "2.0", "id": rid, "result": result}
    return _error(rid, -32601, f"Method not found: {method}")


@router.post("/mcp")
async def mcp_endpoint(request: Request) -> Response:
    if not os.environ.get("FDA_MCP_TOKEN"):
        return JSONResponse(
            {"detail": "MCP is disabled: set FDA_MCP_TOKEN in .env"}, status_code=503
        )
    if not _token_ok(request):
        return JSONResponse({"detail": "Invalid or missing bearer token"}, status_code=401)
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse(_error(None, -32700, "Parse error"), status_code=400)
    if isinstance(body, list):
        answers = [a for a in [await handle(request, m) for m in body if isinstance(m, dict)] if a]
        return JSONResponse(answers) if answers else Response(status_code=202)
    if not isinstance(body, dict):
        return JSONResponse(_error(None, -32600, "Invalid request"), status_code=400)
    answer = await handle(request, body)
    return JSONResponse(answer) if answer is not None else Response(status_code=202)
