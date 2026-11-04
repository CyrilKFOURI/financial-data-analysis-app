"""MCP stdio bridge for Claude Desktop / Claude Code: reads JSON-RPC messages line by line on
stdin, forwards each to the running API's `/mcp` endpoint with the bearer token, and writes the
answers on stdout. Standard library only; logs go to stderr.

    FDA_API_URL=http://127.0.0.1:8000 FDA_MCP_TOKEN=… uv run python -m fda_api.mcp_stdio
"""

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


def forward(url: str, token: str, message: dict[str, Any]) -> dict[str, Any] | None:
    data = json.dumps(message).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        return _error(message.get("id"), f"API error {exc.code}: {detail}")
    except urllib.error.URLError as exc:
        return _error(message.get("id"), f"API not reachable at {url}: {exc.reason}")
    if not body:
        return None
    answer: dict[str, Any] = json.loads(body)
    return answer


def _error(rid: Any, text: str) -> dict[str, Any] | None:
    if rid is None:
        return None
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32000, "message": text}}


def main() -> None:
    url = os.environ.get("FDA_API_URL", "http://127.0.0.1:8000").rstrip("/") + "/mcp"
    token = os.environ.get("FDA_MCP_TOKEN", "")
    if not token:
        print("FDA_MCP_TOKEN is not set", file=sys.stderr)
        sys.exit(1)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            print("ignored a non-JSON line", file=sys.stderr)
            continue
        answer = forward(url, token, message)
        if answer is not None:
            sys.stdout.write(json.dumps(answer) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
