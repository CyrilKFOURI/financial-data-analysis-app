"""Neutral types shared by every provider adapter."""

import copy
import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx2 as httpx


class LLMError(RuntimeError):
    """A provider call failed; the message never contains the API key."""


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class Reply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop: str = ""


class Session(Protocol):
    """One conversation in the provider's own format (thinking blocks, signatures and call ids
    are kept exactly as the provider returned them and sent back unchanged)."""

    def send(
        self, user_text: str | None = None, results: list[tuple[ToolCall, str]] | None = None
    ) -> Reply: ...


RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504, 529})
BACKOFF = (2.0, 6.0, 15.0)


def post_json(
    url: str, headers: dict[str, str], body: dict[str, Any], timeout: float = 180.0
) -> dict[str, Any]:
    """POST with retries on overload / rate limits (429, 5xx), honouring Retry-After."""
    where = url.split("?")[0]
    for attempt in range(len(BACKOFF) + 1):
        try:
            response = httpx.post(url, headers=headers, json=body, timeout=timeout)
        except httpx.TimeoutException as exc:
            raise LLMError(f"{type(exc).__name__} calling {where}") from exc
        except httpx.HTTPError as exc:
            if attempt == len(BACKOFF):
                raise LLMError(f"{type(exc).__name__} calling {where}") from exc
            time.sleep(BACKOFF[attempt])
            continue
        if response.status_code in RETRY_STATUS and attempt < len(BACKOFF):
            wait = response.headers.get("retry-after", "")
            time.sleep(
                min(float(wait), 30.0) if wait.replace(".", "", 1).isdigit() else BACKOFF[attempt]
            )
            continue
        break
    if response.status_code >= 400:
        raise LLMError(f"HTTP {response.status_code} from {where}: {response.text[:400]}")
    try:
        data = response.json()
    except ValueError as exc:
        raise LLMError(f"Non-JSON answer from {where}") from exc
    if not isinstance(data, dict):
        raise LLMError("Unexpected answer shape")
    return data


def parse_arguments(raw: Any) -> dict[str, Any]:
    """Tool arguments arrive as an object or a JSON string (OpenAI); bad JSON → {}."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            value = json.loads(raw)
        except ValueError:
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve local `$ref`s into plain nested schemas (Gemini and small local models do not
    follow `$defs`)."""
    defs = schema.get("$defs", {})

    def walk(node: Any, depth: int = 0) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/") and depth < 8:
                target = copy.deepcopy(defs.get(ref.split("/")[-1], {}))
                merged = {**target, **{k: v for k, v in node.items() if k != "$ref"}}
                return walk(merged, depth + 1)
            return {k: walk(v, depth) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [walk(v, depth) for v in node]
        return node

    result = walk(schema)
    return result if isinstance(result, dict) else {"type": "object", "properties": {}}
