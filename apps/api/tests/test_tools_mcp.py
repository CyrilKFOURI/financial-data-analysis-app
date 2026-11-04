"""J24: tool generation from the registry, result shaping, BM25 tool search, MCP JSON-RPC over
HTTP (token, read-only mode), and the write-action approval gate (acceptance M30 §10.2: a write
tool call is never executed without an explicit approval event)."""

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fda_api.tools import all_tools, search, shape
from fda_core.registry import REGISTRY

TOKEN = "test-token-123"


def _refs(node: Any) -> list[str]:
    if isinstance(node, dict):
        own = [node["$ref"]] if "$ref" in node else []
        return own + [r for v in node.values() for r in _refs(v)]
    if isinstance(node, list):
        return [r for v in node for r in _refs(v)]
    return []


def test_every_analysis_is_a_tool_with_a_valid_schema() -> None:
    tools = all_tools()
    names = {t.name for t in tools}
    analyses = [s for s in REGISTRY.all() if not s.diagnostic]
    assert len(tools) >= len(analyses) + 10
    assert len(names) == len(tools)
    for tool in tools:
        assert len(tool.name) <= 64
        assert tool.name.replace("_", "").isalnum()
        schema = tool.input_schema
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        for ref in _refs(schema):
            assert ref.startswith("#/$defs/")
            assert ref.removeprefix("#/$defs/") in schema.get("$defs", {}), (tool.name, ref)
    assert not [t for t in tools if "live" in t.name and t.safety == "read" and "order" in t.name]


def test_search_and_shaping() -> None:
    hits = [h["analysis_id"] for h in search("value at risk", 5)]
    assert any(h and h.startswith("R-") for h in hits)
    assert search("R-001", 3)[0]["analysis_id"] == "R-001"
    shaped = shape(
        {
            "dates": [f"2024-01-{i:02d}" for i in range(1, 31)],
            "lines": {"x": list(range(30))},
            "rows": [{"a": i} for i in range(40)],
            "note": "y" * 2000,
            "v": float("nan"),
        }
    )
    assert shaped["dates"] == {"n": 30, "first": "2024-01-01", "last": "2024-01-30"}
    assert shaped["lines"]["x"]["max"] == 29
    assert shaped["rows"][-1] == {"truncated": 25}
    assert len(shaped["note"]) < 700
    assert shaped["v"] is None


def _mcp(
    client: TestClient,
    method: str,
    params: dict[str, Any] | None = None,
    rid: int = 1,
    token: str = TOKEN,
) -> Any:
    return client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}},
        headers={"Authorization": f"Bearer {token}"},
    )


def test_mcp_and_approval_gate(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    disabled = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
    assert disabled.status_code == 503
    monkeypatch.setenv("FDA_MCP_TOKEN", TOKEN)
    assert _mcp(client, "ping", token="wrong").status_code == 401
    init = _mcp(client, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}}).json()
    assert init["result"]["protocolVersion"] == "2025-06-18"
    assert "cite" in init["result"]["instructions"]
    note = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert note.status_code == 202
    listed = _mcp(client, "tools/list").json()["result"]["tools"]
    names = {t["name"] for t in listed}
    assert "place_paper_order" not in names  # read-only by default
    assert all(t["annotations"]["readOnlyHint"] for t in listed)

    client.post("/v1/ingest/FAKEMCP")
    tool = next(t.name for t in all_tools() if t.analysis_id == "T-001")
    call = _mcp(
        client,
        "tools/call",
        {
            "name": tool,
            "arguments": {"subject_type": "asset", "subject_id": "FAKEMCP", "range": "1Y"},
        },
    ).json()
    result = call["result"]
    assert result["isError"] is False
    content = json.loads(result["content"][0]["text"])
    assert len(content["result_id"]) == 12
    assert content["ui_ref"].endswith("/a/FAKEMCP/overview")
    bad = _mcp(
        client,
        "tools/call",
        {"name": tool, "arguments": {"subject_type": "asset", "subject_id": "NOPE"}},
    ).json()["result"]
    assert bad["isError"] is True
    assert _mcp(client, "tools/call", {"name": "place_paper_order", "arguments": {}}).json()[
        "error"
    ]

    monkeypatch.setenv("FDA_MCP_READ_ONLY", "0")
    aid = client.post("/v1/paper", json={"name": "Agent", "starting_cash": 10_000}).json()["id"]
    proposed = _mcp(
        client,
        "tools/call",
        {
            "name": "place_paper_order",
            "arguments": {"account_id": aid, "symbol": "FAKEMCP", "side": "buy", "quantity": 3},
        },
    ).json()
    action = json.loads(proposed["result"]["content"][0]["text"])
    assert action["status"] == "pending"
    assert client.get(f"/v1/paper/{aid}").json()["orders"] == []  # nothing executed
    # the MCP path cannot approve: approval is a web-app route behind the client header
    assert (
        client.post(
            f"/v1/actions/{action['action_id']}/approve", headers={"X-FDA-Client": ""}
        ).status_code
        == 403
    )
    approved = client.post(f"/v1/actions/{action['action_id']}/approve").json()
    assert approved["status"] == "executed"
    assert approved["result"]["order_id"] > 0
    assert len(client.get(f"/v1/paper/{aid}").json()["orders"]) == 1
    assert client.post(f"/v1/actions/{action['action_id']}/approve").status_code == 409
    status = _mcp(
        client,
        "tools/call",
        {"name": "get_action_status", "arguments": {"action_id": action["action_id"]}},
    ).json()
    assert json.loads(status["result"]["content"][0]["text"])["status"] == "executed"
    rejected = _mcp(
        client,
        "tools/call",
        {"name": "create_paper_account", "arguments": {"name": "X", "starting_cash": 5}},
    ).json()
    rid = json.loads(rejected["result"]["content"][0]["text"])["action_id"]
    assert client.post(f"/v1/actions/{rid}/reject").json()["status"] == "rejected"
    assert len(client.get("/v1/paper").json()) == 1
    audit = client.get("/v1/actions").json()
    assert {a["status"] for a in audit} == {"executed", "rejected"}
    assert all(a["origin"] == "mcp" for a in audit)


def test_direct_route_calls_validate_their_inputs(client: TestClient) -> None:
    for tool in ("get_data_status", "request_ingestion"):
        out = client.post(f"/v1/tools/{tool}:call", json={"arguments": {"symbol": "../../etc"}})
        assert out.json()["is_error"] is True
        assert "Invalid symbol" in out.json()["content"]["error"]
    empty = client.post("/v1/tools/search_instruments:call", json={"arguments": {"query": " "}})
    assert empty.json()["is_error"] is True
