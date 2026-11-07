"""J19 endpoints offline: hypothesis-first strategies, every TR analysis, the trial ledger."""

import json
from typing import Any

from fastapi.testclient import TestClient

SPEC = {
    "symbol": "FAKEBT",
    "template": "ema_cross",
    "grid": {"fast": [5, 10, 20], "slow": [50, 100]},
}
HYPOTHESIS = (
    "Trend persistence: medium-term momentum in broad equity indices (behavioural under-reaction)."
)


def _run(client: TestClient, sid: int, ids: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    r = client.post(
        "/v1/analyses:run",
        json={
            "subject": {"type": "strategy", "id": str(sid)},
            "context": {"range": "MAX"},
            "analyses": [{"id": i, "params": p} for i, p in ids],
        },
    )
    out = {}
    for line in r.text.splitlines():
        x = json.loads(line)
        out[(x.get("result") or x.get("error"))["analysis_id"]] = x
    return out


def test_strategy_lab(client: TestClient) -> None:
    client.post("/v1/ingest/FAKEBT")
    bad = client.post("/v1/strategies", json={"name": "x", "hypothesis": "too short", "spec": SPEC})
    assert bad.status_code == 422
    free = client.post(
        "/v1/strategies",
        json={
            "name": "y",
            "hypothesis": HYPOTHESIS,
            "spec": SPEC | {"costs": {"fee_bps": 0, "slippage_bps": 0}},
        },
    )
    assert free.status_code == 422  # costs are always on
    sid = client.post(
        "/v1/strategies", json={"name": "Trend", "hypothesis": HYPOTHESIS, "spec": SPEC}
    ).json()["id"]
    ids = [
        "TR-001",
        "TR-002",
        "TR-003",
        "TR-007",
        "TR-008",
        "TR-009",
        "TR-014",
        "TR-015",
        "TR-016",
        "TR-017",
        "TR-004",
        "TR-024",
        "TR-025",
    ]
    out = _run(client, sid, [(i, {}) for i in ids])
    for aid in ids:
        assert "result" in out[aid] or aid == "TR-017", (aid, out[aid])
    run = out["TR-001"]["result"]["payload"]
    assert run["validation"]["complete"] is True
    assert run["validation"]["trials_total"] == 6  # 3 × 2 configurations, all with fast < slow
    assert run["validation"]["lookahead_suspicious"] is False
    assert run["lines"]["strategy"]
    assert out["TR-014"]["result"]["payload"]["paths"] == 5
    assert len(out["TR-008"]["result"]["payload"]["rows"]) == 7
    robust = out["TR-024"]["result"]["payload"]
    assert [c["check"] for c in robust["checks"]][:3] == [
        "Plateau",
        "Trades a year",
        "Execution delay",
    ]
    assert {r["series"] for r in robust["epochs"]} >= {"strategy", "vol_targeted_hold"}
    # Same spec again: no new trials; a new grid adds its configurations to the ledger.
    _run(client, sid, [("TR-007", {})])
    assert client.get(f"/v1/strategies/{sid}").json()["trials"] == 6
    client.put(f"/v1/strategies/{sid}", json=SPEC | {"grid": {"fast": [5, 10], "slow": [30, 60]}})
    again = _run(client, sid, [("TR-007", {})])["TR-007"]["result"]["payload"]
    assert again["trials_total"] == 10


def test_strategy_builder_graph(client: TestClient) -> None:
    """Feature 321: the block catalogue, a template graph, its run, and the lab on the graph."""
    client.post("/v1/ingest/FAKEGR")
    catalogue = client.get("/v1/strategies/blocks").json()
    assert {"Candlestick", "PortfolioSimulation", "BelowCrossesAbove"} <= {
        b["type"] for b in catalogue
    }
    graph = client.get(
        "/v1/strategies/blocks/templates/ema_cross", params={"symbol": "FAKEGR"}
    ).json()
    assert client.post("/v1/strategies/blocks/validate", json={"graph": graph}).json() == {
        "problems": []
    }
    broken = {"blocks": graph["blocks"], "wires": graph["wires"][:-4]}
    assert client.post("/v1/strategies/blocks/validate", json={"graph": broken}).json()["problems"]
    bad = client.post(
        "/v1/strategies",
        json={
            "name": "Broken",
            "hypothesis": HYPOTHESIS,
            "spec": {"symbol": "FAKEGR", "template": "graph", "graph": broken},
        },
    )
    assert bad.status_code == 422
    sid = client.post(
        "/v1/strategies",
        json={
            "name": "EMA Cross",
            "hypothesis": HYPOTHESIS,
            "spec": {"symbol": "FAKEGR", "template": "graph", "graph": graph},
        },
    ).json()["id"]
    spec = client.get(f"/v1/strategies/{sid}").json()["spec"]
    assert spec["grid"] == {
        "ema_fast.Period": [8.0, 12.0, 16.0],
        "ema_slow.Period": [21.0, 26.0, 34.0],
    }
    out = _run(client, sid, [("TR-027", {}), ("TR-001", {}), ("TR-024", {})])
    failed = {k: v["error"]["detail"] for k, v in out.items() if "error" in v}
    assert not failed, failed
    summary = out["TR-027"]["result"]["payload"]["performance_summary"]
    assert summary["total_trades"] == summary["winning_trades"] + summary["losing_trades"]
    assert out["TR-001"]["result"]["payload"]["validation"]["trials_total"] == 9


def test_graph_preview_runs_an_unsaved_graph_on_an_asset(client: TestClient) -> None:
    client.post("/v1/ingest/FAKEGP")
    graph = client.get(
        "/v1/strategies/blocks/templates/rsi_bounce", params={"symbol": "FAKEGP"}
    ).json()
    response = client.post(
        "/v1/analyses:run",
        json={
            "subject": {"type": "asset", "id": "FAKEGP"},
            "context": {"range": "MAX"},
            "analyses": [{"id": "TR-028", "params": {"graph": graph}}],
        },
    )
    line = json.loads(response.text.splitlines()[0])
    assert "result" in line, line
    payload = line["result"]["payload"]
    assert payload["run_configuration"]["blocks"] == 9
    assert len(payload["dates"]) == len(payload["lines"]["equity"])
