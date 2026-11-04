"""J17 endpoints offline: every optimiser on a synthetic 4-holding portfolio, validation gate."""

import json
from typing import Any

from fastapi.testclient import TestClient

SYMBOLS = ["FAKEW", "FAKEX", "FAKEY", "FAKEZ"]


def _run(client: TestClient, pid: int, items: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    r = client.post(
        "/v1/analyses:run",
        json={
            "subject": {"type": "portfolio", "id": str(pid)},
            "context": {"range": "3Y"},
            "analyses": [{"id": i, "params": p} for i, p in items],
        },
    )
    out = {}
    for line in r.text.splitlines():
        x = json.loads(line)
        body = x.get("result") or x.get("error")
        out[f"{body['analysis_id']}:{json.dumps(body.get('params', {}), sort_keys=True)}"] = x
        out[body["analysis_id"]] = x
    return out


def test_optimisers_and_validation(client: TestClient) -> None:
    for s in SYMBOLS:
        client.post(f"/v1/ingest/{s}")
    pf = client.post(
        "/v1/portfolios",
        json={"name": "Opt", "holdings": [{"symbol": s, "weight": 0.25} for s in SYMBOLS]},
    ).json()
    items = [
        ("IN-010", {"method": "inverse_vol"}),
        ("IN-011", {"method": "gmv"}),
        ("IN-012", {"views": [{"asset": "FAKEW", "annual_return": 0.1, "confidence": 0.6}]}),
        ("IN-013", {}),
        ("IN-014", {}),
        ("IN-015", {"method": "min_cvar"}),
        ("IN-016", {"method": "robust"}),
    ]
    out = _run(client, pf["id"], items)
    for aid, _ in items:
        assert "result" in out[aid], (aid, out[aid])
        payload = out[aid]["result"]["payload"]
        if "rows" in payload and aid != "IN-012":
            assert abs(sum(r["weight"] for r in payload["rows"]) - 1) < 1e-8
            assert payload["validated"] is False
    assert len(out["IN-011"]["result"]["payload"]["frontier"]) >= 5
    assert out["IN-014"]["result"]["payload"]["leaf_order"]
    validation = _run(client, pf["id"], [("IN-018", {"method": "gmv", "lookback": 252})])["IN-018"]
    assert "result" in validation, validation
    v = validation["result"]["payload"]
    assert v["validated"] is True
    assert {r["method"] for r in v["rows"]} >= {
        "gmv",
        "equal",
        "inverse_vol",
        "hrp",
        "current (held)",
    }
    assert 0 <= v["deflated_sharpe"] <= 1
