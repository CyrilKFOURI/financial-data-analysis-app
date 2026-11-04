"""J15 endpoints offline: synthetic chain → every DV analysis resolves or explains itself."""

import json
from typing import Any

from fastapi.testclient import TestClient

ALL = [
    "DV-001",
    "DV-003",
    "DV-004",
    "DV-005",
    "DV-006",
    "DV-007",
    "DV-009",
    "DV-010",
    "DV-011",
    "DV-012",
]


def _run(
    client: TestClient, items: list[tuple[str, dict[str, Any]]], symbol: str = "FAKEO"
) -> dict[str, Any]:
    r = client.post(
        "/v1/analyses:run",
        json={
            "subject": {"type": "asset", "id": symbol},
            "context": {"range": "1Y"},
            "analyses": [{"id": i, "params": p} for i, p in items],
        },
    )
    out = {}
    for line in r.text.splitlines():
        x = json.loads(line)
        body = x.get("result") or x.get("error")
        out[body["analysis_id"]] = x
    return out


def test_options_flow(client: TestClient) -> None:
    client.post("/v1/ingest/FAKEO")
    before = _run(client, [("DV-001", {})])
    assert "ingest options" in before["DV-001"]["error"]["detail"]
    job = client.post("/v1/options/FAKEO/ingest")
    assert job.status_code == 202
    snaps = client.get("/v1/options/FAKEO/snapshots").json()
    assert len(snaps["dates"]) == 1
    assert snaps["iv_history"][0]["atm_iv_30d"] > 0

    results = _run(client, [(a, {}) for a in ALL])
    for aid in (
        "DV-001",
        "DV-003",
        "DV-005",
        "DV-006",
        "DV-007",
        "DV-009",
        "DV-010",
        "DV-011",
        "DV-012",
    ):
        assert "result" in results[aid], (aid, results[aid])
    chain = results["DV-001"]["result"]["payload"]
    assert chain["rows"]
    assert chain["rows"][0]["c_iv"] is not None
    surface = results["DV-005"]["result"]["payload"]
    assert surface["rmse_max"] < 0.01  # the synthetic chain is priced on an SVI smile
    assert results["DV-006"]["result"]["payload"]["history_source"].startswith("realised")
    rnd = results["DV-011"]["result"]["payload"]
    assert abs(rnd["mass"] - 1) < 0.05

    pricer = _run(
        client,
        [
            (
                "DV-002",
                {
                    "kind": "put",
                    "spot": 42,
                    "strike": 40,
                    "days": 182.5,
                    "rate": 0.1,
                    "volatility": 0.2,
                    "model": "bsm",
                },
            )
        ],
    )
    assert round(pricer["DV-002"]["result"]["payload"]["price"], 2) == 0.81
