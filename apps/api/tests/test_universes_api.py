"""J14 endpoints offline: custom universe → ingest → snapshot → screens, ranks, breadth."""

import json
from typing import Any

from fastapi.testclient import TestClient

SYMBOLS = ["FAKEA", "FAKEB", "FAKEC", "FAKED", "FAKEE", "FAKEF"]


def _run(
    client: TestClient, uid: int, items: list[tuple[str, dict[str, Any]]]
) -> list[dict[str, Any]]:
    r = client.post(
        "/v1/analyses:run",
        json={
            "subject": {"type": "universe", "id": str(uid)},
            "context": {"range": "1Y"},
            "analyses": [{"id": i, "params": p} for i, p in items],
        },
    )
    return [json.loads(line) for line in r.text.splitlines()]


def test_universe_flow(client: TestClient) -> None:
    created = client.post(
        "/v1/universes",
        json={"name": "Fakes", "symbols": "\n".join(f"{s};Co {s};Tech" for s in SYMBOLS)},
    )
    assert created.status_code == 201, created.text
    uid = created.json()["id"]
    assert client.post("/v1/universes", json={"name": "Fakes", "symbols": "X"}).status_code == 422
    assert client.post(f"/v1/universes/{uid}/screen", json={}).status_code == 409  # no snapshot yet
    queued = client.post(f"/v1/universes/{uid}/ingest").json()
    assert (queued["prices"], queued["fundamentals"]) == (6, 0)
    assert queued["snapshot_job"] is not None  # chained after the ingestion tasks
    listed = client.get("/v1/universes").json()
    assert [(u["name"], u["members"], len(u["snapshots"])) for u in listed] == [("Fakes", 6, 1)]
    detail = client.get(f"/v1/universes/{uid}").json()
    assert len(detail["members"]) == 6
    assert len(detail["snapshots"]) == 1

    everyone = client.post(
        f"/v1/universes/{uid}/screen", json={"columns": ["return_1m", "volatility_1y"]}
    ).json()
    assert everyone["count"] == 6
    assert everyone["rows"][0]["sector"] == "Tech"
    half = client.post(
        f"/v1/universes/{uid}/screen",
        json={
            "filters": {
                "all": [{"metric": "return_1m", "op": ">", "value": 50, "basis": "percentile"}]
            }
        },
    ).json()
    assert half["count"] == 3

    saved = client.post(
        "/v1/screens",
        json={
            "name": "Top half",
            "universe_id": uid,
            "filters": {
                "all": [{"metric": "return_1m", "op": ">", "value": 50, "basis": "percentile"}]
            },
        },
    ).json()
    first = client.post(f"/v1/screens/{saved['id']}/run").json()
    assert first["diff"] is None
    second = client.post(f"/v1/screens/{saved['id']}/run").json()
    assert second["diff"] == {"entrants": [], "leavers": []}

    ranks, heat, dist, breadth = _run(
        client,
        uid,
        [
            ("SC-004", {}),
            ("SC-005", {"metric": "return_1m"}),
            ("SC-006", {"metric": "volatility_1y"}),
            ("MK-012", {}),
        ],
    )
    for line in (ranks, heat, dist, breadth):
        assert "result" in line, line
    assert len(ranks["result"]["payload"]["rows"]) == 6
    assert breadth["result"]["payload"]["members_with_prices"] == 6
    assert client.get("/v1/screener/metrics").json()["presets"]
