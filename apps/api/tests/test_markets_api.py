"""J13 endpoints offline: the market subject resolves stored inputs or explains the ingest."""

import json

from fastapi.testclient import TestClient


def test_market_subject_explains_missing_data_and_ingest_queues_everything(
    client: TestClient,
) -> None:
    r = client.post(
        "/v1/analyses:run",
        json={
            "subject": {"type": "market", "id": "global"},
            "context": {"range": "1Y"},
            "analyses": [{"id": "MK-001", "params": {}}, {"id": "MK-004", "params": {}}],
        },
    )
    lines = [json.loads(line) for line in r.text.splitlines()]
    assert all("markets/ingest" in line["error"]["detail"] for line in lines), lines
    asset = client.post(
        "/v1/analyses:run",
        json={
            "subject": {"type": "asset", "id": "FAKE"},
            "context": {"range": "1Y"},
            "analyses": [{"id": "MK-001", "params": {}}],
        },
    )
    assert "applies to" in json.loads(asset.text.splitlines()[0])["error"]["detail"]
    jobs = client.post("/v1/markets/ingest").json()
    assert len(jobs["prices"]) > 50
    assert len(jobs["series"]) > 20
