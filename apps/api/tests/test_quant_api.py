"""J12 endpoints offline: quant analyses resolve their inputs or explain what is missing."""

import json
from typing import Any

from fastapi.testclient import TestClient


def _run(
    client: TestClient, kind: str, subject: str, items: list[tuple[str, dict[str, Any]]]
) -> list[dict[str, Any]]:
    r = client.post(
        "/v1/analyses:run",
        json={
            "subject": {"type": kind, "id": subject},
            "context": {"range": "MAX"},
            "analyses": [{"id": i, "params": p} for i, p in items],
        },
    )
    return [json.loads(line) for line in r.text.splitlines()]


def test_quant_analyses_on_an_asset_and_a_portfolio(client: TestClient) -> None:
    for s in ("FAKE", "FAKEB", "FAKEC"):
        client.post(f"/v1/ingest/{s}")
    lines = _run(
        client,
        "asset",
        "FAKE",
        [
            ("Q-001", {"regressors": "FAKEB,FAKEC", "frequency": "W"}),
            ("Q-010", {}),
            ("Q-011", {}),
            ("Q-015", {"other": "FAKEB"}),
            ("Q-016", {"other": "FAKEB", "events": "2024-06-03,2024-09-02"}),
        ],
    )
    reg, ff, macro, cop, event = lines
    assert "result" in reg, reg
    terms = [c["term"] for c in reg["result"]["payload"]["coefficients"]]
    assert terms == ["const", "FAKEB", "FAKEC"]
    assert "Ken French" in ff["error"]["detail"]
    assert "Ingest macro series" in macro["error"]["detail"]
    assert "result" in cop, cop
    assert "result" in event or "window" in event["error"]["detail"], event

    pf = client.post(
        "/v1/portfolios",
        json={
            "name": "Three",
            "holdings": [{"symbol": s, "weight": 1 / 3} for s in ("FAKE", "FAKEB", "FAKEC")],
        },
    ).json()
    pca = _run(client, "portfolio", str(pf["id"]), [("Q-009", {})])[0]
    assert "result" in pca, pca
    assert pca["result"]["payload"]["n_series"] == 3


def test_factor_ingestion_queues_seven_series(client: TestClient) -> None:
    r = client.post("/v1/factors/EU/ingest")
    assert r.status_code == 202
    assert len(r.json()) == 7
    assert client.post("/v1/factors/JP/ingest").status_code == 422
