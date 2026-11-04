"""AI-004 offline: synthetic 10-Ks → index, cached documents, sections and the YoY risk diff."""

import json

from fastapi.testclient import TestClient


def test_filings_flow(client: TestClient) -> None:
    assert client.get("/v1/filings/FAKEFL").status_code == 404
    before = client.post(
        "/v1/analyses:run",
        json={"subject": {"type": "asset", "id": "FAKEFL"}, "analyses": [{"id": "AI-004"}]},
    )
    assert "ingest" in json.loads(before.text.splitlines()[0])["error"]["detail"]
    first = client.post("/v1/filings/FAKEFL/ingest").json()
    assert first["fetched"] == 2
    assert client.post("/v1/filings/FAKEFL/ingest").json()["fetched"] == 0  # cache-first
    index = client.get("/v1/filings/FAKEFL").json()
    assert len(index["documents"]) == 2
    assert "text" not in index["documents"][0]
    assert client.post("/v1/filings/NOT_A_FAKE/ingest").status_code in (404, 422)

    run = client.post(
        "/v1/analyses:run",
        json={
            "subject": {"type": "asset", "id": "FAKEFL"},
            "analyses": [{"id": "AI-004", "params": {"section": "7"}}],
        },
    )
    payload = json.loads(run.text.splitlines()[0])["result"]["payload"]
    assert payload["document"]["filed"].startswith("2025")
    assert payload["compared_with"]["filed"].startswith("2024")
    assert {r["item"] for r in payload["sections"]} >= {"1", "1A", "7", "8"}
    assert payload["excerpt"].startswith("Revenue grew.")
    assert payload["risk_factors"] == 3
    assert payload["new_risks"] == 1
    assert payload["removed_risks"] == 1
    assert payload["changed_risks"] == 1
    assert payload["risk_diff"]["new"][0]["heading"].startswith("Cyberattacks")


def test_filings_paths_cannot_escape_the_lake(client: TestClient) -> None:
    for subject in ("../../secrets", r"..\..\x", "a/b"):
        run = client.post(
            "/v1/analyses:run",
            json={"subject": {"type": "asset", "id": subject}, "analyses": [{"id": "AI-004"}]},
        )
        line = json.loads(run.text.splitlines()[0])
        assert "error" in line
        assert "ingest" in line["error"]["detail"]
