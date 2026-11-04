import json

from fastapi.testclient import TestClient


def test_fake_symbol_is_ingested_end_to_end(client: TestClient) -> None:
    before = client.get("/v1/data-status/FAKE").json()
    assert before["instrument"] is None
    assert before["can_ingest"] is True

    accepted = client.post("/v1/ingest/fake")
    assert accepted.status_code == 202
    assert accepted.json()["runner"] == "inline"

    status = client.get("/v1/data-status/FAKE").json()
    [dataset] = status["datasets"]
    assert dataset["dataset"] == "prices_daily"
    assert dataset["source"] == "fake"
    assert dataset["rows"] > 2_400
    assert status["jobs"][0]["status"] == "succeeded"
    assert status["instrument"]["asset_class"] == "equity"

    prices = client.get("/v1/instruments/FAKE/prices", params={"start": "2024-01-01"}).json()
    assert prices["source"] == "fake"
    assert len(prices["dates"]) == len(prices["close"]) == len(prices["volume"])
    assert prices["dates"][0] >= "2024-01-01"


def test_job_events_stream_ends_on_terminal_status(client: TestClient) -> None:
    job_id = client.post("/v1/ingest/FAKE").json()["job_id"]

    with client.stream("GET", f"/v1/jobs/{job_id}/events") as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        events = [
            json.loads(line.removeprefix("data: "))
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]
    assert events[-1]["status"] == "succeeded"
    assert events[-1]["progress"] == 1.0


def test_symbol_without_a_configured_provider_is_refused(client: TestClient) -> None:
    status = client.get("/v1/data-status/AAPL").json()
    assert status["can_ingest"] is False
    assert "No configured provider" in status["reason"]

    client.post("/v1/ingest/AAPL")
    after = client.get("/v1/data-status/AAPL").json()
    assert after["jobs"][0]["status"] == "failed"
    assert after["instrument"] is None
    assert client.get("/v1/instruments/AAPL/prices").status_code == 404


def test_search_lists_provider_candidates_then_ingested_instruments(client: TestClient) -> None:
    [candidate] = client.get("/v1/instruments", params={"q": "fake"}).json()
    assert candidate["ingested"] is False
    assert candidate["source"] == "fake"

    client.post("/v1/ingest/FAKE")
    [stored] = client.get("/v1/instruments", params={"q": "FAKE"}).json()
    assert stored["ingested"] is True


def test_symbols_are_validated(client: TestClient) -> None:
    assert client.get("/v1/data-status/bad%20symbol").status_code == 422


def test_reference_series_ingest_status_and_read(client: TestClient) -> None:
    before = client.get("/v1/series-status/fakerate").json()
    assert before["can_ingest"] is True
    assert before["dataset"] is None

    assert client.post("/v1/series/FAKERATE/ingest").status_code == 202
    status = client.get("/v1/series-status/FAKERATE").json()
    assert status["dataset"]["dataset"] == "series"
    assert status["jobs"][0]["status"] == "succeeded"

    series = client.get("/v1/series/FAKERATE", params={"start": "2024-01-01"}).json()
    assert series["unit"] == "percent"
    assert series["source"] == "fake"
    assert len(series["dates"]) == len(series["values"]) > 100


def test_unknown_series_is_refused_and_unread(client: TestClient) -> None:
    assert client.get("/v1/series-status/DGS10").json()["can_ingest"] is False
    assert client.get("/v1/series/DGS10").status_code == 404
    assert client.get("/v1/series/bad-code").status_code == 422


def test_actions_endpoint_lists_stored_actions(client: TestClient) -> None:
    assert client.get("/v1/instruments/FAKE/actions").status_code == 404
    client.post("/v1/ingest/FAKE")
    assert client.get("/v1/instruments/FAKE/actions").json() == []


def test_asset_type_conventions_and_overrides(client: TestClient) -> None:
    assert client.get("/v1/instruments/FAKE/type").status_code == 404
    client.post("/v1/ingest/FAKE")
    asset_type = client.get("/v1/instruments/FAKE/type").json()
    assert (asset_type["asset_class"], asset_type["asset_subclass"]) == ("equity", "synthetic")
    assert asset_type["conventions"]["annualisation"] == 252
    assert asset_type["conventions"]["price_field"] == "adj_close"

    fixed = client.put(
        "/v1/instruments/FAKE/overrides",
        json={"field": "asset_subclass", "value": "reit", "reason": "test"},
    ).json()
    assert fixed["asset_subclass"] == "reit"
    assert fixed["source"] == "override"
    assert fixed["overrides"] == {"asset_subclass": "reit"}

    refused = client.put(
        "/v1/instruments/FAKE/overrides", json={"field": "asset_subclass", "value": "stablecoin"}
    )
    assert refused.status_code == 422
    assert client.get("/v1/instruments/FAKE/type").json()["asset_subclass"] == "reit"

    restored = client.delete("/v1/instruments/FAKE/overrides/asset_subclass").json()
    assert restored["asset_subclass"] == "synthetic"
    assert restored["overrides"] == {}
