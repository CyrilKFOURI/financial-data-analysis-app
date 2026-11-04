import asyncio
import json
from typing import Any

from fastapi.testclient import TestClient

from fda_api.cache import cache_key

SUBJECT = {"type": "asset", "id": "AAPL"}


def run(
    client: TestClient, analyses: list[dict[str, Any]], subject: dict[str, str] = SUBJECT
) -> list[dict[str, Any]]:
    response = client.post("/v1/analyses:run", json={"subject": subject, "analyses": analyses})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    return [json.loads(line) for line in response.text.splitlines() if line]


def test_catalogue_lists_dev_analysis_with_tool_metadata(client: TestClient) -> None:
    analyses = {a["id"]: a for a in client.get("/v1/analyses").json()}

    dev = analyses["DEV-001"]
    assert dev["tool_name"] == "dev_synthetic_statistics"
    assert dev["diagnostic"] is True
    assert set(dev["params_schema"]["properties"]) == {"n", "seed", "mu", "sigma"}


def test_second_identical_run_is_a_cache_hit(client: TestClient) -> None:
    [first] = run(client, [{"id": "DEV-001", "params": {"seed": 3}}])
    [second] = run(client, [{"id": "DEV-001", "params": {"seed": 3}}])

    assert first["result"]["cache"] == "miss"
    assert second["result"]["cache"] == "hit"
    assert second["result"]["payload"] == first["result"]["payload"]
    assert first["result"]["params"] == {"n": 252, "seed": 3, "mu": 0.08, "sigma": 0.2}


def test_errors_are_reported_per_line_without_failing_the_batch(client: TestClient) -> None:
    lines = run(
        client,
        [
            {"id": "X-404"},
            {"id": "DEV-001", "params": {"n": 5}},
            {"id": "DEV-001"},
        ],
    )

    assert lines[0]["error"]["type"] == "/errors/unknown-analysis"
    assert lines[1]["error"]["type"] == "/errors/invalid-params"
    assert lines[2]["result"]["analysis_id"] == "DEV-001"


def test_subject_type_outside_applicability_is_rejected(client: TestClient) -> None:
    [line] = run(client, [{"id": "DEV-001"}], subject={"type": "comparison", "id": "c1"})

    assert line["error"]["type"] == "/errors/not-applicable"


def test_cache_key_ignores_parameter_order() -> None:
    common = {
        "analysis_id": "DEV-001",
        "analysis_version": "1.0.0",
        "subject": SUBJECT,
        "context": {"range": "1Y"},
        "data_version": "v1",
    }
    assert cache_key(params={"a": 1, "b": 2}, **common) == cache_key(
        params={"b": 2, "a": 1}, **common
    )
    assert cache_key(params={"a": 1}, **common) != cache_key(params={"a": 2}, **common)


def test_memory_cache_is_bounded_lru() -> None:
    from fda_api.cache import MemoryCache

    async def scenario() -> None:
        cache = MemoryCache(max_entries=2)
        await cache.set("a", b"1")
        await cache.set("b", b"2")
        assert await cache.get("a") == b"1"  # touch a → b becomes least recent
        await cache.set("c", b"3")
        assert await cache.get("b") is None
        assert await cache.get("a") == b"1"
        assert await cache.get("c") == b"3"

    asyncio.run(scenario())


def test_context_fields_are_bounded(client: TestClient) -> None:
    response = client.post(
        "/v1/analyses:run",
        json={"subject": SUBJECT, "context": {"bench": "x" * 500}, "analyses": [{"id": "DEV-001"}]},
    )
    assert response.status_code == 422


def _run(client: TestClient, symbol: str, ids: list[str], range_: str = "1Y") -> list[dict]:
    response = client.post(
        "/v1/analyses:run",
        json={
            "subject": {"type": "asset", "id": symbol},
            "context": {"range": range_},
            "analyses": [{"id": analysis_id} for analysis_id in ids],
        },
    )
    return [json.loads(line) for line in response.text.splitlines()]


def test_price_analyses_run_on_stored_prices_with_type_conventions(client: TestClient) -> None:
    client.post("/v1/ingest/FAKE")
    ids = ["T-001", "T-002", "T-003", "T-005", "T-007", "T-011", "T-012", "T-013", "T-030"]
    lines = _run(client, "FAKE", ids)
    assert [line.get("result", {}).get("analysis_id") for line in lines] == ids, lines
    vol = lines[4]["result"]
    assert 0.1 < vol["payload"]["value"] < 0.4  # FakeProvider σ = 22 %
    assert vol["payload"]["annualisation"] == 252
    assert ":equity/synthetic:" in vol["data_version"]

    again = _run(client, "FAKE", ["T-007"])[0]["result"]
    assert again["cache"] == "hit"


def test_price_analysis_on_a_symbol_without_prices_is_a_clean_problem(client: TestClient) -> None:
    [line] = _run(client, "AAPL", ["T-007"])
    assert line["error"]["type"] == "/errors/insufficient-data"
    assert "ingest it first" in line["error"]["detail"]


def test_time_series_lab_runs_through_the_api_and_serialises(client: TestClient) -> None:
    client.post("/v1/ingest/FAKE")
    ids = ["TS-001", "TS-002", "TS-005", "TS-006", "TS-009", "TS-013", "TS-014", "TS-017", "TS-022"]
    lines = _run(client, "FAKE", ids, range_="3Y")
    assert [line.get("result", {}).get("analysis_id") for line in lines] == ids, lines
    adf = lines[0]["result"]["payload"]["tests"][0]
    assert adf["test"] == "adf"
    assert isinstance(adf["reject"], bool)


def test_pair_analysis_without_a_second_asset_explains_itself(client: TestClient) -> None:
    client.post("/v1/ingest/FAKE")
    [line] = _run(client, "FAKE", ["TS-018"])
    assert "second asset" in line["error"]["detail"]
    client.post("/v1/ingest/FAKEB")
    response = client.post(
        "/v1/analyses:run",
        json={
            "subject": {"type": "asset", "id": "FAKE"},
            "analyses": [{"id": "TS-018", "params": {"other": "FAKEB"}}],
        },
    )
    result = json.loads(response.text.splitlines()[0])["result"]
    assert result["payload"]["other"] == "FAKEB"
    assert "|FAKEB:" in result["data_version"]


def test_tiered_cache_serves_l1_and_bounds_bytes() -> None:
    from fda_api.cache import MemoryCache

    l1 = MemoryCache(max_entries=10, max_bytes=10)
    asyncio.run(l1.set("a", b"12345"))
    asyncio.run(l1.set("b", b"12345"))
    asyncio.run(l1.set("c", b"1"))  # 11 bytes > 10: the oldest entry is evicted
    assert asyncio.run(l1.get("a")) is None
    assert asyncio.run(l1.get("b")) == b"12345"
