from fastapi.testclient import TestClient


def test_health_reports_ok_version_and_backends(client: TestClient) -> None:
    response = client.get("/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": "0.1.0",
        "cache": "memory",
        "jobs": "inline",
    }


def test_unsafe_methods_require_the_client_header(client: TestClient) -> None:
    refused = client.post("/v1/ingest/FAKE", headers={"X-FDA-Client": ""})
    assert refused.status_code == 403
    assert refused.headers["content-type"].startswith("application/problem+json")
    assert client.get("/v1/data-status/FAKE", headers={"X-FDA-Client": ""}).status_code == 200


def test_cross_site_reads_are_refused(client: TestClient) -> None:
    response = client.get(
        "/v1/instruments", params={"q": "FAKE"}, headers={"Sec-Fetch-Site": "cross-site"}
    )
    assert response.status_code == 403
    assert (
        client.get(
            "/v1/instruments", params={"q": "FAKE"}, headers={"Sec-Fetch-Site": "same-origin"}
        ).status_code
        == 200
    )
