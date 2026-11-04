"""J11 endpoints offline: comparisons, presets, reports, alerts, snapshots."""

import json
from typing import Any

from fastapi.testclient import TestClient

from fda_api.runner import value_at


def _run(
    client: TestClient, subject: str, items: list[tuple[str, dict[str, Any]]]
) -> list[dict[str, Any]]:
    r = client.post(
        "/v1/analyses:run",
        json={
            "subject": {"type": "comparison", "id": subject},
            "context": {"range": "MAX"},
            "analyses": [{"id": i, "params": p} for i, p in items],
        },
    )
    return [json.loads(line) for line in r.text.splitlines()]


def _seed(client: TestClient) -> int:
    for s in ("FAKE", "FAKEB", "FAKEC"):
        client.post(f"/v1/ingest/{s}")
    pf = client.post(
        "/v1/portfolios",
        json={
            "name": "Core",
            "holdings": [{"symbol": "FAKE", "weight": 0.5}, {"symbol": "FAKEB", "weight": 0.5}],
        },
    )
    return int(pf.json()["id"])


def test_adhoc_and_saved_comparisons(client: TestClient) -> None:
    pid = _seed(client)
    lines = _run(
        client,
        f"FAKE,FAKEC,portfolio:{pid}",
        [("CP-001", {}), ("CP-001", {"view": "overlay"}), ("CP-003", {})],
    )
    assert all("result" in line for line in lines), lines
    kpi = lines[0]["result"]["payload"]
    assert kpi["members"] == ["FAKE", "FAKEC", "Core"]
    assert "volatility" in kpi["grid"]["rows"]
    assert lines[2]["result"]["payload"]["focus"] == "Core"
    saved = client.post(
        "/v1/comparisons", json={"name": "Mine", "members": f"FAKE, portfolio:{pid}"}
    )
    assert saved.status_code == 201
    cid = saved.json()["id"]
    assert _run(client, str(cid), [("CP-001", {"view": "calendar"})])[0]["result"]["payload"][
        "grid"
    ]["rows"]
    assert (
        client.post("/v1/comparisons", json={"name": "Bad", "members": "FAKE"}).status_code == 422
    )
    assert [c["name"] for c in client.get("/v1/comparisons").json()] == ["Mine"]
    problem = _run(client, "FAKE,NOPE", [("CP-001", {})])[0]["error"]
    assert "ingest" in problem["detail"]


def test_presets_create_benchmark_portfolios_once(client: TestClient) -> None:
    presets = {p["key"]: p for p in client.get("/v1/benchmarks/presets").json()}
    assert presets["sp500"]["symbol"] == "SPY"
    single = client.post("/v1/benchmarks/presets/sp500").json()
    assert single["member"] == "asset:SPY"
    blend = client.post("/v1/benchmarks/presets/6040_usd").json()
    again = client.post("/v1/benchmarks/presets/6040_usd").json()
    assert blend["member"] == again["member"]
    detail = client.get(f"/v1/portfolios/{blend['portfolio_id']}").json()
    assert detail["is_benchmark"] is True
    assert {h["symbol"]: h["weight"] for h in detail["holdings"]} == {"SPY": 0.6, "AGG": 0.4}


def test_reports_render_sections_and_appendix(client: TestClient) -> None:
    pid = _seed(client)
    assert {t["key"] for t in client.get("/v1/reports/templates").json()} == {
        "company",
        "risk",
        "portfolio",
        "comparison",
        "ips",
    }
    risk = client.get("/v1/reports/risk", params={"subject": "FAKE", "range": "3Y"})
    assert risk.status_code == 200
    page = risk.text
    assert page.startswith("<!doctype html>")
    assert "Security risk sheet" in page
    assert "Sources, formulas" in page
    assert "<svg" in page
    assert "R-020" in page
    review = client.get("/v1/reports/portfolio", params={"subject": str(pid)}).text
    assert "Portfolio review" in review
    assert "P-009" in review
    company = client.get("/v1/reports/company", params={"subject": "FAKE"}).text
    assert "No statements stored" in company  # explained, not failed
    assert client.get("/v1/reports/nope", params={"subject": "FAKE"}).status_code == 404


def test_alert_fires_once_then_notification_is_read(client: TestClient) -> None:
    _seed(client)
    rule = {
        "name": "FAKE vol",
        "subject_type": "asset",
        "subject_id": "FAKE",
        "analysis_id": "T-007",
        "path": "value",
        "op": ">",
        "threshold": 0.0,
    }
    assert client.post("/v1/alerts", json=rule).status_code == 201
    never = rule | {"name": "never", "threshold": 1e9}
    client.post("/v1/alerts", json=never)
    first = client.post("/v1/alerts/evaluate").json()
    assert [r["fired"] for r in first] == [True, False]
    second = client.post("/v1/alerts/evaluate").json()
    assert [r["fired"] for r in second] == [False, False]  # still true: no new notification
    notes = client.get("/v1/notifications", params={"unread": True}).json()
    assert len(notes) == 1
    assert notes[0]["message"].startswith("FAKE vol: value =")
    assert client.post("/v1/notifications/read").json() == {"read": 1}
    assert client.get("/v1/notifications", params={"unread": True}).json() == []
    bad = rule | {"analysis_id": "X-999"}
    assert client.post("/v1/alerts", json=bad).status_code == 422


def test_snapshots_and_diff(client: TestClient) -> None:
    _seed(client)
    body = {
        "subject_type": "asset",
        "subject_id": "FAKE",
        "label": "Q1",
        "analyses": ["T-007", "R-020"],
        "context": {"range": "1Y"},
    }
    a = client.post("/v1/snapshots", json=body).json()
    b = client.post("/v1/snapshots", json=body | {"label": "Q2", "context": {"range": "3Y"}}).json()
    assert a["analyses"] == ["T-007", "R-020"]
    listed = client.get(
        "/v1/snapshots", params={"subject_type": "asset", "subject_id": "FAKE"}
    ).json()
    assert [s["label"] for s in listed] == ["Q2", "Q1"]
    diff = client.get("/v1/snapshots/diff", params={"a": a["id"], "b": b["id"]}).json()
    vol = next(c for c in diff["changes"] if c["analysis"] == "T-007" and c["metric"] == "value")
    assert vol["change"] == vol["after"] - vol["before"]


def test_value_paths() -> None:
    payload = {
        "rows": [{"symbol": "AAPL", "weight": 0.3}, {"symbol": "MSFT", "weight": 0.2}],
        "grid": {"values": [[1, 2], [3, 4]]},
        "volatility": 0.2,
        "label": "x",
    }
    assert value_at(payload, "volatility") == 0.2
    assert value_at(payload, "rows.1.weight") == 0.2
    assert value_at(payload, "rows[symbol=AAPL].weight") == 0.3
    assert value_at(payload, "grid.values.1.0") == 3
    assert value_at(payload, "label") is None
    assert value_at(payload, "missing.path") is None
