"""Portfolio endpoints and the portfolio Subject through /v1/analyses:run (offline, fake data)."""

import json
from typing import Any

from fastapi.testclient import TestClient


def _run(client: TestClient, subject: str, ids: list[str], **params: Any) -> dict[str, Any]:
    response = client.post(
        "/v1/analyses:run",
        json={
            "subject": {"type": "portfolio", "id": subject},
            "context": {"range": "MAX"},
            "analyses": [{"id": i, "params": params.get(i, {})} for i in ids],
        },
    )
    assert response.status_code == 200
    out = {}
    for line in response.text.splitlines():
        item = json.loads(line)
        result = item.get("result") or item.get("error")
        out[result["analysis_id"]] = item
    return out


def _weights_portfolio(client: TestClient, name: str = "Core", **extra: Any) -> int:
    for symbol in ("FAKE", "FAKEB"):
        assert client.post(f"/v1/ingest/{symbol}").status_code == 202
    body = {
        "name": name,
        "mode": "weights",
        "base_currency": "USD",
        "holdings": [
            {"symbol": "FAKE", "weight": 0.6},
            {"symbol": "FAKEB", "weight": 0.3},
            {"symbol": "CASH", "weight": 0.1, "is_cash": True},
        ],
    } | extra
    response = client.post("/v1/portfolios", json=body)
    assert response.status_code == 201, response.text
    return int(response.json()["id"])


def test_create_list_version_and_validation(client: TestClient) -> None:
    pid = _weights_portfolio(client)
    listed = client.get("/v1/portfolios").json()
    assert [p["name"] for p in listed] == ["Core"]
    assert listed[0]["holdings"] == 3
    bad = client.post(
        "/v1/portfolios", json={"name": "Bad", "holdings": [{"symbol": "FAKE", "weight": 0.5}]}
    )
    assert bad.status_code == 422
    assert "sum" in bad.json()["detail"]
    dup = client.post(
        "/v1/portfolios", json={"name": "Core", "holdings": [{"symbol": "FAKE", "weight": 1}]}
    )
    assert dup.status_code == 422
    draft = client.put(
        f"/v1/portfolios/{pid}", json={"holdings": [{"symbol": "FAKE", "weight": 1}], "draft": True}
    )
    assert draft.json() == {"id": pid, "version": 2, "draft": True, "subject": f"{pid}@2"}
    current = client.get(f"/v1/portfolios/{pid}").json()
    assert current["version"] == 1  # the what-if draft leaves the saved version untouched
    assert len(current["holdings"]) == 3
    assert [v["version_no"] for v in current["versions"]] == [2, 1]
    assert (
        client.get(f"/v1/portfolios/{pid}", params={"version": 2}).json()["holdings"][0]["weight"]
        == 1
    )


def test_portfolio_subject_runs_portfolio_and_price_analyses(client: TestClient) -> None:
    pid = _weights_portfolio(client)
    ids = [
        "P-001",
        "P-003",
        "P-004",
        "P-005",
        "P-006",
        "P-007",
        "P-008",
        "P-009",
        "P-010",
        "P-011",
        "P-012",
        "P-017",
        "P-020",
        "P-021",
        "P-022",
        "P-024",
        "C-001",
        "C-004",
        "C-005",
        "T-001",
        "T-030",
        "R-001",
        "R-020",
        "TS-001",
        "TR-026",
    ]
    out = _run(client, str(pid), ids)
    failed = {k: v["error"]["detail"] for k, v in out.items() if "error" in v}
    assert set(failed) <= {"P-022"}, failed  # synthetic history starts after the crises
    d = out["P-009"]["result"]["payload"]
    assert abs(sum(r["risk_contribution"] for r in d["rows"]) - d["volatility"]) < 1e-9
    assert out["P-003"]["result"]["analysis_id"] == "P-003"
    book = out["TR-026"]["result"]["payload"]
    assert [r["series"] for r in book["rows"]] == [
        "carver_book",
        "vol_targeted_hold",
        "buy_and_hold",
    ]
    assert [w["symbol"] for w in book["weights"]] == ["FAKE", "FAKEB"]  # cash is not traded
    # What-if draft analysed side by side: a different subject, a different result.
    client.put(
        f"/v1/portfolios/{pid}", json={"holdings": [{"symbol": "FAKE", "weight": 1}], "draft": True}
    )
    draft = _run(client, f"{pid}@2", ["P-007"])["P-007"]["result"]["payload"]
    assert draft["holdings"] == 1
    missing = _run(client, "999", ["P-003"])["P-003"]["error"]
    assert "not found" in missing["detail"]


def test_ledger_import_dry_run_idempotency_void_and_convert(client: TestClient) -> None:
    for symbol in ("FAKE", "FAKEB"):
        client.post(f"/v1/ingest/{symbol}")
    pid = client.post("/v1/portfolios", json={"name": "Broker", "mode": "transactions"}).json()[
        "id"
    ]
    template = client.get("/v1/portfolios-import/template").json()
    csv_text = (
        template["csv"]
        .replace("AAPL", "FAKE")
        .replace("2024-01-02", "2025-01-02")
        .replace("2024-03-15", "2025-03-14")
    )
    csv_text += "2025-02-03,BUY,FAKEB,5,100,USD,0,,,t-2\nnot-a-date,BUY,FAKE,1,1,USD,0,,,x\n"
    dry = client.post(
        f"/v1/portfolios/{pid}/import", json={"csv": csv_text, "dry_run": True}
    ).json()
    assert dry["rows"] == 4
    assert any("not-a-date" in e for e in dry["errors"])
    clean = "\n".join(csv_text.splitlines()[:-1]) + "\n"
    first = client.post(
        f"/v1/portfolios/{pid}/import", json={"csv": clean, "dry_run": False}
    ).json()
    assert first["inserted"] == 4
    again = client.post(
        f"/v1/portfolios/{pid}/import", json={"csv": clean, "dry_run": False}
    ).json()
    assert again["already_imported"] is True
    re_export = clean.replace("t-2", "t-2") + "2025-04-01,FEE,,2,1,USD,0,,,f-1\n"
    partial = client.post(
        f"/v1/portfolios/{pid}/import", json={"csv": re_export, "dry_run": False}
    ).json()
    assert (partial["inserted"], partial["duplicates"]) == (1, 4)  # external ids dedupe re-exports
    tx = client.get(f"/v1/portfolios/{pid}/transactions").json()
    assert len(tx) == 5
    fee = next(t for t in tx if t["type"] == "FEE")
    assert client.post(f"/v1/portfolios/{pid}/transactions/{fee['id']}/void").json() == {
        "voided": True
    }
    assert len(client.get(f"/v1/portfolios/{pid}/transactions").json()) == 4
    out = _run(client, str(pid), ["P-002", "P-004", "P-005"])
    positions = out["P-005"]["result"]["payload"]["positions"]
    assert {p["symbol"] for p in positions} >= {"FAKE", "FAKEB"}
    assert out["P-002"]["result"]["payload"]["counts"]["BUY"] == 2
    converted = client.post(f"/v1/portfolios/{pid}/convert", json={"to": "weights"}).json()
    assert converted["mode"] == "weights"
    weights = client.get(f"/v1/portfolios/{pid}").json()
    assert abs(sum(h["weight"] for h in weights["holdings"]) - 1) < 1e-9


def test_weights_to_transactions_and_benchmark_portfolio(client: TestClient) -> None:
    pid = _weights_portfolio(client)
    bench = _weights_portfolio(
        client,
        "Bench",
        holdings=[{"symbol": "FAKE", "weight": 0.5}, {"symbol": "FAKEB", "weight": 0.5}],
    )
    client.put(f"/v1/portfolios/{pid}", json={"benchmark": f"portfolio:{bench}"})
    out = _run(client, str(pid), ["P-018", "P-025", "P-026"])
    assert all("result" in v for v in out.values()), out
    # min(0.6, 0.5) + min(0.3, 0.5), up to drift since the last rebalance
    assert abs(out["P-026"]["result"]["payload"]["overlap"] - 0.8) < 0.05
    converted = client.post(f"/v1/portfolios/{pid}/convert", json={"to": "transactions"}).json()
    assert converted["mode"] == "transactions"
    mwr = _run(client, str(pid), ["P-004"])["P-004"]["result"]["payload"]
    assert abs(mwr["net_flows"] - 100_000) < 1e-6  # synthetic deposit = the initial value
