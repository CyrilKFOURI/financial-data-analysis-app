"""J23 endpoints offline: paper accounts, orders (API + back-dated through the data layer so
they fill on stored bars), cancellation, strategy-driven orders, TR-040 / TR-042."""

import json
from datetime import date
from typing import Any

from fastapi.testclient import TestClient

from fda_data.paper import place_order, validate_order


def _run(client: TestClient, aid: int, items: list[str]) -> dict[str, Any]:
    r = client.post(
        "/v1/analyses:run",
        json={
            "subject": {"type": "paper", "id": str(aid)},
            "context": {"range": "MAX"},
            "analyses": [{"id": i, "params": {}} for i in items],
        },
    )
    out = {}
    for line in r.text.splitlines():
        x = json.loads(line)
        out[(x.get("result") or x.get("error"))["analysis_id"]] = x
    return out


def test_paper_flow(client: TestClient) -> None:
    client.post("/v1/ingest/FAKEPT")
    bad = client.post("/v1/paper", json={"name": "Neg", "starting_cash": -5})
    assert bad.status_code == 422
    aid = client.post("/v1/paper", json={"name": "Main", "starting_cash": 50_000}).json()["id"]
    assert client.post("/v1/paper", json={"name": "Main"}).status_code == 422

    empty = _run(client, aid, ["TR-042"])["TR-042"]
    assert "No orders yet" in empty["error"]["detail"]

    unknown = client.post(
        f"/v1/paper/{aid}/orders", json={"symbol": "NOPE", "side": "buy", "quantity": 1}
    )
    assert unknown.status_code == 422
    limit_without_price = client.post(
        f"/v1/paper/{aid}/orders",
        json={"symbol": "FAKEPT", "side": "buy", "quantity": 1, "order_type": "limit"},
    )
    assert limit_without_price.status_code == 422
    placed = client.post(
        f"/v1/paper/{aid}/orders",
        json={
            "symbol": "FAKEPT",
            "side": "buy",
            "quantity": 5,
            "order_type": "limit",
            "limit_price": 1,
        },
    ).json()
    assert placed["decision_price"] > 0
    cancel = client.post(f"/v1/paper/{aid}/orders/{placed['id']}/cancel")
    assert cancel.status_code == 200
    assert client.post(f"/v1/paper/{aid}/orders/{placed['id']}/cancel").status_code == 409

    # back-dated orders fill on the stored bars
    engine = client.app.state.engine  # type: ignore[attr-defined]
    account = client.get(f"/v1/paper/{aid}").json()
    decided = date.fromisoformat(placed["decision_date"])
    early = date(decided.year - 1, decided.month, 1)
    place_order(
        engine,
        aid,
        validate_order({"symbol": "FAKEPT", "side": "buy", "quantity": 40}),
        early,
        100.0,
    )
    place_order(
        engine,
        aid,
        validate_order(
            {
                "symbol": "FAKEPT",
                "side": "sell",
                "quantity": 15,
                "order_type": "trailing_stop",
                "trail_pct": 0.05,
            }
        ),
        early,
        100.0,
    )
    assert len(client.get(f"/v1/paper/{aid}").json()["orders"]) == len(account["orders"]) + 2

    out = _run(client, aid, ["TR-040", "TR-042"])
    statement = out["TR-040"]
    assert "result" in statement, statement
    p = statement["result"]["payload"]
    assert p["reconciliation"]["difference"] == 0.0
    assert p["reconciliation"]["cash_difference"] == 0.0
    assert p["fees"] > 0
    assert len(p["dates"]) == len(p["lines"]["equity"])
    statuses = {o["status"] for o in p["orders"]}
    assert "filled" in statuses
    assert "cancelled" in statuses or "working" in statuses
    execution = out["TR-042"]["result"]["payload"]
    assert execution["summary"]["fee_bps"] > 0
    assert 0 < execution["fill_rate"] <= 1

    sid = client.post(
        "/v1/strategies",
        json={
            "name": "Paper trend",
            "hypothesis": "Trend persistence after the crossing of averages",
            "spec": {"symbol": "FAKEPT", "template": "sma_cross"},
        },
    ).json()["id"]
    signal = client.post(
        f"/v1/paper/{aid}/orders/from-strategy", json={"strategy_id": sid, "weight": 0.5}
    )
    assert signal.status_code == 201
    assert signal.json()["signal"] in (0.0, 1.0)
    assert client.delete(f"/v1/paper/{aid}").status_code == 204
    assert client.get(f"/v1/paper/{aid}").status_code == 404
