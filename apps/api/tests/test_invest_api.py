"""J16 endpoints offline: questionnaire → profile → IN-001/IN-003 → IPS."""

import json

from fastapi.testclient import TestClient

ANSWERS = {
    "horizon": "10-20y",
    "income_stability": "stable",
    "emergency_fund": "3-6m",
    "savings_rate": "5-15%",
    "debt": "low",
    "liquidity_needs": "none",
    "drawdown_reaction": "hold",
    "max_loss": "20%",
    "preference": "balanced",
    "knowledge": "good",
    "experience": "3-10",
    "goal": {"amount": 300000, "years": 15, "current": 40000, "monthly_saving": 600},
}


def test_profile_flow(client: TestClient) -> None:
    questions = client.get("/v1/invest/questionnaire").json()
    assert len(questions["questions"]) == 11
    assert (
        client.post(
            "/v1/invest/profiles", json={"name": "Me", "answers": {"horizon": "<2y"}}
        ).status_code
        == 422
    )
    created = client.post("/v1/invest/profiles", json={"name": "Me", "answers": ANSWERS}).json()
    assert created["risk_score"] == 5
    r = client.post(
        "/v1/analyses:run",
        json={
            "subject": {"type": "profile", "id": str(created["id"])},
            "context": {"range": "MAX"},
            "analyses": [{"id": "IN-001", "params": {}}, {"id": "IN-003", "params": {}}],
        },
    )
    profile, allocation = [json.loads(line) for line in r.text.splitlines()]
    payload = profile["result"]["payload"]
    assert payload["label"].startswith("Model output")
    assert payload["required_return"] > 0
    # No asset-class proxies offline: the allocation still lists weights, without a volatility.
    alloc = allocation["result"]["payload"]
    assert abs(sum(row["weight"] for row in alloc["rows"]) - 1) < 1e-9
    assert alloc["volatility"] is None
    ips = client.get(f"/v1/invest/ips/{created['id']}")
    assert ips.status_code == 200
    assert "Investment policy statement" in ips.text
    assert "Rebalancing" in ips.text
    assert "not personalised advice" in ips.text
    assert [p["name"] for p in client.get("/v1/invest/profiles").json()] == ["Me"]
