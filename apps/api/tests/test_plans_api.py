"""J18 endpoints offline: DCA, rebalancing, risk budgets, hedging, goals, Kelly."""

import json
from typing import Any

from fastapi.testclient import TestClient


def _run(
    client: TestClient, kind: str, sid: str, items: list[tuple[str, dict[str, Any]]]
) -> dict[str, Any]:
    r = client.post(
        "/v1/analyses:run",
        json={
            "subject": {"type": kind, "id": sid},
            "context": {"range": "3Y"},
            "analyses": [{"id": i, "params": p} for i, p in items],
        },
    )
    out = {}
    for line in r.text.splitlines():
        x = json.loads(line)
        out[(x.get("result") or x.get("error"))["analysis_id"]] = x
    return out


def test_plans(client: TestClient) -> None:
    for s in ("FAKEP1", "FAKEP2", "FAKEP3"):
        client.post(f"/v1/ingest/{s}")
    asset = _run(
        client, "asset", "FAKEP1", [("IN-020", {"months": 36, "simulations": 200}), ("IN-025", {})]
    )
    dca = asset["IN-020"]["result"]["payload"]
    assert {r["variant"] for r in dca["rows"]} == {
        "lump_sum",
        "dca",
        "value_averaging",
        "vix_scaled",
        "trend_filtered",
        "hybrid",
    }
    assert dca["vix_available"] is False  # no ^VIX offline: the multiplier falls back to 1×
    assert asset["IN-025"]["result"]["payload"]["warning"]
    pf = client.post(
        "/v1/portfolios",
        json={
            "name": "Plan",
            "benchmark": "FAKEP3",
            "holdings": [{"symbol": "FAKEP1", "weight": 0.6}, {"symbol": "FAKEP2", "weight": 0.4}],
        },
    ).json()
    out = _run(
        client,
        "portfolio",
        str(pf["id"]),
        [("IN-021", {}), ("IN-022", {"other": "FAKEP3"}), ("IN-023", {})],
    )
    for aid in ("IN-021", "IN-022", "IN-023"):
        assert "result" in out[aid], (aid, out[aid])
    assert out["IN-021"]["result"]["payload"]["recommended"] is not None
    hedge = out["IN-022"]["result"]["payload"]
    assert hedge["tail"]["vol_source"].startswith("1-year realised")
    profile = client.post(
        "/v1/invest/profiles",
        json={
            "name": "Goals",
            "answers": {
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
            },
        },
    ).json()
    goal = _run(client, "profile", str(profile["id"]), [("IN-024", {})])["IN-024"]
    assert "invest/ingest" in goal["error"]["detail"]  # no asset-class data offline
