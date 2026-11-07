"""J20 endpoints offline: library report, signal monitor, strategy alerts with webhook, costs."""

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fda_api import workbench

HYPOTHESIS = "Breakouts persist because of slow information diffusion across participants."


def _run(
    client: TestClient, kind: str, sid: str, items: list[tuple[str, dict[str, Any]]]
) -> dict[str, Any]:
    r = client.post(
        "/v1/analyses:run",
        json={
            "subject": {"type": kind, "id": sid},
            "context": {"range": "MAX"},
            "analyses": [{"id": i, "params": p} for i, p in items],
        },
    )
    return {
        (json.loads(x).get("result") or json.loads(x).get("error"))["analysis_id"]: json.loads(x)
        for x in r.text.splitlines()
    }


def test_library_monitor_alerts_and_costs(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    client.post("/v1/ingest/FAKELIB")
    asset = _run(
        client,
        "asset",
        "FAKELIB",
        [("TR-010", {}), ("TR-013", {"shares": 5000, "schedule": "almgren_chriss"})],
    )
    report = asset["TR-010"]["result"]["payload"]
    assert (
        len(report["rows"]) == 16
    )  # 11 + carver_trend, macd_v, donchian_ensemble, tema_events, kalman
    assert all("fails_in" in r for r in report["rows"])
    costs = asset["TR-013"]["result"]["payload"]
    assert abs(sum(r["shares"] for r in costs["rows"]) - 5000) < 1e-6
    sid = client.post(
        "/v1/strategies",
        json={
            "name": "Breakout",
            "hypothesis": HYPOTHESIS,
            "spec": {
                "symbol": "FAKELIB",
                "template": "donchian_breakout",
                "overlays": ["trend_filter"],
            },
        },
    ).json()["id"]
    monitor = _run(client, "strategy", str(sid), [("TR-011", {})])["TR-011"]["result"]["payload"]
    assert monitor["action"] in {"buy", "sell", "hold"}
    assert monitor["overlays"] == ["trend_filter"]
    sent: list[dict[str, Any]] = []
    monkeypatch.setattr(
        workbench,
        "send_webhook",
        lambda url, payload: sent.append(payload.model_dump()) or "webhook delivered",
    )
    monkeypatch.setattr(workbench, "unsafe_webhook", lambda url: None)  # no DNS in tests
    rule = client.post(
        "/v1/alerts",
        json={
            "name": "breakout state",
            "subject_type": "strategy",
            "subject_id": str(sid),
            "analysis_id": "TR-011",
            "path": "position",
            "op": ">=",
            "threshold": -1,
            "webhook_url": "https://example.test/hook",
        },
    )
    assert rule.status_code == 201, rule.text
    assert (
        client.post(
            "/v1/alerts",
            json={
                "name": "x",
                "subject_type": "strategy",
                "subject_id": str(sid),
                "analysis_id": "TR-011",
                "path": "position",
                "op": ">",
                "threshold": 0,
                "webhook_url": "http://example.test/hook",
            },
        ).status_code
        == 422
    )
    first = client.post("/v1/alerts/evaluate").json()
    assert any(r["fired"] for r in first)
    assert sent
    assert sent[0]["ticker"] == "FAKELIB"
    assert set(sent[0]) == {"ticker", "action", "size", "price", "timestamp", "rule", "value"}
    client.post("/v1/alerts/evaluate")
    assert len(sent) == 1  # only on the transition


def test_webhook_targets_must_be_public_https() -> None:
    assert workbench.unsafe_webhook("http://example.com/hook") is not None
    assert workbench.unsafe_webhook("https://127.0.0.1/hook") is not None
    assert workbench.unsafe_webhook("https://localhost/hook") is not None
    assert workbench.unsafe_webhook("https://10.0.0.5/hook") is not None
    assert workbench.unsafe_webhook("https://169.254.169.254/latest") is not None
