"""AI-006 guardrails offline: model-output labels, validation-before-claim, no live orders."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from fda_api import guardrails, tools
from fda_api.runner import Run

HYPOTHESIS = "Trend persistence: medium-term momentum in broad equity indices (under-reaction)."


def _call(
    client: TestClient, analysis_id: str, subject_type: str, subject_id: Any
) -> dict[str, Any]:
    name = next(t.name for t in tools.all_tools() if t.analysis_id == analysis_id)
    body = {
        "arguments": {"subject_type": subject_type, "subject_id": str(subject_id), "range": "MAX"}
    }
    response = client.post(f"/v1/tools/{name}:call", json=body).json()
    assert response["is_error"] is False, response
    return dict(response["content"])


@pytest.fixture
def subjects(client: TestClient) -> dict[str, int]:
    for symbol in ("FAKE", "FAKEB", "FAKEBT"):
        client.post(f"/v1/ingest/{symbol}")
    pid = client.post(
        "/v1/portfolios",
        json={
            "name": "G",
            "mode": "weights",
            "holdings": [{"symbol": "FAKE", "weight": 0.5}, {"symbol": "FAKEB", "weight": 0.5}],
        },
    ).json()["id"]
    sid = client.post(
        "/v1/strategies",
        json={
            "name": "G",
            "hypothesis": HYPOTHESIS,
            "spec": {
                "symbol": "FAKEBT",
                "template": "ema_cross",
                "grid": {"fast": [5, 10], "slow": [50, 100]},
            },
        },
    ).json()["id"]
    return {"pid": int(pid), "sid": int(sid)}


def test_performance_comes_with_its_validation(
    client: TestClient, subjects: dict[str, int]
) -> None:
    stats = _call(client, "TR-016", "strategy", subjects["sid"])
    assert stats["model_output"] is True
    assert stats["label"].startswith("Model output")
    assert stats["evidence"]["analysis_id"] == "TR-007"
    assert {"dsr", "pbo"} <= set(stats["evidence"]["summary"])
    assert "sharpe" in stats["summary"]

    weights = _call(client, "IN-011", "portfolio", subjects["pid"])
    assert weights["evidence"]["analysis_id"] == "IN-018"
    assert "deflated_sharpe" in weights["evidence"]["summary"]
    assert weights["assumptions"]["range"] == "MAX"

    own = _call(client, "TR-007", "strategy", subjects["sid"])  # carries its own evidence
    assert own["evidence"]["analysis_id"] == "TR-007"

    plain = _call(client, "R-005", "asset", "FAKE")  # V1 measurement: no model-output wrapper
    assert "model_output" not in plain


def test_performance_is_withheld_without_validation(
    client: TestClient, subjects: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(*args: Any, **kwargs: Any) -> Run:
        return Run("TR-007", "", "", "", {}, None, [], "not enough out-of-sample data", None)

    monkeypatch.setattr(tools, "run_analysis", failing)
    stats = _call(client, "TR-016", "strategy", subjects["sid"])
    assert "sharpe" in stats["withheld"]
    assert "sharpe" not in stats["summary"]
    assert "cagr" not in stats["summary"]
    assert "TR-007 could not be computed" in stats["withheld_reason"]


def test_no_live_order_tool() -> None:
    every = tools.all_tools()
    for tool in every:
        text = f"{tool.name} {tool.description}".lower()
        assert "live order" not in text or "no live" in text
        assert "broker" not in tool.name
        if "order" in tool.name:
            assert tool.safety == "write"  # proposals approved in the web app
            assert "paper" in tool.name


def test_classification_rules() -> None:
    assert guardrails.needs_validation("TR-016", {"sharpe": 1.2, "cagr": 0.1})
    assert not guardrails.needs_validation("TR-016", {"sharpe": 1.2, "dsr": 0.4})
    assert not guardrails.needs_validation("TR-040", {"total_return": 0.02})  # paper records
    assert guardrails.needs_validation("IN-011", {"sharpe": 1.0, "validated": False})
    assert not guardrails.needs_validation("IN-018", {"deflated_sharpe": 0.3, "validated": True})
    assert guardrails.evidence_request(
        "portfolio", {"method": "hrp"}, {"mean": "shrunk", "trials": 3}
    ) == (
        "IN-018",
        {"mean": "shrunk", "method": "hrp"},
    )
    assert guardrails.evidence_request("asset", {}, {}) is None
