"""NFR-12 cost meter and monthly budget: prices from .env, local models free, usage per model,
calls refused once the token cap is reached."""

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fda_api import agent
from fda_api.llm import Reply, ToolCall
from fda_api.llm.costs import cost, price


class Echo:
    def send(
        self, text: str | None = None, results: list[tuple[ToolCall, str]] | None = None
    ) -> Reply:
        return Reply("No figures needed.", [], {"input": 1000, "output": 200})


class EchoProvider:
    name, label, local, key = "gemini", "Gemini", False, None

    def model(self, requested: str | None = None) -> str:
        return "g-1"

    def session(self, *args: Any) -> Echo:
        return Echo()


def test_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FDA_LLM_PRICE_GEMINI", raising=False)
    assert price("gemini") is None
    assert cost("gemini", {"input": 10}) is None
    assert price("ollama") == (0.0, 0.0)  # local
    monkeypatch.setenv("FDA_LLM_PRICE_GEMINI", "0.30,2.50")
    assert cost("gemini", {"input": 1_000_000, "output": 100_000}) == pytest.approx(0.55)
    monkeypatch.setenv("FDA_LLM_PRICE_ANTHROPIC", "5,25")
    # cache reads at a tenth of the input price
    assert cost("anthropic", {"input": 0, "cache_read": 1_000_000}) == pytest.approx(0.5)
    monkeypatch.setenv("FDA_LLM_PRICE_GEMINI", "nonsense")
    assert price("gemini") is None


def test_usage_and_budget(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent, "get_provider", lambda name=None: EchoProvider())
    monkeypatch.setenv("FDA_LLM_PRICE_GEMINI", "0.30,2.50")
    monkeypatch.setenv("FDA_LLM_MONTHLY_TOKENS", "2000")

    def ask() -> dict[str, Any]:
        out = client.post("/v1/agent/ask", json={"question": "hello"})
        event: dict[str, Any] = json.loads(out.text.splitlines()[-1])
        return event

    first = ask()
    assert first["type"] == "answer"
    assert first["cost"] == pytest.approx((1000 * 0.30 + 200 * 2.50) / 1e6)
    usage = client.get("/v1/agent/usage").json()
    assert usage["runs"] == 1
    assert usage["tokens"] == 1200
    assert usage["models"][0] | {"cost": None} == {
        "provider": "gemini", "model": "g-1", "runs": 1, "input": 1000, "output": 200,
        "cost": None, "priced": True,
    }  # fmt: skip
    assert usage["over_budget"] is False
    assert client.get("/v1/agent/runs").json()[0]["cost"] == pytest.approx(first["cost"])
    assert ask()["type"] == "answer"  # 1,200 < 2,000 before this call
    refused = ask()
    assert refused["type"] == "error"
    assert "budget" in refused["detail"]
    assert client.get("/v1/agent/usage").json()["over_budget"] is True
