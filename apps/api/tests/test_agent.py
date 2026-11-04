"""J25 slices 251–252 offline: the agent loop with a scripted model, and each wire format."""

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fda_api import agent
from fda_api.llm import LLMError, Reply, ToolCall, ToolSpec, providers
from fda_api.llm.base import inline_refs


class Scripted:
    """A model that calls the volatility tool once, then answers citing what it received."""

    def __init__(self) -> None:
        self.calls = 0

    def send(
        self, user_text: str | None = None, results: list[tuple[ToolCall, str]] | None = None
    ) -> Reply:
        self.calls += 1
        if self.calls == 1:
            args = {
                "name": "t_annualised_volatility",
                "arguments": {"subject_type": "asset", "subject_id": "FAKE", "range": "1Y"},
            }
            return Reply("", [ToolCall("c1", "call_tool", args)], {"input": 100, "output": 10})
        content = json.loads(results[0][1]) if results else {}
        value = content["summary"]["value"]
        rid = content["result_id"]
        return Reply(
            f"FAKE's annualised volatility is {value} [r:{rid}].", [], {"input": 300, "output": 20}
        )


class FakeProvider:
    name, label, local, key = "fake", "Fake", False, None

    def model(self, requested: str | None = None) -> str:
        return requested or "fake-1"

    def session(self, *args: Any) -> Scripted:
        return Scripted()


def _events(response: Any) -> list[dict[str, Any]]:
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def test_agent_loop_cites_and_records(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    client.post("/v1/ingest/FAKE")
    monkeypatch.setattr(agent, "get_provider", lambda name=None: FakeProvider())
    events = _events(client.post("/v1/agent/ask", json={"question": "Volatility of FAKE?"}))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "start"
    assert "tool_call" in kinds
    assert kinds[-1] == "answer"
    answer = events[-1]
    assert answer["verification"]["unverified"] == []
    assert answer["verification"]["coverage"] == 1.0
    assert answer["usage"] == {"input": 400, "output": 30}
    assert next(iter(answer["citations"].values())).endswith("/a/FAKE/overview")
    runs = client.get("/v1/agent/runs").json()
    assert runs[0]["id"] == answer["run_id"]
    assert runs[0]["steps"][0]["result_id"]
    assert runs[0]["model"] == "fake-1"


def test_unknown_model_is_refused(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent, "get_provider", lambda name=None: FakeProvider())
    monkeypatch.setattr(agent, "model_list", lambda provider: ["fake-1", "fake-2"])
    events = _events(client.post("/v1/agent/ask", json={"question": "hi", "model": "pricey-9"}))
    assert events[0]["type"] == "error"
    assert "pricey-9" in events[0]["detail"]


def test_agent_reports_provider_errors(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str | None = None) -> Any:
        raise LLMError("No LLM configured: add an API key to .env or start Ollama / LM Studio")

    monkeypatch.setattr(agent, "get_provider", missing)
    events = _events(client.post("/v1/agent/ask", json={"question": "hi"}))
    assert events == [
        {
            "type": "error",
            "detail": "No LLM configured: add an API key to .env or start Ollama / LM Studio",
        }
    ]
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-secret-123")
    monkeypatch.setattr(providers, "remote_models", lambda p: [])  # no network in tests
    monkeypatch.setattr(providers, "local_models", lambda p: [])
    response = client.get("/v1/agent/providers")
    listed = {p["name"]: p for p in response.json()}
    assert {"anthropic", "gemini", "openai", "mistral", "ollama", "lmstudio"} <= set(listed)
    assert listed["mistral"]["configured"] is True
    assert "sk-secret-123" not in response.text  # keys never leave the server


def test_french_answers_are_read_as_figures() -> None:
    text = (
        "Pour le portefeuille 3 sur 1 an : volatilité 15,01 % [r:a], perte 155,27 EUR [r:a], "
        "pic le 28/10/2025, 10 000 EUR investis [r:a], total 12,000 [r:a]."
    )
    out = agent.checkable(text)
    assert "15.01 %" in out
    assert "155.27 EUR" in out
    assert "12,000" in out  # thousands comma kept
    assert agent.checkable("1.  **First**\n2) Second") == "  **First**\n Second"
    for gone in ("portefeuille 3", "1 an", "28/10/2025"):
        assert gone not in out


def test_non_figures_are_not_checked() -> None:
    text = "SPY (S&P 500) vol 0.1 [r:a]. See [SPY](http://127.0.0.1:5173/a/SPY) and CAC 40."
    assert "500" not in agent.checkable(text)
    assert "5173" not in agent.checkable(text)
    assert "0.1 [r:a]" in agent.checkable(text)


TOOL = ToolSpec(
    "get_x",
    "Get x",
    {
        "type": "object",
        "properties": {"p": {"$ref": "#/$defs/P"}},
        "$defs": {"P": {"type": "string"}},
    },
)


def _capture(monkeypatch: pytest.MonkeyPatch, answer: dict[str, Any]) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []

    def fake_post(
        url: str, headers: dict[str, str], body: dict[str, Any], timeout: float = 0
    ) -> dict[str, Any]:
        sent.append({"url": url, "headers": headers, "body": json.loads(json.dumps(body))})
        return answer

    monkeypatch.setattr(providers, "post_json", fake_post)
    return sent


def test_anthropic_wire(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    answer = {
        "content": [
            {"type": "thinking", "thinking": "…", "signature": "sig"},
            {"type": "tool_use", "id": "tu1", "name": "get_x", "input": {"p": "a"}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 3},
    }
    sent = _capture(monkeypatch, answer)
    session = providers.PROVIDERS["anthropic"].session("claude-opus-5-5", "sys", [TOOL], [])
    reply = session.send("hi")
    assert reply.tool_calls == [ToolCall("tu1", "get_x", {"p": "a"})]
    assert reply.usage == {"input": 10, "output": 5, "cache_read": 3}
    assert sent[0]["url"].endswith("/v1/messages")
    assert sent[0]["headers"]["anthropic-version"] == "2026-10-21"
    assert sent[0]["body"]["tools"][0]["input_schema"]["$defs"]  # Claude follows $refs
    session.send(results=[(reply.tool_calls[0], "42")])
    second = sent[1]["body"]["messages"]
    assert second[1]["content"][0]["type"] == "thinking"  # assistant turn echoed verbatim
    assert second[2]["content"][0] == {"type": "tool_result", "tool_use_id": "tu1", "content": "42"}


def test_gemini_wire(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    answer = {
        "status": "requires_action",
        "steps": [
            {"type": "thought", "signature": "abc"},
            {"type": "function_call", "id": "fc1", "name": "get_x", "arguments": {"p": "a"}},
        ],
        "usage": {"total_input_tokens": 7, "total_output_tokens": 2},
    }
    sent = _capture(monkeypatch, answer)
    session = providers.PROVIDERS["gemini"].session(
        "gemini-3.8-flash", "sys", [TOOL], [("user", "a"), ("assistant", "b")]
    )
    reply = session.send("hi")
    assert reply.tool_calls == [ToolCall("fc1", "get_x", {"p": "a"})]
    body = sent[0]["body"]
    assert sent[0]["url"].endswith("/v1beta/interactions")
    assert body["store"] is False
    assert body["system_instruction"] == "sys"
    assert body["tools"][0]["parameters"]["properties"]["p"] == {"type": "string"}  # refs inlined
    assert [s["type"] for s in body["input"]] == ["user_input", "model_output", "user_input"]
    session.send(results=[(reply.tool_calls[0], "42")])
    steps = sent[1]["body"]["input"]
    assert steps[3]["type"] == "thought"  # signatures go back unchanged
    assert steps[-1] == {
        "type": "function_result",
        "name": "get_x",
        "call_id": "fc1",
        "result": [{"type": "text", "text": "42"}],
    }

    final = {
        "status": "completed",
        "steps": [{"type": "model_output", "content": [{"type": "text", "text": "done"}]}],
    }
    _capture(monkeypatch, final)
    assert session.send("again").text == "done"


def test_openai_compatible_wire(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "m-test")
    answer = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_x", "arguments": '{"p": "a"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 4},
    }
    sent = _capture(monkeypatch, answer)
    session = providers.PROVIDERS["mistral"].session("mistral-large-latest", "sys", [TOOL], [])
    reply = session.send("hi")
    assert reply.tool_calls == [ToolCall("call_1", "get_x", {"p": "a"})]
    assert sent[0]["headers"]["authorization"] == "Bearer m-test"
    assert sent[0]["url"] == "https://api.mistral.ai/v1/chat/completions"
    session.send(results=[(reply.tool_calls[0], "42")])
    assert sent[1]["body"]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "42",
    }


def test_provider_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    for p in providers.PROVIDERS.values():
        if p.key_env:
            monkeypatch.delenv(p.key_env, raising=False)
    monkeypatch.delenv("FDA_LLM_PROVIDER", raising=False)
    monkeypatch.setattr(providers, "local_models", lambda p: [])
    with pytest.raises(LLMError, match="No LLM configured"):
        providers.get_provider()
    monkeypatch.setenv("OPENROUTER_API_KEY", "or")
    assert providers.get_provider().name == "openrouter"
    monkeypatch.setenv("FDA_LLM_MODEL_OPENROUTER", "vendor/model")
    assert providers.PROVIDERS["openrouter"].model() == "vendor/model"
    with pytest.raises(LLMError, match="Unknown provider"):
        providers.get_provider("nope")
    assert inline_refs(TOOL.schema) == {"type": "object", "properties": {"p": {"type": "string"}}}


def test_unwrap_nested_call_tool() -> None:
    args = {"range": "1Y", "subject_id": "AAPL", "subject_type": "asset"}
    plain = {"name": "t_vol", "arguments": args}
    assert agent.unwrap("call_tool", plain) == ("t_vol", args)
    # seen with gemma4 (Ollama): the call wrapped once more, under either name
    assert agent.unwrap("call_tool", {"name": "call_tool", "arguments": plain}) == ("t_vol", args)
    assert agent.unwrap("call_tool", {"name": "t_vol", "arguments": plain}) == ("t_vol", args)
    assert agent.unwrap("search_tools", {"query": "vol"}) == ("search_tools", {"query": "vol"})
    assert agent.unwrap("call_tool", {"name": "t_vol", "arguments": "x"}) == ("t_vol", {})


def test_page_context_and_rank_words() -> None:
    prompt = agent.system_prompt({"type": "asset", "id": "AAPL", "range": "3Y"})
    assert "asset AAPL" in prompt
    assert "range is 3Y" in prompt
    assert "range is" not in agent.system_prompt({"type": "asset", "id": "AAPL", "range": "evil"})
    assert "t_dispersion: Dispersion — Std, variance" in agent.analysis_index()
    text = agent.checkable("The 75th percentile is 1.2 %, the 1st quartile 0.3 %; 3e quartile 2 %.")
    assert "75" not in text
    assert "1st" not in text
    assert "3e" not in text
    assert "1.2 %" in text
    assert "0.3 %" in text
    assert "2 %" in text


class Sloppy(Scripted):
    """Adds an invented figure; on the citation check, rewrites without it (or not at all)."""

    def __init__(self, fixes: bool) -> None:
        super().__init__()
        self.fixes, self.good = fixes, ""

    def send(
        self, user_text: str | None = None, results: list[tuple[ToolCall, str]] | None = None
    ) -> Reply:
        if user_text and user_text.startswith("Citation check"):
            assert "42 %" in user_text
            text = self.good if self.fixes else self.good + " Roughly 42 % lower than peers."
            return Reply(text, [], {"input": 5, "output": 5})
        reply = super().send(user_text, results)
        if reply.text:
            self.good = reply.text
            reply.text += " Roughly 42 % lower than peers."
        return reply


@pytest.mark.parametrize("fixes", [True, False])
def test_citation_check_round(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fixes: bool
) -> None:
    client.post("/v1/ingest/FAKE")

    class P(FakeProvider):
        def session(self, *args: Any) -> Sloppy:
            return Sloppy(fixes)

    monkeypatch.setattr(agent, "get_provider", lambda name=None: P())
    events = _events(client.post("/v1/agent/ask", json={"question": "Volatility of FAKE?"}))
    answer = events[-1]
    assert answer["type"] == "answer"
    if fixes:
        assert answer["verification"]["unverified"] == []
        assert "42" not in answer["text"]
    else:  # no better rewrite: the original stays, flagged
        assert answer["verification"]["unverified"] == ["42 %"]
    run = client.get("/v1/agent/runs").json()[0]
    assert run["steps"][-1]["tool"] == "citation_check"
    assert run["usage"]["input"] == 405
