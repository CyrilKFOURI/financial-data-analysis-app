"""Agent-level evaluation offline: a scripted model answers gold questions through the real agent
loop; the grader passes correct cited answers and fails wrong, uncited or tool-less ones."""

import json
from datetime import date
from typing import Any

import pytest
from fastapi.testclient import TestClient

import fda_data.ingest as ingest
from fda_api import agent
from fda_api.agent_eval import grade, markdown, summarise
from fda_api.evaluation import load_answers, load_gold
from fda_api.llm import Reply, ToolCall
from fda_api.tools import all_tools


class FixedDate(date):
    @classmethod
    def today(cls) -> "FixedDate":
        return cls(2025, 6, 30)


GOLD = {item["question"]: item for item in load_gold()}
NAMES = {t.analysis_id: t.name for t in all_tools() if t.analysis_id}


class GoldModel:
    """Calls the gold tool for the question, then writes the value (or a wrong one)."""

    def __init__(self, question: str, mode: str) -> None:
        self.item, self.mode, self.calls, self.last = GOLD[question], mode, 0, ""

    def send(
        self, user_text: str | None = None, results: list[tuple[ToolCall, str]] | None = None
    ) -> Reply:
        if user_text and user_text.startswith("Citation check"):
            return Reply(self.last, [], {"output": 1})  # a model that cannot do better
        self.calls += 1
        if self.calls == 1 and self.mode != "no-tool":
            item = self.item
            args = {
                "name": NAMES[item["analysis"]],
                "arguments": {
                    "subject_type": item["subject"]["type"],
                    "subject_id": item["subject"]["id"],
                    "range": item["range"],
                    "params": item["params"],
                },
            }
            return Reply("", [ToolCall("c1", "call_tool", args)], {"input": 10, "output": 5})
        if self.mode == "no-tool":
            self.last = "About 12.5 %."
            return Reply(self.last, [], {"input": 10, "output": 5})
        content = json.loads(results[0][1]) if results else {}
        node: Any = content
        for part in self.item["path"].split("."):
            node = node[int(part)] if isinstance(node, list) else node[part]
        value = float(node) * (1.1 if self.mode == "wrong" else 1.0)
        self.last = f"The answer is {value:.6g} [r:{content['result_id']}]."
        return Reply(self.last, [], {"output": 7})


class GoldProvider:
    name, label, local, key = "scripted", "Scripted", False, None

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.question = ""

    def model(self, requested: str | None = None) -> str:
        return "gold-1"

    def session(self, *args: Any) -> "GoldProvider":
        return self

    def send(self, user_text: str | None = None, results: Any = None) -> Reply:
        if user_text in GOLD:
            self.question, self.inner = user_text, GoldModel(user_text, self.mode)
        return self.inner.send(user_text, results)


def _ask(client: TestClient, question: str) -> list[dict[str, Any]]:
    response = client.post("/v1/agent/ask", json={"question": question})
    return [json.loads(x) for x in response.text.splitlines() if x.strip()]


@pytest.mark.parametrize("mode", ["right", "wrong", "no-tool"])
def test_grader(client: TestClient, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.setattr(ingest, "date", FixedDate)
    monkeypatch.setattr(agent, "date", FixedDate)
    client.post("/v1/ingest/FAKEGOLD")
    provider = GoldProvider(mode)
    monkeypatch.setattr(agent, "get_provider", lambda name=None: provider)
    answers = load_answers()
    graded = []
    for item in load_gold()[:6]:
        events = _ask(client, item["question"])
        graded.append(grade(item, answers[item["id"]], events, 1.0))
    summary = summarise(graded, "scripted", "gold-1")
    if mode == "right":
        assert summary["passed"] == 6, [(g.id, g.answer, g.unverified) for g in graded]
        assert summary["gate"] is True
        assert graded[0].tools == [NAMES[load_gold()[0]["analysis"]]]
        assert graded[0].model_rounds == 2
    elif mode == "wrong":
        assert summary["correct"] == 0
        assert summary["fully_cited"] == 0  # the wrong number is not in the cited result
        assert summary["uncited_numbers"] == 0  # cited, but not verified
        assert summary["gate"] is False
    else:
        assert summary["passed"] == 0
        assert summary["unverified_numbers"] == 6
        assert summary["uncited_numbers"] == 6
    assert "| g01 |" in markdown(summary, graded)


def test_percent_and_relative_tolerance() -> None:
    item = {"id": "x", "question": "q", "rel_tol": 0.05}
    answer = {
        "type": "answer",
        "text": "CAGR 39.02 % [r:abcdef123456]",
        "verification": {"numbers": 1, "cited": 1, "unverified": [], "coverage": 1.0},
        "usage": {},
    }
    assert grade({"id": "x", "question": "q"}, 0.390172, [answer]).passed
    assert not grade({"id": "x", "question": "q"}, 0.40, [answer]).correct
    assert grade(item, 0.40, [answer]).correct  # within 5 %
