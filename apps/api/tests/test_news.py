"""AI-005 offline: synthetic headlines → store (merged, idempotent) → LLM scores from a scripted
model → daily sentiment index and lead–lag reading; cited digest; score parsing."""

import json
import re
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fda_api import news
from fda_api.llm import Reply, ToolCall


class Rater:
    """Scores titles by keyword, like a model would; writes a digest citing one real id."""

    def __init__(self, system: str) -> None:
        self.system = system

    def send(
        self, text: str | None = None, results: list[tuple[ToolCall, str]] | None = None
    ) -> Reply:
        lines = (text or "").splitlines()
        if "digest" in self.system:
            first = re.search(r"\[h:([0-9a-f]{10})\]", lines[0])
            assert first
            cited = first.group(1)
            body = f"- Results beat estimates [h:{cited}]\n- A probe [h:ffffffffff]\nNo source."
            return Reply(body, [], {"input": 50, "output": 20})
        rows = []
        for n, line in enumerate(lines, 1):
            score = 0.7 if re.search("beats|raises|lift|wins", line) else 0.0
            score = -0.8 if re.search("misses|cuts|probe|recalls", line) else score
            rows.append({"i": n, "score": score})
        return Reply("```json\n" + json.dumps(rows) + "\n```", [], {"input": 100, "output": 40})


class RaterProvider:
    name, label, local, key = "scripted", "Scripted", False, None

    def model(self, requested: str | None = None) -> str:
        return "rater-1"

    def session(self, model: str, system: str, *args: Any) -> Rater:
        return Rater(system)


def _run(client: TestClient, symbol: str) -> dict[str, Any]:
    out = client.post(
        "/v1/analyses:run",
        json={"subject": {"type": "asset", "id": symbol}, "analyses": [{"id": "AI-005"}]},
    )
    event: dict[str, Any] = json.loads(out.text.splitlines()[0])
    return event.get("result") or event["error"]


def test_news_flow(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    client.post("/v1/ingest/FAKENEWS")
    assert client.get("/v1/news/FAKENEWS").status_code == 404
    assert "ingest" in _run(client, "FAKENEWS")["detail"]
    first = client.post("/v1/news/FAKENEWS/ingest").json()
    assert first == {"fetched": 40, "new": 40, "stored": 40}
    assert client.post("/v1/news/FAKENEWS/ingest").json()["new"] == 0  # merged by id

    before = _run(client, "FAKENEWS")
    assert before["payload"]["headlines_scored"] == 0
    assert any("not scored" in w for w in before["warnings"])

    monkeypatch.setattr(news, "get_provider", lambda name=None: RaterProvider())
    scored = client.post("/v1/news/FAKENEWS/score", json={}).json()
    assert scored["scored"] == 40
    assert scored["unscored_left"] == 0
    assert scored["usage"] == {"input": 200, "output": 80}  # two batches of ≤ 25
    stored = client.get("/v1/news/FAKENEWS").json()
    labels = {h["label"] for h in stored["items"]}
    assert labels == {"positive", "negative", "neutral"}
    assert stored["items"][0]["sentiment"]["model"] == "scripted:rater-1"

    after = _run(client, "FAKENEWS")["payload"]
    assert after["headlines_scored"] == 40
    assert after["sessions_with_news"] >= 20
    assert len(after["dates"]) == len(after["sentiment_index"]) == len(after["headline_count"])
    assert sum(after["headline_count"]) <= 40
    assert after["corr_next_session"]["n"] == after["sessions_with_news"] - (
        1 if after["daily_sentiment"][-1] is not None else 0
    )
    assert after["positive_share"] + after["negative_share"] + after["neutral_share"] == 1.0

    digest = client.post("/v1/news/FAKENEWS/digest", json={"days": 90}).json()
    assert len(digest["sources"]) == 1
    assert digest["unknown_sources"] == ["ffffffffff"]  # an invented id is reported
    assert digest["uncited_lines"] == 1
    assert _run(client, "FAKENEWS")["payload"]["digest"]["sources"] == digest["sources"]
    runs = client.get("/v1/agent/runs").json()
    assert {r["question"].split("]")[0] for r in runs} == {"[news sentiment", "[news digest"}


def test_invalid_symbols_and_unknown_news(client: TestClient) -> None:
    assert client.post("/v1/news/AAPL/ingest").status_code == 404  # offline: FAKE* only
    assert client.post("/v1/news/..%2Fx/ingest").status_code in (404, 422)


def test_parse_scores() -> None:
    assert news.parse_scores('Here: [{"i": 1, "score": 3}, {"i": 2, "score": -0.2}]', 2) == {
        1: 1.0,
        2: -0.2,
    }
    assert news.parse_scores('[{"i": 9, "score": 0.5}]', 2) == {}
    assert news.parse_scores("no json", 2) == {}


def test_budget_stops_scoring_between_batches(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    client.post("/v1/ingest/FAKENEWB")
    client.post("/v1/news/FAKENEWB/ingest")
    monkeypatch.setattr(news, "get_provider", lambda name=None: RaterProvider())
    monkeypatch.setenv("FDA_LLM_MONTHLY_TOKENS", "100")  # one batch (140 tokens) passes
    refused = client.post("/v1/news/FAKENEWB/score", json={})
    assert refused.status_code == 400
    assert "budget" in refused.json()["detail"]
    stored = client.get("/v1/news/FAKENEWB").json()["items"]
    assert sum(1 for h in stored if h["sentiment"]) == 25  # the paid batch is kept
    assert client.get("/v1/agent/usage").json()["tokens"] == 140


def test_only_web_links_are_stored() -> None:
    from fda_data.news import YahooNews

    class Ticker:
        def __init__(self, symbol: str) -> None:
            self.news = [
                {"title": "Good", "link": "javascript:alert(1)", "providerPublishTime": 1},
                {"title": "Fine", "link": "https://example.com/a", "providerPublishTime": 2},
            ]

    import yfinance

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(yfinance, "Ticker", Ticker)
        urls = [h.url for h in YahooNews().headlines("AAPL")]
    assert urls == ["", "https://example.com/a"]


def test_grouped_citations_and_headings() -> None:
    text = "**Products**\n* Launch [h:aaaaaaaaaa, h:bbbbbbbbbb].\n* Record [h:cccccccccc]"
    assert news.cited_ids(text) == ["aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc"]
    assert news.HEADING.match("**Products**")
    assert news.HEADING.match("## Products")
    assert not news.HEADING.match("* Launch [h:aaaaaaaaaa]")
