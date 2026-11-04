"""J26 slice 262 workflows offline: cited briefs, verifier-clean highlights, schedule and tool."""

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from fda_api.evaluation import verify


def _portfolio(client: TestClient) -> int:
    for symbol in ("FAKE", "FAKEB"):
        client.post(f"/v1/ingest/{symbol}")
    holdings = [{"symbol": "FAKE", "weight": 0.6}, {"symbol": "FAKEB", "weight": 0.4}]
    body = {"name": "Review", "mode": "weights", "holdings": holdings}
    return int(client.post("/v1/portfolios", json=body).json()["id"])


def _results(brief: dict[str, Any]) -> dict[str, Any]:
    return {s["result_id"]: s["summary"] for s in brief["sections"] if "result_id" in s}


def test_portfolio_review_brief(client: TestClient) -> None:
    pid = _portfolio(client)
    snap = {
        "subject_type": "portfolio",
        "subject_id": str(pid),
        "label": "base",
        "analyses": ["R-020"],
    }
    assert client.post("/v1/snapshots", json=snap).status_code == 201
    rule = {
        "name": "Vol watch",
        "subject_type": "portfolio",
        "subject_id": str(pid),
        "analysis_id": "R-020",
        "path": "volatility",
        "op": ">",
        "threshold": 0.0,
    }
    assert client.post("/v1/alerts", json=rule).status_code == 201
    client.post("/v1/alerts/evaluate")

    listed = {w["name"] for w in client.get("/v1/workflows").json()}
    assert listed == {"portfolio_review", "earnings_preview"}
    brief = client.post("/v1/workflows/portfolio_review:run", json={"subject_id": str(pid)}).json()
    assert brief["subject"] == {"type": "portfolio", "id": str(pid)}
    assert {s["analysis_id"] for s in brief["sections"]} == {
        "P-003",
        "P-004",
        "R-020",
        "P-008",
        "P-010",
        "P-018",
    }
    assert all("result_id" in s or "problem" in s for s in brief["sections"])
    assert any(h.startswith("Annualised volatility") for h in brief["highlights"])
    assert any("Vol watch" in h for h in brief["highlights"])
    assert brief["alerts"][0]["triggered"] is True
    # every number of the highlights is backed by the result it cites
    check = verify("\n".join(brief["highlights"]), _results(brief))
    assert check.unverified == [], check.unverified
    assert check.cited >= 2
    assert check.citation_coverage == 1.0

    assert client.post("/v1/workflows/nope:run", json={"subject_id": str(pid)}).status_code == 404
    assert (
        client.post("/v1/workflows/portfolio_review:run", json={"subject_id": "999"}).status_code
        == 404
    )


def test_earnings_preview_and_schedule(client: TestClient) -> None:
    pid = _portfolio(client)
    for symbol in ("FAKE", "FAKEB"):
        client.post(f"/v1/company-data/{symbol}/ingest")
    brief = client.post(
        "/v1/workflows/earnings_preview:run", json={"subject_id": str(pid), "horizon_days": 120}
    ).json()
    assert [r["symbol"] for r in brief["holdings"]] != []
    assert {r["symbol"] for r in brief["holdings"]} == {"FAKE", "FAKEB"}
    for row in brief["holdings"]:
        assert {"next_report", "problem", "note"} & set(row)  # explained when data is missing
    check = verify("\n".join(brief["highlights"]), _results(brief))
    assert check.unverified == [], check.unverified

    made = client.post(
        "/v1/schedules",
        json={
            "name": "Weekly review",
            "kind": "workflow",
            "cron": "0 7 * * 1",
            "spec": {
                "workflow": "portfolio_review",
                "subject": str(pid),
                "email": "me@example.com",
            },
        },
    )
    assert made.status_code == 201, made.text
    ran = client.post(f"/v1/schedules/{made.json()['id']}/run").json()
    assert ran["status"] == "succeeded", ran
    assert Path(ran["result"]["file"]).read_text("utf-8").startswith("<!doctype html>")
    assert Path(ran["result"]["file"]).with_suffix(".json").exists()
    assert "SMTP_HOST" in ran["result"]["delivery"]
    bad = client.post(
        "/v1/schedules",
        json={"name": "Bad", "kind": "workflow", "cron": "0 7 * * 1", "spec": {"workflow": "x"}},
    )
    assert bad.status_code == 422

    tool = client.post(
        "/v1/tools/run_workflow:call",
        json={"arguments": {"workflow": "portfolio_review", "portfolio_id": pid}},
    ).json()
    assert tool["is_error"] is False
    assert tool["content"]["highlights"]
    missing = client.post(
        "/v1/tools/run_workflow:call",
        json={"arguments": {"workflow": "portfolio_review", "portfolio_id": 999}},
    ).json()
    assert missing["is_error"] is True


class Narrator:
    """Writes one sentence from the first highlight (its figure and citation) and one invented
    figure, like a model that partly hallucinates."""

    def __init__(self, system: str) -> None:
        self.system = system

    def send(self, text: str | None = None, results: Any = None) -> Any:
        import json as _json

        from fda_api.llm import Reply

        brief = _json.loads(text or "{}")
        first = brief["highlights"][0]
        return Reply(
            f"{first}\nThe portfolio also gained 99.9 % [r:000000000000].", [], {"output": 9}
        )


class NarratorProvider:
    name, label, local, key = "scripted", "Scripted", False, None

    def model(self, requested: str | None = None) -> str:
        return "narrator-1"

    def session(self, model: str, system: str, *args: Any) -> Narrator:
        return Narrator(system)


def test_narration_is_verified_and_recorded(client: TestClient, monkeypatch: Any) -> None:
    from fda_api import narrate

    pid = _portfolio(client)
    monkeypatch.setattr(narrate, "get_provider", lambda name=None: NarratorProvider())
    brief = client.post(
        "/v1/workflows/portfolio_review:run", json={"subject_id": str(pid), "narrate": True}
    ).json()
    story = brief["narrative"]
    assert story["model"] == "scripted:narrator-1"
    assert story["verification"]["unverified"] == ["99.9 %"]  # the invented figure only
    assert story["verification"]["numbers"] >= 2
    run = client.get("/v1/agent/runs").json()[0]
    assert run["question"].startswith("[narration] Weekly portfolio review")
    from fda_api.workflows import render_html

    page = render_html(brief)
    assert "Narrative" in page
    assert "Unverified figures: 99.9 %" in page
