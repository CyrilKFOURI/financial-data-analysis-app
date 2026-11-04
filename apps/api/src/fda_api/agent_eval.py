"""Agent-level evaluation (PRD §7 V6, M30 §7): the 50-question gold set asked to the real agent.

Each question goes through `/v1/agent/ask` as from the console opened on the item's subject
(page context: subject and range, as the console sends them), against the frozen
fake-provider snapshot of the tool-level gold set (fixed "today", throw-away SQLite DB). An answer
passes when:
- it is numerically correct: one of its numbers equals the gold value at the precision it is
  written in (39.02 % for 0.390172), or within the item's `rel_tol`;
- it is fully cited: every number verifies against the result it cites.

The V6 gate (PRD §7) asks for ≥ 90 % passing answers and zero uncited numbers (a number
without any citation) over the whole set. Cited numbers that do not verify (computed by the
model, or from another result) fail their answer and are reported separately.

    uv run python -m fda_api.agent_eval --provider gemini --model gemini-3.5-flash
    uv run python -m fda_api.agent_eval --provider ollama --limit 5 --out eval.json

Runs spend LLM credits (or local compute): tests use a scripted model, never this CLI.
"""

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from fda_api.evaluation import Mention, _matches, load_answers, load_gold, verify

TODAY = date(2025, 6, 30)  # the snapshot date of gold_answers.json


@dataclass(slots=True)
class Graded:
    id: str
    question: str
    expected: float
    answer: str = ""
    correct: bool = False
    cited: bool = False
    numbers: int = 0
    uncited: int = 0
    unverified: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    model_rounds: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.correct and self.cited and self.error is None


def _close(m: Mention, expected: float, rel_tol: float | None) -> bool:
    if _matches(m, [expected]):
        return True
    if rel_tol is None:
        return False
    value = m.value * (0.01 if m.percent else 1.0)
    return abs(abs(value) - abs(expected)) <= rel_tol * abs(expected)


def grade(
    item: dict[str, Any], expected: float, events: list[dict[str, Any]], seconds: float = 0.0
) -> Graded:
    """Score one agent run (its NDJSON events) against the gold value."""
    from fda_api.agent import checkable, unwrap

    out = Graded(item["id"], item["question"], expected, seconds=round(seconds, 1))
    for event in events:
        if event["type"] == "tool_result":
            out.tools.append(unwrap(event["tool"], event["arguments"])[0])
        elif event["type"] == "model":
            out.model_rounds += 1
        elif event["type"] == "error":
            out.error = str(event["detail"])
        elif event["type"] == "answer":
            out.answer = event["text"]
            v = event["verification"]
            out.numbers, out.unverified = v["numbers"], list(v["unverified"])
            out.uncited = int(v["numbers"]) - int(v["cited"])
            out.cited = v["numbers"] > 0 and not v["unverified"] and v["coverage"] == 1.0
            out.tokens_in = int(event["usage"].get("input", 0))
            out.tokens_out = int(event["usage"].get("output", 0))
    if out.answer:
        mentions = verify(checkable(out.answer), {}).mentions
        out.correct = any(_close(m, expected, item.get("rel_tol")) for m in mentions)
    return out


def summarise(graded: list[Graded], provider: str, model: str) -> dict[str, Any]:
    n = len(graded) or 1
    numbers = sum(g.numbers for g in graded)
    unverified = sum(len(g.unverified) for g in graded)
    uncited = sum(g.uncited for g in graded)
    passed = sum(g.passed for g in graded)
    return {
        "provider": provider,
        "model": model,
        "questions": len(graded),
        "passed": passed,
        "pass_rate": round(passed / n, 4),
        "correct": sum(g.correct for g in graded),
        "fully_cited": sum(g.cited for g in graded),
        "errors": sum(g.error is not None for g in graded),
        "numbers": numbers,
        "unverified_numbers": unverified,
        "uncited_numbers": uncited,
        "median_seconds": sorted(g.seconds for g in graded)[len(graded) // 2] if graded else 0,
        "mean_tool_calls": round(sum(len(g.tools) for g in graded) / n, 2),
        "tokens_in": sum(g.tokens_in for g in graded),
        "tokens_out": sum(g.tokens_out for g in graded),
        "gate": passed / n >= 0.9 and uncited == 0,
    }


def markdown(summary: dict[str, Any], graded: list[Graded]) -> str:
    s = summary
    lines = [
        f"# Agent evaluation · {s['provider']} · `{s['model']}`",
        "",
        f"- Passed (correct and fully cited): **{s['passed']} / {s['questions']}** "
        f"({s['pass_rate'] * 100:.0f} %) · gate ≥ 90 % and zero uncited numbers: "
        f"**{'met' if s['gate'] else 'not met'}**",
        f"- Numerically correct {s['correct']} · fully cited {s['fully_cited']} · "
        f"errors {s['errors']} · uncited numbers {s['uncited_numbers']} · cited but unverified "
        f"{s['unverified_numbers']} of {s['numbers']}",
        f"- Median {s['median_seconds']} s per question · {s['mean_tool_calls']} tool calls · "
        f"{s['tokens_in']:,} input / {s['tokens_out']:,} output tokens",
        "",
        "| id | pass | correct | cited | tools | s | note |",
        "|---|---|---|---|---|---|---|",
    ]
    for g in graded:
        note = g.error or (", ".join(g.unverified) if g.unverified else "")
        lines.append(
            f"| {g.id} | {'✓' if g.passed else '✗'} | {'✓' if g.correct else '✗'} | "
            f"{'✓' if g.cited else '✗'} | {', '.join(g.tools)} | {g.seconds} | "
            f"{note.replace('|', '/')[:120]} |"
        )
    return "\n".join(lines) + "\n"


def run(
    provider: str | None, model: str | None, limit: int | None, pause: float
) -> tuple[dict[str, Any], list[Graded]]:
    """Fresh SQLite DB + lake, offline fake data frozen at TODAY, then every gold question."""
    workdir = Path(tempfile.mkdtemp(prefix="fda-agent-eval-"))
    database_url = f"sqlite:///{(workdir / 'eval.db').as_posix()}"
    os.environ.update(
        {
            "DATABASE_URL": database_url,
            "DATA_DIR": str(workdir / "data"),
            "REDIS_URL": "",
            "FDA_PROVIDERS": "fake",
            "FDA_JOB_RUNNER": "inline",
            "FDA_AUTH_MODE": "single",
        }
    )
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine

    import fda_api.agent as agent_module
    import fda_data.ingest as ingest
    from fda_api.main import app
    from fda_api.security import CLIENT_HEADER
    from fda_data.schema import metadata

    class Frozen(date):
        @classmethod
        def today(cls) -> "Frozen":
            return cls(TODAY.year, TODAY.month, TODAY.day)

    ingest.date = Frozen  # type: ignore[misc]
    agent_module.date = Frozen  # type: ignore[misc]
    engine = create_engine(database_url)
    metadata.create_all(engine)
    engine.dispose()

    gold, answers = load_gold(), load_answers()
    if limit:
        gold = gold[:limit]
    graded: list[Graded] = []
    with TestClient(app, headers={CLIENT_HEADER: "agent-eval"}) as client:
        client.post("/v1/ingest/FAKEGOLD")
        used_model = model or ""
        for n, item in enumerate(gold, 1):
            # the page context the console would send: the subject and its range
            page = {**item["subject"], "range": item["range"]}
            body = {
                "question": item["question"],
                "provider": provider,
                "model": model,
                "subject": page,
            }
            started = time.perf_counter()
            response = client.post("/v1/agent/ask", json=body)
            events = [json.loads(x) for x in response.text.splitlines() if x.strip()]
            for event in events:
                if event["type"] == "start":
                    provider, used_model = event["provider"], event["model"]
            g = grade(item, answers[item["id"]], events, time.perf_counter() - started)
            graded.append(g)
            mark = "pass" if g.passed else ("error" if g.error else "fail")
            print(f"[{n}/{len(gold)}] {g.id} {mark} {g.seconds}s {g.error or ''}", flush=True)
            if pause and n < len(gold):
                time.sleep(pause)
    return summarise(graded, provider or "", used_model), graded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--provider", help="anthropic, gemini, openai, ollama… (default: first)")
    parser.add_argument("--model", help="model id (default: the provider's default)")
    parser.add_argument("--limit", type=int, help="first N questions only")
    parser.add_argument("--pause", type=float, default=0.0, help="seconds between questions")
    parser.add_argument("--out", type=Path, help="write JSON results here (and .md alongside)")
    args = parser.parse_args(argv)
    summary, graded = run(args.provider, args.model, args.limit, args.pause)
    print(json.dumps(summary, indent=1))
    if args.out:
        payload = {"summary": summary, "items": [asdict(g) for g in graded]}
        args.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", "utf-8")
        args.out.with_suffix(".md").write_text(markdown(summary, graded), "utf-8")
    return 0 if summary["gate"] else 1


if __name__ == "__main__":
    sys.exit(main())
