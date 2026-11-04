"""AI-007 gold set at the tool layer: 50 questions → tool path → value in the LLM summary, against
a frozen snapshot of the offline fake provider (fixed "today"). Also the citations verifier: a
cited correct answer verifies, a wrong or uncited number does not.

Regenerate the snapshot after an intended formula change (and bump the analysis version):
    FDA_GOLD_UPDATE=1 uv run pytest apps/api/tests/test_gold_set.py
"""

import json
import os
from datetime import date
from typing import Any

import pytest
from fastapi.testclient import TestClient

import fda_data.ingest as ingest
from fda_api.evaluation import GOLD_DIR, load_answers, load_gold, pick, score, verify
from fda_api.tools import all_tools


class FixedDate(date):
    @classmethod
    def today(cls) -> "FixedDate":
        return cls(2025, 6, 30)


def _fmt(value: float, area: str) -> str:
    if isinstance(value, int) or abs(value) >= 100:
        return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:g}"
    return (
        f"{value * 100:.2f} %"
        if abs(value) < 1 and area in ("performance", "risk")
        else f"{value:.4f}"
    )


def test_gold_set(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ingest, "date", FixedDate)
    client.post("/v1/ingest/FAKEGOLD")
    gold = load_gold()
    assert len(gold) >= 50
    names = {t.analysis_id: t.name for t in all_tools() if t.analysis_id}
    expected = load_answers()
    update = os.environ.get("FDA_GOLD_UPDATE") == "1"
    found: dict[str, float] = {}
    correct, failures = 0, []
    for item in gold:
        arguments = {
            "subject_type": item["subject"]["type"],
            "subject_id": item["subject"]["id"],
            "range": item["range"],
            "params": item["params"],
        }
        out: dict[str, Any] = client.post(
            f"/v1/tools/{names[item['analysis']]}:call", json={"arguments": arguments}
        ).json()
        assert out["is_error"] is False, (item["id"], out)
        value = pick(out["content"], item["path"])
        found[item["id"]] = value
        if update:
            continue
        if score(value, expected[item["id"]], rel=item.get("rel_tol", 1e-6)):
            correct += 1
        else:
            failures.append((item["id"], item["analysis"], value, expected[item["id"]]))
        # the verifier accepts the value as an answer would print it, cited to this result
        rid = out["content"]["result_id"]
        text = f"{item['question'].rstrip('?')}: {_fmt(value, item['area'])} [r:{rid}]."
        check = verify(text, {rid: out["content"]})
        assert check.unverified == [], (item["id"], text)
    if update:
        (GOLD_DIR / "gold_answers.json").write_text(json.dumps(found, indent=1) + "\n", "utf-8")
        pytest.skip("gold snapshot regenerated")
    assert correct / len(gold) == 1.0, failures


def test_verifier_catches_wrong_and_uncited_numbers() -> None:
    results = {
        "abc123def456": {"summary": {"value": 0.390172, "max_drawdown": -0.196985, "count": 41}}
    }
    ok = verify(
        "CAGR was 39.02 % [r:abc123def456]; max drawdown 19.7 % with 41 episodes [r:abc123def456].",
        results,
    )
    assert ok.unverified == []
    assert ok.citation_coverage == 1.0
    wrong = verify("CAGR was 41.5 % [r:abc123def456].", results)
    assert wrong.unverified == ["41.5 %"]
    uncited = verify("Since 2019 the CAGR was 39.02 %.", results)
    assert uncited.citation_coverage == 0.0
    assert uncited.unverified == ["39.02 %"]  # the year is ignored, the number is not


def test_numbers_from_the_question_are_inputs() -> None:
    results = {"abc123def456": {"summary": {"participation": 0.003386}}}
    answer = "A 10,000-share order is 0.34 % of ADV [r:abc123def456]; that is 10 001 shares."
    check = verify(answer, results, given="What would a 10 000-share order represent?")
    assert check.unverified == ["10 001"]  # restating 10 000 is fine, a new number is not
