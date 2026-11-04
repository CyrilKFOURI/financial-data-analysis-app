"""J26 slice 262, LLM part: a short narrative over a workflow brief (M30 §6).

The brief stays the report of record. The narrative is written by the user's LLM from the brief
only (highlights + section summaries with their result ids), and goes through the same citations
verifier as the console: every figure must appear in the result it cites. Unverified figures are
returned and flagged in the HTML report; the call is logged in `agent_runs` with its usage and
counts against the monthly budget.
"""

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Engine, insert

from fda_api.evaluation import verify
from fda_api.llm import LLMError, get_provider
from fda_api.llm.costs import check_budget, cost
from fda_api.llm.providers import model_list
from fda_data.actor import current_actor
from fda_data.schema import agent_runs

SYSTEM = """You turn a workflow brief from an analytics app into a short narrative (120 to 200
words, 2 or 3 paragraphs) for its owner. Use only figures that appear in the brief and cite each
one right after it as [r:RESULT_ID], with the result_id of the section it comes from. Describe
measurements; no buy/sell verdicts, no price targets, no forecasts. Results labelled as model
output stay labelled as such. If a section reports a problem, mention it briefly. Write in
English."""
MAX_BRIEF_CHARS = 14000


def _compact(brief: dict[str, Any]) -> str:
    sections = [
        {k: s.get(k) for k in ("analysis_id", "title", "result_id", "summary", "problem")}
        for s in brief.get("sections") or []
    ]
    payload = {
        "title": brief.get("title"),
        "subject": brief.get("subject"),
        "range": brief.get("range"),
        "highlights": brief.get("highlights"),
        "sections": sections,
    }
    return json.dumps(payload, default=str)[:MAX_BRIEF_CHARS]


def narrate_brief(
    engine: Engine, brief: dict[str, Any], provider_name: str | None, model_name: str | None
) -> dict[str, Any]:
    """Narrative + verification for a brief, as the current user (never a viewer)."""
    from fda_api.agent import checkable  # the agent module imports the tool layer

    actor = current_actor()
    if actor is None or actor.role == "viewer":
        raise LLMError("Narration needs a signed-in analyst (it spends LLM credits)")
    provider = get_provider(provider_name)
    model = provider.model(model_name)
    listed = model_list(provider)
    if model_name and listed and model_name not in listed:
        raise LLMError(f"{provider.label} does not offer the model {model_name!r}")
    if not model:
        raise LLMError(f"{provider.label}: choose a model")
    check_budget(engine, actor.user_id)
    reply = provider.session(model, SYSTEM, [], []).send(_compact(brief), None)
    results = {
        s["result_id"]: s.get("summary") or {}
        for s in brief.get("sections") or []
        if isinstance(s.get("result_id"), str)
    }
    check = verify(checkable(reply.text), results)
    verification = {
        "numbers": check.numbers,
        "cited": check.cited,
        "unverified": check.unverified,
        "coverage": round(check.citation_coverage, 4),
    }
    with engine.begin() as conn:
        conn.execute(
            insert(agent_runs).values(
                owner_id=actor.user_id,
                provider=provider.name,
                model=model,
                question=f"[narration] {brief.get('title')} · {brief.get('subject', {}).get('id')}",
                answer=reply.text,
                steps=[],
                usage=reply.usage,
                verification=verification,
                duration_ms=0,
                created_at=datetime.now(UTC),
            )
        )
    return {
        "text": reply.text.strip(),
        "verification": verification,
        "model": f"{provider.name}:{model}",
        "usage": reply.usage,
        "cost": cost(provider.name, reply.usage),
        "label": "LLM narrative of the brief above; figures checked against their citations.",
    }
