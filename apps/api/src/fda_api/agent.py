"""J25 slices 251–252: the analyst agent, provider-neutral (M30 §3–4).

The loop:
1. The model gets the behaviour contract (system prompt), the core tools and `call_tool`.
   `call_tool` reaches any of the ~260 registry tools found with `search_tools`; this is the
   provider-neutral form of tool search.
2. Every tool runs as the signed-in user, through the same layer as MCP:
   - rows are scoped to that user;
   - write tools only propose actions, which the user approves in the web app;
   - V4/V5 results carry their validation evidence (J26 guardrails).
3. The final answer goes through the citations verifier: every number must appear in the
   result it cites `[r:result_id]`. Unverified numbers are returned and stored with the run.
4. A run is bounded by `MAX_STEPS` tool rounds. Token usage is recorded per run (`agent_runs`).
"""

import functools
import json
import re
import time
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime
from typing import Any

import anyio
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import insert, select

from fda_api.evaluation import verify
from fda_api.llm import LLMError, Reply, ToolCall, ToolSpec, get_provider, provider_status
from fda_api.llm.costs import check_budget, cost, month_usage
from fda_api.llm.providers import model_list
from fda_api.tools import CORE_TOOLS, ToolError, call_tool, tools_by_name
from fda_core.registry import REGISTRY
from fda_data.actor import current_actor
from fda_data.schema import agent_runs

router = APIRouter(prefix="/v1/agent", tags=["agent"])

MAX_STEPS = 10
RANGE = re.compile(r"^(?:\d{1,2}[MY]|YTD|MAX)$")  # page context, never free text in the prompt
MAX_RESULT_CHARS = 6000
MAX_HISTORY = 20
EXPOSED = (*sorted(CORE_TOOLS), "run_workflow")

SYSTEM = """You are a senior financial analyst and quant working inside the FDA Workbench, an
analytics app. Today is {today}. The user's role is {role}.

Rules (non-negotiable):
- Numbers come only from tool results obtained in this conversation. Cite every number with the
  result it comes from, written exactly as [r:RESULT_ID] right after the number, using the
  `result_id` field of that tool result. If no tool can provide a number, say so: never estimate.
- Pick the analysis from the index below and call it directly with call_tool, with the
  ticker the user wrote and empty params (they default to the expert settings): do not look the
  instrument up, check data status or explain the analysis first. Use search_tools only when the
  index does not cover the question, and the lookup tools only after an error. Analysis tools take
  {{"subject_type": "asset"|"portfolio"|"comparison"|"strategy"|"profile"|"paper"|"universe",
  "subject_id": "...", "range": "1Y", "params": {{}}}}; use explain_analysis for their parameters.
  If data is missing, check get_data_status and request_ingestion.
- Measurements (risk, returns, statements, valuation, time series) are descriptions: no buy/sell
  verdicts, no price targets. Third-party data and opinions are labelled as such.
- Investment and trading results (IN-, TR-) are model outputs: state their assumptions and the
  out-of-sample evidence attached to them (DSR, PBO, walk-forward). Never present a backtest or
  an optimisation without that evidence; if performance was withheld, say why.
- Write tools only propose an action that the user must approve in the web app (Agent actions
  page). Say so; never claim an order or change was executed. There is no live trading.
- Be concise and dense: short paragraphs or a table, then the page to open (ui_ref) when useful.
- Answer in the user's language.
{context}

Analysis index (tool name: title [subjects]):
{index}"""

REPAIR = """Citation check: these numbers in your answer are not backed by the tool result they
cite (or cite none): {numbers}. Rewrite your whole answer: keep only numbers that appear in a tool
result of this conversation, each followed by its [r:RESULT_ID] in the same sentence; drop numbers
you computed, converted or recalled, and side notes with uncited figures. Do not call tools.
Reply with the corrected answer only."""

CALL_TOOL = ToolSpec(
    "call_tool",
    "Call any workbench tool by name (find names with search_tools). `arguments` follows that "
    "tool's input schema; analysis tools take subject_type, subject_id, range and params.",
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Tool name, e.g. r_volatility"},
            "arguments": {"type": "object", "description": "The tool's arguments"},
        },
        "required": ["name", "arguments"],
    },
)


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    text: str = Field(max_length=20000)


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    provider: str | None = Field(None, max_length=32)
    model: str | None = Field(None, max_length=120)
    history: list[Turn] = Field(default_factory=list, max_length=MAX_HISTORY)
    subject: dict[str, str] | None = None


def tool_specs() -> list[ToolSpec]:
    tools = tools_by_name()
    specs = [ToolSpec(n, tools[n].description, tools[n].input_schema) for n in EXPOSED]
    return [*specs, CALL_TOOL]


def _short(text: str, limit: int = 80) -> str:
    """The head of a description, cut at a word: what the analysis outputs, in a few words."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0].rstrip(",;:(") + "…"


@functools.cache
def analysis_index() -> str:
    """Every analysis tool in one line each: lets the model call the right one at once instead
    of spending model round trips on search_tools / explain_analysis (each costs seconds)."""
    lines = []
    for tool in tools_by_name().values():
        if tool.analysis_id:
            subjects = ", ".join(REGISTRY.get(tool.analysis_id).applies_to)
            hint = _short(REGISTRY.get(tool.analysis_id).description)
            lines.append(f"{tool.name}: {tool.title} — {hint} [{subjects}]")
    return "\n".join(sorted(lines))


def system_prompt(subject: dict[str, str] | None) -> str:
    actor = current_actor()
    context = ""
    if subject and subject.get("type") and subject.get("id"):
        context = f"\nThe user is looking at {subject['type']} {subject['id']} in the app."
        if RANGE.match(subject.get("range") or ""):
            context += (
                f" The page's range is {subject['range']}: use it unless the question asks "
                "for another one."
            )
    return SYSTEM.format(
        today=date.today().isoformat(),
        role=actor.role if actor else "admin",
        context=context,
        index=analysis_index(),
    )


def unwrap(name: str, arguments: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """The tool really meant by a `call_tool` call. Some models wrap it again:
    call_tool(name="call_tool", arguments={name, arguments}) or name=X with arguments={name: X,
    arguments: {...}}; up to three levels are unwrapped."""
    for _ in range(3):
        inner = arguments.get("arguments")
        wrapped = set(arguments) <= {"name", "arguments"} and isinstance(inner, dict)
        if name == "call_tool" or (wrapped and arguments.get("name") in (name, None)):
            name = str(arguments.get("name") or name)
            arguments = inner if isinstance(inner, dict) else {}
        else:
            break
    return name, arguments


async def _execute(request: Request, call: ToolCall) -> tuple[dict[str, Any], bool]:
    name, arguments = unwrap(call.name, call.arguments)
    try:
        out = await call_tool(request, name, arguments, "agent")
    except ToolError as exc:
        return {"tool": name, "error": str(exc)}, True
    except HTTPException as exc:
        return {"tool": name, "error": str(exc.detail)}, True
    content = out.get("content")
    return (content if isinstance(content, dict) else {"result": content}), bool(out["is_error"])


def _collect(content: dict[str, Any], results: dict[str, Any]) -> None:
    """Every result_id in a tool output (analysis results, evidence, workflow sections)."""
    if isinstance(content.get("result_id"), str):
        results[content["result_id"]] = content
    evidence = content.get("evidence")
    if isinstance(evidence, dict) and isinstance(evidence.get("result_id"), str):
        results[evidence["result_id"]] = evidence
    for section in content.get("sections") or []:
        if isinstance(section, dict) and isinstance(section.get("result_id"), str):
            results[section["result_id"]] = section


# numbers that are names or addresses, not figures: links, index names, years in labels
NOT_FIGURES = re.compile(
    r"\]\([^)]*\)|https?://\S+|\b(?:S&P|FTSE|CAC|DAX|STOXX(?: Europe)?|Russell|Nasdaq|"
    r"NASDAQ|Nikkei|Euro Stoxx|MSCI \w+|Dow(?: Jones)?)[ -]?\d+\b"
)


# dates and identifiers carry digits that are not figures either
MONTHS = (
    r"(?:jan|feb|fév|fev|mar|apr|avr|may|mai|jun|juin|jul|juil|aug|août|aou|sep|oct|nov|dec|déc)"
    r"[a-zéû]*\.?"
)
DATES = re.compile(
    r"\b\d{1,2}[/.]\d{1,2}[/.]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b|"
    rf"\b{MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b|"
    rf"\b\d{{1,2}}(?:er)?\s+{MONTHS}\s+\d{{4}}\b",
    re.IGNORECASE,
)
IDS = re.compile(
    r"\b(?:portfolio|portefeuille|strategy|stratégie|strategie|account|compte|universe|univers|"
    r"version|run|#)\s*#?\d+\b",
    re.IGNORECASE,
)
# the horizon asked for ("over the last 3 years", "sur 1 an") is a parameter, not a result
HORIZON = re.compile(
    r"\b\d+[\s-]*(?:ans?|années?|years?|yrs?)\b|\b\d+[YM]\b|\b\d+[\s-]*(?:mois|months?|"
    r"semaines?|weeks?|days?|jours?)[\s-]+(?:period|window|horizon|période|fenêtre)\b|"
    r"\b(?:last|past|derni\w+)\s+\d+\s*(?:mois|months?|semaines?|weeks?)\b",
    re.IGNORECASE,
)
# list numbering at the start of a line ("1. ", "2) ")
ORDINALS = re.compile(r"^\s*\d{1,2}[.)]\s", re.MULTILINE)
# ranks written as words: "75th percentile", "1st quartile", "3e quartile"
RANKS = re.compile(r"\b\d{1,3}(?:st|nd|rd|th|e|ème|er)\b", re.IGNORECASE)
# a decimal comma (15,01 · 155,27) — but not a thousands comma (10,000)
DECIMAL_COMMA = re.compile(r"(?<=\d),(?=\d{1,2}(?!\d)|\d{4,})")


def checkable(answer: str) -> str:
    """The answer as the verifier should read it: link targets, index names, dates and ids
    removed (their digits are not figures), decimal commas read as points."""
    text = NOT_FIGURES.sub(lambda m: "]" if m.group(0).startswith("]") else "", answer)
    text = DATES.sub(" ", text)
    text = IDS.sub(" ", text)
    text = HORIZON.sub(" ", text)
    text = ORDINALS.sub(" ", text)
    text = RANKS.sub(" ", text)
    return DECIMAL_COMMA.sub(".", text)


def _line(event: dict[str, Any]) -> str:
    return json.dumps(event, default=str, ensure_ascii=False) + "\n"


async def run_agent(request: Request, body: AskIn) -> AsyncIterator[str]:
    started = time.perf_counter()
    actor = current_actor()
    if actor is None:  # never a system run: every run belongs to the user who asked
        yield _line({"type": "error", "detail": "Sign in to use the agent."})
        return
    if actor.role == "viewer":  # agent runs spend paid LLM credits
        yield _line(
            {"type": "error", "detail": "Viewers cannot run the agent (it spends LLM credits)."}
        )
        return
    try:
        provider = get_provider(body.provider)
        model = provider.model(body.model)
        listed = model_list(provider)
        if body.model and listed and body.model not in listed:
            raise LLMError(f"{provider.label} does not offer the model {body.model!r}")
        if not model:
            env = f"FDA_LLM_MODEL_{provider.name.upper()}"
            raise LLMError(f"{provider.label}: choose a model (or set {env} in .env)")
        check_budget(request.app.state.engine, actor.user_id)
        session = provider.session(
            model,
            system_prompt(body.subject),
            tool_specs(),
            [(t.role, t.text) for t in body.history],
        )
    except LLMError as exc:
        yield _line({"type": "error", "detail": str(exc)})
        return
    yield _line({"type": "start", "provider": provider.name, "model": model})
    results: dict[str, Any] = {}
    steps: list[dict[str, Any]] = []
    usage: dict[str, int] = {}
    reply: Reply | None = None
    pending: list[tuple[ToolCall, str]] | None = None
    text_in: str | None = body.question
    try:
        for _ in range(MAX_STEPS + 1):
            tick = time.perf_counter()
            reply = await anyio.to_thread.run_sync(session.send, text_in, pending)
            yield _line({"type": "model", "ms": round((time.perf_counter() - tick) * 1000)})
            text_in, pending = None, []
            for key, value in reply.usage.items():
                usage[key] = usage.get(key, 0) + value
            if not reply.tool_calls:
                break
            for call in reply.tool_calls:
                yield _line({"type": "tool_call", "name": call.name, "arguments": call.arguments})
                tick = time.perf_counter()
                content, is_error = await _execute(request, call)
                _collect(content, results)
                step = {
                    "tool": call.name,
                    "arguments": call.arguments,
                    "is_error": is_error,
                    "result_id": content.get("result_id"),
                    "ui_ref": content.get("ui_ref"),
                    "ms": round((time.perf_counter() - tick) * 1000),
                }
                steps.append(step)
                yield _line({"type": "tool_result", **step})
                pending.append((call, json.dumps(content, default=str)[:MAX_RESULT_CHARS]))
        else:
            reply = await anyio.to_thread.run_sync(
                session.send,
                "Tool budget reached: answer now with what you have, citing results.",
                pending,
            )
    except LLMError as exc:
        yield _line({"type": "error", "detail": str(exc), "steps": steps})
        return
    answer = reply.text if reply else ""
    given = checkable(body.question)
    check = verify(checkable(answer), results, given=given)
    if check.unverified and results:
        # one correction round: name the unbacked numbers, keep the rewrite only if it is better
        tick = time.perf_counter()
        try:
            fixed = await anyio.to_thread.run_sync(
                session.send, REPAIR.format(numbers=", ".join(check.unverified)), None
            )
        except LLMError:
            fixed = None
        yield _line({"type": "model", "ms": round((time.perf_counter() - tick) * 1000)})
        if fixed is not None:
            for key, value in fixed.usage.items():
                usage[key] = usage.get(key, 0) + value
            again = verify(checkable(fixed.text), results, given=given)
            if fixed.text.strip() and len(again.unverified) < len(check.unverified):
                answer, check = fixed.text, again
            steps.append({"tool": "citation_check", "arguments": {}, "is_error": False,
                          "result_id": None, "ui_ref": None,
                          "ms": round((time.perf_counter() - tick) * 1000)})  # fmt: skip
    verification = {
        "numbers": check.numbers,
        "cited": check.cited,
        "unverified": check.unverified,
        "coverage": round(check.citation_coverage, 4),
    }
    elapsed = round((time.perf_counter() - started) * 1000)
    with request.app.state.engine.begin() as conn:
        run_id = conn.execute(
            insert(agent_runs).values(
                owner_id=actor.user_id,
                provider=provider.name,
                model=model,
                question=body.question,
                answer=answer,
                steps=steps,
                usage=usage,
                verification=verification,
                duration_ms=elapsed,
                created_at=datetime.now(UTC),
            )
        ).inserted_primary_key[0]
    yield _line(
        {
            "type": "answer",
            "run_id": int(run_id),
            "text": answer,
            "verification": verification,
            "citations": {rid: r.get("ui_ref") for rid, r in results.items()},
            "usage": usage,
            "cost": cost(provider.name, usage),
            "duration_ms": elapsed,
        }
    )


@router.get("/providers")
def get_providers() -> list[dict[str, Any]]:
    """Which LLM providers have a key in .env (or a running local server), and their model."""
    return provider_status()


@router.post("/ask")
async def ask(request: Request, body: AskIn) -> StreamingResponse:
    """NDJSON events: start, tool_call, tool_result, then answer (or error)."""
    return StreamingResponse(run_agent(request, body), media_type="application/x-ndjson")


@router.get("/runs")
def list_runs(request: Request) -> list[dict[str, Any]]:
    """Your own runs only (explicit here, on top of the row scoping and Postgres RLS)."""
    actor = current_actor()
    if actor is None:
        raise HTTPException(401, "Sign in to see agent runs")
    query = (
        select(agent_runs)
        .where(agent_runs.c.owner_id == actor.user_id)
        .order_by(agent_runs.c.id.desc())
        .limit(50)
    )
    with request.app.state.engine.connect() as conn:
        rows = [dict(r) for r in conn.execute(query).mappings()]
    return [r | {"cost": cost(r["provider"], r["usage"] or {})} for r in rows]


@router.get("/usage")
def get_usage(request: Request) -> dict[str, Any]:
    """This month's LLM tokens and estimated cost per model (yours), and the budget caps."""
    actor = current_actor()
    if actor is None:
        raise HTTPException(401, "Sign in to see LLM usage")
    return month_usage(request.app.state.engine, actor.user_id)
