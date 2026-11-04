"""V6 tool layer (M30 §2.1, AI-001): every registry analysis plus data, navigation and write
actions as typed, model-facing tools; the same definitions are served over MCP (`fda_api.mcp`).

Safety classes: `read` tools run immediately; `write` tools never execute on call — they record
a pending action that only the web app can approve (audit log `agent_actions`); live orders are
`forbidden` and have no tool at all. Results are shaped for an LLM: long series become summaries,
long tables are truncated, and every result carries a `result_id` for citations and a `ui_ref`
pointing at the subject's page.
"""

import hashlib
import json
import math
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import anyio
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import Engine, insert, select, update

import fda_core.analyses  # noqa: F401  (registers analyses)
from fda_api import guardrails
from fda_api.analyses import _run_one
from fda_api.resolve import Memo, _frame, _instrument
from fda_api.runner import run_analysis
from fda_contracts.envelopes import AnalysisRequestItem, RunContext, RunRequest, SubjectRef
from fda_core.registry import REGISTRY, AnalysisSpec, InsufficientDataError
from fda_data.actor import current_actor
from fda_data.paper import (
    PaperError,
    cancel_order,
    create_account,
    place_order,
    validate_order,
)
from fda_data.schema import agent_actions
from fda_data.strategies import StrategyError, create_strategy

router = APIRouter(prefix="/v1", tags=["tools"])

Safety = Literal["read", "write"]
SUBJECT_ROUTES = {
    "asset": "/a/{id}/overview",
    "portfolio": "/p/{id}/overview",
    "comparison": "/c/{id}/overview",
    "market": "/m/overview",
    "universe": "/u/{id}/screen",
    "profile": "/invest/{id}/allocation",
    "strategy": "/t/{id}/validation",
    "paper": "/paper/{id}/account",
}
CORE_TOOLS = {
    "search_tools",
    "search_instruments",
    "get_data_status",
    "explain_analysis",
    "open_page",
    "get_action_status",
}
MAX_LIST = 15
MAX_SERIES = 12
MAX_TEXT = 600


def web_url() -> str:
    return os.environ.get("FDA_WEB_URL", "http://127.0.0.1:5173").rstrip("/")


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    title: str
    description: str
    input_schema: dict[str, Any]
    safety: Safety
    family: str
    analysis_id: str | None = None
    deferred: bool = True
    keywords: tuple[str, ...] = field(default_factory=tuple)

    def public(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "input_schema": self.input_schema,
            "safety_class": self.safety,
            "family": self.family,
            "analysis_id": self.analysis_id,
            "defer_loading": self.deferred,
        }


# ── generation from the registry ────────────────────────────────────────────

FAMILIES = {
    "T": "price",
    "TS": "time_series",
    "R": "risk",
    "F": "fundamentals",
    "S": "scores",
    "V": "valuation",
    "PE": "peers",
    "E": "earnings",
    "O": "ownership",
    "TA": "technical",
    "P": "portfolio",
    "C": "portfolio",
    "CP": "comparison",
    "Q": "quant",
    "MK": "markets",
    "SC": "screener",
    "DV": "derivatives",
    "IN": "investment",
    "TR": "trading",
    "DEV": "diagnostics",
}


def _hoist(schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Params JSON schema → (object schema without $defs, $defs)."""
    defs = schema.pop("$defs", {})
    schema.pop("title", None)
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    schema["additionalProperties"] = False
    return schema, defs


def analysis_tool(spec: AnalysisSpec) -> Tool:
    params, defs = _hoist(spec.params_schema())
    subjects = sorted(spec.applies_to)
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "subject_type": {"type": "string", "enum": subjects, "description": "Kind of subject"},
            "subject_id": {
                "type": "string",
                "description": "Symbol (asset), or the saved object id (portfolio, comparison, "
                "universe, profile, strategy, paper); 'global' for markets",
            },
            "range": {
                "type": "string",
                "description": "Window: 1M, 3M, 6M, YTD, 1Y, 3Y, 5Y, 10Y or MAX (default 1Y)",
            },
            "params": params,
        },
        "required": ["subject_type", "subject_id"],
        "additionalProperties": False,
    }
    if defs:
        schema["$defs"] = defs
    prefix = spec.id.split("-")[0]
    description = (
        f"{spec.title} ({spec.id} v{spec.version}). {spec.description} "
        f"Applies to: {', '.join(subjects)}. Returns a compact {spec.result_kind} summary with "
        "result_id (cite it as [r:result_id]) and ui_ref (the page showing the full result)."
    )
    return Tool(
        name=spec.tool_name[:64],
        title=spec.title,
        description=description,
        input_schema=schema,
        safety="read" if spec.safety_class == "read" else "write",
        family=FAMILIES.get(prefix, prefix.lower()),
        analysis_id=spec.id,
        keywords=(spec.id, spec.title, prefix),
    )


def _obj(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


S = {"type": "string"}
N = {"type": "number"}
I = {"type": "integer"}  # noqa: E741

STATIC_TOOLS = [
    Tool(
        "search_tools",
        "Search tools",
        "Find the right tool among ~250 by keywords (e.g. 'value at risk', 'DCF', 'Hurst', "
        "'implied volatility', 'drawdown'). Returns tool names, titles and one-line purposes; "
        "call explain_analysis for the full parameter schema before using an unfamiliar tool.",
        _obj({"query": S, "limit": {"type": "integer", "minimum": 1, "maximum": 25}}, ["query"]),
        "read",
        "data",
        deferred=False,
    ),
    Tool(
        "search_instruments",
        "Search instruments",
        "Search stored and provider instruments by symbol or name (stocks, ETFs, indices, FX, "
        "crypto, futures). Returns symbol, name, asset class, currency and whether prices are "
        "already stored (ingested).",
        _obj({"query": S}, ["query"]),
        "read",
        "data",
        deferred=False,
    ),
    Tool(
        "get_data_status",
        "Data status",
        "Coverage of an instrument's stored prices: first/last date, rows, source, last ingest. "
        "Use before analysing to state the sample; if nothing is stored, call request_ingestion.",
        _obj({"symbol": S}, ["symbol"]),
        "read",
        "data",
        deferred=False,
    ),
    Tool(
        "request_ingestion",
        "Request ingestion",
        "Fetch and store an instrument's daily prices (and profile) from the configured providers "
        "(cache-first, rate-limited). Returns the job; analyses work once it has completed.",
        _obj({"symbol": S}, ["symbol"]),
        "read",
        "data",
    ),
    Tool(
        "explain_analysis",
        "Explain analysis",
        "Definition, version, subjects, result kind and parameter schema of an analysis, by "
        "catalog id (e.g. R-001) or tool name. Use it to pick parameters and explain a metric.",
        _obj({"analysis": S}, ["analysis"]),
        "read",
        "data",
        deferred=False,
    ),
    Tool(
        "open_page",
        "Open page",
        "Link to a workbench page for a subject so the user sees the evidence (panels with the "
        "full results). Returns the URL; it does not change any data.",
        _obj(
            {
                "subject_type": {"type": "string", "enum": sorted(SUBJECT_ROUTES)},
                "subject_id": S,
                "page": S,
            },
            ["subject_type", "subject_id"],
        ),
        "read",
        "navigation",
        deferred=False,
    ),
    Tool(
        "get_action_status",
        "Action status",
        "Status of a proposed write action (pending, approved, rejected, executed, failed) and its "
        "result once the user approved it in the workbench.",
        _obj({"action_id": I}, ["action_id"]),
        "read",
        "actions",
        deferred=False,
    ),
    Tool(
        "create_paper_account",
        "Create paper account",
        "WRITE (needs the user's approval in the workbench): create a simulated cash account. "
        "Returns a pending action id; nothing is created until approved.",
        _obj({"name": S, "starting_cash": N, "currency": S}, ["name", "starting_cash"]),
        "write",
        "actions",
    ),
    Tool(
        "place_paper_order",
        "Place paper order",
        "WRITE (needs approval): simulated order on a paper account (market, limit, stop, "
        "stop_limit, trailing_stop; day/gtc). Decided at the last stored close, fills from the next"
        " session. Paper only — there is no live-order tool.",
        _obj(
            {
                "account_id": I,
                "symbol": S,
                "side": {"type": "string", "enum": ["buy", "sell"]},
                "quantity": N,
                "order_type": {
                    "type": "string",
                    "enum": ["market", "limit", "stop", "stop_limit", "trailing_stop"],
                },
                "limit_price": N,
                "stop_price": N,
                "trail_pct": N,
                "tif": {"type": "string", "enum": ["day", "gtc"]},
            },
            ["account_id", "symbol", "side", "quantity"],
        ),
        "write",
        "actions",
    ),
    Tool(
        "cancel_paper_order",
        "Cancel paper order",
        "WRITE (needs approval): cancel a working paper order as of the last stored session.",
        _obj({"account_id": I, "order_id": I}, ["account_id", "order_id"]),
        "write",
        "actions",
    ),
    Tool(
        "run_workflow",
        "Run workflow",
        "Recurring review brief for a portfolio: `portfolio_review` (weekly portfolio review: "
        "performance, risk, contributions, alert states, moves since the last snapshot) or "
        "`earnings_preview` (earnings preview for holdings: next report dates, last surprise, "
        "beat rate). Every figure in its highlights cites a result_id.",
        _obj(
            {
                "workflow": {"type": "string", "enum": ["portfolio_review", "earnings_preview"]},
                "portfolio_id": I,
                "range": S,
                "horizon_days": I,
            },
            ["workflow", "portfolio_id"],
        ),
        "read",
        "reports",
        keywords=("weekly review", "monitoring", "earnings calendar", "holdings", "brief"),
    ),
    Tool(
        "create_strategy",
        "Create strategy",
        "WRITE (needs approval): save a hypothesis-first strategy (symbol, template) for the "
        "backtest lab; its validation (DSR, PBO) is computed before any performance is shown.",
        _obj(
            {"name": S, "hypothesis": S, "symbol": S, "template": S},
            ["name", "hypothesis", "symbol", "template"],
        ),
        "write",
        "actions",
    ),
]


def all_tools() -> list[Tool]:
    tools = [analysis_tool(s) for s in REGISTRY.all() if not s.diagnostic] + STATIC_TOOLS
    names = [t.name for t in tools]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise RuntimeError(f"Duplicate tool names: {sorted(duplicates)}")
    return tools


_TOOLS: dict[str, Tool] | None = None


def tools_by_name() -> dict[str, Tool]:
    global _TOOLS
    if _TOOLS is None:
        _TOOLS = {t.name: t for t in all_tools()}
    return _TOOLS


# ── search (BM25 over names, titles, descriptions) ──────────────────────────

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower().replace("-", " "))


# Abbreviations in titles → the words people ask with (titles are the strongest search field).
SYNONYMS = {
    "var": "value at risk",
    "cvar": "conditional value at risk expected shortfall",
    "es": "expected shortfall",
    "sharpe": "sharpe ratio",
    "sortino": "sortino ratio",
    "calmar": "calmar ratio",
    "cagr": "compound annual growth rate return",
    "iv": "implied volatility",
    "dcf": "discounted cash flow",
    "pca": "principal components",
    "garch": "conditional volatility",
    "dsr": "deflated sharpe",
    "pbo": "backtest overfitting",
    "dca": "dollar cost averaging",
    "ips": "investment policy statement",
    "evt": "extreme value",
}


def _expand(title: str) -> str:
    words = _tokens(title)
    return " ".join([title, *(SYNONYMS[w] for w in words if w in SYNONYMS)])


def search(query: str, limit: int = 10) -> list[dict[str, Any]]:
    tools = list(tools_by_name().values())
    # names and titles weigh three times the description (a title match is the strongest signal)
    docs = [
        _tokens(" ".join((*(t.name.replace("_", " "), _expand(t.title)) * 3, t.description)))
        for t in tools
    ]
    q = _tokens(query)
    if not q:
        return []
    n = len(docs)
    avg = sum(len(d) for d in docs) / n
    df = {term: sum(1 for d in docs if term in d) for term in set(q)}
    k1, b = 1.2, 0.75
    scored = []
    for tool, doc in zip(tools, docs, strict=True):
        score = 0.0
        for term in q:
            tf = doc.count(term)
            if not tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc) / avg))
        if tool.analysis_id and tool.analysis_id.lower() in query.lower():
            score += 10
        if score > 0:
            scored.append((score, tool))
    scored.sort(key=lambda x: -x[0])
    return [
        {
            "name": t.name,
            "title": t.title,
            "analysis_id": t.analysis_id,
            "family": t.family,
            "safety_class": t.safety,
            "purpose": t.description.split(". ")[1][:200] if t.analysis_id else t.description[:200],
            "score": round(s, 2),
        }
        for s, t in scored[:limit]
    ]


# ── result shaping ───────────────────────────────────────────────────────────


def _num(x: Any) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool)


def shape(value: Any, key: str = "", depth: int = 0) -> Any:
    """Compact LLM view: numeric series → n/first/last/min/max/mean; dates → n/first/last; long
    record lists truncated; long strings clipped; nesting capped."""
    if isinstance(value, float):
        return None if not math.isfinite(value) else round(value, 6)
    if isinstance(value, str):
        return value if len(value) <= MAX_TEXT else value[:MAX_TEXT] + "…"
    if isinstance(value, list):
        if key == "dates" and len(value) > 2:
            return {"n": len(value), "first": value[0], "last": value[-1]}
        nums = [v for v in value if _num(v) and math.isfinite(float(v))]
        if len(value) > MAX_SERIES and value and len(nums) >= 0.8 * len(value):
            return {
                "n": len(value),
                "first": shape(value[0]),
                "last": shape(value[-1]),
                "min": round(min(nums), 6),
                "max": round(max(nums), 6),
                "mean": round(sum(nums) / len(nums), 6),
            }
        items = [shape(v, "", depth + 1) for v in value[:MAX_LIST]]
        if len(value) > MAX_LIST:
            items.append({"truncated": len(value) - MAX_LIST})
        return items
    if isinstance(value, dict):
        if depth >= 4:
            return {"keys": sorted(value)[:20]}
        return {k: shape(v, k, depth + 1) for k, v in value.items()}
    return value


def result_id(analysis_id: str, data_version: str, params: Any, subject: Any) -> str:
    raw = json.dumps([analysis_id, data_version, params, subject], sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def ui_ref(subject_type: str, subject_id: str) -> str:
    return web_url() + SUBJECT_ROUTES.get(subject_type, "/").format(id=subject_id)


# ── execution ────────────────────────────────────────────────────────────────


class ToolError(Exception):
    pass


async def call_tool(
    request: Request, name: str, arguments: dict[str, Any], origin: str
) -> dict[str, Any]:
    """Runs a tool; returns {content, is_error}. Write tools only create a pending action."""
    tool = tools_by_name().get(name)
    if tool is None:
        raise ToolError(f"Unknown tool {name!r}: use search_tools to find one")
    actor = current_actor()
    if (
        actor is not None
        and actor.role == "viewer"
        and (tool.safety == "write" or name in SIDE_EFFECTS)
    ):
        raise ToolError(f"{name} changes data or spends provider quota; viewers cannot call it")
    if tool.safety == "write":
        return {"content": propose(_engine(request), tool, arguments, origin), "is_error": False}
    if tool.analysis_id:
        return await _call_analysis(request, tool, arguments)
    handler = READ_HANDLERS[name]
    return {"content": await handler(request, arguments), "is_error": False}


async def _call_analysis(request: Request, tool: Tool, arguments: dict[str, Any]) -> dict[str, Any]:
    assert tool.analysis_id is not None
    try:
        run = RunRequest(
            subject=SubjectRef(type=arguments["subject_type"], id=str(arguments["subject_id"])),
            context=RunContext(range=str(arguments.get("range") or "1Y").upper()),
            analyses=[
                AnalysisRequestItem(id=tool.analysis_id, params=arguments.get("params") or {})
            ],
        )
    except (KeyError, ValueError) as exc:
        raise ToolError(f"Invalid arguments: {exc}") from exc
    memo: Memo = {}
    line = await _run_one(run.analyses[0], run, request, memo)
    if line.error is not None:
        hint = {
            "/errors/insufficient-data": "Try a longer range, ingest the data, or another subject.",
            "/errors/invalid-params": "Check the parameter schema with explain_analysis.",
            "/errors/not-applicable": "This analysis does not apply to this subject or asset type.",
        }.get(line.error.type, "")
        return {
            "content": {"error": line.error.title, "detail": line.error.detail, "hint": hint},
            "is_error": True,
        }
    env = line.result
    assert env is not None
    rid = result_id(env.analysis_id, env.data_version, env.params, env.subject.model_dump())
    content: dict[str, Any] = {
        "result_id": rid,
        "analysis_id": env.analysis_id,
        "analysis_version": env.analysis_version,
        "subject": env.subject.model_dump(),
        "range": env.context.range,
        "params": env.params,
        "data_version": env.data_version,
        "warnings": env.warnings,
        "summary": shape(env.payload),
        "ui_ref": ui_ref(env.subject.type, env.subject.id),
    }
    if guardrails.is_model_output(env.analysis_id):
        await _guard(request, env, content)
    return {"content": content, "is_error": False}


async def _guard(request: Request, env: Any, content: dict[str, Any]) -> None:
    """AI-006 validation-before-claim: label, assumptions, and evidence for any performance."""
    payload = dict(env.payload)
    params = dict(env.params or {})
    content["model_output"] = True
    content["label"] = guardrails.LABEL
    content["assumptions"] = {"params": params, "range": env.context.range}
    own = guardrails.evidence(payload)
    if not guardrails.needs_validation(env.analysis_id, payload):
        content["evidence"] = (
            {"analysis_id": env.analysis_id, "summary": shape(own)} if own else None
        )
        return
    wanted = guardrails.evidence_request(env.subject.type, payload, params)
    problem = "no validation analysis exists for this subject type"
    if wanted is not None:
        analysis_id, ev_params = wanted
        run = await anyio.to_thread.run_sync(
            run_analysis,
            _engine(request),
            request.app.state.lake,
            env.subject.type,
            env.subject.id,
            analysis_id,
            ev_params,
            {"range": env.context.range},
        )
        if run.payload is not None:
            subject = env.subject.model_dump()
            content["evidence"] = {
                "analysis_id": analysis_id,
                "result_id": result_id(analysis_id, run.data_version or "", ev_params, subject),
                "summary": shape(guardrails.evidence(run.payload) or run.payload),
            }
            return
        problem = f"{analysis_id} could not be computed: {run.problem}"
    content["summary"] = shape(guardrails.withhold(payload))
    content["withheld"] = guardrails.performance_keys(payload)
    content["withheld_reason"] = f"Performance withheld until validated ({problem})."


def _engine(request: Request) -> Engine:
    engine: Engine = request.app.state.engine
    return engine


async def _search_tools(request: Request, args: dict[str, Any]) -> Any:
    return {"tools": search(str(args.get("query", "")), int(args.get("limit") or 10))}


async def _search_instruments(request: Request, args: dict[str, Any]) -> Any:
    from fda_api.data import search_instruments

    query = str(args.get("query", "")).strip()[:40]
    if not query:
        raise ToolError("Empty query")
    rows = search_instruments(request, query)
    return {"instruments": [r.model_dump() for r in rows]}


def _symbol(args: dict[str, Any]) -> str:
    """Route functions called directly skip FastAPI's path validation: apply the same pattern."""
    from fda_api.data import SYMBOL_PATTERN

    symbol = str(args.get("symbol", "")).strip()
    if not re.match(SYMBOL_PATTERN, symbol):
        raise ToolError(f"Invalid symbol {symbol[:24]!r}: letters, digits and . ^ = - only, ≤ 20")
    return symbol.upper()


async def _data_status(request: Request, args: dict[str, Any]) -> Any:
    from fda_api.data import data_status

    return data_status(request, _symbol(args)).model_dump(mode="json")


async def _ingest(request: Request, args: dict[str, Any]) -> Any:
    from fda_api.data import request_ingestion

    background = BackgroundTasks()
    job = await request_ingestion(request, background, _symbol(args))
    await background()
    return job.model_dump(mode="json")


async def _explain(request: Request, args: dict[str, Any]) -> Any:
    key = str(args["analysis"]).strip()
    spec = next((s for s in REGISTRY.all() if key.upper() == s.id or key == s.tool_name), None)
    if spec is None:
        raise ToolError(f"Unknown analysis {key!r}: use search_tools")
    return {
        "analysis_id": spec.id,
        "tool": spec.tool_name,
        "title": spec.title,
        "version": spec.version,
        "definition": spec.description,
        "applies_to": sorted(spec.applies_to),
        "result_kind": spec.result_kind,
        "params_schema": spec.params_schema(),
    }


async def _open_page(request: Request, args: dict[str, Any]) -> Any:
    kind = str(args["subject_type"])
    if kind not in SUBJECT_ROUTES:
        raise ToolError(f"Unknown subject type {kind!r}")
    url = ui_ref(kind, str(args["subject_id"]))
    page = args.get("page")
    if page and kind != "market":
        url = url.rsplit("/", 1)[0] + "/" + re.sub(r"[^a-z0-9-]", "", str(page).lower())
    return {"url": url}


async def _action_status(request: Request, args: dict[str, Any]) -> Any:
    row = load_action(_engine(request), int(args["action_id"]))
    if row is None:
        raise ToolError(f"Unknown action {args['action_id']}")
    return row


async def _workflow(request: Request, args: dict[str, Any]) -> Any:
    from fda_api.workflows import run_workflow  # workflows build on this module

    try:
        portfolio = str(int(args["portfolio_id"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ToolError("portfolio_id must be a portfolio id") from exc
    state = request.app.state
    try:
        return await anyio.to_thread.run_sync(
            run_workflow,
            state.engine,
            state.lake,
            str(args.get("workflow")),
            portfolio,
            str(args.get("range") or "1Y"),
            int(args.get("horizon_days") or 30),
        )
    except HTTPException as exc:
        raise ToolError(str(exc.detail)) from exc


# read tools that still change stored data (provider fetches): not for viewers
SIDE_EFFECTS = frozenset({"request_ingestion"})
READ_HANDLERS: dict[str, Callable[[Request, dict[str, Any]], Awaitable[Any]]] = {
    "search_tools": _search_tools,
    "search_instruments": _search_instruments,
    "get_data_status": _data_status,
    "request_ingestion": _ingest,
    "explain_analysis": _explain,
    "open_page": _open_page,
    "get_action_status": _action_status,
    "run_workflow": _workflow,
}


# ── write actions: propose → approve in the web app → execute ───────────────


def _summary(tool: Tool, args: dict[str, Any]) -> str:
    if tool.name == "place_paper_order":
        price = args.get("limit_price") or args.get("stop_price") or ""
        return (
            f"Paper account {args.get('account_id')}: {args.get('side')} {args.get('quantity')} "
            f"{args.get('symbol')} {args.get('order_type', 'market')} {price}"
        ).strip()
    if tool.name == "cancel_paper_order":
        return f"Paper account {args.get('account_id')}: cancel order {args.get('order_id')}"
    if tool.name == "create_paper_account":
        cash = f"{args.get('starting_cash')} {args.get('currency', 'USD')}"
        return f"New paper account {args.get('name')!r} with {cash}"
    if tool.name == "create_strategy":
        return f"New strategy {args.get('name')!r}: {args.get('template')} on {args.get('symbol')}"
    return tool.title


def propose(engine: Engine, tool: Tool, args: dict[str, Any], origin: str) -> dict[str, Any]:
    with engine.begin() as conn:
        aid = conn.execute(
            insert(agent_actions).values(
                tool=tool.name,
                arguments=args,
                summary=_summary(tool, args),
                status="pending",
                origin=origin,
            )
        ).inserted_primary_key[0]  # type: ignore[index]
    return {
        "action_id": int(aid),
        "status": "pending",
        "summary": _summary(tool, args),
        "approve_in": f"{web_url()}/ai/actions",
        "note": "Nothing has been executed. The user must approve this action in the workbench; "
        "check it with get_action_status.",
    }


def load_action(engine: Engine, action_id: int) -> dict[str, Any] | None:
    with engine.connect() as conn:
        row = (
            conn.execute(select(agent_actions).where(agent_actions.c.id == action_id))
            .mappings()
            .one_or_none()
        )
    return None if row is None else _row(row)


def _row(row: Any) -> dict[str, Any]:
    out = dict(row)
    for k in ("created_at", "decided_at"):
        if out.get(k) is not None:
            out[k] = out[k].isoformat()
    return out


def _last_close(request: Request, symbol: str) -> tuple[Any, float]:
    row = _instrument(_engine(request), symbol)
    frame = _frame(request.app.state.lake, row)
    if frame.empty:
        raise InsufficientDataError(f"{symbol} has no stored prices")
    return frame.index[-1].date(), float(frame["close"].iloc[-1])


def execute(request: Request, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    engine = _engine(request)
    if tool == "create_paper_account":
        aid = create_account(
            engine,
            str(args["name"]),
            str(args.get("currency") or "USD"),
            float(args["starting_cash"]),
        )
        return {"account_id": aid, "ui_ref": ui_ref("paper", str(aid))}
    if tool == "place_paper_order":
        order = validate_order(args)
        decided, price = _last_close(request, order["symbol"])
        oid = place_order(engine, int(args["account_id"]), order, decided, price, "agent")
        return {"order_id": oid, "decision_date": decided.isoformat(), "decision_price": price}
    if tool == "cancel_paper_order":
        account = int(args["account_id"])
        from fda_data.paper import load_account

        acc = load_account(engine, account)
        row = next(
            (o for o in (acc or {}).get("orders", []) if o["id"] == int(args["order_id"])), None
        )
        if row is None:
            raise PaperError(f"Unknown order {args['order_id']}")
        decided, _ = _last_close(request, row["symbol"])
        return {"cancelled": cancel_order(engine, account, int(args["order_id"]), decided)}
    if tool == "create_strategy":
        sid = create_strategy(
            engine,
            str(args["name"]),
            str(args["hypothesis"]),
            {"symbol": args["symbol"], "template": args["template"]},
        )
        return {"strategy_id": sid, "ui_ref": ui_ref("strategy", str(sid))}
    raise ToolError(f"No executor for {tool}")


def decide(request: Request, action_id: int, approve: bool) -> dict[str, Any]:
    engine = _engine(request)
    row = load_action(engine, action_id)
    if row is None:
        raise HTTPException(404, f"Unknown action {action_id}")
    if row["status"] != "pending":
        raise HTTPException(409, f"Action {action_id} is already {row['status']}")
    now = datetime.now(UTC)
    if not approve:
        status, result = "rejected", None
    else:
        try:
            status, result = "executed", execute(request, row["tool"], row["arguments"])
        except (
            PaperError,
            StrategyError,
            InsufficientDataError,
            ToolError,
            KeyError,
            ValueError,
        ) as exc:
            status, result = "failed", {"error": str(exc)}
    with engine.begin() as conn:
        conn.execute(
            update(agent_actions)
            .where((agent_actions.c.id == action_id) & (agent_actions.c.status == "pending"))
            .values(status=status, result=result, decided_at=now)
        )
    out = load_action(engine, action_id)
    assert out is not None
    return out


# ── REST endpoints ───────────────────────────────────────────────────────────


class CallBody(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)


@router.get("/tools")
def list_tools(core_only: bool = False, family: str | None = None) -> list[dict[str, Any]]:
    tools = tools_by_name().values()
    return [
        t.public()
        for t in tools
        if (not core_only or t.name in CORE_TOOLS) and (family is None or t.family == family)
    ]


@router.post("/tools/{name}:call")
async def post_call(request: Request, name: str, body: CallBody) -> dict[str, Any]:
    try:
        return await call_tool(request, name, body.arguments, "api")
    except ToolError as exc:
        return {"content": {"error": str(exc)}, "is_error": True}


@router.get("/actions")
def list_actions(request: Request, status: str | None = None) -> list[dict[str, Any]]:
    query = select(agent_actions).order_by(agent_actions.c.id.desc()).limit(200)
    if status:
        query = query.where(agent_actions.c.status == status)
    with _engine(request).connect() as conn:
        return [_row(r) for r in conn.execute(query).mappings()]


@router.post("/actions/{action_id}/approve")
def approve_action(request: Request, action_id: int) -> dict[str, Any]:
    """Approval is a web-app action: the MCP endpoint cannot reach this route."""
    return decide(request, action_id, True)


@router.post("/actions/{action_id}/reject")
def reject_action(request: Request, action_id: int) -> dict[str, Any]:
    return decide(request, action_id, False)
