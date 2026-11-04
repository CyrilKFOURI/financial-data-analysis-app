"""J26 slice 262 (M30 §6, M31 §3): recurring analysis workflows that produce a cited brief.

A workflow runs a fixed set of analyses on a subject and returns a *brief*:
- every section carries the analysis id, a `result_id` (the same id the tool layer cites) and the
  page to open;
- deterministic highlights state only figures taken from those results, each one cited
  `[r:result_id]`.

The brief itself is deterministic: it is the report, saved and e-mailed by a `workflow`
schedule. On request (`narrate`), the user's LLM adds a short narrative over it, verified like a
console answer (`fda_api.narrate`). V1–V3 measurements are described, never
turned into verdicts. V4 model outputs keep their label.

Workflows:
- `portfolio_review`: weekly portfolio review — performance, risk and contributions, the
  alert rules on the portfolio, and the largest moves since its last snapshot.
- `earnings_preview`: earnings preview for holdings — next report dates within the horizon, the
  last surprise and the beat rate (third-party data, labelled).
"""

import html
import json
import math
from datetime import UTC, datetime, timedelta
from typing import Any

import anyio
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import Engine, select

from fda_api.llm import LLMError
from fda_api.narrate import narrate_brief
from fda_api.runner import Run, run_analysis, value_at
from fda_api.tools import result_id, shape, ui_ref
from fda_api.workbench import _flat
from fda_data.lake import PriceLake
from fda_data.portfolios import PortfolioError, load_portfolio
from fda_data.schema import alert_rules, instruments, snapshots

router = APIRouter(prefix="/v1/workflows", tags=["workflows"])

REVIEW_ANALYSES = ("P-003", "P-004", "R-020", "P-008", "P-010", "P-018")
WORKFLOWS: dict[str, dict[str, Any]] = {
    "portfolio_review": {
        "title": "Weekly portfolio review",
        "subject": "portfolio",
        "description": "Performance, risk and contributions, alert states, largest moves since "
        "the last snapshot.",
    },
    "earnings_preview": {
        "title": "Earnings preview for holdings",
        "subject": "portfolio",
        "description": "Next report dates within the horizon, last surprise and beat rate per "
        "holding (third-party data).",
    },
}


class WorkflowIn(BaseModel):
    subject_id: str = Field(min_length=1, max_length=64)
    range: str = Field("1Y", max_length=8)
    horizon_days: int = Field(30, ge=1, le=120)
    narrate: bool = Field(False, description="Add an LLM narrative (cited, verified)")
    provider: str | None = Field(None, max_length=32)
    model: str | None = Field(None, max_length=120)


def _num(value: Any) -> str:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        return str(value)
    return f"{value:.4g}"


def _section(run: Run, subject_type: str, subject_id: str) -> dict[str, Any]:
    out: dict[str, Any] = {"analysis_id": run.analysis_id, "title": run.title}
    if run.payload is None:
        out["problem"] = run.problem
        return out
    subject = {"type": subject_type, "id": subject_id}
    out["result_id"] = result_id(run.analysis_id, run.data_version or "", run.params, subject)
    out["ui_ref"] = ui_ref(subject_type, subject_id)
    out["summary"] = shape(run.payload)
    out["warnings"] = run.warnings
    return out


def _cite(section: dict[str, Any]) -> str:
    return f"[r:{section['result_id']}]"


def _first_number(payload: dict[str, Any], paths: tuple[str, ...]) -> tuple[str, float] | None:
    for path in paths:
        value = value_at(payload, path)
        if value is not None:
            return path, value
    return None


def portfolio_review(
    engine: Engine, lake: PriceLake, pid: str, context: dict[str, Any]
) -> dict[str, Any]:
    memo: dict[Any, Any] = {}
    runs = {
        aid: run_analysis(engine, lake, "portfolio", pid, aid, {}, context, memo)
        for aid in REVIEW_ANALYSES
    }
    sections = [_section(r, "portfolio", pid) for r in runs.values()]
    by_id = {s["analysis_id"]: s for s in sections}
    highlights: list[str] = []
    for aid, label, paths in (
        ("P-003", "Time-weighted return over the range", ("twr", "inception_twr")),
        ("P-003", "Annualised time-weighted return", ("twr_annualised",)),
        ("R-020", "Annualised volatility", ("volatility",)),
        ("R-020", "Maximum drawdown", ("max_drawdown",)),
        ("P-004", "Money-weighted return (your flows)", ("mwr",)),
    ):
        run, section = runs[aid], by_id[aid]
        found = _first_number(run.payload or {}, paths) if run.payload else None
        if found:
            highlights.append(f"{label}: {_num(found[1])} {_cite(section)}")
    with engine.connect() as conn:
        rules = conn.execute(
            select(alert_rules).where(
                alert_rules.c.subject_type == "portfolio", alert_rules.c.subject_id == pid
            )
        ).mappings()
        alerts = [
            {
                "name": r["name"],
                "condition": f"{r['analysis_id']} {r['path']} {r['op']} {r['threshold']:g}",
                "last_value": r["last_value"],
                "triggered": r["last_state"],
                "evaluated_at": r["last_evaluated_at"].isoformat()
                if r["last_evaluated_at"]
                else None,
            }
            for r in rules
        ]
        last = (
            conn.execute(
                select(snapshots)
                .where(snapshots.c.subject_type == "portfolio", snapshots.c.subject_id == pid)
                .order_by(snapshots.c.created_at.desc(), snapshots.c.id.desc())
                .limit(1)
            )
            .mappings()
            .first()
        )
    for alert in alerts:
        if alert["triggered"]:
            highlights.append(f"Alert '{alert['name']}' is triggered (see alerts).")
    moves = _moves(last, runs) if last else []
    for move in moves[:5]:
        section = by_id[move["analysis_id"]]
        highlights.append(
            f"Largest move since the last snapshot: `{move['path']}` is now "
            f"{_num(move['now'])} {_cite(section)}"
        )
    return {
        "sections": sections,
        "alerts": alerts,
        "moves_since_snapshot": moves[:20],
        "highlights": highlights,
    }


def _moves(last: Any, runs: dict[str, Run]) -> list[dict[str, Any]]:
    out = []
    for aid, before in (last["metrics"] or {}).items():
        run = runs.get(aid)
        if run is None or run.payload is None:
            continue
        now = _flat(run.payload)
        for path, old in before.items():
            new = now.get(path)
            if not isinstance(old, int | float) or not isinstance(new, int | float) or old == new:
                continue
            scale = max(abs(old), 1e-9)
            out.append(
                {
                    "analysis_id": aid,
                    "path": path,
                    "before": old,
                    "now": new,
                    "relative": (new - old) / scale,
                    "snapshot": last["label"],
                }
            )
    return sorted(out, key=lambda m: -abs(m["relative"]))


def earnings_preview(
    engine: Engine, lake: PriceLake, pid: str, context: dict[str, Any], horizon_days: int
) -> dict[str, Any]:
    try:
        definition = load_portfolio(engine, int(pid))
    except (PortfolioError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc
    symbols = [h["symbol"] for h in definition["holdings"] if not h.get("is_cash")] or list(
        definition.get("symbols") or []
    )
    limit = datetime.now(UTC) + timedelta(days=horizon_days)
    with engine.connect() as conn:
        classes = dict(
            conn.execute(
                select(instruments.c.symbol, instruments.c.asset_class).where(
                    instruments.c.symbol.in_(symbols)
                )
            ).all()
        )
    rows, sections = [], []
    for symbol in symbols:
        if classes.get(symbol, "equity") != "equity":
            rows.append({"symbol": symbol, "note": f"no earnings reports ({classes[symbol]})"})
            continue
        calendar = run_analysis(engine, lake, "asset", symbol, "E-001", {}, context)
        surprises = run_analysis(engine, lake, "asset", symbol, "E-002", {}, context)
        row: dict[str, Any] = {"symbol": symbol}
        if calendar.payload is None:
            row["problem"] = calendar.problem
            rows.append(row)
            continue
        section = _section(calendar, "asset", symbol)
        sections.append(section)
        next_report = calendar.payload.get("next_report")
        history = [
            h for h in calendar.payload.get("history") or [] if h.get("reported") is not None
        ]
        row |= {
            "next_report": next_report,
            "last_surprise_pct": history[0].get("surprise_pct") if history else None,
            "source": calendar.payload.get("source"),
            "as_of": calendar.payload.get("as_of"),
            "result_id": section["result_id"],
        }
        if surprises.payload is not None:
            beat = _first_number(surprises.payload, ("beat_rate",))
            row["beat_rate"] = beat[1] if beat else None
            sections.append(_section(surprises, "asset", symbol))
        row["within_horizon"] = bool(next_report) and datetime.fromisoformat(next_report) <= limit
        rows.append(row)
    rows.sort(key=lambda r: (r.get("next_report") is None, r.get("next_report") or ""))
    highlights = [
        f"{r['symbol']} reports on {r['next_report'][:10]} (third-party calendar, {r['source']}) "
        f"[r:{r['result_id']}]"
        for r in rows
        if r.get("within_horizon")
    ]
    return {"holdings": rows, "sections": sections, "highlights": highlights}


def run_workflow(
    engine: Engine,
    lake: PriceLake,
    name: str,
    subject_id: str,
    range_: str = "1Y",
    horizon_days: int = 30,
) -> dict[str, Any]:
    spec = WORKFLOWS.get(name)
    if spec is None:
        raise HTTPException(404, f"Unknown workflow {name}; one of {', '.join(WORKFLOWS)}")
    context = {"range": range_.upper()}
    if name == "portfolio_review":
        try:
            load_portfolio(engine, int(subject_id))
        except (PortfolioError, ValueError) as exc:
            raise HTTPException(404, str(exc)) from exc
        body = portfolio_review(engine, lake, subject_id, context)
    else:
        body = earnings_preview(engine, lake, subject_id, context, horizon_days)
    return {
        "workflow": name,
        "title": spec["title"],
        "subject": {"type": spec["subject"], "id": subject_id},
        "range": context["range"],
        "as_of": datetime.now(UTC).isoformat(),
        "note": "Measurements and third-party data with their sources; no buy/sell verdicts. "
        "Every figure cites the result it comes from.",
        **body,
    }


def render_html(brief: dict[str, Any]) -> str:
    """A plain HTML version of a brief (saved under data/reports, e-mailed by schedules)."""
    esc = html.escape
    items = "".join(f"<li>{esc(h)}</li>" for h in brief.get("highlights") or []) or (
        "<li>No highlight: nothing stood out in the stored data.</li>"
    )
    narrative = brief.get("narrative") or {}
    story = ""
    if narrative.get("text"):
        unverified = narrative["verification"]["unverified"]
        flag = (
            f"<p style='color:#b42318'>Unverified figures: {esc(', '.join(unverified))}</p>"
            if unverified
            else "<p>Every figure verified against its citation.</p>"
        )
        paragraphs = "".join(f"<p>{esc(p)}</p>" for p in narrative["text"].split("\n") if p)
        story = (
            f"<h2>Narrative</h2><p><i>{esc(narrative['label'])} {esc(narrative['model'])}</i>"
            f"</p>{paragraphs}{flag}"
        )
    elif narrative.get("error"):
        story = f"<h2>Narrative</h2><p>Not written: {esc(narrative['error'])}</p>"
    sections = "".join(
        f"<h3>{esc(s['analysis_id'])} · {esc(s['title'])}</h3>"
        + (
            f"<p>{esc(str(s['problem']))}</p>"
            if "problem" in s
            else f"<p>result {esc(s['result_id'])}</p><pre>"
            f"{esc(json.dumps(s['summary'], indent=1, default=str)[:4000])}</pre>"
        )
        for s in brief.get("sections") or []
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{esc(brief['title'])}</title></head><body style='font:13px sans-serif'>"
        f"<h1>{esc(brief['title'])}</h1><p>{esc(brief['subject']['type'])} "
        f"{esc(brief['subject']['id'])} · range {esc(brief['range'])} · "
        f"{esc(brief['as_of'][:16])} UTC"
        f"</p><p>{esc(brief['note'])}</p><h2>Highlights</h2><ul>{items}</ul>{story}"
        f"<h2>Results</h2>{sections}</body></html>"
    )


@router.get("")
def list_workflows() -> list[dict[str, Any]]:
    return [{"name": k} | v for k, v in WORKFLOWS.items()]


@router.post("/{name}:run")
async def post_run(request: Request, name: str, body: WorkflowIn) -> dict[str, Any]:
    state = request.app.state
    brief = await anyio.to_thread.run_sync(
        run_workflow, state.engine, state.lake, name, body.subject_id, body.range, body.horizon_days
    )
    if body.narrate:
        try:
            brief["narrative"] = await anyio.to_thread.run_sync(
                narrate_brief, state.engine, brief, body.provider, body.model
            )
        except LLMError as exc:
            raise HTTPException(400, str(exc)) from exc
    return brief
