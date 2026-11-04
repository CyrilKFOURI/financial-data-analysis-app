"""M30 §4 endpoints: news headlines, their LLM sentiment and a cited digest (AI-005 reads them).

Scoring and digests use the LLM chosen by the user (any provider, J25 slice 251), never a local
torch model. Only headline titles are sent; each call is recorded with its token usage in
`agent_runs` (it spends credits, so viewers cannot run it).
"""

import json
import re
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import anyio
from fastapi import APIRouter, HTTPException, Path, Request
from pydantic import BaseModel, Field
from sqlalchemy import insert

from fda_api.data import SYMBOL_PATTERN
from fda_api.llm import LLMError, get_provider
from fda_api.llm.costs import check_budget
from fda_api.llm.providers import model_list
from fda_core.analyses.news_sentiment import label_of
from fda_data.actor import current_actor
from fda_data.lake import PriceLake
from fda_data.news import ingest_news, read_news, write_news
from fda_data.providers.base import ProviderError, SymbolNotFoundError
from fda_data.schema import agent_runs

router = APIRouter(prefix="/v1/news", tags=["news"])

BATCH = 25
SCORE_SYSTEM = """You rate the tone of financial news headlines for the shareholders of {symbol}.
For each numbered headline give a sentiment score from -1 (clearly negative for the company) to 1
(clearly positive); 0 when neutral, mixed or unrelated to the company. Judge the headline only.
Reply with JSON only, no prose: [{{"i": 1, "score": 0.6}}, ...]"""
DIGEST_SYSTEM = """You write a short news digest for {symbol} from the headlines given (and nothing
else): 3 to 5 bullet points, at most 120 words, grouped by theme. End every bullet with the ids
of the headlines it rests on, written exactly as [h:ID]. No numbers that are not in the
headlines, no forecasts, no buy/sell opinion. Answer in English."""
CITE = re.compile(r"\[(h:[0-9a-f]+(?:\s*,\s*h:[0-9a-f]+)*)\]")  # [h:id] or [h:a, h:b]


HEADING = re.compile(r"^\s*(?:#{1,4}\s.*|\*\*[^*]+\*\*:?|__[^_]+__:?)\s*$")


def cited_ids(text: str) -> list[str]:
    return [i for group in CITE.findall(text) for i in re.findall(r"h:([0-9a-f]+)", group)]


class LlmChoice(BaseModel):
    provider: str | None = Field(None, max_length=32)
    model: str | None = Field(None, max_length=120)


class ScoreIn(LlmChoice):
    limit: int = Field(100, ge=1, le=500, description="Unscored headlines to rate (newest first)")


class DigestIn(LlmChoice):
    days: int = Field(7, ge=1, le=90)


def _lake(request: Request) -> PriceLake:
    lake: PriceLake = request.app.state.lake
    return lake


def _stored(request: Request, symbol: str) -> dict[str, Any]:
    data = read_news(_lake(request), symbol)
    if data is None:
        raise HTTPException(404, f"No headlines stored for {symbol.upper()}: POST …/ingest first")
    return data


def _session(request: Request, choice: LlmChoice, system: str) -> tuple[Any, str, str]:
    actor = current_actor()
    if actor is None:
        raise HTTPException(401, "Sign in to use the LLM")
    if actor.role == "viewer":
        raise HTTPException(403, "Viewers cannot run LLM calls (they spend credits)")
    try:
        check_budget(request.app.state.engine, actor.user_id)
        provider = get_provider(choice.provider)
        model = provider.model(choice.model)
        listed = model_list(provider)
        if choice.model and listed and choice.model not in listed:
            raise LLMError(f"{provider.label} does not offer the model {choice.model!r}")
        if not model:
            raise LLMError(f"{provider.label}: choose a model")
        return provider.session(model, system, [], []), provider.name, model
    except LLMError as exc:
        raise HTTPException(400, str(exc)) from exc


def _record(
    request: Request, provider: str, model: str, question: str, answer: str, usage: dict[str, int]
) -> None:
    actor = current_actor()
    with request.app.state.engine.begin() as conn:
        conn.execute(
            insert(agent_runs).values(
                owner_id=actor.user_id if actor else None,
                provider=provider,
                model=model,
                question=question,
                answer=answer[:20000],
                steps=[],
                usage=usage,
                verification={},
                duration_ms=0,
                created_at=datetime.now(UTC),
            )
        )


def parse_scores(text: str, count: int) -> dict[int, float]:
    """`[{"i": 1, "score": 0.6}, …]` (possibly fenced) → {1: 0.6}; out-of-range clamped."""
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return {}
    try:
        rows = json.loads(text[start : end + 1])
    except ValueError:
        return {}
    out: dict[int, float] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        i, score = row.get("i"), row.get("score")
        if isinstance(i, int) and 1 <= i <= count and isinstance(score, int | float):
            out[i] = max(-1.0, min(1.0, float(score)))
    return out


@router.get("/{symbol}")
def get_news(request: Request, symbol: str = Path(pattern=SYMBOL_PATTERN)) -> dict[str, Any]:
    data = _stored(request, symbol)
    for h in data["items"]:
        s = h.get("sentiment")
        h["label"] = label_of(s["score"]) if isinstance(s, dict) else None
    return data


@router.post("/{symbol}/ingest")
async def post_ingest(
    request: Request, symbol: str = Path(pattern=SYMBOL_PATTERN)
) -> dict[str, Any]:
    """Latest provider headlines merged into the store (yfinance; offline: synthetic)."""
    try:
        return await anyio.to_thread.run_sync(lambda: ingest_news(_lake(request), symbol))
    except SymbolNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(502, str(exc)) from exc


@router.post("/{symbol}/score")
async def post_score(
    request: Request, body: ScoreIn, symbol: str = Path(pattern=SYMBOL_PATTERN)
) -> dict[str, Any]:
    """Rate unscored headlines with the chosen LLM, in batches of 25 titles."""
    data = _stored(request, symbol)
    todo = [h for h in data["items"] if h.get("sentiment") is None][: body.limit]
    usage: dict[str, int] = {}
    scored = 0
    started = time.perf_counter()
    try:
        for at in range(0, len(todo), BATCH):
            batch = todo[at : at + BATCH]
            prompt = "\n".join(f"{n}. {h['title']}" for n, h in enumerate(batch, 1))
            # a fresh conversation per batch keeps the prompt small; the budget is checked
            # before every batch against usage recorded after the previous one
            session, provider, model = _session(
                request, body, SCORE_SYSTEM.format(symbol=symbol.upper())
            )
            try:
                reply = await anyio.to_thread.run_sync(session.send, prompt, None)
            except LLMError as exc:
                raise HTTPException(502, str(exc)) from exc
            for key, value in reply.usage.items():
                usage[key] = usage.get(key, 0) + value
            stamp = datetime.now(UTC).isoformat(timespec="seconds")
            done = parse_scores(reply.text, len(batch))
            for n, score in done.items():
                batch[n - 1]["sentiment"] = {
                    "score": score,
                    "model": f"{provider}:{model}",
                    "scored_at": stamp,
                }
            scored += len(done)
            _record(
                request,
                provider,
                model,
                f"[news sentiment] {symbol.upper()}: {len(batch)} headlines",
                f"{len(done)} scored",
                reply.usage,
            )
    finally:  # scores already paid for are kept even when a later batch fails
        write_news(_lake(request), symbol, data)
    return {
        "requested": len(todo),
        "scored": scored,
        "unscored_left": sum(1 for h in data["items"] if h.get("sentiment") is None),
        "usage": usage,
        "seconds": round(time.perf_counter() - started, 1),
    }


@router.post("/{symbol}/digest")
async def post_digest(
    request: Request, body: DigestIn, symbol: str = Path(pattern=SYMBOL_PATTERN)
) -> dict[str, Any]:
    """Bullet digest of the last `days` of headlines; every bullet cites its headlines."""
    data = _stored(request, symbol)
    since = (datetime.now(UTC) - timedelta(days=body.days)).isoformat()
    recent = [h for h in data["items"] if h["published"] >= since][:80]
    if not recent:
        raise HTTPException(422, f"No headline in the last {body.days} days: fetch headlines")
    session, provider, model = _session(request, body, DIGEST_SYSTEM.format(symbol=symbol.upper()))
    prompt = "\n".join(f"[h:{h['id']}] {h['published'][:10]} · {h['title']}" for h in recent)
    try:
        reply = await anyio.to_thread.run_sync(session.send, prompt, None)
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc
    known = {h["id"] for h in recent}
    cited = cited_ids(reply.text)
    digest = {
        "text": reply.text.strip(),
        "sources": sorted({c for c in cited if c in known}),
        "unknown_sources": sorted({c for c in cited if c not in known}),
        "uncited_lines": sum(  # bullets and sentences; headings carry no claim
            1
            for line in reply.text.splitlines()
            if line.strip() and not HEADING.match(line) and not CITE.search(line)
        ),
        "days": body.days,
        "headlines": len(recent),
        "model": f"{provider}:{model}",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    data["digest"] = digest
    write_news(_lake(request), symbol, data)
    _record(request, provider, model, f"[news digest] {symbol.upper()}", reply.text, reply.usage)
    return digest
