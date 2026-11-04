"""LLM cost meter and monthly budget (PRD NFR-12, D-AI-6).

Prices change and differ per account, so none are hard-coded: `FDA_LLM_PRICE_<PROVIDER>` holds
"input,output" in USD per million tokens (e.g. `FDA_LLM_PRICE_GEMINI=0.30,2.50`); cache reads
are priced at a tenth of input. Local servers cost nothing. Without a price the meter shows
tokens only.

Budgets are per user and calendar month (UTC), over every recorded LLM call (`agent_runs`):
`FDA_LLM_MONTHLY_TOKENS` caps input + output tokens, `FDA_LLM_MONTHLY_USD` caps the priced cost.
A call is refused once a cap is reached.
"""

import os
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Engine

from fda_api.llm.base import LLMError
from fda_api.llm.providers import PROVIDERS
from fda_data.schema import agent_runs

CACHE_READ_FACTOR = 0.1


def price(provider: str) -> tuple[float, float] | None:
    """USD per million (input, output) tokens, or None when not configured."""
    known = PROVIDERS.get(provider)
    if known is not None and known.local:
        return (0.0, 0.0)
    raw = os.getenv(f"FDA_LLM_PRICE_{provider.upper()}", "")
    try:
        p_in, p_out = (float(x) for x in raw.split(","))
    except ValueError:
        return None
    return (p_in, p_out) if p_in >= 0 and p_out >= 0 else None


def cost(provider: str, usage: dict[str, Any]) -> float | None:
    rates = price(provider)
    if rates is None:
        return None
    p_in, p_out = rates
    tokens_in = float(usage.get("input") or 0) + CACHE_READ_FACTOR * float(
        usage.get("cache_read") or 0
    )
    return (tokens_in * p_in + float(usage.get("output") or 0) * p_out) / 1e6


def _float_env(name: str) -> float | None:
    try:
        value = float(os.getenv(name, ""))
    except ValueError:
        return None
    return value if value > 0 else None


def month_start(now: datetime | None = None) -> datetime:
    now = now or datetime.now(UTC)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def month_usage(engine: Engine, user_id: int | None) -> dict[str, Any]:
    """This month's LLM calls of a user, per provider and model, with the budget left."""
    query = select(
        agent_runs.c.provider, agent_runs.c.model, agent_runs.c.usage, agent_runs.c.created_at
    ).where(agent_runs.c.owner_id == user_id)
    start = month_start()
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with engine.connect() as conn:
        for provider, model, usage, created in conn.execute(query):
            stamp = created if created.tzinfo else created.replace(tzinfo=UTC)
            if stamp < start:
                continue
            row = rows.setdefault(
                (provider, model),
                {"provider": provider, "model": model, "runs": 0, "input": 0, "output": 0,
                 "cost": 0.0, "priced": price(provider) is not None},
            )  # fmt: skip
            row["runs"] += 1
            row["input"] += int((usage or {}).get("input") or 0)
            row["output"] += int((usage or {}).get("output") or 0)
            row["cost"] += cost(provider, usage or {}) or 0.0
    models = sorted(rows.values(), key=lambda r: -(r["input"] + r["output"]))
    for row in models:
        if not row["priced"]:
            row["cost"] = None
    tokens = sum(r["input"] + r["output"] for r in models)
    spent = sum(r["cost"] or 0.0 for r in models)
    cap_tokens = _float_env("FDA_LLM_MONTHLY_TOKENS")
    cap_usd = _float_env("FDA_LLM_MONTHLY_USD")
    return {
        "month": start.date().isoformat()[:7],
        "models": models,
        "runs": sum(r["runs"] for r in models),
        "tokens": tokens,
        "cost": spent,
        "unpriced_runs": sum(r["runs"] for r in models if not r["priced"]),
        "cap_tokens": cap_tokens,
        "cap_usd": cap_usd,
        "over_budget": bool(
            (cap_tokens is not None and tokens >= cap_tokens)
            or (cap_usd is not None and spent >= cap_usd)
        ),
    }


def check_budget(engine: Engine, user_id: int | None) -> None:
    """Refuse a new LLM call once this month's token or cost cap is reached."""
    if _float_env("FDA_LLM_MONTHLY_TOKENS") is None and _float_env("FDA_LLM_MONTHLY_USD") is None:
        return
    used = month_usage(engine, user_id)
    if used["over_budget"]:
        raise LLMError(
            f"Monthly LLM budget reached ({used['tokens']:,} tokens, ${used['cost']:.2f} this "
            "month): raise FDA_LLM_MONTHLY_TOKENS / FDA_LLM_MONTHLY_USD in .env"
        )
