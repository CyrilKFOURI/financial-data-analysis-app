"""Company data beyond statements (M12–M13): peer groups (DECISIONS D-PE-1/2) and third-party
snapshots (earnings dates & surprises, estimates, analyst views, holders, insiders, short
interest). yfinance is the free source; every snapshot keeps its source and date.

The yfinance client is injected (`YahooClient`) so tests replay recorded payloads offline.
"""

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sqlalchemy import Engine, delete, insert, select, update

from fda_data.ingest import (
    IngestResult,
    _finish,
    _start_job,
    _update_job,
    create_job,
    network_enabled,
)
from fda_data.schema import instruments, peer_group_members, peer_groups, provider_snapshots

logger = logging.getLogger(__name__)

PEER_ALGORITHM_VERSION = "d-pe-2.v1"
DEFAULT_WEIGHTS = {"industry": 0.4, "correlation": 0.3, "size": 0.2, "margins": 0.1}
SIZE_BAND = (0.25, 4.0)
PEER_COUNT = 10
INFO_FIELDS = (
    "longName",
    "shortName",
    "industryKey",
    "industry",
    "sectorKey",
    "sector",
    "country",
    "currency",
    "marketCap",
    "enterpriseValue",
    "grossMargins",
    "operatingMargins",
    "profitMargins",
    "revenueGrowth",
    "earningsGrowth",
    "returnOnEquity",
    "returnOnAssets",
    "debtToEquity",
    "trailingPE",
    "forwardPE",
    "enterpriseToEbitda",
    "enterpriseToRevenue",
    "priceToBook",
    "priceToSalesTrailing12Months",
    "freeCashflow",
    "totalRevenue",
    "ebitda",
    "dividendYield",
    "beta",
    "sharesShort",
    "shortRatio",
    "shortPercentOfFloat",
    "floatShares",
    "sharesOutstanding",
    "heldPercentInsiders",
    "heldPercentInstitutions",
    "dateShortInterest",
    "averageVolume",
    "averageDailyVolume10Day",
)


class YahooClient(Protocol):
    def info(self, symbol: str) -> dict[str, Any]: ...

    def industry_companies(self, industry_key: str) -> list[dict[str, Any]]: ...

    def sector_companies(self, sector_key: str) -> list[dict[str, Any]]: ...

    def weekly_closes(self, symbols: list[str]) -> pd.DataFrame: ...

    def frames(self, symbol: str) -> dict[str, Any]: ...


class LiveYahoo:
    """yfinance-backed client (network)."""

    def info(self, symbol: str) -> dict[str, Any]:
        import yfinance as yf

        raw: dict[str, Any] = yf.Ticker(symbol).get_info()
        return {k: raw.get(k) for k in INFO_FIELDS if raw.get(k) is not None}

    def industry_companies(self, industry_key: str) -> list[dict[str, Any]]:
        import yfinance as yf

        top = yf.Industry(industry_key).top_companies
        if top is None or top.empty:
            return []
        return [
            {"symbol": str(sym), "name": str(row.get("name", sym))} for sym, row in top.iterrows()
        ]

    def sector_companies(self, sector_key: str) -> list[dict[str, Any]]:
        import yfinance as yf

        top = yf.Sector(sector_key).top_companies
        if top is None or top.empty:
            return []
        return [
            {"symbol": str(sym), "name": str(row.get("name", sym))} for sym, row in top.iterrows()
        ]

    def weekly_closes(self, symbols: list[str]) -> pd.DataFrame:
        import yfinance as yf

        data = yf.download(
            symbols, period="2y", interval="1wk", auto_adjust=True, progress=False, threads=False
        )["Close"]
        return data if isinstance(data, pd.DataFrame) else data.to_frame(symbols[0])

    def frames(self, symbol: str) -> dict[str, Any]:
        import yfinance as yf

        t = yf.Ticker(symbol)
        out: dict[str, Any] = {}
        for name, getter in (
            ("earnings_dates", lambda: t.get_earnings_dates(limit=40)),
            ("earnings_estimate", lambda: t.earnings_estimate),
            ("revenue_estimate", lambda: t.revenue_estimate),
            ("eps_trend", lambda: t.eps_trend),
            ("eps_revisions", lambda: t.eps_revisions),
            ("recommendations_summary", lambda: t.recommendations_summary),
            ("price_targets", lambda: t.analyst_price_targets),
            ("insider_transactions", lambda: t.insider_transactions),
            ("institutional_holders", lambda: t.institutional_holders),
            ("major_holders", lambda: t.major_holders),
        ):
            try:
                out[name] = getter()
            except Exception:  # each feed is optional; missing ones are reported as absent
                logger.info("yfinance %s unavailable for %s", name, symbol, exc_info=True)
        return out


def _jsonable(value: Any) -> Any:
    """DataFrames → records with ISO dates; NaN → None."""
    if isinstance(value, pd.DataFrame):
        frame = value.reset_index()
        frame.columns = [str(c) for c in frame.columns]
        records = frame.to_dict("records")
        return [{k: _jsonable(v) for k, v in r.items()} for r in records]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _upsert_snapshot(
    engine: Engine, instrument_id: int, kind: str, payload: Any, source: str = "yfinance"
) -> None:
    with engine.begin() as conn:
        conn.execute(
            delete(provider_snapshots).where(
                (provider_snapshots.c.instrument_id == instrument_id)
                & (provider_snapshots.c.kind == kind)
            )
        )
        conn.execute(
            insert(provider_snapshots).values(
                instrument_id=instrument_id,
                kind=kind,
                source=source,
                payload=payload,
                as_of=datetime.now(UTC),
            )
        )


def _instrument_row(engine: Engine, symbol: str) -> Mapping[str, Any] | None:
    with engine.connect() as conn:
        return (
            conn.execute(select(instruments).where(instruments.c.symbol == symbol.upper()))
            .mappings()
            .one_or_none()
        )


def ingest_company_data(
    engine: Engine, symbol: str, job_id: int | None = None, client: YahooClient | None = None
) -> IngestResult:
    """Snapshot earnings, estimates, analyst views, holders, insiders and short interest."""
    symbol = symbol.upper()
    job_id = job_id or create_job(engine, "company_data", {"symbol": symbol})
    _start_job(engine, job_id, "Downloading company data")
    result = IngestResult(job_id=job_id, symbol=symbol, status="running", source="yfinance")
    row = _instrument_row(engine, symbol)
    if row is None or row["asset_class"] != "equity":
        return _finish(
            engine,
            result,
            "failed",
            "Company data applies to stored companies",
            error="not a company",
        )
    if client is None and not network_enabled():
        return _finish(
            engine,
            result,
            "failed",
            "Company data providers disabled (offline mode)",
            error="offline",
        )
    client = client or LiveYahoo()
    result.instrument_id = int(row["id"])
    try:
        info = client.info(symbol)
        frames = client.frames(symbol)
    except Exception as exc:
        return _finish(
            engine, result, "failed", "Provider error", error=f"{type(exc).__name__}: {exc}"
        )
    _upsert_snapshot(engine, result.instrument_id, "info", _jsonable(info))
    for kind, value in frames.items():
        if value is None or (isinstance(value, pd.DataFrame) and value.empty):
            continue
        _upsert_snapshot(engine, result.instrument_id, kind, _jsonable(value))
    result.rows_written = 1 + sum(1 for v in frames.values() if v is not None)
    return _finish(engine, result, "succeeded", f"{result.rows_written} snapshots")


def load_snapshots(engine: Engine, instrument_id: int) -> dict[str, Any]:
    with engine.connect() as conn:
        rows = (
            conn.execute(
                select(provider_snapshots).where(
                    provider_snapshots.c.instrument_id == instrument_id
                )
            )
            .mappings()
            .all()
        )
    return {
        r["kind"]: {
            "payload": r["payload"],
            "as_of": pd.Timestamp(r["as_of"]).isoformat(),
            "source": r["source"],
        }
        for r in rows
    }


# ── peers (D-PE-2) ───────────────────────────────────────────────────────────


@dataclass
class Candidate:
    symbol: str
    name: str
    info: dict[str, Any]
    breakdown: dict[str, float] = field(default_factory=dict)
    score: float = 0.0


def peer_metrics(info: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "marketCap",
        "trailingPE",
        "forwardPE",
        "enterpriseToEbitda",
        "enterpriseToRevenue",
        "priceToBook",
        "priceToSalesTrailing12Months",
        "grossMargins",
        "operatingMargins",
        "profitMargins",
        "revenueGrowth",
        "earningsGrowth",
        "returnOnEquity",
        "debtToEquity",
        "freeCashflow",
        "totalRevenue",
        "ebitda",
        "dividendYield",
        "beta",
        "currency",
    )
    return {k: info.get(k) for k in keys if info.get(k) is not None}


def score_candidates(
    subject: Mapping[str, Any],
    candidates: list[Candidate],
    closes: pd.DataFrame,
    subject_symbol: str,
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
) -> list[Candidate]:
    """D-PE-2: same industry, market cap within 0.25×–4×, then score = 0.4·industry depth +
    0.3·2y weekly-return correlation + 0.2·size proximity (log) + 0.1·margin similarity."""
    cap = subject.get("marketCap")
    returns = closes.pct_change().dropna(how="all")
    # D-PE-2 band 0.25×–4×, widened step by step for mega/micro caps with too few neighbours.
    for low, high in (SIZE_BAND, (0.1, 10.0), (0.0, math.inf)):
        ranked = _score(subject, candidates, returns, subject_symbol, weights, cap, low, high)
        if len(ranked) >= 5 or high == math.inf:
            band = f"{low:g}–{high:g}×" if math.isfinite(high) else "any size"
            for c in ranked:
                c.breakdown["size_band"] = band
            return ranked
    return []


def _score(
    subject: Mapping[str, Any],
    candidates: list[Candidate],
    returns: pd.DataFrame,
    subject_symbol: str,
    weights: Mapping[str, float],
    cap: Any,
    low: float,
    high: float,
) -> list[Candidate]:
    base = returns.get(subject_symbol, None)
    kept = []
    for c in candidates:
        other_cap = c.info.get("marketCap")
        if not cap or not other_cap or not (low <= other_cap / cap <= high):
            continue
        industry = (
            1.0
            if c.info.get("industryKey") == subject.get("industryKey")
            else (0.5 if c.info.get("sectorKey") == subject.get("sectorKey") else 0.0)
        )
        corr = 0.0
        if base is not None and c.symbol in returns:
            both = pd.concat([base, returns[c.symbol]], axis=1).dropna()
            corr = float(both.corr().iloc[0, 1]) if len(both) >= 26 else 0.0
        size = 1 - abs(math.log(other_cap / cap)) / math.log(10.0)
        gaps = [
            abs((c.info.get(k) or 0) - (subject.get(k) or 0))
            for k in ("grossMargins", "operatingMargins")
            if c.info.get(k) is not None and subject.get(k) is not None
        ]
        margins = max(0.0, 1 - float(np.mean(gaps)) / 0.3) if gaps else 0.0
        c.breakdown = {
            "industry": industry,
            "correlation": max(corr, 0.0),
            "size": max(size, 0.0),
            "margins": margins,
        }
        c.score = sum(weights[k] * v for k, v in c.breakdown.items())
        kept.append(c)
    return sorted(kept, key=lambda c: c.score, reverse=True)


def build_peer_group(
    engine: Engine,
    symbol: str,
    job_id: int | None = None,
    client: YahooClient | None = None,
    force: bool = False,
) -> IngestResult:
    """Propose 10 peers; a manual (edited) group is never overwritten unless `force`."""
    symbol = symbol.upper()
    job_id = job_id or create_job(engine, "peers", {"symbol": symbol})
    _start_job(engine, job_id, "Finding industry candidates")
    result = IngestResult(job_id=job_id, symbol=symbol, status="running", source="yfinance")
    row = _instrument_row(engine, symbol)
    if row is None or row["asset_class"] != "equity":
        return _finish(
            engine, result, "failed", "Peers apply to stored companies", error="not a company"
        )
    with engine.connect() as conn:
        group = (
            conn.execute(select(peer_groups).where(peer_groups.c.instrument_id == row["id"]))
            .mappings()
            .one_or_none()
        )
    if group is not None and group["mode"] == "manual" and not force:
        return _finish(engine, result, "succeeded", "Manual peer group kept (reset to rebuild)")
    if client is None and not network_enabled():
        return _finish(
            engine, result, "failed", "Peer providers disabled (offline mode)", error="offline"
        )
    client = client or LiveYahoo()
    result.instrument_id = int(row["id"])
    try:
        subject = client.info(symbol)
        industry_key = subject.get("industryKey")
        if not industry_key:
            return _finish(
                engine,
                result,
                "failed",
                "Industry unknown: add peers manually",
                error="no industry",
            )
        pool = [c for c in client.industry_companies(industry_key) if c["symbol"].upper() != symbol]
        if subject.get("sectorKey"):
            # Sector leaders widen the pool for mega caps whose industry list is small or tiny
            # (industry match then scores 0.5 instead of 1, per D-PE-2's depth term).
            pool += [
                c
                for c in client.sector_companies(subject["sectorKey"])
                if c["symbol"].upper() != symbol
            ]
        with engine.connect() as conn:
            stored = conn.execute(
                select(instruments.c.symbol, instruments.c.name).where(
                    (instruments.c.industry == row["industry"])
                    & (instruments.c.symbol != symbol)
                    & (instruments.c.asset_class == "equity")
                )
            ).all()
        known = {c["symbol"].upper() for c in pool}
        pool += [{"symbol": s, "name": n} for s, n in stored if s.upper() not in known]
        _update_job(engine, job_id, progress=0.4, message=f"Scoring {len(pool)} candidates")
        unique = list({c["symbol"].upper(): c for c in pool}.values())[:40]
        candidates = [
            Candidate(c["symbol"].upper(), c["name"], client.info(c["symbol"])) for c in unique
        ]
        closes = client.weekly_closes([symbol] + [c.symbol for c in candidates])
    except Exception as exc:
        return _finish(
            engine, result, "failed", "Provider error", error=f"{type(exc).__name__}: {exc}"
        )
    ranked = score_candidates(subject, candidates, closes, symbol)[:PEER_COUNT]
    basket = closes[[c.symbol for c in ranked if c.symbol in closes]].pct_change().mean(axis=1)
    details = {
        "subject": peer_metrics(subject),
        "industry_key": industry_key,
        "candidates": len(candidates),
        "weekly": {
            "dates": [d.date().isoformat() for d in closes.index],
            "subject": [None if not np.isfinite(v) else float(v) for v in closes[symbol]]
            if symbol in closes
            else [],
            "basket_returns": [None if not np.isfinite(v) else float(v) for v in basket],
        },
    }
    _store_group(engine, int(row["id"]), "auto", ranked, details)
    result.rows_written = len(ranked)
    return _finish(engine, result, "succeeded", f"{len(ranked)} peers proposed")


def _store_group(
    engine: Engine, instrument_id: int, mode: str, members: list[Candidate], details: dict[str, Any]
) -> None:
    with engine.begin() as conn:
        old_ids = select(peer_groups.c.id).where(peer_groups.c.instrument_id == instrument_id)
        # Explicit: SQLite does not enforce ON DELETE CASCADE without a pragma.
        conn.execute(
            delete(peer_group_members).where(peer_group_members.c.peer_group_id.in_(old_ids))
        )
        conn.execute(delete(peer_groups).where(peer_groups.c.instrument_id == instrument_id))
        group_id = conn.execute(
            insert(peer_groups).values(
                instrument_id=instrument_id,
                mode=mode,
                algorithm_version=PEER_ALGORITHM_VERSION,
                weights=DEFAULT_WEIGHTS,
                details=_jsonable(details),
                frozen_at=datetime.now(UTC) if mode == "manual" else None,
            )
        ).inserted_primary_key[0]
        for m in members:
            conn.execute(
                insert(peer_group_members).values(
                    peer_group_id=group_id,
                    symbol=m.symbol,
                    name=m.name,
                    score=m.score,
                    breakdown=m.breakdown,
                    metrics=_jsonable(peer_metrics(m.info)),
                    pinned=False,
                    source="auto",
                )
            )


def edit_peer_group(
    engine: Engine,
    symbol: str,
    add: list[str],
    remove: list[str],
    pin: list[str],
    info_lookup: Callable[[str], dict[str, Any]] | None = None,
) -> None:
    """Manual edit (PE-002): the group becomes `manual` and frozen until reset."""
    row = _instrument_row(engine, symbol)
    if row is None:
        raise ValueError(f"{symbol} unknown")
    with engine.begin() as conn:
        group = (
            conn.execute(select(peer_groups).where(peer_groups.c.instrument_id == row["id"]))
            .mappings()
            .one_or_none()
        )
        if group is None:
            group_id = conn.execute(
                insert(peer_groups).values(
                    instrument_id=row["id"],
                    mode="manual",
                    algorithm_version=PEER_ALGORITHM_VERSION,
                    weights=DEFAULT_WEIGHTS,
                    details={},
                    frozen_at=datetime.now(UTC),
                )
            ).inserted_primary_key[0]
        else:
            group_id = group["id"]
            conn.execute(
                update(peer_groups)
                .where(peer_groups.c.id == group_id)
                .values(mode="manual", frozen_at=datetime.now(UTC))
            )
        for sym in remove:
            conn.execute(
                delete(peer_group_members).where(
                    (peer_group_members.c.peer_group_id == group_id)
                    & (peer_group_members.c.symbol == sym.upper())
                )
            )
        existing = set(
            conn.execute(
                select(peer_group_members.c.symbol).where(
                    peer_group_members.c.peer_group_id == group_id
                )
            ).scalars()
        )
        for sym in add:
            if sym.upper() in existing or sym.upper() == row["symbol"]:
                continue
            info = info_lookup(sym) if info_lookup else {}
            conn.execute(
                insert(peer_group_members).values(
                    peer_group_id=group_id,
                    symbol=sym.upper(),
                    name=info.get("longName") or sym.upper(),
                    score=None,
                    breakdown={},
                    metrics=_jsonable(peer_metrics(info)),
                    pinned=True,
                    source="user",
                )
            )
        for sym in pin:
            conn.execute(
                update(peer_group_members)
                .where(
                    (peer_group_members.c.peer_group_id == group_id)
                    & (peer_group_members.c.symbol == sym.upper())
                )
                .values(pinned=True)
            )


def load_peer_group(engine: Engine, instrument_id: int) -> dict[str, Any] | None:
    with engine.connect() as conn:
        group = (
            conn.execute(select(peer_groups).where(peer_groups.c.instrument_id == instrument_id))
            .mappings()
            .one_or_none()
        )
        if group is None:
            return None
        members = (
            conn.execute(
                select(peer_group_members)
                .where(peer_group_members.c.peer_group_id == group["id"])
                .order_by(peer_group_members.c.score.desc().nulls_last())
            )
            .mappings()
            .all()
        )
    return {
        "mode": group["mode"],
        "algorithm_version": group["algorithm_version"],
        "weights": group["weights"],
        "details": group["details"],
        "created_at": pd.Timestamp(group["created_at"]).isoformat(),
        "frozen_at": pd.Timestamp(group["frozen_at"]).isoformat() if group["frozen_at"] else None,
        "members": [
            {
                "symbol": m["symbol"],
                "name": m["name"],
                "score": m["score"],
                "breakdown": m["breakdown"],
                "metrics": m["metrics"],
                "pinned": m["pinned"],
                "source": m["source"],
            }
            for m in members
        ],
    }
