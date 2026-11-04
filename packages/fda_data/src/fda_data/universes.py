"""Universes (SC-001): constituent sources, membership with dates, metric snapshots (SC-002).

Sources: Wikipedia constituent tables (S&P 500, Nasdaq-100, Dow 30, CAC 40), CoinGecko top
coins by market cap (stablecoins and wrapped tokens excluded), a pasted list, or a portfolio's
holdings. Wikipedia pages are parsed with the standard library (`html.parser`): the first table
whose header has the symbol column and enough rows. Free sources give today's list only, so
membership is tracked from the first refresh on (leavers get `valid_to`); screens on earlier
dates are survivorship-biased (documented in the UI).

Snapshots live in the lake: `universe_snapshots/universe=<id>/as_of=<date>/part.parquet`, one row
per member, one column per metric.
"""

import os
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import httpx2 as httpx
import pandas as pd
import pyarrow as pa
from sqlalchemy import Engine, delete, func, insert, select, update

from fda_data.actor import current_owner
from fda_data.lake import PriceLake, _atomic_write
from fda_data.providers.base import ProviderError
from fda_data.providers.http import get, make_client
from fda_data.schema import saved_screens, universe_members, universes

WIKI = "https://en.wikipedia.org/wiki/"
COINGECKO_MARKETS = "https://api.coingecko.com/api/v3/coins/markets"
STABLE_OR_WRAPPED = {
    "USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE", "USDS", "PYUSD", "BUSD", "USDD", "USD1",
    "FRAX", "WBTC", "WETH", "STETH", "WSTETH", "WEETH", "CBBTC", "RETH", "BSC-USD", "SUSDE",
    "USDTB", "BUIDL", "USDF", "RLUSD", "USDG", "USYC", "WBETH", "LBTC", "SOLVBTC", "JITOSOL",
    "XAUT", "PAXG", "BFUSD", "USDC.E", "SUSDS", "MSOL", "BNSOL", "CLBTC", "WBNB",
}  # fmt: skip


class UniverseError(ValueError):
    """Invalid universe request (unknown source, empty list, unknown id)."""


@dataclass(frozen=True, slots=True)
class Member:
    symbol: str
    name: str | None = None
    sector: str | None = None


@dataclass(frozen=True, slots=True)
class IndexSource:
    label: str
    page: str
    symbol_columns: tuple[str, ...]
    name_columns: tuple[str, ...]
    sector_columns: tuple[str, ...]
    minimum: int


SOURCES: dict[str, IndexSource] = {
    "sp500": IndexSource("S&P 500", "List_of_S%26P_500_companies", ("symbol",), ("security",),
                         ("gics sector",), 480),
    "nasdaq100": IndexSource("Nasdaq-100", "List_of_NASDAQ-100_companies", ("ticker",),
                             ("company",), ("icb industry", "gics sector"), 95),
    "dow30": IndexSource("Dow 30", "List_of_Dow_Jones_Industrial_Average_companies",
                         ("symbol",), ("company",), ("sector", "industry"), 30),
    "cac40": IndexSource("CAC 40", "CAC_40", ("ticker",), ("company",), ("sector", "gics sector"),
                         40),
}  # fmt: skip
CRYPTO_SOURCE = "crypto50"


class _Tables(HTMLParser):
    """Collects top-level <table>s as rows of cell texts (th and td)."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._depth = 0
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._depth += 1
            if self._depth == 1:
                self.tables.append([])
        elif self._depth == 1 and tag == "tr":
            self._row = []
        elif self._depth == 1 and tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "table":
            self._depth -= 1
        elif self._depth != 1:
            return
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.tables[-1].append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def _header(cell: str) -> str:
    return re.sub(r"\[\d+\]", "", cell).strip().lower()


def yahoo_symbol(raw: str) -> str:
    """Share classes use a dash on Yahoo (BRK.B → BRK-B); exchange suffixes stay (AI.PA)."""
    return re.sub(r"\.([A-Z])$", r"-\1", raw.strip().upper())


def parse_constituents(html: str, source: IndexSource) -> list[Member]:
    parser = _Tables()
    parser.feed(html)
    for table in parser.tables:
        if len(table) <= source.minimum:
            continue
        header = [_header(h) for h in table[0]]
        sym = next((header.index(c) for c in source.symbol_columns if c in header), None)
        if sym is None:
            continue
        name = next((header.index(c) for c in source.name_columns if c in header), None)
        sector = next((header.index(c) for c in source.sector_columns if c in header), None)

        def cell(row: list[str], i: int | None) -> str | None:
            return row[i] or None if i is not None and len(row) > i else None

        members = [
            Member(yahoo_symbol(row[sym].split()[0]), cell(row, name), cell(row, sector))
            for row in table[1:]
            if len(row) > sym and row[sym].strip()
        ]
        if len(members) >= source.minimum:
            return members
    raise ProviderError(f"No constituents table with ≥ {source.minimum} rows on {source.page}")


def parse_pasted(text: str) -> list[Member]:
    """One symbol per line or comma-separated; an optional ';name;sector' tail per line."""
    members: dict[str, Member] = {}
    for line in re.split(r"[\n,]", text):
        parts = [p.strip() for p in line.split(";")]
        if not parts[0]:
            continue
        symbol = parts[0].upper()
        if not re.fullmatch(r"[A-Z0-9^.=\-]{1,20}", symbol):
            raise UniverseError(f"Invalid symbol {parts[0]!r}")
        members[symbol] = Member(
            symbol,
            parts[1] if len(parts) > 1 and parts[1] else None,
            parts[2] if len(parts) > 2 and parts[2] else None,
        )
    if not members:
        raise UniverseError("Paste at least one symbol")
    return list(members.values())


def _client() -> httpx.Client:
    """Wikipedia requires a contact in the User-Agent: reuse the SEC one from .env."""
    client = make_client(timeout=30.0)
    agent = os.environ.get("SEC_USER_AGENT")
    if agent:
        client.headers["User-Agent"] = agent
    return client


def fetch_members(source: str, client: httpx.Client | None = None, top: int = 50) -> list[Member]:
    client = client or _client()
    if source == CRYPTO_SOURCE:
        coins = get(
            client,
            "coingecko",
            COINGECKO_MARKETS,
            vs_currency="usd",
            order="market_cap_desc",
            per_page="100",
            page="1",
        ).json()
        members = [
            Member(f"{str(c['symbol']).upper()}-USD", str(c.get("name") or ""), "Crypto")
            for c in coins
            if str(c.get("symbol", "")).upper() not in STABLE_OR_WRAPPED
        ]
        return members[:top]
    spec = SOURCES.get(source)
    if spec is None:
        raise UniverseError(f"Unknown universe source {source!r}")
    return parse_constituents(get(client, "wikipedia", WIKI + spec.page).text, spec)


# ── persistence ──────────────────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now(UTC)


def save_members(
    engine: Engine, universe_id: int, members: list[Member], today: date
) -> dict[str, list[str]]:
    """Diffs today's list with the open membership: entrants open a row, leavers get valid_to."""
    with engine.begin() as conn:
        current = {
            r.symbol: r
            for r in conn.execute(
                select(universe_members).where(
                    universe_members.c.universe_id == universe_id,
                    universe_members.c.valid_to.is_(None),
                )
            )
        }
        incoming = {m.symbol: m for m in members}
        entrants = sorted(set(incoming) - set(current))
        leavers = sorted(set(current) - set(incoming))
        if entrants:
            conn.execute(
                insert(universe_members),
                [
                    {
                        "universe_id": universe_id,
                        "symbol": s,
                        "name": incoming[s].name,
                        "sector": incoming[s].sector,
                        "valid_from": today,
                    }
                    for s in entrants
                ],
            )
        for s in set(incoming) & set(current):
            m = incoming[s]
            if (m.name, m.sector) != (current[s].name, current[s].sector):
                conn.execute(
                    update(universe_members)
                    .where(universe_members.c.id == current[s].id)
                    .values(name=m.name or current[s].name, sector=m.sector or current[s].sector)
                )
        if leavers:
            conn.execute(
                update(universe_members)
                .where(universe_members.c.id.in_([current[s].id for s in leavers]))
                .values(valid_to=today)
            )
        conn.execute(
            update(universes)
            .where(universes.c.id == universe_id)
            .values(refreshed_at=_now(), updated_at=_now())
        )
    return {"entrants": entrants, "leavers": leavers}


SYSTEM_KINDS = frozenset({"index", "crypto"})


def create_universe(
    engine: Engine, name: str, kind: str, source: str | None, members: list[Member], today: date
) -> int:
    with engine.begin() as conn:
        exists = conn.execute(select(universes.c.id).where(universes.c.name == name)).scalar()
        if exists is not None:
            raise UniverseError(f"A universe named {name!r} exists")
        uid = int(
            conn.execute(
                insert(universes).values(
                    name=name,
                    kind=kind,
                    source=source,
                    # index / crypto lists are shared reference data (system rows)
                    owner_id=None if kind in SYSTEM_KINDS else current_owner(),
                )
            ).inserted_primary_key[0]
        )  # type: ignore[index]
    save_members(engine, uid, members, today)
    return uid


def list_universes(engine: Engine) -> list[dict[str, Any]]:
    with engine.connect() as conn:
        rows = conn.execute(select(universes).order_by(universes.c.name)).mappings().all()
        counts = dict(
            conn.execute(
                select(universe_members.c.universe_id, func.count(universe_members.c.id))
                .where(universe_members.c.valid_to.is_(None))
                .group_by(universe_members.c.universe_id)
            ).all()
        )
    return [dict(r) | {"members": int(counts.get(r["id"], 0))} for r in rows]


def load_universe(engine: Engine, universe_id: int) -> dict[str, Any]:
    with engine.connect() as conn:
        row = (
            conn.execute(select(universes).where(universes.c.id == universe_id))
            .mappings()
            .one_or_none()
        )
        if row is None:
            raise UniverseError(f"Unknown universe {universe_id}")
        members = (
            conn.execute(
                select(
                    universe_members.c.symbol,
                    universe_members.c.name,
                    universe_members.c.sector,
                    universe_members.c.valid_from,
                )
                .where(
                    universe_members.c.universe_id == universe_id,
                    universe_members.c.valid_to.is_(None),
                )
                .order_by(universe_members.c.symbol)
            )
            .mappings()
            .all()
        )
    return dict(row) | {"members": [dict(m) for m in members]}


def delete_universe(engine: Engine, universe_id: int) -> None:
    with engine.begin() as conn:
        conn.execute(delete(saved_screens).where(saved_screens.c.universe_id == universe_id))
        conn.execute(delete(universe_members).where(universe_members.c.universe_id == universe_id))
        conn.execute(delete(universes).where(universes.c.id == universe_id))


# ── snapshots (lake) ─────────────────────────────────────────────────────────


def _snapshot_dir(lake: PriceLake, universe_id: int) -> Path:
    return lake.root / "universe_snapshots" / f"universe={universe_id}"


def write_snapshot(lake: PriceLake, universe_id: int, as_of: date, frame: pd.DataFrame) -> Path:

    target = _snapshot_dir(lake, universe_id) / f"as_of={as_of.isoformat()}" / "part.parquet"
    table = pa.Table.from_pandas(frame.reset_index(drop=True), preserve_index=False)
    _atomic_write(frame.reset_index(drop=True), table.schema, target)
    return target


def snapshot_dates(lake: PriceLake, universe_id: int) -> list[date]:
    root = _snapshot_dir(lake, universe_id)
    if not root.exists():
        return []
    return sorted(
        date.fromisoformat(p.name.split("=", 1)[1])
        for p in root.glob("as_of=*")
        if (p / "part.parquet").exists()
    )


def read_snapshot(
    lake: PriceLake, universe_id: int, as_of: date | None = None
) -> tuple[date, pd.DataFrame] | None:
    dates = snapshot_dates(lake, universe_id)
    if not dates:
        return None
    chosen = (
        as_of
        if as_of in dates
        else dates[-1]
        if as_of is None
        else max((d for d in dates if d <= as_of), default=None)
    )
    if chosen is None:
        return None
    path = _snapshot_dir(lake, universe_id) / f"as_of={chosen.isoformat()}" / "part.parquet"
    return chosen, pd.read_parquet(path)
