"""Fundamentals ingestion (M02 §3, BACKEND_SCHEMA §4): SEC companyfacts first (US filers,
true point-in-time `filed` dates), yfinance statements as fallback (≈ 4 FY / 5 Q, filing
dates estimated as period end + 75 d FY / 45 d Q and flagged).

companyfacts' `fy`/`fp` describe the *filing*, not the fact: a fact's own period comes from
its start/end dates (≈ 3 months → Q, ≈ 12 months → FY) and the company's fiscal-year-end month.
"""

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Any

import httpx2 as httpx
import pandas as pd
import yaml
from sqlalchemy import Engine, delete, insert, select

from fda_data.ingest import (
    IngestResult,
    _finish,
    _record_catalog,
    _start_job,
    _update_job,
    create_job,
    network_enabled,
)
from fda_data.providers.base import ProviderError, SymbolNotFoundError
from fda_data.providers.http import get, make_client
from fda_data.ratelimit import TokenBucket
from fda_data.schema import fundamental_facts, instruments

logger = logging.getLogger(__name__)

MAPPING_DIR = Path(__file__).parent / "mapping"
FUNDAMENTALS_DATASET = "fundamentals"
FACT_COLUMNS = [
    "line_item",
    "period_type",
    "fiscal_year",
    "fiscal_quarter",
    "period_start",
    "period_end",
    "value",
    "unit",
    "filed_at",
    "source",
    "source_ref",
    "form",
    "is_restated",
    "estimated_pit",
]


@cache
def line_items() -> list[dict[str, Any]]:
    data = yaml.safe_load((MAPPING_DIR / "line_items.yaml").read_text(encoding="utf-8"))
    return list(data["items"])


@cache
def mapping(source: str) -> dict[str, dict[str, Any]]:
    """{code: {"tags": [...], "unit": str|None, "negate": bool}} from the source YAML."""
    data = yaml.safe_load((MAPPING_DIR / f"{source}.yaml").read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for code, spec in data["items"].items():
        if isinstance(spec, list):
            out[code] = {"tags": spec, "unit": None, "negate": False}
        else:
            out[code] = {
                "tags": spec.get("tags") or spec.get("fields"),
                "unit": spec.get("unit"),
                "negate": bool(spec.get("negate")),
            }
    return out


def mapping_version() -> str:
    versions = [
        yaml.safe_load((MAPPING_DIR / f"{name}.yaml").read_text(encoding="utf-8"))["version"]
        for name in ("line_items", "sec_xbrl", "yfinance")
    ]
    return ".".join(str(v) for v in versions)


# ── period logic ─────────────────────────────────────────────────────────────


def _month(day: date) -> int:
    """Month of a period end, shifted a week back so 52/53-week years ending on the first
    days of a month count as the previous month (Oct 1 → September)."""
    return (day - timedelta(days=7)).month


def fiscal_period(end: date, fy_end_month: int) -> tuple[int, int]:
    """(fiscal_year, fiscal_quarter) of a quarter ending on `end`."""
    month = _month(end)
    year = (end - timedelta(days=7)).year
    fiscal_year = year if month <= fy_end_month else year + 1
    quarter = ((month - fy_end_month - 1) % 12) // 3 + 1
    return fiscal_year, quarter


def duration_type(start: date | None, end: date) -> str | None:
    if start is None:
        return "I"
    days = (end - start).days
    if 80 <= days <= 100:
        return "Q"
    if 350 <= days <= 380:
        return "FY"
    if 170 <= days <= 190:
        return "H"
    return None  # 9-month YTD and other cumulative spans are skipped


# ── SEC companyfacts ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SecCompany:
    cik: int
    name: str


class SecProvider:
    """SEC EDGAR: ≤ 9 req/s with the owner's User-Agent from `.env` (SEC requirement)."""

    name = "sec_xbrl"
    TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

    def __init__(self, client: httpx.Client | None = None, user_agent: str | None = None) -> None:
        agent = user_agent or os.getenv("SEC_USER_AGENT", "")
        if not agent:
            raise ProviderError("sec: SEC_USER_AGENT is not set in .env")
        self._client = client or make_client(timeout=60)
        self._client.headers["User-Agent"] = agent
        self._limiter = TokenBucket(rate_per_second=8, burst=8)
        self._tickers: dict[str, SecCompany] | None = None

    def _get(self, url: str) -> Any:
        if not self._limiter.acquire(timeout=10):
            raise ProviderError("sec: local rate limit")
        return get(self._client, self.name, url).json()

    def company(self, symbol: str) -> SecCompany:
        if self._tickers is None:
            raw = self._get(self.TICKERS_URL)
            self._tickers = {
                str(row["ticker"]).upper(): SecCompany(int(row["cik_str"]), str(row["title"]))
                for row in raw.values()
            }
        key = symbol.upper().replace("-", ".")
        found = self._tickers.get(key) or self._tickers.get(symbol.upper().replace(".", "-"))
        if found is None:
            raise SymbolNotFoundError(f"sec: {symbol} is not an SEC filer")
        return found

    def company_facts(self, cik: int) -> dict[str, Any]:
        facts: dict[str, Any] = self._get(self.FACTS_URL.format(cik=cik))
        return facts


def _fy_end_month(facts: Mapping[str, Any]) -> int:
    """Most frequent end month of annual (≈ 12-month) facts reported in 10-K filings."""
    months: list[int] = []
    for taxonomy in facts.get("facts", {}).values():
        for concept in taxonomy.values():
            for rows in concept.get("units", {}).values():
                for row in rows:
                    if row.get("form") in {"10-K", "20-F", "40-F"} and row.get("start"):
                        start, end = (
                            date.fromisoformat(row["start"]),
                            date.fromisoformat(row["end"]),
                        )
                        if 350 <= (end - start).days <= 380:
                            months.append(_month(end))
            if len(months) > 50:
                break
    return max(set(months), key=months.count) if months else 12


def normalize_sec(facts: Mapping[str, Any]) -> pd.DataFrame:
    """companyfacts JSON → canonical long facts, one row per (item, period, filing)."""
    fy_end = _fy_end_month(facts)
    rows: list[dict[str, Any]] = []
    ytd_rows: list[dict[str, Any]] = []
    for code, spec in mapping("sec_xbrl").items():
        unit = spec["unit"] or "USD"
        claimed: set[tuple[str, date]] = set()  # (period_type, end) won by a higher-priority tag
        for priority, tag in enumerate(spec["tags"]):
            taxonomy, concept = tag.split(":")
            series = (
                facts.get("facts", {}).get(taxonomy, {}).get(concept, {}).get("units", {}).get(unit)
            )
            if not series:
                continue
            won: set[tuple[str, date]] = set()
            for row in series:
                if row.get("form") not in {
                    "10-K",
                    "10-Q",
                    "10-K/A",
                    "10-Q/A",
                    "20-F",
                    "40-F",
                    "8-K",
                }:
                    continue
                end = date.fromisoformat(row["end"])
                start = date.fromisoformat(row["start"]) if row.get("start") else None
                kind = duration_type(start, end)
                is_10q = row.get("form", "").startswith("10-Q")
                if start is not None and is_10q and kind in {None, "H"}:
                    days = (end - start).days
                    if 170 <= days <= 190 or 260 <= days <= 285:
                        ytd_rows.append(
                            {
                                "code": code,
                                "priority": priority,
                                "tag": tag,
                                "row": row,
                                "start": start,
                                "end": end,
                                "unit": unit,
                            }
                        )
                    continue  # 10-Q 6/9-month figures are year-to-date: quarters derive from them
                if kind is None:
                    continue
                if kind == "FY" and _month(end) != fy_end:
                    continue  # trailing-12-month disclosures in 10-Qs are not fiscal years
                if kind == "I":
                    kind = (
                        "FY"
                        if _month(end) == fy_end
                        and row.get("form", "").startswith(("10-K", "20-F", "40-F"))
                        else "Q"
                    )
                key = (kind, end)
                if key in claimed:
                    continue
                won.add(key)
                fiscal_year, quarter = fiscal_period(end, fy_end)
                if kind == "FY":
                    quarter = None
                elif kind == "H":
                    quarter = 2 if quarter <= 2 else 4
                rows.append(
                    {
                        "line_item": code,
                        "period_type": kind,
                        "fiscal_year": fiscal_year,
                        "fiscal_quarter": quarter,
                        "period_start": start,
                        "period_end": end,
                        "value": float(row["val"]),
                        "unit": unit,
                        "filed_at": date.fromisoformat(row["filed"]),
                        "source": "sec_xbrl",
                        "source_ref": f"{tag}|{row.get('accn', '')}",
                        "form": row.get("form"),
                        "priority": priority,
                    }
                )
            claimed |= won
    rows += _quarters_from_ytd(rows, ytd_rows, fy_end)
    rows = _snap_cover_instants(rows, fy_end)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=FACT_COLUMNS)
    return _finalise(frame, estimated=False)


def _quarters_from_ytd(
    rows: list[dict[str, Any]], ytd_rows: list[dict[str, Any]], fy_end: int
) -> list[dict[str, Any]]:
    """US 10-Qs often report cash flows only year-to-date (3, 6, 9 months): derive the
    missing quarters Q2 = H1 − Q1 and Q3 = 9M − H1, known on the later filing's date."""
    quarters = {
        (r["line_item"], r["fiscal_year"], r["fiscal_quarter"]): r
        for r in sorted(rows, key=lambda r: r["filed_at"], reverse=True)
        if r["period_type"] == "Q" and r["period_start"] is not None
    }
    # Highest-priority tag per (code, end), first filing.
    best: dict[tuple[str, date], dict[str, Any]] = {}
    for item in sorted(ytd_rows, key=lambda i: (i["priority"], i["row"]["filed"])):
        best.setdefault((item["code"], item["end"]), item)
    cumulative: dict[tuple[str, int, int], tuple[float, dict[str, Any]]] = {}
    for (code, _), item in best.items():
        fiscal_year, quarter = fiscal_period(item["end"], fy_end)
        cumulative[(code, fiscal_year, quarter)] = (float(item["row"]["val"]), item)
    derived: list[dict[str, Any]] = []
    for (code, fiscal_year, quarter), (value, item) in sorted(cumulative.items()):
        if (code, fiscal_year, quarter) in quarters or quarter not in (2, 3):
            continue
        if quarter == 2:
            previous = quarters.get((code, fiscal_year, 1))
            prev_value = previous["value"] if previous else None
        else:
            h1 = cumulative.get((code, fiscal_year, 2))
            prev_value = h1[0] if h1 else None
        if prev_value is None:
            continue
        row = item["row"]
        derived.append(
            {
                "line_item": code,
                "period_type": "Q",
                "fiscal_year": fiscal_year,
                "fiscal_quarter": quarter,
                "period_start": None,
                "period_end": item["end"],
                "value": value - prev_value,
                "unit": item["unit"],
                "filed_at": date.fromisoformat(row["filed"]),
                "source": "sec_xbrl",
                "source_ref": f"{item['tag']}|{row.get('accn', '')}|ytd-derived",
                "form": row.get("form"),
                "priority": item["priority"],
            }
        )
    return derived


def _snap_cover_instants(rows: list[dict[str, Any]], fy_end: int) -> list[dict[str, Any]]:
    """Cover-page share counts are dated after the quarter end: attach them to the latest
    reporting period end within 120 days before, and drop those that match none."""
    period_ends = sorted({r["period_end"] for r in rows if r["period_start"] is not None})
    out = []
    for r in rows:
        if r["line_item"] != "shares_outstanding" or r["period_end"] in period_ends:
            out.append(r)
            continue
        earlier = [
            e for e in period_ends if e <= r["period_end"] and (r["period_end"] - e).days <= 120
        ]
        if earlier:
            end = earlier[-1]
            fiscal_year, quarter = fiscal_period(end, fy_end)
            is_fy = r["period_type"] == "FY" and _month(end) == fy_end
            out.append(
                r
                | {
                    "period_end": end,
                    "fiscal_year": fiscal_year,
                    "fiscal_quarter": None if is_fy else quarter,
                }
            )
    return out


def _finalise(frame: pd.DataFrame, estimated: bool) -> pd.DataFrame:
    """One row per filing; flag values changed by later filings as restated."""
    frame = frame.sort_values(["line_item", "period_type", "period_end", "filed_at"])
    frame = frame.drop_duplicates(
        ["line_item", "period_type", "period_end", "filed_at"], keep="first"
    )
    first_value = frame.groupby(["line_item", "period_type", "period_end"])["value"].transform(
        "first"
    )
    frame["is_restated"] = (frame["value"] - first_value).abs() > 1e-6 * first_value.abs().clip(
        lower=1
    )
    frame["estimated_pit"] = estimated
    return frame[FACT_COLUMNS].reset_index(drop=True)


# ── yfinance statements ──────────────────────────────────────────────────────


def normalize_yfinance(
    statements: Mapping[str, pd.DataFrame], currency: str, fy_end_month: int | None = None
) -> pd.DataFrame:
    """yfinance annual/quarterly statement frames (rows = fields, columns = period ends)."""
    rows: list[dict[str, Any]] = []
    annual_ends = [
        pd.Timestamp(c).date()
        for name, f in statements.items()
        if name.startswith("annual")
        for c in f.columns
    ]
    fy_end = fy_end_month or (
        max({_month(d) for d in annual_ends}, key=[_month(d) for d in annual_ends].count)
        if annual_ends
        else 12
    )
    for name, frame in statements.items():
        kind = "FY" if name.startswith("annual") else "Q"
        for code, spec in mapping("yfinance").items():
            for field in spec["tags"]:
                if field not in frame.index:
                    continue
                for column, value in frame.loc[field].items():
                    if pd.isna(value):
                        continue
                    end = pd.Timestamp(column).date()
                    fiscal_year, quarter = fiscal_period(end, fy_end)
                    lag = 75 if kind == "FY" else 45
                    unit = (
                        "shares"
                        if code.startswith("shares")
                        else f"{currency}/shares"
                        if code.startswith("eps")
                        else currency
                    )
                    rows.append(
                        {
                            "line_item": code,
                            "period_type": kind,
                            "fiscal_year": fiscal_year,
                            "fiscal_quarter": None if kind == "FY" else quarter,
                            "period_start": None,
                            "period_end": end,
                            "value": -float(value) if spec["negate"] else float(value),
                            "unit": unit,
                            "filed_at": end + timedelta(days=lag),
                            "source": "yfinance",
                            "source_ref": field,
                            "form": "annual_report" if kind == "FY" else "interim",
                        }
                    )
                break  # first field present wins for this code and statement frequency
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=FACT_COLUMNS)
    frame = frame.drop_duplicates(["line_item", "period_type", "period_end"], keep="first")
    return _finalise(frame, estimated=True)


# Share counts and EPS SEC filers may tag only per share class (dimensional facts, absent from
# companyfacts): Visa, Berkshire. Spec 274.
SHARE_ITEMS = (
    "shares_outstanding",
    "shares_basic_wavg",
    "shares_diluted_wavg",
    "eps_basic",
    "eps_diluted",
)
SHARE_MAX_AGE_DAYS = 400


def _latest_results(facts: pd.DataFrame) -> pd.Timestamp:
    ends = pd.to_datetime(facts["period_end"])
    income = ends[facts["line_item"] == "net_income"]
    return (income if len(income) else ends).max()


def needs_share_supplement(facts: pd.DataFrame) -> bool:
    """True when no share count or no diluted EPS is current (≤ 400 days before the latest
    net income period end)."""
    if facts.empty:
        return False
    ends = pd.to_datetime(facts["period_end"])
    reference = _latest_results(facts)

    def current(codes: tuple[str, ...]) -> bool:
        mine = ends[facts["line_item"].isin(codes)]
        return bool(len(mine)) and (reference - mine.max()).days <= SHARE_MAX_AGE_DAYS

    return not current(("shares_outstanding", "shares_diluted_wavg")) or not current(
        ("eps_diluted",)
    )


def supplement_shares(sec: pd.DataFrame, yahoo: pd.DataFrame) -> pd.DataFrame:
    """SEC facts + yfinance share/EPS facts for the (item, period type, period end) keys SEC
    lacks, up to SEC's latest results (a later period would open a statement column with no
    flows); the added rows keep `source = yfinance` and `estimated_pit = True`."""
    extra = yahoo[
        yahoo["line_item"].isin(SHARE_ITEMS)
        & (pd.to_datetime(yahoo["period_end"]) <= _latest_results(sec))
    ]
    if extra.empty:
        return sec
    keys = ["line_item", "period_type", "period_end"]
    have = set(map(tuple, sec[keys].astype({"period_end": str}).to_numpy()))
    new = extra[[tuple(k) not in have for k in extra[keys].astype({"period_end": str}).to_numpy()]]
    if new.empty:
        return sec
    return pd.concat([sec, new], ignore_index=True)[FACT_COLUMNS]


def _yfinance_statements(symbol: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    return {
        "annual_income": ticker.income_stmt,
        "annual_balance": ticker.balance_sheet,
        "annual_cashflow": ticker.cashflow,
        "quarterly_income": ticker.quarterly_income_stmt,
        "quarterly_balance": ticker.quarterly_balance_sheet,
        "quarterly_cashflow": ticker.quarterly_cashflow,
    }


# ── ingestion ────────────────────────────────────────────────────────────────


def ingest_fundamentals(
    engine: Engine,
    symbol: str,
    job_id: int | None = None,
    sec: SecProvider | None = None,
    yahoo_statements: Any = _yfinance_statements,
) -> IngestResult:
    """Replace the instrument's facts by a fresh normalised set (SEC first, else yfinance)."""
    symbol = symbol.upper()
    if job_id is None:
        job_id = create_job(engine, "fundamentals", {"symbol": symbol})
    _start_job(engine, job_id, "Resolving filer")
    result = IngestResult(job_id=job_id, symbol=symbol, status="running")
    with engine.connect() as conn:
        row = (
            conn.execute(select(instruments).where(instruments.c.symbol == symbol))
            .mappings()
            .one_or_none()
        )
    if row is None:
        return _finish(
            engine,
            result,
            "failed",
            "Ingest prices first (instrument unknown)",
            error="unknown instrument",
        )
    if row["asset_class"] != "equity":
        return _finish(
            engine,
            result,
            "failed",
            f"No company statements for a {row['asset_class']} asset",
            error="not a company",
        )
    result.instrument_id = int(row["id"])
    if sec is None and not network_enabled():
        return _finish(
            engine,
            result,
            "failed",
            "Statement providers disabled (offline mode)",
            error="FDA_PROVIDERS has no network provider",
        )
    facts = pd.DataFrame(columns=FACT_COLUMNS)
    try:
        provider = sec or SecProvider()
        company = provider.company(symbol)
        _update_job(
            engine,
            job_id,
            progress=0.3,
            message=f"Downloading SEC companyfacts (CIK {company.cik})",
        )
        facts = normalize_sec(provider.company_facts(company.cik))
        result.source = "sec_xbrl"
    except ProviderError as exc:
        logger.info("SEC unavailable for %s (%s); falling back to yfinance", symbol, exc)
    online = network_enabled() or yahoo_statements is not _yfinance_statements
    if not facts.empty and online and needs_share_supplement(facts):
        try:
            _update_job(engine, job_id, progress=0.5, message="Share counts from yfinance")
            fy_ends = [_month(d) for d in facts.loc[facts["period_type"] == "FY", "period_end"]]
            yahoo = normalize_yfinance(
                yahoo_statements(symbol),
                row["currency"],
                max(set(fy_ends), key=fy_ends.count) if fy_ends else None,
            )
            facts = supplement_shares(facts, yahoo)
        except Exception as exc:  # yfinance raises many unrelated types
            logger.info("yfinance share supplement failed for %s (%s)", symbol, exc)
    if facts.empty:
        try:
            _update_job(engine, job_id, progress=0.3, message="Downloading yfinance statements")
            facts = normalize_yfinance(yahoo_statements(symbol), row["currency"])
            result.source = "yfinance"
        except Exception as exc:  # yfinance raises many unrelated types
            return _finish(engine, result, "failed", "No statements available", error=str(exc))
    if facts.empty:
        return _finish(engine, result, "failed", "No statements available", error="empty")
    _update_job(engine, job_id, progress=0.7, message="Writing facts")
    now = datetime.now(UTC)
    records = facts.assign(instrument_id=result.instrument_id, ingested_at=now).to_dict("records")
    for record in records:
        for key in ("period_start", "fiscal_quarter"):
            if pd.isna(record[key]):
                record[key] = None
        record["is_restated"] = bool(record["is_restated"])
        record["estimated_pit"] = bool(record["estimated_pit"])
        record["fiscal_quarter"] = (
            int(record["fiscal_quarter"]) if record["fiscal_quarter"] is not None else None
        )
    with engine.begin() as conn:
        conn.execute(
            delete(fundamental_facts).where(
                fundamental_facts.c.instrument_id == result.instrument_id
            )
        )
        for start in range(0, len(records), 1_000):
            conn.execute(insert(fundamental_facts), records[start : start + 1_000])
        ends = pd.to_datetime(facts["period_end"]).dt.date
        _record_catalog(
            conn,
            FUNDAMENTALS_DATASET,
            str(result.instrument_id),
            (ends.min(), ends.max(), len(facts)),
            str(result.source),
            result.instrument_id,
        )
    result.rows_written = len(records)
    result.qc = {
        "facts": len(records),
        "items": int(facts["line_item"].nunique()),
        "restated": int(facts["is_restated"].sum()),
        "mapping_version": mapping_version(),
    }
    return _finish(engine, result, "succeeded", f"{len(records)} facts from {result.source}")


def load_facts(engine: Engine, instrument_id: int, as_of: date | None = None) -> pd.DataFrame:
    """Point-in-time view: for each (item, period type, period end) the latest filing with
    filed_at ≤ as_of (BACKEND_SCHEMA §4.1 PIT rule)."""
    with engine.connect() as conn:
        frame = pd.DataFrame(
            conn.execute(
                select(fundamental_facts).where(fundamental_facts.c.instrument_id == instrument_id)
            )
            .mappings()
            .all()
        )
    if frame.empty:
        return frame
    if as_of is not None:
        frame = frame[pd.to_datetime(frame["filed_at"]).dt.date <= as_of]
    frame = frame.sort_values("filed_at").drop_duplicates(
        ["line_item", "period_type", "period_end"], keep="last"
    )
    return frame.reset_index(drop=True)


def load_all_facts(engine: Engine, instrument_id: int) -> pd.DataFrame:
    """Every stored filing (original and restated values) for point-in-time histories."""
    with engine.connect() as conn:
        return pd.DataFrame(
            conn.execute(
                select(fundamental_facts).where(fundamental_facts.c.instrument_id == instrument_id)
            )
            .mappings()
            .all()
        )
