"""Filings (M30 §5, AI-004): EDGAR filing index and documents, cached as extracted text in the
lake (`filings/cik=…/<accession>.json`). Cache-first; SEC limits respected (≤ 8 req/s with the
owner's User-Agent). Offline runs (`FDA_PROVIDERS=fake`) use deterministic synthetic 10-Ks."""

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, ClassVar, Protocol

from fda_core.filings import html_to_text
from fda_data.fundamentals import SecProvider
from fda_data.ingest import network_enabled
from fda_data.lake import PriceLake
from fda_data.providers.base import ProviderError, SymbolNotFoundError
from fda_data.providers.http import get

FORMS = ("10-K", "10-K/A", "10-Q", "20-F", "40-F", "8-K")


@dataclass(frozen=True, slots=True)
class Filing:
    accession: str
    form: str
    filed: str
    period: str | None
    document: str
    url: str


class FilingSource(Protocol):
    def filings(self, symbol: str) -> tuple[int, str, list[Filing]]: ...
    def text(self, cik: int, filing: Filing) -> str: ...


class SecFilings:
    SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
    DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"

    def __init__(self, provider: SecProvider | None = None) -> None:
        self._sec = provider or SecProvider()

    def filings(self, symbol: str) -> tuple[int, str, list[Filing]]:
        company = self._sec.company(symbol)
        data = self._sec._get(self.SUBMISSIONS_URL.format(cik=company.cik))
        recent = data.get("filings", {}).get("recent", {})
        rows = zip(
            recent.get("accessionNumber", []),
            recent.get("form", []),
            recent.get("filingDate", []),
            recent.get("reportDate", []),
            recent.get("primaryDocument", []),
            strict=False,
        )
        out = []
        for acc, form, filed, period, doc in rows:
            if form not in FORMS or not doc:
                continue
            url = self.DOC_URL.format(cik=company.cik, acc=acc.replace("-", ""), doc=doc)
            out.append(Filing(acc, form, filed, period or None, doc, url))
        return company.cik, company.name, out

    def text(self, cik: int, filing: Filing) -> str:
        if not self._sec._limiter.acquire(timeout=10):
            raise ProviderError("sec: local rate limit")
        response = get(self._sec._client, "sec_edgar", filing.url)
        return html_to_text(response.text)


class FakeFilings:
    """Two synthetic annual reports per FAKE* symbol: one risk added, one removed, one expanded."""

    LOREM = "Results could be affected in ways that are difficult to predict and may be material. "
    RISKS: ClassVar[dict[int, dict[str, int]]] = {
        2024: {
            "Competition could reduce our margins and market share.": 3,
            "We depend on a small number of suppliers for key components.": 2,
            "Changes in interest rates could raise our financing costs.": 2,
        },
        2025: {
            "Competition could reduce our margins and our market share.": 3,
            "We depend on a small number of suppliers for key components.": 6,
            "Cyberattacks could disrupt operations and expose customer data.": 3,
        },
    }

    def filings(self, symbol: str) -> tuple[int, str, list[Filing]]:
        if not symbol.upper().startswith("FAKE"):
            raise SymbolNotFoundError(f"fake: {symbol} has no filings")
        cik = 900000 + sum(map(ord, symbol.upper())) % 9999
        return (
            cik,
            f"{symbol.upper()} Corp",
            [
                Filing(
                    f"0000{cik}-{y % 100}-000001",
                    "10-K",
                    f"{y}-02-15",
                    f"{y - 1}-12-31",
                    "10k.htm",
                    f"https://example.invalid/{symbol}/{y}",
                )
                for y in (2025, 2024)
            ],
        )

    def text(self, cik: int, filing: Filing) -> str:
        year = int(filing.filed[:4])
        risks = "\n".join(f"{h}\n{self.LOREM * n}" for h, n in self.RISKS[year].items())
        return (
            "PART I\nItem 1. Business\nWe design and sell widgets worldwide. "
            + self.LOREM * 6
            + "\n"
            "Item 1A. Risk Factors\nInvesting in our shares involves risks.\n" + risks + "\n"
            "Item 1B. Unresolved Staff Comments\nNone.\n"
            "Item 7. Management's Discussion and Analysis\nRevenue grew. " + self.LOREM * 10 + "\n"
            "Item 8. Financial Statements and Supplementary Data\nSee the statements.\n"
        )


def default_source() -> FilingSource:
    return SecFilings() if network_enabled() else FakeFilings()


SYMBOL = re.compile(r"^[A-Za-z0-9.^=-]{1,20}$")  # no path separators: names stay in the folder
ACCESSION = re.compile(r"^[0-9]{10}-[0-9]{2}-[0-9]{6}$")


def _dir(lake: PriceLake, cik: int) -> Path:
    return lake.root / "filings" / f"cik={int(cik)}"


def _pointer(lake: PriceLake, symbol: str) -> Path | None:
    """Symbol pointer file, or None for anything that is not a plain symbol."""
    if not SYMBOL.match(symbol):
        return None
    root = (lake.root / "filings").resolve()
    path = (root / f"symbol={symbol.upper()}.json").resolve()
    return path if path.parent == root else None


def ingest_filings(
    lake: PriceLake,
    symbol: str,
    forms: tuple[str, ...] = ("10-K", "20-F"),
    count: int = 2,
    source: FilingSource | None = None,
) -> dict[str, Any]:
    """Index + the latest `count` documents of the given forms, cache-first (a cached accession
    is never re-downloaded: filings are immutable; amendments are separate accessions)."""
    pointer = _pointer(lake, symbol)
    if pointer is None:
        raise SymbolNotFoundError(f"Invalid symbol {symbol[:24]!r}")
    source = source or default_source()
    cik, name, filings = source.filings(symbol)
    folder = _dir(lake, cik)
    folder.mkdir(parents=True, exist_ok=True)
    index = {
        "symbol": symbol.upper(),
        "cik": cik,
        "name": name,
        "as_of": date.today().isoformat(),
        "filings": [asdict(f) for f in filings[:200]],
    }
    (folder / "index.json").write_text(json.dumps(index), "utf-8")
    pointer.write_text(json.dumps({"cik": cik}), "utf-8")
    wanted = [f for f in filings if f.form in forms][:count]
    fetched = 0
    for filing in wanted:
        if not ACCESSION.match(filing.accession):
            continue  # never build a path from an unexpected accession string
        target = folder / f"{filing.accession}.json"
        if target.exists():
            continue
        text = source.text(cik, filing)
        target.write_text(json.dumps({**asdict(filing), "text": text}), "utf-8")
        fetched += 1
    return {
        "cik": cik,
        "name": name,
        "indexed": len(filings),
        "documents": len(wanted),
        "fetched": fetched,
    }


def read_filings(lake: PriceLake, symbol: str) -> dict[str, Any] | None:
    """Index + cached documents (newest first) for a symbol; None when never ingested."""
    pointer = _pointer(lake, symbol)
    if pointer is None or not pointer.exists():
        return None
    cik = int(json.loads(pointer.read_text("utf-8"))["cik"])
    folder = _dir(lake, cik)
    index = json.loads((folder / "index.json").read_text("utf-8"))
    docs = [
        json.loads(p.read_text("utf-8")) for p in folder.glob("*.json") if p.name != "index.json"
    ]
    docs.sort(key=lambda d: d["filed"], reverse=True)
    return index | {"documents": docs}


def filings_version(lake: PriceLake, symbol: str) -> str:
    pointer = _pointer(lake, symbol)
    if pointer is None or not pointer.exists():
        return "none"
    cik = int(json.loads(pointer.read_text("utf-8"))["cik"])
    names = sorted(p.stem for p in _dir(lake, cik).glob("*.json"))
    stamp = os.path.getmtime(_dir(lake, cik) / "index.json")
    return f"{cik}:{len(names)}:{names[-1] if names else ''}:{stamp:.0f}"
