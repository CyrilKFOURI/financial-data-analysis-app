"""News headlines (M30 §4, AI-005): free provider headlines per ticker, cached in the lake
(`news/symbol=<SYMBOL>.json`) and accumulated across fetches (providers only return the latest
few). Each headline keeps its sentiment once scored (by the configured LLM, API layer) and a
digest can be stored with its sources. Offline runs (`FDA_PROVIDERS=fake`) use deterministic
synthetic headlines for FAKE* symbols."""

import hashlib
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Protocol

from fda_data.ingest import network_enabled
from fda_data.lake import PriceLake
from fda_data.providers.base import ProviderError, SymbolNotFoundError

SYMBOL = re.compile(r"^[A-Za-z0-9.^=-]{1,20}$")  # no path separators: names stay in the folder
MAX_ITEMS = 2000


@dataclass(frozen=True, slots=True)
class Headline:
    id: str
    published: str  # ISO 8601, UTC
    title: str
    publisher: str
    url: str
    source: str
    summary: str = ""


def headline_id(url: str, title: str) -> str:
    return hashlib.sha1(f"{url}|{title}".encode()).hexdigest()[:10]


class NewsSource(Protocol):
    def headlines(self, symbol: str) -> list[Headline]: ...


def _iso(value: Any) -> str | None:
    if isinstance(value, int | float):
        return datetime.fromtimestamp(float(value), UTC).isoformat(timespec="seconds")
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.astimezone(UTC).isoformat(timespec="seconds")
    return None


class YahooNews:
    """yfinance `Ticker.news`: the latest headlines (about 10–20), both payload layouts."""

    def headlines(self, symbol: str) -> list[Headline]:
        import yfinance as yf

        try:
            raw = yf.Ticker(symbol).news or []
            if not raw:  # yfinance 1.7: Ticker.news is often empty, the search endpoint is not
                found = yf.Search(symbol, max_results=1, news_count=40).news or []
                upper = symbol.upper()
                raw = [
                    n
                    for n in found
                    if upper in [str(t).upper() for t in n.get("relatedTickers") or [upper]]
                ]
        except Exception as exc:  # yfinance raises bare exceptions on throttling / bad symbols
            raise ProviderError(f"yfinance news: {type(exc).__name__}") from exc
        out = []
        for entry in raw:
            content = entry.get("content") if isinstance(entry.get("content"), dict) else entry
            title = str(content.get("title") or "").strip()
            published = _iso(content.get("pubDate") or content.get("providerPublishTime"))
            if not title or not published:
                continue
            provider = content.get("provider")
            publisher = (
                provider.get("displayName") if isinstance(provider, dict) else None
            ) or str(content.get("publisher") or "")
            link = content.get("canonicalUrl") or content.get("clickThroughUrl")
            url = (link.get("url") if isinstance(link, dict) else None) or str(
                content.get("link") or ""
            )
            if not re.match(r"^https?://", url, re.IGNORECASE):
                url = ""  # never store javascript:, data: or relative links
            out.append(
                Headline(
                    headline_id(url, title),
                    published,
                    title[:300],
                    str(publisher)[:80],
                    url[:500],
                    "yfinance",
                    str(content.get("summary") or "")[:600],
                )
            )
        return out


class FakeNews:
    """Forty deterministic headlines over the 90 days to yesterday, for FAKE* symbols only."""

    GOOD = (
        "{s} beats quarterly earnings estimates",
        "{s} raises full-year guidance",
        "Analysts lift {s} price targets after strong demand",
        "{s} wins multi-year supply contract",
    )
    BAD = (
        "{s} misses revenue expectations",
        "{s} cuts outlook on weaker orders",
        "Regulator opens probe into {s} accounting",
        "{s} recalls products over safety concerns",
    )
    NEUTRAL = (
        "{s} to present at industry conference",
        "{s} schedules annual shareholder meeting",
        "{s} appoints new board member",
    )

    def headlines(self, symbol: str) -> list[Headline]:
        if not symbol.upper().startswith("FAKE"):
            raise SymbolNotFoundError(f"No fake news for {symbol!r}")
        s = symbol.upper()
        rng = random.Random(s)
        end = date.today() - timedelta(days=1)
        out = []
        for n in range(40):
            day = end - timedelta(days=rng.randrange(0, 90))
            pool = rng.choice((self.GOOD, self.BAD, self.NEUTRAL))
            title = rng.choice(pool).format(s=s) + f" ({n + 1})"
            url = f"https://news.example.invalid/{s.lower()}/{n + 1}"
            when = datetime.combine(day, time(13 + n % 8, 0), UTC).isoformat(timespec="seconds")
            out.append(Headline(headline_id(url, title), when, title, "Example Wire", url, "fake"))
        return out


def default_source() -> NewsSource:
    return YahooNews() if network_enabled() else FakeNews()


def _path(lake: PriceLake, symbol: str) -> Path | None:
    """Store file for a plain symbol, or None for anything else (no path tricks)."""
    if not SYMBOL.match(symbol):
        return None
    root = (lake.root / "news").resolve()
    path = (root / f"symbol={symbol.upper()}.json").resolve()
    return path if path.parent == root else None


def read_news(lake: PriceLake, symbol: str) -> dict[str, Any] | None:
    """Stored headlines (newest first) with their sentiment, and the last digest."""
    path = _path(lake, symbol)
    if path is None or not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text("utf-8"))
    return data


def write_news(lake: PriceLake, symbol: str, data: dict[str, Any]) -> None:
    path = _path(lake, symbol)
    if path is None:
        raise SymbolNotFoundError(f"Invalid symbol {symbol[:24]!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data["items"] = sorted(data["items"], key=lambda h: h["published"], reverse=True)[:MAX_ITEMS]
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
    tmp.replace(path)


def ingest_news(lake: PriceLake, symbol: str, source: NewsSource | None = None) -> dict[str, Any]:
    """Fetch the latest headlines and merge them into the store (scores are kept)."""
    if _path(lake, symbol) is None:
        raise SymbolNotFoundError(f"Invalid symbol {symbol[:24]!r}")
    fetched = (source or default_source()).headlines(symbol)
    data = read_news(lake, symbol) or {"symbol": symbol.upper(), "items": [], "digest": None}
    known = {h["id"] for h in data["items"]}
    new = [asdict(h) | {"sentiment": None} for h in fetched if h.id not in known]
    data["items"].extend(new)
    data["as_of"] = datetime.now(UTC).isoformat(timespec="seconds")
    write_news(lake, symbol, data)
    return {"fetched": len(fetched), "new": len(new), "stored": len(data["items"])}


def news_version(lake: PriceLake, symbol: str) -> str:
    path = _path(lake, symbol)
    if path is None or not path.exists():
        return "none"
    return f"{os.path.getmtime(path):.3f}"
