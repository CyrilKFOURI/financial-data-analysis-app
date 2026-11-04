"""yfinance adapter: daily bars, corporate actions, profile and search (TRD §5.2).

Yahoo's `Close` is already split-adjusted (not dividend-adjusted); `Adj Close` is split- and
dividend-adjusted. Both are stored as `close` / `adj_close`; performance always uses `adj_close`.
"""

import re
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any, Protocol

import pandas as pd

from fda_data.providers.base import (
    ACTION_COLUMNS,
    InstrumentProfile,
    ProviderError,
    RateLimitedError,
    SymbolNotFoundError,
)

# quoteType → (asset_class, default asset_subclass); slice 012 refines the subclass.
QUOTE_TYPES: dict[str, tuple[str, str]] = {
    "EQUITY": ("equity", "common_stock"),
    "ETF": ("etf", "equity_etf"),
    "MUTUALFUND": ("fund", "mutual_fund"),
    "MONEYMARKET": ("fund", "money_market"),
    "INDEX": ("index", "equity_index"),
    "CRYPTOCURRENCY": ("crypto", "coin"),
    "CURRENCY": ("fx", "spot_pair"),
    "FUTURE": ("commodity", "continuous_future"),
}

EXTRA_KEYS = (
    "quoteType",
    "typeDisp",
    "exchange",
    "fullExchangeName",
    "category",
    "fundFamily",
    "legalType",
    "isin",
    "firstTradeDateEpochUtc",
)

# Yahoo returns full country names; the schema stores ISO 3166-1 alpha-2.
COUNTRIES = {
    "United States": "US", "Canada": "CA", "Mexico": "MX", "Brazil": "BR",
    "United Kingdom": "GB", "Ireland": "IE", "France": "FR", "Germany": "DE",
    "Netherlands": "NL", "Belgium": "BE", "Luxembourg": "LU", "Switzerland": "CH",
    "Italy": "IT", "Spain": "ES", "Portugal": "PT", "Austria": "AT", "Denmark": "DK",
    "Sweden": "SE", "Norway": "NO", "Finland": "FI", "Poland": "PL", "Greece": "GR",
    "Israel": "IL", "Japan": "JP", "China": "CN", "Hong Kong": "HK", "Taiwan": "TW",
    "South Korea": "KR", "Singapore": "SG", "India": "IN", "Australia": "AU",
    "New Zealand": "NZ", "South Africa": "ZA", "Bermuda": "BM", "Cayman Islands": "KY",
    "Jersey": "JE", "Guernsey": "GG", "Uruguay": "UY", "Argentina": "AR", "Chile": "CL",
}  # fmt: skip

BAR_COLUMNS = ["session_date", "open", "high", "low", "close", "adj_close", "volume"]
_SYMBOL_RE = re.compile(r"^[A-Z0-9.^=\-]{1,20}$")


class Ticker(Protocol):
    def history(self, **kwargs: Any) -> pd.DataFrame: ...

    def get_info(self) -> dict[str, Any]: ...


TickerFactory = Callable[[str], Ticker]
SearchFn = Callable[[str], list[dict[str, Any]]]


def _yf_ticker(symbol: str) -> Ticker:
    import yfinance as yf

    ticker: Ticker = yf.Ticker(symbol)
    return ticker


def _yf_search(query: str) -> list[dict[str, Any]]:
    import yfinance as yf

    quotes: list[dict[str, Any]] = yf.Search(query, max_results=8, news_count=0).quotes
    return quotes


def _is_rate_limit(exc: Exception) -> bool:
    return type(exc).__name__ == "YFRateLimitError" or "Too Many Requests" in str(exc)


def _session_dates(index: pd.Index) -> list[date]:
    """Exchange-local session dates (the index is tz-aware at the exchange's midnight)."""
    return [pd.Timestamp(value).date() for value in index]


class YahooProvider:
    name = "yfinance"

    def __init__(
        self, ticker_factory: TickerFactory = _yf_ticker, search_fn: SearchFn = _yf_search
    ) -> None:
        self._ticker = ticker_factory
        self._search = search_fn
        self._currency: dict[str, str] = {}
        # Bars and actions come from the same history call: keep the last one to avoid a refetch.
        self._last_history: tuple[tuple[str, date, date], pd.DataFrame] | None = None

    def supports(self, symbol: str) -> bool:
        upper = symbol.upper()
        return bool(_SYMBOL_RE.match(upper)) and not upper.startswith("FAKE")

    def _call[T](self, symbol: str, fn: Callable[[], T]) -> T:
        try:
            return fn()
        except ProviderError:
            raise
        except Exception as exc:
            if _is_rate_limit(exc):
                raise RateLimitedError(f"yfinance: rate limited on {symbol}") from exc
            raise ProviderError(f"yfinance: {type(exc).__name__} on {symbol}") from exc

    def profile(self, symbol: str) -> InstrumentProfile:
        symbol = symbol.upper()
        info = self._call(symbol, lambda: self._ticker(symbol).get_info())
        quote_type = str(info.get("quoteType") or "").upper()
        if quote_type not in QUOTE_TYPES or not info.get("currency"):
            raise SymbolNotFoundError(f"yfinance: no usable profile for {symbol}")
        asset_class, asset_subclass = QUOTE_TYPES[quote_type]
        currency = str(info["currency"]).upper()
        self._currency[symbol] = currency
        return InstrumentProfile(
            symbol=symbol,
            name=str(info.get("longName") or info.get("shortName") or symbol),
            asset_class=asset_class,
            asset_subclass=asset_subclass,
            currency=currency,
            country=COUNTRIES.get(str(info.get("country"))),
            sector=info.get("sector"),
            industry=info.get("industry"),
            timezone=info.get("exchangeTimezoneName"),
            provider_symbols={self.name: symbol},
            extras={key: info[key] for key in EXTRA_KEYS if info.get(key) is not None},
        )

    def _history(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        key = (symbol, start, end)
        if self._last_history is not None and self._last_history[0] == key:
            return self._last_history[1]
        # yfinance treats `end` as exclusive.
        history = self._call(
            symbol,
            lambda: self._ticker(symbol).history(
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                interval="1d",
                auto_adjust=False,
                actions=True,
                raise_errors=True,
            ),
        )
        self._last_history = (key, history)
        return history

    def daily_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        raw = self._history(symbol.upper(), start, end)
        if raw.empty:
            return pd.DataFrame(columns=BAR_COLUMNS)
        return pd.DataFrame(
            {
                "session_date": _session_dates(raw.index),
                "open": raw["Open"].to_numpy(dtype=float),
                "high": raw["High"].to_numpy(dtype=float),
                "low": raw["Low"].to_numpy(dtype=float),
                "close": raw["Close"].to_numpy(dtype=float),
                "adj_close": raw["Adj Close"].to_numpy(dtype=float),
                "volume": raw["Volume"].to_numpy(dtype=float),
            }
        )

    def actions(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        symbol = symbol.upper()
        raw = self._history(symbol, start, end)
        currency = self._currency.get(symbol)
        dates = _session_dates(raw.index)
        rows: list[dict[str, Any]] = []
        if "Dividends" in raw:
            for ex_date, amount in zip(dates, raw["Dividends"].to_numpy(dtype=float), strict=True):
                if amount > 0:
                    rows.append(
                        {"ex_date": ex_date, "action": "dividend", "ratio": None,
                         "amount": amount, "currency": currency}
                    )  # fmt: skip
        if "Stock Splits" in raw:
            for ex_date, ratio in zip(
                dates, raw["Stock Splits"].to_numpy(dtype=float), strict=True
            ):
                if ratio > 0:
                    rows.append(
                        {"ex_date": ex_date, "action": "split", "ratio": ratio,
                         "amount": None, "currency": None}
                    )  # fmt: skip
        frame = pd.DataFrame(rows, columns=ACTION_COLUMNS)
        return frame.sort_values("ex_date", ignore_index=True)

    def search(self, query: str) -> list[InstrumentProfile]:
        quotes = self._call(query, lambda: self._search(query))
        results: list[InstrumentProfile] = []
        for quote in quotes:
            quote_type = str(quote.get("quoteType") or "").upper()
            symbol = str(quote.get("symbol") or "").upper()
            if quote_type not in QUOTE_TYPES or not self.supports(symbol):
                continue
            asset_class, asset_subclass = QUOTE_TYPES[quote_type]
            results.append(
                InstrumentProfile(
                    symbol=symbol,
                    name=str(quote.get("longname") or quote.get("shortname") or symbol),
                    asset_class=asset_class,
                    asset_subclass=asset_subclass,
                    currency="",  # unknown until the full profile is fetched
                    sector=quote.get("sector"),
                    industry=quote.get("industry"),
                    provider_symbols={self.name: symbol},
                    extras={"exchange": quote.get("exchDisp") or quote.get("exchange")},
                )
            )
        return results
