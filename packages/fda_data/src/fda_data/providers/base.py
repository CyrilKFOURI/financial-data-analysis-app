"""Provider protocols and value objects (TRD §5.1)."""

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

import pandas as pd


class ProviderError(Exception):
    """The provider cannot serve this request; the chain may try the next provider."""


class RateLimitedError(ProviderError):
    """Throttled by the provider (HTTP 429 or equivalent)."""


class SymbolNotFoundError(ProviderError):
    pass


@dataclass(frozen=True, slots=True)
class InstrumentProfile:
    symbol: str
    name: str
    asset_class: str
    currency: str
    asset_subclass: str | None = None
    exchange_mic: str | None = None
    country: str | None = None
    sector: str | None = None
    industry: str | None = None
    timezone: str | None = None
    provider_symbols: dict[str, str] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


class PriceProvider(Protocol):
    name: str

    def supports(self, symbol: str) -> bool: ...

    def profile(self, symbol: str) -> InstrumentProfile: ...

    def daily_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Columns: session_date, open, high, low, close, adj_close, volume."""
        ...

    def search(self, query: str) -> list[InstrumentProfile]: ...


ACTION_COLUMNS = ["ex_date", "action", "ratio", "amount", "currency"]


class ActionsProvider(Protocol):
    name: str

    def actions(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Columns: ex_date, action ('dividend' | 'split'), ratio, amount, currency."""
        ...


@dataclass(frozen=True, slots=True)
class SeriesInfo:
    code: str
    title: str
    unit: str
    frequency: str
    provider_key: str


class SeriesProvider(Protocol):
    """Rates, FX reference rates and macro series: levels by date, not tradable prices."""

    name: str

    def supports_series(self, code: str) -> bool: ...

    def series_info(self, code: str) -> SeriesInfo: ...

    def series(self, code: str, start: date, end: date) -> pd.DataFrame:
        """Columns: date, value (observations only — gaps are never filled)."""
        ...
