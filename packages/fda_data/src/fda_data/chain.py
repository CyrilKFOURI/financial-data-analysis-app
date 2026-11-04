"""Ordered provider fallback chain (TRD §5.1-5.3)."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from fda_data.providers.base import (
    ACTION_COLUMNS,
    InstrumentProfile,
    PriceProvider,
    ProviderError,
    RateLimitedError,
    SeriesInfo,
    SeriesProvider,
)
from fda_data.ratelimit import TokenBucket


class NoProviderError(ProviderError):
    """No provider in the chain can serve the symbol."""

    def __init__(self, message: str, rate_limited: bool = False) -> None:
        super().__init__(message)
        # Every attempt failed on throttling only: the job is worth retrying later.
        self.rate_limited = rate_limited


@dataclass(frozen=True, slots=True)
class Sourced[T]:
    value: T
    source: str


class ProviderChain:
    def __init__(
        self,
        providers: Sequence[PriceProvider],
        limiters: dict[str, TokenBucket] | None = None,
    ) -> None:
        self._providers = list(providers)
        self._limiters = limiters or {}

    def _attempt[T](self, symbol: str, call: Callable[[PriceProvider], T]) -> Sourced[T]:
        errors: list[str] = []
        throttled = 0
        for provider in self._providers:
            if not provider.supports(symbol):
                continue
            limiter = self._limiters.get(provider.name)
            if limiter is not None and not limiter.acquire():
                errors.append(f"{provider.name}: local rate limit")
                throttled += 1
                continue
            try:
                return Sourced(call(provider), provider.name)
            except RateLimitedError as exc:
                errors.append(f"{provider.name}: rate limited ({exc})")
                throttled += 1
            except ProviderError as exc:
                errors.append(f"{provider.name}: {exc}")
        if not errors:
            raise NoProviderError(f"No provider can serve {symbol}")
        raise NoProviderError(
            f"All providers failed for {symbol}: " + "; ".join(errors),
            rate_limited=throttled == len(errors),
        )

    def can_serve(self, symbol: str) -> bool:
        return any(provider.supports(symbol) for provider in self._providers)

    def profile(self, symbol: str) -> Sourced[InstrumentProfile]:
        return self._attempt(symbol, lambda provider: provider.profile(symbol))

    def daily_bars(self, symbol: str, start: date, end: date) -> Sourced[pd.DataFrame]:
        return self._attempt(symbol, lambda provider: provider.daily_bars(symbol, start, end))

    def actions(self, symbol: str, start: date, end: date, source: str) -> pd.DataFrame:
        """Corporate actions from the provider that served the bars (empty if it has none)."""
        for provider in self._providers:
            fetch = getattr(provider, "actions", None)
            if provider.name == source and callable(fetch):
                frame: pd.DataFrame = fetch(symbol, start, end)
                return frame
        return pd.DataFrame(columns=ACTION_COLUMNS)

    def search(self, query: str) -> list[Sourced[InstrumentProfile]]:
        results: list[Sourced[InstrumentProfile]] = []
        seen: set[str] = set()
        for provider in self._providers:
            limiter = self._limiters.get(provider.name)
            if limiter is not None and not limiter.try_acquire():
                continue  # search is interactive: skip a throttled provider rather than wait
            try:
                found = provider.search(query)
            except ProviderError:
                continue
            for profile in found:
                if profile.symbol not in seen:
                    seen.add(profile.symbol)
                    results.append(Sourced(profile, provider.name))
        return results


class SeriesChain:
    """Rates/FX/macro series: the first provider that knows the code serves it."""

    def __init__(
        self,
        providers: Sequence[SeriesProvider],
        limiters: dict[str, TokenBucket] | None = None,
    ) -> None:
        self._providers = list(providers)
        self._limiters = limiters or {}

    def _provider(self, code: str) -> SeriesProvider:
        for provider in self._providers:
            if provider.supports_series(code):
                return provider
        raise NoProviderError(f"No provider knows series {code}")

    def can_serve(self, code: str) -> bool:
        return any(provider.supports_series(code) for provider in self._providers)

    def info(self, code: str) -> Sourced[SeriesInfo]:
        provider = self._provider(code)
        return Sourced(provider.series_info(code), provider.name)

    def series(self, code: str, start: date, end: date) -> Sourced[pd.DataFrame]:
        provider = self._provider(code)
        limiter = self._limiters.get(provider.name)
        if limiter is not None and not limiter.acquire():
            raise NoProviderError(f"{provider.name}: local rate limit", rate_limited=True)
        try:
            return Sourced(provider.series(code, start, end), provider.name)
        except RateLimitedError as exc:
            raise NoProviderError(f"{provider.name}: {exc}", rate_limited=True) from exc
