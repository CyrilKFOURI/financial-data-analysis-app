"""Deterministic synthetic provider for tests and demos.

Serves only symbols starting with ``FAKE`` so synthetic data can never masquerade as a real
instrument such as AAPL.
"""

import hashlib
import math
from datetime import date

import numpy as np
import pandas as pd

from fda_data.providers.base import InstrumentProfile, SeriesInfo, SymbolNotFoundError

ORIGIN = date(2000, 1, 3)


class FakeProvider:
    name = "fake"

    def __init__(self, mu: float = 0.07, sigma: float = 0.22) -> None:
        self._mu = mu
        self._sigma = sigma

    def supports(self, symbol: str) -> bool:
        return symbol.upper().startswith("FAKE")

    def _require(self, symbol: str) -> None:
        if not self.supports(symbol):
            raise SymbolNotFoundError(f"{self.name} serves FAKE* symbols only, not {symbol}")

    def profile(self, symbol: str) -> InstrumentProfile:
        self._require(symbol)
        return InstrumentProfile(
            symbol=symbol.upper(),
            name=f"Synthetic test asset {symbol.upper()}",
            asset_class="equity",
            asset_subclass="synthetic",
            currency="USD",
            exchange_mic="XFAK",
            country="US",
            timezone="America/New_York",
            provider_symbols={self.name: symbol.upper()},
        )

    def search(self, query: str) -> list[InstrumentProfile]:
        return [self.profile(query)] if self.supports(query) else []

    def daily_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Business-day GBM path anchored at 2000-01-03, so any window is reproducible."""
        self._require(symbol)
        seed = int(hashlib.sha256(symbol.upper().encode()).hexdigest()[:8], 16)
        sessions = pd.bdate_range(ORIGIN, end)
        rng = np.random.default_rng(seed)
        daily_sigma = self._sigma / math.sqrt(252)
        log_returns = rng.normal(self._mu / 252 - daily_sigma**2 / 2, daily_sigma, len(sessions))
        close = 100.0 * np.exp(np.cumsum(log_returns))
        intraday = np.abs(rng.normal(0, daily_sigma / 2, len(sessions)))
        frame = pd.DataFrame(
            {
                "session_date": sessions.date,
                "open": close * (1 + rng.normal(0, daily_sigma / 4, len(sessions))),
                "close": close,
                "adj_close": close,
                "volume": rng.integers(1_000_000, 5_000_000, len(sessions)).astype(float),
            }
        )
        frame["high"] = np.maximum(frame["open"], frame["close"]) * (1 + intraday)
        frame["low"] = np.minimum(frame["open"], frame["close"]) * (1 - intraday)
        mask = (frame["session_date"] >= start) & (frame["session_date"] <= end)
        return frame.loc[mask].reset_index(drop=True)


class FakeSeriesProvider:
    """Synthetic rate series (`FAKE*` codes) for offline tests of the series pipeline."""

    name = "fake"

    def supports_series(self, code: str) -> bool:
        return code.upper().startswith("FAKE")

    def series_info(self, code: str) -> SeriesInfo:
        if not self.supports_series(code):
            raise SymbolNotFoundError(f"fake serves FAKE* series only, not {code}")
        return SeriesInfo(code.upper(), f"Synthetic rate {code.upper()}", "percent", "B", code)

    def series(self, code: str, start: date, end: date) -> pd.DataFrame:
        self.series_info(code)
        days = pd.bdate_range(start, end)
        level = 2.0 + np.sin(np.arange(len(days)) / 60.0)
        return pd.DataFrame({"date": days.date, "value": np.round(level, 4)})
