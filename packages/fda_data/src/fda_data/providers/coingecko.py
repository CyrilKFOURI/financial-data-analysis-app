"""CoinGecko public API: crypto metadata (rank, supply, categories). Best effort, no key."""

from typing import Any

import httpx2 as httpx

from fda_data.providers.base import ProviderError, SymbolNotFoundError
from fda_data.providers.http import get, make_client
from fda_data.ratelimit import TokenBucket

BASE_URL = "https://api.coingecko.com/api/v3"
QUOTE_SUFFIXES = ("-USD", "-EUR", "-USDT", "-GBP", "-JPY")


def base_asset(symbol: str) -> str:
    """BTC-USD → BTC (Yahoo crypto pairs are `<base>-<quote>`)."""
    upper = symbol.upper()
    for suffix in QUOTE_SUFFIXES:
        if upper.endswith(suffix):
            return upper.removesuffix(suffix)
    return upper


class CoinGeckoProvider:
    name = "coingecko"

    def __init__(
        self, client: httpx.Client | None = None, limiter: TokenBucket | None = None
    ) -> None:
        self._client = client or make_client()
        # Public tier: ~10-30 calls/min depending on load → stay well below.
        self._limiter = limiter or TokenBucket(rate_per_second=0.2, burst=3)

    def _get(self, path: str, **params: str) -> Any:
        if not self._limiter.acquire(timeout=15):
            raise ProviderError("coingecko: local rate limit")
        return get(self._client, self.name, f"{BASE_URL}{path}", **params).json()

    def coin_id(self, symbol: str) -> str:
        """Highest-ranked coin whose ticker matches exactly (BTC → bitcoin, not wrapped BTC)."""
        base = base_asset(symbol)
        coins = self._get("/search", query=base).get("coins", [])
        matches = [c for c in coins if str(c.get("symbol", "")).upper() == base]
        if not matches:
            raise SymbolNotFoundError(f"coingecko: no coin with ticker {base}")
        ranked = sorted(matches, key=lambda c: c.get("market_cap_rank") or 1_000_000)
        return str(ranked[0]["id"])

    def crypto_metadata(self, symbol: str) -> dict[str, Any]:
        coin_id = self.coin_id(symbol)
        coin = self._get(
            f"/coins/{coin_id}",
            localization="false",
            tickers="false",
            market_data="true",
            community_data="false",
            developer_data="false",
            sparkline="false",
        )
        market = coin.get("market_data") or {}
        return {
            "coingecko_id": coin_id,
            "market_cap_rank": coin.get("market_cap_rank"),
            "categories": [c for c in coin.get("categories") or [] if c],
            "hashing_algorithm": coin.get("hashing_algorithm"),
            "genesis_date": coin.get("genesis_date"),
            "circulating_supply": market.get("circulating_supply"),
            "total_supply": market.get("total_supply"),
            "max_supply": market.get("max_supply"),
        }
