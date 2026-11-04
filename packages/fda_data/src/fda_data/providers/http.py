"""Shared HTTP plumbing for REST providers: one client each, polite headers, typed errors."""

import httpx2 as httpx

from fda_data.providers.base import ProviderError, RateLimitedError, SymbolNotFoundError

USER_AGENT = "fda-workbench/0.1 (personal research tool)"
TIMEOUT_S = 20.0


def make_client(
    transport: httpx.BaseTransport | None = None, timeout: float = TIMEOUT_S
) -> httpx.Client:
    """`transport` lets tests replay recorded responses (httpx MockTransport)."""
    return httpx.Client(
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
        transport=transport,
    )


def get(client: httpx.Client, provider: str, url: str, **params: str) -> httpx.Response:
    try:
        response = client.get(url, params=params)
    except httpx.HTTPError as exc:
        raise ProviderError(f"{provider}: {type(exc).__name__}") from exc
    if response.status_code == 429:
        raise RateLimitedError(f"{provider}: HTTP 429")
    if response.status_code == 404:
        raise SymbolNotFoundError(f"{provider}: not found")
    if response.status_code >= 400:
        raise ProviderError(f"{provider}: HTTP {response.status_code}")
    return response
