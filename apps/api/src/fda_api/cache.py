"""Content-addressed result cache (TRD §7.2): an in-process LRU (L1) in front of Redis (L2)
when Redis is reachable, the LRU alone otherwise. L1 keeps warm reads off the network: a
Redis round-trip through Docker Desktop port forwarding costs ~20 ms on Windows."""

import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Any, Protocol
from urllib.parse import urlsplit

from redis.asyncio import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 7 * 24 * 3600


class ResultCache(Protocol):
    backend: str

    async def get(self, key: str) -> bytes | None: ...
    async def set(self, key: str, value: bytes, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None: ...
    async def close(self) -> None: ...


class MemoryCache:
    """Bounded LRU (entries and bytes): the L1 tier, or the whole cache without Redis."""

    backend = "memory"

    def __init__(self, max_entries: int = 4_096, max_bytes: int = 128 * 1024 * 1024) -> None:
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._bytes = 0
        self._items: OrderedDict[str, tuple[float, bytes]] = OrderedDict()

    async def get(self, key: str) -> bytes | None:
        item = self._items.get(key)
        if item is None:
            return None
        if item[0] < time.monotonic():
            self._drop(key)
            return None
        self._items.move_to_end(key)
        return item[1]

    def _drop(self, key: str) -> None:
        _, value = self._items.pop(key)
        self._bytes -= len(value)

    async def set(self, key: str, value: bytes, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        if key in self._items:
            self._drop(key)
        self._items[key] = (time.monotonic() + ttl_seconds, value)
        self._bytes += len(value)
        while self._items and (
            len(self._items) > self._max_entries or self._bytes > self._max_bytes
        ):
            self._drop(next(iter(self._items)))

    async def close(self) -> None:
        self._items.clear()
        self._bytes = 0


class RedisCache:
    backend = "redis"
    _prefix = "fda:result:"

    def __init__(self, client: Redis) -> None:
        self._client = client

    async def get(self, key: str) -> bytes | None:
        try:
            value: bytes | None = await self._client.get(self._prefix + key)
            return value
        except RedisError:
            logger.warning("Redis read failed; treating as cache miss", exc_info=True)
            return None

    async def set(self, key: str, value: bytes, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        try:
            await self._client.set(self._prefix + key, value, ex=ttl_seconds)
        except RedisError:
            logger.warning("Redis write failed; result not cached", exc_info=True)

    async def close(self) -> None:
        await self._client.aclose()


class TieredCache:
    """L1 in-process LRU, L2 Redis shared with the workers and other API processes."""

    backend = "redis+memory"

    def __init__(self, l1: MemoryCache, l2: RedisCache) -> None:
        self._l1, self._l2 = l1, l2

    async def get(self, key: str) -> bytes | None:
        value = await self._l1.get(key)
        if value is not None:
            return value
        value = await self._l2.get(key)
        if value is not None:
            await self._l1.set(key, value)
        return value

    async def set(self, key: str, value: bytes, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        await self._l1.set(key, value, ttl_seconds)
        await self._l2.set(key, value, ttl_seconds)

    async def close(self) -> None:
        await self._l1.close()
        await self._l2.close()


async def connect_cache(redis_url: str | None) -> ResultCache:
    if not redis_url:
        return MemoryCache()
    client = Redis.from_url(redis_url, socket_connect_timeout=0.5, socket_timeout=1.0)
    try:
        await client.ping()
    except (RedisError, OSError):
        location = urlsplit(redis_url)
        logger.warning(
            "Redis unreachable at %s:%s; using in-memory result cache",
            location.hostname,
            location.port,
        )
        await client.aclose()
        return MemoryCache()
    return TieredCache(MemoryCache(), RedisCache(client))


def cache_key(
    *,
    analysis_id: str,
    analysis_version: str,
    params: dict[str, Any],
    subject: dict[str, Any],
    context: dict[str, Any],
    data_version: str,
) -> str:
    """sha256 over a canonical JSON document: key order and whitespace never matter."""
    document = json.dumps(
        {
            "analysis": analysis_id,
            "version": analysis_version,
            "params": params,
            "subject": subject,
            "context": context,
            "data_version": data_version,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(document.encode()).hexdigest()
