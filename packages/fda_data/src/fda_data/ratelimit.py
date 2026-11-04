"""Token bucket per provider (TRD §5.3).

In-process limiter; a Redis-backed variant shared by all workers arrives with the real
providers (slice 011), behind the same `acquire()` interface.
"""

import threading
import time


class TokenBucket:
    def __init__(self, rate_per_second: float, burst: int) -> None:
        if rate_per_second <= 0 or burst < 1:
            raise ValueError("rate must be > 0 and burst >= 1")
        self._rate = rate_per_second
        self._capacity = float(burst)
        self._tokens = float(burst)
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self, now: float) -> None:
        self._tokens = min(self._capacity, self._tokens + (now - self._updated) * self._rate)
        self._updated = now

    def try_acquire(self) -> bool:
        with self._lock:
            self._refill(time.monotonic())
            if self._tokens >= 1:
                self._tokens -= 1
                return True
            return False

    def acquire(self, timeout: float = 30.0) -> bool:
        """Block until a token is available or `timeout` seconds elapse."""
        deadline = time.monotonic() + timeout
        while True:
            if self.try_acquire():
                return True
            with self._lock:
                wait = (1 - self._tokens) / self._rate
            if time.monotonic() + wait > deadline:
                return False
            time.sleep(wait)
