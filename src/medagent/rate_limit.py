from __future__ import annotations

import time
from typing import Callable, Dict


class TokenBucket:
    def __init__(self, rate_per_sec: float, burst: int, now: Callable[[], float] | None = None):
        self.rate = float(rate_per_sec)
        self.capacity = int(max(1, burst))
        self.tokens = float(self.capacity)
        self.last = (now or time.perf_counter)()
        self.now = now or time.perf_counter

    def allow(self) -> bool:
        t = self.now()
        elapsed = t - self.last
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last = t
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    def remaining_tokens(self) -> float:
        return max(0.0, self.tokens)

    def next_refill_seconds(self) -> float:
        if self.tokens >= 1.0:
            return 0.0
        need = 1.0 - self.tokens
        if self.rate <= 0:
            return float("inf")
        return max(0.0, need / self.rate)


class RateLimiter:
    def __init__(self, rate_per_sec: float, burst: int, now: Callable[[], float] | None = None):
        self.rate = rate_per_sec
        self.burst = burst
        self.now = now or time.perf_counter
        self.buckets: Dict[str, TokenBucket] = {}

    def check(self, key: str) -> bool:
        b = self.buckets.get(key)
        if b is None:
            b = self.buckets[key] = TokenBucket(self.rate, self.burst, now=self.now)
        return b.allow()

    def state(self, key: str) -> tuple[float, float]:
        b = self.buckets.get(key)
        if b is None:
            b = self.buckets[key] = TokenBucket(self.rate, self.burst, now=self.now)
        return b.remaining_tokens(), b.next_refill_seconds()


class Quota:
    """Simple in-memory daily quota per key. Not persisted across restarts."""

    def __init__(self, daily_limit: int, now: Callable[[], float] | None = None):
        self.daily_limit = int(max(0, daily_limit))
        self.now = now or time.time
        self.usage: Dict[str, tuple[int, int]] = {}
        # usage[key] = (yyyymmdd, count)

    def _today(self) -> int:
        t = time.localtime(self.now())
        return t.tm_year * 10000 + t.tm_mon * 100 + t.tm_mday

    def consume(self, key: str) -> bool:
        if self.daily_limit == 0:
            return True
        today = self._today()
        d, c = self.usage.get(key, (today, 0))
        if d != today:
            d, c = today, 0
        if c + 1 > self.daily_limit:
            self.usage[key] = (d, c)
            return False
        self.usage[key] = (d, c + 1)
        return True

    def remaining(self, key: str) -> int:
        if self.daily_limit == 0:
            return 2**31 - 1
        today = self._today()
        d, c = self.usage.get(key, (today, 0))
        if d != today:
            c = 0
        return max(0, self.daily_limit - c)

    def reset_seconds(self) -> int:
        t = time.localtime(self.now())
        # seconds to midnight
        secs_today = t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec
        return 86400 - secs_today
