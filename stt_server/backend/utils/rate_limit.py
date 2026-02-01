"""Simple token-bucket rate limiter with per-key tracking."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict


@dataclass
class _BucketState:
    tokens: float
    last_refill: float
    last_seen: float


class KeyedRateLimiter:
    """Token-bucket limiter keyed by arbitrary identifiers."""

    def __init__(
        self,
        rate_per_sec: float,
        burst: float | None = None,
        time_fn: Callable[[], float] | None = None,
        *,
        max_keys: int | None = 10000,
        ttl_sec: float | None = 300.0,
        prune_interval_sec: float = 60.0,
        prune_every: int = 1000,
    ) -> None:
        self._rate_per_sec = max(0.0, float(rate_per_sec))
        self._burst = (
            max(0.0, float(burst))
            if burst is not None and burst > 0
            else self._rate_per_sec
        )
        self._time_fn = time_fn or time.monotonic
        self._states: Dict[str, _BucketState] = {}
        self._lock = threading.Lock()
        self._max_keys = max_keys if max_keys and max_keys > 0 else None
        self._ttl_sec = ttl_sec if ttl_sec and ttl_sec > 0 else None
        self._prune_interval_sec = max(0.0, float(prune_interval_sec))
        self._prune_every = max(0, int(prune_every))
        self._last_prune = self._time_fn()
        self._call_count = 0

    def allow(self, key: str, amount: float = 1.0) -> bool:
        """Consume tokens for key; returns True if allowed."""
        if self._rate_per_sec <= 0 or self._burst <= 0:
            return True
        if amount <= 0:
            return True
        now = self._time_fn()
        with self._lock:
            self._call_count += 1
            self._prune_if_needed(now)
            state = self._states.get(key)
            if state is None:
                state = _BucketState(tokens=self._burst, last_refill=now, last_seen=now)
                self._states[key] = state
                if self._max_keys is not None and len(self._states) > self._max_keys:
                    self._prune(now)
            elapsed = max(0.0, now - state.last_refill)
            state.tokens = min(self._burst, state.tokens + elapsed * self._rate_per_sec)
            state.last_refill = now
            state.last_seen = now
            if state.tokens < amount:
                return False
            state.tokens -= amount
            return True

    def _prune_if_needed(self, now: float) -> None:
        if self._prune_every > 0 and self._call_count % self._prune_every == 0:
            self._prune(now)
            return
        if self._prune_interval_sec <= 0:
            self._prune(now)
            return
        if now - self._last_prune < self._prune_interval_sec:
            return
        self._prune(now)

    def _prune(self, now: float) -> None:
        self._last_prune = now
        if self._ttl_sec is not None:
            cutoff = now - self._ttl_sec
            stale_keys = [
                key for key, state in self._states.items() if state.last_seen < cutoff
            ]
            for key in stale_keys:
                self._states.pop(key, None)
        if self._max_keys is not None and len(self._states) > self._max_keys:
            ordered = sorted(self._states.items(), key=lambda item: item[1].last_seen)
            excess = len(self._states) - self._max_keys
            for key, _state in ordered[:excess]:
                self._states.pop(key, None)
