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


class KeyedRateLimiter:
    """Token-bucket limiter keyed by arbitrary identifiers."""

    def __init__(
        self,
        rate_per_sec: float,
        burst: float | None = None,
        time_fn: Callable[[], float] | None = None,
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

    def allow(self, key: str, amount: float = 1.0) -> bool:
        """Consume tokens for key; returns True if allowed."""
        if self._rate_per_sec <= 0 or self._burst <= 0:
            return True
        if amount <= 0:
            return True
        now = self._time_fn()
        with self._lock:
            state = self._states.get(key)
            if state is None:
                state = _BucketState(tokens=self._burst, last_refill=now)
                self._states[key] = state
            elapsed = max(0.0, now - state.last_refill)
            state.tokens = min(
                self._burst, state.tokens + elapsed * self._rate_per_sec
            )
            state.last_refill = now
            if state.tokens < amount:
                return False
            state.tokens -= amount
            return True
