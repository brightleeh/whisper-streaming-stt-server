from __future__ import annotations

from stt_server.backend.utils.rate_limit import KeyedRateLimiter


def test_keyed_rate_limiter_prunes_by_ttl() -> None:
    now = 0.0

    def time_fn() -> float:
        return now

    limiter = KeyedRateLimiter(
        rate_per_sec=1.0,
        burst=1.0,
        time_fn=time_fn,
        ttl_sec=1.0,
        prune_interval_sec=0.0,
        max_keys=10,
    )

    assert limiter.allow("old")
    now = 2.0
    assert limiter.allow("new")

    assert "old" not in limiter._states
    assert "new" in limiter._states


def test_keyed_rate_limiter_prunes_to_max_keys() -> None:
    now = 0.0

    def time_fn() -> float:
        return now

    limiter = KeyedRateLimiter(
        rate_per_sec=1.0,
        burst=1.0,
        time_fn=time_fn,
        ttl_sec=None,
        prune_interval_sec=0.0,
        max_keys=2,
    )

    assert limiter.allow("first")
    now = 1.0
    assert limiter.allow("second")
    now = 2.0
    assert limiter.allow("third")

    assert len(limiter._states) == 2
    assert "first" not in limiter._states
    assert "second" in limiter._states
    assert "third" in limiter._states
