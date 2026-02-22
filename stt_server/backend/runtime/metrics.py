"""Runtime metrics for streaming STT sessions."""

import bisect
import hashlib
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict

import grpc


@dataclass(frozen=True)
class HistogramSnapshot:
    """Serializable histogram snapshot for metrics export."""

    bounds: tuple[float, ...]
    cumulative_counts: tuple[int, ...]
    count: int
    sum: float


class Histogram:
    """Thread-safe(under external lock) histogram with fixed buckets."""

    def __init__(self, bounds: tuple[float, ...]):
        normalized = []
        for value in bounds:
            value = float(value)
            if value < 0:
                continue
            if normalized and value <= normalized[-1]:
                continue
            normalized.append(value)
        self._bounds = tuple(normalized)
        self._bucket_counts = [0] * (len(self._bounds) + 1)  # includes +Inf bucket
        self._count = 0
        self._sum = 0.0

    def observe(self, value: float) -> None:
        """Observe a non-negative sample."""
        if value < 0:
            return
        index = bisect.bisect_left(self._bounds, value)
        self._bucket_counts[index] += 1
        self._count += 1
        self._sum += value

    def snapshot(self) -> HistogramSnapshot:
        """Return cumulative counts for Prometheus exposition."""
        cumulative = []
        running = 0
        for count in self._bucket_counts:
            running += count
            cumulative.append(running)
        return HistogramSnapshot(
            bounds=self._bounds,
            cumulative_counts=tuple(cumulative),
            count=self._count,
            sum=self._sum,
        )


class Metrics:
    """Thread-safe counters and aggregations for server metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_sessions = 0
        self._api_key_sessions: Dict[str, int] = defaultdict(int)
        self._expose_api_key_metrics = False
        self._buffer_bytes_total = 0
        self._stream_buffer_bytes: Dict[str, int] = {}
        self._partial_drop_count = 0
        self._rate_limit_blocks: Dict[str, int] = defaultdict(int)
        self._rate_limit_blocks_by_key: Dict[str, int] = defaultdict(int)
        self._decode_count = 0
        self._decode_total = 0.0
        self._decode_max = 0.0
        self._decode_pending = 0
        self._decode_buffer_wait_count = 0
        self._decode_buffer_wait_total = 0.0
        self._decode_buffer_wait_max = 0.0
        self._decode_queue_wait_count = 0
        self._decode_queue_wait_total = 0.0
        self._decode_queue_wait_max = 0.0
        self._decode_response_emit_count = 0
        self._decode_response_emit_total = 0.0
        self._decode_response_emit_max = 0.0
        self._decode_cancelled = 0
        self._decode_orphaned = 0
        self._rtf_count = 0
        self._rtf_total = 0.0
        self._rtf_max = 0.0
        self._vad_triggers = 0
        self._active_vad_utterances = 0
        self._vad_model_deepcopy_fallback_total = 0
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._decode_latency_hist = Histogram(
            (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.5, 5.0, 10.0)
        )
        self._decode_buffer_wait_hist = Histogram(
            (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
        )
        self._decode_queue_wait_hist = Histogram(
            (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
        )
        self._decode_response_emit_hist = Histogram(
            (0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
        )

    def _hash_key(self, value: str) -> str:
        if not value:
            return ""
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        return digest[:16]

    def increase_active_sessions(self, api_key: str) -> None:
        """Increment the active session counters."""
        with self._lock:
            self._active_sessions += 1
            if api_key:
                self._api_key_sessions[api_key] += 1

    def decrease_active_sessions(self, api_key: str) -> None:
        """Decrement the active session counters."""
        with self._lock:
            if self._active_sessions > 0:
                self._active_sessions -= 1
            if api_key and self._api_key_sessions.get(api_key):
                self._api_key_sessions[api_key] -= 1
                if self._api_key_sessions[api_key] <= 0:
                    self._api_key_sessions.pop(api_key, None)

    def set_buffer_total(self, total_bytes: int) -> None:
        """Set the global buffered audio byte total."""
        with self._lock:
            self._buffer_bytes_total = max(0, int(total_bytes))

    def set_decode_pending(self, pending: int) -> None:
        """Set the current number of pending decode tasks."""
        with self._lock:
            self._decode_pending = max(0, int(pending))

    def set_stream_buffer_bytes(self, session_id: str, buffer_bytes: int) -> None:
        """Set per-stream buffer bytes (tracked by hashed session id)."""
        if not session_id:
            return
        key = self._hash_key(f"session:{session_id}")
        with self._lock:
            self._stream_buffer_bytes[key] = max(0, int(buffer_bytes))

    def clear_stream_buffer(self, session_id: str) -> None:
        """Remove per-stream buffer bytes entry for a session."""
        if not session_id:
            return
        key = self._hash_key(f"session:{session_id}")
        with self._lock:
            self._stream_buffer_bytes.pop(key, None)

    def record_decode(
        self,
        inference_sec: float,
        real_time_factor: float,
        queue_wait_sec: float | None = None,
        buffer_wait_sec: float | None = None,
        response_emit_sec: float | None = None,
    ) -> None:
        """Record decode timing metrics for a single decode."""
        with self._lock:
            self._decode_count += 1
            self._decode_total += inference_sec
            self._decode_max = max(self._decode_max, inference_sec)
            self._decode_latency_hist.observe(inference_sec)
            if buffer_wait_sec is not None and buffer_wait_sec >= 0:
                self._decode_buffer_wait_count += 1
                self._decode_buffer_wait_total += buffer_wait_sec
                self._decode_buffer_wait_max = max(
                    self._decode_buffer_wait_max, buffer_wait_sec
                )
                self._decode_buffer_wait_hist.observe(buffer_wait_sec)
            if queue_wait_sec is not None and queue_wait_sec >= 0:
                self._decode_queue_wait_count += 1
                self._decode_queue_wait_total += queue_wait_sec
                self._decode_queue_wait_max = max(
                    self._decode_queue_wait_max, queue_wait_sec
                )
                self._decode_queue_wait_hist.observe(queue_wait_sec)
            if response_emit_sec is not None and response_emit_sec >= 0:
                self._decode_response_emit_count += 1
                self._decode_response_emit_total += response_emit_sec
                self._decode_response_emit_max = max(
                    self._decode_response_emit_max, response_emit_sec
                )
                self._decode_response_emit_hist.observe(response_emit_sec)
            if real_time_factor >= 0:
                self._rtf_count += 1
                self._rtf_total += real_time_factor
                self._rtf_max = max(self._rtf_max, real_time_factor)

    def record_decode_cancelled(self, count: int) -> None:
        """Record cancelled decode count."""
        with self._lock:
            self._decode_cancelled += max(count, 0)

    def record_decode_orphaned(self, count: int) -> None:
        """Record orphaned decode count."""
        with self._lock:
            self._decode_orphaned += max(count, 0)

    def record_partial_drop(self, count: int) -> None:
        """Record dropped partial decode count."""
        with self._lock:
            self._partial_drop_count += max(count, 0)

    def record_vad_trigger(self) -> None:
        """Record a completed VAD trigger."""
        with self._lock:
            self._vad_triggers += 1

    def increase_active_vad_utterances(self) -> None:
        """Increment the number of active VAD utterances."""
        with self._lock:
            self._active_vad_utterances += 1

    def decrease_active_vad_utterances(self) -> None:
        """Decrement the number of active VAD utterances."""
        with self._lock:
            if self._active_vad_utterances > 0:
                self._active_vad_utterances -= 1

    def record_vad_model_deepcopy_fallback(self, count: int = 1) -> None:
        """Record fallback model reloads caused by deepcopy failures."""
        with self._lock:
            self._vad_model_deepcopy_fallback_total += max(0, int(count))

    def active_vad_utterances(self) -> int:
        """Return the current active VAD utterance count."""
        with self._lock:
            return self._active_vad_utterances

    def record_error(self, status_code: grpc.StatusCode) -> None:
        """Record an error code occurrence."""
        with self._lock:
            self._error_counts[status_code.name] += 1

    def record_rate_limit_block(self, scope: str, key: str | None = None) -> None:
        """Record a rate limit block for a scope and optional key."""
        if not scope:
            scope = "unknown"
        with self._lock:
            self._rate_limit_blocks[scope] += 1
            if key:
                hashed = self._hash_key(key)
                if hashed:
                    self._rate_limit_blocks_by_key[f"{scope}_{hashed}"] += 1

    def set_expose_api_key_metrics(self, enabled: bool) -> None:
        """Enable or disable per-api-key metrics exposure."""
        with self._lock:
            self._expose_api_key_metrics = bool(enabled)

    def render(self) -> Dict[str, Any]:
        """Render metrics as a serializable payload."""
        with self._lock:
            payload = {
                "active_sessions": self._active_sessions,
                "buffer_bytes_total": self._buffer_bytes_total,
                "decode_latency_total": self._decode_total,
                "decode_latency_count": self._decode_count,
                "decode_latency_max": self._decode_max,
                "decode_pending": self._decode_pending,
                "decode_buffer_wait_total": self._decode_buffer_wait_total,
                "decode_buffer_wait_count": self._decode_buffer_wait_count,
                "decode_buffer_wait_max": self._decode_buffer_wait_max,
                "decode_queue_wait_total": self._decode_queue_wait_total,
                "decode_queue_wait_count": self._decode_queue_wait_count,
                "decode_queue_wait_max": self._decode_queue_wait_max,
                "decode_response_emit_total": self._decode_response_emit_total,
                "decode_response_emit_count": self._decode_response_emit_count,
                "decode_response_emit_max": self._decode_response_emit_max,
                "decode_cancelled": self._decode_cancelled,
                "decode_orphaned": self._decode_orphaned,
                "partial_drop_count": self._partial_drop_count,
                "rtf_total": self._rtf_total,
                "rtf_count": self._rtf_count,
                "rtf_max": self._rtf_max,
                "vad_triggers_total": self._vad_triggers,
                "active_vad_utterances": self._active_vad_utterances,
                "vad_model_deepcopy_fallback_total": self._vad_model_deepcopy_fallback_total,
                "error_counts": dict(self._error_counts),
                "rate_limit_blocks": dict(self._rate_limit_blocks),
            }
            if self._expose_api_key_metrics:
                payload["active_sessions_by_api"] = dict(self._api_key_sessions)
            if self._stream_buffer_bytes:
                payload["stream_buffer_bytes"] = dict(self._stream_buffer_bytes)
            if self._rate_limit_blocks_by_key:
                payload["rate_limit_blocks_by_key"] = dict(
                    self._rate_limit_blocks_by_key
                )
            payload["histograms"] = self._render_histograms()
            return payload

    def _render_histograms(self) -> Dict[str, Dict[str, Any]]:
        """Render histogram values as JSON-friendly maps."""
        return {
            "decode_latency_sec": self._histogram_payload(self._decode_latency_hist),
            "decode_buffer_wait_sec": self._histogram_payload(
                self._decode_buffer_wait_hist
            ),
            "decode_queue_wait_sec": self._histogram_payload(
                self._decode_queue_wait_hist
            ),
            "decode_response_emit_sec": self._histogram_payload(
                self._decode_response_emit_hist
            ),
        }

    @staticmethod
    def _histogram_payload(histogram: Histogram) -> Dict[str, Any]:
        snap = histogram.snapshot()
        buckets: Dict[str, int] = {}
        for idx, bound in enumerate(snap.bounds):
            buckets[str(bound)] = snap.cumulative_counts[idx]
        buckets["+Inf"] = snap.cumulative_counts[-1]
        return {"buckets": buckets, "count": snap.count, "sum": snap.sum}

    def snapshot(self) -> Dict[str, float]:
        """Return a snapshot with averages and maxima for key metrics."""
        with self._lock:
            decode_avg = (
                (self._decode_total / self._decode_count) if self._decode_count else 0.0
            )
            rtf_avg = (self._rtf_total / self._rtf_count) if self._rtf_count else 0.0
            buffer_wait_avg = (
                (self._decode_buffer_wait_total / self._decode_buffer_wait_count)
                if self._decode_buffer_wait_count
                else 0.0
            )
            queue_wait_avg = (
                (self._decode_queue_wait_total / self._decode_queue_wait_count)
                if self._decode_queue_wait_count
                else 0.0
            )
            response_emit_avg = (
                (self._decode_response_emit_total / self._decode_response_emit_count)
                if self._decode_response_emit_count
                else 0.0
            )
            return {
                "active_sessions": self._active_sessions,
                "decode_latency_avg": decode_avg,
                "decode_latency_max": self._decode_max,
                "decode_pending": float(self._decode_pending),
                "decode_buffer_wait_avg": buffer_wait_avg,
                "decode_buffer_wait_max": self._decode_buffer_wait_max,
                "decode_queue_wait_avg": queue_wait_avg,
                "decode_queue_wait_max": self._decode_queue_wait_max,
                "decode_response_emit_avg": response_emit_avg,
                "decode_response_emit_max": self._decode_response_emit_max,
                "decode_cancelled": float(self._decode_cancelled),
                "decode_orphaned": float(self._decode_orphaned),
                "rtf_avg": rtf_avg,
                "rtf_max": self._rtf_max,
                "vad_triggers": self._vad_triggers,
                "active_vad_utterances": self._active_vad_utterances,
            }
