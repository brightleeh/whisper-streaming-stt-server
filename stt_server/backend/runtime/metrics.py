import threading
from collections import defaultdict
from typing import Any, Dict

import grpc


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_sessions = 0
        self._api_key_sessions: Dict[str, int] = defaultdict(int)
        self._expose_api_key_metrics = False
        self._decode_count = 0
        self._decode_total = 0.0
        self._decode_max = 0.0
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
        self._error_counts: Dict[str, int] = defaultdict(int)

    def increase_active_sessions(self, api_key: str) -> None:
        with self._lock:
            self._active_sessions += 1
            if api_key:
                self._api_key_sessions[api_key] += 1

    def decrease_active_sessions(self, api_key: str) -> None:
        with self._lock:
            if self._active_sessions > 0:
                self._active_sessions -= 1
            if api_key and self._api_key_sessions.get(api_key):
                self._api_key_sessions[api_key] -= 1
                if self._api_key_sessions[api_key] <= 0:
                    self._api_key_sessions.pop(api_key, None)

    def record_decode(
        self,
        inference_sec: float,
        real_time_factor: float,
        queue_wait_sec: float | None = None,
        buffer_wait_sec: float | None = None,
        response_emit_sec: float | None = None,
    ) -> None:
        with self._lock:
            self._decode_count += 1
            self._decode_total += inference_sec
            self._decode_max = max(self._decode_max, inference_sec)
            if buffer_wait_sec is not None and buffer_wait_sec >= 0:
                self._decode_buffer_wait_count += 1
                self._decode_buffer_wait_total += buffer_wait_sec
                self._decode_buffer_wait_max = max(
                    self._decode_buffer_wait_max, buffer_wait_sec
                )
            if queue_wait_sec is not None and queue_wait_sec >= 0:
                self._decode_queue_wait_count += 1
                self._decode_queue_wait_total += queue_wait_sec
                self._decode_queue_wait_max = max(
                    self._decode_queue_wait_max, queue_wait_sec
                )
            if response_emit_sec is not None and response_emit_sec >= 0:
                self._decode_response_emit_count += 1
                self._decode_response_emit_total += response_emit_sec
                self._decode_response_emit_max = max(
                    self._decode_response_emit_max, response_emit_sec
                )
            if real_time_factor >= 0:
                self._rtf_count += 1
                self._rtf_total += real_time_factor
                self._rtf_max = max(self._rtf_max, real_time_factor)

    def record_decode_cancelled(self, count: int) -> None:
        with self._lock:
            self._decode_cancelled += max(count, 0)

    def record_decode_orphaned(self, count: int) -> None:
        with self._lock:
            self._decode_orphaned += max(count, 0)

    def record_vad_trigger(self) -> None:
        with self._lock:
            self._vad_triggers += 1

    def increase_active_vad_utterances(self) -> None:
        with self._lock:
            self._active_vad_utterances += 1

    def decrease_active_vad_utterances(self) -> None:
        with self._lock:
            if self._active_vad_utterances > 0:
                self._active_vad_utterances -= 1

    def active_vad_utterances(self) -> int:
        with self._lock:
            return self._active_vad_utterances

    def record_error(self, status_code: grpc.StatusCode) -> None:
        with self._lock:
            self._error_counts[status_code.name] += 1

    def set_expose_api_key_metrics(self, enabled: bool) -> None:
        with self._lock:
            self._expose_api_key_metrics = bool(enabled)

    def render(self) -> Dict[str, Any]:
        with self._lock:
            payload = {
                "active_sessions": self._active_sessions,
                "decode_latency_total": self._decode_total,
                "decode_latency_count": self._decode_count,
                "decode_latency_max": self._decode_max,
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
                "rtf_total": self._rtf_total,
                "rtf_count": self._rtf_count,
                "rtf_max": self._rtf_max,
                "vad_triggers_total": self._vad_triggers,
                "active_vad_utterances": self._active_vad_utterances,
                "error_counts": dict(self._error_counts),
            }
            if self._expose_api_key_metrics:
                payload["active_sessions_by_api"] = dict(self._api_key_sessions)
            return payload

    def snapshot(self) -> Dict[str, float]:
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
