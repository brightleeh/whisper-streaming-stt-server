import threading
from collections import defaultdict
from typing import Dict

import grpc


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_sessions = 0
        self._api_key_sessions: Dict[str, int] = defaultdict(int)
        self._decode_count = 0
        self._decode_total = 0.0
        self._decode_max = 0.0
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

    def record_decode(self, latency_sec: float, rtf: float) -> None:
        with self._lock:
            self._decode_count += 1
            self._decode_total += latency_sec
            self._decode_max = max(self._decode_max, latency_sec)
            if rtf >= 0:
                self._rtf_count += 1
                self._rtf_total += rtf
                self._rtf_max = max(self._rtf_max, rtf)

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

    def render(self) -> str:
        with self._lock:
            lines = [
                f"active_sessions {self._active_sessions}",
                f"decode_latency_total {self._decode_total:.6f}",
                f"decode_latency_count {self._decode_count}",
                f"decode_latency_max {self._decode_max:.6f}",
                f"rtf_total {self._rtf_total:.6f}",
                f"rtf_count {self._rtf_count}",
                f"rtf_max {self._rtf_max:.6f}",
                f"vad_triggers_total {self._vad_triggers}",
                f"active_vad_utterances {self._active_vad_utterances}",
            ]
            for api_key, count in self._api_key_sessions.items():
                lines.append(f'active_sessions_by_api{{api_key="{api_key}"}} {count}')
            for code, count in self._error_counts.items():
                lines.append(f'error_count{{code="{code}"}} {count}')
            return "\n".join(lines) + "\n"

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            decode_avg = (
                (self._decode_total / self._decode_count) if self._decode_count else 0.0
            )
            rtf_avg = (self._rtf_total / self._rtf_count) if self._rtf_count else 0.0
            return {
                "active_sessions": self._active_sessions,
                "decode_latency_avg": decode_avg,
                "decode_latency_max": self._decode_max,
                "rtf_avg": rtf_avg,
                "rtf_max": self._rtf_max,
                "vad_triggers": self._vad_triggers,
                "active_vad_utterances": self._active_vad_utterances,
            }
