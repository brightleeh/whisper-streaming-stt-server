"""Session registration and lifecycle management."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

from stt_server.backend.core.metrics import Metrics


@dataclass
class SessionInfo:
    attributes: Dict[str, str]
    vad_mode: int
    vad_silence: float
    vad_threshold: float
    token: str
    token_required: bool
    api_key: str
    decode_profile: str
    decode_options: Dict[str, Any]
    language_code: str
    task: str


class SessionManager:
    """Thread-safe registry for active STT sessions."""

    def __init__(self, metrics: Metrics) -> None:
        self._metrics = metrics
        self._lock = threading.Lock()
        self._sessions: Dict[str, SessionInfo] = {}

    def create_session(self, session_id: str, info: SessionInfo) -> None:
        """Register a session, raising ValueError if it already exists."""
        with self._lock:
            if session_id in self._sessions:
                raise ValueError("session already exists")
            self._sessions[session_id] = info
        if info.api_key:
            self._metrics.increase_active_sessions(info.api_key)

    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        with self._lock:
            return self._sessions.get(session_id)

    def remove_session(self, session_id: str) -> Optional[SessionInfo]:
        with self._lock:
            info = self._sessions.pop(session_id, None)
        if info and info.api_key:
            self._metrics.decrease_active_sessions(info.api_key)
        return info

    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)
