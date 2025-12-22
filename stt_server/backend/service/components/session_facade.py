from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import grpc

from gen.stt.python.v1 import stt_pb2
from stt_server.backend.core.session_manager import SessionInfo, SessionManager
from stt_server.utils.logger import LOGGER


@dataclass
class SessionState:
    """Represents resolved session context for a streaming RPC."""

    session_id: str
    session_info: SessionInfo
    decode_options: Dict[str, Any]


class SessionFacade:
    """Centralizes session lookup, validation, and lifecycle helpers.

    The STT service treats sessions as single-use resources: each CreateSession
    must be followed by exactly one StreamingRecognize, and the session is
    removed as soon as streaming finishes (or aborts). This facade enforces the
    policy and ensures every session identifier is normalized to `str`.
    """

    def __init__(self, session_manager: SessionManager) -> None:
        self._session_manager = session_manager

    def resolve_from_metadata(
        self,
        metadata: Dict[str, str | bytes],
        context: grpc.ServicerContext,
    ) -> Optional[SessionState]:
        session_id = self._normalize_session_id(
            metadata.get("session-id") or metadata.get("session_id")
        )
        if not session_id:
            return None
        return self._build_state(session_id, context)

    def ensure_session_from_chunk(
        self,
        current_state: Optional[SessionState],
        chunk: stt_pb2.AudioChunk,
        context: grpc.ServicerContext,
    ) -> SessionState:
        session_id = self._normalize_session_id(chunk.session_id) or (
            current_state.session_id if current_state else None
        )
        if not session_id:
            LOGGER.error("ERR1004 Unknown or missing session_id")
            context.abort(
                grpc.StatusCode.UNAUTHENTICATED,
                "ERR1004 Unknown or missing session_id",
            )
        if current_state and session_id == current_state.session_id:
            return current_state
        return self._build_state(session_id, context)

    def validate_token(
        self,
        state: Optional[SessionState],
        chunk: stt_pb2.AudioChunk,
        context: grpc.ServicerContext,
    ) -> None:
        if not state:
            return
        session_info = state.session_info
        if session_info.token_required and chunk.session_token != session_info.token:
            self.remove_session(state, reason="invalid_token")
            LOGGER.error("ERR1005 Invalid session token")
            context.abort(
                grpc.StatusCode.PERMISSION_DENIED, "ERR1005 Invalid session token"
            )

    def remove_session(self, state: Optional[SessionState], reason: str = "") -> None:
        if not state:
            return
        self._session_manager.remove_session(state.session_id)
        if reason:
            LOGGER.info("Removed session %s (%s)", state.session_id, reason)

    def remove_session_by_id(self, session_id: str | bytes | None) -> None:
        normalized = self._normalize_session_id(session_id)
        if normalized:
            self._session_manager.remove_session(normalized)

    def _build_state(
        self, session_id: str, context: grpc.ServicerContext
    ) -> SessionState:
        session_info = self._session_manager.get_session(session_id)
        if not session_info:
            LOGGER.error("ERR1004 Unknown or missing session_id")
            context.abort(
                grpc.StatusCode.UNAUTHENTICATED,
                "ERR1004 Unknown or missing session_id",
            )
        return SessionState(
            session_id=session_id,
            session_info=session_info,
            decode_options=dict(session_info.decode_options),
        )

    def _normalize_session_id(self, value: str | bytes | None) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8", errors="ignore")
            except Exception:
                return None
        normalized = str(value).strip()
        return normalized or None
