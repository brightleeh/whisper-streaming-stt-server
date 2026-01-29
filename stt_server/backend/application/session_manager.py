"""Session management helpers."""

from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import grpc

from gen.stt.python.v1 import stt_pb2
from stt_server.backend.application.model_registry import ModelRegistry
from stt_server.backend.component.vad_gate import release_vad_slot, reserve_vad_slot
from stt_server.backend.utils.profile_resolver import (
    profile_enum_from_name,
    profile_name_from_enum,
    resolve_decode_profile,
    resolve_language_code,
    resolve_task,
    task_enum_from_name,
)
from stt_server.config.default.model import DEFAULT_MODEL_ID
from stt_server.config.languages import SupportedLanguages
from stt_server.errors import ErrorCode, abort_with_error, format_error
from stt_server.utils.logger import LOGGER, clear_session_id, set_session_id


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
    model_id: str = DEFAULT_MODEL_ID
    vad_reserved: bool = False


def _noop_session_hook(_: "SessionInfo") -> None:
    return None


@dataclass(frozen=True)
class SessionRegistryHooks:
    on_create: Callable[["SessionInfo"], None] = _noop_session_hook
    on_remove: Callable[["SessionInfo"], None] = _noop_session_hook


class SessionRegistry:
    """Thread-safe registry for active STT sessions."""

    def __init__(self, hooks: SessionRegistryHooks | None = None) -> None:
        self._hooks = hooks or SessionRegistryHooks()
        self._lock = threading.Lock()
        self._sessions: Dict[str, SessionInfo] = {}

    def create_session(self, session_id: str, info: SessionInfo) -> None:
        """Register a session, raising ValueError if it already exists."""
        with self._lock:
            if session_id in self._sessions:
                raise ValueError("session already exists")
            self._sessions[session_id] = info
        self._hooks.on_create(info)

    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        with self._lock:
            return self._sessions.get(session_id)

    def remove_session(self, session_id: str) -> Optional[SessionInfo]:
        with self._lock:
            info = self._sessions.pop(session_id, None)
        if info:
            self._hooks.on_remove(info)
        return info

    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)


@dataclass
class SessionState:
    """Represents resolved session context for a streaming RPC."""

    session_id: str
    session_info: SessionInfo
    decode_options: Dict[str, Any]


class SessionFacade:
    """Centralizes session lookup, validation, and lifecycle helpers."""

    def __init__(self, session_registry: SessionRegistry) -> None:
        self._session_registry = session_registry

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
            LOGGER.error(format_error(ErrorCode.SESSION_ID_MISSING))
            abort_with_error(context, ErrorCode.SESSION_ID_MISSING)
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
            LOGGER.error(format_error(ErrorCode.SESSION_TOKEN_INVALID))
            abort_with_error(context, ErrorCode.SESSION_TOKEN_INVALID)

    def remove_session(self, state: Optional[SessionState], reason: str = "") -> None:
        if not state:
            return
        self._session_registry.remove_session(state.session_id)
        if reason:
            LOGGER.info("Removed session %s (%s)", state.session_id, reason)

    def remove_session_by_id(self, session_id: str | bytes | None) -> None:
        normalized = self._normalize_session_id(session_id)
        if normalized:
            self._session_registry.remove_session(normalized)

    def _build_state(
        self, session_id: str, context: grpc.ServicerContext
    ) -> SessionState:
        session_info = self._session_registry.get_session(session_id)
        if not session_info:
            LOGGER.error(format_error(ErrorCode.SESSION_ID_MISSING))
            abort_with_error(context, ErrorCode.SESSION_ID_MISSING)
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


class CreateSessionHandler:
    """Handles CreateSession requests for the STT backend servicer."""

    def __init__(
        self,
        session_registry: SessionRegistry,
        model_registry: ModelRegistry,
        decode_profiles: Dict[str, Dict[str, Any]],
        default_decode_profile: str,
        default_language: str,
        language_fix: bool,
        default_task: str,
        supported_languages: SupportedLanguages,
        default_vad_silence: float,
        default_vad_threshold: float,
    ) -> None:
        self._session_registry = session_registry
        self._model_registry = model_registry
        self._decode_profiles = decode_profiles
        self._default_decode_profile = default_decode_profile
        self._default_language = default_language
        self._language_fix = language_fix
        self._default_task = default_task
        self._supported_languages = supported_languages
        self._default_vad_silence = max(0.0, default_vad_silence)
        self._default_vad_threshold = max(0.0, default_vad_threshold)

    def handle(
        self, request: stt_pb2.SessionRequest, context: grpc.ServicerContext
    ) -> stt_pb2.SessionResponse:
        if not request.session_id:
            LOGGER.error(format_error(ErrorCode.SESSION_ID_REQUIRED))
            abort_with_error(context, ErrorCode.SESSION_ID_REQUIRED)

        session_id = request.session_id
        set_session_id(session_id)
        vad_mode = (
            request.vad_mode
            if request.vad_mode in (stt_pb2.VAD_CONTINUE, stt_pb2.VAD_AUTO_END)
            else stt_pb2.VAD_CONTINUE
        )
        token_required = bool(request.require_token)
        token = secrets.token_hex(16) if token_required else ""
        api_key = (
            request.attributes.get("api_key") or request.attributes.get("api-key") or ""
        )

        requested_profile = profile_name_from_enum(request.decode_profile)
        if not requested_profile:
            requested_profile = request.attributes.get(
                "decode_profiles"
            ) or request.attributes.get("decode_profile")
        profile_name, profile_options = resolve_decode_profile(
            requested_profile,
            self._decode_profiles,
            self._default_decode_profile,
        )
        language_code = resolve_language_code(
            request.language_code,
            self._default_language,
            self._language_fix,
            self._supported_languages,
        )
        session_task = resolve_task(request.task, self._default_task)

        model_id = (
            request.attributes.get("model_id")
            or request.attributes.get("model")
            or self._model_registry.get_next_model_id()
            or DEFAULT_MODEL_ID
        )

        options = profile_options.copy()
        if session_task:
            options["task"] = session_task
        if language_code:
            options["language"] = language_code

        vad_silence = self._resolve_vad_silence(request.vad_silence, context)
        if request.HasField("vad_threshold_override"):
            vad_threshold = self._resolve_vad_threshold(
                request.vad_threshold_override, context, allow_default=False
            )
        else:
            vad_threshold = self._resolve_vad_threshold(request.vad_threshold, context)
        vad_reserved = False
        if vad_threshold > 0:
            if not reserve_vad_slot():
                LOGGER.error("VAD pool exhausted; rejecting session_id=%s", session_id)
                abort_with_error(context, ErrorCode.VAD_POOL_EXHAUSTED)
            vad_reserved = True
        session_info = SessionInfo(
            attributes=dict(request.attributes),
            vad_mode=vad_mode,
            vad_silence=vad_silence,
            vad_threshold=vad_threshold,
            token=token,
            token_required=token_required,
            api_key=api_key,
            decode_profile=profile_name,
            decode_options=options,
            language_code=language_code,
            task=session_task,
            model_id=model_id,
            vad_reserved=vad_reserved,
        )
        try:
            try:
                self._session_registry.create_session(session_id, session_info)
            except ValueError:
                if vad_reserved:
                    release_vad_slot()
                LOGGER.error(format_error(ErrorCode.SESSION_ID_ALREADY_ACTIVE))
                abort_with_error(context, ErrorCode.SESSION_ID_ALREADY_ACTIVE)

            response_attributes = dict(request.attributes)
            response_attributes["decode_profile"] = profile_name
            if language_code:
                response_attributes["language_code"] = language_code

            LOGGER.info(
                "Created session_id=%s vad_mode=%s token_required=%s decode_profile=%s language=%s task=%s vad_silence=%.3f vad_threshold=%.4f attributes=%s model_id=%s",
                session_id,
                "AUTO_END" if vad_mode == stt_pb2.VAD_AUTO_END else "CONTINUE",
                token_required,
                profile_name,
                language_code or "auto",
                session_task,
                vad_silence,
                vad_threshold,
                dict(request.attributes),
                model_id,
            )

            return stt_pb2.SessionResponse(
                attributes=response_attributes,
                vad_mode=vad_mode,
                vad_silence=vad_silence,
                vad_threshold=vad_threshold,
                token=token,
                token_required=token_required,
                language_code=language_code,
                task=task_enum_from_name(session_task),
                decode_profile=profile_enum_from_name(profile_name),
            )
        finally:
            clear_session_id()

    def _resolve_vad_silence(
        self, value: float, context: grpc.ServicerContext
    ) -> float:
        if value <= 0:
            return self._default_vad_silence
        return value

    def _resolve_vad_threshold(
        self,
        value: float,
        context: grpc.ServicerContext,
        allow_default: bool = True,
    ) -> float:
        if value < 0:
            LOGGER.error(format_error(ErrorCode.VAD_THRESHOLD_NEGATIVE))
            abort_with_error(context, ErrorCode.VAD_THRESHOLD_NEGATIVE)
        if allow_default and value == 0:
            return self._default_vad_threshold
        return value


__all__ = [
    "SessionInfo",
    "SessionRegistry",
    "SessionState",
    "SessionFacade",
    "CreateSessionHandler",
]
