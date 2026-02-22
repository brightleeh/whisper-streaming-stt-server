"""Session management helpers."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import grpc

from gen.stt.python.v1 import stt_pb2
from stt_server.backend.application.model_registry import ModelRegistry
from stt_server.backend.component.vad_gate import (
    VADModelPool,
    default_vad_model_pool,
)
from stt_server.backend.utils.profile_resolver import (
    invalid_decode_options,
    profile_enum_from_name,
    profile_name_from_enum,
    resolve_decode_profile,
    resolve_language_code,
    resolve_task,
    task_enum_from_name,
)
from stt_server.backend.utils.rate_limit import KeyedRateLimiter
from stt_server.config.default.model import DEFAULT_MODEL_ID
from stt_server.config.languages import SupportedLanguages
from stt_server.errors import ErrorCode, abort_with_error, format_error
from stt_server.utils.logger import LOGGER, clear_session_id, set_session_id

if TYPE_CHECKING:
    from stt_server.backend.runtime.metrics import Metrics

_AUTH_PROFILE_NONE = "none"
_AUTH_PROFILE_API_KEY = "api_key"
_AUTH_PROFILE_SIGNED_TOKEN = "signed_token"
_AUTH_PROFILE_ALIASES = {
    "none": _AUTH_PROFILE_NONE,
    "off": _AUTH_PROFILE_NONE,
    "false": _AUTH_PROFILE_NONE,
    "0": _AUTH_PROFILE_NONE,
    "api_key": _AUTH_PROFILE_API_KEY,
    "api-key": _AUTH_PROFILE_API_KEY,
    "apikey": _AUTH_PROFILE_API_KEY,
    "signed_token": _AUTH_PROFILE_SIGNED_TOKEN,
    "signed": _AUTH_PROFILE_SIGNED_TOKEN,
    "signature": _AUTH_PROFILE_SIGNED_TOKEN,
    "hmac": _AUTH_PROFILE_SIGNED_TOKEN,
}
_AUTH_SIG_KEYS = ("auth_sig", "auth_signature", "signature")
_AUTH_TS_KEYS = ("auth_ts", "auth_timestamp", "timestamp")
_AUTH_METADATA_SIG_KEYS = (
    "authorization",
    "x-stt-auth",
    "x-auth-sig",
    "x-auth-signature",
)
_AUTH_METADATA_TS_KEYS = (
    "x-stt-auth-ts",
    "x-auth-ts",
    "x-auth-timestamp",
)
_AUTH_ATTRIBUTE_KEYS = set(_AUTH_SIG_KEYS + _AUTH_TS_KEYS)


@dataclass
class SessionInfo:
    """Session attributes and resolved settings for a client session."""

    attributes: Dict[str, str]
    vad_mode: int
    vad_silence: float
    vad_threshold: float
    token: str
    token_required: bool
    client_ip: str
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
    """Callbacks invoked on session create/remove."""

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
        """Return session info if the session is active."""
        with self._lock:
            return self._sessions.get(session_id)

    def remove_session(self, session_id: str) -> Optional[SessionInfo]:
        """Remove a session and return its info if it existed."""
        with self._lock:
            info = self._sessions.pop(session_id, None)
        if info:
            self._hooks.on_remove(info)
        return info

    def active_count(self) -> int:
        """Return the current number of active sessions."""
        with self._lock:
            return len(self._sessions)

    def active_count_by_ip(self, client_ip: str) -> int:
        """Return active session count for the given client IP."""
        if not client_ip:
            return 0
        with self._lock:
            return sum(
                1 for info in self._sessions.values() if info.client_ip == client_ip
            )

    def active_count_by_api_key(self, api_key: str) -> int:
        """Return active session count for the given API key."""
        if not api_key:
            return 0
        with self._lock:
            return sum(1 for info in self._sessions.values() if info.api_key == api_key)


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
        """Resolve session state from request metadata."""
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
        """Resolve or validate session state from an audio chunk."""
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
        """Validate the session token in the incoming chunk."""
        if not state:
            return
        session_info = state.session_info
        if session_info.token_required and chunk.session_token != session_info.token:
            self.remove_session(state, reason="invalid_token")
            LOGGER.error(format_error(ErrorCode.SESSION_TOKEN_INVALID))
            abort_with_error(context, ErrorCode.SESSION_TOKEN_INVALID)

    def remove_session(self, state: Optional[SessionState], reason: str = "") -> None:
        """Remove a session and log the reason when provided."""
        if not state:
            return
        self._session_registry.remove_session(state.session_id)
        if reason:
            LOGGER.info("Removed session %s (%s)", state.session_id, reason)

    def remove_session_by_id(self, session_id: str | bytes | None) -> None:
        """Remove a session by raw identifier if it can be normalized."""
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
            except UnicodeDecodeError:
                return None
        normalized = str(value).strip()
        return normalized or None


@dataclass(frozen=True)
class CreateSessionConfig:
    """Configuration for CreateSessionHandler defaults."""

    decode_profiles: Dict[str, Dict[str, Any]]
    default_decode_profile: str
    default_language: str
    language_fix: bool
    default_task: str
    supported_languages: SupportedLanguages
    default_vad_silence: float
    default_vad_threshold: float
    require_api_key: bool = False
    create_session_auth_profile: str = _AUTH_PROFILE_NONE
    create_session_auth_secret: str = ""
    create_session_auth_ttl_sec: float = 0.0
    create_session_rps: float = 0.0
    create_session_burst: float = 0.0
    max_sessions_per_ip: int = 0
    max_sessions_per_api_key: int = 0
    allow_new_sessions: Callable[[], bool] = lambda: True
    allow_overload_sessions: Callable[[], bool] = lambda: True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "default_vad_silence", max(0.0, self.default_vad_silence)
        )
        object.__setattr__(
            self, "default_vad_threshold", max(0.0, self.default_vad_threshold)
        )


class CreateSessionHandler:
    """Handles CreateSession requests for the STT backend servicer."""

    def __init__(
        self,
        session_registry: SessionRegistry,
        model_registry: ModelRegistry,
        config: CreateSessionConfig,
        metrics: "Metrics | None" = None,
        vad_model_pool: VADModelPool | None = None,
    ) -> None:
        self._session_registry = session_registry
        self._model_registry = model_registry
        self._config = config
        self._metrics = metrics
        self._vad_model_pool = vad_model_pool or default_vad_model_pool()
        self._create_session_limiter = None
        if config.create_session_rps > 0:
            self._create_session_limiter = KeyedRateLimiter(
                config.create_session_rps,
                config.create_session_burst or None,
            )

    def _normalize_auth_profile(self) -> str:
        raw = (self._config.create_session_auth_profile or "").strip().lower()
        return _AUTH_PROFILE_ALIASES.get(raw, raw)

    @staticmethod
    def _build_metadata(context: grpc.ServicerContext) -> Dict[str, str]:
        metadata: Dict[str, str] = {}
        if context is None:
            return metadata
        for key, value in context.invocation_metadata():
            if not key:
                continue
            norm_key = str(key).lower()
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8", errors="ignore")
                except UnicodeDecodeError:
                    value = ""
            metadata[norm_key] = str(value).strip()
        return metadata

    def _get_metadata_value(
        self, metadata: Dict[str, str], keys: tuple[str, ...]
    ) -> str:
        for key in keys:
            value = metadata.get(key)
            if value:
                return str(value).strip()
        return ""

    def _extract_signature(self, metadata: Dict[str, str]) -> str:
        auth = self._get_metadata_value(metadata, ("authorization",))
        if auth:
            parts = auth.split(None, 1)
            if len(parts) == 2 and parts[0].lower() in {
                "bearer",
                "token",
                "signature",
                "hmac",
            }:
                return parts[1].strip()
            return auth.strip()
        return self._get_metadata_value(metadata, _AUTH_METADATA_SIG_KEYS)

    def _sanitize_attributes(self, attributes: Dict[str, str]) -> Dict[str, str]:
        if not attributes:
            return {}
        sanitized = dict(attributes)
        for key in _AUTH_ATTRIBUTE_KEYS:
            sanitized.pop(key, None)
        return sanitized

    def _validate_signed_token(
        self,
        session_id: str,
        metadata: Dict[str, str],
        context: grpc.ServicerContext,
    ) -> None:
        secret = (self._config.create_session_auth_secret or "").strip()
        if not secret:
            LOGGER.error("CreateSession auth profile requires secret")
            abort_with_error(context, ErrorCode.CREATE_SESSION_AUTH_INVALID)
        ts_raw = self._get_metadata_value(metadata, _AUTH_METADATA_TS_KEYS)
        sig_raw = self._extract_signature(metadata)
        used_legacy_auth_format = False
        if (not ts_raw or not sig_raw) and metadata.get("authorization"):
            raw_auth = metadata.get("authorization", "").strip()
            parts = raw_auth.split(None, 1)
            if len(parts) == 2 and parts[0].lower() in {
                "bearer",
                "token",
                "signature",
                "hmac",
            }:
                raw_auth = parts[1].strip()
            if ":" in raw_auth:
                maybe_ts, maybe_sig = raw_auth.split(":", 1)
                if not ts_raw:
                    ts_raw = maybe_ts.strip()
                if not sig_raw or sig_raw == raw_auth or ":" in sig_raw:
                    sig_raw = maybe_sig.strip()
                used_legacy_auth_format = True

        if used_legacy_auth_format:
            LOGGER.warning(
                "CreateSession auth used legacy authorization format; "
                "prefer 'authorization: Bearer <signature>' + 'x-stt-auth-ts'."
            )
        if not ts_raw or not sig_raw:
            LOGGER.warning("CreateSession auth token missing timestamp/signature")
            abort_with_error(context, ErrorCode.CREATE_SESSION_AUTH_INVALID)
        try:
            timestamp_raw = int(float(ts_raw))
        except (TypeError, ValueError):
            LOGGER.warning("CreateSession auth timestamp invalid: %s", ts_raw)
            abort_with_error(context, ErrorCode.CREATE_SESSION_AUTH_INVALID)
        timestamp_sec = timestamp_raw
        if timestamp_sec > 100_000_000_000:
            # Treat large epoch values as milliseconds for TTL checks.
            timestamp_sec = int(timestamp_sec / 1000)
        ttl = float(self._config.create_session_auth_ttl_sec or 0.0)
        if ttl > 0:
            now = time.time()
            if abs(now - timestamp_sec) > ttl:
                LOGGER.warning(
                    "CreateSession auth token expired (ts=%s)", timestamp_raw
                )
                abort_with_error(context, ErrorCode.CREATE_SESSION_AUTH_INVALID)
        payload = f"{session_id}:{timestamp_raw}".encode("utf-8")
        expected = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, sig_raw):
            LOGGER.warning("CreateSession auth signature mismatch")
            abort_with_error(context, ErrorCode.CREATE_SESSION_AUTH_INVALID)

    def _enforce_create_session_auth(
        self,
        profile: str,
        session_id: str,
        metadata: Dict[str, str],
        context: grpc.ServicerContext,
    ) -> None:
        if profile in ("", _AUTH_PROFILE_NONE):
            return
        if profile == _AUTH_PROFILE_API_KEY:
            return
        if profile == _AUTH_PROFILE_SIGNED_TOKEN:
            self._validate_signed_token(session_id, metadata, context)
            return
        LOGGER.error("Unknown CreateSession auth profile: %s", profile)
        abort_with_error(context, ErrorCode.CREATE_SESSION_AUTH_INVALID)

    def handle(
        self, request: stt_pb2.SessionRequest, context: grpc.ServicerContext
    ) -> stt_pb2.SessionResponse:
        """Validate a CreateSession request and return a session response."""
        if not self._config.allow_new_sessions():
            LOGGER.warning("CreateSession rejected during shutdown")
            abort_with_error(context, ErrorCode.SERVER_SHUTTING_DOWN)
        if not self._config.allow_overload_sessions():
            LOGGER.warning("CreateSession rejected due to overload")
            abort_with_error(context, ErrorCode.CREATE_SESSION_RATE_LIMITED)
        if not request.session_id:
            LOGGER.error(format_error(ErrorCode.SESSION_ID_REQUIRED))
            abort_with_error(context, ErrorCode.SESSION_ID_REQUIRED)

        session_id = request.session_id
        set_session_id(session_id)
        client_ip = _extract_client_ip(context)
        attributes = dict(request.attributes)
        metadata = self._build_metadata(context)
        vad_mode = (
            request.vad_mode
            if request.vad_mode in (stt_pb2.VAD_CONTINUE, stt_pb2.VAD_AUTO_END)
            else stt_pb2.VAD_CONTINUE
        )
        token_required = bool(request.require_token)
        token = secrets.token_hex(16) if token_required else ""
        api_key = (attributes.get("api_key") or attributes.get("api-key") or "").strip()
        api_key_required = (
            attributes.get("api_key_required")
            or attributes.get("api-key-required")
            or ""
        )
        api_key_required = str(api_key_required).lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        auth_profile = self._normalize_auth_profile()
        api_key_required = api_key_required or auth_profile == _AUTH_PROFILE_API_KEY
        if (self._config.require_api_key or api_key_required) and not api_key:
            LOGGER.error(format_error(ErrorCode.API_KEY_MISSING))
            abort_with_error(context, ErrorCode.API_KEY_MISSING)

        self._enforce_create_session_auth(auth_profile, session_id, metadata, context)

        self._enforce_session_limits(session_id, api_key, client_ip, context)

        requested_profile = profile_name_from_enum(request.decode_profile)
        if not requested_profile:
            requested_profile = attributes.get("decode_profiles") or attributes.get(
                "decode_profile"
            )
        profile_name, profile_options = resolve_decode_profile(
            requested_profile,
            self._config.decode_profiles,
            self._config.default_decode_profile,
        )
        language_code = resolve_language_code(
            request.language_code,
            self._config.default_language,
            self._config.language_fix,
            self._config.supported_languages,
        )
        session_task = resolve_task(request.task, self._config.default_task)

        model_id = (
            attributes.get("model_id")
            or attributes.get("model")
            or self._model_registry.get_next_model_id()
            or DEFAULT_MODEL_ID
        )

        options = profile_options.copy()
        if session_task:
            options["task"] = session_task
        if language_code:
            options["language"] = language_code
        invalid_options = invalid_decode_options(options)
        if invalid_options:
            detail = f"invalid decode option(s): {', '.join(sorted(invalid_options))}"
            LOGGER.error(format_error(ErrorCode.DECODE_OPTION_INVALID, detail))
            abort_with_error(context, ErrorCode.DECODE_OPTION_INVALID, detail)

        vad_silence = self._resolve_vad_silence(request.vad_silence)
        if request.HasField("vad_threshold_override"):
            vad_threshold = self._resolve_vad_threshold(
                request.vad_threshold_override, context, allow_default=False
            )
        else:
            vad_threshold = self._resolve_vad_threshold(request.vad_threshold, context)
        vad_reserved = False
        if vad_threshold > 0 and not token_required:
            if not self._vad_model_pool.reserve_slot():
                LOGGER.error("VAD pool exhausted; rejecting session_id=%s", session_id)
                abort_with_error(context, ErrorCode.VAD_POOL_EXHAUSTED)
            vad_reserved = True
        sanitized_attributes = self._sanitize_attributes(attributes)
        session_info = SessionInfo(
            attributes=sanitized_attributes,
            vad_mode=vad_mode,
            vad_silence=vad_silence,
            vad_threshold=vad_threshold,
            token=token,
            token_required=token_required,
            client_ip=client_ip,
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
                    self._vad_model_pool.release_slot()
                LOGGER.error(format_error(ErrorCode.SESSION_ID_ALREADY_ACTIVE))
                abort_with_error(context, ErrorCode.SESSION_ID_ALREADY_ACTIVE)

            response_attributes = dict(sanitized_attributes)
            response_attributes["decode_profile"] = profile_name
            if language_code:
                response_attributes["language_code"] = language_code

            LOGGER.info(
                "Created session_id=%s vad_mode=%s token_required=%s "
                "decode_profile=%s language=%s task=%s vad_silence=%.3f "
                "vad_threshold=%.4f attributes=%s model_id=%s",
                session_id,
                "AUTO_END" if vad_mode == stt_pb2.VAD_AUTO_END else "CONTINUE",
                token_required,
                profile_name,
                language_code or "auto",
                session_task,
                vad_silence,
                vad_threshold,
                dict(sanitized_attributes),
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

    def _resolve_vad_silence(self, value: float) -> float:
        if value <= 0:
            return self._config.default_vad_silence
        return value

    def _enforce_session_limits(
        self,
        session_id: str,
        api_key: str,
        client_ip: str,
        context: grpc.ServicerContext,
    ) -> None:
        limiter = self._create_session_limiter
        if limiter:
            key = api_key or client_ip or "anonymous"
            if not limiter.allow(key):
                if self._metrics is not None:
                    self._metrics.record_rate_limit_block("create_session", key)
                LOGGER.warning(
                    "CreateSession rate limited (key=%s, session_id=%s)",
                    key,
                    session_id,
                )
                abort_with_error(context, ErrorCode.CREATE_SESSION_RATE_LIMITED)
        if self._config.max_sessions_per_ip > 0 and client_ip:
            if (
                self._session_registry.active_count_by_ip(client_ip)
                >= self._config.max_sessions_per_ip
            ):
                LOGGER.warning(
                    "Max sessions per IP exceeded (ip=%s limit=%d)",
                    client_ip,
                    self._config.max_sessions_per_ip,
                )
                abort_with_error(context, ErrorCode.SESSION_LIMIT_EXCEEDED)
        if self._config.max_sessions_per_api_key > 0 and api_key:
            if (
                self._session_registry.active_count_by_api_key(api_key)
                >= self._config.max_sessions_per_api_key
            ):
                LOGGER.warning(
                    "Max sessions per API key exceeded (api_key=%s limit=%d)",
                    api_key,
                    self._config.max_sessions_per_api_key,
                )
                abort_with_error(context, ErrorCode.SESSION_LIMIT_EXCEEDED)

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
            return self._config.default_vad_threshold
        return value


def _extract_client_ip(context: grpc.ServicerContext) -> str:
    peer = context.peer() if context else ""
    if not peer:
        return ""
    for prefix in ("ipv4:", "ipv6:"):
        if peer.startswith(prefix):
            rest = peer[len(prefix) :]
            if rest.startswith("[") and "]" in rest:
                return rest[1 : rest.index("]")]
            return rest.split(":", 1)[0]
    return ""


__all__ = [
    "SessionInfo",
    "SessionRegistry",
    "SessionState",
    "SessionFacade",
    "CreateSessionConfig",
    "CreateSessionHandler",
]
