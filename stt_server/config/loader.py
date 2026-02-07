"""Load server and model configuration from YAML files."""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from stt_server.config.default import (
    DEFAULT_ADAPTIVE_BATCH_WINDOW_MIN_MS,
    DEFAULT_ADAPTIVE_BUFFER_RATIO_HIGH,
    DEFAULT_ADAPTIVE_CREATE_SESSION_BACKOFF_SEC,
    DEFAULT_ADAPTIVE_ORPHAN_RATE_HIGH,
    DEFAULT_ADAPTIVE_PARTIAL_INTERVAL_MAX_SEC,
    DEFAULT_ADAPTIVE_PARTIAL_INTERVAL_SCALE,
    DEFAULT_ADAPTIVE_PENDING_RATIO_HIGH,
    DEFAULT_ADAPTIVE_THROTTLE_ENABLED,
    DEFAULT_ADAPTIVE_THROTTLE_INTERVAL_SEC,
    DEFAULT_AUDIO_STORAGE_DIR,
    DEFAULT_AUDIO_STORAGE_QUEUE_MAX_CHUNKS,
    DEFAULT_BUFFER_OVERLAP_SEC,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_CREATE_SESSION_AUTH_PROFILE,
    DEFAULT_CREATE_SESSION_AUTH_SECRET,
    DEFAULT_CREATE_SESSION_AUTH_TTL_SEC,
    DEFAULT_CREATE_SESSION_BURST,
    DEFAULT_CREATE_SESSION_RPS,
    DEFAULT_DECODE_BATCH_WINDOW_MS,
    DEFAULT_DECODE_PROFILE,
    DEFAULT_DECODE_PROFILE_NAME,
    DEFAULT_DECODE_QUEUE_TIMEOUT_SEC,
    DEFAULT_DECODE_TIMEOUT,
    DEFAULT_DEVICE,
    DEFAULT_EMIT_FINAL_ON_VAD,
    DEFAULT_EXPOSE_API_KEY_METRICS,
    DEFAULT_GRPC_MAX_RECEIVE_MESSAGE_BYTES,
    DEFAULT_GRPC_MAX_SEND_MESSAGE_BYTES,
    DEFAULT_GRPC_WORKER_THREADS,
    DEFAULT_HEALTH_MAX_TIMEOUT_RATIO,
    DEFAULT_HEALTH_MIN_EVENTS,
    DEFAULT_HEALTH_MIN_SUCCESS_RATIO,
    DEFAULT_HEALTH_WINDOW_SEC,
    DEFAULT_HTTP_HOST,
    DEFAULT_HTTP_RATE_LIMIT_BURST,
    DEFAULT_HTTP_RATE_LIMIT_RPS,
    DEFAULT_HTTP_TRUSTED_PROXIES,
    DEFAULT_LANGUAGE,
    DEFAULT_LANGUAGE_FIX,
    DEFAULT_LOG_FILE,
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOG_METRICS,
    DEFAULT_LOG_TRANSCRIPTS,
    DEFAULT_MAX_AUDIO_BYTES_PER_SEC,
    DEFAULT_MAX_AUDIO_BYTES_PER_SEC_BATCH,
    DEFAULT_MAX_AUDIO_BYTES_PER_SEC_BURST,
    DEFAULT_MAX_AUDIO_BYTES_PER_SEC_BURST_BATCH,
    DEFAULT_MAX_AUDIO_BYTES_PER_SEC_BURST_REALTIME,
    DEFAULT_MAX_AUDIO_BYTES_PER_SEC_REALTIME,
    DEFAULT_MAX_AUDIO_SECONDS_PER_SESSION,
    DEFAULT_MAX_BUFFER_SEC,
    DEFAULT_MAX_CHUNK_MS,
    DEFAULT_MAX_DECODE_BATCH_SIZE,
    DEFAULT_MAX_PENDING_DECODES_GLOBAL,
    DEFAULT_MAX_PENDING_DECODES_PER_STREAM,
    DEFAULT_MAX_SESSIONS,
    DEFAULT_MAX_SESSIONS_PER_API_KEY,
    DEFAULT_MAX_SESSIONS_PER_IP,
    DEFAULT_MAX_TOTAL_BUFFER_BYTES,
    DEFAULT_METRICS_PORT,
    DEFAULT_MODEL_BACKEND,
    DEFAULT_MODEL_LOAD_PROFILE_NAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_POOL_SIZE,
    DEFAULT_PARTIAL_DECODE_INTERVAL_SEC,
    DEFAULT_PARTIAL_DECODE_WINDOW_SEC,
    DEFAULT_PERSIST_AUDIO,
    DEFAULT_PORT,
    DEFAULT_REQUIRE_API_KEY,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SPEECH_RMS_THRESHOLD,
    DEFAULT_TASK,
    DEFAULT_TLS_CERT_FILE,
    DEFAULT_TLS_KEY_FILE,
    DEFAULT_TLS_REQUIRED,
    DEFAULT_TRANSCRIPT_LOG_FILE,
    DEFAULT_TRANSCRIPT_RETENTION_DAYS,
    DEFAULT_VAD_MODEL_POOL_GROWTH_FACTOR,
    DEFAULT_VAD_MODEL_POOL_SIZE,
    DEFAULT_VAD_MODEL_PREWARM,
    DEFAULT_VAD_SILENCE,
    DEFAULT_VAD_THRESHOLD,
    DEFAULT_WS_HOST,
    DEFAULT_WS_PORT,
    MODEL_SECTION_MAP,
    SERVER_SECTION_MAP,
    default_decode_profiles,
)


@dataclass
class ServerConfig:
    """Server + runtime configuration with defaults applied."""

    model: str = DEFAULT_MODEL_NAME
    device: str = DEFAULT_DEVICE
    compute_type: str = DEFAULT_COMPUTE_TYPE
    language: str = DEFAULT_LANGUAGE
    language_fix: bool = DEFAULT_LANGUAGE_FIX
    task: str = DEFAULT_TASK
    decode_profiles: Dict[str, Dict[str, Any]] = field(
        default_factory=default_decode_profiles
    )
    default_decode_profile: str = DEFAULT_DECODE_PROFILE_NAME
    model_load_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    default_model_load_profile: str = DEFAULT_MODEL_LOAD_PROFILE_NAME
    model_backend: str = DEFAULT_MODEL_BACKEND
    model_pool_size: int = DEFAULT_MODEL_POOL_SIZE
    port: int = DEFAULT_PORT
    max_sessions: int = DEFAULT_MAX_SESSIONS
    metrics_port: int = DEFAULT_METRICS_PORT
    http_host: str = DEFAULT_HTTP_HOST
    ws_host: str = DEFAULT_WS_HOST
    http_rate_limit_rps: float = DEFAULT_HTTP_RATE_LIMIT_RPS
    http_rate_limit_burst: float = DEFAULT_HTTP_RATE_LIMIT_BURST
    http_trusted_proxies: list[str] = field(
        default_factory=lambda: list(DEFAULT_HTTP_TRUSTED_PROXIES)
    )
    ws_port: int = DEFAULT_WS_PORT
    create_session_rps: float = DEFAULT_CREATE_SESSION_RPS
    create_session_burst: float = DEFAULT_CREATE_SESSION_BURST
    max_sessions_per_ip: int = DEFAULT_MAX_SESSIONS_PER_IP
    max_sessions_per_api_key: int = DEFAULT_MAX_SESSIONS_PER_API_KEY
    max_audio_seconds_per_session: float = DEFAULT_MAX_AUDIO_SECONDS_PER_SESSION
    max_audio_bytes_per_sec: int = DEFAULT_MAX_AUDIO_BYTES_PER_SEC
    max_audio_bytes_per_sec_burst: int = DEFAULT_MAX_AUDIO_BYTES_PER_SEC_BURST
    max_audio_bytes_per_sec_realtime: Optional[int] = (
        DEFAULT_MAX_AUDIO_BYTES_PER_SEC_REALTIME
    )
    max_audio_bytes_per_sec_burst_realtime: Optional[int] = (
        DEFAULT_MAX_AUDIO_BYTES_PER_SEC_BURST_REALTIME
    )
    max_audio_bytes_per_sec_batch: Optional[int] = DEFAULT_MAX_AUDIO_BYTES_PER_SEC_BATCH
    max_audio_bytes_per_sec_burst_batch: Optional[int] = (
        DEFAULT_MAX_AUDIO_BYTES_PER_SEC_BURST_BATCH
    )
    decode_timeout_sec: float = DEFAULT_DECODE_TIMEOUT
    log_metrics: bool = DEFAULT_LOG_METRICS
    log_transcripts: bool = DEFAULT_LOG_TRANSCRIPTS
    vad_silence: float = DEFAULT_VAD_SILENCE
    vad_threshold: float = DEFAULT_VAD_THRESHOLD
    vad_model_pool_size: int = DEFAULT_VAD_MODEL_POOL_SIZE
    vad_model_prewarm: int = DEFAULT_VAD_MODEL_PREWARM
    vad_model_pool_growth_factor: float = DEFAULT_VAD_MODEL_POOL_GROWTH_FACTOR
    expose_api_key_metrics: bool = DEFAULT_EXPOSE_API_KEY_METRICS
    speech_rms_threshold: float = DEFAULT_SPEECH_RMS_THRESHOLD
    log_level: str = DEFAULT_LOG_LEVEL
    log_file: Optional[str] = DEFAULT_LOG_FILE
    faster_whisper_log_level: Optional[str] = None
    transcript_log_file: Optional[str] = DEFAULT_TRANSCRIPT_LOG_FILE
    transcript_retention_days: Optional[int] = DEFAULT_TRANSCRIPT_RETENTION_DAYS
    sample_rate: int = DEFAULT_SAMPLE_RATE
    session_timeout_sec: float = 60.0
    max_buffer_sec: Optional[float] = DEFAULT_MAX_BUFFER_SEC
    max_buffer_bytes: Optional[int] = None
    max_chunk_ms: Optional[int] = DEFAULT_MAX_CHUNK_MS
    partial_decode_interval_sec: Optional[float] = DEFAULT_PARTIAL_DECODE_INTERVAL_SEC
    partial_decode_window_sec: Optional[float] = DEFAULT_PARTIAL_DECODE_WINDOW_SEC
    emit_final_on_vad: bool = DEFAULT_EMIT_FINAL_ON_VAD
    adaptive_throttle_enabled: bool = DEFAULT_ADAPTIVE_THROTTLE_ENABLED
    adaptive_throttle_interval_sec: float = DEFAULT_ADAPTIVE_THROTTLE_INTERVAL_SEC
    adaptive_pending_ratio_high: float = DEFAULT_ADAPTIVE_PENDING_RATIO_HIGH
    adaptive_buffer_ratio_high: float = DEFAULT_ADAPTIVE_BUFFER_RATIO_HIGH
    adaptive_orphan_rate_high: float = DEFAULT_ADAPTIVE_ORPHAN_RATE_HIGH
    adaptive_partial_interval_scale: float = DEFAULT_ADAPTIVE_PARTIAL_INTERVAL_SCALE
    adaptive_partial_interval_max_sec: Optional[float] = (
        DEFAULT_ADAPTIVE_PARTIAL_INTERVAL_MAX_SEC
    )
    adaptive_batch_window_min_ms: int = DEFAULT_ADAPTIVE_BATCH_WINDOW_MIN_MS
    adaptive_create_session_backoff_sec: float = (
        DEFAULT_ADAPTIVE_CREATE_SESSION_BACKOFF_SEC
    )
    max_pending_decodes_per_stream: int = DEFAULT_MAX_PENDING_DECODES_PER_STREAM
    max_pending_decodes_global: int = DEFAULT_MAX_PENDING_DECODES_GLOBAL
    max_total_buffer_bytes: Optional[int] = DEFAULT_MAX_TOTAL_BUFFER_BYTES
    decode_queue_timeout_sec: float = DEFAULT_DECODE_QUEUE_TIMEOUT_SEC
    decode_batch_window_ms: int = DEFAULT_DECODE_BATCH_WINDOW_MS
    max_decode_batch_size: int = DEFAULT_MAX_DECODE_BATCH_SIZE
    buffer_overlap_sec: float = DEFAULT_BUFFER_OVERLAP_SEC
    grpc_max_receive_message_bytes: Optional[int] = (
        DEFAULT_GRPC_MAX_RECEIVE_MESSAGE_BYTES
    )
    grpc_max_send_message_bytes: Optional[int] = DEFAULT_GRPC_MAX_SEND_MESSAGE_BYTES
    grpc_worker_threads: int = DEFAULT_GRPC_WORKER_THREADS
    tls_cert_file: Optional[str] = DEFAULT_TLS_CERT_FILE
    tls_key_file: Optional[str] = DEFAULT_TLS_KEY_FILE
    tls_required: bool = DEFAULT_TLS_REQUIRED
    require_api_key: bool = DEFAULT_REQUIRE_API_KEY
    create_session_auth_profile: str = DEFAULT_CREATE_SESSION_AUTH_PROFILE
    create_session_auth_secret: str = DEFAULT_CREATE_SESSION_AUTH_SECRET
    create_session_auth_ttl_sec: float = DEFAULT_CREATE_SESSION_AUTH_TTL_SEC
    health_window_sec: float = DEFAULT_HEALTH_WINDOW_SEC
    health_min_events: int = DEFAULT_HEALTH_MIN_EVENTS
    health_max_timeout_ratio: float = DEFAULT_HEALTH_MAX_TIMEOUT_RATIO
    health_min_success_ratio: float = DEFAULT_HEALTH_MIN_SUCCESS_RATIO
    persist_audio: bool = DEFAULT_PERSIST_AUDIO
    audio_storage_dir: str = DEFAULT_AUDIO_STORAGE_DIR
    audio_storage_queue_max_chunks: Optional[int] = (
        DEFAULT_AUDIO_STORAGE_QUEUE_MAX_CHUNKS
    )
    audio_storage_max_bytes: Optional[int] = None
    audio_storage_max_files: Optional[int] = None
    audio_storage_max_age_days: Optional[int] = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "server.yaml"
DEFAULT_MODEL_CONFIG_PATH = PROJECT_ROOT / "config" / "model.yaml"

SECTION_MAP: Dict[str, Dict[str, str]] = {"model": MODEL_SECTION_MAP}
SECTION_MAP.update(SERVER_SECTION_MAP)


def load_config(
    server_path: Optional[Path] = None, model_path: Optional[Path] = None
) -> ServerConfig:
    """Load server + model configuration from YAML, falling back to defaults."""
    cfg = ServerConfig()
    server_data = _read_yaml(server_path or DEFAULT_CONFIG_PATH)
    if server_data:
        _apply_sections(cfg, server_data)
    model_data = _read_yaml(model_path or DEFAULT_MODEL_CONFIG_PATH)
    if model_data:
        _apply_sections(cfg, model_data)

    _ensure_default_model_load_profile(cfg)

    return cfg


def _read_yaml(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if not path or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if isinstance(data, dict):
        return data
    return None


def _apply_sections(cfg: ServerConfig, raw: Dict[str, Any]) -> None:
    field_names = {f.name for f in fields(ServerConfig)}
    for section, mapping in SECTION_MAP.items():
        data = raw.get(section)
        if not isinstance(data, dict):
            continue
        for key, attr in mapping.items():
            if key in data and data[key] is not None:
                setattr(cfg, attr, data[key])
        if section == "model":
            _apply_decode_profiles(cfg, data.get("decode_profiles"))
            _apply_model_load_profiles(cfg, data.get("model_load_profiles"))
        if (
            section == "server"
            and "session_timeout_sec" in data
            and data["session_timeout_sec"] is not None
        ):
            cfg.session_timeout_sec = float(data["session_timeout_sec"])

    _apply_decode_profiles(cfg, raw.get("decode_profiles"))
    _apply_model_load_profiles(cfg, raw.get("model_load_profiles"))

    for key, value in raw.items():
        if key in SECTION_MAP:
            continue
        if key in field_names and value is not None:
            setattr(cfg, key, value)


def _apply_decode_profiles(
    cfg: ServerConfig, profiles: Optional[Dict[str, Any]]
) -> None:
    normalized = _normalize_profiles(profiles)
    if normalized:
        cfg.decode_profiles = normalized


def _apply_model_load_profiles(
    cfg: ServerConfig, profiles: Optional[Dict[str, Any]]
) -> None:
    normalized = _normalize_profiles(profiles)
    if normalized:
        cfg.model_load_profiles = normalized


def _normalize_profiles(
    profiles: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    if not isinstance(profiles, dict):
        return {}
    normalized: Dict[str, Dict[str, Any]] = {}
    for name, options in profiles.items():
        if isinstance(options, dict):
            normalized[name] = dict(options)
    return normalized


def _build_default_model_load_profile(cfg: ServerConfig) -> Dict[str, Any]:
    return {
        "model_size": cfg.model,
        "device": cfg.device,
        "compute_type": cfg.compute_type,
        "pool_size": max(1, int(cfg.model_pool_size)),
        "language": cfg.language,
        "language_fix": cfg.language_fix,
        "task": cfg.task,
        "backend": cfg.model_backend,
        "log_metrics": cfg.log_metrics,
    }


def _ensure_default_model_load_profile(cfg: ServerConfig) -> None:
    if cfg.model_load_profiles:
        if cfg.default_model_load_profile not in cfg.model_load_profiles:
            cfg.default_model_load_profile = next(iter(cfg.model_load_profiles))
        return
    cfg.model_load_profiles = {
        cfg.default_model_load_profile: _build_default_model_load_profile(cfg)
    }


__all__ = [
    "ServerConfig",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_MODEL_CONFIG_PATH",
    "DEFAULT_DECODE_PROFILE",
    "load_config",
]
