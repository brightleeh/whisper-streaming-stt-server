from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from stt_server.config.default import (
    DEFAULT_AUDIO_STORAGE_DIR,
    DEFAULT_AUDIO_STORAGE_QUEUE_MAX_CHUNKS,
    DEFAULT_BUFFER_OVERLAP_SEC,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DECODE_PROFILE,
    DEFAULT_DECODE_QUEUE_TIMEOUT_SEC,
    DEFAULT_DECODE_TIMEOUT,
    DEFAULT_DEVICE,
    DEFAULT_EXPOSE_API_KEY_METRICS,
    DEFAULT_GRPC_MAX_RECEIVE_MESSAGE_BYTES,
    DEFAULT_GRPC_MAX_SEND_MESSAGE_BYTES,
    DEFAULT_HEALTH_MAX_TIMEOUT_RATIO,
    DEFAULT_HEALTH_MIN_EVENTS,
    DEFAULT_HEALTH_MIN_SUCCESS_RATIO,
    DEFAULT_HEALTH_WINDOW_SEC,
    DEFAULT_LANGUAGE,
    DEFAULT_LANGUAGE_FIX,
    DEFAULT_LOG_FILE,
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOG_METRICS,
    DEFAULT_MAX_BUFFER_SEC,
    DEFAULT_MAX_CHUNK_MS,
    DEFAULT_MAX_PENDING_DECODES_GLOBAL,
    DEFAULT_MAX_PENDING_DECODES_PER_STREAM,
    DEFAULT_MAX_SESSIONS,
    DEFAULT_MAX_TOTAL_BUFFER_BYTES,
    DEFAULT_METRICS_PORT,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_POOL_SIZE,
    DEFAULT_PERSIST_AUDIO,
    DEFAULT_PORT,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SPEECH_RMS_THRESHOLD,
    DEFAULT_TASK,
    DEFAULT_VAD_MODEL_POOL_GROWTH_FACTOR,
    DEFAULT_VAD_MODEL_POOL_SIZE,
    DEFAULT_VAD_MODEL_PREWARM,
    DEFAULT_VAD_SILENCE,
    DEFAULT_VAD_THRESHOLD,
    MODEL_SECTION_MAP,
    SERVER_SECTION_MAP,
    default_decode_profiles,
)


@dataclass
class ServerConfig:
    model: str = DEFAULT_MODEL_NAME
    device: str = DEFAULT_DEVICE
    compute_type: str = DEFAULT_COMPUTE_TYPE
    language: str = DEFAULT_LANGUAGE
    language_fix: bool = DEFAULT_LANGUAGE_FIX
    task: str = DEFAULT_TASK
    decode_profiles: Dict[str, Dict[str, Any]] = field(
        default_factory=default_decode_profiles
    )
    default_decode_profile: str = "realtime"
    model_pool_size: int = DEFAULT_MODEL_POOL_SIZE
    port: int = DEFAULT_PORT
    max_sessions: int = DEFAULT_MAX_SESSIONS
    metrics_port: int = DEFAULT_METRICS_PORT
    decode_timeout_sec: float = DEFAULT_DECODE_TIMEOUT
    log_metrics: bool = DEFAULT_LOG_METRICS
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
    sample_rate: int = DEFAULT_SAMPLE_RATE
    session_timeout_sec: float = 60.0
    max_buffer_sec: Optional[float] = DEFAULT_MAX_BUFFER_SEC
    max_buffer_bytes: Optional[int] = None
    max_chunk_ms: Optional[int] = DEFAULT_MAX_CHUNK_MS
    max_pending_decodes_per_stream: int = DEFAULT_MAX_PENDING_DECODES_PER_STREAM
    max_pending_decodes_global: int = DEFAULT_MAX_PENDING_DECODES_GLOBAL
    max_total_buffer_bytes: Optional[int] = DEFAULT_MAX_TOTAL_BUFFER_BYTES
    decode_queue_timeout_sec: float = DEFAULT_DECODE_QUEUE_TIMEOUT_SEC
    buffer_overlap_sec: float = DEFAULT_BUFFER_OVERLAP_SEC
    grpc_max_receive_message_bytes: Optional[int] = (
        DEFAULT_GRPC_MAX_RECEIVE_MESSAGE_BYTES
    )
    grpc_max_send_message_bytes: Optional[int] = DEFAULT_GRPC_MAX_SEND_MESSAGE_BYTES
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
        if (
            section == "server"
            and "session_timeout_sec" in data
            and data["session_timeout_sec"] is not None
        ):
            cfg.session_timeout_sec = float(data["session_timeout_sec"])

    _apply_decode_profiles(cfg, raw.get("decode_profiles"))

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


__all__ = [
    "ServerConfig",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_MODEL_CONFIG_PATH",
    "DEFAULT_DECODE_PROFILE",
    "load_config",
]
