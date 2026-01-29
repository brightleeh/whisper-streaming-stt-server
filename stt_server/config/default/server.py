"""Default values for server/runtime configuration."""

from typing import Dict

DEFAULT_PORT = 50051
DEFAULT_MAX_SESSIONS = 4
DEFAULT_METRICS_PORT = 8000
DEFAULT_DECODE_TIMEOUT = 30.0
DEFAULT_LOG_METRICS = False
DEFAULT_VAD_SILENCE = 0.8
DEFAULT_VAD_THRESHOLD = 0.5
DEFAULT_SPEECH_RMS_THRESHOLD = 0.02
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MAX_BUFFER_SEC = 60.0
DEFAULT_HEALTH_WINDOW_SEC = 60.0
DEFAULT_HEALTH_MIN_EVENTS = 5
DEFAULT_HEALTH_MAX_TIMEOUT_RATIO = 0.5
DEFAULT_HEALTH_MIN_SUCCESS_RATIO = 0.5
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FILE = None
DEFAULT_PERSIST_AUDIO = False
DEFAULT_AUDIO_STORAGE_DIR = "data/audio"
DEFAULT_AUDIO_STORAGE_QUEUE_MAX_CHUNKS = 256

SERVER_SECTION_MAP: Dict[str, Dict[str, str]] = {
    "server": {
        "port": "port",
        "max_sessions": "max_sessions",
        "metrics_port": "metrics_port",
        "decode_timeout_sec": "decode_timeout_sec",
        "max_buffer_sec": "max_buffer_sec",
        "max_buffer_bytes": "max_buffer_bytes",
        "log_metrics": "log_metrics",
        "sample_rate": "sample_rate",
    },
    "vad": {
        "silence": "vad_silence",
        "threshold": "vad_threshold",
    },
    "safety": {
        "speech_rms_threshold": "speech_rms_threshold",
    },
    "logging": {
        "level": "log_level",
        "file": "log_file",
        "faster_whisper_level": "faster_whisper_log_level",
    },
    "storage": {
        "persist_audio": "persist_audio",
        "directory": "audio_storage_dir",
        "queue_max_chunks": "audio_storage_queue_max_chunks",
        "max_bytes": "audio_storage_max_bytes",
        "max_files": "audio_storage_max_files",
        "max_age_days": "audio_storage_max_age_days",
    },
    "health": {
        "window_sec": "health_window_sec",
        "min_events": "health_min_events",
        "max_timeout_ratio": "health_max_timeout_ratio",
        "min_success_ratio": "health_min_success_ratio",
    },
}

__all__ = [
    "DEFAULT_PORT",
    "DEFAULT_MAX_SESSIONS",
    "DEFAULT_METRICS_PORT",
    "DEFAULT_DECODE_TIMEOUT",
    "DEFAULT_LOG_METRICS",
    "DEFAULT_VAD_SILENCE",
    "DEFAULT_VAD_THRESHOLD",
    "DEFAULT_SPEECH_RMS_THRESHOLD",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_MAX_BUFFER_SEC",
    "DEFAULT_HEALTH_WINDOW_SEC",
    "DEFAULT_HEALTH_MIN_EVENTS",
    "DEFAULT_HEALTH_MAX_TIMEOUT_RATIO",
    "DEFAULT_HEALTH_MIN_SUCCESS_RATIO",
    "DEFAULT_LOG_LEVEL",
    "DEFAULT_LOG_FILE",
    "DEFAULT_PERSIST_AUDIO",
    "DEFAULT_AUDIO_STORAGE_DIR",
    "DEFAULT_AUDIO_STORAGE_QUEUE_MAX_CHUNKS",
    "SERVER_SECTION_MAP",
]
