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
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FILE = None
DEFAULT_PERSIST_AUDIO = False
DEFAULT_AUDIO_STORAGE_DIR = "data/audio"

SERVER_SECTION_MAP: Dict[str, Dict[str, str]] = {
    "server": {
        "port": "port",
        "max_sessions": "max_sessions",
        "metrics_port": "metrics_port",
        "decode_timeout_sec": "decode_timeout_sec",
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
    },
    "storage": {
        "persist_audio": "persist_audio",
        "directory": "audio_storage_dir",
        "max_bytes": "audio_storage_max_bytes",
        "max_files": "audio_storage_max_files",
        "max_age_days": "audio_storage_max_age_days",
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
    "DEFAULT_LOG_LEVEL",
    "DEFAULT_LOG_FILE",
    "DEFAULT_PERSIST_AUDIO",
    "DEFAULT_AUDIO_STORAGE_DIR",
    "SERVER_SECTION_MAP",
]
