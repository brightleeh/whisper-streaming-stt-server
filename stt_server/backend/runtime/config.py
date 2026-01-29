"""Runtime configuration models for the STT application layer."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ModelRuntimeConfig:
    model_size: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    log_metrics: bool = False
    model_pool_size: int = 1
    language: str = ""
    language_fix: bool = False
    task: str = "transcribe"
    decode_profiles: Optional[Dict[str, Dict[str, Any]]] = None
    default_decode_profile: str = "realtime"


@dataclass
class StreamingRuntimeConfig:
    vad_silence: float = 0.8
    vad_threshold: float = 0.5
    speech_rms_threshold: float = 0.02
    session_timeout_sec: float = 60.0
    sample_rate: int = 16000
    decode_timeout_sec: float = 30.0
    max_buffer_sec: Optional[float] = 60.0
    max_buffer_bytes: Optional[int] = None


@dataclass
class StorageRuntimeConfig:
    enabled: bool = False
    directory: str = "data/audio"
    max_bytes: Optional[int] = None
    max_files: Optional[int] = None
    max_age_days: Optional[int] = None


@dataclass
class ServicerConfig:
    model: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    streaming: StreamingRuntimeConfig = field(default_factory=StreamingRuntimeConfig)
    storage: StorageRuntimeConfig = field(default_factory=StorageRuntimeConfig)
