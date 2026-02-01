"""Runtime configuration models for the STT application layer."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ModelRuntimeConfig:
    """Model-related runtime configuration."""

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
    require_api_key: bool = False
    model_load_profiles: Optional[Dict[str, Dict[str, Any]]] = None
    default_model_load_profile: str = "default"


@dataclass
class StreamingRuntimeConfig:
    """Streaming/VAD runtime configuration."""

    vad_silence: float = 0.8
    vad_threshold: float = 0.5
    vad_model_pool_size: int = 4
    vad_model_prewarm: int = 1
    vad_model_pool_max_size: Optional[int] = None
    vad_model_pool_growth_factor: float = 1.5
    speech_rms_threshold: float = 0.02
    session_timeout_sec: float = 60.0
    sample_rate: int = 16000
    decode_timeout_sec: float = 30.0
    max_buffer_sec: Optional[float] = 20.0
    max_buffer_bytes: Optional[int] = None
    max_chunk_ms: Optional[int] = 2000
    partial_decode_interval_sec: Optional[float] = 1.5
    partial_decode_window_sec: Optional[float] = 10.0
    max_pending_decodes_per_stream: int = 8
    max_pending_decodes_global: int = 64
    max_total_buffer_bytes: Optional[int] = 64 * 1024 * 1024
    decode_queue_timeout_sec: float = 1.0
    buffer_overlap_sec: float = 0.5
    health_window_sec: float = 60.0
    health_min_events: int = 5
    health_max_timeout_ratio: float = 0.5
    health_min_success_ratio: float = 0.5
    expose_api_key_metrics: bool = False
    log_transcripts: bool = False


@dataclass
class StorageRuntimeConfig:
    """Audio storage runtime configuration."""

    enabled: bool = False
    directory: str = "data/audio"
    queue_max_chunks: Optional[int] = 256
    max_bytes: Optional[int] = None
    max_files: Optional[int] = None
    max_age_days: Optional[int] = None


@dataclass
class ServicerConfig:
    """Top-level configuration for the gRPC servicer runtime."""

    model: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    streaming: StreamingRuntimeConfig = field(default_factory=StreamingRuntimeConfig)
    storage: StorageRuntimeConfig = field(default_factory=StorageRuntimeConfig)
