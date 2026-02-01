"""Types and helpers for the stream orchestrator."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional, TypeAlias

from stt_server.backend.component.decode_scheduler import DecodeSchedulerHooks
from stt_server.config.languages import SupportedLanguages
from stt_server.utils import audio
from stt_server.utils.logger import LOGGER

if TYPE_CHECKING:
    from stt_server.backend.application.session_manager import SessionState
    from stt_server.backend.component.audio_storage import SessionAudioRecorder
    from stt_server.backend.component.decode_scheduler import DecodeStream
    from stt_server.backend.component.vad_gate import VADGate


def _noop() -> None:
    """Default no-op hook."""
    return None


def _zero() -> int:
    """Default hook returning zero."""
    return 0


ScheduleDecodeFn: TypeAlias = Callable[..., bool]
EmitWithActivityFn: TypeAlias = Callable[..., Iterator[Any]]


@dataclass(frozen=True)
class StreamSettings:
    """Streaming control settings."""

    vad_threshold: float
    vad_silence: float
    speech_rms_threshold: float
    session_timeout_sec: float
    default_sample_rate: int
    decode_timeout_sec: float
    language_lookup: SupportedLanguages
    log_transcripts: bool = False
    max_audio_seconds_per_session: float = 0.0
    max_audio_bytes_per_sec: int = 0
    max_audio_bytes_per_sec_burst: int = 0


@dataclass(frozen=True)
class StorageSettings:
    """Audio storage settings."""

    enabled: bool
    directory: str
    queue_max_chunks: Optional[int] = None
    max_bytes: Optional[int] = None
    max_files: Optional[int] = None
    max_age_days: Optional[int] = None


@dataclass(frozen=True)
class VADPoolSettings:
    """Silero VAD model pool settings."""

    size: Optional[int] = None
    prewarm: Optional[int] = None
    max_size: Optional[int] = None
    growth_factor: float = 1.5


@dataclass(frozen=True)
class BufferLimits:
    """Limits for per-stream and global audio buffering."""

    max_buffer_sec: Optional[float] = 20.0
    max_buffer_bytes: Optional[int] = None
    max_chunk_ms: Optional[int] = 2000
    max_total_buffer_bytes: Optional[int] = 64 * 1024 * 1024
    buffer_overlap_sec: float = 0.5


@dataclass(frozen=True)
class PartialDecodeSettings:
    """Controls periodic partial decoding behavior."""

    interval_sec: Optional[float] = 1.5
    window_sec: Optional[float] = 10.0


@dataclass(frozen=True)
class DecodeQueueSettings:
    """Controls per-stream and global decode queue limits."""

    max_pending_decodes_per_stream: int = 8
    max_pending_decodes_global: int = 64
    decode_queue_timeout_sec: float = 1.0


@dataclass(frozen=True)
class HealthSettings:
    """Thresholds for decode health monitoring."""

    window_sec: float = 60.0
    min_events: int = 5
    max_timeout_ratio: float = 0.5
    min_success_ratio: float = 0.5


@dataclass(frozen=True)
class StreamOrchestratorConfig:
    """Configuration for streaming recognition."""

    stream: StreamSettings
    storage: StorageSettings
    vad_pool: VADPoolSettings = field(default_factory=VADPoolSettings)
    buffer_limits: BufferLimits = field(default_factory=BufferLimits)
    partial_decode: PartialDecodeSettings = field(default_factory=PartialDecodeSettings)
    decode_queue: DecodeQueueSettings = field(default_factory=DecodeQueueSettings)
    health: HealthSettings = field(default_factory=HealthSettings)


@dataclass(frozen=True)
class StreamOrchestratorHooks:
    """Hook callbacks for stream lifecycle events."""

    on_vad_trigger: Callable[[], None] = _noop
    on_vad_utterance_start: Callable[[], None] = _noop
    active_vad_utterances: Callable[[], int] = _zero
    decode_hooks: DecodeSchedulerHooks = field(default_factory=DecodeSchedulerHooks)


@dataclass(frozen=True)
class FlowDecodeOps:
    """Decode operations used by the flow helpers."""

    ensure_capacity: Callable[
        [Optional["DecodeStream"], bool, Optional["SessionState"]], bool
    ]
    schedule: ScheduleDecodeFn
    emit_with_activity: EmitWithActivityFn
    maybe_schedule_periodic_partial: Callable[["_StreamState", Any, Any], None]
    cancel_pending_decodes: Callable[[Optional["DecodeStream"], Optional[str]], None]


@dataclass(frozen=True)
class FlowBufferOps:
    """Buffer operations used by the flow helpers."""

    clear: Callable[["_StreamState"], None]
    enforce_limit: Callable[["_StreamState", Any], None]
    apply_global_limit: Callable[["_StreamState", int], int]


@dataclass(frozen=True)
class FlowSessionOps:
    """Session operations used by the flow helpers."""

    ensure_session_from_chunk: Callable[
        [Optional["SessionState"], Any, Any], "SessionState"
    ]
    validate_token: Callable[[Optional["SessionState"], Any, Any], None]
    create_vad_state: Callable[["SessionState"], "VADGate"]
    step_init: Callable[["_StreamState"], None]


@dataclass(frozen=True)
class FlowHooksOps:
    """Hook operations used by the flow helpers."""

    on_vad_trigger: Callable[[], None]
    on_vad_utterance_start: Callable[[], None]
    active_vad_utterances: Callable[[], int]


@dataclass(frozen=True)
class FlowAudioOps:
    """Audio operations used by the flow helpers."""

    capture_audio_chunk: Callable[
        [
            Optional["SessionAudioRecorder"],
            Optional["SessionState"],
            Optional[int],
            bytes,
        ],
        Optional["SessionAudioRecorder"],
    ]
    max_chunk_bytes: Callable[[Optional[int]], Optional[int]]


@dataclass(frozen=True)
class FlowActivityOps:
    """Activity tracking operations used by the flow helpers."""

    mark: Callable[["_StreamState"], None]


@dataclass(frozen=True)
class FlowLimitOps:
    """Limit enforcement operations used by the flow helpers."""

    enforce_chunk: Callable[["_StreamState", Any, Any], None]


@dataclass(frozen=True)
class StreamFlowContext:
    """Adapter interface for stream flow helpers."""

    config: "StreamOrchestratorConfig"
    decode: FlowDecodeOps
    buffer: FlowBufferOps
    session: FlowSessionOps
    hooks: FlowHooksOps
    audio: FlowAudioOps
    activity: FlowActivityOps
    limits: FlowLimitOps


class StreamPhase(Enum):
    """High-level phases of the streaming loop."""

    INIT = "init"
    STREAMING = "streaming"
    DRAINING = "draining"
    DONE = "done"


@dataclass
class _StreamSessionState:
    """Session-related state for a streaming call."""

    session_state: Optional["SessionState"] = None
    session_logged: bool = False
    final_reason: str = "stream_end"
    session_start: float = field(default_factory=time.monotonic)
    client_disconnected: bool = False
    sample_rate: Optional[int] = None
    audio_recorder: Optional["SessionAudioRecorder"] = None


@dataclass
class _StreamVADState:
    """VAD-related state for a streaming call."""

    vad_state: Optional["VADGate"] = None
    vad_count: int = 0


@dataclass
class _StreamDecodeState:
    """Decode stream state for a streaming call."""

    decode_stream: Optional["DecodeStream"] = None


@dataclass
class _StreamBufferState:
    """Buffer state for a streaming call."""

    buffer: bytearray = field(default_factory=bytearray)
    buffer_start_sec: float = 0.0
    buffer_start_time: Optional[float] = None
    buffer_has_new_audio: bool = False
    last_partial_decode_sec: Optional[float] = None


@dataclass
class _StreamActivityState:
    """Activity tracking state for a streaming call."""

    audio_received_sec: float = 0.0
    last_activity: float = field(default_factory=time.monotonic)


@dataclass
class _StreamEventState:
    """Event flags for a streaming call."""

    stop_watchdog: threading.Event = field(default_factory=threading.Event)
    timeout_event: threading.Event = field(default_factory=threading.Event)
    disconnect_event: threading.Event = field(default_factory=threading.Event)
    processing_event: threading.Event = field(default_factory=threading.Event)
    stop_stream: bool = False


@dataclass
class _StreamState:
    """Mutable state tracked across a streaming call."""

    session: _StreamSessionState = field(default_factory=_StreamSessionState)
    vad: _StreamVADState = field(default_factory=_StreamVADState)
    decode: _StreamDecodeState = field(default_factory=_StreamDecodeState)
    buffer: _StreamBufferState = field(default_factory=_StreamBufferState)
    activity: _StreamActivityState = field(default_factory=_StreamActivityState)
    events: _StreamEventState = field(default_factory=_StreamEventState)
    phase: StreamPhase = StreamPhase.INIT


class _AudioBufferManager:
    """Tracks per-stream and global audio buffer sizes."""

    def __init__(self, config: StreamOrchestratorConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._total_bytes = 0

    def update_total(self, delta: int) -> None:
        """Update the global buffered audio byte total."""
        if delta == 0:
            return
        with self._lock:
            self._total_bytes = max(0, self._total_bytes + delta)

    def apply_global_limit(self, state: _StreamState, incoming_len: int) -> int:
        """Apply global buffer limits and return bytes allowed to keep."""
        if incoming_len <= 0:
            return 0
        limit = self._config.buffer_limits.max_total_buffer_bytes
        if not limit or limit <= 0:
            self.update_total(incoming_len)
            return incoming_len

        with self._lock:
            total = self._total_bytes
            overflow = total + incoming_len - limit
            if overflow <= 0:
                self._total_bytes = total + incoming_len
                return incoming_len

            drop_from_buffer = min(overflow, len(state.buffer.buffer))
            if drop_from_buffer > 0:
                del state.buffer.buffer[:drop_from_buffer]
                self._total_bytes = max(0, self._total_bytes - drop_from_buffer)
                rate = (
                    state.session.sample_rate or self._config.stream.default_sample_rate
                )
                dropped_sec = audio.chunk_duration_seconds(drop_from_buffer, rate)
                state.buffer.buffer_start_sec += dropped_sec
                if state.buffer.buffer_start_time is not None:
                    state.buffer.buffer_start_time += dropped_sec
                overflow -= drop_from_buffer

            if overflow > 0:
                LOGGER.warning(
                    "Global buffer limit reached; dropping %d bytes of incoming audio",
                    overflow,
                )

            incoming_keep = max(0, incoming_len - overflow)
            self._total_bytes = max(0, self._total_bytes + incoming_keep)
            return incoming_keep

    def clear(self, state: _StreamState) -> None:
        """Clear buffer and reset related state fields."""
        if state.buffer.buffer:
            self.update_total(-len(state.buffer.buffer))
            state.buffer.buffer = bytearray()
        state.buffer.buffer_start_time = None
        state.buffer.buffer_has_new_audio = False
        state.buffer.last_partial_decode_sec = None

    def buffer_limit_bytes(self, sample_rate: Optional[int]) -> Optional[int]:
        """Return per-stream buffer byte limit for the given sample rate."""
        limit_bytes: Optional[int] = None
        max_bytes = self._config.buffer_limits.max_buffer_bytes
        if max_bytes is not None and max_bytes > 0:
            limit_bytes = int(max_bytes)
        max_sec = self._config.buffer_limits.max_buffer_sec
        if max_sec is not None and max_sec > 0:
            rate = sample_rate or self._config.stream.default_sample_rate
            sec_limit = int(max_sec * rate * 2)
            if sec_limit > 0:
                limit_bytes = (
                    sec_limit if limit_bytes is None else min(limit_bytes, sec_limit)
                )
        return limit_bytes

    def partial_decode_window_bytes(self, sample_rate: Optional[int]) -> Optional[int]:
        """Return the byte window for partial decode scheduling."""
        window_sec = self._config.partial_decode.window_sec
        if window_sec is None or window_sec <= 0:
            return None
        rate = sample_rate or self._config.stream.default_sample_rate
        if rate <= 0:
            return None
        return max(1, int(window_sec * rate * 2))


__all__ = [
    "BufferLimits",
    "DecodeQueueSettings",
    "HealthSettings",
    "PartialDecodeSettings",
    "StorageSettings",
    "StreamFlowContext",
    "StreamOrchestratorConfig",
    "StreamOrchestratorHooks",
    "StreamPhase",
    "StreamSettings",
    "VADPoolSettings",
]
