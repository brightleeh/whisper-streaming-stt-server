from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, Optional

import grpc

from gen.stt.python.v1 import stt_pb2
from stt_server.backend.application.model_registry import ModelRegistry
from stt_server.backend.application.session_manager import SessionFacade, SessionState
from stt_server.backend.component.audio_storage import (
    AudioStorageConfig,
    AudioStorageManager,
    SessionAudioRecorder,
)
from stt_server.backend.component.decode_scheduler import (
    DecodeScheduler,
    DecodeSchedulerHooks,
    DecodeStream,
)
from stt_server.backend.component.vad_gate import (
    VADGate,
    buffer_is_speech,
    configure_vad_model_pool,
)
from stt_server.config.languages import SupportedLanguages
from stt_server.errors import ErrorCode, abort_with_error
from stt_server.utils import audio
from stt_server.utils.logger import LOGGER, clear_session_id, set_session_id

if TYPE_CHECKING:
    from stt_server.model.worker import ModelWorker


@dataclass(frozen=True)
class StreamOrchestratorConfig:
    # Streaming control settings
    vad_threshold: float
    vad_silence: float
    speech_rms_threshold: float
    session_timeout_sec: float
    default_sample_rate: int
    decode_timeout_sec: float
    language_lookup: SupportedLanguages

    # Audio storage settings
    storage_enabled: bool
    storage_directory: str
    storage_queue_max_chunks: Optional[int] = None
    storage_max_bytes: Optional[int] = None
    storage_max_files: Optional[int] = None
    storage_max_age_days: Optional[int] = None

    # VAD model pool settings
    vad_model_pool_size: Optional[int] = None
    vad_model_prewarm: Optional[int] = None
    vad_model_pool_max_size: Optional[int] = None
    vad_model_pool_growth_factor: float = 1.5

    # Buffer control settings
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

    # Health check thresholds
    health_window_sec: float = 60.0
    health_min_events: int = 5
    health_max_timeout_ratio: float = 0.5
    health_min_success_ratio: float = 0.5


def _noop() -> None:
    return None


def _zero() -> int:
    return 0


@dataclass(frozen=True)
class StreamOrchestratorHooks:
    on_vad_trigger: Callable[[], None] = _noop
    on_vad_utterance_start: Callable[[], None] = _noop
    active_vad_utterances: Callable[[], int] = _zero
    decode_hooks: DecodeSchedulerHooks = field(default_factory=DecodeSchedulerHooks)


@dataclass
class _StreamState:
    session_state: Optional[SessionState] = None
    vad_state: Optional[VADGate] = None
    decode_stream: Optional[DecodeStream] = None
    session_logged: bool = False
    final_reason: str = "stream_end"
    session_start: float = field(default_factory=time.monotonic)
    vad_count: int = 0
    audio_recorder: Optional[SessionAudioRecorder] = None
    audio_received_sec: float = 0.0
    buffer_start_sec: float = 0.0
    buffer_start_time: Optional[float] = None
    client_disconnected: bool = False
    buffer_has_new_audio: bool = False
    last_partial_decode_sec: Optional[float] = None
    buffer: bytearray = field(default_factory=bytearray)
    sample_rate: Optional[int] = None
    last_activity: float = field(default_factory=time.monotonic)
    stop_watchdog: threading.Event = field(default_factory=threading.Event)
    timeout_event: threading.Event = field(default_factory=threading.Event)
    disconnect_event: threading.Event = field(default_factory=threading.Event)
    stop_stream: bool = False


class StreamOrchestrator:
    """Executes the streaming recognition loop for the gRPC servicer."""

    def __init__(
        self,
        session_facade: SessionFacade,
        model_registry: ModelRegistry,
        config: StreamOrchestratorConfig,
        hooks: StreamOrchestratorHooks | None = None,
    ) -> None:
        self._session_facade = session_facade
        self._model_registry = model_registry
        self._config = config
        self._hooks = hooks or StreamOrchestratorHooks()
        self._decode_scheduler = self._create_decode_scheduler(config)
        self._audio_storage: Optional[AudioStorageManager] = None
        self._buffer_bytes_lock = threading.Lock()
        self._buffer_bytes_total = 0
        configure_vad_model_pool(
            config.vad_model_pool_size,
            config.vad_model_prewarm,
            config.vad_model_pool_max_size,
            config.vad_model_pool_growth_factor,
        )
        if config.storage_enabled:
            storage_directory = Path(config.storage_directory).expanduser()
            storage_policy = AudioStorageConfig(
                enabled=True,
                directory=storage_directory,
                queue_max_chunks=config.storage_queue_max_chunks,
                max_bytes=config.storage_max_bytes,
                max_files=config.storage_max_files,
                max_age_days=config.storage_max_age_days,
            )
            self._audio_storage = AudioStorageManager(storage_policy)
            LOGGER.info(
                "Audio storage enabled directory=%s queue_max_chunks=%s max_bytes=%s max_files=%s max_age_days=%s",
                storage_directory,
                config.storage_queue_max_chunks,
                config.storage_max_bytes,
                config.storage_max_files,
                config.storage_max_age_days,
            )

    @property
    def decode_scheduler(self) -> DecodeScheduler:
        return self._decode_scheduler

    def load_model(self, model_id: str, config: Dict[str, Any]) -> None:
        self._model_registry.load_model(model_id, config)

    def acquire_worker(self, model_id: str) -> ModelWorker:
        worker = self._model_registry.get_worker(model_id)
        if not worker:
            raise RuntimeError(f"No worker available for model_id='{model_id}'")
        return worker

    @property
    def model_registry(self) -> ModelRegistry:
        return self._model_registry

    def _create_decode_scheduler(
        self, config: StreamOrchestratorConfig
    ) -> DecodeScheduler:
        return DecodeScheduler(
            self,
            decode_timeout_sec=config.decode_timeout_sec,
            language_lookup=config.language_lookup,
            max_pending_decodes_global=config.max_pending_decodes_global,
            health_window_sec=config.health_window_sec,
            health_min_events=config.health_min_events,
            health_max_timeout_ratio=config.health_max_timeout_ratio,
            health_min_success_ratio=config.health_min_success_ratio,
            hooks=self._hooks.decode_hooks,
        )

    def _on_vad_trigger(self) -> None:
        self._hooks.on_vad_trigger()

    def _on_vad_utterance_start(self) -> None:
        self._hooks.on_vad_utterance_start()

    def _active_vad_utterances(self) -> int:
        return self._hooks.active_vad_utterances()

    def _ensure_decode_capacity(
        self,
        decode_stream: Optional[DecodeStream],
        is_final: bool,
        session_state: Optional[SessionState],
    ) -> bool:
        if decode_stream is None:
            return False
        limit = self._config.max_pending_decodes_per_stream
        if limit <= 0:
            return True
        pending = decode_stream.pending_count()
        current_session_id = session_state.session_id if session_state else "unknown"
        if is_final:
            if pending >= limit:
                cancelled, orphaned = decode_stream.drop_pending_partials()
                if cancelled or orphaned:
                    LOGGER.warning(
                        "Dropped %d pending partial decodes for final decode (session_id=%s)",
                        cancelled + orphaned,
                        current_session_id,
                    )
            return True
        if pending < limit:
            return True
        decode_stream.drop_pending_partials(1)
        if decode_stream.pending_count() >= limit:
            LOGGER.warning(
                "Pending decode limit reached; dropping partial decode (session_id=%s pending=%d limit=%d)",
                current_session_id,
                pending,
                limit,
            )
            return False
        return True

    def _build_metadata(self, context: grpc.ServicerContext) -> Dict[str, str | bytes]:
        return {k.lower(): v for (k, v) in context.invocation_metadata()}

    def _apply_metadata_session_id(self, metadata: Dict[str, str | bytes]) -> None:
        metadata_session_id = metadata.get("session-id") or metadata.get("session_id")
        if metadata_session_id:
            if isinstance(metadata_session_id, bytes):
                try:
                    metadata_session_id = metadata_session_id.decode(
                        "utf-8", errors="ignore"
                    )
                except Exception:
                    metadata_session_id = None
            if metadata_session_id:
                set_session_id(str(metadata_session_id).strip())

    def _mark_activity(self, state: _StreamState) -> None:
        state.last_activity = time.monotonic()

    def _update_buffer_total(self, delta: int) -> None:
        if delta == 0:
            return
        with self._buffer_bytes_lock:
            self._buffer_bytes_total = max(0, self._buffer_bytes_total + delta)

    def _apply_global_buffer_limit(self, state: _StreamState, incoming_len: int) -> int:
        if incoming_len <= 0:
            return 0
        limit = self._config.max_total_buffer_bytes
        if not limit or limit <= 0:
            self._update_buffer_total(incoming_len)
            return incoming_len

        with self._buffer_bytes_lock:
            total = self._buffer_bytes_total
            overflow = total + incoming_len - limit
            if overflow <= 0:
                self._buffer_bytes_total = total + incoming_len
                return incoming_len

            drop_from_buffer = min(overflow, len(state.buffer))
            if drop_from_buffer > 0:
                del state.buffer[:drop_from_buffer]
                self._buffer_bytes_total = max(
                    0, self._buffer_bytes_total - drop_from_buffer
                )
                rate = state.sample_rate or self._config.default_sample_rate
                dropped_sec = audio.chunk_duration_seconds(drop_from_buffer, rate)
                state.buffer_start_sec += dropped_sec
                if state.buffer_start_time is not None:
                    state.buffer_start_time += dropped_sec
                overflow -= drop_from_buffer

            if overflow > 0:
                LOGGER.warning(
                    "Global buffer limit reached; dropping %d bytes of incoming audio",
                    overflow,
                )
            incoming_keep = max(0, incoming_len - overflow)
            self._buffer_bytes_total = max(0, self._buffer_bytes_total + incoming_keep)
            return incoming_keep

    def _clear_buffer(self, state: _StreamState) -> None:
        if state.buffer:
            self._update_buffer_total(-len(state.buffer))
            state.buffer = bytearray()
        state.buffer_start_time = None
        state.buffer_has_new_audio = False
        state.last_partial_decode_sec = None

    def _acquire_decode_slot(
        self,
        state: _StreamState,
        is_final: bool,
        context: grpc.ServicerContext,
    ) -> bool:
        limit = self._config.max_pending_decodes_global
        if not limit or limit <= 0:
            return True
        timeout = self._config.decode_queue_timeout_sec if is_final else 0.0
        acquired = self._decode_scheduler.acquire_pending_slot(
            block=is_final, timeout=timeout
        )
        if acquired:
            return True
        if not is_final:
            LOGGER.warning(
                "Global pending decode limit reached; dropping partial decode (session_id=%s)",
                state.session_state.session_id if state.session_state else "unknown",
            )
            return False
        LOGGER.error(
            "Global pending decode limit reached; aborting session (session_id=%s)",
            state.session_state.session_id if state.session_state else "unknown",
        )
        state.final_reason = "decode_backpressure"
        abort_with_error(context, ErrorCode.DECODE_TIMEOUT)
        return False

    def _schedule_decode(
        self,
        state: _StreamState,
        pcm: bytes,
        is_final: bool,
        offset_sec: float,
        count_vad: bool,
        buffer_started_at: Optional[float],
        context: grpc.ServicerContext,
    ) -> bool:
        if not state.decode_stream:
            return False
        if not self._acquire_decode_slot(state, is_final, context):
            return False
        state.decode_stream.schedule_decode(
            pcm,
            state.sample_rate or self._config.default_sample_rate,
            state.session_state.decode_options if state.session_state else {},
            is_final,
            offset_sec,
            count_vad=count_vad,
            buffer_started_at=buffer_started_at,
            holds_slot=True,
        )
        self._mark_activity(state)
        return True

    def _emit_with_activity(
        self, state: _StreamState, block: bool
    ) -> Iterator[stt_pb2.STTResult]:
        if not state.decode_stream:
            return
        self._mark_activity(state)
        for result in self._emit_results_with_session(
            state.decode_stream, block, state.session_state
        ):
            self._mark_activity(state)
            yield result

    def _watchdog_loop(self, state: _StreamState) -> None:
        while not state.stop_watchdog.is_set():
            if state.decode_stream and state.decode_stream.has_pending_results():
                self._mark_activity(state)
            # Calculate time elapsed since last activity
            elapsed = time.monotonic() - state.last_activity
            remaining = self._config.session_timeout_sec - elapsed

            if remaining <= 0:
                LOGGER.warning("Session timeout detected.")
                state.timeout_event.set()
                return

            # Wait for remaining time (wake up immediately if stop signal received)
            if state.stop_watchdog.wait(remaining):
                break

    def _start_watchdog(self, state: _StreamState) -> threading.Thread:
        thread = threading.Thread(
            target=lambda: self._watchdog_loop(state), daemon=True
        )
        thread.start()
        return thread

    def _cancel_pending_decodes(
        self, decode_stream: Optional[DecodeStream], session_id: Optional[str]
    ) -> None:
        if not decode_stream:
            return
        cancelled, running = decode_stream.cancel_pending()
        if cancelled:
            LOGGER.info(
                "Cancelled %d pending decodes for session_id=%s",
                cancelled,
                session_id or "unknown",
            )
        if running:
            LOGGER.info(
                "Pending decodes already running; cannot cancel (count=%d, session_id=%s)",
                running,
                session_id or "unknown",
            )

    def _handle_disconnect(self, state: _StreamState) -> None:
        if state.disconnect_event.is_set():
            return
        state.disconnect_event.set()
        current_session_id = (
            state.session_state.session_id if state.session_state else None
        )
        LOGGER.info(
            "Client disconnect callback received for session %s", current_session_id
        )
        self._cancel_pending_decodes(state.decode_stream, current_session_id)

    def _bootstrap_stream(
        self,
        state: _StreamState,
        metadata: Dict[str, str | bytes],
        context: grpc.ServicerContext,
    ) -> None:
        state.session_state = self._session_facade.resolve_from_metadata(
            metadata, context
        )
        if state.session_state:
            set_session_id(state.session_state.session_id)
            state.session_logged = self._log_session_start(state.session_state)
            state.vad_state = self._create_vad_state(state.session_state)
        state.decode_stream = self._decode_scheduler.new_stream()
        if state.session_state and state.decode_stream:
            state.decode_stream.set_session_id(state.session_state.session_id)
            state.decode_stream.set_model_id(state.session_state.session_info.model_id)

    def _handle_vad_trigger(
        self,
        state: _StreamState,
        vad_update: Any,
        context: grpc.ServicerContext,
    ) -> Iterator[stt_pb2.STTResult]:
        if not state.vad_state or not state.decode_stream or not state.session_state:
            return
        if not buffer_is_speech(state.buffer, self._config.speech_rms_threshold):
            LOGGER.info(
                "Skipping decode: chunk RMS %.4f below speech threshold %.4f",
                vad_update.chunk_rms,
                self._config.speech_rms_threshold,
            )
            LOGGER.info(
                "session_id=%s ignored low-energy buffer",
                state.session_state.session_id if state.session_state else "unknown",
            )
            self._clear_buffer(state)
            state.vad_state.reset_after_trigger()
            return
        self._on_vad_trigger()
        state.vad_count += 1
        self._on_vad_utterance_start()
        session_info = state.session_state.session_info
        is_final = session_info.vad_mode == stt_pb2.VAD_AUTO_END
        if state.disconnect_event.is_set() or state.timeout_event.is_set():
            LOGGER.info("Skipping decode due to shutdown signal.")
            state.final_reason = (
                "client_disconnect" if state.disconnect_event.is_set() else "timeout"
            )
            state.client_disconnected = state.disconnect_event.is_set()
            state.stop_stream = True
            return
        if not self._ensure_decode_capacity(
            state.decode_stream, is_final, state.session_state
        ):
            self._clear_buffer(state)
            state.vad_state.reset_after_trigger()
            return
        self._schedule_decode(
            state,
            bytes(state.buffer),
            is_final=is_final,
            offset_sec=state.buffer_start_sec,
            count_vad=True,
            buffer_started_at=state.buffer_start_time,
            context=context,
        )
        self._clear_buffer(state)
        LOGGER.info(
            "VAD count=%d for current session (pending=%d, mode=%s, active_vad=%d)",
            state.vad_count,
            state.decode_stream.pending_partial_decodes(),
            (
                "AUTO_END"
                if session_info.vad_mode == stt_pb2.VAD_AUTO_END
                else "CONTINUE"
            ),
            self._active_vad_utterances(),
        )
        if is_final:
            for result in self._emit_with_activity(state, False):
                yield result
            state.final_reason = "auto_vad_finalized"
            state.stop_stream = True
            return
        state.vad_state.reset_after_trigger()

    def _handle_final_chunk(
        self, state: _StreamState, context: grpc.ServicerContext
    ) -> Iterator[stt_pb2.STTResult]:
        if not state.decode_stream:
            return
        if state.buffer and state.buffer_has_new_audio:
            if state.disconnect_event.is_set() or state.timeout_event.is_set():
                LOGGER.info("Skipping final decode due to shutdown signal.")
                state.final_reason = (
                    "client_disconnect"
                    if state.disconnect_event.is_set()
                    else "timeout"
                )
                state.client_disconnected = state.disconnect_event.is_set()
                state.stop_stream = True
                return
            self._ensure_decode_capacity(state.decode_stream, True, state.session_state)
            self._schedule_decode(
                state,
                bytes(state.buffer),
                is_final=True,
                offset_sec=state.buffer_start_sec,
                count_vad=False,
                buffer_started_at=state.buffer_start_time,
                context=context,
            )
            self._clear_buffer(state)
        for result in self._emit_with_activity(state, False):
            yield result
        state.final_reason = "client_sent_final_chunk"
        state.stop_stream = True

    def _drain_pending_results(
        self, state: _StreamState, context: grpc.ServicerContext
    ) -> Iterator[stt_pb2.STTResult]:
        if state.timeout_event.is_set():
            LOGGER.info("Stopping stream due to timeout signal.")
            state.final_reason = "timeout"
            abort_with_error(context, ErrorCode.SESSION_TIMEOUT)
        if state.decode_stream:
            if (
                not state.client_disconnected
                and state.buffer
                and state.buffer_has_new_audio
                and buffer_is_speech(state.buffer, self._config.speech_rms_threshold)
            ):
                self._ensure_decode_capacity(
                    state.decode_stream, True, state.session_state
                )
                if self._schedule_decode(
                    state,
                    bytes(state.buffer),
                    is_final=True,
                    offset_sec=state.buffer_start_sec,
                    count_vad=False,
                    buffer_started_at=state.buffer_start_time,
                    context=context,
                ):
                    self._clear_buffer(state)
            state.buffer_start_time = None

            while True:
                if state.timeout_event.is_set():
                    LOGGER.info("Stopping stream due to timeout signal.")
                    state.final_reason = "timeout"
                    abort_with_error(context, ErrorCode.SESSION_TIMEOUT)
                emitted = list(
                    self._emit_with_activity(
                        state,
                        block=state.decode_stream.has_pending_results(),
                    )
                )
                if not emitted:
                    break
                for result in emitted:
                    yield result

    def _handle_chunk(
        self,
        state: _StreamState,
        chunk: stt_pb2.AudioChunk,
        context: grpc.ServicerContext,
    ) -> Iterator[stt_pb2.STTResult]:
        if state.disconnect_event.is_set():
            LOGGER.info("Stopping stream due to disconnect signal.")
            state.final_reason = "client_disconnect"
            state.client_disconnected = True
            self._clear_buffer(state)
            state.stop_stream = True
            return
        if state.timeout_event.is_set():
            LOGGER.info("Stopping stream due to timeout signal.")
            state.final_reason = "timeout"
            abort_with_error(context, ErrorCode.SESSION_TIMEOUT)

        # Update activity timestamp
        self._mark_activity(state)

        current_session_id = (
            state.session_state.session_id if state.session_state else None
        )
        if current_session_id:
            set_session_id(current_session_id)
        if not context.is_active():
            LOGGER.info("Client disconnected; stopping session %s", current_session_id)
            state.final_reason = "client_disconnect"
            state.client_disconnected = True
            self._cancel_pending_decodes(state.decode_stream, current_session_id)
            self._clear_buffer(state)
            state.stop_stream = True
            return
        if (
            chunk.session_id
            and current_session_id
            and chunk.session_id != current_session_id
        ):
            LOGGER.warning(
                "Received chunk with mismatched session_id=%s (expected %s)",
                chunk.session_id,
                current_session_id,
            )
            return

        if state.session_state is None:
            state.session_state = self._session_facade.ensure_session_from_chunk(
                state.session_state, chunk, context
            )

        if state.session_state and state.decode_stream:
            state.decode_stream.set_session_id(state.session_state.session_id)
            set_session_id(state.session_state.session_id)
        if state.session_state and not state.session_logged:
            state.session_logged = self._log_session_start(state.session_state)
        if state.vad_state is None:
            state.vad_state = self._create_vad_state(state.session_state)

        self._session_facade.validate_token(state.session_state, chunk, context)

        if state.vad_state is None:
            # Should not happen, but guard against update before initialization.
            state.vad_state = self._create_vad_state(state.session_state)

        state.sample_rate = (
            chunk.sample_rate
            if chunk.sample_rate > 0
            else state.sample_rate or self._config.default_sample_rate
        )
        max_chunk_bytes = self._max_chunk_bytes(state.sample_rate)
        if max_chunk_bytes is not None and len(chunk.pcm16) > max_chunk_bytes:
            LOGGER.warning(
                "Chunk size exceeds limit (bytes=%d max=%d session_id=%s)",
                len(chunk.pcm16),
                max_chunk_bytes,
                state.session_state.session_id if state.session_state else "unknown",
            )
            abort_with_error(
                context,
                ErrorCode.AUDIO_CHUNK_TOO_LARGE,
                detail=f"chunk bytes {len(chunk.pcm16)} exceeds max {max_chunk_bytes}",
            )
        state.audio_recorder = self._capture_audio_chunk(
            state.audio_recorder,
            state.session_state,
            state.sample_rate,
            chunk.pcm16,
        )

        if not state.buffer and chunk.pcm16:
            state.buffer_start_sec = state.audio_received_sec
            state.buffer_start_time = time.perf_counter()
        incoming = chunk.pcm16
        incoming_len = len(incoming)
        if incoming_len:
            allowed = self._apply_global_buffer_limit(state, incoming_len)
            if allowed < incoming_len:
                incoming = incoming[-allowed:] if allowed > 0 else b""
        if incoming:
            state.buffer.extend(incoming)
            state.buffer_has_new_audio = True
        elif not state.buffer:
            state.buffer_start_time = None
        state.audio_received_sec += audio.chunk_duration_seconds(
            len(chunk.pcm16), state.sample_rate
        )
        vad_update = state.vad_state.update(chunk.pcm16, state.sample_rate)

        if vad_update.triggered:
            for result in self._handle_vad_trigger(state, vad_update, context):
                yield result
            if state.stop_stream:
                return
        else:
            self._maybe_schedule_periodic_partial(state, vad_update, context)

        if not chunk.is_final:
            if state.disconnect_event.is_set() or state.timeout_event.is_set():
                LOGGER.info("Skipping buffer management due to shutdown signal.")
                state.final_reason = (
                    "client_disconnect"
                    if state.disconnect_event.is_set()
                    else "timeout"
                )
                state.client_disconnected = state.disconnect_event.is_set()
                state.stop_stream = True
                return
            self._enforce_buffer_limit(state, context)

        for result in self._emit_with_activity(state, False):
            yield result

        if chunk.is_final:
            for result in self._handle_final_chunk(state, context):
                yield result

    def _finalize_stream(
        self, state: _StreamState, context: grpc.ServicerContext
    ) -> None:
        # Send termination signal to stop watchdog cleanly
        state.stop_watchdog.set()

        if state.timeout_event.is_set():
            state.final_reason = "timeout"

        if state.vad_state:
            state.vad_state.close()

        if state.decode_stream:
            (
                buffer_wait_total,
                queue_wait_total,
                inference_total,
                response_emit_total,
                decode_count,
            ) = state.decode_stream.timing_summary()
            try:
                # Decode timing totals per stream (accumulated across decode tasks):
                # - buffer_wait: time spent buffering audio before scheduling decode
                # - queue_wait: time waiting for a worker after scheduling decode
                # - inference: model execution time
                # - response_emit: time spent yielding results to the client
                # - total: sum of buffer_wait + queue_wait + inference + response_emit
                context.set_trailing_metadata(
                    (
                        (
                            "stt-decode-buffer-wait-sec",
                            f"{buffer_wait_total:.6f}",
                        ),
                        (
                            "stt-decode-queue-wait-sec",
                            f"{queue_wait_total:.6f}",
                        ),
                        ("stt-decode-inference-sec", f"{inference_total:.6f}"),
                        (
                            "stt-decode-response-emit-sec",
                            f"{response_emit_total:.6f}",
                        ),
                        (
                            "stt-decode-total-sec",
                            f"{(buffer_wait_total + queue_wait_total + inference_total + response_emit_total):.6f}",
                        ),
                        ("stt-decode-count", str(decode_count)),
                    )
                )
            except Exception:
                pass

        if state.audio_recorder and self._audio_storage:
            self._audio_storage.finalize_recording(
                state.audio_recorder, state.final_reason
            )
        if state.buffer:
            self._update_buffer_total(-len(state.buffer))
            state.buffer = bytearray()
        if state.session_state:
            duration = time.monotonic() - state.session_start
            LOGGER.info(
                "Streaming finished for session_id=%s reason=%s vad_count=%d duration=%.2fs",
                state.session_state.session_id,
                state.final_reason,
                state.vad_count,
                duration,
            )
        self._session_facade.remove_session(
            state.session_state, reason=state.final_reason
        )
        clear_session_id()

    def run(
        self,
        request_iterator: Iterable[stt_pb2.AudioChunk],
        context: grpc.ServicerContext,
    ) -> Iterator[stt_pb2.STTResult]:
        state = _StreamState()
        metadata = self._build_metadata(context)
        self._apply_metadata_session_id(metadata)

        context.add_callback(lambda: self._handle_disconnect(state))
        self._start_watchdog(state)

        try:
            self._bootstrap_stream(state, metadata, context)
            for chunk in request_iterator:
                for result in self._handle_chunk(state, chunk, context):
                    yield result
                if state.stop_stream:
                    break

            for result in self._drain_pending_results(state, context):
                yield result

        except Exception:
            # Handle errors occurring during timeout abort
            if state.timeout_event.is_set():
                state.final_reason = "timeout"
            else:
                raise

        finally:
            self._finalize_stream(state, context)

    def _create_vad_state(self, state: SessionState) -> VADGate:
        info = state.session_info
        silence = info.vad_silence
        if silence <= 0:
            silence = self._config.vad_silence
        threshold = info.vad_threshold
        if threshold < 0:
            threshold = self._config.vad_threshold
        return VADGate(threshold, silence)

    def _capture_audio_chunk(
        self,
        recorder: Optional[SessionAudioRecorder],
        session_state: Optional[SessionState],
        sample_rate: Optional[int],
        pcm16: bytes,
    ) -> Optional[SessionAudioRecorder]:
        if (
            self._audio_storage is None
            or session_state is None
            or not pcm16
            or sample_rate is None
        ):
            return recorder
        if recorder is None:
            effective_rate = sample_rate or self._config.default_sample_rate
            recorder = self._audio_storage.start_recording(
                session_state.session_id, effective_rate
            )
        recorder.append(pcm16)
        return recorder

    def _buffer_limit_bytes(self, sample_rate: Optional[int]) -> Optional[int]:
        limit_bytes: Optional[int] = None
        max_bytes = self._config.max_buffer_bytes
        if max_bytes is not None and max_bytes > 0:
            limit_bytes = int(max_bytes)
        max_sec = self._config.max_buffer_sec
        if max_sec is not None and max_sec > 0:
            rate = sample_rate or self._config.default_sample_rate
            sec_limit = int(max_sec * rate * 2)
            if sec_limit > 0:
                limit_bytes = (
                    sec_limit if limit_bytes is None else min(limit_bytes, sec_limit)
                )
        return limit_bytes

    def _partial_decode_window_bytes(self, sample_rate: Optional[int]) -> Optional[int]:
        window_sec = self._config.partial_decode_window_sec
        if window_sec is None or window_sec <= 0:
            return None
        rate = sample_rate or self._config.default_sample_rate
        if rate <= 0:
            return None
        return max(1, int(window_sec * rate * 2))

    def _maybe_schedule_periodic_partial(
        self,
        state: _StreamState,
        vad_update: Any,
        context: grpc.ServicerContext,
    ) -> None:
        interval = self._config.partial_decode_interval_sec
        if interval is None or interval <= 0:
            return
        if not state.session_state or not state.decode_stream:
            return
        if state.session_state.session_info.vad_mode != stt_pb2.VAD_CONTINUE:
            return
        if not vad_update.speech_active or not state.buffer:
            return
        limit_bytes = self._buffer_limit_bytes(state.sample_rate)
        if limit_bytes is not None and len(state.buffer) > limit_bytes:
            return
        if state.disconnect_event.is_set() or state.timeout_event.is_set():
            return
        if not buffer_is_speech(state.buffer, self._config.speech_rms_threshold):
            return
        current_sec = state.audio_received_sec
        if state.last_partial_decode_sec is None:
            if current_sec - state.buffer_start_sec < interval:
                return
        elif current_sec - state.last_partial_decode_sec < interval:
            return
        if not self._ensure_decode_capacity(
            state.decode_stream, False, state.session_state
        ):
            return
        rate = state.sample_rate or self._config.default_sample_rate
        window_bytes = self._partial_decode_window_bytes(state.sample_rate)
        buffer_bytes = state.buffer
        offset_sec = state.buffer_start_sec
        if window_bytes is not None and len(buffer_bytes) > window_bytes:
            drop = len(buffer_bytes) - window_bytes
            offset_sec += audio.chunk_duration_seconds(drop, rate)
            pcm = bytes(buffer_bytes[-window_bytes:])
        else:
            pcm = bytes(buffer_bytes)
        if self._schedule_decode(
            state,
            pcm,
            is_final=False,
            offset_sec=offset_sec,
            count_vad=False,
            buffer_started_at=state.buffer_start_time,
            context=context,
        ):
            state.last_partial_decode_sec = current_sec

    def _max_chunk_bytes(self, sample_rate: Optional[int]) -> Optional[int]:
        max_ms = self._config.max_chunk_ms
        if max_ms is None or max_ms <= 0:
            return None
        rate = sample_rate or self._config.default_sample_rate
        if rate <= 0:
            return None
        return int((max_ms / 1000.0) * rate * 2)

    def _enforce_buffer_limit(
        self,
        state: _StreamState,
        context: grpc.ServicerContext,
    ) -> None:
        buffer = state.buffer
        limit_bytes = self._buffer_limit_bytes(state.sample_rate)
        if limit_bytes is None or len(buffer) <= limit_bytes:
            return

        if (
            state.session_state
            and state.decode_stream
            and state.session_state.session_info.vad_mode == stt_pb2.VAD_CONTINUE
        ):
            if not buffer_is_speech(buffer, self._config.speech_rms_threshold):
                LOGGER.info(
                    "Buffer limit reached with low-energy audio; dropping buffer."
                )
                self._clear_buffer(state)
                return
            LOGGER.warning(
                "Buffer limit reached (%d bytes); scheduling partial decode.",
                len(buffer),
            )
            if not self._ensure_decode_capacity(
                state.decode_stream, False, state.session_state
            ):
                self._clear_buffer(state)
                return
            rate = state.sample_rate or self._config.default_sample_rate
            window_drop = max(0, len(buffer) - limit_bytes)
            window_offset_sec = state.buffer_start_sec + audio.chunk_duration_seconds(
                window_drop, rate
            )
            window = bytes(buffer[-limit_bytes:])
            if not self._schedule_decode(
                state,
                window,
                is_final=False,
                offset_sec=window_offset_sec,
                count_vad=False,
                buffer_started_at=state.buffer_start_time,
                context=context,
            ):
                self._clear_buffer(state)
                return
            state.last_partial_decode_sec = state.audio_received_sec
            overlap_sec = max(0.0, self._config.buffer_overlap_sec)
            overlap_bytes = int(overlap_sec * rate * 2)
            retain = min(overlap_bytes, len(buffer))
            dropped = len(buffer) - retain
            if dropped > 0:
                new_buffer = bytearray(buffer[-retain:]) if retain > 0 else bytearray()
                dropped_sec = audio.chunk_duration_seconds(dropped, rate)
                state.buffer_start_sec += dropped_sec
                if state.buffer_start_time is not None:
                    state.buffer_start_time += dropped_sec
            else:
                new_buffer = bytearray()
            before_len = len(state.buffer)
            state.buffer = new_buffer
            self._update_buffer_total(len(state.buffer) - before_len)
            state.buffer_has_new_audio = False
            return

        before_len = len(buffer)
        overflow = len(buffer) - limit_bytes
        if overflow > 0:
            del buffer[:overflow]
            rate = state.sample_rate or self._config.default_sample_rate
            dropped_sec = audio.chunk_duration_seconds(overflow, rate)
            state.buffer_start_sec += dropped_sec
            if state.buffer_start_time is not None:
                state.buffer_start_time += dropped_sec
            LOGGER.warning(
                "Buffer limit reached (%d bytes); trimmed %.2fs of audio.",
                limit_bytes,
                dropped_sec,
            )
        after_len = len(buffer)
        if after_len != before_len:
            self._update_buffer_total(after_len - before_len)
        state.buffer = buffer

    def _emit_results_with_session(
        self,
        decode_stream: DecodeStream,
        block: bool,
        session_state: Optional[SessionState],
    ) -> Iterator[stt_pb2.STTResult]:
        if session_state and decode_stream.session_id != session_state.session_id:
            decode_stream.set_session_id(session_state.session_id)
        yield from decode_stream.emit_ready(block)

    def _log_session_start(self, state: SessionState) -> bool:
        info = state.session_info
        LOGGER.info(
            "Streaming started for session_id=%s vad_mode=%s decode_profile=%s vad_silence=%.3f vad_threshold=%.4f model_id=%s",
            state.session_id,
            "AUTO_END" if info.vad_mode == stt_pb2.VAD_AUTO_END else "CONTINUE",
            info.decode_profile,
            info.vad_silence,
            info.vad_threshold,
            info.model_id,
        )
        return True
