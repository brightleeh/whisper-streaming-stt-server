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

    # Buffer control settings
    max_buffer_sec: Optional[float] = 60.0
    max_buffer_bytes: Optional[int] = None
    max_pending_decodes_per_stream: int = 8

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
        configure_vad_model_pool(config.vad_model_pool_size, config.vad_model_prewarm)
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
            state.buffer = bytearray()
            state.buffer_start_time = None
            state.vad_state.reset_after_trigger()
            return
        self._on_vad_trigger()
        state.vad_count += 1
        self._on_vad_utterance_start()
        session_info = state.session_state.session_info
        is_final = session_info.vad_mode == stt_pb2.VAD_AUTO_END
        effective_rate = state.sample_rate or self._config.default_sample_rate
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
            state.buffer = bytearray()
            state.buffer_start_time = None
            state.vad_state.reset_after_trigger()
            return
        state.decode_stream.schedule_decode(
            bytes(state.buffer),
            effective_rate,
            state.session_state.decode_options,
            is_final,
            state.buffer_start_sec,
            count_vad=True,
            buffer_started_at=state.buffer_start_time,
        )
        self._mark_activity(state)
        state.buffer = bytearray()
        state.buffer_start_time = None
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
        if state.buffer:
            effective_rate = state.sample_rate or self._config.default_sample_rate
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
            state.decode_stream.schedule_decode(
                bytes(state.buffer),
                effective_rate,
                state.session_state.decode_options if state.session_state else {},
                True,
                state.buffer_start_sec,
                count_vad=False,
                buffer_started_at=state.buffer_start_time,
            )
            self._mark_activity(state)
            state.buffer = bytearray()
            state.buffer_start_time = None
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
                and buffer_is_speech(state.buffer, self._config.speech_rms_threshold)
            ):
                self._ensure_decode_capacity(
                    state.decode_stream, True, state.session_state
                )
                state.decode_stream.schedule_decode(
                    bytes(state.buffer),
                    state.sample_rate or self._config.default_sample_rate,
                    state.session_state.decode_options if state.session_state else {},
                    True,
                    state.buffer_start_sec,
                    count_vad=False,
                    buffer_started_at=state.buffer_start_time,
                )
                self._mark_activity(state)
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
            state.buffer = bytearray()
            state.buffer_start_time = None
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
            state.buffer = bytearray()
            state.buffer_start_time = None
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
        state.audio_recorder = self._capture_audio_chunk(
            state.audio_recorder,
            state.session_state,
            state.sample_rate,
            chunk.pcm16,
        )

        if not state.buffer and chunk.pcm16:
            state.buffer_start_sec = state.audio_received_sec
            state.buffer_start_time = time.perf_counter()
        state.buffer.extend(chunk.pcm16)
        state.audio_received_sec += audio.chunk_duration_seconds(
            len(chunk.pcm16), state.sample_rate
        )
        vad_update = state.vad_state.update(chunk.pcm16, state.sample_rate)

        if vad_update.triggered:
            for result in self._handle_vad_trigger(state, vad_update):
                yield result
            if state.stop_stream:
                return

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
            (
                state.buffer,
                state.buffer_start_sec,
                state.buffer_start_time,
            ) = self._enforce_buffer_limit(
                state.buffer,
                state.buffer_start_sec,
                state.buffer_start_time,
                state.audio_received_sec,
                state.sample_rate,
                state.session_state,
                state.decode_stream,
                activity_callback=lambda: self._mark_activity(state),
            )

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

        except Exception as e:
            # Handle errors occurring during timeout abort
            if state.timeout_event.is_set():
                state.final_reason = "timeout"
            else:
                raise e

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

    def _enforce_buffer_limit(
        self,
        buffer: bytearray,
        buffer_start_sec: float,
        buffer_start_time: Optional[float],
        audio_received_sec: float,
        sample_rate: Optional[int],
        session_state: Optional[SessionState],
        decode_stream: Optional[DecodeStream],
        activity_callback: Optional[Callable[[], None]] = None,
    ) -> tuple[bytearray, float, Optional[float]]:
        limit_bytes = self._buffer_limit_bytes(sample_rate)
        if limit_bytes is None or len(buffer) <= limit_bytes:
            return buffer, buffer_start_sec, buffer_start_time

        if (
            session_state
            and decode_stream
            and session_state.session_info.vad_mode == stt_pb2.VAD_CONTINUE
        ):
            if not buffer_is_speech(buffer, self._config.speech_rms_threshold):
                LOGGER.info(
                    "Buffer limit reached with low-energy audio; dropping buffer."
                )
                return bytearray(), audio_received_sec, None
            LOGGER.warning(
                "Buffer limit reached (%d bytes); scheduling partial decode.",
                len(buffer),
            )
            if self._ensure_decode_capacity(decode_stream, False, session_state):
                decode_stream.schedule_decode(
                    bytes(buffer),
                    sample_rate or self._config.default_sample_rate,
                    session_state.decode_options,
                    False,
                    buffer_start_sec,
                    count_vad=False,
                    buffer_started_at=buffer_start_time,
                )
                if activity_callback:
                    activity_callback()
            return bytearray(), audio_received_sec, None

        overflow = len(buffer) - limit_bytes
        if overflow > 0:
            del buffer[:overflow]
            rate = sample_rate or self._config.default_sample_rate
            dropped_sec = audio.chunk_duration_seconds(overflow, rate)
            buffer_start_sec += dropped_sec
            if buffer_start_time is not None:
                buffer_start_time += dropped_sec
            LOGGER.warning(
                "Buffer limit reached (%d bytes); trimmed %.2fs of audio.",
                limit_bytes,
                dropped_sec,
            )
        return buffer, buffer_start_sec, buffer_start_time

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
