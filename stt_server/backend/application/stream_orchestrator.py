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
from stt_server.backend.component.vad_gate import VADGate, buffer_is_speech
from stt_server.config.languages import SupportedLanguages
from stt_server.errors import ErrorCode, abort_with_error
from stt_server.utils import audio
from stt_server.utils.logger import LOGGER

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
    storage_max_bytes: Optional[int]
    storage_max_files: Optional[int]
    storage_max_age_days: Optional[int]

    # Buffer control settings
    max_buffer_sec: Optional[float] = 60.0
    max_buffer_bytes: Optional[int] = None


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
        if config.storage_enabled:
            storage_directory = Path(config.storage_directory).expanduser()
            storage_policy = AudioStorageConfig(
                enabled=True,
                directory=storage_directory,
                max_bytes=config.storage_max_bytes,
                max_files=config.storage_max_files,
                max_age_days=config.storage_max_age_days,
            )
            self._audio_storage = AudioStorageManager(storage_policy)
            LOGGER.info(
                "Audio storage enabled directory=%s max_bytes=%s max_files=%s max_age_days=%s",
                storage_directory,
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

    def _create_decode_scheduler(
        self, config: StreamOrchestratorConfig
    ) -> DecodeScheduler:
        return DecodeScheduler(
            self,
            decode_timeout_sec=config.decode_timeout_sec,
            language_lookup=config.language_lookup,
            hooks=self._hooks.decode_hooks,
        )

    def _on_vad_trigger(self) -> None:
        self._hooks.on_vad_trigger()

    def _on_vad_utterance_start(self) -> None:
        self._hooks.on_vad_utterance_start()

    def _active_vad_utterances(self) -> int:
        return self._hooks.active_vad_utterances()

    def run(
        self,
        request_iterator: Iterable[stt_pb2.AudioChunk],
        context: grpc.ServicerContext,
    ) -> Iterator[stt_pb2.STTResult]:
        session_state: Optional[SessionState] = None
        vad_state: Optional[VADGate] = None
        decode_stream = None
        metadata = {k.lower(): v for (k, v) in context.invocation_metadata()}
        session_logged = False
        final_reason = "stream_end"
        session_start = time.monotonic()
        vad_count = 0
        audio_recorder: Optional[SessionAudioRecorder] = None
        audio_received_sec = 0.0
        buffer_start_sec = 0.0
        buffer_start_time: Optional[float] = None
        client_disconnected = False

        # Initialize Watchdog variables
        last_activity = time.monotonic()
        stop_watchdog = threading.Event()
        timeout_event = threading.Event()
        disconnect_event = threading.Event()

        def watchdog_loop():
            while not stop_watchdog.is_set():
                # Calculate time elapsed since last activity
                elapsed = time.monotonic() - last_activity
                remaining = self._config.session_timeout_sec - elapsed

                if remaining <= 0:
                    LOGGER.warning("Session timeout detected.")
                    timeout_event.set()
                    return

                # Wait for remaining time (wake up immediately if stop signal received)
                if stop_watchdog.wait(remaining):
                    break

        def on_disconnect() -> None:
            if disconnect_event.is_set():
                return
            disconnect_event.set()
            current_session_id = session_state.session_id if session_state else None
            LOGGER.info(
                "Client disconnect callback received for session %s", current_session_id
            )
            if decode_stream:
                cancelled, running = decode_stream.cancel_pending()
                if cancelled:
                    LOGGER.info(
                        "Cancelled %d pending decodes for session_id=%s",
                        cancelled,
                        current_session_id or "unknown",
                    )
                if running:
                    LOGGER.info(
                        "Pending decodes already running; cannot cancel (count=%d, session_id=%s)",
                        running,
                        current_session_id or "unknown",
                    )

        context.add_callback(on_disconnect)

        # Start Watchdog thread (run as daemon thread)
        watchdog_thread = threading.Thread(target=watchdog_loop, daemon=True)
        watchdog_thread.start()

        try:
            session_state = self._session_facade.resolve_from_metadata(
                metadata, context
            )
            if session_state:
                session_logged = self._log_session_start(session_state)
                vad_state = self._create_vad_state(session_state)
            decode_stream = self._decode_scheduler.new_stream()
            if session_state and decode_stream:
                decode_stream.set_session_id(session_state.session_id)
                decode_stream.set_model_id(session_state.session_info.model_id)

            buffer = bytearray()
            sample_rate: Optional[int] = None

            for chunk in request_iterator:
                if disconnect_event.is_set():
                    LOGGER.info("Stopping stream due to disconnect signal.")
                    final_reason = "client_disconnect"
                    client_disconnected = True
                    buffer = bytearray()
                    buffer_start_time = None
                    break
                # Check for timeout signal at the start of the loop
                if timeout_event.is_set():
                    LOGGER.info("Stopping stream due to timeout signal.")
                    final_reason = "timeout"
                    abort_with_error(context, ErrorCode.SESSION_TIMEOUT)

                # Update activity timestamp
                last_activity = time.monotonic()

                current_session_id = session_state.session_id if session_state else None
                if not context.is_active():
                    LOGGER.info(
                        "Client disconnected; stopping session %s", current_session_id
                    )
                    final_reason = "client_disconnect"
                    client_disconnected = True
                    if decode_stream:
                        cancelled, running = decode_stream.cancel_pending()
                        if cancelled:
                            LOGGER.info(
                                "Cancelled %d pending decodes for session_id=%s",
                                cancelled,
                                current_session_id or "unknown",
                            )
                        if running:
                            LOGGER.info(
                                "Pending decodes already running; cannot cancel (count=%d, session_id=%s)",
                                running,
                                current_session_id or "unknown",
                            )
                    buffer = bytearray()
                    buffer_start_time = None
                    break
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
                    continue

                if session_state is None:
                    session_state = self._session_facade.ensure_session_from_chunk(
                        session_state, chunk, context
                    )

                if session_state and decode_stream:
                    decode_stream.set_session_id(session_state.session_id)
                if session_state and not session_logged:
                    session_logged = self._log_session_start(session_state)
                if vad_state is None:
                    vad_state = self._create_vad_state(session_state)

                self._session_facade.validate_token(session_state, chunk, context)

                if vad_state is None:
                    # Should not happen, but guard against update before initialization.
                    vad_state = self._create_vad_state(session_state)

                sample_rate = (
                    chunk.sample_rate
                    if chunk.sample_rate > 0
                    else sample_rate or self._config.default_sample_rate
                )
                audio_recorder = self._capture_audio_chunk(
                    audio_recorder,
                    session_state,
                    sample_rate,
                    chunk.pcm16,
                )

                if not buffer and chunk.pcm16:
                    buffer_start_sec = audio_received_sec
                    buffer_start_time = time.perf_counter()
                buffer.extend(chunk.pcm16)
                audio_received_sec += audio.chunk_duration_seconds(
                    len(chunk.pcm16), sample_rate
                )
                vad_update = vad_state.update(chunk.pcm16, sample_rate)

                if vad_update.triggered:
                    if not buffer_is_speech(buffer, self._config.speech_rms_threshold):
                        LOGGER.info(
                            "Skipping decode: chunk RMS %.4f below speech threshold %.4f",
                            vad_update.chunk_rms,
                            self._config.speech_rms_threshold,
                        )
                        LOGGER.info(
                            "session_id=%s ignored low-energy buffer",
                            session_state.session_id if session_state else "unknown",
                        )
                        buffer = bytearray()
                        buffer_start_time = None
                        vad_state.reset_after_trigger()
                        continue
                    self._on_vad_trigger()
                    vad_count += 1
                    self._on_vad_utterance_start()
                    session_info = session_state.session_info
                    is_final = session_info.vad_mode == stt_pb2.VAD_AUTO_END
                    decode_stream.schedule_decode(
                        bytes(buffer),
                        sample_rate,
                        session_state.decode_options,
                        is_final,
                        buffer_start_sec,
                        count_vad=True,
                        buffer_started_at=buffer_start_time,
                    )
                    buffer = bytearray()
                    buffer_start_time = None
                    LOGGER.info(
                        "VAD count=%d for current session (pending=%d, mode=%s, active_vad=%d)",
                        vad_count,
                        decode_stream.pending_partial_decodes(),
                        (
                            "AUTO_END"
                            if session_info.vad_mode == stt_pb2.VAD_AUTO_END
                            else "CONTINUE"
                        ),
                        self._active_vad_utterances(),
                    )
                    if is_final:
                        yield from self._emit_results_with_session(
                            decode_stream, False, session_state
                        )
                        final_reason = "auto_vad_finalized"
                        break
                    vad_state.reset_after_trigger()

                if not chunk.is_final:
                    buffer, buffer_start_sec, buffer_start_time = (
                        self._enforce_buffer_limit(
                            buffer,
                            buffer_start_sec,
                            buffer_start_time,
                            audio_received_sec,
                            sample_rate,
                            session_state,
                            decode_stream,
                        )
                    )

                yield from self._emit_results_with_session(
                    decode_stream, False, session_state
                )

                if chunk.is_final:
                    if buffer:
                        decode_stream.schedule_decode(
                            bytes(buffer),
                            sample_rate,
                            session_state.decode_options,
                            True,
                            buffer_start_sec,
                            count_vad=False,
                            buffer_started_at=buffer_start_time,
                        )
                        buffer = bytearray()
                        buffer_start_time = None
                    yield from self._emit_results_with_session(
                        decode_stream, False, session_state
                    )
                    final_reason = "client_sent_final_chunk"
                    break

            # Process remaining buffer after loop ends
            if timeout_event.is_set():
                LOGGER.info("Stopping stream due to timeout signal.")
                final_reason = "timeout"
                abort_with_error(context, ErrorCode.SESSION_TIMEOUT)
            if decode_stream:
                if (
                    not client_disconnected
                    and buffer
                    and buffer_is_speech(buffer, self._config.speech_rms_threshold)
                ):
                    decode_stream.schedule_decode(
                        bytes(buffer),
                        sample_rate or self._config.default_sample_rate,
                        session_state.decode_options if session_state else {},
                        True,
                        buffer_start_sec,
                        count_vad=False,
                        buffer_started_at=buffer_start_time,
                    )
                buffer_start_time = None

                while True:
                    if timeout_event.is_set():
                        LOGGER.info("Stopping stream due to timeout signal.")
                        final_reason = "timeout"
                        abort_with_error(context, ErrorCode.SESSION_TIMEOUT)
                    emitted = list(
                        self._emit_results_with_session(
                            decode_stream,
                            block=decode_stream.has_pending_results(),
                            session_state=session_state,
                        )
                    )
                    if not emitted:
                        break
                    for result in emitted:
                        yield result

        except Exception as e:
            # Handle errors occurring during timeout abort
            if timeout_event.is_set():
                final_reason = "timeout"
            else:
                raise e

        finally:
            # Send termination signal to stop watchdog cleanly
            stop_watchdog.set()

            if timeout_event.is_set():
                final_reason = "timeout"

            if decode_stream:
                (
                    buffer_wait_total,
                    queue_wait_total,
                    inference_total,
                    response_emit_total,
                    decode_count,
                ) = decode_stream.timing_summary()
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

            if audio_recorder and self._audio_storage:
                self._audio_storage.finalize_recording(audio_recorder, final_reason)
            if session_state:
                duration = time.monotonic() - session_start
                LOGGER.info(
                    "Streaming finished for session_id=%s reason=%s vad_count=%d duration=%.2fs",
                    session_state.session_id,
                    final_reason,
                    vad_count,
                    duration,
                )
            self._session_facade.remove_session(session_state, reason=final_reason)

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
            decode_stream.schedule_decode(
                bytes(buffer),
                sample_rate or self._config.default_sample_rate,
                session_state.decode_options,
                False,
                buffer_start_sec,
                count_vad=False,
                buffer_started_at=buffer_start_time,
            )
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
