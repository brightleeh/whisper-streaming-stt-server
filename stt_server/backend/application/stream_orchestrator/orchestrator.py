"""Streaming orchestrator implementation."""

from __future__ import annotations

import threading
import time
from concurrent import futures
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, Optional

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
    DecodeStream,
)
from stt_server.backend.component.vad_gate import (
    VADGate,
    buffer_is_speech,
    configure_vad_model_pool,
    reserve_vad_slot,
)
from stt_server.backend.utils.rate_limit import KeyedRateLimiter
from stt_server.errors import ErrorCode, abort_with_error
from stt_server.utils import audio
from stt_server.utils.logger import LOGGER, clear_session_id, set_session_id

from .flow import (
    drain_pending_results,
    handle_chunk,
    handle_final_chunk,
    handle_vad_trigger,
    step_drain,
    step_streaming,
    step_streaming_buffer,
    step_streaming_emit,
    step_streaming_vad,
)
from .helpers import (
    build_partial_decode_window,
    emit_results_with_session,
    log_session_start,
    should_attempt_periodic_partial,
)
from .types import (
    FlowActivityOps,
    FlowAudioOps,
    FlowBufferOps,
    FlowDecodeOps,
    FlowHooksOps,
    FlowLimitOps,
    FlowSessionOps,
    StreamFlowContext,
    StreamOrchestratorConfig,
    StreamOrchestratorHooks,
    StreamPhase,
    StreamSettings,
    _AudioBufferManager,
    _StreamState,
)

if TYPE_CHECKING:
    from stt_server.backend.application.model_registry import ModelWorkerProtocol


VAD_CONTINUE = stt_pb2.VAD_CONTINUE
VAD_AUTO_END = stt_pb2.VAD_AUTO_END


class StreamOrchestrator:
    """Executes the streaming recognition loop for the gRPC servicer."""

    def __init__(
        self,
        session_facade: SessionFacade,
        model_registry: ModelRegistry,
        config: StreamOrchestratorConfig,
        hooks: StreamOrchestratorHooks | None = None,
    ) -> None:
        """Initialize orchestrator dependencies and configure internal pools."""
        self._session_facade = session_facade
        self._model_registry = model_registry
        self._config = config
        self._hooks = hooks or StreamOrchestratorHooks()
        self._decode_scheduler = self._create_decode_scheduler(config)
        self._audio_storage: Optional[AudioStorageManager] = None
        self._buffer_manager = _AudioBufferManager(config)
        self._stream_rate_limiters: Dict[str, Optional[KeyedRateLimiter]] = {}
        self._configure_stream_rate_limiters(config.stream)
        configure_vad_model_pool(
            config.vad_pool.size,
            config.vad_pool.prewarm,
            config.vad_pool.max_size,
            config.vad_pool.growth_factor,
        )
        if config.storage.enabled:
            storage_directory = Path(config.storage.directory).expanduser()
            storage_policy = AudioStorageConfig(
                enabled=True,
                directory=storage_directory,
                queue_max_chunks=config.storage.queue_max_chunks,
                max_bytes=config.storage.max_bytes,
                max_files=config.storage.max_files,
                max_age_days=config.storage.max_age_days,
            )
            self._audio_storage = AudioStorageManager(storage_policy)
            LOGGER.info(
                "Audio storage enabled directory=%s queue_max_chunks=%s max_bytes=%s "
                "max_files=%s max_age_days=%s",
                storage_directory,
                config.storage.queue_max_chunks,
                config.storage.max_bytes,
                config.storage.max_files,
                config.storage.max_age_days,
            )
        # Flow context is created on demand to keep instance attributes lean.

    @property
    def decode_scheduler(self) -> DecodeScheduler:
        """Return the decode scheduler for this orchestrator."""
        return self._decode_scheduler

    @property
    def config(self) -> StreamOrchestratorConfig:
        """Expose the orchestrator configuration."""
        return self._config

    def load_model(self, model_id: str, config: Dict[str, Any]) -> None:
        """Load a model into the registry."""
        self._model_registry.load_model(model_id, config)

    def acquire_worker(self, model_id: str) -> ModelWorkerProtocol:
        """Acquire an available worker for the given model id."""
        worker = self._model_registry.get_worker(model_id)
        if not worker:
            raise RuntimeError(f"No worker available for model_id='{model_id}'")
        return worker

    def submit_decode(
        self,
        model_id: str,
        session_id: str,
        pcm: bytes,
        sample_rate: int,
        decode_options: Optional[Dict[str, Any]],
        is_final: bool,
    ) -> futures.Future:
        """Submit a decode request to the shared queue."""
        return self._model_registry.submit_decode(
            model_id, session_id, pcm, sample_rate, decode_options, is_final
        )

    @property
    def model_registry(self) -> ModelRegistry:
        """Expose the underlying model registry."""
        return self._model_registry

    def _create_decode_scheduler(
        self, config: StreamOrchestratorConfig
    ) -> DecodeScheduler:
        return DecodeScheduler(
            self,
            decode_timeout_sec=config.stream.decode_timeout_sec,
            language_lookup=config.stream.language_lookup,
            max_pending_decodes_global=config.decode_queue.max_pending_decodes_global,
            health_window_sec=config.health.window_sec,
            health_min_events=config.health.min_events,
            health_max_timeout_ratio=config.health.max_timeout_ratio,
            health_min_success_ratio=config.health.min_success_ratio,
            log_transcripts=config.stream.log_transcripts,
            hooks=self._hooks.decode_hooks,
        )

    def _flow(self) -> StreamFlowContext:
        return StreamFlowContext(
            config=self._config,
            decode=FlowDecodeOps(
                ensure_capacity=self._ensure_decode_capacity,
                schedule=self._schedule_decode,
                emit_with_activity=self._emit_with_activity,
                maybe_schedule_periodic_partial=self._maybe_schedule_periodic_partial,
                cancel_pending_decodes=self._cancel_pending_decodes,
            ),
            buffer=FlowBufferOps(
                clear=self._clear_buffer,
                enforce_limit=self._enforce_buffer_limit,
                apply_global_limit=self._apply_global_buffer_limit,
            ),
            session=FlowSessionOps(
                ensure_session_from_chunk=self._session_facade.ensure_session_from_chunk,
                validate_token=self._session_facade.validate_token,
                create_vad_state=self._create_vad_state,
                step_init=self._step_init,
            ),
            hooks=FlowHooksOps(
                on_vad_trigger=self._on_vad_trigger,
                on_vad_utterance_start=self._on_vad_utterance_start,
                active_vad_utterances=self._active_vad_utterances,
            ),
            audio=FlowAudioOps(
                capture_audio_chunk=self._capture_audio_chunk,
                max_chunk_bytes=self._max_chunk_bytes,
            ),
            activity=FlowActivityOps(
                mark=self._mark_activity,
            ),
            limits=FlowLimitOps(
                enforce_chunk=self._enforce_stream_limits,
            ),
        )

    def _on_vad_trigger(self) -> None:
        self._hooks.on_vad_trigger()

    def _on_vad_utterance_start(self) -> None:
        self._hooks.on_vad_utterance_start()

    def _active_vad_utterances(self) -> int:
        return self._hooks.active_vad_utterances()

    def _stream_rate_key(self, session_state: SessionState) -> str:
        info = session_state.session_info
        if info.api_key:
            return f"api:{info.api_key}"
        if info.client_ip:
            return f"ip:{info.client_ip}"
        return f"session:{session_state.session_id}"

    def _configure_stream_rate_limiters(self, stream_config: StreamSettings) -> None:
        realtime_limit = stream_config.max_audio_bytes_per_sec_realtime
        realtime_burst = stream_config.max_audio_bytes_per_sec_burst_realtime
        if realtime_limit is None:
            realtime_limit = stream_config.max_audio_bytes_per_sec
        if realtime_burst is None:
            realtime_burst = stream_config.max_audio_bytes_per_sec_burst

        batch_limit = stream_config.max_audio_bytes_per_sec_batch
        batch_burst = stream_config.max_audio_bytes_per_sec_burst_batch
        if batch_limit is None:
            batch_limit = stream_config.max_audio_bytes_per_sec
        if batch_burst is None:
            batch_burst = stream_config.max_audio_bytes_per_sec_burst

        self._stream_rate_limiters["realtime"] = (
            KeyedRateLimiter(realtime_limit, realtime_burst or None)
            if realtime_limit and realtime_limit > 0
            else None
        )
        self._stream_rate_limiters["batch"] = (
            KeyedRateLimiter(batch_limit, batch_burst or None)
            if batch_limit and batch_limit > 0
            else None
        )

    def _session_stream_mode(self, session_state: SessionState) -> str:
        mode = (
            session_state.session_info.attributes.get("upload_mode", "").strip().lower()
        )
        if mode in {"batch", "realtime"}:
            return mode
        return "realtime"

    def _stream_rate_limiter_for(
        self, session_state: SessionState
    ) -> Optional[KeyedRateLimiter]:
        mode = self._session_stream_mode(session_state)
        return self._stream_rate_limiters.get(mode)

    def _enforce_stream_limits(
        self,
        state: _StreamState,
        chunk: stt_pb2.AudioChunk,
        context: grpc.ServicerContext,
    ) -> None:
        if not state.session.session_state:
            return
        chunk_bytes = len(chunk.pcm16)
        if chunk_bytes <= 0:
            return
        limiter = self._stream_rate_limiter_for(state.session.session_state)
        if limiter:
            key = self._stream_rate_key(state.session.session_state)
            if not limiter.allow(key, amount=chunk_bytes):
                self._hooks.on_rate_limit_block("stream", key)
                LOGGER.warning(
                    "Stream rate limit exceeded (key=%s session_id=%s)",
                    key,
                    state.session.session_state.session_id,
                )
                abort_with_error(context, ErrorCode.STREAM_RATE_LIMITED)
        max_seconds = self._config.stream.max_audio_seconds_per_session
        if max_seconds and max_seconds > 0:
            sample_rate = (
                state.session.sample_rate or self._config.stream.default_sample_rate
            )
            next_total = (
                state.activity.audio_received_sec
                + audio.chunk_duration_seconds(chunk_bytes, sample_rate)
            )
            if next_total > max_seconds:
                LOGGER.warning(
                    "Stream audio limit exceeded (session_id=%s total=%.2f limit=%.2f)",
                    state.session.session_state.session_id,
                    next_total,
                    max_seconds,
                )
                abort_with_error(context, ErrorCode.STREAM_AUDIO_LIMIT_EXCEEDED)

    def _ensure_decode_capacity(
        self,
        decode_stream: Optional[DecodeStream],
        is_final: bool,
        session_state: Optional[SessionState],
    ) -> bool:
        if decode_stream is None:
            return False
        limit = self._config.decode_queue.max_pending_decodes_per_stream
        if limit <= 0:
            return True
        pending = decode_stream.pending_count()
        current_session_id = session_state.session_id if session_state else "unknown"
        if is_final:
            if pending >= limit:
                cancelled, orphaned = decode_stream.drop_pending_partials()
                dropped = cancelled + orphaned
                if dropped:
                    self._hooks.on_partial_drop(dropped)
                    LOGGER.warning(
                        "Dropped %d pending partial decodes for final decode (session_id=%s)",
                        dropped,
                        current_session_id,
                    )
            return True
        if pending < limit:
            return True
        cancelled, orphaned = decode_stream.drop_pending_partials(1)
        dropped = cancelled + orphaned
        if dropped:
            self._hooks.on_partial_drop(dropped)
        if decode_stream.pending_count() >= limit:
            LOGGER.warning(
                "Pending decode limit reached; dropping partial decode "
                "(session_id=%s pending=%d limit=%d)",
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
                except UnicodeDecodeError:
                    metadata_session_id = None
            if metadata_session_id:
                set_session_id(str(metadata_session_id).strip())

    def _mark_activity(self, state: _StreamState) -> None:
        state.activity.last_activity = time.monotonic()

    def _update_buffer_total(self, delta: int) -> None:
        self._buffer_manager.update_total(delta)
        self._hooks.on_buffer_total_bytes(self._buffer_manager.total_bytes())

    def _apply_global_buffer_limit(self, state: _StreamState, incoming_len: int) -> int:
        allowed = self._buffer_manager.apply_global_limit(state, incoming_len)
        self._hooks.on_buffer_total_bytes(self._buffer_manager.total_bytes())
        return allowed

    def _clear_buffer(self, state: _StreamState) -> None:
        self._buffer_manager.clear(state)

    def _acquire_decode_slot(
        self,
        state: _StreamState,
        is_final: bool,
        context: grpc.ServicerContext,
    ) -> bool:
        limit = self._config.decode_queue.max_pending_decodes_global
        if not limit or limit <= 0:
            return True
        timeout = (
            self._config.decode_queue.decode_queue_timeout_sec if is_final else 0.0
        )
        acquired = self._decode_scheduler.acquire_pending_slot(
            block=is_final, timeout=timeout
        )
        if acquired:
            return True
        session_id = (
            state.session.session_state.session_id
            if state.session.session_state
            else "unknown"
        )
        if not is_final:
            LOGGER.warning(
                "Global pending decode limit reached; dropping partial decode (session_id=%s)",
                session_id,
            )
            return False
        LOGGER.error(
            "Global pending decode limit reached; aborting session (session_id=%s)",
            session_id,
        )
        state.session.final_reason = "decode_backpressure"
        abort_with_error(context, ErrorCode.DECODE_TIMEOUT)

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
        if not state.decode.decode_stream:
            return False
        if not self._acquire_decode_slot(state, is_final, context):
            return False
        state.decode.decode_stream.schedule_decode(
            pcm,
            state.session.sample_rate or self._config.stream.default_sample_rate,
            (
                state.session.session_state.decode_options
                if state.session.session_state
                else {}
            ),
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
        if not state.decode.decode_stream:
            return
        self._mark_activity(state)
        for result in emit_results_with_session(
            state.decode.decode_stream, block, state.session.session_state
        ):
            self._mark_activity(state)
            yield result

    def _watchdog_loop(self, state: _StreamState) -> None:
        while not state.events.stop_watchdog.is_set():
            if state.events.processing_event.is_set():
                self._mark_activity(state)
            if (
                state.decode.decode_stream
                and state.decode.decode_stream.has_pending_results()
            ):
                self._mark_activity(state)
            # Calculate time elapsed since last activity
            elapsed = time.monotonic() - state.activity.last_activity
            remaining = self._config.stream.session_timeout_sec - elapsed

            if remaining <= 0:
                LOGGER.warning("Session timeout detected.")
                state.events.timeout_event.set()
                return

            # Wait for remaining time (wake up immediately if stop signal received)
            if state.events.stop_watchdog.wait(remaining):
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
        if state.events.disconnect_event.is_set():
            return
        state.events.disconnect_event.set()
        current_session_id = (
            state.session.session_state.session_id
            if state.session.session_state
            else None
        )
        LOGGER.info(
            "Client disconnect callback received for session %s", current_session_id
        )
        self._cancel_pending_decodes(state.decode.decode_stream, current_session_id)

    def _bootstrap_stream(
        self,
        state: _StreamState,
        metadata: Dict[str, str | bytes],
        context: grpc.ServicerContext,
    ) -> None:
        state.session.session_state = self._session_facade.resolve_from_metadata(
            metadata, context
        )
        session_state = state.session.session_state
        if session_state:
            set_session_id(session_state.session_id)
            if not session_state.session_info.token_required:
                state.session.session_logged = log_session_start(
                    session_state, VAD_AUTO_END
                )
                state.vad.vad_state = self._create_vad_state(session_state, context)
        state.decode.decode_stream = self._decode_scheduler.new_stream()
        if session_state and state.decode.decode_stream:
            state.decode.decode_stream.set_session_id(session_state.session_id)
            state.decode.decode_stream.set_model_id(session_state.session_info.model_id)

    def _step_init(self, state: _StreamState) -> None:
        if state.phase == StreamPhase.INIT:
            state.phase = StreamPhase.STREAMING

    def _handle_vad_trigger(
        self,
        state: _StreamState,
        vad_update: Any,
        context: grpc.ServicerContext,
    ) -> Iterator[stt_pb2.STTResult]:
        flow = self._flow()
        yield from handle_vad_trigger(flow, state, vad_update, context, VAD_AUTO_END)

    def _handle_final_chunk(
        self, state: _StreamState, context: grpc.ServicerContext
    ) -> Iterator[stt_pb2.STTResult]:
        flow = self._flow()
        yield from handle_final_chunk(flow, state, context)

    def _drain_pending_results(
        self, state: _StreamState, context: grpc.ServicerContext
    ) -> Iterator[stt_pb2.STTResult]:
        flow = self._flow()
        yield from drain_pending_results(flow, state, context)

    def _step_streaming_vad(
        self,
        state: _StreamState,
        vad_update: Any,
        context: grpc.ServicerContext,
    ) -> Iterator[stt_pb2.STTResult]:
        flow = self._flow()
        yield from step_streaming_vad(flow, state, vad_update, context, VAD_AUTO_END)

    def _step_streaming_buffer(
        self,
        state: _StreamState,
        chunk: stt_pb2.AudioChunk,
        context: grpc.ServicerContext,
    ) -> bool:
        flow = self._flow()
        ok = step_streaming_buffer(flow, state, chunk, context)
        session_state = state.session.session_state
        if session_state is not None:
            self._hooks.on_stream_buffer_bytes(
                session_state.session_id, len(state.buffer.buffer)
            )
        self._hooks.on_buffer_total_bytes(self._buffer_manager.total_bytes())
        return ok

    def _step_streaming_emit(
        self,
        state: _StreamState,
        chunk: stt_pb2.AudioChunk,
        context: grpc.ServicerContext,
    ) -> Iterator[stt_pb2.STTResult]:
        flow = self._flow()
        yield from step_streaming_emit(flow, state, chunk, context)

    def _step_streaming(
        self,
        state: _StreamState,
        chunk: stt_pb2.AudioChunk,
        context: grpc.ServicerContext,
    ) -> Iterator[stt_pb2.STTResult]:
        flow = self._flow()
        yield from step_streaming(flow, state, chunk, context, VAD_AUTO_END)

    def _handle_chunk(
        self,
        state: _StreamState,
        chunk: stt_pb2.AudioChunk,
        context: grpc.ServicerContext,
    ) -> Iterator[stt_pb2.STTResult]:
        flow = self._flow()
        yield from handle_chunk(flow, state, chunk, context, VAD_AUTO_END)

    def _step_drain(
        self,
        state: _StreamState,
        context: grpc.ServicerContext,
    ) -> Iterator[stt_pb2.STTResult]:
        flow = self._flow()
        yield from step_drain(flow, state, context)

    def _finalize_stream(
        self, state: _StreamState, context: grpc.ServicerContext
    ) -> None:
        # Send termination signal to stop watchdog cleanly
        state.events.stop_watchdog.set()

        if state.events.timeout_event.is_set():
            state.session.final_reason = "timeout"

        if state.vad.vad_state:
            state.vad.vad_state.close()

        if state.decode.decode_stream:
            (
                buffer_wait_total,
                queue_wait_total,
                inference_total,
                response_emit_total,
                decode_count,
            ) = state.decode.decode_stream.timing_summary()
            try:
                # Decode timing totals per stream (accumulated across decode tasks):
                # - buffer_wait: time spent buffering audio before scheduling decode
                # - queue_wait: time waiting for a worker after scheduling decode
                # - inference: model execution time
                # - response_emit: time spent yielding results to the client
                # - total: sum of buffer_wait + queue_wait + inference + response_emit
                decode_total_sec = (
                    buffer_wait_total
                    + queue_wait_total
                    + inference_total
                    + response_emit_total
                )
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
                            f"{decode_total_sec:.6f}",
                        ),
                        ("stt-decode-count", str(decode_count)),
                    )
                )
            except (grpc.RpcError, RuntimeError, ValueError):
                pass

        if state.session.audio_recorder and self._audio_storage:
            self._audio_storage.finalize_recording(
                state.session.audio_recorder, state.session.final_reason
            )
        if state.buffer.buffer:
            self._update_buffer_total(-len(state.buffer.buffer))
            state.buffer.buffer = bytearray()
            if state.session.session_state:
                self._hooks.on_stream_buffer_bytes(
                    state.session.session_state.session_id, 0
                )
        if state.session.session_state:
            self._hooks.on_stream_end(state.session.session_state.session_id)
            duration = time.monotonic() - state.session.session_start
            LOGGER.info(
                "Streaming finished for session_id=%s reason=%s vad_count=%d duration=%.2fs",
                state.session.session_state.session_id,
                state.session.final_reason,
                state.vad.vad_count,
                duration,
            )
        self._session_facade.remove_session(
            state.session.session_state, reason=state.session.final_reason
        )
        clear_session_id()

    def run(
        self,
        request_iterator: Iterable[stt_pb2.AudioChunk],
        context: grpc.ServicerContext,
    ) -> Iterator[stt_pb2.STTResult]:
        """Process incoming audio chunks and yield recognition results."""
        state = _StreamState()
        metadata = self._build_metadata(context)
        self._apply_metadata_session_id(metadata)

        context.add_callback(lambda: self._handle_disconnect(state))
        self._start_watchdog(state)

        try:
            state.events.processing_event.set()
            try:
                self._bootstrap_stream(state, metadata, context)
                self._step_init(state)
            finally:
                state.events.processing_event.clear()
            for chunk in request_iterator:
                state.events.processing_event.set()
                try:
                    for result in self._handle_chunk(state, chunk, context):
                        yield result
                finally:
                    state.events.processing_event.clear()
                if state.events.stop_stream:
                    break

            state.events.processing_event.set()
            try:
                for result in self._step_drain(state, context):
                    yield result
            finally:
                state.events.processing_event.clear()

        except (RuntimeError, grpc.RpcError):
            # Handle errors occurring during timeout abort
            if state.events.timeout_event.is_set():
                state.session.final_reason = "timeout"
            else:
                raise

        finally:
            state.phase = StreamPhase.DONE
            self._finalize_stream(state, context)

    def _create_vad_state(
        self, state: SessionState, context: grpc.ServicerContext
    ) -> VADGate:
        info = state.session_info
        silence = info.vad_silence
        if silence <= 0:
            silence = self._config.stream.vad_silence
        threshold = info.vad_threshold
        if threshold < 0:
            threshold = self._config.stream.vad_threshold
        if threshold > 0 and not info.vad_reserved:
            if not reserve_vad_slot():
                LOGGER.error(
                    "VAD pool exhausted; rejecting session_id=%s", state.session_id
                )
                self._session_facade.remove_session(state, reason="vad_pool_exhausted")
                abort_with_error(context, ErrorCode.VAD_POOL_EXHAUSTED)
            info.vad_reserved = True
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
            effective_rate = sample_rate or self._config.stream.default_sample_rate
            recorder = self._audio_storage.start_recording(
                session_state.session_id, effective_rate
            )
        recorder.append(pcm16)
        return recorder

    def _buffer_limit_bytes(self, sample_rate: Optional[int]) -> Optional[int]:
        return self._buffer_manager.buffer_limit_bytes(sample_rate)

    def _partial_decode_window_bytes(self, sample_rate: Optional[int]) -> Optional[int]:
        return self._buffer_manager.partial_decode_window_bytes(sample_rate)

    def _partial_enabled(self, state: _StreamState) -> bool:
        session_state = state.session.session_state
        if not session_state:
            return False
        attrs = session_state.session_info.attributes
        raw = attrs.get("partial") or attrs.get("partial_mode") or ""
        value = str(raw).strip().lower()
        if value in {"1", "true", "yes", "on", "enable", "enabled"}:
            return True
        if value in {"0", "false", "no", "off", "disable", "disabled"}:
            return False
        return False

    def _maybe_schedule_periodic_partial(
        self,
        state: _StreamState,
        vad_update: Any,
        context: grpc.ServicerContext,
    ) -> None:
        if not self._partial_enabled(state):
            return
        interval = self._config.partial_decode.interval_sec
        if interval is None:
            return
        limit_bytes = self._buffer_limit_bytes(state.session.sample_rate)
        if not should_attempt_periodic_partial(
            state,
            vad_update,
            interval,
            limit_bytes,
            self._config.stream.speech_rms_threshold,
            VAD_CONTINUE,
        ):
            return
        current_sec = state.activity.audio_received_sec
        last_sec = (
            state.buffer.buffer_start_sec
            if state.buffer.last_partial_decode_sec is None
            else state.buffer.last_partial_decode_sec
        )
        if current_sec - last_sec < interval:
            return
        if not self._ensure_decode_capacity(
            state.decode.decode_stream, False, state.session.session_state
        ):
            return
        window_bytes = self._partial_decode_window_bytes(state.session.sample_rate)
        pcm, offset_sec = build_partial_decode_window(
            state,
            window_bytes,
            self._config.stream.default_sample_rate,
        )
        if self._schedule_decode(
            state,
            pcm,
            is_final=False,
            offset_sec=offset_sec,
            count_vad=False,
            buffer_started_at=state.buffer.buffer_start_time,
            context=context,
        ):
            state.buffer.last_partial_decode_sec = current_sec

    def _max_chunk_bytes(self, sample_rate: Optional[int]) -> Optional[int]:
        max_ms = self._config.buffer_limits.max_chunk_ms
        if max_ms is None or max_ms <= 0:
            return None
        rate = sample_rate or self._config.stream.default_sample_rate
        if rate <= 0:
            return None
        return int((max_ms / 1000.0) * rate * 2)

    def _enforce_buffer_limit(
        self,
        state: _StreamState,
        context: grpc.ServicerContext,
    ) -> None:
        buffer = state.buffer.buffer
        limit_bytes = self._buffer_limit_bytes(state.session.sample_rate)
        if limit_bytes is None or len(buffer) <= limit_bytes:
            return

        if (
            state.session.session_state
            and state.decode.decode_stream
            and state.session.session_state.session_info.vad_mode == VAD_CONTINUE
            and self._partial_enabled(state)
        ):
            if not buffer_is_speech(buffer, self._config.stream.speech_rms_threshold):
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
                state.decode.decode_stream, False, state.session.session_state
            ):
                self._clear_buffer(state)
                return
            rate = state.session.sample_rate or self._config.stream.default_sample_rate
            window_drop = max(0, len(buffer) - limit_bytes)
            window_offset_sec = (
                state.buffer.buffer_start_sec
                + audio.chunk_duration_seconds(window_drop, rate)
            )
            window = bytes(buffer[-limit_bytes:])
            if not self._schedule_decode(
                state,
                window,
                is_final=False,
                offset_sec=window_offset_sec,
                count_vad=False,
                buffer_started_at=state.buffer.buffer_start_time,
                context=context,
            ):
                self._clear_buffer(state)
                return
            state.buffer.last_partial_decode_sec = state.activity.audio_received_sec
            overlap_sec = max(0.0, self._config.buffer_limits.buffer_overlap_sec)
            overlap_bytes = int(overlap_sec * rate * 2)
            retain = min(overlap_bytes, len(buffer))
            dropped = len(buffer) - retain
            if dropped > 0:
                new_buffer = bytearray(buffer[-retain:]) if retain > 0 else bytearray()
                dropped_sec = audio.chunk_duration_seconds(dropped, rate)
                state.buffer.buffer_start_sec += dropped_sec
                if state.buffer.buffer_start_time is not None:
                    state.buffer.buffer_start_time += dropped_sec
            else:
                new_buffer = bytearray()
            before_len = len(state.buffer.buffer)
            state.buffer.buffer = new_buffer
            self._update_buffer_total(len(state.buffer.buffer) - before_len)
            state.buffer.buffer_has_new_audio = False
            return

        before_len = len(buffer)
        overflow = len(buffer) - limit_bytes
        if overflow > 0:
            del buffer[:overflow]
            rate = state.session.sample_rate or self._config.stream.default_sample_rate
            dropped_sec = audio.chunk_duration_seconds(overflow, rate)
            state.buffer.buffer_start_sec += dropped_sec
            if state.buffer.buffer_start_time is not None:
                state.buffer.buffer_start_time += dropped_sec
            LOGGER.warning(
                "Buffer limit reached (%d bytes); trimmed %.2fs of audio.",
                limit_bytes,
                dropped_sec,
            )
        after_len = len(buffer)
        if after_len != before_len:
            self._update_buffer_total(after_len - before_len)
        state.buffer.buffer = buffer
