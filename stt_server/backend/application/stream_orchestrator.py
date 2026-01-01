from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import grpc

from gen.stt.python.v1 import stt_pb2
from stt_server.backend.application.session_registry import SessionFacade, SessionState
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
from stt_server.utils.logger import LOGGER


@dataclass(frozen=True)
class StreamOrchestratorConfig:
    vad_threshold: float
    vad_silence: float
    speech_rms_threshold: float
    session_timeout_sec: float
    default_sample_rate: int
    model_size: str
    device: str
    compute_type: str
    language_cycle: list[Optional[str]]
    pool_size: int
    task: str
    log_metrics: bool
    decode_timeout_sec: float
    language_lookup: SupportedLanguages
    storage_enabled: bool
    storage_directory: str
    storage_max_bytes: Optional[int]
    storage_max_files: Optional[int]
    storage_max_age_days: Optional[int]


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
        config: StreamOrchestratorConfig,
        hooks: StreamOrchestratorHooks | None = None,
    ) -> None:
        self._session_facade = session_facade
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

    def _create_decode_scheduler(
        self, config: StreamOrchestratorConfig
    ) -> DecodeScheduler:
        return DecodeScheduler(
            model_size=config.model_size,
            device=config.device,
            compute_type=config.compute_type,
            language_cycle=config.language_cycle,
            pool_size=config.pool_size,
            task=config.task,
            log_metrics=config.log_metrics,
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

            buffer = bytearray()
            sample_rate: Optional[int] = None
            last_activity = time.time()

            for chunk in request_iterator:
                current_session_id = session_state.session_id if session_state else None
                if not context.is_active():
                    LOGGER.info(
                        "Client disconnected; stopping session %s", current_session_id
                    )
                    final_reason = "client_disconnect"
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
                if len(chunk.pcm16) > 0 or chunk.is_final:
                    last_activity = time.time()

                buffer.extend(chunk.pcm16)
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
                        vad_state.reset_after_trigger()
                        continue
                    self._on_vad_trigger()
                    vad_count += 1
                    self._on_vad_utterance_start()
                    offset_sec = time.monotonic() - session_start
                    session_info = session_state.session_info
                    is_final = session_info.vad_mode == stt_pb2.VAD_AUTO_END
                    decode_stream.schedule_decode(
                        bytes(buffer),
                        sample_rate,
                        session_state.decode_options,
                        is_final,
                        offset_sec,
                        count_vad=True,
                    )
                    buffer = bytearray()
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

                yield from self._emit_results_with_session(
                    decode_stream, False, session_state
                )

                if chunk.is_final:
                    if buffer and buffer_is_speech(
                        buffer, self._config.speech_rms_threshold
                    ):
                        offset_sec = time.monotonic() - session_start
                        decode_stream.schedule_decode(
                            bytes(buffer),
                            sample_rate,
                            session_state.decode_options,
                            True,
                            offset_sec,
                            count_vad=False,
                        )
                        buffer = bytearray()
                    yield from self._emit_results_with_session(
                        decode_stream, False, session_state
                    )
                    final_reason = "client_sent_final_chunk"
                    break

                if time.time() - last_activity > self._config.session_timeout_sec:
                    final_reason = "timeout"
                    self._session_facade.remove_session(session_state, reason="timeout")
                    LOGGER.error("ERR1006 Session timeout (no audio)")
                    context.abort(
                        grpc.StatusCode.DEADLINE_EXCEEDED,
                        "ERR1006 Session timeout (no audio)",
                    )

            if decode_stream:
                if buffer and buffer_is_speech(
                    buffer, self._config.speech_rms_threshold
                ):
                    offset_sec = time.monotonic() - session_start
                    decode_stream.schedule_decode(
                        bytes(buffer),
                        sample_rate or self._config.default_sample_rate,
                        session_state.decode_options if session_state else {},
                        True,
                        offset_sec,
                        count_vad=False,
                    )

                while True:
                    emitted = list(
                        self._emit_results_with_session(
                            decode_stream,
                            block=bool(decode_stream.pending_partial_decodes()),
                            session_state=session_state,
                        )
                    )
                    if not emitted:
                        break
                    for result in emitted:
                        yield result

        finally:
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
            "Streaming started for session_id=%s vad_mode=%s decode_profile=%s vad_silence=%.3f vad_threshold=%.4f",
            state.session_id,
            "AUTO_END" if info.vad_mode == stt_pb2.VAD_AUTO_END else "CONTINUE",
            info.decode_profile,
            info.vad_silence,
            info.vad_threshold,
        )
        return True
