from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import grpc

from gen.stt.python.v1 import stt_pb2
from stt_server.backend.core.decode_scheduler import DecodeScheduler, DecodeStream
from stt_server.backend.core.epd import EPDState, buffer_is_speech
from stt_server.backend.core.metrics import Metrics
from stt_server.backend.service.components.audio_storage import (
    AudioStorageManager,
    SessionAudioRecorder,
)
from stt_server.backend.service.components.session_facade import (
    SessionFacade,
    SessionState,
)
from stt_server.utils.logger import LOGGER


@dataclass(frozen=True)
class StreamingRunnerConfig:
    epd_threshold: float
    epd_silence: float
    speech_rms_threshold: float
    session_timeout_sec: float
    default_sample_rate: int


class StreamingRunner:
    """Executes the streaming recognition loop for the gRPC servicer."""

    def __init__(
        self,
        session_facade: SessionFacade,
        decode_scheduler: DecodeScheduler,
        metrics: Metrics,
        config: StreamingRunnerConfig,
        audio_storage: Optional[AudioStorageManager] = None,
    ) -> None:
        self._session_facade = session_facade
        self._decode_scheduler = decode_scheduler
        self._metrics = metrics
        self._config = config
        self._audio_storage = audio_storage

    def run(
        self,
        request_iterator: Iterable[stt_pb2.AudioChunk],
        context: grpc.ServicerContext,
    ) -> Iterator[stt_pb2.STTResult]:
        session_state: Optional[SessionState] = None
        epd_state: Optional[EPDState] = None
        decode_stream = None
        metadata = {k.lower(): v for (k, v) in context.invocation_metadata()}
        session_logged = False
        final_reason = "stream_end"
        session_start = time.monotonic()
        epd_count = 0
        audio_recorder: Optional[SessionAudioRecorder] = None
        try:
            session_state = self._session_facade.resolve_from_metadata(
                metadata, context
            )
            if session_state:
                session_logged = self._log_session_start(session_state)
                epd_state = self._create_epd_state(session_state)
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
                if epd_state is None:
                    epd_state = self._create_epd_state(session_state)

                self._session_facade.validate_token(session_state, chunk, context)

                if epd_state is None:
                    # Should not happen, but guard against update before initialization.
                    epd_state = self._create_epd_state(session_state)

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
                epd_update = epd_state.update(chunk.pcm16, sample_rate)

                if epd_update.triggered:
                    if not buffer_is_speech(buffer, self._config.speech_rms_threshold):
                        LOGGER.info(
                            "Skipping decode: chunk RMS %.4f below speech threshold %.4f",
                            epd_update.chunk_rms,
                            self._config.speech_rms_threshold,
                        )
                        LOGGER.info(
                            "session_id=%s ignored low-energy buffer",
                            session_state.session_id if session_state else "unknown",
                        )
                        buffer = bytearray()
                        epd_state.reset_after_trigger()
                        continue
                    epd_count += 1
                    offset_sec = time.monotonic() - session_start
                    session_info = session_state.session_info
                    is_final = session_info.epd_mode == stt_pb2.EPD_AUTO_END
                    decode_stream.schedule_decode(
                        bytes(buffer),
                        sample_rate,
                        session_state.decode_options,
                        is_final,
                        offset_sec,
                    )
                    buffer = bytearray()
                    LOGGER.info(
                        "EPD count=%d for current session (pending=%d, mode=%s)",
                        epd_count,
                        decode_stream.pending_partial_decodes(),
                        (
                            "AUTO_END"
                            if session_info.epd_mode == stt_pb2.EPD_AUTO_END
                            else "CONTINUE"
                        ),
                    )
                    if is_final:
                        yield from self._emit_results_with_session(
                            decode_stream, False, session_state
                        )
                        final_reason = "auto_epd_finalized"
                        break
                    epd_state.reset_after_trigger()

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
                    context.abort(
                        grpc.StatusCode.DEADLINE_EXCEEDED,
                        "Session timeout (no audio)",
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
                    "Streaming finished for session_id=%s reason=%s epd_count=%d duration=%.2fs",
                    session_state.session_id,
                    final_reason,
                    epd_count,
                    duration,
                )
            self._session_facade.remove_session(session_state, reason=final_reason)

    def _create_epd_state(self, state: SessionState) -> EPDState:
        info = state.session_info
        silence = info.epd_silence
        if silence <= 0:
            silence = self._config.epd_silence
        threshold = info.epd_threshold
        if threshold < 0:
            threshold = self._config.epd_threshold
        return EPDState(threshold, silence)

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
            "Streaming started for session_id=%s epd_mode=%s decode_profile=%s epd_silence=%.3f epd_threshold=%.4f",
            state.session_id,
            "AUTO_END" if info.epd_mode == stt_pb2.EPD_AUTO_END else "CONTINUE",
            info.decode_profile,
            info.epd_silence,
            info.epd_threshold,
        )
        return True
