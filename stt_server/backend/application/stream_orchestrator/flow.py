"""Streaming flow helpers for the stream orchestrator."""

from __future__ import annotations

import time
from typing import Any, Iterator

import grpc

from gen.stt.python.v1 import stt_pb2
from stt_server.backend.component.vad_gate import buffer_is_speech
from stt_server.errors import ErrorCode, abort_with_error
from stt_server.utils import audio
from stt_server.utils.logger import LOGGER, set_session_id

from .helpers import log_session_start
from .types import (
    StreamFlowContext,
    StreamPhase,
    _StreamState,
)


def handle_vad_trigger(
    flow: StreamFlowContext,
    state: _StreamState,
    vad_update: Any,
    context: grpc.ServicerContext,
    vad_auto_end: int,
) -> Iterator[stt_pb2.STTResult]:
    """Handle VAD-triggered decode scheduling."""
    if (
        not state.vad.vad_state
        or not state.decode.decode_stream
        or not state.session.session_state
    ):
        return
    stream_config = flow.config.stream
    if not buffer_is_speech(state.buffer.buffer, stream_config.speech_rms_threshold):
        LOGGER.info(
            "Skipping decode: chunk RMS %.4f below speech threshold %.4f",
            vad_update.chunk_rms,
            stream_config.speech_rms_threshold,
        )
        LOGGER.info(
            "session_id=%s ignored low-energy buffer",
            (
                state.session.session_state.session_id
                if state.session.session_state
                else "unknown"
            ),
        )
        flow.buffer.clear(state)
        state.vad.vad_state.reset_after_trigger()
        return
    flow.hooks.on_vad_trigger()
    state.vad.vad_count += 1
    flow.hooks.on_vad_utterance_start()
    session_info = state.session.session_state.session_info
    is_final = session_info.vad_mode == vad_auto_end
    if state.events.disconnect_event.is_set() or state.events.timeout_event.is_set():
        LOGGER.info("Skipping decode due to shutdown signal.")
        state.session.final_reason = (
            "client_disconnect" if state.events.disconnect_event.is_set() else "timeout"
        )
        state.session.client_disconnected = state.events.disconnect_event.is_set()
        state.events.stop_stream = True
        return
    if not flow.decode.ensure_capacity(
        state.decode.decode_stream, is_final, state.session.session_state
    ):
        flow.buffer.clear(state)
        state.vad.vad_state.reset_after_trigger()
        return
    flow.decode.schedule(
        state,
        bytes(state.buffer.buffer),
        is_final=is_final,
        offset_sec=state.buffer.buffer_start_sec,
        count_vad=True,
        buffer_started_at=state.buffer.buffer_start_time,
        context=context,
    )
    flow.buffer.clear(state)
    LOGGER.info(
        "VAD count=%d for current session (pending=%d, mode=%s, active_vad=%d)",
        state.vad.vad_count,
        state.decode.decode_stream.pending_partial_decodes(),
        "AUTO_END" if session_info.vad_mode == vad_auto_end else "CONTINUE",
        flow.hooks.active_vad_utterances(),
    )
    if is_final:
        for result in flow.decode.emit_with_activity(state, False):
            yield result
        state.session.final_reason = "auto_vad_finalized"
        state.events.stop_stream = True
        return
    state.vad.vad_state.reset_after_trigger()


def handle_final_chunk(
    flow: StreamFlowContext,
    state: _StreamState,
    context: grpc.ServicerContext,
) -> Iterator[stt_pb2.STTResult]:
    """Handle final chunk processing."""
    if not state.decode.decode_stream:
        return
    if state.buffer.buffer and state.buffer.buffer_has_new_audio:
        if (
            state.events.disconnect_event.is_set()
            or state.events.timeout_event.is_set()
        ):
            LOGGER.info("Skipping final decode due to shutdown signal.")
            state.session.final_reason = (
                "client_disconnect"
                if state.events.disconnect_event.is_set()
                else "timeout"
            )
            state.session.client_disconnected = state.events.disconnect_event.is_set()
            state.events.stop_stream = True
            return
        flow.decode.ensure_capacity(
            state.decode.decode_stream, True, state.session.session_state
        )
        flow.decode.schedule(
            state,
            bytes(state.buffer.buffer),
            is_final=True,
            offset_sec=state.buffer.buffer_start_sec,
            count_vad=False,
            buffer_started_at=state.buffer.buffer_start_time,
            context=context,
        )
        flow.buffer.clear(state)
    for result in flow.decode.emit_with_activity(state, False):
        yield result
    state.session.final_reason = "client_sent_final_chunk"
    state.events.stop_stream = True


def drain_pending_results(
    flow: StreamFlowContext,
    state: _StreamState,
    context: grpc.ServicerContext,
) -> Iterator[stt_pb2.STTResult]:
    """Drain pending results during shutdown."""
    if state.events.timeout_event.is_set():
        LOGGER.info("Stopping stream due to timeout signal.")
        state.session.final_reason = "timeout"
        abort_with_error(context, ErrorCode.SESSION_TIMEOUT)
    if state.decode.decode_stream:
        if (
            not state.session.client_disconnected
            and state.buffer.buffer
            and state.buffer.buffer_has_new_audio
            and buffer_is_speech(
                state.buffer.buffer, flow.config.stream.speech_rms_threshold
            )
        ):
            flow.decode.ensure_capacity(
                state.decode.decode_stream, True, state.session.session_state
            )
            if flow.decode.schedule(
                state,
                bytes(state.buffer.buffer),
                is_final=True,
                offset_sec=state.buffer.buffer_start_sec,
                count_vad=False,
                buffer_started_at=state.buffer.buffer_start_time,
                context=context,
            ):
                flow.buffer.clear(state)
        state.buffer.buffer_start_time = None

        while True:
            if state.events.timeout_event.is_set():
                LOGGER.info("Stopping stream due to timeout signal.")
                state.session.final_reason = "timeout"
                abort_with_error(context, ErrorCode.SESSION_TIMEOUT)
            emitted = list(
                flow.decode.emit_with_activity(
                    state,
                    block=state.decode.decode_stream.has_pending_results(),
                )
            )
            if not emitted:
                break
            for result in emitted:
                yield result


def step_streaming_vad(
    flow: StreamFlowContext,
    state: _StreamState,
    vad_update: Any,
    context: grpc.ServicerContext,
    vad_auto_end: int,
) -> Iterator[stt_pb2.STTResult]:
    """Handle VAD steps in the streaming loop."""
    if vad_update.triggered:
        for result in handle_vad_trigger(
            flow, state, vad_update, context, vad_auto_end
        ):
            yield result
        return
    flow.decode.maybe_schedule_periodic_partial(state, vad_update, context)


def step_streaming_buffer(
    flow: StreamFlowContext,
    state: _StreamState,
    chunk: stt_pb2.AudioChunk,
    context: grpc.ServicerContext,
) -> bool:
    """Handle buffer management during streaming."""
    if chunk.is_final:
        return True
    if state.events.disconnect_event.is_set() or state.events.timeout_event.is_set():
        LOGGER.info("Skipping buffer management due to shutdown signal.")
        state.session.final_reason = (
            "client_disconnect" if state.events.disconnect_event.is_set() else "timeout"
        )
        state.session.client_disconnected = state.events.disconnect_event.is_set()
        state.events.stop_stream = True
        return False
    flow.buffer.enforce_limit(state, context)
    return not state.events.stop_stream


def step_streaming_emit(
    flow: StreamFlowContext,
    state: _StreamState,
    chunk: stt_pb2.AudioChunk,
    context: grpc.ServicerContext,
) -> Iterator[stt_pb2.STTResult]:
    """Emit results during streaming."""
    for result in flow.decode.emit_with_activity(state, False):
        yield result
    if chunk.is_final:
        for result in handle_final_chunk(flow, state, context):
            yield result


def step_streaming(
    flow: StreamFlowContext,
    state: _StreamState,
    chunk: stt_pb2.AudioChunk,
    context: grpc.ServicerContext,
    vad_auto_end: int,
) -> Iterator[stt_pb2.STTResult]:
    """Process a streaming chunk end-to-end."""
    if state.events.disconnect_event.is_set():
        LOGGER.info("Stopping stream due to disconnect signal.")
        state.session.final_reason = "client_disconnect"
        state.session.client_disconnected = True
        flow.buffer.clear(state)
        state.events.stop_stream = True
        return
    if state.events.timeout_event.is_set():
        LOGGER.info("Stopping stream due to timeout signal.")
        state.session.final_reason = "timeout"
        abort_with_error(context, ErrorCode.SESSION_TIMEOUT)

    # Update activity timestamp
    flow.activity.mark(state)

    session_state = state.session.session_state
    current_session_id = session_state.session_id if session_state else None
    if current_session_id:
        set_session_id(current_session_id)
    if not context.is_active():
        LOGGER.info("Client disconnected; stopping session %s", current_session_id)
        state.session.final_reason = "client_disconnect"
        state.session.client_disconnected = True
        flow.decode.cancel_pending_decodes(
            state.decode.decode_stream, current_session_id
        )
        flow.buffer.clear(state)
        state.events.stop_stream = True
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

    if state.session.session_state is None:
        state.session.session_state = flow.session.ensure_session_from_chunk(
            state.session.session_state, chunk, context
        )

    if state.session.session_state and state.decode.decode_stream:
        state.decode.decode_stream.set_session_id(
            state.session.session_state.session_id
        )
        set_session_id(state.session.session_state.session_id)

    flow.session.validate_token(state.session.session_state, chunk, context)

    if state.session.session_state and not state.session.session_logged:
        state.session.session_logged = log_session_start(
            state.session.session_state, vad_auto_end
        )
    if state.vad.vad_state is None and state.session.session_state:
        # Initialize VAD state before processing audio.
        state.vad.vad_state = flow.session.create_vad_state(
            state.session.session_state, context
        )

    state.session.sample_rate = (
        chunk.sample_rate
        if chunk.sample_rate > 0
        else state.session.sample_rate or flow.config.stream.default_sample_rate
    )
    max_chunk_bytes = flow.audio.max_chunk_bytes(state.session.sample_rate)
    if max_chunk_bytes is not None and len(chunk.pcm16) > max_chunk_bytes:
        LOGGER.warning(
            "Chunk size exceeds limit (bytes=%d max=%d session_id=%s)",
            len(chunk.pcm16),
            max_chunk_bytes,
            (
                state.session.session_state.session_id
                if state.session.session_state
                else "unknown"
            ),
        )
        abort_with_error(
            context,
            ErrorCode.AUDIO_CHUNK_TOO_LARGE,
            detail=f"chunk bytes {len(chunk.pcm16)} exceeds max {max_chunk_bytes}",
        )
    flow.limits.enforce_chunk(state, chunk, context)
    state.session.audio_recorder = flow.audio.capture_audio_chunk(
        state.session.audio_recorder,
        state.session.session_state,
        state.session.sample_rate,
        chunk.pcm16,
    )

    if not state.buffer.buffer and chunk.pcm16:
        state.buffer.buffer_start_sec = state.activity.audio_received_sec
        state.buffer.buffer_start_time = time.perf_counter()
    incoming = chunk.pcm16
    incoming_len = len(incoming)
    if incoming_len:
        allowed = flow.buffer.apply_global_limit(state, incoming_len)
        if allowed < incoming_len:
            incoming = incoming[-allowed:] if allowed > 0 else b""
    if incoming:
        state.buffer.buffer.extend(incoming)
        state.buffer.buffer_has_new_audio = True
    elif not state.buffer.buffer:
        state.buffer.buffer_start_time = None
    state.activity.audio_received_sec += audio.chunk_duration_seconds(
        len(chunk.pcm16), state.session.sample_rate
    )
    vad_state = state.vad.vad_state
    if vad_state is None:
        LOGGER.error(
            "VAD state missing for session_id=%s",
            (
                state.session.session_state.session_id
                if state.session.session_state
                else "unknown"
            ),
        )
        abort_with_error(context, ErrorCode.STREAM_UNEXPECTED)
    vad_update = vad_state.update(chunk.pcm16, state.session.sample_rate)

    for result in step_streaming_vad(flow, state, vad_update, context, vad_auto_end):
        yield result
    if state.events.stop_stream:
        return

    if not step_streaming_buffer(flow, state, chunk, context):
        return

    for result in step_streaming_emit(flow, state, chunk, context):
        yield result


def handle_chunk(
    flow: StreamFlowContext,
    state: _StreamState,
    chunk: stt_pb2.AudioChunk,
    context: grpc.ServicerContext,
    vad_auto_end: int,
) -> Iterator[stt_pb2.STTResult]:
    """Handle a single chunk with phase transitions."""
    match state.phase:
        case StreamPhase.INIT:
            flow.session.step_init(state)
        case StreamPhase.STREAMING:
            pass
        case StreamPhase.DRAINING | StreamPhase.DONE:
            return
    for result in step_streaming(flow, state, chunk, context, vad_auto_end):
        yield result
    if state.events.stop_stream and state.phase == StreamPhase.STREAMING:
        state.phase = StreamPhase.DRAINING


def step_drain(
    flow: StreamFlowContext,
    state: _StreamState,
    context: grpc.ServicerContext,
) -> Iterator[stt_pb2.STTResult]:
    """Drain pending results and close the stream."""
    if state.phase == StreamPhase.DONE:
        return
    if state.phase != StreamPhase.DRAINING:
        state.phase = StreamPhase.DRAINING
    for result in drain_pending_results(flow, state, context):
        yield result
    state.phase = StreamPhase.DONE
