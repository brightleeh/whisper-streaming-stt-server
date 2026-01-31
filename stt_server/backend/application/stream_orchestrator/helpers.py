"""Helper functions for stream orchestrator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator, Optional

from gen.stt.python.v1 import stt_pb2
from stt_server.backend.application.session_manager import SessionState
from stt_server.backend.component.decode_scheduler import DecodeStream
from stt_server.backend.component.vad_gate import buffer_is_speech
from stt_server.utils import audio
from stt_server.utils.logger import LOGGER

if TYPE_CHECKING:
    from .types import _StreamState


def emit_results_with_session(
    decode_stream: DecodeStream,
    block: bool,
    session_state: Optional[SessionState],
) -> Iterator[stt_pb2.STTResult]:
    """Yield ready results after aligning stream session id."""
    if session_state and decode_stream.session_id != session_state.session_id:
        decode_stream.set_session_id(session_state.session_id)
    yield from decode_stream.emit_ready(block)


def log_session_start(state: SessionState, vad_auto_end: int) -> bool:
    """Log session start details and return True."""
    info = state.session_info
    LOGGER.info(
        "Streaming started for session_id=%s vad_mode=%s decode_profile=%s "
        "vad_silence=%.3f vad_threshold=%.4f model_id=%s",
        state.session_id,
        "AUTO_END" if info.vad_mode == vad_auto_end else "CONTINUE",
        info.decode_profile,
        info.vad_silence,
        info.vad_threshold,
        info.model_id,
    )
    return True


def should_attempt_periodic_partial(
    state: "_StreamState",
    vad_update: Any,
    interval: float,
    limit_bytes: Optional[int],
    speech_threshold: float,
    vad_continue_mode: int,
) -> bool:
    """Return True when a periodic partial decode should be scheduled."""
    if interval <= 0:
        return False
    if state.events.disconnect_event.is_set() or state.events.timeout_event.is_set():
        return False
    if not state.session.session_state or not state.decode.decode_stream:
        return False
    if state.session.session_state.session_info.vad_mode != vad_continue_mode:
        return False
    if not vad_update.speech_active or not state.buffer.buffer:
        return False
    if limit_bytes is not None and len(state.buffer.buffer) > limit_bytes:
        return False
    if not buffer_is_speech(state.buffer.buffer, speech_threshold):
        return False
    return True


def build_partial_decode_window(
    state: "_StreamState",
    window_bytes: Optional[int],
    default_sample_rate: int,
) -> tuple[bytes, float]:
    """Return PCM bytes and offset for a partial decode window."""
    rate = state.session.sample_rate or default_sample_rate
    buffer_bytes = state.buffer.buffer
    offset_sec = state.buffer.buffer_start_sec
    if window_bytes is not None and len(buffer_bytes) > window_bytes:
        drop = len(buffer_bytes) - window_bytes
        offset_sec += audio.chunk_duration_seconds(drop, rate)
        pcm = bytes(buffer_bytes[-window_bytes:])
    else:
        pcm = bytes(buffer_bytes)
    return pcm, offset_sec
