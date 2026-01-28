"""Voice activity detection helpers."""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol, TypeAlias, cast

import numpy as np

from stt_server.utils import audio

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency in some environments
    torch = None

try:
    from silero_vad import load_silero_vad
except ImportError:  # pragma: no cover - optional dependency in some environments
    load_silero_vad = None

if TYPE_CHECKING:
    import torch as torch_mod

    TensorLike: TypeAlias = torch_mod.Tensor
else:
    TensorLike: TypeAlias = object


class VADModel(Protocol):
    def __call__(self, audio: TensorLike, sample_rate: int): ...

    def eval(self) -> None: ...

    def reset_states(self) -> None: ...


_SILERO_SAMPLE_RATE = 16000
_SILERO_BASE_MODEL: Optional[VADModel] = None


def _load_silero_base_model():
    if torch is None or load_silero_vad is None:
        raise RuntimeError(
            "silero-vad requires torch + silero-vad. Install them to enable VAD-based detection."
        )
    assert load_silero_vad is not None
    global _SILERO_BASE_MODEL
    if _SILERO_BASE_MODEL is None:
        _SILERO_BASE_MODEL = cast(VADModel, load_silero_vad())
    return _SILERO_BASE_MODEL


def _new_silero_model():
    base_model = _load_silero_base_model()
    try:
        model = copy.deepcopy(base_model)
    except Exception:
        assert load_silero_vad is not None
        model = cast(VADModel, load_silero_vad())
    model.eval()
    if hasattr(model, "reset_states"):
        model.reset_states()
    return model


def _silero_frame_probability(model: VADModel, frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    assert torch is not None
    tensor = torch.from_numpy(frame).unsqueeze(0)
    with torch.no_grad():
        prob = model(tensor, _SILERO_SAMPLE_RATE)
    if isinstance(prob, torch.Tensor):
        return float(prob.mean().item())
    if prob is None:
        return 0.0
    return float(prob)


@dataclass
class VADGateUpdate:
    triggered: bool
    speech_active: bool
    silence_duration: float
    chunk_duration: float
    chunk_rms: float


class VADGate:
    """Tracks silence windows and determines when to trigger VAD."""

    def __init__(self, vad_threshold: float, vad_silence: float) -> None:
        self.vad_threshold = vad_threshold
        self.vad_silence = vad_silence
        self.speech_active = False
        self.silence_duration = 0.0
        self._vad_model = _new_silero_model() if vad_threshold > 0 else None
        self._vad_chunks: "deque[np.ndarray]" = deque()
        self._vad_chunk_offset = 0
        self._vad_buffered = 0

    def _vad_probability(self, chunk_bytes: bytes, sample_rate: int) -> float:
        if not chunk_bytes:
            return 0.0
        if self._vad_model is None:
            return 0.0
        audio_f32 = audio.pcm16_to_float32(chunk_bytes)
        if sample_rate and sample_rate != _SILERO_SAMPLE_RATE:
            audio_f32 = audio.ensure_16k(audio_f32, sample_rate)
        if audio_f32.size == 0:
            return 0.0
        model = self._vad_model
        frame_size = 512
        max_prob = 0.0
        self._vad_chunks.append(audio_f32)
        self._vad_buffered += audio_f32.size
        # Buffer incoming chunks and run VAD on 32ms (512-sample) frames; keep the max score per chunk.
        while self._vad_buffered >= frame_size:
            frame = np.empty(frame_size, dtype=np.float32)
            filled = 0
            while filled < frame_size and self._vad_chunks:
                chunk = self._vad_chunks[0]
                remaining = chunk.size - self._vad_chunk_offset
                take = min(frame_size - filled, remaining)
                frame[filled : filled + take] = chunk[
                    self._vad_chunk_offset : self._vad_chunk_offset + take
                ]
                filled += take
                self._vad_chunk_offset += take
                if self._vad_chunk_offset >= chunk.size:
                    self._vad_chunks.popleft()
                    self._vad_chunk_offset = 0
            self._vad_buffered -= frame_size
            prob = _silero_frame_probability(model, frame)
            if prob > max_prob:
                max_prob = prob
        return max_prob

    def update(self, chunk_bytes: bytes, sample_rate: int) -> VADGateUpdate:
        chunk_duration = audio.chunk_duration_seconds(len(chunk_bytes), sample_rate)
        chunk_rms = audio.chunk_rms(chunk_bytes)
        triggered = False

        if chunk_bytes:
            if self._vad_model is None:
                speech_detected = True
            else:
                speech_prob = self._vad_probability(chunk_bytes, sample_rate)
                speech_detected = speech_prob >= self.vad_threshold
            if speech_detected:
                self.speech_active = True
                self.silence_duration = 0.0
            else:
                self.silence_duration += chunk_duration
        elif chunk_duration > 0:
            self.silence_duration += chunk_duration

        if self.speech_active and self.silence_duration >= self.vad_silence:
            triggered = True

        return VADGateUpdate(
            triggered=triggered,
            speech_active=self.speech_active,
            silence_duration=self.silence_duration,
            chunk_duration=chunk_duration,
            chunk_rms=chunk_rms,
        )

    def reset_after_trigger(self) -> None:
        self.speech_active = False
        self.silence_duration = 0.0


def buffer_is_speech(buffer_bytes: bytes, threshold: float) -> bool:
    """Return True when buffer RMS meets the speech threshold."""
    if threshold <= 0:
        return True
    return audio.chunk_rms(buffer_bytes) >= threshold
