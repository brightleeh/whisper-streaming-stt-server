"""Voice activity detection helpers."""

from __future__ import annotations

import copy
import math
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Protocol, TypeAlias, cast

import numpy as np

from stt_server.utils import audio
from stt_server.utils.logger import LOGGER

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
    """Protocol for Silero VAD models."""

    def __call__(self, audio_tensor: TensorLike, sample_rate: int):
        """Return VAD probability for the given audio tensor."""
        raise NotImplementedError

    def eval(self) -> None:
        """Switch the model to evaluation mode."""
        raise NotImplementedError

    def reset_states(self) -> None:
        """Reset any internal model states."""
        raise NotImplementedError


_SILERO_SAMPLE_RATE = 16000


@dataclass
class _VADPoolState:
    """Mutable container for VAD model pool state."""

    base_model: Optional[VADModel] = None
    pool_lock: threading.Lock = field(default_factory=threading.Lock)
    pool: "deque[VADModel]" = field(default_factory=deque)
    pool_capacity: int = 0
    pool_max_capacity: int = 0
    pool_growth_factor: float = 1.5
    pool_total_created: int = 0
    pool_reserved: int = 0


_VAD_STATE = _VADPoolState()


def _load_silero_model() -> VADModel:
    """Load a Silero VAD model with ONNX preferred when available."""
    assert load_silero_vad is not None
    try:
        return cast(VADModel, load_silero_vad(onnx=True))
    except Exception as exc:  # pylint: disable=broad-except
        # Silero loader can raise many runtime errors depending on backends.
        LOGGER.warning(
            "Failed to load ONNX Silero VAD model; falling back to TorchScript. "
            "Install onnxruntime to avoid torch.jit.load deprecation. Error: %s",
            exc,
        )
        return cast(VADModel, load_silero_vad())


def _load_silero_base_model():
    """Return a cached base Silero VAD model."""
    if torch is None or load_silero_vad is None:
        raise RuntimeError(
            "silero-vad requires torch + silero-vad. Install them to enable VAD-based detection."
        )
    if _VAD_STATE.base_model is None:
        _VAD_STATE.base_model = _load_silero_model()
    return _VAD_STATE.base_model


def _new_silero_model():
    """Create a fresh Silero VAD model instance."""
    base_model = _load_silero_base_model()
    try:
        model = copy.deepcopy(base_model)
    except Exception:  # pylint: disable=broad-except
        # deepcopy can fail for non-picklable model components; reload instead.
        model = _load_silero_model()
    if hasattr(model, "eval"):
        model.eval()
    if hasattr(model, "reset_states"):
        model.reset_states()
    return model


def configure_vad_model_pool(
    max_size: Optional[int] = None,
    prewarm: Optional[int] = None,
    max_capacity: Optional[int] = None,
    growth_factor: Optional[float] = None,
) -> None:
    """Configure the VAD model pool and optionally prewarm instances."""
    state = _VAD_STATE
    max_size_value = int(max_size) if max_size is not None else 0
    prewarm_value = int(prewarm) if prewarm is not None else 0
    max_capacity_value = int(max_capacity) if max_capacity is not None else 0
    max_size_value = max(0, max_size_value)
    prewarm_value = max(0, prewarm_value)
    max_capacity_value = max(0, max_capacity_value)
    if growth_factor is None:
        growth_value = 1.5
    else:
        try:
            growth_value = float(growth_factor)
        except (TypeError, ValueError):
            growth_value = 1.5
    if growth_value < 1.0:
        growth_value = 1.0
    if max_capacity_value == 0:
        max_capacity_value = max_size_value
    if max_size_value == 0 and max_capacity_value == 0:
        with state.pool_lock:
            state.pool.clear()
            state.pool_capacity = 0
            state.pool_max_capacity = 0
            state.pool_total_created = 0
            state.pool_reserved = 0
        return

    with state.pool_lock:
        state.pool_capacity = max_size_value or max_capacity_value
        state.pool_max_capacity = max_capacity_value
        state.pool_growth_factor = growth_value
        if state.pool_capacity > state.pool_max_capacity:
            state.pool_capacity = state.pool_max_capacity
        if state.pool_reserved > state.pool_capacity:
            state.pool_reserved = state.pool_capacity
        while len(state.pool) > state.pool_capacity:
            state.pool.pop()
            state.pool_total_created = max(0, state.pool_total_created - 1)

    target_pool = min(prewarm_value, state.pool_capacity)
    while True:
        with state.pool_lock:
            if len(state.pool) >= target_pool:
                break
            if state.pool_total_created >= state.pool_capacity:
                break
        try:
            model = _new_silero_model()
        except Exception:  # pylint: disable=broad-except
            LOGGER.exception("Failed to prewarm VAD model pool")
            break
        with state.pool_lock:
            if len(state.pool) >= state.pool_capacity:
                break
            state.pool.append(model)
            state.pool_total_created += 1


def _acquire_vad_model() -> VADModel:
    """Acquire a VAD model from the pool, creating one if needed."""
    state = _VAD_STATE
    with state.pool_lock:
        if state.pool_capacity > 0 and state.pool:
            model = state.pool.popleft()
            if hasattr(model, "reset_states"):
                model.reset_states()
            return model
        if state.pool_capacity > 0:
            overflow = state.pool_total_created >= state.pool_capacity
            state.pool_total_created += 1
        else:
            overflow = False
    if overflow:
        LOGGER.warning("VAD pool capacity exceeded; creating overflow model instance.")
    return _new_silero_model()


def _release_vad_model(model: VADModel) -> None:
    """Return a VAD model to the pool."""
    state = _VAD_STATE
    with state.pool_lock:
        if state.pool_capacity <= 0:
            return
        if len(state.pool) >= state.pool_capacity:
            state.pool_total_created = max(0, state.pool_total_created - 1)
            return
        if hasattr(model, "reset_states"):
            model.reset_states()
        state.pool.append(model)


def reserve_vad_slot() -> bool:
    """Reserve a VAD slot without instantiating a model."""
    state = _VAD_STATE
    with state.pool_lock:
        if state.pool_capacity <= 0:
            return True
        if state.pool_reserved < state.pool_capacity:
            state.pool_reserved += 1
            return True
        if state.pool_capacity < state.pool_max_capacity:
            new_capacity = max(
                1,
                int(math.ceil(state.pool_capacity * state.pool_growth_factor)),
            )
            if new_capacity > state.pool_max_capacity:
                new_capacity = state.pool_max_capacity
            if new_capacity > state.pool_capacity:
                state.pool_capacity = new_capacity
                LOGGER.info("Expanded VAD pool capacity to %d", state.pool_capacity)
            if state.pool_reserved < state.pool_capacity:
                state.pool_reserved += 1
                return True
        return False


def release_vad_slot() -> None:
    """Release a previously reserved VAD slot."""
    state = _VAD_STATE
    with state.pool_lock:
        if state.pool_capacity <= 0:
            return
        if state.pool_reserved > 0:
            state.pool_reserved -= 1


def _silero_frame_probability(model: VADModel, frame: np.ndarray) -> float:
    """Return Silero VAD probability for a single frame."""
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
    """Structured update returned by VADGate.update."""

    triggered: bool
    speech_active: bool
    silence_duration: float
    chunk_duration: float
    chunk_rms: float


class VADGate:
    """Tracks silence windows and determines when to trigger VAD."""

    def __init__(self, vad_threshold: float, vad_silence: float) -> None:
        """Initialize VAD thresholds and internal buffers."""
        self.vad_threshold = vad_threshold
        self.vad_silence = vad_silence
        self.speech_active = False
        self.silence_duration = 0.0
        self._vad_model = _acquire_vad_model() if vad_threshold > 0 else None
        self._vad_chunks: "deque[np.ndarray]" = deque()
        self._vad_chunk_offset = 0
        self._vad_buffered = 0

    def _vad_probability(self, chunk_bytes: bytes, sample_rate: int) -> float:
        """Compute max VAD probability for a PCM16 chunk."""
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
        # Buffer incoming chunks and run VAD on 32ms (512-sample) frames.
        # Keep the max score per chunk.
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
        """Update VAD state from a PCM16 chunk."""
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
        """Reset internal state after a trigger event."""
        self.speech_active = False
        self.silence_duration = 0.0

    def close(self) -> None:
        """Release pooled VAD resources."""
        if self._vad_model is None:
            return
        _release_vad_model(self._vad_model)
        self._vad_model = None


def buffer_is_speech(buffer_bytes: bytes, threshold: float) -> bool:
    """Return True when buffer RMS meets the speech threshold."""
    if threshold <= 0:
        return True
    return audio.chunk_rms(buffer_bytes) >= threshold
