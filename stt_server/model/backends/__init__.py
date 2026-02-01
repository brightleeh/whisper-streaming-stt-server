"""Backend registry for model implementations."""

from typing import Type

from stt_server.model.backends.base import ModelBackend
from stt_server.model.backends.faster_whisper import FasterWhisperBackend


def get_backend(name: str) -> Type[ModelBackend]:
    """Resolve a backend implementation by name."""
    normalized = (name or "faster_whisper").lower()
    if normalized in {"faster_whisper", "faster-whisper", "fw"}:
        return FasterWhisperBackend
    if normalized in {"torch_whisper", "torch-whisper", "whisper", "pytorch"}:
        try:
            from stt_server.model.backends.torch_whisper import TorchWhisperBackend
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "torch_whisper backend requires the openai-whisper package."
            ) from exc

        return TorchWhisperBackend
    raise ValueError(f"Unknown model backend: {name}")


__all__ = ["get_backend"]
