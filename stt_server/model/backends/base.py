"""Backend interface for Whisper model implementations."""

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Tuple


@dataclass(frozen=True)
class Segment:
    """Minimal segment data used by the decode pipeline."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class BackendInfo:
    """Metadata returned by a backend transcription pass."""

    language: str
    language_probability: float


class ModelBackend(Protocol):
    """Backend interface for model implementations."""

    def __init__(self, model_size: str, device: str, compute_type: str) -> None:
        """Initialize backend with model configuration."""
        raise NotImplementedError

    def transcribe(
        self, audio: Any, options: Dict[str, Any]
    ) -> Tuple[List[Segment], BackendInfo]:
        """Transcribe audio and return segments with metadata."""
        raise NotImplementedError
