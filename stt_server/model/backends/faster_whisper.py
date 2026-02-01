"""faster-whisper backend implementation."""

from typing import Any, Dict, List, Tuple

from faster_whisper import WhisperModel

from stt_server.model.backends.base import BackendInfo, ModelBackend, Segment


class FasterWhisperBackend(ModelBackend):
    """Backend wrapper for faster-whisper."""

    def __init__(self, model_size: str, device: str, compute_type: str) -> None:
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(
        self, audio: Any, options: Dict[str, Any]
    ) -> Tuple[List[Segment], BackendInfo]:
        segments, info = self.model.transcribe(audio, **options)
        parsed = [Segment(seg.start, seg.end, seg.text) for seg in segments]
        language = info.language if info else ""
        language_probability = info.language_probability if info else -1.0
        return parsed, BackendInfo(language, language_probability)
