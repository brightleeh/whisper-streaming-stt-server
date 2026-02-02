"""faster-whisper backend implementation."""

from typing import Any, Dict, List, Tuple

from faster_whisper import WhisperModel
from faster_whisper.transcribe import BatchedInferencePipeline

from stt_server.model.backends.base import BackendInfo, ModelBackend, Segment


class FasterWhisperBackend(ModelBackend):
    """Backend wrapper for faster-whisper."""

    def __init__(self, model_size: str, device: str, compute_type: str) -> None:
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._batched_pipeline = BatchedInferencePipeline(self.model)

    def transcribe(
        self, audio: Any, options: Dict[str, Any]
    ) -> Tuple[List[Segment], BackendInfo]:
        opts = dict(options)
        batch_size = opts.pop("batch_size", None)
        if isinstance(batch_size, bool):
            batch_size = None
        if batch_size is not None:
            try:
                batch_size = int(batch_size)
            except (TypeError, ValueError):
                batch_size = None
        if batch_size and batch_size > 1:
            segments, info = self._batched_pipeline.transcribe(
                audio, batch_size=batch_size, **opts
            )
        else:
            segments, info = self.model.transcribe(audio, **opts)
        parsed = [Segment(seg.start, seg.end, seg.text) for seg in segments]
        language = info.language if info else ""
        language_probability = info.language_probability if info else -1.0
        return parsed, BackendInfo(language, language_probability)
