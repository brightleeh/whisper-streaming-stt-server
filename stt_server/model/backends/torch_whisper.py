"""PyTorch Whisper backend implementation."""

import logging
from typing import Any, Dict, List, Tuple

import whisper

from stt_server.model.backends.base import BackendInfo, ModelBackend, Segment

LOGGER = logging.getLogger("stt_server.model_backend")

FLOAT16_ALIASES = {"float16", "fp16", "half"}


class TorchWhisperBackend(ModelBackend):
    """Backend wrapper for openai-whisper."""

    def __init__(self, model_size: str, device: str, compute_type: str) -> None:
        self.device = device
        self.compute_type = compute_type
        self.model = whisper.load_model(model_size, device=device)
        LOGGER.info(
            "torch_whisper loaded model=%s device=%s compute_type=%s",
            model_size,
            device,
            compute_type,
        )
        self._apply_compute_type()

    def _apply_compute_type(self) -> None:
        if self.compute_type in FLOAT16_ALIASES and self.device != "cpu":
            if self.device == "mps":
                LOGGER.warning(
                    "MPS does not reliably support fp16 for Whisper decoding; "
                    "falling back to float32 (requested compute_type=%s)",
                    self.compute_type,
                )
                self.compute_type = "float32"
            else:
                self.model = self.model.half()
                return
        if self.compute_type not in {"float32", "fp32", "int8", "int8_float16"}:
            LOGGER.warning(
                "Unsupported compute_type=%s for torch_whisper; using float32",
                self.compute_type,
            )
        self.model = self.model.float()

    def transcribe(
        self, audio: Any, options: Dict[str, Any]
    ) -> Tuple[List[Segment], BackendInfo]:
        opts = self._normalize_options(options)
        if "fp16" not in opts:
            opts["fp16"] = self.compute_type in FLOAT16_ALIASES and self.device != "cpu"
        result = self.model.transcribe(audio, **opts)
        raw_segments = result.get("segments", [])
        segments: List[Segment] = []
        for seg in raw_segments:
            if not isinstance(seg, dict):
                continue
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            text = seg.get("text", "") or ""
            try:
                start_f = float(start)
            except (TypeError, ValueError):
                start_f = 0.0
            try:
                end_f = float(end)
            except (TypeError, ValueError):
                end_f = 0.0
            segments.append(Segment(start_f, end_f, str(text)))
        language = result.get("language") or ""
        if not isinstance(language, str):
            language = str(language)
        return segments, BackendInfo(language, -1.0)

    def _normalize_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        supported = {
            "temperature",
            "compression_ratio_threshold",
            "logprob_threshold",
            "no_speech_threshold",
            "condition_on_previous_text",
            "initial_prompt",
            "word_timestamps",
            "prepend_punctuations",
            "append_punctuations",
            "language",
            "task",
            "beam_size",
            "best_of",
            "patience",
            "length_penalty",
            "fp16",
            "prompt",
        }
        opts = dict(options)
        if "log_prob_threshold" in opts and "logprob_threshold" not in opts:
            opts["logprob_threshold"] = opts.pop("log_prob_threshold")
        if "without_timestamps" in opts and "word_timestamps" not in opts:
            opts["word_timestamps"] = not bool(opts.pop("without_timestamps"))

        dropped = {key: value for key, value in opts.items() if key not in supported}
        for key, value in dropped.items():
            LOGGER.warning(
                "Dropping unsupported torch_whisper option %s=%s", key, value
            )
            opts.pop(key, None)
        return opts
