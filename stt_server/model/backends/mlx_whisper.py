"""MLX Whisper backend implementation for Apple Silicon."""

import logging
from typing import Any, Dict, List, Tuple

import mlx.core as mx
import mlx_whisper

from stt_server.model.backends.base import BackendInfo, ModelBackend, Segment

LOGGER = logging.getLogger("stt_server.model_backend")

FLOAT16_ALIASES = {"float16", "fp16", "half"}

# Map standard Whisper model size names to mlx-community HuggingFace repos.
MODEL_REPO_MAP = {
    "tiny": "mlx-community/whisper-tiny",
    "tiny.en": "mlx-community/whisper-tiny.en",
    "base": "mlx-community/whisper-base",
    "base.en": "mlx-community/whisper-base.en",
    "small": "mlx-community/whisper-small",
    "small.en": "mlx-community/whisper-small.en",
    "medium": "mlx-community/whisper-medium",
    "medium.en": "mlx-community/whisper-medium.en",
    "large": "mlx-community/whisper-large-v3",
    "large-v1": "mlx-community/whisper-large-v1-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


class MlxWhisperBackend(ModelBackend):
    """Backend wrapper for mlx-whisper (Apple Silicon)."""

    def __init__(self, model_size: str, device: str, compute_type: str) -> None:
        self.device = device
        self.compute_type = compute_type
        # If model_size contains "/" assume it is already a HF repo path.
        if "/" in model_size:
            self.model_repo = model_size
        else:
            self.model_repo = MODEL_REPO_MAP.get(
                model_size, f"mlx-community/whisper-{model_size}"
            )
        self._dtype = mx.float16 if compute_type in FLOAT16_ALIASES else mx.float32
        LOGGER.info(
            "mlx_whisper initialized repo=%s device=%s compute_type=%s",
            self.model_repo,
            device,
            compute_type,
        )

    def transcribe(
        self, audio: Any, options: Dict[str, Any]
    ) -> Tuple[List[Segment], BackendInfo]:
        opts = self._normalize_options(options)
        if "fp16" not in opts:
            opts["fp16"] = self.compute_type in FLOAT16_ALIASES
        result = mlx_whisper.transcribe(audio, path_or_hf_repo=self.model_repo, **opts)
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
            "best_of",
            "length_penalty",
            "fp16",
            "prompt",
        }
        opts = dict(options)
        if "log_prob_threshold" in opts and "logprob_threshold" not in opts:
            opts["logprob_threshold"] = opts.pop("log_prob_threshold")
        if "without_timestamps" in opts and "word_timestamps" not in opts:
            opts["word_timestamps"] = not bool(opts.pop("without_timestamps"))

        # beam_size and patience are not yet supported by mlx_whisper (beam search
        # decoder is not implemented). Drop them silently to use greedy decoding.
        for key in ("beam_size", "patience"):
            if key in opts:
                opts.pop(key)

        dropped = {key: value for key, value in opts.items() if key not in supported}
        for key, value in dropped.items():
            LOGGER.warning("Dropping unsupported mlx_whisper option %s=%s", key, value)
            opts.pop(key, None)
        return opts
