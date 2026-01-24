import logging
import time
from concurrent import futures
from typing import Any, Dict, List, NamedTuple, Optional

from faster_whisper import WhisperModel

from stt_server.utils.audio import ensure_16k, pcm16_to_float32

LOGGER = logging.getLogger("stt_server.model_worker")


class DecodeResult(NamedTuple):
    segments: List[Any]
    latency_sec: float
    audio_duration: float
    rtf: float
    language_code: str
    language_probability: float


class ModelWorker:
    """Encapsulates a Whisper model and a single-thread executor for decode."""

    def __init__(
        self,
        model_size: str,
        device: str,
        compute_type: str,
        language: Optional[str],
        log_metrics: bool,
        base_options: Optional[Dict[str, Any]] = None,
    ):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.language = language
        self.log_metrics = log_metrics
        self.executor = futures.ThreadPoolExecutor(max_workers=1)
        self.base_options = base_options.copy() if base_options else {}
        if language:
            # language parameter controls the input language Whisper should expect.
            self.base_options.setdefault("language", language)

    def submit(
        self,
        pcm_bytes: bytes,
        src_rate: int,
        decode_options: Optional[Dict[str, Any]] = None,
    ) -> futures.Future:
        """Submit PCM bytes for asynchronous decode."""
        opts = decode_options.copy() if decode_options else None
        return self.executor.submit(self._decode, pcm_bytes, src_rate, opts)

    def _decode(
        self,
        pcm_bytes: bytes,
        src_rate: int,
        decode_options: Optional[Dict[str, Any]],
    ) -> DecodeResult:
        """Decode bytes inside the worker thread."""
        if len(pcm_bytes) == 0:
            return DecodeResult(
                segments=[],
                latency_sec=0.0,
                audio_duration=0.0,
                rtf=-1.0,
                language_code="",
                language_probability=-1.0,
            )

        audio = pcm16_to_float32(pcm_bytes)
        audio = ensure_16k(audio, src_rate)
        options: Dict[str, Any] = self.base_options.copy()
        if decode_options:
            options.update(decode_options)

        start = time.perf_counter()
        segments, info = self.model.transcribe(
            audio,
            **options,
        )
        language_code = info.language if info else ""
        language_probability = info.language_probability if info else -1.0
        elapsed = time.perf_counter() - start
        audio_duration = len(audio) / 16000.0 if len(audio) > 0 else 0.0
        rtf = elapsed / audio_duration if audio_duration > 0 else -1.0
        if self.log_metrics:
            LOGGER.info(
                "decode metrics audio=%.2fs elapsed=%.2fs real_time_factor=%.2f",
                audio_duration,
                elapsed,
                rtf if rtf >= 0 else float("inf"),
            )
        return DecodeResult(
            segments=list(segments),
            latency_sec=elapsed,
            audio_duration=audio_duration,
            rtf=rtf,
            language_code=language_code,
            language_probability=language_probability,
        )

    def close(self) -> None:
        self.executor.shutdown(wait=True)
