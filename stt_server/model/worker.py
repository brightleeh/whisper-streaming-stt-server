import logging
import threading
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
    queue_wait_sec: float


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
        self._active_lock = threading.Lock()
        self._active_cond = threading.Condition(self._active_lock)
        self._active_tasks = 0
        if language:
            # language parameter controls the input language Whisper should expect.
            self.base_options.setdefault("language", language)

    def _on_future_done(self, _future: futures.Future) -> None:
        with self._active_cond:
            if self._active_tasks > 0:
                self._active_tasks -= 1
            if self._active_tasks == 0:
                self._active_cond.notify_all()

    def submit(
        self,
        pcm_bytes: bytes,
        src_rate: int,
        decode_options: Optional[Dict[str, Any]] = None,
    ) -> futures.Future:
        """Submit PCM bytes for asynchronous decode."""
        opts = decode_options.copy() if decode_options else None
        submitted_at = time.perf_counter()
        future = self.executor.submit(
            self._decode, pcm_bytes, src_rate, opts, submitted_at
        )
        with self._active_cond:
            self._active_tasks += 1
        future.add_done_callback(self._on_future_done)
        return future

    def _decode(
        self,
        pcm_bytes: bytes,
        src_rate: int,
        decode_options: Optional[Dict[str, Any]],
        submitted_at: float,
    ) -> DecodeResult:
        """Decode bytes inside the worker thread."""
        start = time.perf_counter()
        queue_wait_sec = max(0.0, start - submitted_at)
        if len(pcm_bytes) == 0:
            return DecodeResult(
                segments=[],
                latency_sec=0.0,
                audio_duration=0.0,
                rtf=-1.0,
                language_code="",
                language_probability=-1.0,
                queue_wait_sec=queue_wait_sec,
            )

        audio = pcm16_to_float32(pcm_bytes)
        audio = ensure_16k(audio, src_rate)
        options: Dict[str, Any] = self.base_options.copy()
        if decode_options:
            options.update(decode_options)

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
            queue_wait_sec=queue_wait_sec,
        )

    def wait_for_idle(self, timeout_sec: Optional[float] = None) -> bool:
        with self._active_cond:
            if self._active_tasks == 0:
                return True
            return self._active_cond.wait_for(
                lambda: self._active_tasks == 0, timeout=timeout_sec
            )

    def close(self, timeout_sec: Optional[float] = None) -> None:
        if timeout_sec is None:
            self.executor.shutdown(wait=True)
            return
        drained = self.wait_for_idle(timeout_sec)
        if drained:
            self.executor.shutdown(wait=True)
            return
        LOGGER.warning("Timed out waiting for decode tasks; forcing worker shutdown")
        self.executor.shutdown(wait=False, cancel_futures=True)
