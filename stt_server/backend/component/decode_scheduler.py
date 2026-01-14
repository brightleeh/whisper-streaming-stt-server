"""Decode worker pool and per-stream scheduling."""

from __future__ import annotations

import threading
from concurrent import futures
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple

import grpc

from gen.stt.python.v1 import stt_pb2
from stt_server.config.default.model import DEFAULT_MODEL_ID
from stt_server.config.languages import SupportedLanguages
from stt_server.utils.logger import LOGGER

if TYPE_CHECKING:
    from stt_server.backend.application.stream_orchestrator import StreamOrchestrator


def _noop() -> None:
    return None


def _noop_status(_: grpc.StatusCode) -> None:
    return None


def _noop_decode(_: float, __: float) -> None:
    return None


@dataclass(frozen=True)
class DecodeSchedulerHooks:
    on_error: Callable[[grpc.StatusCode], None] = _noop_status
    on_decode_result: Callable[[float, float], None] = _noop_decode
    on_vad_utterance_end: Callable[[], None] = _noop


class DecodeScheduler:
    """Owns the pool of ModelWorkers and dispenses workers to streams."""

    def __init__(
        self,
        stream_orchestrator: StreamOrchestrator,
        decode_timeout_sec: float,
        language_lookup: SupportedLanguages,
        hooks: DecodeSchedulerHooks | None = None,
    ) -> None:
        self._hooks = hooks or DecodeSchedulerHooks()
        self.stream_orchestrator = stream_orchestrator
        self.decode_timeout_sec = decode_timeout_sec
        self.language_lookup = language_lookup
        self._pending_lock = threading.Lock()
        self._pending_tasks = 0
        self._next_worker = 0

    def new_stream(self) -> "DecodeStream":
        return DecodeStream(self)

    def workers_healthy(self) -> bool:
        return True

    def pending_decodes(self) -> int:
        with self._pending_lock:
            return self._pending_tasks

    def _increment_pending(self) -> None:
        with self._pending_lock:
            self._pending_tasks += 1

    def _decrement_pending(self) -> None:
        with self._pending_lock:
            if self._pending_tasks > 0:
                self._pending_tasks -= 1

    def _on_decode_error(self, status_code: grpc.StatusCode) -> None:
        self._hooks.on_error(status_code)

    def _on_decode_result(self, latency_sec: float, rtf: float) -> None:
        self._hooks.on_decode_result(latency_sec, rtf)

    def _on_vad_utterance_end(self) -> None:
        self._hooks.on_vad_utterance_end()


class DecodeStream:
    """Manages decode futures for a single streaming recognizer call."""

    def __init__(self, scheduler: DecodeScheduler) -> None:
        self.scheduler = scheduler
        self.pending_results: List[Tuple[futures.Future, bool, float, bool]] = []
        self.pending_partials = 0
        self.session_id: Optional[str] = None
        self.model_id: str = DEFAULT_MODEL_ID

    def set_session_id(self, session_id: Optional[str]) -> None:
        self.session_id = session_id

    def set_model_id(self, model_id: str) -> None:
        self.model_id = model_id

    def schedule_decode(
        self,
        pcm: bytes,
        sample_rate: int,
        decode_options: Dict[str, Any] | None,
        is_final: bool,
        offset_sec: float,
        count_vad: bool = False,
    ) -> None:
        if not pcm:
            LOGGER.debug("Skip decode for empty buffer (final=%s)", is_final)
            return

        worker = self.scheduler.stream_orchestrator.acquire_worker(self.model_id)
        future = worker.submit(pcm, sample_rate, decode_options)
        self.scheduler._increment_pending()
        self.pending_results.append((future, is_final, offset_sec, count_vad))
        if not is_final:
            self.pending_partials += 1
        LOGGER.info(
            "Scheduled decode session_id=%s bytes=%d final=%s pending=%d offset=%.2f, model_id=%s",
            self.session_id or "unknown",
            len(pcm),
            is_final,
            len(self.pending_results),
            offset_sec,
            self.model_id,
        )

    def pending_partial_decodes(self) -> int:
        return self.pending_partials

    def has_pending_results(self) -> bool:
        return bool(self.pending_results)

    def emit_ready(self, block: bool) -> Iterable[stt_pb2.STTResult]:
        ready: List[Tuple[futures.Future, bool, float, bool]] = []

        still_pending: List[Tuple[futures.Future, bool, float, bool]] = []
        for future, is_final, offset_sec, count_vad in self.pending_results:
            if future.done():
                ready.append((future, is_final, offset_sec, count_vad))
            else:
                still_pending.append((future, is_final, offset_sec, count_vad))
        self.pending_results[:] = still_pending

        if not ready and block and self.pending_results:
            wait_timeout = (
                self.scheduler.decode_timeout_sec
                if self.scheduler.decode_timeout_sec > 0
                else None
            )
            done, _ = futures.wait(
                [future for future, _, _, _ in self.pending_results],
                timeout=wait_timeout,
                return_when=futures.FIRST_COMPLETED,
            )
            if not done:
                raise TimeoutError(f"ERR2001 Decode timeout after {wait_timeout}s")
            else:
                remaining: List[Tuple[futures.Future, bool, float, bool]] = []
                for future, is_final, offset_sec, count_vad in self.pending_results:
                    if future in done:
                        ready.append((future, is_final, offset_sec, count_vad))
                    else:
                        remaining.append((future, is_final, offset_sec, count_vad))
                self.pending_results[:] = remaining

        for future, is_final, offset_sec, count_vad in ready:
            try:
                result = future.result()
            except Exception as e:
                raise RuntimeError(f"ERR2002 Decode task failed: {e}") from e

            if not is_final and self.pending_partials > 0:
                self.pending_partials -= 1
            if result.latency_sec >= 0:
                self.scheduler._on_decode_result(result.latency_sec, result.rtf)
            language_name = self.scheduler.language_lookup.get_name(
                result.language_code
            )
            for seg in result.segments:
                LOGGER.info(
                    "session_id=%s %s result='%s' [%.2f, %.2f] lang=%s prob=%.2f",
                    self.session_id or "unknown",
                    "final" if is_final else "partial",
                    seg.text,
                    seg.start + offset_sec,
                    seg.end + offset_sec,
                    result.language_code or "auto",
                    (
                        result.language_probability
                        if result.language_probability >= 0
                        else -1.0
                    ),
                )
                yield stt_pb2.STTResult(
                    text=seg.text,
                    is_final=is_final,
                    start_sec=seg.start + offset_sec,
                    end_sec=seg.end + offset_sec,
                    language_code=result.language_code or "",
                    language=language_name,
                    probability=(
                        result.language_probability
                        if result.language_probability >= 0
                        else 0.0
                    ),
                )
            if count_vad:
                self.scheduler._on_vad_utterance_end()
            self.scheduler._decrement_pending()
