"""Decode worker pool and per-stream scheduling."""

from __future__ import annotations

import threading
import time
from collections import deque
from concurrent import futures
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple

import grpc

from gen.stt.python.v1 import stt_pb2
from stt_server.config.default.model import DEFAULT_MODEL_ID
from stt_server.config.languages import SupportedLanguages
from stt_server.errors import ErrorCode, STTError
from stt_server.utils.logger import LOGGER

if TYPE_CHECKING:
    from stt_server.backend.application.model_registry import ModelWorkerProtocol
    from stt_server.backend.application.stream_orchestrator import StreamOrchestrator


def _noop() -> None:
    return None


def _noop_status(_: grpc.StatusCode) -> None:
    return None


def _noop_decode(_: float, __: float, ___: float, ____: float, _____: float) -> None:
    return None


def _noop_count(_: int) -> None:
    return None


@dataclass(frozen=True)
class DecodeSchedulerHooks:
    """Callback hooks invoked by the decode scheduler."""

    on_error: Callable[[grpc.StatusCode], None] = _noop_status
    on_decode_result: Callable[[float, float, float, float, float], None] = _noop_decode
    on_vad_utterance_end: Callable[[], None] = _noop
    on_decode_cancelled: Callable[[int], None] = _noop_count
    on_decode_orphaned: Callable[[int], None] = _noop_count


class DecodeScheduler:
    """Owns the pool of ModelWorkers and dispenses workers to streams."""

    def __init__(
        self,
        stream_orchestrator: StreamOrchestrator,
        decode_timeout_sec: float,
        language_lookup: SupportedLanguages,
        max_pending_decodes_global: int | None = None,
        health_window_sec: float = 60.0,
        health_min_events: int = 5,
        health_max_timeout_ratio: float = 0.5,
        health_min_success_ratio: float = 0.5,
        log_transcripts: bool = False,
        hooks: DecodeSchedulerHooks | None = None,
    ) -> None:
        self._hooks = hooks or DecodeSchedulerHooks()
        self.stream_orchestrator = stream_orchestrator
        self.decode_timeout_sec = decode_timeout_sec
        self.language_lookup = language_lookup
        self._pending_lock = threading.Lock()
        self._pending_tasks = 0
        self._next_worker = 0
        self._health_lock = threading.Lock()
        self._health_events: "deque[tuple[float, str, int]]" = deque()
        self._health_window_sec = max(1.0, float(health_window_sec))
        self._health_min_events = max(1, int(health_min_events))
        self._health_max_timeout_ratio = max(0.0, min(1.0, health_max_timeout_ratio))
        self._health_min_success_ratio = max(0.0, min(1.0, health_min_success_ratio))
        self.log_transcripts = log_transcripts
        self._pending_limit = (
            max(0, int(max_pending_decodes_global))
            if max_pending_decodes_global is not None
            else 0
        )
        self._pending_sem = (
            threading.BoundedSemaphore(self._pending_limit)
            if self._pending_limit > 0
            else None
        )

    def new_stream(self) -> "DecodeStream":
        """Create a decode stream associated with this scheduler."""
        return DecodeStream(self)

    def acquire_pending_slot(self, block: bool, timeout: float | None) -> bool:
        """Acquire a global pending-decode slot if configured."""
        if not self._pending_sem:
            return True
        if not block:
            return self._pending_sem.acquire(blocking=False)
        return self._pending_sem.acquire(timeout=timeout)

    def release_pending_slot(self) -> None:
        """Release a pending-decode slot when one was acquired."""
        if not self._pending_sem:
            return
        try:
            self._pending_sem.release()
        except ValueError:
            pass

    def workers_healthy(self) -> bool:
        """Return True when worker pools and health ratios are acceptable."""
        registry_summary = self.stream_orchestrator.model_registry.health_summary()
        if not registry_summary["models_loaded"]:
            return False
        if registry_summary["total_workers"] <= 0:
            return False
        if registry_summary["empty_pools"] > 0:
            return False
        if registry_summary["shutdown_workers"] > 0:
            return False

        counts = self._health_counts()
        total = counts["success"] + counts["timeout"] + counts["error"]
        if total < self._health_min_events:
            return True
        timeout_ratio = counts["timeout"] / total if total else 0.0
        success_ratio = counts["success"] / total if total else 0.0
        if timeout_ratio >= self._health_max_timeout_ratio:
            return False
        if success_ratio < self._health_min_success_ratio:
            return False
        return True

    def pending_decodes(self) -> int:
        """Return the number of pending decode tasks."""
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

    def _on_decode_result(
        self,
        inference_sec: float,
        real_time_factor: float,
        queue_wait_sec: float,
        buffer_wait_sec: float,
        response_emit_sec: float,
    ) -> None:
        self._hooks.on_decode_result(
            inference_sec,
            real_time_factor,
            queue_wait_sec,
            buffer_wait_sec,
            response_emit_sec,
        )

    def _record_health_event(self, outcome: str, count: int = 1) -> None:
        if count <= 0:
            return
        now = time.monotonic()
        with self._health_lock:
            self._health_events.append((now, outcome, count))
            self._trim_health_events(now)

    def _trim_health_events(self, now: Optional[float] = None) -> None:
        if now is None:
            now = time.monotonic()
        cutoff = now - self._health_window_sec
        while self._health_events and self._health_events[0][0] < cutoff:
            self._health_events.popleft()

    def _health_counts(self) -> Dict[str, int]:
        with self._health_lock:
            self._trim_health_events()
            counts = {"success": 0, "timeout": 0, "error": 0}
            for _, outcome, count in self._health_events:
                if outcome in counts:
                    counts[outcome] += count
            return counts

    def _on_vad_utterance_end(self) -> None:
        self._hooks.on_vad_utterance_end()

    def _on_decode_cancelled(self, count: int) -> None:
        self._hooks.on_decode_cancelled(count)

    def _on_decode_orphaned(self, count: int) -> None:
        self._hooks.on_decode_orphaned(count)


class DecodeStream:
    """Manages decode futures for a single streaming recognizer call."""

    def __init__(self, scheduler: DecodeScheduler) -> None:
        self.scheduler = scheduler
        self.pending_results: List[
            Tuple[futures.Future, bool, float, bool, float, bool]
        ] = []
        self.pending_partials = 0
        self.session_id: Optional[str] = None
        self.model_id: str = DEFAULT_MODEL_ID
        self._lock = threading.Lock()
        self._queue_wait_total = 0.0
        self._infer_total = 0.0
        self._buffer_wait_total = 0.0
        self._response_emit_total = 0.0
        self._decode_count = 0

    def set_session_id(self, session_id: Optional[str]) -> None:
        """Associate this decode stream with a session identifier."""
        self.session_id = session_id

    def set_model_id(self, model_id: str) -> None:
        """Associate this decode stream with a model identifier."""
        self.model_id = model_id

    def schedule_decode(
        self,
        pcm: bytes,
        sample_rate: int,
        decode_options: Dict[str, Any] | None,
        is_final: bool,
        offset_sec: float,
        count_vad: bool = False,
        buffer_started_at: Optional[float] = None,
        holds_slot: bool = False,
    ) -> None:
        """Submit a decode request and track its pending result."""
        if not pcm:
            LOGGER.debug("Skip decode for empty buffer (final=%s)", is_final)
            if holds_slot:
                self.scheduler.release_pending_slot()
            return

        worker: ModelWorkerProtocol = self.scheduler.stream_orchestrator.acquire_worker(
            self.model_id
        )
        future = worker.submit(pcm, sample_rate, decode_options)
        buffer_wait_sec = (
            max(0.0, time.perf_counter() - buffer_started_at)
            if buffer_started_at is not None
            else 0.0
        )
        self.scheduler._increment_pending()
        with self._lock:
            self.pending_results.append(
                (future, is_final, offset_sec, count_vad, buffer_wait_sec, holds_slot)
            )
            if not is_final:
                self.pending_partials += 1
            pending_count = len(self.pending_results)
        LOGGER.info(
            "Scheduled decode session_id=%s bytes=%d final=%s pending=%d offset=%.2f, model_id=%s",
            self.session_id or "unknown",
            len(pcm),
            is_final,
            pending_count,
            offset_sec,
            self.model_id,
        )

    def _finalize_pending(self, is_final: bool, holds_slot: bool) -> None:
        with self._lock:
            if not is_final and self.pending_partials > 0:
                self.pending_partials -= 1
        self.scheduler._decrement_pending()
        if holds_slot:
            self.scheduler.release_pending_slot()

    def _drop_pending_results(self) -> None:
        with self._lock:
            if not self.pending_results:
                return
            dropped = list(self.pending_results)
            self.pending_results.clear()
            partials_to_drop = sum(
                1 for _, is_final, _, _, _, _ in dropped if not is_final
            )
            if partials_to_drop:
                self.pending_partials = max(0, self.pending_partials - partials_to_drop)
        cancelled = 0
        orphaned = 0
        for future, _, _, _, _, holds_slot in dropped:
            if future.cancel():
                cancelled += 1
            else:
                orphaned += 1
            self.scheduler._decrement_pending()
            if holds_slot:
                self.scheduler.release_pending_slot()
        if cancelled:
            self.scheduler._on_decode_cancelled(cancelled)
        if orphaned:
            self.scheduler._on_decode_orphaned(orphaned)

    def pending_partial_decodes(self) -> int:
        """Return the number of pending partial decodes."""
        with self._lock:
            return self.pending_partials

    def pending_count(self) -> int:
        """Return the total number of pending decode results."""
        with self._lock:
            return len(self.pending_results)

    def drop_pending_partials(self, max_drop: Optional[int] = None) -> Tuple[int, int]:
        """Cancel pending partial decodes and return (cancelled, orphaned)."""
        if max_drop is not None and max_drop <= 0:
            return 0, 0
        dropped: List[Tuple[futures.Future, bool, float, bool, float, bool]] = []
        with self._lock:
            if not self.pending_results:
                return 0, 0
            remaining: List[Tuple[futures.Future, bool, float, bool, float, bool]] = []
            remaining_drop = max_drop if max_drop is not None else float("inf")
            for item in self.pending_results:
                if remaining_drop > 0 and not item[1]:
                    dropped.append(item)
                    remaining_drop -= 1
                else:
                    remaining.append(item)
            if not dropped:
                return 0, 0
            self.pending_results[:] = remaining
            self.pending_partials = max(0, self.pending_partials - len(dropped))
        cancelled = 0
        orphaned = 0
        for future, _, _, _, _, holds_slot in dropped:
            if future.cancel():
                cancelled += 1
            else:
                orphaned += 1
            self.scheduler._decrement_pending()
            if holds_slot:
                self.scheduler.release_pending_slot()
        if cancelled:
            self.scheduler._on_decode_cancelled(cancelled)
        if orphaned:
            self.scheduler._on_decode_orphaned(orphaned)
        return cancelled, orphaned

    def has_pending_results(self) -> bool:
        """Return True when any pending decode results remain."""
        with self._lock:
            return bool(self.pending_results)

    def cancel_pending(self) -> Tuple[int, int]:
        """Cancel all pending decodes and return (cancelled, orphaned)."""
        with self._lock:
            if not self.pending_results:
                return 0, 0
            pending = list(self.pending_results)
            self.pending_results.clear()
            partials_to_drop = sum(
                1 for _, is_final, _, _, _, _ in pending if not is_final
            )
            if partials_to_drop:
                self.pending_partials = max(0, self.pending_partials - partials_to_drop)
        cancelled = 0
        not_cancelled = 0
        for future, _, _, _, _, holds_slot in pending:
            if future.cancel():
                cancelled += 1
            else:
                not_cancelled += 1
            self.scheduler._decrement_pending()
            if holds_slot:
                self.scheduler.release_pending_slot()
        if cancelled:
            self.scheduler._on_decode_cancelled(cancelled)
        if not_cancelled:
            self.scheduler._on_decode_orphaned(not_cancelled)
        return cancelled, not_cancelled

    def timing_summary(self) -> Tuple[float, float, float, float, int]:
        """Return aggregate timing totals and decode count."""
        with self._lock:
            return (
                self._buffer_wait_total,
                self._queue_wait_total,
                self._infer_total,
                self._response_emit_total,
                self._decode_count,
            )

    def _record_timing(
        self,
        buffer_wait_sec: float,
        queue_wait_sec: float,
        inference_sec: float,
        response_emit_sec: float,
    ) -> None:
        with self._lock:
            if buffer_wait_sec >= 0:
                self._buffer_wait_total += buffer_wait_sec
            if queue_wait_sec >= 0:
                self._queue_wait_total += queue_wait_sec
            if inference_sec >= 0:
                self._infer_total += inference_sec
            if response_emit_sec >= 0:
                self._response_emit_total += response_emit_sec
            self._decode_count += 1

    def emit_ready(self, block: bool) -> Iterable[stt_pb2.STTResult]:
        """Yield ready decode results, optionally blocking for completion."""
        ready: List[Tuple[futures.Future, bool, float, bool, float, bool]] = []

        with self._lock:
            still_pending: List[
                Tuple[futures.Future, bool, float, bool, float, bool]
            ] = []
            for (
                future,
                is_final,
                offset_sec,
                count_vad,
                buffer_wait_sec,
                holds_slot,
            ) in self.pending_results:
                if future.done():
                    ready.append(
                        (
                            future,
                            is_final,
                            offset_sec,
                            count_vad,
                            buffer_wait_sec,
                            holds_slot,
                        )
                    )
                else:
                    still_pending.append(
                        (
                            future,
                            is_final,
                            offset_sec,
                            count_vad,
                            buffer_wait_sec,
                            holds_slot,
                        )
                    )
            self.pending_results[:] = still_pending
            pending_snapshot = list(self.pending_results)

        if not ready and block and pending_snapshot:
            wait_timeout = (
                self.scheduler.decode_timeout_sec
                if self.scheduler.decode_timeout_sec > 0
                else None
            )
            done, _ = futures.wait(
                [future for future, _, _, _, _, _ in pending_snapshot],
                timeout=wait_timeout,
                return_when=futures.FIRST_COMPLETED,
            )
            if not done:
                self.scheduler._on_decode_error(grpc.StatusCode.INTERNAL)
                self.scheduler._record_health_event("timeout", len(pending_snapshot))
                self._drop_pending_results()
                detail = (
                    f"decode timeout after {wait_timeout}s"
                    if wait_timeout is not None
                    else None
                )
                raise STTError(ErrorCode.DECODE_TIMEOUT, detail)
            else:
                with self._lock:
                    remaining: List[
                        Tuple[futures.Future, bool, float, bool, float, bool]
                    ] = []
                    for (
                        future,
                        is_final,
                        offset_sec,
                        count_vad,
                        buffer_wait_sec,
                        holds_slot,
                    ) in self.pending_results:
                        if future in done:
                            ready.append(
                                (
                                    future,
                                    is_final,
                                    offset_sec,
                                    count_vad,
                                    buffer_wait_sec,
                                    holds_slot,
                                )
                            )
                        else:
                            remaining.append(
                                (
                                    future,
                                    is_final,
                                    offset_sec,
                                    count_vad,
                                    buffer_wait_sec,
                                    holds_slot,
                                )
                            )
                    self.pending_results[:] = remaining

        for (
            future,
            is_final,
            offset_sec,
            count_vad,
            buffer_wait_sec,
            holds_slot,
        ) in ready:
            try:
                result = future.result()
            except STTError as exc:
                self.scheduler._on_decode_error(exc.status)
                self.scheduler._record_health_event("error")
                self._finalize_pending(is_final, holds_slot)
                raise
            except (
                futures.CancelledError,
                futures.TimeoutError,
                RuntimeError,
                ValueError,
                TypeError,
                OSError,
            ) as exc:
                self.scheduler._on_decode_error(grpc.StatusCode.INTERNAL)
                self.scheduler._record_health_event("error")
                self._finalize_pending(is_final, holds_slot)
                raise STTError(
                    ErrorCode.DECODE_TASK_FAILED, f"decode task failed: {exc}"
                ) from exc

            language_name = self.scheduler.language_lookup.get_name(
                result.language_code
            )
            emit_start = time.perf_counter()
            for seg in result.segments:
                text = seg.text or ""
                if self.scheduler.log_transcripts:
                    LOGGER.info(
                        "session_id=%s %s result='%s' [%.2f, %.2f] lang=%s prob=%.2f",
                        self.session_id or "unknown",
                        "final" if is_final else "partial",
                        text,
                        seg.start + offset_sec,
                        seg.end + offset_sec,
                        result.language_code or "auto",
                        (
                            result.language_probability
                            if result.language_probability >= 0
                            else -1.0
                        ),
                    )
                else:
                    LOGGER.debug(
                        "session_id=%s %s result_len=%d [%.2f, %.2f] lang=%s prob=%.2f",
                        self.session_id or "unknown",
                        "final" if is_final else "partial",
                        len(text),
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
            response_emit_sec = max(0.0, time.perf_counter() - emit_start)
            if count_vad:
                self.scheduler._on_vad_utterance_end()
            if result.latency_sec >= 0:
                self.scheduler._on_decode_result(
                    result.latency_sec,
                    result.rtf,
                    result.queue_wait_sec,
                    buffer_wait_sec,
                    response_emit_sec,
                )
                self.scheduler._record_health_event("success")
                self._record_timing(
                    buffer_wait_sec,
                    result.queue_wait_sec,
                    result.latency_sec,
                    response_emit_sec,
                )
                total_sec = (
                    buffer_wait_sec
                    + result.queue_wait_sec
                    + result.latency_sec
                    + response_emit_sec
                )
                LOGGER.info(
                    "decode_timing session_id=%s final=%s buffer_wait=%.2fs queue_wait=%.2fs inference=%.2fs response_emit=%.2fs total=%.2fs audio_duration=%.2fs real_time_factor=%.2f",
                    self.session_id or "unknown",
                    is_final,
                    buffer_wait_sec,
                    result.queue_wait_sec,
                    result.latency_sec,
                    response_emit_sec,
                    total_sec,
                    result.audio_duration,
                    result.rtf if result.rtf >= 0 else -1.0,
                )
            self._finalize_pending(is_final, holds_slot)
