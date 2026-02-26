"""Model registry for dynamic model loading and switching."""

import logging
import sys
import threading
import time
from collections import deque
from concurrent import futures
from dataclasses import dataclass
from queue import Queue
from typing import Any, Deque, Dict, List, Optional, Protocol, Set, cast

from stt_server.config.default.model import (
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_LANGUAGE,
    DEFAULT_LANGUAGE_FIX,
    DEFAULT_MODEL_BACKEND,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_POOL_SIZE,
    DEFAULT_TASK,
)

try:
    from stt_server.model.worker import ModelWorker
except ImportError:  # pragma: no cover - optional runtime dependency in some contexts
    ModelWorker = None

LOGGER = logging.getLogger("stt_server.model_registry")


@dataclass(frozen=True)
class _DecodeTask:
    pcm: bytes
    sample_rate: int
    decode_options: Optional[Dict[str, Any]]
    session_id: str
    is_final: bool
    submitted_at: float
    future: futures.Future
    cancel_event: threading.Event


class ModelWorkerProtocol(Protocol):
    """Minimal protocol needed by the model registry."""

    executor: Any

    def pending_tasks(self) -> int:
        """Return count of pending decode tasks."""
        raise NotImplementedError

    def submit(
        self,
        pcm_bytes: bytes,
        src_rate: int,
        decode_options: Optional[Dict[str, Any]] = None,
    ) -> futures.Future:
        """Submit PCM bytes for asynchronous decode."""
        raise NotImplementedError

    def decode_sync(
        self,
        pcm_bytes: bytes,
        src_rate: int,
        decode_options: Optional[Dict[str, Any]],
        submitted_at: float,
    ) -> Any:
        """Decode bytes synchronously."""
        raise NotImplementedError

    def close(self, timeout_sec: Optional[float] = None) -> None:
        """Close worker resources."""
        raise NotImplementedError


class ModelWorkerFactory(Protocol):
    """Callable that builds a model worker."""

    def __call__(
        self,
        model_size: str,
        device: str,
        compute_type: str,
        language: Optional[str],
        log_metrics: bool,
        base_options: Optional[Dict[str, Any]] = None,
        backend: Optional[str] = None,
    ) -> ModelWorkerProtocol:
        """Create a model worker instance."""
        raise NotImplementedError


class ModelRegistry:
    """Manages pools of ModelWorkers keyed by model_id."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._pools: Dict[str, List[ModelWorkerProtocol]] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._rr_counters: Dict[str, int] = {}
        self._model_rr_index = 0
        self._task_queues: Dict[str, Queue] = {}
        self._worker_threads: Dict[str, List[threading.Thread]] = {}
        self._session_queues: Dict[str, Dict[str, Deque[_DecodeTask]]] = {}
        self._session_order: Dict[str, Deque[str]] = {}
        self._session_inflight: Dict[str, Set[str]] = {}
        self._dispatch_conds: Dict[str, threading.Condition] = {}
        self._dispatcher_threads: Dict[str, threading.Thread] = {}
        self._dispatcher_shutdown: Dict[str, bool] = {}
        self._cancel_lock = threading.Lock()
        self._cancel_events: Dict[futures.Future, threading.Event] = {}

    def request_cancel(self, future: futures.Future) -> bool:
        """Signal a running decode to cancel if possible."""
        with self._cancel_lock:
            event = self._cancel_events.get(future)
        if event is None:
            return False
        event.set()
        return True

    def _register_cancel_event(
        self, future: futures.Future, cancel_event: threading.Event
    ) -> None:
        with self._cancel_lock:
            self._cancel_events[future] = cancel_event

    def _clear_cancel_event(self, future: futures.Future) -> None:
        with self._cancel_lock:
            self._cancel_events.pop(future, None)

    def is_loaded(self, model_id: str) -> bool:
        """Check if a model is currently loaded."""
        with self._lock:
            return model_id in self._pools

    def list_models(self) -> List[str]:
        """Return a list of loaded model IDs."""
        with self._lock:
            return list(self._pools.keys())

    def health_summary(self) -> Dict[str, Any]:
        """Return a lightweight health summary of loaded pools and workers."""
        with self._lock:
            pools = {model_id: list(pool) for model_id, pool in self._pools.items()}

        total_workers = 0
        empty_pools = 0
        shutdown_workers = 0
        for pool in pools.values():
            if not pool:
                empty_pools += 1
                continue
            total_workers += len(pool)
            for worker in pool:
                if getattr(worker.executor, "_shutdown", False):
                    shutdown_workers += 1

        return {
            "models_loaded": bool(pools),
            "model_count": len(pools),
            "total_workers": total_workers,
            "empty_pools": empty_pools,
            "shutdown_workers": shutdown_workers,
        }

    def get_next_model_id(self) -> Optional[str]:
        """Get the next model_id in round-robin fashion."""
        with self._lock:
            models = list(self._pools.keys())
            if not models:
                return None
            self._model_rr_index %= len(models)
            model_id = models[self._model_rr_index]
            self._model_rr_index += 1
            return model_id

    def load_model(self, model_id: str, config: Dict[str, Any]) -> None:
        """Load a model pool into the registry."""
        worker_cls = self._resolve_model_worker()
        with self._lock:
            if model_id in self._pools:
                LOGGER.info("Model '%s' is already loaded", model_id)
                return

            LOGGER.info("Loading model '%s' with config=%s", model_id, config)
            try:
                pool_size = int(config.get("pool_size", DEFAULT_MODEL_POOL_SIZE))
            except (TypeError, ValueError) as exc:
                raise ValueError("pool_size must be an integer >= 1") from exc
            if pool_size <= 0:
                raise ValueError("pool_size must be >= 1")
            workers: List[ModelWorkerProtocol] = []

            # Extract worker-specific arguments
            model_path_or_size = (
                config.get("model_path")
                or config.get("model_size")
                or config.get("name")
                or DEFAULT_MODEL_NAME
            )
            device = config.get("device", DEFAULT_DEVICE)
            compute_type = config.get("compute_type", DEFAULT_COMPUTE_TYPE)
            language_fix = config.get("language_fix", DEFAULT_LANGUAGE_FIX)
            language = (
                config.get("language", DEFAULT_LANGUAGE) if language_fix else None
            )
            log_metrics = config.get("log_metrics", False)
            base_options = dict(config.get("base_options") or {})
            backend = config.get("backend") or config.get("model_backend")
            if not backend:
                backend = DEFAULT_MODEL_BACKEND

            # Merge task into base_options if provided
            if "task" in config:
                base_options["task"] = config.get("task", DEFAULT_TASK)

            backend_norm = backend.lower().replace("-", "_")
            if backend_norm == "mlx_whisper" and pool_size > 1:
                raise ValueError(
                    "mlx_whisper does not support pool_size > 1 "
                    "(MLX is not thread-safe). Use pool_size=1 or "
                    "switch to torch_whisper for multi-worker pools."
                )

            try:
                self._validate_device_backend(backend, device)
                for i in range(pool_size):
                    LOGGER.debug(
                        "Initializing worker %d/%d for model '%s'",
                        i + 1,
                        pool_size,
                        model_id,
                    )
                    workers.append(
                        worker_cls(
                            model_size=model_path_or_size,
                            device=device,
                            compute_type=compute_type,
                            language=language,
                            log_metrics=log_metrics,
                            base_options=base_options,
                            backend=backend,
                        )
                    )

                self._pools[model_id] = workers
                self._configs[model_id] = config
                self._rr_counters[model_id] = 0
                queue: Queue = Queue()
                self._task_queues[model_id] = queue
                self._session_queues[model_id] = {}
                self._session_order[model_id] = deque()
                self._session_inflight[model_id] = set()
                self._dispatch_conds[model_id] = threading.Condition()
                self._dispatcher_shutdown[model_id] = False
                threads: List[threading.Thread] = []
                for idx, worker in enumerate(workers):
                    thread = threading.Thread(
                        target=self._worker_loop,
                        args=(model_id, worker, queue),
                        name=f"model-worker-{model_id}-{idx}",
                        daemon=True,
                    )
                    thread.start()
                    threads.append(thread)
                self._worker_threads[model_id] = threads
                dispatcher = threading.Thread(
                    target=self._dispatch_loop,
                    args=(model_id,),
                    name=f"model-dispatcher-{model_id}",
                    daemon=True,
                )
                dispatcher.start()
                self._dispatcher_threads[model_id] = dispatcher
                LOGGER.info(
                    "Successfully loaded model '%s' (pool_size=%d)", model_id, pool_size
                )
            except (RuntimeError, ValueError, OSError, TypeError):
                LOGGER.exception("Failed to load model '%s'", model_id)
                for worker in workers:
                    try:
                        worker.close()
                    except (RuntimeError, ValueError, OSError) as exc:
                        LOGGER.exception("Failed to close model worker: %s", exc)
                workers.clear()
                raise

    def _validate_device_backend(self, backend: str, device: str) -> None:
        device_norm = (device or "").lower()
        backend_norm = (backend or "").lower()
        if not device_norm or device_norm == "cpu" or device_norm == "auto":
            return
        if device_norm.startswith("cuda"):
            if sys.platform == "darwin":
                raise ValueError("CUDA device requested on macOS")
            if backend_norm == "torch_whisper":
                try:
                    import torch
                except ImportError as exc:  # pragma: no cover - torch optional
                    raise ValueError(
                        "CUDA requested but torch is not available"
                    ) from exc
                if not torch.cuda.is_available():
                    raise ValueError("CUDA requested but torch reports no CUDA devices")
            return
        if device_norm == "mps":
            if backend_norm not in ("torch_whisper", "mlx_whisper"):
                raise ValueError(
                    "MPS device requires torch_whisper or mlx_whisper backend"
                )
            if sys.platform != "darwin":
                raise ValueError("MPS device requested on non-macOS platform")
            if backend_norm == "mlx_whisper":
                try:
                    import mlx.core  # noqa: F401
                except ImportError as exc:  # pragma: no cover - mlx optional
                    raise ValueError(
                        "MPS requested with mlx_whisper but mlx package is not available"
                    ) from exc
            else:
                try:
                    import torch
                except ImportError as exc:  # pragma: no cover - torch optional
                    raise ValueError(
                        "MPS requested but torch is not available"
                    ) from exc
                mps_backend = getattr(torch.backends, "mps", None)
                if not (mps_backend and mps_backend.is_available()):
                    raise ValueError("MPS requested but torch reports no MPS device")
            return
        if device_norm == "mlx":
            if backend_norm != "mlx_whisper":
                raise ValueError("MLX device requires mlx_whisper backend")
            if sys.platform != "darwin":
                raise ValueError("MLX device requested on non-macOS platform")
            try:
                import mlx.core  # noqa: F401
            except ImportError as exc:  # pragma: no cover - mlx optional
                raise ValueError(
                    "MLX requested but mlx package is not available"
                ) from exc

    def get_worker(self, model_id: str) -> Optional[ModelWorkerProtocol]:
        """Acquire a worker for the given model_id (Round-Robin)."""
        with self._lock:
            pool = self._pools.get(model_id)
            if not pool:
                # Fallback to default if requested model not found
                if model_id != DEFAULT_MODEL_ID:
                    LOGGER.warning(
                        "Model '%s' not found, falling back to '%s'",
                        model_id,
                        DEFAULT_MODEL_ID,
                    )
                    return self.get_worker(DEFAULT_MODEL_ID)

                if self._pools:
                    fallback_id = next(iter(self._pools))
                    LOGGER.warning(
                        "Model '%s' not found, falling back to available model '%s'",
                        model_id,
                        fallback_id,
                    )
                    return self.get_worker(fallback_id)
                return None

            start_idx = self._rr_counters[model_id]
            best_idx = start_idx
            best_pending = None
            for offset in range(len(pool)):
                idx = (start_idx + offset) % len(pool)
                pending = pool[idx].pending_tasks()
                if best_pending is None or pending < best_pending:
                    best_pending = pending
                    best_idx = idx
                    if pending == 0:
                        break
            worker = pool[best_idx]
            self._rr_counters[model_id] = (best_idx + 1) % len(pool)
            return worker

    def submit_decode(
        self,
        model_id: str,
        session_id: str,
        pcm: bytes,
        sample_rate: int,
        decode_options: Optional[Dict[str, Any]],
        is_final: bool,
    ) -> futures.Future:
        """Submit a decode task to the shared queue for the model."""
        with self._lock:
            if model_id not in self._task_queues:
                if model_id != DEFAULT_MODEL_ID:
                    return self.submit_decode(
                        DEFAULT_MODEL_ID,
                        session_id,
                        pcm,
                        sample_rate,
                        decode_options,
                        is_final,
                    )
                if self._task_queues:
                    fallback_id = next(iter(self._task_queues))
                    return self.submit_decode(
                        fallback_id,
                        session_id,
                        pcm,
                        sample_rate,
                        decode_options,
                        is_final,
                    )
                future = futures.Future()
                future.set_exception(RuntimeError("No model workers available"))
                return future
            session_queues = self._session_queues.get(model_id)
            order = self._session_order.get(model_id)
            condition = self._dispatch_conds.get(model_id)
            if session_queues is None or order is None or condition is None:
                future = futures.Future()
                future.set_exception(RuntimeError("Decode dispatcher unavailable"))
                return future
        future = futures.Future()
        cancel_event = threading.Event()
        self._register_cancel_event(future, cancel_event)
        task = _DecodeTask(
            pcm=pcm,
            sample_rate=sample_rate,
            decode_options=decode_options.copy() if decode_options else None,
            session_id=session_id or "unknown",
            is_final=is_final,
            submitted_at=time.perf_counter(),
            future=future,
            cancel_event=cancel_event,
        )
        with condition:
            queue = session_queues.setdefault(task.session_id, deque())
            if task.is_final:
                self._cancel_stale_partials(queue)
            queue.append(task)
            if task.session_id not in order:
                order.append(task.session_id)
            LOGGER.debug(
                "Enqueued decode task session_id=%s final=%s model_id=%s queue_len=%d",
                task.session_id,
                task.is_final,
                model_id,
                len(queue),
            )
            condition.notify_all()
        return future

    def unload_model(
        self, model_id: str, drain_timeout_sec: Optional[float] = None
    ) -> bool:
        """Unload a model to free resources (blocks until workers finish)."""
        workers: List[ModelWorkerProtocol] = []
        with self._lock:
            if model_id not in self._pools:
                return False

            if len(self._pools) <= 1:
                LOGGER.warning("Cannot unload the last remaining model '%s'", model_id)
                return False

            workers = self._pools.pop(model_id)
            del self._configs[model_id]
            del self._rr_counters[model_id]
            queue = self._task_queues.pop(model_id, None)
            threads = self._worker_threads.pop(model_id, [])
            self._session_queues.pop(model_id, None)
            self._session_order.pop(model_id, None)
            self._session_inflight.pop(model_id, None)
            dispatcher = self._dispatcher_threads.pop(model_id, None)
            condition = self._dispatch_conds.pop(model_id, None)
            self._dispatcher_shutdown[model_id] = True
            LOGGER.info("Unloaded model '%s'", model_id)
        # Close workers outside the lock to avoid blocking other operations.
        timeout = None
        if drain_timeout_sec is not None:
            timeout = max(0.0, float(drain_timeout_sec))
        if dispatcher and condition:
            with condition:
                condition.notify_all()
            dispatcher.join(timeout=timeout)
        if queue is not None and threads:
            for _ in threads:
                queue.put(None)
            for thread in threads:
                thread.join(timeout=timeout)
        for worker in workers:
            try:
                worker.close(timeout)
            except (RuntimeError, ValueError) as exc:
                LOGGER.exception("Failed to close model worker: %s", exc)
        # Force garbage collection advice could be placed here
        return True

    def close(self) -> None:
        """Close all model workers and clear registry state."""
        with self._lock:
            pools = list(self._pools.values())
            self._pools.clear()
            self._configs.clear()
            self._rr_counters.clear()
            queue_threads = [
                (self._task_queues[model_id], threads)
                for model_id, threads in self._worker_threads.items()
                if model_id in self._task_queues
            ]
            self._task_queues.clear()
            self._worker_threads.clear()
            dispatchers = list(self._dispatcher_threads.values())
            conditions = list(self._dispatch_conds.values())
            self._dispatcher_threads.clear()
            self._dispatch_conds.clear()
            self._session_queues.clear()
            self._session_order.clear()
            self._session_inflight.clear()
            for model_id in list(self._dispatcher_shutdown.keys()):
                self._dispatcher_shutdown[model_id] = True
        for queue, threads in queue_threads:
            for _ in threads:
                queue.put(None)
            for thread in threads:
                thread.join(timeout=1.0)
        for condition in conditions:
            with condition:
                condition.notify_all()
        for dispatcher in dispatchers:
            dispatcher.join(timeout=1.0)
        for pool in pools:
            for worker in pool:
                try:
                    worker.close()
                except (RuntimeError, ValueError) as exc:
                    LOGGER.exception("Failed to close model worker: %s", exc)

    def _skip_cancelled_task(
        self, model_id: str, task: _DecodeTask, queue: Queue
    ) -> bool:
        if task.future.cancelled():
            self._release_inflight(model_id, task.session_id)
            self._clear_cancel_event(task.future)
            queue.task_done()
            return True
        if not task.future.set_running_or_notify_cancel():
            self._release_inflight(model_id, task.session_id)
            self._clear_cancel_event(task.future)
            queue.task_done()
            return True
        if task.cancel_event.is_set():
            if not task.future.done():
                task.future.set_exception(futures.CancelledError())
            self._release_inflight(model_id, task.session_id)
            self._clear_cancel_event(task.future)
            queue.task_done()
            return True
        return False

    def _worker_loop(
        self, model_id: str, worker: ModelWorkerProtocol, queue: Queue
    ) -> None:
        while True:
            task = queue.get()
            if task is None:
                queue.task_done()
                break
            if self._skip_cancelled_task(model_id, task, queue):
                continue
            try:
                LOGGER.debug(
                    "Worker starting decode session_id=%s final=%s model_id=%s bytes=%d",
                    task.session_id,
                    task.is_final,
                    model_id,
                    len(task.pcm),
                )
                result = worker.decode_sync(
                    task.pcm,
                    task.sample_rate,
                    task.decode_options,
                    task.submitted_at,
                )
                if task.future.cancelled() or task.cancel_event.is_set():
                    if not task.future.done():
                        task.future.set_exception(futures.CancelledError())
                else:
                    task.future.set_result(result)
                LOGGER.debug(
                    "Worker finished decode session_id=%s final=%s model_id=%s",
                    task.session_id,
                    task.is_final,
                    model_id,
                )
            except Exception as exc:  # pragma: no cover - defensive
                if not task.future.cancelled():
                    task.future.set_exception(exc)
                LOGGER.exception("Decode task failed for model '%s'", model_id)
            finally:
                self._release_inflight(model_id, task.session_id)
                self._clear_cancel_event(task.future)
                queue.task_done()

    def _dispatch_loop(self, model_id: str) -> None:
        condition = self._dispatch_conds[model_id]
        task_queue = self._task_queues[model_id]
        while True:
            try:
                with condition:
                    task = self._pop_next_task(model_id)
                    while task is None:
                        if self._dispatcher_shutdown.get(model_id, False):
                            return
                        condition.wait(timeout=0.1)
                        task = self._pop_next_task(model_id)
                LOGGER.debug(
                    "Dispatching decode task session_id=%s final=%s model_id=%s",
                    task.session_id,
                    task.is_final,
                    model_id,
                )
                task_queue.put(task)
            except Exception:  # pragma: no cover - defensive logging
                LOGGER.exception("Dispatcher loop failed for model '%s'", model_id)
                return

    def _pop_next_task(self, model_id: str) -> Optional[_DecodeTask]:
        session_queues = self._session_queues.get(model_id)
        order = self._session_order.get(model_id)
        inflight = self._session_inflight.get(model_id)
        if session_queues is None or order is None or inflight is None:
            return None
        if not order:
            return None
        checks = len(order)
        while checks > 0 and order:
            session_id = order.popleft()
            checks -= 1
            queue = session_queues.get(session_id)
            if not queue:
                session_queues.pop(session_id, None)
                continue
            if session_id in inflight:
                order.append(session_id)
                continue
            task = self._select_task(queue)
            if task is None:
                session_queues.pop(session_id, None)
                continue
            inflight.add(session_id)
            if queue:
                order.append(session_id)
            else:
                session_queues.pop(session_id, None)
            return task
        return None

    def _select_task(self, queue: Deque[_DecodeTask]) -> Optional[_DecodeTask]:
        if not queue:
            return None
        if any(task.is_final for task in queue):
            self._cancel_stale_partials(queue)
        return queue.popleft() if queue else None

    def _cancel_stale_partials(self, queue: Deque[_DecodeTask]) -> None:
        if not queue:
            return
        kept: Deque[_DecodeTask] = deque()
        for task in queue:
            if task.is_final:
                kept.append(task)
            else:
                task.future.cancel()
                self._clear_cancel_event(task.future)
        queue.clear()
        queue.extend(kept)

    def _release_inflight(self, model_id: str, session_id: str) -> None:
        condition = self._dispatch_conds.get(model_id)
        inflight = self._session_inflight.get(model_id)
        if inflight is None:
            return
        if condition is None:
            inflight.discard(session_id)
            LOGGER.debug(
                "Decode task completed session_id=%s model_id=%s inflight=%d",
                session_id,
                model_id,
                len(inflight),
            )
            return
        with condition:
            inflight.discard(session_id)
            LOGGER.debug(
                "Decode task completed session_id=%s model_id=%s inflight=%d",
                session_id,
                model_id,
                len(inflight),
            )
            condition.notify_all()

    @staticmethod
    def _resolve_model_worker() -> ModelWorkerFactory:
        """Return the ModelWorker class, loading it lazily if needed."""
        if ModelWorker is None:
            raise RuntimeError(
                "ModelWorker dependency is missing; install model dependencies."
            )
        return cast(ModelWorkerFactory, ModelWorker)
