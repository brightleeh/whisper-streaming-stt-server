"""Model registry for dynamic model loading and switching."""

import logging
import threading
import time
from concurrent import futures
from dataclasses import dataclass
from queue import Queue
from typing import Any, Dict, List, Optional, Protocol, cast

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
    submitted_at: float
    future: futures.Future


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

            try:
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
        pcm: bytes,
        sample_rate: int,
        decode_options: Optional[Dict[str, Any]],
    ) -> futures.Future:
        """Submit a decode task to the shared queue for the model."""
        with self._lock:
            queue = self._task_queues.get(model_id)
            if not queue:
                if model_id != DEFAULT_MODEL_ID:
                    return self.submit_decode(
                        DEFAULT_MODEL_ID, pcm, sample_rate, decode_options
                    )
                if self._task_queues:
                    fallback_id = next(iter(self._task_queues))
                    return self.submit_decode(
                        fallback_id, pcm, sample_rate, decode_options
                    )
                future = futures.Future()
                future.set_exception(RuntimeError("No model workers available"))
                return future
        future = futures.Future()
        task = _DecodeTask(
            pcm=pcm,
            sample_rate=sample_rate,
            decode_options=decode_options.copy() if decode_options else None,
            submitted_at=time.perf_counter(),
            future=future,
        )
        queue.put(task)
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
            LOGGER.info("Unloaded model '%s'", model_id)
        # Close workers outside the lock to avoid blocking other operations.
        timeout = None
        if drain_timeout_sec is not None:
            timeout = max(0.0, float(drain_timeout_sec))
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
        for queue, threads in queue_threads:
            for _ in threads:
                queue.put(None)
            for thread in threads:
                thread.join(timeout=1.0)
        for pool in pools:
            for worker in pool:
                try:
                    worker.close()
                except (RuntimeError, ValueError) as exc:
                    LOGGER.exception("Failed to close model worker: %s", exc)

    @staticmethod
    def _worker_loop(model_id: str, worker: ModelWorkerProtocol, queue: Queue) -> None:
        while True:
            task = queue.get()
            if task is None:
                queue.task_done()
                break
            if task.future.cancelled():
                queue.task_done()
                continue
            try:
                result = worker.decode_sync(
                    task.pcm,
                    task.sample_rate,
                    task.decode_options,
                    task.submitted_at,
                )
                task.future.set_result(result)
            except Exception as exc:  # pragma: no cover - defensive
                task.future.set_exception(exc)
                LOGGER.exception("Decode task failed for model '%s'", model_id)
            finally:
                queue.task_done()

    @staticmethod
    def _resolve_model_worker() -> ModelWorkerFactory:
        """Return the ModelWorker class, loading it lazily if needed."""
        if ModelWorker is None:
            raise RuntimeError(
                "ModelWorker dependency is missing; install model dependencies."
            )
        return cast(ModelWorkerFactory, ModelWorker)
