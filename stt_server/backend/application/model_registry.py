"""Model registry for dynamic model loading and switching."""

import logging
import threading
from typing import Any, Dict, List, Optional

from stt_server.config.default.model import (
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_LANGUAGE,
    DEFAULT_LANGUAGE_FIX,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_POOL_SIZE,
    DEFAULT_TASK,
)
from stt_server.model.worker import ModelWorker

LOGGER = logging.getLogger("stt_server.model_registry")


class ModelRegistry:
    """Manages pools of ModelWorkers keyed by model_id."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._pools: Dict[str, List[ModelWorker]] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._rr_counters: Dict[str, int] = {}
        self._model_rr_index = 0

    def is_loaded(self, model_id: str) -> bool:
        """Check if a model is currently loaded."""
        with self._lock:
            return model_id in self._pools

    def list_models(self) -> List[str]:
        """Return a list of loaded model IDs."""
        with self._lock:
            return list(self._pools.keys())

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
        with self._lock:
            if model_id in self._pools:
                LOGGER.info("Model '%s' is already loaded", model_id)
                return

            LOGGER.info("Loading model '%s' with config=%s", model_id, config)
            pool_size = config.get("pool_size", DEFAULT_MODEL_POOL_SIZE)
            workers: List[ModelWorker] = []

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
            base_options = config.get("base_options", {})

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
                        ModelWorker(
                            model_size=model_path_or_size,
                            device=device,
                            compute_type=compute_type,
                            language=language,
                            log_metrics=log_metrics,
                            base_options=base_options,
                        )
                    )

                self._pools[model_id] = workers
                self._configs[model_id] = config
                self._rr_counters[model_id] = 0
                LOGGER.info(
                    "Successfully loaded model '%s' (pool_size=%d)", model_id, pool_size
                )
            except Exception:
                LOGGER.exception("Failed to load model '%s'", model_id)
                # Cleanup partially loaded workers
                for w in workers:
                    # ModelWorker doesn't have explicit close, but we drop references
                    pass
                raise

    def get_worker(self, model_id: str) -> Optional[ModelWorker]:
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

            idx = self._rr_counters[model_id]
            worker = pool[idx]
            self._rr_counters[model_id] = (idx + 1) % len(pool)
            return worker

    def unload_model(
        self, model_id: str, drain_timeout_sec: Optional[float] = None
    ) -> bool:
        """Unload a model to free resources (blocks until workers finish)."""
        workers: List[ModelWorker] = []
        with self._lock:
            if model_id not in self._pools:
                return False

            if len(self._pools) <= 1:
                LOGGER.warning("Cannot unload the last remaining model '%s'", model_id)
                return False

            workers = self._pools.pop(model_id)
            del self._configs[model_id]
            del self._rr_counters[model_id]
            LOGGER.info("Unloaded model '%s'", model_id)
        # Close workers outside the lock to avoid blocking other operations.
        timeout = None
        if drain_timeout_sec is not None:
            timeout = max(0.0, float(drain_timeout_sec))
        for worker in workers:
            try:
                worker.close(timeout)
            except Exception:
                LOGGER.exception("Failed to close model worker")
        # Force garbage collection advice could be placed here
        return True

    def close(self) -> None:
        """Close all model workers and clear registry state."""
        with self._lock:
            pools = list(self._pools.values())
            self._pools.clear()
            self._configs.clear()
            self._rr_counters.clear()
        for pool in pools:
            for worker in pool:
                try:
                    worker.close()
                except Exception:
                    LOGGER.exception("Failed to close model worker")
