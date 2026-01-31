from typing import List, cast

import pytest

from stt_server.backend.application import model_registry
from stt_server.backend.application.model_registry import (
    ModelRegistry,
    ModelWorkerProtocol,
)
from stt_server.config.default.model import DEFAULT_MODEL_ID


def test_unload_model_closes_workers(monkeypatch):
    """Test unload model closes workers."""
    closed = []

    class FakeWorker:
        """Test helper FakeWorker."""

        def __init__(self, *args, **kwargs):
            """Helper for   init  ."""
            self.args = args
            self.kwargs = kwargs

        def close(self, *args, **kwargs) -> None:
            """Helper for close."""
            closed.append(self)

    monkeypatch.setattr(model_registry, "ModelWorker", FakeWorker)

    registry = ModelRegistry()
    registry.load_model("model-a", {"pool_size": 2})
    registry.load_model("model-b", {"pool_size": 1})

    assert registry.unload_model("model-b") is True
    assert len(closed) == 1
    assert "model-b" not in registry.list_models()
    assert "model-a" in registry.list_models()


def test_unload_model_passes_drain_timeout(monkeypatch):
    """Test unload model passes drain timeout."""
    timeouts = []

    class FakeWorker:
        """Test helper FakeWorker."""

        def __init__(self, *args, **kwargs):
            """Helper for   init  ."""
            self.args = args
            self.kwargs = kwargs

        def close(self, timeout=None) -> None:
            """Helper for close."""
            timeouts.append(timeout)

    monkeypatch.setattr(model_registry, "ModelWorker", FakeWorker)

    registry = ModelRegistry()
    registry.load_model("model-a", {"pool_size": 1})
    registry.load_model("model-b", {"pool_size": 2})

    assert registry.unload_model("model-b", drain_timeout_sec=1.5) is True
    assert timeouts == [1.5, 1.5]


def test_load_model_rejects_non_positive_pool_size():
    """Test load model rejects non positive pool size."""
    registry = ModelRegistry()
    with pytest.raises(ValueError):
        registry.load_model("model-a", {"pool_size": 0})


def test_get_worker_prefers_lowest_pending():
    """Test get worker prefers lowest pending."""

    class FakeWorker:
        """Test helper FakeWorker."""

        def __init__(self, pending: int):
            """Helper for   init  ."""
            self._pending = pending

        def pending_tasks(self) -> int:
            """Helper for pending tasks."""
            return self._pending

    registry = ModelRegistry()
    workers = [FakeWorker(2), FakeWorker(0), FakeWorker(1)]
    registry._pools[DEFAULT_MODEL_ID] = cast(List[ModelWorkerProtocol], workers)
    registry._rr_counters[DEFAULT_MODEL_ID] = 0

    worker = registry.get_worker(DEFAULT_MODEL_ID)

    assert worker is workers[1]
    assert registry._rr_counters[DEFAULT_MODEL_ID] == 2


def test_load_model_closes_workers_on_failure(monkeypatch):
    """Test load model closes partial workers on failure."""
    closed = []

    class FakeWorker:
        """Test helper FakeWorker that fails on second init."""

        created = 0

        def __init__(self, *args, **kwargs):
            if FakeWorker.created == 1:
                raise RuntimeError("boom")
            FakeWorker.created += 1

        def close(self, *args, **kwargs) -> None:
            closed.append(self)

    monkeypatch.setattr(model_registry, "ModelWorker", FakeWorker)

    registry = ModelRegistry()
    with pytest.raises(RuntimeError):
        registry.load_model("bad-model", {"pool_size": 2})

    assert len(closed) == 1
