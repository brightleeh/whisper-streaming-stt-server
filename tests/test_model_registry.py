from stt_server.backend.application import model_registry
from stt_server.backend.application.model_registry import ModelRegistry


def test_unload_model_closes_workers(monkeypatch):
    closed = []

    class FakeWorker:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def close(self, *args, **kwargs) -> None:
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
    timeouts = []

    class FakeWorker:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def close(self, timeout=None) -> None:
            timeouts.append(timeout)

    monkeypatch.setattr(model_registry, "ModelWorker", FakeWorker)

    registry = ModelRegistry()
    registry.load_model("model-a", {"pool_size": 1})
    registry.load_model("model-b", {"pool_size": 2})

    assert registry.unload_model("model-b", drain_timeout_sec=1.5) is True
    assert timeouts == [1.5, 1.5]
