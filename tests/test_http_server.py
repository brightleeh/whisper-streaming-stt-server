import threading
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from stt_server.backend.transport.http_server import build_http_app


def test_http_load_model_thread_daemon_and_tracked():
    runtime = MagicMock()
    runtime.metrics = MagicMock()
    runtime.health_snapshot.return_value = {"model_pool_healthy": True}
    runtime.model_registry.is_loaded.return_value = False

    load_started = threading.Event()
    block_event = threading.Event()

    def slow_load(model_id, cfg):
        load_started.set()
        block_event.wait(timeout=0.2)

    runtime.model_registry.load_model.side_effect = slow_load

    app, load_threads, load_threads_lock = build_http_app(
        runtime, {"grpc_running": True}
    )
    client = TestClient(app)

    response = client.post("/admin/load_model", json={"model_id": "test-model"})
    assert response.status_code == 200
    assert load_started.wait(timeout=0.5)

    with load_threads_lock:
        threads = list(load_threads)
    assert len(threads) == 1
    assert threads[0].daemon is True
    block_event.set()
    threads[0].join(timeout=0.5)
    assert not threads[0].is_alive()
