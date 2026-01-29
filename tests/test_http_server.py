import threading
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from stt_server.backend.transport.http_server import build_http_app
from stt_server.errors import ErrorCode, http_payload_for, http_status_for


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


def _build_runtime():
    runtime = MagicMock()
    runtime.metrics = MagicMock()
    runtime.health_snapshot.return_value = {"model_pool_healthy": True}
    return runtime


def test_http_admin_api_disabled_returns_error_payload():
    runtime = _build_runtime()
    runtime.model_registry.load_model = None

    app, _, _ = build_http_app(runtime, {"grpc_running": True})
    client = TestClient(app)

    response = client.post("/admin/load_model", json={"model_id": "test-model"})

    assert response.status_code == http_status_for(ErrorCode.ADMIN_API_DISABLED)
    assert response.json() == http_payload_for(ErrorCode.ADMIN_API_DISABLED)


def test_http_admin_model_already_loaded_returns_error_payload():
    runtime = _build_runtime()
    runtime.model_registry.is_loaded.return_value = True

    app, _, _ = build_http_app(runtime, {"grpc_running": True})
    client = TestClient(app)

    response = client.post("/admin/load_model", json={"model_id": "test-model"})

    assert response.status_code == http_status_for(ErrorCode.MODEL_ALREADY_LOADED)
    assert response.json() == http_payload_for(
        ErrorCode.MODEL_ALREADY_LOADED, "Model 'test-model' is already loaded"
    )


def test_http_admin_unload_model_failed_returns_error_payload():
    runtime = _build_runtime()
    runtime.model_registry.unload_model.return_value = False

    app, _, _ = build_http_app(runtime, {"grpc_running": True})
    client = TestClient(app)

    response = client.post("/admin/unload_model", params={"model_id": "test-model"})

    assert response.status_code == http_status_for(ErrorCode.MODEL_UNLOAD_FAILED)
    assert response.json() == http_payload_for(ErrorCode.MODEL_UNLOAD_FAILED)
