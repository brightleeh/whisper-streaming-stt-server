import threading
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from stt_server.backend.transport.http_server import build_http_app
from stt_server.errors import ErrorCode, http_payload_for, http_status_for


def _enable_admin(monkeypatch, token="secret-token"):
    """Helper for  enable admin."""
    monkeypatch.setenv("STT_ADMIN_ENABLED", "true")
    monkeypatch.setenv("STT_ADMIN_TOKEN", token)
    return {"Authorization": f"Bearer {token}"}


def test_http_load_model_thread_daemon_and_tracked(monkeypatch):
    """Test http load model thread daemon and tracked."""
    runtime = MagicMock()
    runtime.metrics = MagicMock()
    runtime.health_snapshot.return_value = {"model_pool_healthy": True}
    runtime.model_registry.is_loaded.return_value = False

    load_started = threading.Event()
    block_event = threading.Event()

    def slow_load(model_id, cfg):
        """Helper for slow load."""
        load_started.set()
        block_event.wait(timeout=0.2)

    runtime.model_registry.load_model.side_effect = slow_load

    app, load_threads, load_threads_lock = build_http_app(
        runtime, {"grpc_running": True}
    )
    client = TestClient(app)
    headers = _enable_admin(monkeypatch)

    response = client.post(
        "/admin/load_model", json={"model_id": "test-model"}, headers=headers
    )
    assert response.status_code == 200
    assert load_started.wait(timeout=0.5)

    with load_threads_lock:
        threads = list(load_threads)
    assert len(threads) == 1
    assert threads[0].daemon is True
    block_event.set()
    threads[0].join(timeout=0.5)
    assert not threads[0].is_alive()

    runtime.model_registry.load_model.side_effect = None
    runtime.model_registry.load_model.return_value = None
    response = client.post(
        "/admin/load_model", json={"model_id": "another-model"}, headers=headers
    )
    assert response.status_code == 200
    with load_threads_lock:
        threads = list(load_threads)
    assert len(threads) == 1


def _build_runtime():
    """Helper for  build runtime."""
    runtime = MagicMock()
    runtime.metrics = MagicMock()
    runtime.health_snapshot.return_value = {"model_pool_healthy": True}
    runtime.metrics.render.return_value = {
        "active_sessions": 2,
        "decode_latency_max": 1.5,
        "active_sessions_by_api": {"key-a": 1},
    }
    return runtime


def _attach_model_profiles(runtime, profiles, default_profile="default"):
    """Helper for attaching model load profiles to runtime config."""
    runtime.config = MagicMock()
    runtime.config.model = MagicMock()
    runtime.config.model.model_load_profiles = profiles
    runtime.config.model.default_model_load_profile = default_profile
    return runtime


def test_http_admin_api_disabled_returns_error_payload(monkeypatch):
    """Test http admin api disabled returns error payload."""
    runtime = _build_runtime()
    runtime.model_registry.load_model = None
    monkeypatch.delenv("STT_ADMIN_ENABLED", raising=False)
    monkeypatch.delenv("STT_ADMIN_TOKEN", raising=False)

    app, _, _ = build_http_app(runtime, {"grpc_running": True})
    client = TestClient(app)

    response = client.post("/admin/load_model", json={"model_id": "test-model"})

    assert response.status_code == http_status_for(ErrorCode.ADMIN_API_DISABLED)
    assert response.json() == http_payload_for(ErrorCode.ADMIN_API_DISABLED)


def test_http_admin_model_already_loaded_returns_error_payload(monkeypatch):
    """Test http admin model already loaded returns error payload."""
    runtime = _build_runtime()
    runtime.model_registry.is_loaded.return_value = True

    app, _, _ = build_http_app(runtime, {"grpc_running": True})
    client = TestClient(app)
    headers = _enable_admin(monkeypatch)

    response = client.post(
        "/admin/load_model", json={"model_id": "test-model"}, headers=headers
    )

    assert response.status_code == http_status_for(ErrorCode.MODEL_ALREADY_LOADED)
    assert response.json() == http_payload_for(
        ErrorCode.MODEL_ALREADY_LOADED, "Model 'test-model' is already loaded"
    )


def test_http_admin_unload_model_failed_returns_error_payload(monkeypatch):
    """Test http admin unload model failed returns error payload."""
    runtime = _build_runtime()
    runtime.model_registry.unload_model = None

    app, _, _ = build_http_app(runtime, {"grpc_running": True})
    client = TestClient(app)
    headers = _enable_admin(monkeypatch)

    response = client.post(
        "/admin/unload_model",
        params={"model_id": "test-model"},
        headers=headers,
    )

    assert response.status_code == http_status_for(ErrorCode.ADMIN_API_DISABLED)
    assert response.json() == http_payload_for(ErrorCode.ADMIN_API_DISABLED)


def test_http_admin_unload_model_passes_drain_timeout(monkeypatch):
    """Test http admin unload model passes drain timeout."""
    runtime = _build_runtime()
    runtime.model_registry.unload_model.return_value = True

    app, _, _ = build_http_app(runtime, {"grpc_running": True})
    client = TestClient(app)
    headers = _enable_admin(monkeypatch)

    response = client.post(
        "/admin/unload_model",
        params={"model_id": "test-model", "drain_timeout_sec": 0.25},
        headers=headers,
    )

    assert response.status_code == 200
    runtime.model_registry.unload_model.assert_called_with(
        "test-model", drain_timeout_sec=0.25
    )


def test_http_admin_list_models_with_token_succeeds(monkeypatch):
    """Test http admin list models with token succeeds."""
    runtime = _build_runtime()
    runtime.model_registry.list_models.return_value = ["model-a", "model-b"]

    app, _, _ = build_http_app(runtime, {"grpc_running": True})
    client = TestClient(app)
    headers = _enable_admin(monkeypatch)

    response = client.get("/admin/list_models", headers=headers)

    assert response.status_code == 200
    assert response.json() == {"models": ["model-a", "model-b"]}


def test_http_admin_unauthorized_returns_error_payload(monkeypatch):
    """Test http admin unauthorized returns error payload."""
    runtime = _build_runtime()
    runtime.model_registry.is_loaded.return_value = False
    _enable_admin(monkeypatch, token="expected")

    app, _, _ = build_http_app(runtime, {"grpc_running": True})
    client = TestClient(app)

    response = client.post("/admin/load_model", json={"model_id": "test-model"})

    assert response.status_code == http_status_for(ErrorCode.ADMIN_UNAUTHORIZED)
    assert response.json() == http_payload_for(ErrorCode.ADMIN_UNAUTHORIZED)


def test_http_admin_model_path_forbidden_returns_error_payload(monkeypatch):
    """Test http admin model path forbidden returns error payload."""
    runtime = _build_runtime()
    runtime.model_registry.is_loaded.return_value = False
    headers = _enable_admin(monkeypatch)

    app, _, _ = build_http_app(runtime, {"grpc_running": True})
    client = TestClient(app)

    response = client.post(
        "/admin/load_model",
        json={"model_id": "test-model", "model_path": "/tmp/evil"},
        headers=headers,
    )

    assert response.status_code == http_status_for(ErrorCode.ADMIN_MODEL_PATH_FORBIDDEN)
    assert response.json() == http_payload_for(ErrorCode.ADMIN_MODEL_PATH_FORBIDDEN)


def test_http_admin_load_model_uses_profile_id(monkeypatch):
    """Test admin load model uses profile_id config."""
    runtime = _build_runtime()
    runtime.model_registry.is_loaded.return_value = False
    profile_cfg = {
        "model_size": "tiny",
        "device": "cpu",
        "compute_type": "int8",
        "pool_size": 2,
    }
    _attach_model_profiles(runtime, {"fast": profile_cfg}, default_profile="fast")
    runtime.model_registry.load_model.return_value = None

    app, load_threads, load_threads_lock = build_http_app(
        runtime, {"grpc_running": True}
    )
    client = TestClient(app)
    headers = _enable_admin(monkeypatch)

    response = client.post(
        "/admin/load_model",
        json={"model_id": "fast-model", "profile_id": "fast"},
        headers=headers,
    )

    assert response.status_code == 200
    with load_threads_lock:
        threads = list(load_threads)
    for thread in threads:
        thread.join(timeout=0.5)

    runtime.model_registry.load_model.assert_called_once_with("fast-model", profile_cfg)


def test_http_admin_load_model_defaults_to_profile(monkeypatch):
    """Test admin load model defaults to model profile."""
    runtime = _build_runtime()
    runtime.model_registry.is_loaded.return_value = False
    profile_cfg = {
        "model_size": "small",
        "device": "cpu",
        "compute_type": "int8",
        "pool_size": 1,
    }
    _attach_model_profiles(runtime, {"default": profile_cfg}, default_profile="default")
    runtime.model_registry.load_model.return_value = None

    app, load_threads, load_threads_lock = build_http_app(
        runtime, {"grpc_running": True}
    )
    client = TestClient(app)
    headers = _enable_admin(monkeypatch)

    response = client.post(
        "/admin/load_model", json={"model_id": "default-model"}, headers=headers
    )

    assert response.status_code == 200
    with load_threads_lock:
        threads = list(load_threads)
    for thread in threads:
        thread.join(timeout=0.5)

    runtime.model_registry.load_model.assert_called_once_with(
        "default-model", profile_cfg
    )


def test_http_admin_load_model_unknown_profile_returns_error_payload(monkeypatch):
    """Test admin load model fails for unknown profile."""
    runtime = _build_runtime()
    runtime.model_registry.is_loaded.return_value = False
    _attach_model_profiles(
        runtime, {"default": {"model_size": "small"}}, default_profile="default"
    )

    app, _, _ = build_http_app(runtime, {"grpc_running": True})
    client = TestClient(app)
    headers = _enable_admin(monkeypatch)

    response = client.post(
        "/admin/load_model",
        json={"model_id": "bad-model", "profile_id": "missing"},
        headers=headers,
    )

    assert response.status_code == http_status_for(
        ErrorCode.ADMIN_MODEL_PROFILE_UNKNOWN
    )
    assert response.json() == http_payload_for(
        ErrorCode.ADMIN_MODEL_PROFILE_UNKNOWN,
        "Unknown model profile 'missing'",
    )


def test_http_admin_load_model_legacy_fields_override_profiles(monkeypatch):
    """Test legacy load model fields override profile defaults."""
    runtime = _build_runtime()
    runtime.model_registry.is_loaded.return_value = False
    _attach_model_profiles(
        runtime, {"default": {"model_size": "small"}}, default_profile="default"
    )
    runtime.model_registry.load_model.return_value = None
    monkeypatch.setenv("STT_ADMIN_ALLOW_MODEL_PATH", "true")

    app, load_threads, load_threads_lock = build_http_app(
        runtime, {"grpc_running": True}
    )
    client = TestClient(app)
    headers = _enable_admin(monkeypatch)

    response = client.post(
        "/admin/load_model",
        json={"model_id": "legacy-model", "model_path": "/tmp/legacy"},
        headers=headers,
    )

    assert response.status_code == 200
    with load_threads_lock:
        threads = list(load_threads)
    for thread in threads:
        thread.join(timeout=0.5)

    args, _ = runtime.model_registry.load_model.call_args
    assert args[0] == "legacy-model"
    assert args[1]["model_path"] == "/tmp/legacy"


def test_http_metrics_requires_observability_token(monkeypatch):
    """Test observability endpoints require token when configured."""
    runtime = _build_runtime()
    monkeypatch.setenv("STT_OBSERVABILITY_TOKEN", "metrics-token")

    app, _, _ = build_http_app(runtime, {"grpc_running": True})
    client = TestClient(app)

    response = client.get("/metrics")
    assert response.status_code == http_status_for(ErrorCode.OBS_UNAUTHORIZED)
    assert response.json() == http_payload_for(ErrorCode.OBS_UNAUTHORIZED)

    response = client.get("/metrics", headers={"Authorization": "Bearer metrics-token"})
    assert response.status_code == 200


def test_http_system_requires_observability_token(monkeypatch):
    """Test system endpoint requires observability token."""
    runtime = _build_runtime()
    monkeypatch.setenv("STT_OBSERVABILITY_TOKEN", "metrics-token")

    app, _, _ = build_http_app(runtime, {"grpc_running": True})
    client = TestClient(app)

    response = client.get("/system")
    assert response.status_code == http_status_for(ErrorCode.OBS_UNAUTHORIZED)
    assert response.json() == http_payload_for(ErrorCode.OBS_UNAUTHORIZED)


def test_http_health_requires_observability_token(monkeypatch):
    """Test health endpoint requires observability token by default."""
    runtime = _build_runtime()
    monkeypatch.setenv("STT_OBSERVABILITY_TOKEN", "metrics-token")

    app, _, _ = build_http_app(runtime, {"grpc_running": True})
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == http_status_for(ErrorCode.OBS_UNAUTHORIZED)
    assert response.json() == http_payload_for(ErrorCode.OBS_UNAUTHORIZED)


def test_http_health_minimal_public_response(monkeypatch):
    """Test public minimal health response omits details."""
    runtime = _build_runtime()
    monkeypatch.setenv("STT_OBSERVABILITY_TOKEN", "metrics-token")
    monkeypatch.setenv("STT_PUBLIC_HEALTH", "minimal")

    app, _, _ = build_http_app(runtime, {"grpc_running": True})
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_http_health_minimal_with_token_includes_details(monkeypatch):
    """Test minimal health returns details when token is provided."""
    runtime = _build_runtime()
    monkeypatch.setenv("STT_OBSERVABILITY_TOKEN", "metrics-token")
    monkeypatch.setenv("STT_PUBLIC_HEALTH", "minimal")

    app, _, _ = build_http_app(runtime, {"grpc_running": True})
    client = TestClient(app)

    response = client.get("/health", headers={"Authorization": "Bearer metrics-token"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["grpc_running"] is True
    assert payload["model_pool_healthy"] is True


def test_http_admin_load_model_status_tracks_success(monkeypatch):
    """Test admin load model status transitions to success."""
    runtime = _build_runtime()
    runtime.model_registry.is_loaded.return_value = False
    load_started = threading.Event()
    block_event = threading.Event()

    def slow_load(_model_id, _cfg):
        load_started.set()
        block_event.wait(timeout=0.2)

    runtime.model_registry.load_model.side_effect = slow_load

    app, load_threads, load_threads_lock = build_http_app(
        runtime, {"grpc_running": True}
    )
    client = TestClient(app)
    headers = _enable_admin(monkeypatch)

    response = client.post(
        "/admin/load_model", json={"model_id": "test-model"}, headers=headers
    )
    assert response.status_code == 200
    assert load_started.wait(timeout=0.5)

    status = client.get(
        "/admin/load_model_status",
        params={"model_id": "test-model"},
        headers=headers,
    )
    assert status.status_code == 200
    assert status.json()["status"] == "running"

    block_event.set()
    with load_threads_lock:
        threads = list(load_threads)
    for thread in threads:
        thread.join(timeout=0.5)

    status = client.get(
        "/admin/load_model_status",
        params={"model_id": "test-model"},
        headers=headers,
    )
    assert status.status_code == 200
    assert status.json()["status"] == "success"


def test_http_admin_load_model_status_tracks_failure(monkeypatch):
    """Test admin load model status transitions to failed."""
    runtime = _build_runtime()
    runtime.model_registry.is_loaded.return_value = False

    def fail_load(_model_id, _cfg):
        raise RuntimeError("boom")

    runtime.model_registry.load_model.side_effect = fail_load

    app, load_threads, load_threads_lock = build_http_app(
        runtime, {"grpc_running": True}
    )
    client = TestClient(app)
    headers = _enable_admin(monkeypatch)

    response = client.post(
        "/admin/load_model", json={"model_id": "bad-model"}, headers=headers
    )
    assert response.status_code == 200

    with load_threads_lock:
        threads = list(load_threads)
    for thread in threads:
        thread.join(timeout=0.5)

    status = client.get(
        "/admin/load_model_status",
        params={"model_id": "bad-model"},
        headers=headers,
    )
    assert status.status_code == 200
    payload = status.json()
    assert payload["status"] == "failed"
    assert "boom" in payload.get("error", "")


def test_http_rate_limit_blocks_excess_requests(monkeypatch):
    """Test HTTP rate limiter blocks excess requests."""
    runtime = _build_runtime()
    monkeypatch.setenv("STT_HTTP_RATE_LIMIT_RPS", "1")
    monkeypatch.setenv("STT_HTTP_RATE_LIMIT_BURST", "1")

    app, _, _ = build_http_app(runtime, {"grpc_running": True})
    client = TestClient(app)

    response = client.get("/metrics")
    assert response.status_code == 200
    response = client.get("/metrics")
    assert response.status_code == http_status_for(ErrorCode.HTTP_RATE_LIMITED)
    assert response.json() == http_payload_for(ErrorCode.HTTP_RATE_LIMITED)


def test_http_allowlist_blocks_unknown_ip(monkeypatch):
    """Test HTTP allowlist denies non-matching client IPs."""
    runtime = _build_runtime()
    monkeypatch.setenv("STT_HTTP_ALLOWLIST", "127.0.0.1/32")

    app, _, _ = build_http_app(runtime, {"grpc_running": True})
    client = TestClient(app)

    response = client.get("/metrics", headers={"X-Forwarded-For": "10.0.0.2"})
    assert response.status_code == http_status_for(ErrorCode.HTTP_IP_FORBIDDEN)
    assert response.json() == http_payload_for(ErrorCode.HTTP_IP_FORBIDDEN)

    response = client.get("/metrics", headers={"X-Forwarded-For": "127.0.0.1"})
    assert response.status_code == 200


def test_metrics_endpoints_format(monkeypatch):
    """Test metrics endpoints format."""
    runtime = _build_runtime()
    app, _, _ = build_http_app(runtime, {"grpc_running": True})
    client = TestClient(app)

    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers.get("content-type", "")
    body = response.text
    assert "stt_active_sessions 2.0" in body
    assert "stt_decode_latency_max 1.5" in body
    assert "stt_active_sessions_by_api_key_a 1.0" in body

    response = client.get("/metrics.json")
    assert response.status_code == 200
    assert response.json()["active_sessions"] == 2
