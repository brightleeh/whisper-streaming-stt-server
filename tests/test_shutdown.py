import threading
from unittest.mock import MagicMock

import pytest

from stt_server.config import ServerConfig


def test_ws_auth_guard_rejects_public_ws_without_auth(monkeypatch):
    """Public WS bind must require CreateSession auth unless explicitly overridden."""
    from stt_server import main as main_module

    monkeypatch.delenv("STT_ALLOW_INSECURE_WS", raising=False)

    config = ServerConfig()
    config.ws_port = 8001
    config.ws_host = "0.0.0.0"
    config.create_session_auth_profile = "none"
    config.require_api_key = False

    with pytest.raises(ValueError, match="WebSocket is bound to a non-loopback host"):
        main_module._enforce_ws_auth_guard(config)


def test_ws_auth_guard_allows_public_ws_with_signed_token(monkeypatch):
    """Public WS bind is allowed when signed-token auth is enabled."""
    from stt_server import main as main_module

    monkeypatch.delenv("STT_ALLOW_INSECURE_WS", raising=False)

    config = ServerConfig()
    config.ws_port = 8001
    config.ws_host = "0.0.0.0"
    config.create_session_auth_profile = "signed_token"
    config.require_api_key = False

    main_module._enforce_ws_auth_guard(config)


def test_graceful_shutdown_on_signal(monkeypatch):
    """Test graceful shutdown on signal."""
    handlers = {}

    def fake_signal(sig, handler):
        """Helper for fake signal."""
        handlers[sig] = handler
        return None

    from stt_server import main as main_module

    monkeypatch.setattr(main_module.signal, "signal", fake_signal)

    fake_executor = MagicMock()
    monkeypatch.setattr(
        main_module.futures, "ThreadPoolExecutor", lambda max_workers: fake_executor
    )

    class FakeFuture:
        """Test helper FakeFuture."""

        def wait(self):
            """Helper for wait."""
            return None

    class FakeServer:
        """Test helper FakeServer."""

        def __init__(self):
            """Helper for   init  ."""
            self.stop_calls = []
            self.wait_calls = 0

        def add_insecure_port(self, *args, **kwargs):
            """Helper for add insecure port."""
            return 0

        def start(self):
            """Helper for start."""
            return None

        def wait_for_termination(self, timeout=None):
            """Helper for wait for termination."""
            self.wait_calls += 1
            if self.wait_calls == 1:
                handler = handlers.get(main_module.signal.SIGTERM)
                if handler:
                    handler(main_module.signal.SIGTERM, None)
            return None

        def stop(self, grace):
            """Helper for stop."""
            self.stop_calls.append(grace)
            return FakeFuture()

    fake_server = FakeServer()
    monkeypatch.setattr(
        main_module.grpc, "server", lambda executor, **_kwargs: fake_server
    )

    fake_http_handle = MagicMock()
    monkeypatch.setattr(
        main_module, "start_http_server", lambda **kwargs: fake_http_handle
    )
    fake_ws_handle = MagicMock()
    monkeypatch.setattr(main_module, "start_ws_server", lambda **kwargs: fake_ws_handle)

    fake_runtime = MagicMock()
    fake_servicer = MagicMock()
    fake_servicer.runtime = fake_runtime
    monkeypatch.setattr(main_module, "STTGrpcServicer", lambda config: fake_servicer)
    monkeypatch.setattr(
        main_module.stt_pb2_grpc,
        "add_STTBackendServicer_to_server",
        lambda servicer, server: None,
    )

    config = ServerConfig()
    config.decode_timeout_sec = 2.5

    main_module.serve(config)

    assert fake_server.stop_calls == [2.5]
    fake_http_handle.stop.assert_called_once_with(timeout=3.5)
    fake_ws_handle.stop.assert_called_once_with(timeout=3.5)
    fake_runtime.stop_accepting_sessions.assert_called_once()
    fake_runtime.shutdown.assert_called_once()
    fake_executor.shutdown.assert_called_once_with(wait=False)


def test_serve_skips_signal_handlers_outside_main_thread(monkeypatch):
    """Test serve skips signal handlers outside main thread."""
    signal_calls = []

    def fake_signal(sig, handler):
        """Helper for fake signal."""
        signal_calls.append(sig)
        return None

    from stt_server import main as main_module

    monkeypatch.setattr(main_module.signal, "signal", fake_signal)

    fake_executor = MagicMock()
    monkeypatch.setattr(
        main_module.futures, "ThreadPoolExecutor", lambda max_workers: fake_executor
    )

    class FakeFuture:
        """Test helper FakeFuture."""

        def wait(self):
            """Helper for wait."""
            return None

    class FakeServer:
        """Test helper FakeServer."""

        def add_insecure_port(self, *args, **kwargs):
            """Helper for add insecure port."""
            return 0

        def start(self):
            """Helper for start."""
            return None

        def wait_for_termination(self, timeout=None):
            """Helper for wait for termination."""
            raise RuntimeError("stop")

        def stop(self, grace):
            """Helper for stop."""
            return FakeFuture()

    monkeypatch.setattr(
        main_module.grpc, "server", lambda executor, **_kwargs: FakeServer()
    )

    fake_http_handle = MagicMock()
    monkeypatch.setattr(
        main_module, "start_http_server", lambda **kwargs: fake_http_handle
    )
    fake_ws_handle = MagicMock()
    monkeypatch.setattr(main_module, "start_ws_server", lambda **kwargs: fake_ws_handle)

    fake_runtime = MagicMock()
    fake_servicer = MagicMock()
    fake_servicer.runtime = fake_runtime
    monkeypatch.setattr(main_module, "STTGrpcServicer", lambda config: fake_servicer)
    monkeypatch.setattr(
        main_module.stt_pb2_grpc,
        "add_STTBackendServicer_to_server",
        lambda servicer, server: None,
    )

    config = ServerConfig()

    error_holder = {}

    def run():
        """Helper for run."""
        try:
            main_module.serve(config)
        except RuntimeError as exc:
            error_holder["exc"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert "exc" in error_holder
    assert signal_calls == []


def test_serve_passes_grpc_message_limits(monkeypatch):
    """Test serve passes grpc message limits."""
    captured = {}

    def fake_signal(*_args, **_kwargs):
        """Helper for fake signal."""
        return None

    from stt_server import main as main_module

    monkeypatch.setattr(main_module.signal, "signal", fake_signal)

    fake_executor = MagicMock()
    monkeypatch.setattr(
        main_module.futures, "ThreadPoolExecutor", lambda max_workers: fake_executor
    )

    class FakeFuture:
        """Test helper FakeFuture."""

        def wait(self):
            """Helper for wait."""
            return None

    class FakeServer:
        """Test helper FakeServer."""

        def add_insecure_port(self, *args, **kwargs):
            """Helper for add insecure port."""
            return 0

        def start(self):
            """Helper for start."""
            return None

        def wait_for_termination(self, timeout=None):
            """Helper for wait for termination."""
            raise RuntimeError("stop")

        def stop(self, grace):
            """Helper for stop."""
            return FakeFuture()

    def fake_server(executor, options=None):
        """Helper for fake server."""
        captured["options"] = options
        return FakeServer()

    monkeypatch.setattr(main_module.grpc, "server", fake_server)

    fake_http_handle = MagicMock()
    monkeypatch.setattr(
        main_module, "start_http_server", lambda **kwargs: fake_http_handle
    )
    fake_ws_handle = MagicMock()
    monkeypatch.setattr(main_module, "start_ws_server", lambda **kwargs: fake_ws_handle)

    fake_runtime = MagicMock()
    fake_servicer = MagicMock()
    fake_servicer.runtime = fake_runtime
    monkeypatch.setattr(main_module, "STTGrpcServicer", lambda config: fake_servicer)
    monkeypatch.setattr(
        main_module.stt_pb2_grpc,
        "add_STTBackendServicer_to_server",
        lambda servicer, server: None,
    )

    config = ServerConfig()
    config.grpc_max_receive_message_bytes = 123
    config.grpc_max_send_message_bytes = 456

    with pytest.raises(RuntimeError, match="stop"):
        main_module.serve(config)

    assert captured["options"] == [
        ("grpc.max_receive_message_length", 123),
        ("grpc.max_send_message_length", 456),
    ]


def test_serve_requires_tls_when_configured(monkeypatch):
    """Test serve refuses to start when TLS is required but missing."""

    def fake_signal(*_args, **_kwargs):
        """Helper for fake signal."""
        return None

    from stt_server import main as main_module

    monkeypatch.setattr(main_module.signal, "signal", fake_signal)

    fake_executor = MagicMock()
    monkeypatch.setattr(
        main_module.futures, "ThreadPoolExecutor", lambda max_workers: fake_executor
    )

    class FakeServer:
        """Test helper FakeServer."""

        def add_insecure_port(self, *args, **kwargs):
            """Helper for add insecure port."""
            return 0

        def start(self):
            """Helper for start."""
            return None

        def wait_for_termination(self, timeout=None):
            """Helper for wait for termination."""
            raise RuntimeError("stop")

        def stop(self, grace):
            """Helper for stop."""
            return MagicMock()

    monkeypatch.setattr(
        main_module.grpc, "server", lambda executor, **_kwargs: FakeServer()
    )

    fake_http_handle = MagicMock()
    monkeypatch.setattr(
        main_module, "start_http_server", lambda **kwargs: fake_http_handle
    )
    fake_ws_handle = MagicMock()
    monkeypatch.setattr(main_module, "start_ws_server", lambda **kwargs: fake_ws_handle)

    fake_runtime = MagicMock()
    fake_servicer = MagicMock()
    fake_servicer.runtime = fake_runtime
    monkeypatch.setattr(main_module, "STTGrpcServicer", lambda config: fake_servicer)
    monkeypatch.setattr(
        main_module.stt_pb2_grpc,
        "add_STTBackendServicer_to_server",
        lambda servicer, server: None,
    )

    config = ServerConfig()
    config.tls_required = True
    config.tls_cert_file = None
    config.tls_key_file = None

    with pytest.raises(ValueError, match="TLS is required"):
        main_module.serve(config)


def test_serve_starts_ws_server_when_enabled(monkeypatch):
    """Test serve starts websocket server when enabled."""

    def fake_signal(*_args, **_kwargs):
        """Helper for fake signal."""
        return None

    from stt_server import main as main_module

    monkeypatch.setattr(main_module.signal, "signal", fake_signal)

    fake_executor = MagicMock()
    monkeypatch.setattr(
        main_module.futures, "ThreadPoolExecutor", lambda max_workers: fake_executor
    )

    class FakeFuture:
        """Test helper FakeFuture."""

        def wait(self):
            """Helper for wait."""
            return None

    class FakeServer:
        """Test helper FakeServer."""

        def add_insecure_port(self, *args, **kwargs):
            """Helper for add insecure port."""
            return 0

        def start(self):
            """Helper for start."""
            return None

        def wait_for_termination(self, timeout=None):
            """Helper for wait for termination."""
            raise RuntimeError("stop")

        def stop(self, grace):
            """Helper for stop."""
            return FakeFuture()

    monkeypatch.setattr(
        main_module.grpc, "server", lambda executor, **_kwargs: FakeServer()
    )

    fake_http_handle = MagicMock()
    monkeypatch.setattr(
        main_module, "start_http_server", lambda **kwargs: fake_http_handle
    )

    fake_ws_handle = MagicMock()
    ws_calls = {}

    def fake_start_ws_server(**kwargs):
        """Helper for fake websocket server."""
        ws_calls["kwargs"] = kwargs
        return fake_ws_handle

    monkeypatch.setattr(main_module, "start_ws_server", fake_start_ws_server)

    fake_runtime = MagicMock()
    fake_servicer = MagicMock()
    fake_servicer.runtime = fake_runtime
    monkeypatch.setattr(main_module, "STTGrpcServicer", lambda config: fake_servicer)
    monkeypatch.setattr(
        main_module.stt_pb2_grpc,
        "add_STTBackendServicer_to_server",
        lambda servicer, server: None,
    )

    config = ServerConfig()
    config.ws_port = 4321
    config.ws_host = "127.0.0.1"

    with pytest.raises(RuntimeError, match="stop"):
        main_module.serve(config)

    assert ws_calls["kwargs"]["host"] == "127.0.0.1"
    assert ws_calls["kwargs"]["port"] == 4321
    fake_ws_handle.stop.assert_called_once()
    fake_http_handle.stop.assert_called_once()


def test_serve_skips_ws_server_when_disabled(monkeypatch):
    """Test serve skips websocket server when disabled."""

    def fake_signal(*_args, **_kwargs):
        """Helper for fake signal."""
        return None

    from stt_server import main as main_module

    monkeypatch.setattr(main_module.signal, "signal", fake_signal)

    fake_executor = MagicMock()
    monkeypatch.setattr(
        main_module.futures, "ThreadPoolExecutor", lambda max_workers: fake_executor
    )

    class FakeFuture:
        """Test helper FakeFuture."""

        def wait(self):
            """Helper for wait."""
            return None

    class FakeServer:
        """Test helper FakeServer."""

        def add_insecure_port(self, *args, **kwargs):
            """Helper for add insecure port."""
            return 0

        def start(self):
            """Helper for start."""
            return None

        def wait_for_termination(self, timeout=None):
            """Helper for wait for termination."""
            raise RuntimeError("stop")

        def stop(self, grace):
            """Helper for stop."""
            return FakeFuture()

    monkeypatch.setattr(
        main_module.grpc, "server", lambda executor, **_kwargs: FakeServer()
    )

    fake_http_handle = MagicMock()
    monkeypatch.setattr(
        main_module, "start_http_server", lambda **kwargs: fake_http_handle
    )

    fake_ws_handle = MagicMock()
    monkeypatch.setattr(main_module, "start_ws_server", fake_ws_handle)

    fake_runtime = MagicMock()
    fake_servicer = MagicMock()
    fake_servicer.runtime = fake_runtime
    monkeypatch.setattr(main_module, "STTGrpcServicer", lambda config: fake_servicer)
    monkeypatch.setattr(
        main_module.stt_pb2_grpc,
        "add_STTBackendServicer_to_server",
        lambda servicer, server: None,
    )

    config = ServerConfig()
    config.ws_port = 0

    with pytest.raises(RuntimeError, match="stop"):
        main_module.serve(config)

    fake_ws_handle.assert_not_called()
