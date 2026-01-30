import threading
from unittest.mock import MagicMock

import pytest

from stt_server.config import ServerConfig


def test_graceful_shutdown_on_signal(monkeypatch):
    handlers = {}

    def fake_signal(sig, handler):
        handlers[sig] = handler
        return None

    from stt_server import main as main_module

    monkeypatch.setattr(main_module.signal, "signal", fake_signal)

    fake_executor = MagicMock()
    monkeypatch.setattr(
        main_module.futures, "ThreadPoolExecutor", lambda max_workers: fake_executor
    )

    class FakeFuture:
        def wait(self):
            return None

    class FakeServer:
        def __init__(self):
            self.stop_calls = []
            self.wait_calls = 0

        def add_insecure_port(self, *args, **kwargs):
            return 0

        def start(self):
            return None

        def wait_for_termination(self, timeout=None):
            self.wait_calls += 1
            if self.wait_calls == 1:
                handler = handlers.get(main_module.signal.SIGTERM)
                if handler:
                    handler(main_module.signal.SIGTERM, None)
            return None

        def stop(self, grace):
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
    fake_runtime.shutdown.assert_called_once()
    fake_executor.shutdown.assert_called_once_with(wait=False)


def test_serve_skips_signal_handlers_outside_main_thread(monkeypatch):
    signal_calls = []

    def fake_signal(sig, handler):
        signal_calls.append(sig)
        return None

    from stt_server import main as main_module

    monkeypatch.setattr(main_module.signal, "signal", fake_signal)

    fake_executor = MagicMock()
    monkeypatch.setattr(
        main_module.futures, "ThreadPoolExecutor", lambda max_workers: fake_executor
    )

    class FakeFuture:
        def wait(self):
            return None

    class FakeServer:
        def add_insecure_port(self, *args, **kwargs):
            return 0

        def start(self):
            return None

        def wait_for_termination(self, timeout=None):
            raise RuntimeError("stop")

        def stop(self, grace):
            return FakeFuture()

    monkeypatch.setattr(
        main_module.grpc, "server", lambda executor, **_kwargs: FakeServer()
    )

    fake_http_handle = MagicMock()
    monkeypatch.setattr(
        main_module, "start_http_server", lambda **kwargs: fake_http_handle
    )

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
    captured = {}

    def fake_signal(*_args, **_kwargs):
        return None

    from stt_server import main as main_module

    monkeypatch.setattr(main_module.signal, "signal", fake_signal)

    fake_executor = MagicMock()
    monkeypatch.setattr(
        main_module.futures, "ThreadPoolExecutor", lambda max_workers: fake_executor
    )

    class FakeFuture:
        def wait(self):
            return None

    class FakeServer:
        def add_insecure_port(self, *args, **kwargs):
            return 0

        def start(self):
            return None

        def wait_for_termination(self, timeout=None):
            raise RuntimeError("stop")

        def stop(self, grace):
            return FakeFuture()

    def fake_server(executor, options=None):
        captured["options"] = options
        return FakeServer()

    monkeypatch.setattr(main_module.grpc, "server", fake_server)

    fake_http_handle = MagicMock()
    monkeypatch.setattr(
        main_module, "start_http_server", lambda **kwargs: fake_http_handle
    )

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
