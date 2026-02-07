from unittest.mock import MagicMock

from stt_server.backend.transport.ws_server import build_ws_app
from stt_server.config import ServerConfig
from stt_server.config.default import DEFAULT_WS_HOST, DEFAULT_WS_PORT


def test_ws_app_exposes_stream_route():
    """Test websocket app exposes stream route."""
    runtime = MagicMock()
    runtime.metrics = MagicMock()
    app = build_ws_app(runtime)

    paths = [getattr(route, "path", "") for route in app.router.routes]
    assert "/ws/stream" in paths


def test_server_config_defaults_include_ws_settings():
    """Test server config defaults include websocket settings."""
    config = ServerConfig()
    assert config.ws_host == DEFAULT_WS_HOST
    assert config.ws_port == DEFAULT_WS_PORT
