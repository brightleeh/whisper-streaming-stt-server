import logging
import os
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from uvicorn.config import LOGGING_CONFIG

from stt_server.backend.runtime import ApplicationRuntime
from stt_server.backend.utils.system_metrics import collect_system_metrics
from stt_server.config.default.model import (
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_MODEL_NAME,
)
from stt_server.errors import ErrorCode, STTError, http_payload_for, http_status_for

_ACCESS_LOG_IGNORED_PATHS = frozenset({"/metrics", "/system", "/health"})
_ADMIN_ENABLE_ENV = "STT_ADMIN_ENABLED"
_ADMIN_TOKEN_ENV = "STT_ADMIN_TOKEN"
_ADMIN_ALLOW_MODEL_PATH_ENV = "STT_ADMIN_ALLOW_MODEL_PATH"
_ADMIN_MODEL_PATH_ALLOWLIST_ENV = "STT_ADMIN_MODEL_PATH_ALLOWLIST"


class _AccessLogPathFilter(logging.Filter):
    def __init__(self, ignored_paths: Tuple[str, ...]) -> None:
        super().__init__()
        self._ignored_paths = set(ignored_paths)

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple) and len(record.args) >= 3:
            path = record.args[2]
            if path in self._ignored_paths:
                return False
        return True


def _build_uvicorn_log_config() -> Dict[str, Any]:
    log_config = deepcopy(LOGGING_CONFIG)
    log_config.setdefault("filters", {})
    log_config["filters"]["ignore_internal_endpoints"] = {
        "()": _AccessLogPathFilter,
        "ignored_paths": tuple(sorted(_ACCESS_LOG_IGNORED_PATHS)),
    }
    access_handler = log_config["handlers"].get("access", {})
    access_filters = access_handler.get("filters", [])
    access_handler["filters"] = [*access_filters, "ignore_internal_endpoints"]
    log_config["handlers"]["access"] = access_handler
    return log_config


def _env_enabled(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _admin_token() -> str:
    return os.getenv(_ADMIN_TOKEN_ENV, "").strip()


def _require_admin(request: Request) -> None:
    if not _env_enabled(_ADMIN_ENABLE_ENV) or not _admin_token():
        raise STTError(ErrorCode.ADMIN_API_DISABLED)
    auth = request.headers.get("authorization", "").strip()
    if not auth.lower().startswith("bearer "):
        raise STTError(ErrorCode.ADMIN_UNAUTHORIZED)
    token = auth[7:].strip()
    if token != _admin_token():
        raise STTError(ErrorCode.ADMIN_UNAUTHORIZED)


def _model_path_allowed(model_path: Optional[str]) -> bool:
    if not model_path:
        return True
    if not _env_enabled(_ADMIN_ALLOW_MODEL_PATH_ENV):
        return False
    allowlist_raw = os.getenv(_ADMIN_MODEL_PATH_ALLOWLIST_ENV, "")
    allowlist = [item.strip() for item in allowlist_raw.split(",") if item.strip()]
    if not allowlist:
        return True
    return any(model_path.startswith(prefix) for prefix in allowlist)


class LoadModelRequest(BaseModel):
    model_id: str
    model_path: Optional[str] = None  # local path or HuggingFace ID
    model_size: Optional[str] = (
        DEFAULT_MODEL_NAME  # size name to use if model_path is missing (e.g. "small")
    )
    device: str = DEFAULT_DEVICE
    compute_type: str = DEFAULT_COMPUTE_TYPE
    language: Optional[str] = None


@dataclass
class HttpServerHandle:
    server: uvicorn.Server
    thread: threading.Thread
    load_threads: List[threading.Thread]
    load_threads_lock: threading.Lock

    def stop(self, timeout: Optional[float] = None) -> None:
        if self.thread.is_alive():
            self.server.should_exit = True
            self.thread.join(timeout=timeout)
        with self.load_threads_lock:
            threads = list(self.load_threads)
        if threads:
            deadline = time.monotonic() + timeout if timeout is not None else None
            for thread in threads:
                remaining = None
                if deadline is not None:
                    remaining = max(0.0, deadline - time.monotonic())
                thread.join(timeout=remaining)


def build_http_app(
    runtime: ApplicationRuntime, server_state: Dict[str, bool]
) -> Tuple[FastAPI, List[threading.Thread], threading.Lock]:
    """Create the FastAPI app and load-model thread tracking state."""
    app = FastAPI()
    metrics = runtime.metrics
    model_registry = runtime.model_registry
    health_snapshot = runtime.health_snapshot
    load_threads: List[threading.Thread] = []
    load_threads_lock = threading.Lock()

    @app.exception_handler(STTError)
    async def stt_error_handler(_request: Request, exc: STTError) -> JSONResponse:
        return JSONResponse(
            http_payload_for(exc.code, exc.detail),
            status_code=http_status_for(exc.code),
        )

    @app.get("/metrics")
    def metrics_endpoint() -> Response:
        return JSONResponse(metrics.render(), status_code=200)

    @app.get("/health")
    def health_endpoint() -> JSONResponse:
        snapshot = health_snapshot()
        snapshot["grpc_running"] = server_state.get("grpc_running", False)
        healthy = snapshot["grpc_running"] and snapshot["model_pool_healthy"]
        status = 200 if healthy else 500
        payload = {"status": "ok" if healthy else "error", **snapshot}
        return JSONResponse(payload, status_code=status)

    @app.get("/system")
    def system_endpoint() -> JSONResponse:
        return JSONResponse(collect_system_metrics(), status_code=200)

    @app.post("/admin/load_model")
    def load_model_endpoint(req: LoadModelRequest, request: Request) -> JSONResponse:
        _require_admin(request)
        if not model_registry.load_model:
            raise STTError(ErrorCode.ADMIN_API_DISABLED)

        if model_registry.is_loaded(req.model_id):
            raise STTError(
                ErrorCode.MODEL_ALREADY_LOADED,
                f"Model '{req.model_id}' is already loaded",
            )
        if not _model_path_allowed(req.model_path):
            raise STTError(ErrorCode.ADMIN_MODEL_PATH_FORBIDDEN)

        # Load in a separate thread to prevent blocking the main loop
        thread = threading.Thread(
            target=model_registry.load_model,
            args=(req.model_id, req.model_dump()),
            daemon=True,
        )
        with load_threads_lock:
            load_threads.append(thread)
        thread.start()

        return JSONResponse(
            {
                "status": "loading_started",
                "message": f"Model '{req.model_id}' is loading in the background.",
            }
        )

    @app.post("/admin/unload_model")
    def unload_model_endpoint(
        model_id: str, request: Request, drain_timeout_sec: float | None = None
    ) -> JSONResponse:
        _require_admin(request)
        if not model_registry.unload_model:
            raise STTError(ErrorCode.ADMIN_API_DISABLED)

        success = model_registry.unload_model(
            model_id, drain_timeout_sec=drain_timeout_sec
        )
        if success:
            return JSONResponse({"status": "unloaded", "model_id": model_id})
        raise STTError(ErrorCode.MODEL_UNLOAD_FAILED)

    @app.get("/admin/list_models")
    def list_models_endpoint(request: Request) -> JSONResponse:
        _require_admin(request)
        return JSONResponse({"models": model_registry.list_models()})

    return app, load_threads, load_threads_lock


def start_http_server(
    runtime: ApplicationRuntime,
    server_state: Dict[str, bool],
    host: str,
    port: int,
) -> HttpServerHandle:
    """Start FastAPI app for /metrics and /health in a background thread."""
    app, load_threads, load_threads_lock = build_http_app(runtime, server_state)
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        log_config=_build_uvicorn_log_config(),
    )
    server = uvicorn.Server(config)

    def run_server() -> None:
        server.run()

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    return HttpServerHandle(
        server=server,
        thread=thread,
        load_threads=load_threads,
        load_threads_lock=load_threads_lock,
    )
