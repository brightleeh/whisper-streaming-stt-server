import threading
from dataclasses import dataclass
from typing import Dict, Optional

import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from stt_server.backend.runtime import ApplicationRuntime
from stt_server.config.default.model import (
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_MODEL_NAME,
)


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

    def stop(self, timeout: Optional[float] = None) -> None:
        if self.thread.is_alive():
            self.server.should_exit = True
            self.thread.join(timeout=timeout)


def start_http_server(
    runtime: ApplicationRuntime,
    server_state: Dict[str, bool],
    host: str,
    port: int,
) -> HttpServerHandle:
    """Start FastAPI app for /metrics and /health in a background thread."""
    app = FastAPI()

    metrics = runtime.metrics
    model_registry = runtime.model_registry
    health_snapshot = runtime.health_snapshot

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

    @app.post("/admin/load_model")
    def load_model_endpoint(req: LoadModelRequest) -> JSONResponse:
        if not model_registry.load_model:
            return JSONResponse({"error": "Admin API not enabled"}, status_code=501)

        if model_registry.is_loaded(req.model_id):
            return JSONResponse(
                {"error": f"Model '{req.model_id}' is already loaded"},
                status_code=409,
            )

        # Load in a separate thread to prevent blocking the main loop
        threading.Thread(
            target=model_registry.load_model, args=(req.model_id, req.model_dump())
        ).start()

        return JSONResponse(
            {
                "status": "loading_started",
                "message": f"Model '{req.model_id}' is loading in the background.",
            }
        )

    @app.post("/admin/unload_model")
    def unload_model_endpoint(model_id: str) -> JSONResponse:
        if not model_registry.unload_model:
            return JSONResponse({"error": "Admin API not enabled"}, status_code=501)

        success = model_registry.unload_model(model_id)
        if success:
            return JSONResponse({"status": "unloaded", "model_id": model_id})
        else:
            return JSONResponse(
                {"status": "failed", "message": "Model not found or is default"},
                status_code=400,
            )

    @app.get("/admin/list_models")
    def list_models_endpoint() -> JSONResponse:
        return JSONResponse({"models": model_registry.list_models()})

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None

    def run_server() -> None:
        server.run()

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    return HttpServerHandle(server=server, thread=thread)
