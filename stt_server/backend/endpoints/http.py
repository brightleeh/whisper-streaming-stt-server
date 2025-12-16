import threading
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

from stt_server.backend.core.metrics import Metrics


def start_observability_server(
    metrics: Metrics,
    servicer: Any,
    server_state: Dict[str, bool],
    host: str,
    port: int,
) -> None:
    """Start FastAPI app for /metrics and /health in a background thread."""
    app = FastAPI()

    @app.get("/metrics")
    def metrics_endpoint() -> Response:
        return Response(content=metrics.render(), media_type="text/plain")

    @app.get("/health")
    def health_endpoint() -> JSONResponse:
        snapshot = servicer.health_snapshot()
        snapshot["grpc_running"] = server_state.get("grpc_running", False)
        healthy = snapshot["grpc_running"] and snapshot["model_pool_healthy"]
        status = 200 if healthy else 500
        payload = {"status": "ok" if healthy else "error", **snapshot}
        return JSONResponse(payload, status_code=status)

    def run_server() -> None:
        uvicorn.run(app, host=host, port=port, log_level="info")

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
