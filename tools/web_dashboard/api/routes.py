"""API routes for the web dashboard controller."""

from __future__ import annotations

import json
import queue
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from tools.web_dashboard.core.run_manager import RunManager


class UploadResponse(BaseModel):
    audio_path: str
    sha256: str
    bytes: int


class TargetResponse(BaseModel):
    id: str
    grpc_target: str
    http_base: str


class TargetStatus(BaseModel):
    grpc_ok: bool
    system_ok: bool
    metrics_ok: bool
    rtt_ms: int
    ts: float
    last_ok_ts: Optional[float] = None
    runtime: Optional[Dict[str, Any]] = None
    server_errors: Optional[Dict[str, Any]] = None


class RunRequest(BaseModel):
    target_id: str
    audio_path: str
    channels: int = Field(1, ge=1)
    duration_sec: Optional[float] = None
    ramp_steps: int = 5
    ramp_interval_sec: float = 2.0
    chunk_ms: int = 20
    realtime: bool = True
    speed: float = 1.0
    task: str = "transcribe"
    language: Optional[str] = None
    vad_mode: str = "auto"
    decode_profile: str = "realtime"
    attrs: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, str] = Field(default_factory=dict)
    token: str = ""
    create_session_backoff_ms: int = 100


class RunResponse(BaseModel):
    run_id: str
    started_at: float


class StopResponse(BaseModel):
    ok: bool


class LatestResponse(BaseModel):
    active: bool
    run_id: Optional[str] = None
    started_at: Optional[float] = None
    target_id: Optional[str] = None
    paths: Optional[Dict[str, str]] = None


class DefaultsResponse(BaseModel):
    defaults: Dict[str, Any]
    languages: List[Dict[str, str]]


def build_router(run_manager: RunManager) -> APIRouter:
    router = APIRouter()

    @router.post("/upload", response_model=UploadResponse)
    async def upload_audio(file: UploadFile = File(...)) -> Dict[str, Any]:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty upload")
        result = run_manager.upload_audio(file.filename or "upload.wav", data)
        return result

    @router.get("/targets", response_model=List[TargetResponse])
    def list_targets() -> List[Dict[str, str]]:
        return [target.__dict__ for target in run_manager.list_targets()]

    @router.get("/targets/{target_id}/status", response_model=TargetStatus)
    def target_status(target_id: str) -> Dict[str, Any]:
        target = run_manager.get_target(target_id)
        if not target:
            raise HTTPException(status_code=404, detail="Unknown target")
        return run_manager.probe_target(target)

    @router.post("/runs", response_model=RunResponse)
    def start_run(req: RunRequest) -> Dict[str, Any]:
        if not Path(req.audio_path).exists():
            raise HTTPException(status_code=400, detail="audio_path not found")
        try:
            state = run_manager.start_run(req.dict())
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"run_id": state.run_id, "started_at": state.config.started_at}

    @router.post("/runs/{run_id}/stop", response_model=StopResponse)
    def stop_run(run_id: str) -> Dict[str, Any]:
        run_manager.stop_run(run_id)
        return {"ok": True}

    @router.get("/runs/latest", response_model=LatestResponse)
    def latest_run() -> Dict[str, Any]:
        return run_manager.get_latest_payload()

    @router.get("/defaults", response_model=DefaultsResponse)
    def defaults() -> Dict[str, Any]:
        return run_manager.get_defaults_payload()

    @router.get("/runs/history")
    def run_history() -> Dict[str, Any]:
        return {"runs": run_manager.list_runs()}

    @router.get("/runs/{run_id}/report")
    def run_report(run_id: str) -> FileResponse:
        report_path = run_manager.ensure_report(run_id)
        if not report_path:
            raise HTTPException(status_code=404, detail="Report not available")
        return FileResponse(
            report_path,
            media_type="text/markdown",
            filename=f"{run_id}.md",
        )

    @router.get("/runs/{run_id}/sessions.csv")
    def run_sessions_csv(run_id: str) -> FileResponse:
        csv_path = run_manager.runs_dir / run_id / "sessions.csv"
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail="Session CSV not available")
        return FileResponse(
            csv_path,
            media_type="text/csv",
            filename=f"{run_id}.csv",
        )

    @router.get("/runs/{run_id}/sessions/preview")
    def run_sessions_preview(run_id: str, limit: int = 200) -> Dict[str, Any]:
        safe_limit = max(1, min(1000, int(limit)))
        return run_manager.get_sessions_preview(run_id, limit=safe_limit)

    @router.get("/runs/{run_id}/live")
    def live_events(run_id: str) -> StreamingResponse:
        state = run_manager.active_run()
        if not state or state.run_id != run_id:
            raise HTTPException(status_code=404, detail="Run not active")
        q = run_manager.subscribe()

        def format_sse(event: str, payload: Dict[str, Any]) -> str:
            return f"event: {event}\ndata: {json.dumps(payload)}\n\n"

        def event_stream():
            try:
                while True:
                    try:
                        message = q.get(timeout=5.0)
                        yield format_sse(message["event"], message["data"])
                    except queue.Empty:
                        yield format_sse("ping", {"ts": time.time()})
                    active = run_manager.active_run()
                    if not active or active.run_id != run_id:
                        yield format_sse("done", {"ts": time.time(), "reason": "ended"})
                        break
            finally:
                run_manager.unsubscribe(q)

        response = StreamingResponse(event_stream(), media_type="text/event-stream")
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Connection"] = "keep-alive"
        response.headers["X-Accel-Buffering"] = "no"
        return response

    return router
