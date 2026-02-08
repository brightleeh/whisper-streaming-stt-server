"""HTTP endpoints for metrics, health, and admin actions."""

import ipaddress
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
from stt_server.backend.utils.rate_limit import KeyedRateLimiter
from stt_server.backend.utils.system_metrics import collect_system_metrics
from stt_server.config.default.model import (
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_MODEL_LOAD_PROFILE_NAME,
    DEFAULT_MODEL_NAME,
)
from stt_server.errors import ErrorCode, STTError, http_payload_for, http_status_for

_ACCESS_LOG_IGNORED_PATHS = frozenset(
    {"/metrics", "/metrics.json", "/system", "/health"}
)
_ADMIN_ENABLE_ENV = "STT_ADMIN_ENABLED"
_ADMIN_TOKEN_ENV = "STT_ADMIN_TOKEN"
_ADMIN_ALLOW_MODEL_PATH_ENV = "STT_ADMIN_ALLOW_MODEL_PATH"
_ADMIN_MODEL_PATH_ALLOWLIST_ENV = "STT_ADMIN_MODEL_PATH_ALLOWLIST"
_OBS_TOKEN_ENV = "STT_OBSERVABILITY_TOKEN"
_PUBLIC_HEALTH_ENV = "STT_PUBLIC_HEALTH"
_HEALTH_DETAIL_MODE_ENV = "STT_HEALTH_DETAIL_MODE"
_HTTP_RATE_LIMIT_RPS_ENV = "STT_HTTP_RATE_LIMIT_RPS"
_HTTP_RATE_LIMIT_BURST_ENV = "STT_HTTP_RATE_LIMIT_BURST"
_HTTP_ALLOWLIST_ENV = "STT_HTTP_ALLOWLIST"
_HTTP_TRUSTED_PROXIES_ENV = "STT_HTTP_TRUSTED_PROXIES"
LOGGER = logging.getLogger("stt_server.http_server")


class _AccessLogPathFilter(logging.Filter):
    """Filter out noisy access logs for internal endpoints."""

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


def _observability_token() -> str:
    return os.getenv(_OBS_TOKEN_ENV, "").strip()


def _public_health_mode() -> str:
    value = os.getenv(_PUBLIC_HEALTH_ENV, "").strip().lower()
    if value in {"1", "true", "yes", "on", "minimal"}:
        return "minimal"
    return ""


def _health_detail_mode() -> str:
    value = os.getenv(_HEALTH_DETAIL_MODE_ENV, "").strip().lower()
    if value in {"1", "true", "yes", "on", "token"}:
        return "token"
    return ""


def _require_admin(request: Request) -> None:
    if not _env_enabled(_ADMIN_ENABLE_ENV) or not _admin_token():
        raise STTError(ErrorCode.ADMIN_API_DISABLED)
    auth = request.headers.get("authorization", "").strip()
    if not auth.lower().startswith("bearer "):
        raise STTError(ErrorCode.ADMIN_UNAUTHORIZED)
    token = auth[7:].strip()
    if token != _admin_token():
        raise STTError(ErrorCode.ADMIN_UNAUTHORIZED)


def _require_observability(request: Request) -> None:
    token = _observability_token()
    if not token:
        return
    auth = request.headers.get("authorization", "").strip()
    if not auth.lower().startswith("bearer "):
        raise STTError(ErrorCode.OBS_UNAUTHORIZED)
    if auth[7:].strip() != token:
        raise STTError(ErrorCode.OBS_UNAUTHORIZED)


def _parse_rate_limit_value(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _create_rate_limiter(
    rate_limit_rps: float, rate_limit_burst: float
) -> KeyedRateLimiter:
    return KeyedRateLimiter(rate_limit_rps, rate_limit_burst or None)


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


def _request_fields_set(model: BaseModel) -> set[str]:
    fields_set = getattr(model, "model_fields_set", None)
    if fields_set is None:
        fields_set = getattr(model, "__fields_set__", set())
    return set(fields_set or ())


class LoadModelRequest(BaseModel):
    """Request body for admin model load endpoint."""

    model_id: str
    profile_id: Optional[str] = None
    model_path: Optional[str] = None  # local path or HuggingFace ID
    model_size: Optional[str] = (
        DEFAULT_MODEL_NAME  # size name to use if model_path is missing (e.g. "small")
    )
    backend: Optional[str] = None
    device: str = DEFAULT_DEVICE
    compute_type: str = DEFAULT_COMPUTE_TYPE
    language: Optional[str] = None


@dataclass
class LoadJobState:
    """Track the status of a background model load."""

    status: str
    model_id: str
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "status": self.status,
            "model_id": self.model_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        if self.error:
            payload["error"] = self.error
        if self.started_at is not None and self.finished_at is not None:
            payload["duration_sec"] = max(0.0, self.finished_at - self.started_at)
        return payload


@dataclass
class HttpServerHandle:
    """Handle for the background HTTP server and admin load threads."""

    server: uvicorn.Server
    thread: threading.Thread
    load_threads: List[threading.Thread]
    load_threads_lock: threading.Lock

    def stop(self, timeout: Optional[float] = None) -> None:
        """Stop the HTTP server and wait for background load threads."""
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
        with self.load_threads_lock:
            self.load_threads[:] = [
                thread for thread in self.load_threads if thread.is_alive()
            ]


def build_http_app(
    runtime: ApplicationRuntime,
    server_state: Dict[str, bool],
    http_rate_limit_rps: float | None = None,
    http_rate_limit_burst: float | None = None,
    http_trusted_proxies: Optional[List[str]] = None,
) -> Tuple[FastAPI, List[threading.Thread], threading.Lock]:
    """Create the FastAPI app and load-model thread tracking state."""
    app = FastAPI()
    metrics = runtime.metrics
    model_registry = runtime.model_registry
    health_snapshot = runtime.health_snapshot
    load_threads: List[threading.Thread] = []
    load_threads_lock = threading.Lock()
    if http_rate_limit_rps is None:
        rate_limit_rps = _parse_rate_limit_value(
            os.getenv(_HTTP_RATE_LIMIT_RPS_ENV, ""), 0.0
        )
    else:
        rate_limit_rps = float(http_rate_limit_rps)
    if http_rate_limit_burst is None:
        rate_limit_burst = _parse_rate_limit_value(
            os.getenv(_HTTP_RATE_LIMIT_BURST_ENV, ""), max(1.0, rate_limit_rps)
        )
    else:
        rate_limit_burst = float(http_rate_limit_burst)
    rate_limiter = _create_rate_limiter(rate_limit_rps, rate_limit_burst)
    allowlist_raw = os.getenv(_HTTP_ALLOWLIST_ENV, "")
    allowlist: List[ipaddress._BaseNetwork] = []
    for entry in [item.strip() for item in allowlist_raw.split(",") if item.strip()]:
        try:
            allowlist.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            LOGGER.warning("Invalid HTTP allowlist entry ignored: %s", entry)
    if http_trusted_proxies is None:
        trusted_proxies_raw = os.getenv(_HTTP_TRUSTED_PROXIES_ENV, "")
        trusted_proxy_entries = [
            item.strip() for item in trusted_proxies_raw.split(",") if item.strip()
        ]
    else:
        trusted_proxy_entries = [item.strip() for item in http_trusted_proxies if item]
    trusted_proxies: List[ipaddress._BaseNetwork] = []
    trusted_proxy_hosts: List[str] = []
    for entry in trusted_proxy_entries:
        try:
            trusted_proxies.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            if any(ch.isalpha() for ch in entry):
                trusted_proxy_hosts.append(entry)
            else:
                LOGGER.warning("Invalid HTTP trusted proxy entry ignored: %s", entry)
    load_statuses: Dict[str, LoadJobState] = {}
    load_statuses_lock = threading.Lock()
    model_profiles: Dict[str, Dict[str, Any]] = {}
    default_model_profile: Optional[str] = None
    runtime_config = getattr(runtime, "config", None)
    if runtime_config is not None:
        model_config = getattr(runtime_config, "model", None)
        if model_config is not None:
            profiles = getattr(model_config, "model_load_profiles", None)
            if isinstance(profiles, dict):
                model_profiles = profiles
            default_model_profile = getattr(
                model_config, "default_model_load_profile", None
            )

    def _runtime_snapshot() -> Dict[str, Any]:
        config = getattr(runtime, "config", None)
        if config is None:
            return {}
        model_cfg = getattr(config, "model", None)
        streaming_cfg = getattr(config, "streaming", None)
        payload: Dict[str, Any] = {}
        if model_cfg is not None:
            payload["model"] = {
                "model_size": getattr(model_cfg, "model_size", None),
                "model_backend": getattr(model_cfg, "model_backend", None),
                "device": getattr(model_cfg, "device", None),
                "compute_type": getattr(model_cfg, "compute_type", None),
                "model_pool_size": getattr(model_cfg, "model_pool_size", None),
                "default_decode_profile": getattr(
                    model_cfg, "default_decode_profile", None
                ),
                "language": getattr(model_cfg, "language", None),
                "task": getattr(model_cfg, "task", None),
            }
        if streaming_cfg is not None:
            payload["streaming"] = {
                "sample_rate": getattr(streaming_cfg, "sample_rate", None),
                "session_timeout_sec": getattr(streaming_cfg, "session_timeout_sec", None),
                "decode_timeout_sec": getattr(streaming_cfg, "decode_timeout_sec", None),
                "create_session_rps": getattr(
                    streaming_cfg, "create_session_rps", None
                ),
                "create_session_burst": getattr(
                    streaming_cfg, "create_session_burst", None
                ),
                "vad_model_pool_size": getattr(
                    streaming_cfg, "vad_model_pool_size", None
                ),
                "vad_model_prewarm": getattr(
                    streaming_cfg, "vad_model_prewarm", None
                ),
                "vad_silence": getattr(streaming_cfg, "vad_silence", None),
                "vad_threshold": getattr(streaming_cfg, "vad_threshold", None),
                "max_chunk_ms": getattr(streaming_cfg, "max_chunk_ms", None),
                "partial_decode_interval_sec": getattr(
                    streaming_cfg, "partial_decode_interval_sec", None
                ),
                "partial_decode_window_sec": getattr(
                    streaming_cfg, "partial_decode_window_sec", None
                ),
                "decode_batch_window_ms": getattr(
                    streaming_cfg, "decode_batch_window_ms", None
                ),
                "max_decode_batch_size": getattr(
                    streaming_cfg, "max_decode_batch_size", None
                ),
                "max_pending_decodes_global": getattr(
                    streaming_cfg, "max_pending_decodes_global", None
                ),
                "max_pending_decodes_per_stream": getattr(
                    streaming_cfg, "max_pending_decodes_per_stream", None
                ),
                "adaptive_throttle_enabled": getattr(
                    streaming_cfg, "adaptive_throttle_enabled", None
                ),
            }
        return payload

    def _prune_load_threads() -> None:
        with load_threads_lock:
            if not load_threads:
                return
            load_threads[:] = [thread for thread in load_threads if thread.is_alive()]

    def _extract_client_ip(request: Request) -> str:
        client = request.client
        client_ip = client.host if client else ""
        trusted = client_ip in trusted_proxy_hosts
        if not trusted and trusted_proxies:
            try:
                addr = ipaddress.ip_address(client_ip)
            except ValueError:
                addr = None
            if addr is not None and any(addr in network for network in trusted_proxies):
                trusted = True
        if not trusted:
            return client_ip
        forwarded_for = request.headers.get("x-forwarded-for", "").strip()
        if not forwarded_for:
            return client_ip
        return forwarded_for.split(",")[-1].strip()

    def _enforce_ip_allowlist(request: Request) -> None:
        if not allowlist:
            return
        client_ip = _extract_client_ip(request)
        try:
            addr = ipaddress.ip_address(client_ip)
        except ValueError:
            raise STTError(ErrorCode.HTTP_IP_FORBIDDEN)
        if not any(addr in network for network in allowlist):
            raise STTError(ErrorCode.HTTP_IP_FORBIDDEN)

    def _enforce_rate_limit(request: Request) -> None:
        key = _extract_client_ip(request) or "unknown"
        if not rate_limiter.allow(key):
            metrics.record_rate_limit_block("http", key)
            raise STTError(ErrorCode.HTTP_RATE_LIMITED)

    def _get_load_status(model_id: str) -> Optional[LoadJobState]:
        with load_statuses_lock:
            return load_statuses.get(model_id)

    def _set_load_status(
        model_id: str,
        status: str,
        *,
        started_at: Optional[float] = None,
        finished_at: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        with load_statuses_lock:
            state = load_statuses.get(model_id)
            if state is None:
                state = LoadJobState(status=status, model_id=model_id)
                load_statuses[model_id] = state
            state.status = status
            if started_at is not None:
                state.started_at = started_at
            if finished_at is not None:
                state.finished_at = finished_at
            state.error = error

    @app.exception_handler(STTError)
    async def stt_error_handler(_request: Request, exc: STTError) -> JSONResponse:
        return JSONResponse(
            http_payload_for(exc.code, exc.detail),
            status_code=http_status_for(exc.code),
        )

    def _sanitize_metric_name(value: str) -> str:
        sanitized = []
        for idx, ch in enumerate(value):
            if ch.isalnum() or ch == "_":
                sanitized.append(ch)
            else:
                sanitized.append("_")
            if idx == 0 and sanitized[-1].isdigit():
                sanitized.insert(0, "m")
        return "".join(sanitized) or "metric"

    def _flatten_metrics(payload: Dict[str, Any]) -> Dict[str, float]:
        flat: Dict[str, float] = {}
        for key, value in payload.items():
            if value is None:
                continue
            if isinstance(value, (int, float, bool)):
                flat[_sanitize_metric_name(key)] = float(value)
            elif isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    if isinstance(sub_val, (int, float, bool)):
                        metric_key = _sanitize_metric_name(f"{key}_{sub_key}")
                        flat[metric_key] = float(sub_val)
        return flat

    def _prometheus_text(payload: Dict[str, Any]) -> str:
        flat = _flatten_metrics(payload)
        lines: List[str] = []
        for key in sorted(flat.keys()):
            metric_name = f"stt_{key}"
            lines.append(
                f"# HELP {metric_name} Server metric '{key}' exposed as a gauge."
            )
            lines.append(f"# TYPE {metric_name} gauge")
            lines.append(f"{metric_name} {flat[key]}")
        return "\n".join(lines) + "\n"

    @app.get("/metrics")
    def metrics_endpoint(request: Request) -> Response:
        _enforce_ip_allowlist(request)
        _enforce_rate_limit(request)
        _require_observability(request)
        payload = metrics.render()
        text = _prometheus_text(payload)
        return Response(content=text, media_type="text/plain; version=0.0.4")

    @app.get("/metrics.json")
    def metrics_json_endpoint(request: Request) -> JSONResponse:
        _enforce_ip_allowlist(request)
        _enforce_rate_limit(request)
        _require_observability(request)
        return JSONResponse(metrics.render(), status_code=200)

    @app.get("/health")
    def health_endpoint(request: Request) -> JSONResponse:
        _enforce_ip_allowlist(request)
        _enforce_rate_limit(request)
        public_mode = _public_health_mode()
        detail_mode = _health_detail_mode()
        if public_mode == "minimal" or detail_mode == "token":
            token = _observability_token()
            auth_ok = False
            if token:
                auth = request.headers.get("authorization", "").strip()
                if auth.lower().startswith("bearer ") and auth[7:].strip() == token:
                    auth_ok = True
            snapshot = health_snapshot()
            snapshot["grpc_running"] = server_state.get("grpc_running", False)
            healthy = snapshot["grpc_running"] and snapshot["model_pool_healthy"]
            status = 200 if healthy else 503
            payload = {"status": "ok" if healthy else "error"}
            if auth_ok:
                payload.update(snapshot)
            return JSONResponse(payload, status_code=status)
        _require_observability(request)
        snapshot = health_snapshot()
        snapshot["grpc_running"] = server_state.get("grpc_running", False)
        healthy = snapshot["grpc_running"] and snapshot["model_pool_healthy"]
        status = 200 if healthy else 500
        payload = {"status": "ok" if healthy else "error", **snapshot}
        return JSONResponse(payload, status_code=status)

    @app.get("/system")
    def system_endpoint(request: Request) -> JSONResponse:
        _enforce_ip_allowlist(request)
        _enforce_rate_limit(request)
        _require_observability(request)
        payload = collect_system_metrics()
        runtime_payload = _runtime_snapshot()
        if runtime_payload:
            payload["runtime"] = runtime_payload
        return JSONResponse(payload, status_code=200)

    @app.post("/admin/load_model")
    def load_model_endpoint(req: LoadModelRequest, request: Request) -> JSONResponse:
        _enforce_ip_allowlist(request)
        _enforce_rate_limit(request)
        _require_admin(request)
        load_fn = getattr(model_registry, "load_model", None)
        if not callable(load_fn):
            raise STTError(ErrorCode.ADMIN_API_DISABLED)

        _prune_load_threads()
        existing = _get_load_status(req.model_id)
        if existing and existing.status in {"queued", "running"}:
            return JSONResponse(
                {
                    "status": existing.status,
                    "message": f"Model '{req.model_id}' is already loading.",
                    "job": existing.to_payload(),
                }
            )
        if model_registry.is_loaded(req.model_id):
            raise STTError(
                ErrorCode.MODEL_ALREADY_LOADED,
                f"Model '{req.model_id}' is already loaded",
            )
        profile_id = req.profile_id
        legacy_fields = {
            "model_path",
            "model_size",
            "backend",
            "device",
            "compute_type",
            "language",
        }
        fields_set = _request_fields_set(req)
        use_legacy = bool(fields_set & legacy_fields)
        if profile_id:
            profile_cfg = model_profiles.get(profile_id)
            if not profile_cfg:
                raise STTError(
                    ErrorCode.ADMIN_MODEL_PROFILE_UNKNOWN,
                    f"Unknown model profile '{profile_id}'",
                )
            load_config = dict(profile_cfg)
        elif model_profiles and not use_legacy:
            profile_id = default_model_profile or DEFAULT_MODEL_LOAD_PROFILE_NAME
            profile_cfg = model_profiles.get(profile_id)
            if not profile_cfg:
                raise STTError(
                    ErrorCode.ADMIN_MODEL_PROFILE_UNKNOWN,
                    f"Unknown model profile '{profile_id}'",
                )
            load_config = dict(profile_cfg)
        else:
            load_config = req.model_dump(exclude={"profile_id"})
        if not load_config.get("backend") and not load_config.get("model_backend"):
            default_backend = getattr(
                runtime.config.model, "model_backend", "faster_whisper"
            )
            load_config["backend"] = default_backend or "faster_whisper"
        if not _model_path_allowed(load_config.get("model_path")):
            raise STTError(ErrorCode.ADMIN_MODEL_PATH_FORBIDDEN)

        _set_load_status(
            req.model_id,
            "queued",
            error=None,
            started_at=None,
            finished_at=None,
        )

        # Load in a separate thread to prevent blocking the main loop
        def _load_model_safe() -> None:
            _set_load_status(
                req.model_id, "running", started_at=time.time(), error=None
            )
            try:
                load_fn(req.model_id, load_config)
            except (OSError, RuntimeError, TypeError, ValueError, STTError) as exc:
                error = str(exc).strip() or exc.__class__.__name__
                _set_load_status(
                    req.model_id,
                    "failed",
                    finished_at=time.time(),
                    error=error,
                )
                LOGGER.exception("Failed to load model '%s'", req.model_id)
                return
            _set_load_status(
                req.model_id, "success", finished_at=time.time(), error=None
            )

        thread = threading.Thread(target=_load_model_safe, daemon=True)
        with load_threads_lock:
            load_threads.append(thread)
        thread.start()

        job_state = _get_load_status(req.model_id)
        return JSONResponse(
            {
                "status": "loading_started",
                "message": f"Model '{req.model_id}' is loading in the background.",
                "job": (
                    job_state.to_payload()
                    if job_state
                    else {"status": "unknown", "model_id": req.model_id}
                ),
            }
        )

    @app.get("/admin/load_model_status")
    def load_model_status_endpoint(model_id: str, request: Request) -> JSONResponse:
        _enforce_ip_allowlist(request)
        _enforce_rate_limit(request)
        _require_admin(request)
        state = _get_load_status(model_id)
        if not state:
            return JSONResponse({"status": "unknown", "model_id": model_id})
        return JSONResponse(state.to_payload())

    @app.post("/admin/unload_model")
    def unload_model_endpoint(
        model_id: str, request: Request, drain_timeout_sec: float | None = None
    ) -> JSONResponse:
        _enforce_ip_allowlist(request)
        _enforce_rate_limit(request)
        _require_admin(request)
        unload_fn = getattr(model_registry, "unload_model", None)
        if not callable(unload_fn):
            raise STTError(ErrorCode.ADMIN_API_DISABLED)

        success = unload_fn(model_id, drain_timeout_sec=drain_timeout_sec)
        if success:
            return JSONResponse({"status": "unloaded", "model_id": model_id})
        raise STTError(ErrorCode.MODEL_UNLOAD_FAILED)

    @app.get("/admin/list_models")
    def list_models_endpoint(request: Request) -> JSONResponse:
        _enforce_ip_allowlist(request)
        _enforce_rate_limit(request)
        _require_admin(request)
        return JSONResponse({"models": model_registry.list_models()})

    return app, load_threads, load_threads_lock


def start_http_server(
    runtime: ApplicationRuntime,
    server_state: Dict[str, bool],
    host: str,
    port: int,
    http_rate_limit_rps: float | None = None,
    http_rate_limit_burst: float | None = None,
    http_trusted_proxies: Optional[List[str]] = None,
) -> HttpServerHandle:
    """Start FastAPI app for /metrics and /health in a background thread."""
    app, load_threads, load_threads_lock = build_http_app(
        runtime,
        server_state,
        http_rate_limit_rps=http_rate_limit_rps,
        http_rate_limit_burst=http_rate_limit_burst,
        http_trusted_proxies=http_trusted_proxies,
    )
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
