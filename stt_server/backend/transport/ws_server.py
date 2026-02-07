"""WebSocket bridge for browser streaming clients."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import queue
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

import grpc
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from gen.stt.python.v1 import stt_pb2
from stt_server.backend.runtime import ApplicationRuntime
from stt_server.backend.utils.rate_limit import KeyedRateLimiter
from stt_server.errors import ErrorCode, STTError, http_payload_for

_HTTP_RATE_LIMIT_RPS_ENV = "STT_HTTP_RATE_LIMIT_RPS"
_HTTP_RATE_LIMIT_BURST_ENV = "STT_HTTP_RATE_LIMIT_BURST"
_HTTP_ALLOWLIST_ENV = "STT_HTTP_ALLOWLIST"
_HTTP_TRUSTED_PROXIES_ENV = "STT_HTTP_TRUSTED_PROXIES"
LOGGER = logging.getLogger("stt_server.ws_server")


def _parse_rate_limit_value(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _create_rate_limiter(
    rate_limit_rps: float, rate_limit_burst: float
) -> KeyedRateLimiter:
    return KeyedRateLimiter(rate_limit_rps, rate_limit_burst or None)


def _extract_ws_client_ip(
    websocket: WebSocket,
    trusted_proxy_hosts: List[str],
    trusted_proxies: List[ipaddress._BaseNetwork],
) -> str:
    client = websocket.client
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
    forwarded_for = websocket.headers.get("x-forwarded-for", "").strip()
    if not forwarded_for:
        return client_ip
    return forwarded_for.split(",")[-1].strip()


def _peer_for_ws(client_ip: str) -> str:
    if not client_ip:
        return ""
    if ":" in client_ip:
        return f"ipv6:[{client_ip}]:0"
    return f"ipv4:{client_ip}:0"


def _parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_decode_profile(value: Any) -> int:
    if isinstance(value, int):
        return value
    raw = str(value or "").strip().lower()
    if raw in {"realtime", "rt", "low", "low_latency"}:
        return stt_pb2.DECODE_PROFILE_REALTIME
    if raw in {"accurate", "accuracy", "high"}:
        return stt_pb2.DECODE_PROFILE_ACCURATE
    return stt_pb2.DECODE_PROFILE_UNSPECIFIED


def _parse_task(value: Any) -> int:
    if isinstance(value, int):
        return value
    raw = str(value or "").strip().lower()
    if raw in {"translate", "translation"}:
        return stt_pb2.TASK_TRANSLATE
    if raw in {"transcribe", "transcription"}:
        return stt_pb2.TASK_TRANSCRIBE
    return stt_pb2.TASK_UNSPECIFIED


def _parse_vad_mode(value: Any) -> int:
    if isinstance(value, int):
        return value
    raw = str(value or "").strip().lower()
    if raw in {"auto", "auto_end", "auto-end", "end"}:
        return stt_pb2.VAD_AUTO_END
    return stt_pb2.VAD_CONTINUE


def _normalize_attributes(raw: Any) -> Dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    normalized: Dict[str, str] = {}
    for key, value in raw.items():
        if key is None:
            continue
        if value is None:
            continue
        normalized[str(key)] = str(value)
    return normalized


def _normalize_metadata(raw: Any) -> List[Tuple[str, str]]:
    if not isinstance(raw, dict):
        return []
    items: List[Tuple[str, str]] = []
    for key, value in raw.items():
        if not key:
            continue
        if value is None:
            continue
        items.append((str(key).lower(), str(value)))
    return items


class _WebSocketAbort(RuntimeError):
    def __init__(self, status: grpc.StatusCode, details: str) -> None:
        super().__init__(details)
        self.status = status
        self.details = details


class _WebSocketContext:
    def __init__(
        self,
        websocket: WebSocket,
        metadata: List[Tuple[str, str]],
        peer: str,
    ) -> None:
        self._websocket = websocket
        self._metadata = metadata
        self._peer = peer
        self._callbacks: List[Any] = []
        self._active = True
        self.trailing_metadata = None

    def invocation_metadata(self):
        return list(self._metadata)

    def add_callback(self, callback):
        self._callbacks.append(callback)

    def is_active(self) -> bool:
        return self._active

    def set_trailing_metadata(self, metadata):
        self.trailing_metadata = metadata

    def peer(self) -> str:
        return self._peer

    def abort(self, code, details):
        raise _WebSocketAbort(code, details)

    def close(self) -> None:
        if not self._active:
            return
        self._active = False
        for callback in self._callbacks:
            try:
                callback()
            except Exception:
                LOGGER.exception("WebSocket callback failed")


def build_ws_app(
    runtime: ApplicationRuntime,
    ws_rate_limit_rps: float | None = None,
    ws_rate_limit_burst: float | None = None,
    ws_trusted_proxies: Optional[List[str]] = None,
) -> FastAPI:
    app = FastAPI()
    metrics = runtime.metrics

    if ws_rate_limit_rps is None:
        rate_limit_rps = _parse_rate_limit_value(
            os.getenv(_HTTP_RATE_LIMIT_RPS_ENV, ""), 0.0
        )
    else:
        rate_limit_rps = float(ws_rate_limit_rps)
    if ws_rate_limit_burst is None:
        rate_limit_burst = _parse_rate_limit_value(
            os.getenv(_HTTP_RATE_LIMIT_BURST_ENV, ""), max(1.0, rate_limit_rps)
        )
    else:
        rate_limit_burst = float(ws_rate_limit_burst)
    rate_limiter = _create_rate_limiter(rate_limit_rps, rate_limit_burst)

    allowlist_raw = os.getenv(_HTTP_ALLOWLIST_ENV, "")
    allowlist: List[ipaddress._BaseNetwork] = []
    for entry in [item.strip() for item in allowlist_raw.split(",") if item.strip()]:
        try:
            allowlist.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            LOGGER.warning("Invalid HTTP allowlist entry ignored: %s", entry)

    if ws_trusted_proxies is None:
        trusted_proxy_entries = [
            item.strip()
            for item in os.getenv(_HTTP_TRUSTED_PROXIES_ENV, "").split(",")
            if item.strip()
        ]
    else:
        trusted_proxy_entries = [item.strip() for item in ws_trusted_proxies if item]
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

    def _enforce_ws_allowlist(client_ip: str) -> None:
        if not allowlist:
            return
        try:
            addr = ipaddress.ip_address(client_ip)
        except ValueError:
            raise STTError(ErrorCode.HTTP_IP_FORBIDDEN)
        if not any(addr in network for network in allowlist):
            raise STTError(ErrorCode.HTTP_IP_FORBIDDEN)

    def _enforce_ws_rate_limit(client_ip: str) -> None:
        key = client_ip or "unknown"
        if not rate_limiter.allow(key):
            metrics.record_rate_limit_block("http", key)
            raise STTError(ErrorCode.HTTP_RATE_LIMITED)

    @app.websocket("/ws/stream")
    async def websocket_stream(websocket: WebSocket) -> None:
        await websocket.accept()
        client_ip = _extract_ws_client_ip(
            websocket, trusted_proxy_hosts, trusted_proxies
        )
        try:
            _enforce_ws_allowlist(client_ip)
            _enforce_ws_rate_limit(client_ip)
        except STTError as exc:
            await websocket.send_json(http_payload_for(exc.code, exc.detail))
            await websocket.close(code=4403)
            return
        try:
            start_payload = await websocket.receive_json()
        except Exception:
            await websocket.close(code=1003)
            return

        if isinstance(start_payload, dict) and start_payload.get("type") == "start":
            payload = start_payload.get("data") or start_payload
        else:
            payload = start_payload if isinstance(start_payload, dict) else {}

        session_id = str(payload.get("session_id") or uuid.uuid4().hex)
        sample_rate = int(payload.get("sample_rate") or 16000)
        attributes = _normalize_attributes(payload.get("attributes"))
        metadata = _normalize_metadata(payload.get("metadata"))

        context = _WebSocketContext(websocket, metadata, _peer_for_ws(client_ip))
        request = stt_pb2.SessionRequest(
            session_id=session_id,
            attributes=attributes,
            vad_mode=cast(
                stt_pb2.VADMode.ValueType, _parse_vad_mode(payload.get("vad_mode"))
            ),
            vad_silence=_parse_float(payload.get("vad_silence"), 0.0),
            vad_threshold=_parse_float(payload.get("vad_threshold"), 0.0),
            require_token=_parse_bool(payload.get("require_token"), False),
            language_code=str(payload.get("language_code") or ""),
            task=cast(stt_pb2.Task.ValueType, _parse_task(payload.get("task"))),
            decode_profile=cast(
                stt_pb2.DecodeProfile.ValueType,
                _parse_decode_profile(payload.get("decode_profile")),
            ),
        )
        if "vad_threshold_override" in payload:
            request.vad_threshold_override = _parse_float(
                payload.get("vad_threshold_override"), 0.0
            )

        try:
            response = runtime.create_session_handler.handle(
                request, cast(grpc.ServicerContext, context)
            )
        except _WebSocketAbort as exc:
            await websocket.send_json(
                {
                    "type": "error",
                    "code": str(exc.details).split()[0],
                    "message": exc.details,
                }
            )
            await websocket.close(code=4401)
            return

        await websocket.send_json(
            {
                "type": "session",
                "session_id": session_id,
                "attributes": dict(response.attributes),
                "token": response.token,
                "token_required": response.token_required,
                "vad_mode": int(response.vad_mode),
                "vad_silence": response.vad_silence,
                "vad_threshold": response.vad_threshold,
                "language_code": response.language_code,
                "task": int(response.task),
                "decode_profile": int(response.decode_profile),
            }
        )

        audio_queue: queue.Queue[bytes | None] = queue.Queue()
        result_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        session_token = response.token if response.token_required else ""

        def audio_iter():
            while True:
                item = audio_queue.get()
                if item is None:
                    yield stt_pb2.AudioChunk(
                        session_id=session_id,
                        sample_rate=sample_rate,
                        is_final=True,
                        session_token=session_token,
                    )
                    return
                yield stt_pb2.AudioChunk(
                    session_id=session_id,
                    sample_rate=sample_rate,
                    pcm16=item,
                    is_final=False,
                    session_token=session_token,
                )

        def run_stream():
            try:
                for result in runtime.stream_orchestrator.run(
                    audio_iter(), cast(grpc.ServicerContext, context)
                ):
                    loop.call_soon_threadsafe(
                        result_queue.put_nowait, ("result", result)
                    )
                loop.call_soon_threadsafe(result_queue.put_nowait, ("done", None))
            except _WebSocketAbort as exc:
                loop.call_soon_threadsafe(result_queue.put_nowait, ("error", exc))
            except Exception as exc:  # pragma: no cover - defensive
                loop.call_soon_threadsafe(result_queue.put_nowait, ("error", exc))

        thread = threading.Thread(target=run_stream, daemon=True)
        thread.start()

        async def recv_audio() -> None:
            try:
                while True:
                    message = await websocket.receive()
                    msg_type = message.get("type")
                    if msg_type == "websocket.disconnect":
                        break
                    if "bytes" in message and message["bytes"]:
                        audio_queue.put(message["bytes"])
                        continue
                    if "text" in message and message["text"]:
                        try:
                            data = json.loads(message["text"])
                        except json.JSONDecodeError:
                            continue
                        if data.get("type") == "end":
                            break
            except WebSocketDisconnect:
                pass
            finally:
                context.close()
                audio_queue.put(None)

        async def send_results() -> None:
            while True:
                kind, payload = await result_queue.get()
                if kind == "result":
                    result = payload
                    try:
                        await websocket.send_json(
                            {
                                "type": "result",
                                "is_final": result.is_final,
                                "text": result.text,
                                "committed_text": result.committed_text,
                                "unstable_text": result.unstable_text,
                                "start_sec": result.start_sec,
                                "end_sec": result.end_sec,
                                "language_code": result.language_code,
                                "language": result.language,
                                "probability": result.probability,
                            }
                        )
                    except (WebSocketDisconnect, RuntimeError):
                        break
                    continue
                if kind == "error":
                    if isinstance(payload, _WebSocketAbort):
                        details = payload.details
                    else:
                        details = str(payload)
                    try:
                        await websocket.send_json({"type": "error", "message": details})
                    except (WebSocketDisconnect, RuntimeError):
                        pass
                    break
                if kind == "done":
                    trailing = (
                        dict(context.trailing_metadata)
                        if context.trailing_metadata
                        else None
                    )
                    try:
                        await websocket.send_json(
                            {"type": "done", "trailing": trailing}
                        )
                    except (WebSocketDisconnect, RuntimeError):
                        pass
                    break

        await asyncio.gather(recv_audio(), send_results())
        try:
            await websocket.close()
        except RuntimeError:
            pass

    return app


@dataclass
class WebSocketServerHandle:
    """Handle for the background WebSocket server."""

    server: uvicorn.Server
    thread: threading.Thread

    def stop(self, timeout: Optional[float] = None) -> None:
        if self.thread.is_alive():
            self.server.should_exit = True
            self.thread.join(timeout=timeout)


def start_ws_server(
    runtime: ApplicationRuntime,
    host: str,
    port: int,
    ws_rate_limit_rps: float | None = None,
    ws_rate_limit_burst: float | None = None,
    ws_trusted_proxies: Optional[List[str]] = None,
) -> WebSocketServerHandle:
    """Start WebSocket app for streaming UI in a background thread."""
    app = build_ws_app(
        runtime,
        ws_rate_limit_rps=ws_rate_limit_rps,
        ws_rate_limit_burst=ws_rate_limit_burst,
        ws_trusted_proxies=ws_trusted_proxies,
    )
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return WebSocketServerHandle(server=server, thread=thread)


__all__ = [
    "WebSocketServerHandle",
    "build_ws_app",
    "start_ws_server",
]
