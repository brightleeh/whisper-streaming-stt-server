"""Minimal streaming client SDK for STT backend."""

from __future__ import annotations

import hashlib
import hmac
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Sequence

import grpc

from gen.stt.python.v1 import stt_pb2, stt_pb2_grpc

_ERROR_CODE_RE = re.compile(r"(ERR\d{4})")


@dataclass(frozen=True)
class RetryConfig:
    """Retry policy for client calls."""

    attempts: int = 3
    base_backoff_sec: float = 0.5
    max_backoff_sec: float = 5.0
    retryable_status: Sequence[grpc.StatusCode] = (
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.RESOURCE_EXHAUSTED,
        grpc.StatusCode.DEADLINE_EXCEEDED,
    )


def parse_error_code(exc: grpc.RpcError) -> Optional[str]:
    """Extract ERR#### code from a gRPC error, if present."""
    details = ""
    try:
        details = exc.details() or ""
    except Exception:
        details = ""
    match = _ERROR_CODE_RE.search(details)
    return match.group(1) if match else None


def _should_retry(exc: grpc.RpcError, retry: RetryConfig, attempt: int) -> bool:
    if attempt >= max(0, retry.attempts):
        return False
    status = exc.code() if hasattr(exc, "code") else None
    if status in retry.retryable_status:
        return True
    return False


def _backoff_delay(retry: RetryConfig, attempt: int) -> float:
    base = max(0.0, retry.base_backoff_sec)
    delay = min(retry.max_backoff_sec, base * (2**attempt))
    jitter = delay * 0.2
    return max(0.0, delay + random.uniform(-jitter, jitter))


def build_signed_token_metadata(
    session_id: str, signed_token_secret: Optional[str]
) -> list[tuple[str, str]]:
    """Build signed token metadata for CreateSession."""
    secret = (signed_token_secret or "").strip()
    if not secret:
        return []
    timestamp = str(int(time.time()))
    payload = f"{session_id}:{timestamp}".encode("utf-8")
    signature = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return [("authorization", f"Bearer {signature}"), ("x-stt-auth-ts", timestamp)]


def _create_channel(
    target: str,
    grpc_max_receive_message_bytes: Optional[int],
    grpc_max_send_message_bytes: Optional[int],
    tls_enabled: bool,
    tls_ca_file: Optional[str],
    keepalive_time_ms: int,
    keepalive_timeout_ms: int,
    keepalive_permit_without_calls: bool,
    keepalive_max_pings_without_data: int,
    keepalive_min_time_between_pings_ms: int,
) -> grpc.Channel:
    options = [
        ("grpc.keepalive_time_ms", keepalive_time_ms),
        ("grpc.keepalive_timeout_ms", keepalive_timeout_ms),
        ("grpc.keepalive_permit_without_calls", int(keepalive_permit_without_calls)),
        ("grpc.http2.max_pings_without_data", keepalive_max_pings_without_data),
        ("grpc.http2.min_time_between_pings_ms", keepalive_min_time_between_pings_ms),
    ]
    if grpc_max_receive_message_bytes and grpc_max_receive_message_bytes > 0:
        options.append(
            ("grpc.max_receive_message_length", grpc_max_receive_message_bytes)
        )
    if grpc_max_send_message_bytes and grpc_max_send_message_bytes > 0:
        options.append(("grpc.max_send_message_length", grpc_max_send_message_bytes))

    root_certificates = None
    if tls_ca_file:
        tls_enabled = True
        cert_path = Path(tls_ca_file).expanduser()
        if not cert_path.exists():
            raise FileNotFoundError(f"TLS CA file not found: {cert_path}")
        root_certificates = cert_path.read_bytes()

    if tls_enabled:
        credentials = grpc.ssl_channel_credentials(root_certificates=root_certificates)
        return grpc.secure_channel(target, credentials, options=options)
    return grpc.insecure_channel(target, options=options)


class StreamingClient:
    """Small streaming client with retry/keepalive support."""

    def __init__(
        self,
        target: str,
        *,
        tls_enabled: bool = False,
        tls_ca_file: Optional[str] = None,
        grpc_max_receive_message_bytes: Optional[int] = None,
        grpc_max_send_message_bytes: Optional[int] = None,
        keepalive_time_ms: int = 30000,
        keepalive_timeout_ms: int = 10000,
        keepalive_permit_without_calls: bool = True,
        keepalive_max_pings_without_data: int = 0,
        keepalive_min_time_between_pings_ms: int = 10000,
        signed_token_secret: Optional[str] = None,
    ) -> None:
        self._channel = _create_channel(
            target,
            grpc_max_receive_message_bytes,
            grpc_max_send_message_bytes,
            tls_enabled,
            tls_ca_file,
            keepalive_time_ms,
            keepalive_timeout_ms,
            keepalive_permit_without_calls,
            keepalive_max_pings_without_data,
            keepalive_min_time_between_pings_ms,
        )
        self._stub = stt_pb2_grpc.STTBackendStub(self._channel)
        self._signed_token_secret = signed_token_secret

    def close(self) -> None:
        self._channel.close()

    def build_signed_metadata(
        self, session_id: str, signed_token_secret: Optional[str] = None
    ) -> list[tuple[str, str]]:
        return build_signed_token_metadata(
            session_id, signed_token_secret or self._signed_token_secret
        )

    def create_session(
        self,
        request: stt_pb2.SessionRequest,
        *,
        metadata: Optional[Iterable[tuple[str, str]]] = None,
        retry: Optional[RetryConfig] = None,
    ) -> stt_pb2.SessionResponse:
        retry = retry or RetryConfig(attempts=0)
        attempt = 0
        while True:
            try:
                return self._stub.CreateSession(request, metadata=metadata)
            except grpc.RpcError as exc:
                if not _should_retry(exc, retry, attempt):
                    raise
                time.sleep(_backoff_delay(retry, attempt))
                attempt += 1

    def streaming_recognize(
        self,
        audio_iter: Iterable[stt_pb2.AudioChunk],
        *,
        metadata: Optional[Iterable[tuple[str, str]]] = None,
        timeout: Optional[float] = None,
    ) -> Iterable[stt_pb2.STTResult]:
        return self._stub.StreamingRecognize(
            audio_iter, metadata=metadata, timeout=timeout
        )

    def streaming_recognize_with_retry(
        self,
        audio_iter_factory: Callable[[], Iterable[stt_pb2.AudioChunk]],
        *,
        metadata: Optional[Iterable[tuple[str, str]]] = None,
        timeout: Optional[float] = None,
        retry: Optional[RetryConfig] = None,
    ) -> Iterator[stt_pb2.STTResult]:
        retry = retry or RetryConfig(attempts=0)
        attempt = 0
        while True:
            got_result = False
            try:
                for result in self._stub.StreamingRecognize(
                    audio_iter_factory(), metadata=metadata, timeout=timeout
                ):
                    got_result = True
                    yield result
                return
            except grpc.RpcError as exc:
                if got_result or not _should_retry(exc, retry, attempt):
                    raise
                time.sleep(_backoff_delay(retry, attempt))
                attempt += 1
