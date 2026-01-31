"""Centralized error codes and status mappings."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final, NoReturn, Optional

import grpc


class ErrorCode(str, Enum):
    """Stable error identifiers surfaced to clients and logs."""

    # session (ERR100x)
    SESSION_ID_REQUIRED = "ERR1001"
    SESSION_ID_ALREADY_ACTIVE = "ERR1002"
    VAD_THRESHOLD_NEGATIVE = "ERR1003"
    SESSION_ID_MISSING = "ERR1004"
    SESSION_TOKEN_INVALID = "ERR1005"
    SESSION_TIMEOUT = "ERR1006"
    AUDIO_CHUNK_TOO_LARGE = "ERR1007"
    VAD_POOL_EXHAUSTED = "ERR1008"
    API_KEY_MISSING = "ERR1009"
    DECODE_OPTION_INVALID = "ERR1010"

    # decode (ERR200x)
    DECODE_TIMEOUT = "ERR2001"
    DECODE_TASK_FAILED = "ERR2002"

    # internal (ERR300x)
    CREATE_SESSION_UNEXPECTED = "ERR3001"
    STREAM_UNEXPECTED = "ERR3002"

    # admin/http (ERR400x)
    ADMIN_API_DISABLED = "ERR4001"
    MODEL_ALREADY_LOADED = "ERR4002"
    MODEL_UNLOAD_FAILED = "ERR4003"
    ADMIN_UNAUTHORIZED = "ERR4004"
    ADMIN_MODEL_PATH_FORBIDDEN = "ERR4005"


@dataclass(frozen=True)
class ErrorSpec:
    """Maps an error code to gRPC/HTTP statuses and message."""

    code: ErrorCode
    status: grpc.StatusCode
    http_status: int
    message: str


ERROR_SPECS: Final[dict[ErrorCode, ErrorSpec]] = {
    ErrorCode.SESSION_ID_REQUIRED: ErrorSpec(
        ErrorCode.SESSION_ID_REQUIRED,
        grpc.StatusCode.INVALID_ARGUMENT,
        400,
        "session_id is required",
    ),
    ErrorCode.SESSION_ID_ALREADY_ACTIVE: ErrorSpec(
        ErrorCode.SESSION_ID_ALREADY_ACTIVE,
        grpc.StatusCode.ALREADY_EXISTS,
        409,
        "session_id already active",
    ),
    ErrorCode.VAD_THRESHOLD_NEGATIVE: ErrorSpec(
        ErrorCode.VAD_THRESHOLD_NEGATIVE,
        grpc.StatusCode.INVALID_ARGUMENT,
        400,
        "vad_threshold must be non-negative",
    ),
    ErrorCode.SESSION_ID_MISSING: ErrorSpec(
        ErrorCode.SESSION_ID_MISSING,
        grpc.StatusCode.UNAUTHENTICATED,
        401,
        "Unknown or missing session_id",
    ),
    ErrorCode.SESSION_TOKEN_INVALID: ErrorSpec(
        ErrorCode.SESSION_TOKEN_INVALID,
        grpc.StatusCode.PERMISSION_DENIED,
        403,
        "Invalid session token",
    ),
    ErrorCode.SESSION_TIMEOUT: ErrorSpec(
        ErrorCode.SESSION_TIMEOUT,
        grpc.StatusCode.DEADLINE_EXCEEDED,
        504,
        "Session timeout due to inactivity",
    ),
    ErrorCode.AUDIO_CHUNK_TOO_LARGE: ErrorSpec(
        ErrorCode.AUDIO_CHUNK_TOO_LARGE,
        grpc.StatusCode.INVALID_ARGUMENT,
        400,
        "audio chunk exceeds maximum size",
    ),
    ErrorCode.VAD_POOL_EXHAUSTED: ErrorSpec(
        ErrorCode.VAD_POOL_EXHAUSTED,
        grpc.StatusCode.RESOURCE_EXHAUSTED,
        503,
        "VAD capacity exhausted",
    ),
    ErrorCode.API_KEY_MISSING: ErrorSpec(
        ErrorCode.API_KEY_MISSING,
        grpc.StatusCode.UNAUTHENTICATED,
        401,
        "API key is required",
    ),
    ErrorCode.DECODE_OPTION_INVALID: ErrorSpec(
        ErrorCode.DECODE_OPTION_INVALID,
        grpc.StatusCode.INVALID_ARGUMENT,
        400,
        "invalid decode option",
    ),
    ErrorCode.DECODE_TIMEOUT: ErrorSpec(
        ErrorCode.DECODE_TIMEOUT,
        grpc.StatusCode.INTERNAL,
        500,
        "decode timeout waiting for pending tasks",
    ),
    ErrorCode.DECODE_TASK_FAILED: ErrorSpec(
        ErrorCode.DECODE_TASK_FAILED,
        grpc.StatusCode.INTERNAL,
        500,
        "decode task failed",
    ),
    ErrorCode.CREATE_SESSION_UNEXPECTED: ErrorSpec(
        ErrorCode.CREATE_SESSION_UNEXPECTED,
        grpc.StatusCode.UNKNOWN,
        500,
        "Unexpected CreateSession error",
    ),
    ErrorCode.STREAM_UNEXPECTED: ErrorSpec(
        ErrorCode.STREAM_UNEXPECTED,
        grpc.StatusCode.UNKNOWN,
        500,
        "Unexpected streaming error",
    ),
    ErrorCode.ADMIN_API_DISABLED: ErrorSpec(
        ErrorCode.ADMIN_API_DISABLED,
        grpc.StatusCode.UNIMPLEMENTED,
        501,
        "Admin API not enabled",
    ),
    ErrorCode.MODEL_ALREADY_LOADED: ErrorSpec(
        ErrorCode.MODEL_ALREADY_LOADED,
        grpc.StatusCode.ALREADY_EXISTS,
        409,
        "Model is already loaded",
    ),
    ErrorCode.MODEL_UNLOAD_FAILED: ErrorSpec(
        ErrorCode.MODEL_UNLOAD_FAILED,
        grpc.StatusCode.FAILED_PRECONDITION,
        400,
        "Model not found or is default",
    ),
    ErrorCode.ADMIN_UNAUTHORIZED: ErrorSpec(
        ErrorCode.ADMIN_UNAUTHORIZED,
        grpc.StatusCode.UNAUTHENTICATED,
        401,
        "Invalid or missing admin token",
    ),
    ErrorCode.ADMIN_MODEL_PATH_FORBIDDEN: ErrorSpec(
        ErrorCode.ADMIN_MODEL_PATH_FORBIDDEN,
        grpc.StatusCode.PERMISSION_DENIED,
        403,
        "model_path is not allowed",
    ),
}

ERROR_STATUS_MAP: Final[dict[ErrorCode, grpc.StatusCode]] = {
    code: spec.status for code, spec in ERROR_SPECS.items()
}

ERROR_HTTP_STATUS_MAP: Final[dict[ErrorCode, int]] = {
    code: spec.http_status for code, spec in ERROR_SPECS.items()
}


def spec_for(code: ErrorCode) -> ErrorSpec:
    """Return the ErrorSpec for a given error code."""
    return ERROR_SPECS[code]


def status_for(code: ErrorCode) -> grpc.StatusCode:
    """Return the gRPC status associated with an error code."""
    return ERROR_SPECS[code].status


def http_status_for(code: ErrorCode) -> int:
    """Return the HTTP status associated with an error code."""
    return ERROR_SPECS[code].http_status


def format_error(code: ErrorCode, detail: Optional[str] = None) -> str:
    """Format an error code and optional detail into a message."""
    spec = ERROR_SPECS[code]
    message = detail if detail else spec.message
    return f"{spec.code.value} {message}"


def http_payload_for(code: ErrorCode, detail: Optional[str] = None) -> dict[str, str]:
    """Build an HTTP error payload for a given error code."""
    spec = ERROR_SPECS[code]
    message = detail if detail else spec.message
    return {"code": spec.code.value, "message": message}


class STTError(RuntimeError):
    """Raised for application-defined errors with status metadata."""

    def __init__(self, code: ErrorCode, detail: Optional[str] = None) -> None:
        """Create an STTError with formatted message and status metadata."""
        self.code = code
        self.status = status_for(code)
        self.http_status = http_status_for(code)
        self.detail = detail or ERROR_SPECS[code].message
        super().__init__(format_error(code, detail))


def abort_with_error(
    context: grpc.ServicerContext,
    code: ErrorCode,
    detail: Optional[str] = None,
) -> NoReturn:
    """Abort a gRPC context with an error code and optional detail."""
    context.abort(status_for(code), format_error(code, detail))
    raise RuntimeError("unreachable")


__all__ = [
    "ErrorCode",
    "ErrorSpec",
    "ERROR_SPECS",
    "ERROR_STATUS_MAP",
    "ERROR_HTTP_STATUS_MAP",
    "STTError",
    "abort_with_error",
    "format_error",
    "http_payload_for",
    "http_status_for",
    "spec_for",
    "status_for",
]
