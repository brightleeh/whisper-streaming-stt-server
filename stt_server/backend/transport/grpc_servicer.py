"""gRPC STT servicer built on top of helper modules."""

from __future__ import annotations

from typing import Any, Callable, Iterable

import grpc

from gen.stt.python.v1 import stt_pb2, stt_pb2_grpc
from stt_server.backend.runtime import ApplicationRuntime, ServicerConfig
from stt_server.utils.logger import LOGGER


class STTGrpcServicer(stt_pb2_grpc.STTBackendServicer):
    """Implements the gRPC STT streaming service."""

    def __init__(
        self,
        config: ServicerConfig,
        metrics: Any = None,
        error_recorder: Callable[[grpc.StatusCode], None] | None = None,
    ) -> None:
        self.runtime = ApplicationRuntime(config, metrics)
        self._error_recorder = error_recorder or self.runtime.metrics.record_error
        self.create_session_handler = self.runtime.create_session_handler
        self.stream_orchestrator = self.runtime.stream_orchestrator
        self.decode_scheduler = self.runtime.decode_scheduler
        self.session_registry = self.runtime.session_registry

    # ------------------------------------------------------------------
    # gRPC methods
    # ------------------------------------------------------------------
    def CreateSession(  # type: ignore[override]
        self, request: stt_pb2.SessionRequest, context: grpc.ServicerContext
    ) -> stt_pb2.SessionResponse:
        try:
            return self.create_session_handler.handle(request, context)
        except grpc.RpcError as exc:
            self._record_error(exc.code())
            raise
        except Exception:
            self._record_error(grpc.StatusCode.UNKNOWN)
            LOGGER.exception("ERR3001 Unexpected CreateSession error")
            raise

    def StreamingRecognize(  # type: ignore[override]
        self,
        request_iterator: Iterable[stt_pb2.AudioChunk],
        context: grpc.ServicerContext,
    ) -> Iterable[stt_pb2.STTResult]:
        try:
            yield from self.stream_orchestrator.run(request_iterator, context)
        except grpc.RpcError as exc:
            self._record_error(exc.code())
            raise
        except Exception:
            self._record_error(grpc.StatusCode.UNKNOWN)
            LOGGER.exception("ERR3002 Unexpected streaming error")
            raise

    def _record_error(self, status_code: grpc.StatusCode) -> None:
        self._error_recorder(status_code)
