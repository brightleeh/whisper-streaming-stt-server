"""gRPC STT servicer built on top of helper modules."""

from __future__ import annotations

from typing import Iterable

import grpc

from gen.stt.python.v1 import stt_pb2, stt_pb2_grpc
from stt_server.backend.runtime import ApplicationRuntime, ServicerConfig
from stt_server.errors import ErrorCode, STTError, format_error, status_for
from stt_server.utils.logger import LOGGER


class STTGrpcServicer(stt_pb2_grpc.STTBackendServicer):
    """Implements the gRPC STT streaming service."""

    def __init__(
        self,
        config: ServicerConfig,
    ) -> None:
        self.runtime = ApplicationRuntime(config)

    # ------------------------------------------------------------------
    # gRPC methods
    # ------------------------------------------------------------------
    def CreateSession(  # type: ignore[override]
        self, request: stt_pb2.SessionRequest, context: grpc.ServicerContext
    ) -> stt_pb2.SessionResponse:
        try:
            return self.runtime.create_session_handler.handle(request, context)
        except grpc.RpcError as exc:
            self._record_error(exc.code())
            raise
        except STTError as exc:
            self._record_error(exc.status)
            LOGGER.error(str(exc))
            context.abort(exc.status, str(exc))
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
            self._record_error(status_for(ErrorCode.CREATE_SESSION_UNEXPECTED))
            LOGGER.exception(format_error(ErrorCode.CREATE_SESSION_UNEXPECTED))
            raise

    def StreamingRecognize(  # type: ignore[override]
        self,
        request_iterator: Iterable[stt_pb2.AudioChunk],
        context: grpc.ServicerContext,
    ) -> Iterable[stt_pb2.STTResult]:
        try:
            yield from self.runtime.stream_orchestrator.run(request_iterator, context)
        except grpc.RpcError as exc:
            self._record_error(exc.code())
            raise
        except STTError as exc:
            self._record_error(exc.status)
            LOGGER.error(str(exc))
            context.abort(exc.status, str(exc))
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
            self._record_error(status_for(ErrorCode.STREAM_UNEXPECTED))
            LOGGER.exception(format_error(ErrorCode.STREAM_UNEXPECTED))
            raise

    def _record_error(self, status_code: grpc.StatusCode) -> None:
        self.runtime.metrics.record_error(status_code)
