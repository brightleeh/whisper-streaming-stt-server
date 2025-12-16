"""gRPC STT servicer built on top of helper modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import grpc

from gen.stt.python.v1 import stt_pb2, stt_pb2_grpc
from stt_server.backend.core.decode_scheduler import DecodeScheduler
from stt_server.backend.core.metrics import Metrics
from stt_server.backend.core.profile_resolver import normalize_decode_profiles
from stt_server.backend.core.session_manager import SessionManager
from stt_server.backend.service.components import (
    AudioStorageConfig,
    AudioStorageManager,
    CreateSessionHandler,
    SessionFacade,
    StreamingRunner,
    StreamingRunnerConfig,
)
from stt_server.config.languages import SupportedLanguages
from stt_server.utils.logger import LOGGER


@dataclass
class ModelRuntimeConfig:
    model_size: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    log_metrics: bool = False
    model_pool_size: int = 1
    language: str = ""
    language_fix: bool = False
    task: str = "transcribe"
    decode_profiles: Optional[Dict[str, Dict[str, Any]]] = None
    default_decode_profile: str = "realtime"


@dataclass
class StreamingRuntimeConfig:
    epd_silence: float = 0.8
    epd_threshold: float = 0.01
    speech_rms_threshold: float = 0.02
    session_timeout_sec: float = 60.0
    sample_rate: int = 16000
    decode_timeout_sec: float = 30.0


@dataclass
class StorageRuntimeConfig:
    enabled: bool = False
    directory: str = "data/audio"
    max_bytes: Optional[int] = None
    max_files: Optional[int] = None
    max_age_days: Optional[int] = None


@dataclass
class ServicerConfig:
    model: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    streaming: StreamingRuntimeConfig = field(default_factory=StreamingRuntimeConfig)
    storage: StorageRuntimeConfig = field(default_factory=StorageRuntimeConfig)


class STTBackendServicer(stt_pb2_grpc.STTBackendServicer):
    """Implements the gRPC STT streaming service."""

    def __init__(
        self,
        config: ServicerConfig,
        metrics: Optional[Metrics] = None,
    ) -> None:
        self.metrics = metrics or Metrics()
        self.config = config
        model_config = self.config.model
        streaming_config = self.config.streaming
        self.default_language = (
            model_config.language.strip().lower() if model_config.language else ""
        )
        self.language_fix = model_config.language_fix
        self.default_task = (model_config.task or "transcribe").lower()
        self.supported_languages = SupportedLanguages()

        self.decode_profiles = normalize_decode_profiles(model_config.decode_profiles)
        default_profile = model_config.default_decode_profile
        if default_profile not in self.decode_profiles:
            LOGGER.warning(
                "Unknown default decode profile '%s'; using 'realtime'",
                default_profile,
            )
            default_profile = "realtime"
        self.default_decode_profile = default_profile

        language_hint_cycle = self._build_language_cycle()
        pool_size = max(model_config.model_pool_size, 1)
        self.decode_scheduler = DecodeScheduler(
            model_size=model_config.model_size,
            device=model_config.device,
            compute_type=model_config.compute_type,
            language_cycle=language_hint_cycle,
            pool_size=pool_size,
            task=self.default_task,
            log_metrics=model_config.log_metrics,
            decode_timeout_sec=streaming_config.decode_timeout_sec,
            metrics=self.metrics,
            language_lookup=self.supported_languages,
        )

        self.session_manager = SessionManager(self.metrics)
        self.session_facade = SessionFacade(self.session_manager)
        self.create_session_handler = CreateSessionHandler(
            session_manager=self.session_manager,
            decode_profiles=self.decode_profiles,
            default_decode_profile=self.default_decode_profile,
            default_language=self.default_language,
            language_fix=self.language_fix,
            default_task=self.default_task,
            supported_languages=self.supported_languages,
            default_epd_silence=streaming_config.epd_silence,
            default_epd_threshold=streaming_config.epd_threshold,
        )
        runner_config = StreamingRunnerConfig(
            epd_threshold=streaming_config.epd_threshold,
            epd_silence=streaming_config.epd_silence,
            speech_rms_threshold=streaming_config.speech_rms_threshold,
            session_timeout_sec=streaming_config.session_timeout_sec,
            default_sample_rate=streaming_config.sample_rate,
        )
        storage_config = self.config.storage
        self.audio_storage_manager: Optional[AudioStorageManager] = None
        if storage_config.enabled:
            storage_directory = Path(storage_config.directory).expanduser()
            storage_policy = AudioStorageConfig(
                enabled=True,
                directory=storage_directory,
                max_bytes=storage_config.max_bytes,
                max_files=storage_config.max_files,
                max_age_days=storage_config.max_age_days,
            )
            self.audio_storage_manager = AudioStorageManager(storage_policy)
            LOGGER.info(
                "Audio storage enabled directory=%s max_bytes=%s max_files=%s max_age_days=%s",
                storage_directory,
                storage_config.max_bytes,
                storage_config.max_files,
                storage_config.max_age_days,
            )
        self.streaming_runner = StreamingRunner(
            session_facade=self.session_facade,
            decode_scheduler=self.decode_scheduler,
            metrics=self.metrics,
            config=runner_config,
            audio_storage=self.audio_storage_manager,
        )

    # ------------------------------------------------------------------
    # gRPC methods
    # ------------------------------------------------------------------
    def CreateSession(  # type: ignore[override]
        self, request: stt_pb2.SessionRequest, context: grpc.ServicerContext
    ) -> stt_pb2.SessionResponse:
        try:
            return self.create_session_handler.handle(request, context)
        except grpc.RpcError as exc:
            self.metrics.record_error(exc.code())
            raise
        except Exception:
            self.metrics.record_error(grpc.StatusCode.UNKNOWN)
            LOGGER.exception("Unexpected CreateSession error")
            raise

    def StreamingRecognize(  # type: ignore[override]
        self,
        request_iterator: Iterable[stt_pb2.AudioChunk],
        context: grpc.ServicerContext,
    ) -> Iterable[stt_pb2.STTResult]:
        try:
            yield from self.streaming_runner.run(request_iterator, context)
        except grpc.RpcError as exc:
            self.metrics.record_error(exc.code())
            raise
        except Exception:
            self.metrics.record_error(grpc.StatusCode.UNKNOWN)
            LOGGER.exception("Unexpected streaming error")
            raise

    # ------------------------------------------------------------------
    def health_snapshot(self) -> Dict[str, Any]:
        metrics_snapshot = self.metrics.snapshot()
        return {
            "model_pool_healthy": self.decode_scheduler.workers_healthy(),
            "models_loaded": True,
            "active_sessions": self.session_manager.active_count(),
            "decode_queue_depth": self.decode_scheduler.pending_decodes(),
            "decode_latency_avg": metrics_snapshot.get("decode_latency_avg"),
            "decode_latency_max": metrics_snapshot.get("decode_latency_max"),
        }

    def _build_language_cycle(self) -> List[Optional[str]]:
        if self.language_fix and self.default_language:
            return [self.default_language]
        return [None]
