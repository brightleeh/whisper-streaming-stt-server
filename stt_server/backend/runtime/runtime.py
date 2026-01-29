"""Application wiring for the STT server."""

from __future__ import annotations

from typing import Any, Dict, Optional

from stt_server.backend.application.model_registry import ModelRegistry
from stt_server.backend.application.session_manager import (
    CreateSessionHandler,
    SessionFacade,
    SessionInfo,
    SessionRegistry,
    SessionRegistryHooks,
)
from stt_server.backend.application.stream_orchestrator import (
    StreamOrchestrator,
    StreamOrchestratorConfig,
    StreamOrchestratorHooks,
)
from stt_server.backend.component.decode_scheduler import DecodeSchedulerHooks
from stt_server.backend.runtime.config import ServicerConfig
from stt_server.backend.runtime.metrics import Metrics
from stt_server.backend.utils.profile_resolver import normalize_decode_profiles
from stt_server.config.default.model import DEFAULT_MODEL_ID
from stt_server.config.languages import SupportedLanguages
from stt_server.utils.logger import LOGGER


class ApplicationRuntime:
    """Builds and owns application-layer dependencies."""

    def __init__(
        self,
        config: ServicerConfig,
    ) -> None:
        self.metrics = Metrics()
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

        self.model_registry = ModelRegistry()

        session_hooks = SessionRegistryHooks(
            on_create=self._on_session_created,
            on_remove=self._on_session_removed,
        )
        self.session_registry = SessionRegistry(session_hooks)
        self.session_facade = SessionFacade(self.session_registry)
        self.create_session_handler = CreateSessionHandler(
            session_registry=self.session_registry,
            model_registry=self.model_registry,
            decode_profiles=self.decode_profiles,
            default_decode_profile=self.default_decode_profile,
            default_language=self.default_language,
            language_fix=self.language_fix,
            default_task=self.default_task,
            supported_languages=self.supported_languages,
            default_vad_silence=streaming_config.vad_silence,
            default_vad_threshold=streaming_config.vad_threshold,
        )
        storage_config = self.config.storage
        orchestrator_config = StreamOrchestratorConfig(
            vad_threshold=streaming_config.vad_threshold,
            vad_silence=streaming_config.vad_silence,
            speech_rms_threshold=streaming_config.speech_rms_threshold,
            session_timeout_sec=streaming_config.session_timeout_sec,
            default_sample_rate=streaming_config.sample_rate,
            decode_timeout_sec=streaming_config.decode_timeout_sec,
            language_lookup=self.supported_languages,
            vad_model_pool_size=streaming_config.vad_model_pool_size,
            vad_model_prewarm=streaming_config.vad_model_prewarm,
            max_buffer_sec=streaming_config.max_buffer_sec,
            max_buffer_bytes=streaming_config.max_buffer_bytes,
            max_chunk_ms=streaming_config.max_chunk_ms,
            max_pending_decodes_per_stream=streaming_config.max_pending_decodes_per_stream,
            max_pending_decodes_global=streaming_config.max_pending_decodes_global,
            max_total_buffer_bytes=streaming_config.max_total_buffer_bytes,
            decode_queue_timeout_sec=streaming_config.decode_queue_timeout_sec,
            buffer_overlap_sec=streaming_config.buffer_overlap_sec,
            health_window_sec=streaming_config.health_window_sec,
            health_min_events=streaming_config.health_min_events,
            health_max_timeout_ratio=streaming_config.health_max_timeout_ratio,
            health_min_success_ratio=streaming_config.health_min_success_ratio,
            storage_enabled=storage_config.enabled,
            storage_directory=storage_config.directory,
            storage_queue_max_chunks=storage_config.queue_max_chunks,
            storage_max_bytes=storage_config.max_bytes,
            storage_max_files=storage_config.max_files,
            storage_max_age_days=storage_config.max_age_days,
        )
        decode_hooks = DecodeSchedulerHooks(
            on_error=self.metrics.record_error,
            on_decode_result=self.metrics.record_decode,
            on_vad_utterance_end=self.metrics.decrease_active_vad_utterances,
            on_decode_cancelled=self.metrics.record_decode_cancelled,
            on_decode_orphaned=self.metrics.record_decode_orphaned,
        )
        stream_hooks = StreamOrchestratorHooks(
            on_vad_trigger=self.metrics.record_vad_trigger,
            on_vad_utterance_start=self.metrics.increase_active_vad_utterances,
            active_vad_utterances=self.metrics.active_vad_utterances,
            decode_hooks=decode_hooks,
        )
        self.stream_orchestrator = StreamOrchestrator(
            session_facade=self.session_facade,
            model_registry=self.model_registry,
            config=orchestrator_config,
            hooks=stream_hooks,
        )
        self.decode_scheduler = self.stream_orchestrator.decode_scheduler

        default_model_config = {
            "name": model_config.model_size,
            "model_size": model_config.model_size,
            "device": model_config.device,
            "compute_type": model_config.compute_type,
            "pool_size": max(model_config.model_pool_size, 1),
            "language": self.default_language if self.language_fix else None,
            "log_metrics": model_config.log_metrics,
            "task": self.default_task,
        }
        self.stream_orchestrator.load_model(DEFAULT_MODEL_ID, default_model_config)

    def _build_language_cycle(self) -> list[Optional[str]]:
        if self.language_fix and self.default_language:
            return [self.default_language]
        return [None]

    def _on_session_created(self, info: SessionInfo) -> None:
        if info.api_key:
            self.metrics.increase_active_sessions(info.api_key)

    def _on_session_removed(self, info: SessionInfo) -> None:
        if info.api_key:
            self.metrics.decrease_active_sessions(info.api_key)

    def health_snapshot(self) -> Dict[str, Any]:
        metrics_snapshot = self.metrics.snapshot()
        registry_summary = self.model_registry.health_summary()
        return {
            "model_pool_healthy": self.decode_scheduler.workers_healthy(),
            "models_loaded": registry_summary["models_loaded"],
            "model_count": registry_summary["model_count"],
            "model_worker_total": registry_summary["total_workers"],
            "model_worker_shutdown": registry_summary["shutdown_workers"],
            "active_sessions": self.session_registry.active_count(),
            "decode_queue_depth": self.decode_scheduler.pending_decodes(),
            "decode_latency_avg": metrics_snapshot.get("decode_latency_avg"),
            "decode_latency_max": metrics_snapshot.get("decode_latency_max"),
        }

    def shutdown(self) -> None:
        self.model_registry.close()
