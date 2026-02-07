"""Application wiring for the STT server."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

from stt_server.backend.application.model_registry import ModelRegistry
from stt_server.backend.application.session_manager import (
    CreateSessionConfig,
    CreateSessionHandler,
    SessionFacade,
    SessionInfo,
    SessionRegistry,
    SessionRegistryHooks,
)
from stt_server.backend.application.stream_orchestrator import (
    StreamOrchestrator,
    StreamOrchestratorHooks,
)
from stt_server.backend.application.stream_orchestrator.types import (
    BufferLimits,
    DecodeQueueSettings,
    HealthSettings,
    PartialDecodeSettings,
    StorageSettings,
    StreamOrchestratorConfig,
    StreamSettings,
    VADPoolSettings,
)
from stt_server.backend.component.decode_scheduler import DecodeSchedulerHooks
from stt_server.backend.component.vad_gate import release_vad_slot
from stt_server.backend.runtime.config import ServicerConfig, StreamingRuntimeConfig
from stt_server.backend.runtime.metrics import Metrics
from stt_server.backend.utils.profile_resolver import normalize_decode_profiles
from stt_server.config.default.model import DEFAULT_MODEL_ID
from stt_server.config.languages import SupportedLanguages
from stt_server.utils.logger import LOGGER


class ApplicationRuntime:  # pylint: disable=too-many-instance-attributes
    """Builds and owns application-layer dependencies."""

    def __init__(
        self,
        config: ServicerConfig,
    ) -> None:
        self.metrics = Metrics()
        self.config = config
        self._accepting_sessions = True
        self._overload_until = 0.0
        self._overload_lock = threading.Lock()
        self._adaptive_throttle: AdaptiveThrottle | None = None
        model_config = self.config.model
        streaming_config = self.config.streaming
        self.metrics.set_expose_api_key_metrics(streaming_config.expose_api_key_metrics)
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

        self.model_registry = ModelRegistry(
            batch_window_ms=streaming_config.decode_batch_window_ms,
            max_batch_size=streaming_config.max_decode_batch_size,
        )

        session_hooks = SessionRegistryHooks(
            on_create=self._on_session_created,
            on_remove=self._on_session_removed,
        )
        self.session_registry = SessionRegistry(session_hooks)
        self.session_facade = SessionFacade(self.session_registry)
        session_config = CreateSessionConfig(
            decode_profiles=self.decode_profiles,
            default_decode_profile=self.default_decode_profile,
            default_language=self.default_language,
            language_fix=self.language_fix,
            default_task=self.default_task,
            supported_languages=self.supported_languages,
            default_vad_silence=streaming_config.vad_silence,
            default_vad_threshold=streaming_config.vad_threshold,
            require_api_key=self.config.model.require_api_key,
            create_session_auth_profile=self.config.model.create_session_auth_profile,
            create_session_auth_secret=self.config.model.create_session_auth_secret,
            create_session_auth_ttl_sec=self.config.model.create_session_auth_ttl_sec,
            create_session_rps=streaming_config.create_session_rps,
            create_session_burst=streaming_config.create_session_burst,
            max_sessions_per_ip=streaming_config.max_sessions_per_ip,
            max_sessions_per_api_key=streaming_config.max_sessions_per_api_key,
            allow_new_sessions=self._allow_new_sessions,
            allow_overload_sessions=self._allow_overload_sessions,
        )
        self.create_session_handler = CreateSessionHandler(
            session_registry=self.session_registry,
            model_registry=self.model_registry,
            config=session_config,
            metrics=self.metrics,
        )
        storage_config = self.config.storage
        stream_settings = StreamSettings(
            vad_threshold=streaming_config.vad_threshold,
            vad_silence=streaming_config.vad_silence,
            speech_rms_threshold=streaming_config.speech_rms_threshold,
            session_timeout_sec=streaming_config.session_timeout_sec,
            default_sample_rate=streaming_config.sample_rate,
            decode_timeout_sec=streaming_config.decode_timeout_sec,
            language_lookup=self.supported_languages,
            log_transcripts=streaming_config.log_transcripts,
            max_audio_seconds_per_session=streaming_config.max_audio_seconds_per_session,
            max_audio_bytes_per_sec=streaming_config.max_audio_bytes_per_sec,
            max_audio_bytes_per_sec_burst=streaming_config.max_audio_bytes_per_sec_burst,
            max_audio_bytes_per_sec_realtime=streaming_config.max_audio_bytes_per_sec_realtime,
            max_audio_bytes_per_sec_burst_realtime=streaming_config.max_audio_bytes_per_sec_burst_realtime,
            max_audio_bytes_per_sec_batch=streaming_config.max_audio_bytes_per_sec_batch,
            max_audio_bytes_per_sec_burst_batch=streaming_config.max_audio_bytes_per_sec_burst_batch,
            emit_final_on_vad=streaming_config.emit_final_on_vad,
        )
        storage_settings = StorageSettings(
            enabled=storage_config.enabled,
            directory=storage_config.directory,
            queue_max_chunks=storage_config.queue_max_chunks,
            max_bytes=storage_config.max_bytes,
            max_files=storage_config.max_files,
            max_age_days=storage_config.max_age_days,
        )
        vad_pool_settings = VADPoolSettings(
            size=streaming_config.vad_model_pool_size,
            prewarm=streaming_config.vad_model_prewarm,
            max_size=streaming_config.vad_model_pool_max_size,
            growth_factor=streaming_config.vad_model_pool_growth_factor,
        )
        buffer_limits = BufferLimits(
            max_buffer_sec=streaming_config.max_buffer_sec,
            max_buffer_bytes=streaming_config.max_buffer_bytes,
            max_chunk_ms=streaming_config.max_chunk_ms,
            max_total_buffer_bytes=streaming_config.max_total_buffer_bytes,
            buffer_overlap_sec=streaming_config.buffer_overlap_sec,
        )
        partial_decode = PartialDecodeSettings(
            interval_sec=streaming_config.partial_decode_interval_sec,
            window_sec=streaming_config.partial_decode_window_sec,
        )
        decode_queue = DecodeQueueSettings(
            max_pending_decodes_per_stream=streaming_config.max_pending_decodes_per_stream,
            max_pending_decodes_global=streaming_config.max_pending_decodes_global,
            decode_queue_timeout_sec=streaming_config.decode_queue_timeout_sec,
        )
        health = HealthSettings(
            window_sec=streaming_config.health_window_sec,
            min_events=streaming_config.health_min_events,
            max_timeout_ratio=streaming_config.health_max_timeout_ratio,
            min_success_ratio=streaming_config.health_min_success_ratio,
        )
        orchestrator_config = StreamOrchestratorConfig(
            stream=stream_settings,
            storage=storage_settings,
            vad_pool=vad_pool_settings,
            buffer_limits=buffer_limits,
            partial_decode=partial_decode,
            decode_queue=decode_queue,
            health=health,
        )
        decode_hooks = DecodeSchedulerHooks(
            on_error=self.metrics.record_error,
            on_decode_result=self.metrics.record_decode,
            on_vad_utterance_end=self.metrics.decrease_active_vad_utterances,
            on_decode_cancelled=self.metrics.record_decode_cancelled,
            on_decode_orphaned=self.metrics.record_decode_orphaned,
            on_decode_pending=self.metrics.set_decode_pending,
        )
        stream_hooks = StreamOrchestratorHooks(
            on_vad_trigger=self.metrics.record_vad_trigger,
            on_vad_utterance_start=self.metrics.increase_active_vad_utterances,
            active_vad_utterances=self.metrics.active_vad_utterances,
            on_buffer_total_bytes=self.metrics.set_buffer_total,
            on_stream_buffer_bytes=self.metrics.set_stream_buffer_bytes,
            on_stream_end=self.metrics.clear_stream_buffer,
            on_partial_drop=self.metrics.record_partial_drop,
            on_rate_limit_block=self.metrics.record_rate_limit_block,
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
            "backend": model_config.model_backend,
            "device": model_config.device,
            "compute_type": model_config.compute_type,
            "pool_size": max(model_config.model_pool_size, 1),
            "language": self.default_language if self.language_fix else None,
            "log_metrics": model_config.log_metrics,
            "task": self.default_task,
        }
        self.stream_orchestrator.load_model(DEFAULT_MODEL_ID, default_model_config)

        if streaming_config.adaptive_throttle_enabled:
            self._adaptive_throttle = AdaptiveThrottle(self, streaming_config)
            self._adaptive_throttle.start()

    def _build_language_cycle(self) -> list[Optional[str]]:
        if self.language_fix and self.default_language:
            return [self.default_language]
        return [None]

    def _on_session_created(self, info: SessionInfo) -> None:
        if info.api_key:
            self.metrics.increase_active_sessions(info.api_key)

    def _on_session_removed(self, info: SessionInfo) -> None:
        if info.vad_reserved:
            release_vad_slot()
            info.vad_reserved = False
        if info.api_key:
            self.metrics.decrease_active_sessions(info.api_key)

    def health_snapshot(self) -> Dict[str, Any]:
        """Return a point-in-time snapshot of runtime health metrics."""
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
        """Release runtime resources before exiting."""
        if self._adaptive_throttle is not None:
            self._adaptive_throttle.stop()
        self.model_registry.close()

    def stop_accepting_sessions(self) -> None:
        """Block new CreateSession requests."""
        self._accepting_sessions = False

    def _allow_new_sessions(self) -> bool:
        return self._accepting_sessions

    def _allow_overload_sessions(self) -> bool:
        with self._overload_lock:
            return time.monotonic() >= self._overload_until

    def _set_overload_until(self, deadline: float) -> None:
        with self._overload_lock:
            self._overload_until = max(self._overload_until, deadline)


class AdaptiveThrottle:
    """Adaptive throttling loop based on runtime pressure signals."""

    def __init__(self, runtime: ApplicationRuntime, config: StreamingRuntimeConfig):
        self._runtime = runtime
        self._config = config
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._base_partial_interval = config.partial_decode_interval_sec
        self._base_batch_window_ms = max(0, int(config.decode_batch_window_ms))
        self._pending_limit = max(0, int(config.max_pending_decodes_global))
        self._buffer_limit = (
            max(0, int(config.max_total_buffer_bytes))
            if config.max_total_buffer_bytes is not None
            else 0
        )
        self._last_orphaned = 0.0
        self._last_cancelled = 0.0
        self._mode = "normal"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        interval = max(0.5, float(self._config.adaptive_throttle_interval_sec))
        while not self._stop.wait(interval):
            self._tick()

    def _tick(self) -> None:
        metrics = self._runtime.metrics.render()
        pending = float(metrics.get("decode_pending", 0.0) or 0.0)
        buffer_total = float(metrics.get("buffer_bytes_total", 0.0) or 0.0)
        orphaned = float(metrics.get("decode_orphaned", 0.0) or 0.0)
        cancelled = float(metrics.get("decode_cancelled", 0.0) or 0.0)

        delta_orphaned = max(0.0, orphaned - self._last_orphaned)
        delta_cancelled = max(0.0, cancelled - self._last_cancelled)
        self._last_orphaned = orphaned
        self._last_cancelled = cancelled

        orphan_rate = 0.0
        denom = delta_orphaned + delta_cancelled
        if denom > 0:
            orphan_rate = delta_orphaned / denom

        pending_ratio = (
            pending / self._pending_limit if self._pending_limit > 0 else 0.0
        )
        buffer_ratio = (
            buffer_total / self._buffer_limit if self._buffer_limit > 0 else 0.0
        )

        pressure = (
            pending_ratio >= self._config.adaptive_pending_ratio_high
            or buffer_ratio >= self._config.adaptive_buffer_ratio_high
            or orphan_rate >= self._config.adaptive_orphan_rate_high
        )

        if pressure:
            self._apply_throttle()
        else:
            self._restore_defaults()

    def _apply_throttle(self) -> None:
        now = time.monotonic()
        self._runtime._set_overload_until(  # pylint: disable=protected-access
            now + max(0.0, float(self._config.adaptive_create_session_backoff_sec))
        )

        interval = self._scaled_partial_interval()
        self._runtime.stream_orchestrator.set_partial_interval_override(interval)

        min_window_ms = max(0, int(self._config.adaptive_batch_window_min_ms))
        window_ms = min(self._base_batch_window_ms, min_window_ms)
        self._runtime.model_registry.set_batch_window_ms(window_ms)

        if self._mode != "throttled":
            self._mode = "throttled"
            LOGGER.warning(
                "Adaptive throttling enabled: partial_interval=%s batch_window_ms=%s",
                interval,
                window_ms,
            )

    def _restore_defaults(self) -> None:
        self._runtime.stream_orchestrator.set_partial_interval_override(
            self._base_partial_interval
        )
        self._runtime.model_registry.set_batch_window_ms(self._base_batch_window_ms)
        if self._mode != "normal":
            self._mode = "normal"
            LOGGER.info("Adaptive throttling disabled; restored defaults.")

    def _scaled_partial_interval(self) -> Optional[float]:
        base = self._base_partial_interval
        if base is None or base <= 0:
            return base
        scaled = base * max(1.0, float(self._config.adaptive_partial_interval_scale))
        max_sec = self._config.adaptive_partial_interval_max_sec
        if max_sec is not None and max_sec > 0:
            return min(scaled, max_sec)
        return scaled
