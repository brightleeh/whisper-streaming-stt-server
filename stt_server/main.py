"""CLI entrypoint for the streaming STT server."""

import argparse
import os
import signal
import threading
from concurrent import futures
from pathlib import Path

import grpc

from gen.stt.python.v1 import stt_pb2_grpc
from stt_server.backend.runtime.config import (
    ModelRuntimeConfig,
    ServicerConfig,
    StorageRuntimeConfig,
    StreamingRuntimeConfig,
)
from stt_server.backend.transport import STTGrpcServicer, start_http_server
from stt_server.config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MODEL_CONFIG_PATH,
    ServerConfig,
    load_config,
)
from stt_server.utils.logger import LOGGER, configure_logging


def serve(config: ServerConfig) -> None:
    """Launch gRPC + HTTP observability servers."""
    server_state = {"grpc_running": False}
    stop_event = threading.Event()
    shutdown_once = threading.Event()
    shutdown_done = threading.Event()
    force_exit_scheduled = threading.Event()
    grpc_workers = config.grpc_worker_threads
    if grpc_workers <= 0:
        grpc_workers = max(4, config.max_sessions + 4)
    grpc_executor = futures.ThreadPoolExecutor(max_workers=grpc_workers)
    grpc_options = []
    if (
        config.grpc_max_receive_message_bytes
        and config.grpc_max_receive_message_bytes > 0
    ):
        grpc_options.append(
            ("grpc.max_receive_message_length", config.grpc_max_receive_message_bytes)
        )
    if config.grpc_max_send_message_bytes and config.grpc_max_send_message_bytes > 0:
        grpc_options.append(
            ("grpc.max_send_message_length", config.grpc_max_send_message_bytes)
        )
    if grpc_options:
        server = grpc.server(grpc_executor, options=grpc_options)
    else:
        server = grpc.server(grpc_executor)
    model_cfg = ModelRuntimeConfig(
        model_size=config.model,
        model_backend=config.model_backend,
        device=config.device,
        compute_type=config.compute_type,
        log_metrics=config.log_metrics,
        task=config.task,
        language=config.language,
        language_fix=config.language_fix,
        model_pool_size=config.model_pool_size,
        decode_profiles=config.decode_profiles,
        default_decode_profile=config.default_decode_profile,
        model_load_profiles=config.model_load_profiles,
        default_model_load_profile=config.default_model_load_profile,
        require_api_key=config.require_api_key,
    )
    vad_pool_size = config.vad_model_pool_size
    if vad_pool_size <= 0:
        vad_pool_size = config.max_sessions
    vad_prewarm = max(0, config.vad_model_prewarm)
    streaming_cfg = StreamingRuntimeConfig(
        vad_silence=config.vad_silence,
        vad_threshold=config.vad_threshold,
        vad_model_pool_size=vad_pool_size,
        vad_model_prewarm=vad_prewarm,
        vad_model_pool_max_size=config.max_sessions,
        vad_model_pool_growth_factor=config.vad_model_pool_growth_factor,
        session_timeout_sec=config.session_timeout_sec,
        sample_rate=config.sample_rate,
        decode_timeout_sec=config.decode_timeout_sec,
        max_buffer_sec=config.max_buffer_sec,
        max_buffer_bytes=config.max_buffer_bytes,
        max_chunk_ms=config.max_chunk_ms,
        partial_decode_interval_sec=config.partial_decode_interval_sec,
        partial_decode_window_sec=config.partial_decode_window_sec,
        expose_api_key_metrics=config.expose_api_key_metrics,
        max_pending_decodes_per_stream=config.max_pending_decodes_per_stream,
        max_pending_decodes_global=config.max_pending_decodes_global,
        max_total_buffer_bytes=config.max_total_buffer_bytes,
        decode_queue_timeout_sec=config.decode_queue_timeout_sec,
        decode_batch_window_ms=config.decode_batch_window_ms,
        max_decode_batch_size=config.max_decode_batch_size,
        buffer_overlap_sec=config.buffer_overlap_sec,
        create_session_rps=config.create_session_rps,
        create_session_burst=config.create_session_burst,
        max_sessions_per_ip=config.max_sessions_per_ip,
        max_sessions_per_api_key=config.max_sessions_per_api_key,
        max_audio_seconds_per_session=config.max_audio_seconds_per_session,
        max_audio_bytes_per_sec=config.max_audio_bytes_per_sec,
        max_audio_bytes_per_sec_burst=config.max_audio_bytes_per_sec_burst,
        max_audio_bytes_per_sec_realtime=config.max_audio_bytes_per_sec_realtime,
        max_audio_bytes_per_sec_burst_realtime=config.max_audio_bytes_per_sec_burst_realtime,
        max_audio_bytes_per_sec_batch=config.max_audio_bytes_per_sec_batch,
        max_audio_bytes_per_sec_burst_batch=config.max_audio_bytes_per_sec_burst_batch,
        health_window_sec=config.health_window_sec,
        health_min_events=config.health_min_events,
        health_max_timeout_ratio=config.health_max_timeout_ratio,
        health_min_success_ratio=config.health_min_success_ratio,
        log_transcripts=config.log_transcripts,
    )
    streaming_cfg.speech_rms_threshold = config.speech_rms_threshold
    storage_cfg = StorageRuntimeConfig(
        enabled=config.persist_audio,
        directory=config.audio_storage_dir,
        queue_max_chunks=config.audio_storage_queue_max_chunks,
        max_bytes=config.audio_storage_max_bytes,
        max_files=config.audio_storage_max_files,
        max_age_days=config.audio_storage_max_age_days,
    )
    servicer_config = ServicerConfig(model_cfg, streaming_cfg, storage_cfg)
    servicer = STTGrpcServicer(servicer_config)
    stt_pb2_grpc.add_STTBackendServicer_to_server(servicer, server)  # type: ignore[name-defined]

    def _bind_grpc_port(bind_address: str, secure: bool) -> None:
        if secure:
            server.add_secure_port(bind_address, credentials)
        else:
            server.add_insecure_port(bind_address)

    if config.tls_required and (not config.tls_cert_file or not config.tls_key_file):
        raise ValueError("TLS is required but tls_cert_file/tls_key_file not set.")
    if config.tls_cert_file or config.tls_key_file:
        if not config.tls_cert_file or not config.tls_key_file:
            raise ValueError(
                "Both tls_cert_file and tls_key_file must be set to enable TLS."
            )
        cert_path = Path(config.tls_cert_file).expanduser()
        key_path = Path(config.tls_key_file).expanduser()
        if not cert_path.exists():
            raise FileNotFoundError(f"TLS cert file not found: {cert_path}")
        if not key_path.exists():
            raise FileNotFoundError(f"TLS key file not found: {key_path}")
        cert_chain = cert_path.read_bytes()
        private_key = key_path.read_bytes()
        credentials = grpc.ssl_server_credentials([(private_key, cert_chain)])
        bind_addr = f"[::]:{config.port}"
        fallback_addr = f"0.0.0.0:{config.port}"
        try:
            _bind_grpc_port(bind_addr, secure=True)
        except RuntimeError as exc:
            LOGGER.warning(
                "Failed to bind gRPC on %s (%s); falling back to %s",
                bind_addr,
                exc,
                fallback_addr,
            )
            _bind_grpc_port(fallback_addr, secure=True)
        LOGGER.info("gRPC TLS enabled cert=%s key=%s", cert_path, key_path)
    else:
        LOGGER.warning(
            "gRPC is running without TLS. Set tls.cert_file/tls.key_file or "
            "--tls-cert-file/--tls-key-file to enable TLS."
        )
        bind_addr = f"[::]:{config.port}"
        fallback_addr = f"0.0.0.0:{config.port}"
        try:
            _bind_grpc_port(bind_addr, secure=False)
        except RuntimeError as exc:
            LOGGER.warning(
                "Failed to bind gRPC on %s (%s); falling back to %s",
                bind_addr,
                exc,
                fallback_addr,
            )
            _bind_grpc_port(fallback_addr, secure=False)
    http_handle = start_http_server(
        runtime=servicer.runtime,
        server_state=server_state,
        host=config.http_host,
        port=config.metrics_port,
        http_rate_limit_rps=config.http_rate_limit_rps,
        http_rate_limit_burst=config.http_rate_limit_burst,
        http_trusted_proxies=config.http_trusted_proxies,
    )
    LOGGER.info(
        "STT Server started on port %s (model=%s, device=%s)",
        config.port,
        config.model,
        config.device,
    )

    def shutdown() -> None:
        if shutdown_once.is_set():
            return
        shutdown_once.set()
        server_state["grpc_running"] = False
        servicer.runtime.stop_accepting_sessions()
        grace = config.decode_timeout_sec if config.decode_timeout_sec > 0 else 5.0
        LOGGER.info("Graceful shutdown started (grace=%.2fs)", grace)
        try:
            server.stop(grace).wait()
        finally:
            http_handle.stop(timeout=grace + 1)
            servicer.runtime.shutdown()
            grpc_executor.shutdown(wait=False)
            shutdown_done.set()

    def _force_exit_after(delay: float) -> None:
        if shutdown_done.wait(timeout=delay):
            return
        LOGGER.error("Graceful shutdown timed out; forcing exit")
        os._exit(1)

    def _handle_signal(signum: int, _frame) -> None:  # type: ignore[no-untyped-def]
        if shutdown_once.is_set():
            LOGGER.error("Second signal %s received; forcing exit", signum)
            os._exit(1)
        LOGGER.info("Received signal %s; shutting down", signum)
        stop_event.set()
        if not force_exit_scheduled.is_set():
            force_exit_scheduled.set()
            delay = (
                config.decode_timeout_sec if config.decode_timeout_sec > 0 else 5.0
            ) + 2.0
            threading.Thread(
                target=_force_exit_after, args=(delay,), daemon=True
            ).start()

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)
    else:
        LOGGER.warning("Signal handlers not installed (not running on main thread)")

    server.start()
    server_state["grpc_running"] = True
    try:
        while not stop_event.is_set():
            server.wait_for_termination(timeout=1.0)
    finally:
        shutdown()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the streaming STT server."""
    parser = argparse.ArgumentParser(description="Streaming STT gRPC server")
    parser.add_argument(
        "--config",
        type=str,
        help=f"Path to YAML config (default search: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        help=f"Path to model YAML (default: {DEFAULT_MODEL_CONFIG_PATH})",
    )
    parser.add_argument("--model", default=None, help="Whisper model size to load")
    parser.add_argument(
        "--model-backend",
        default=None,
        help="Model backend (faster_whisper | torch_whisper)",
    )
    parser.add_argument(
        "--device", default=None, help="Target device passed to the model backend"
    )
    parser.add_argument("--compute-type", default=None, help="Backend compute type")
    parser.add_argument(
        "--language",
        action="append",
        dest="languages",
        help="BCP-47 language code; last occurrence wins (omit for auto-detect)",
    )
    parser.add_argument(
        "--language-fix",
        dest="language_fix",
        action="store_true",
        help="Force the server to decode using the configured language",
    )
    parser.add_argument(
        "--no-language-fix",
        dest="language_fix",
        action="store_false",
        help="Allow automatic language detection (overrides config)",
    )
    parser.add_argument(
        "--task",
        choices=("transcribe", "translate"),
        default=None,
        help="Override the default Whisper task",
    )
    parser.add_argument(
        "--model-pool-size",
        type=int,
        default=None,
        help="Number of Whisper model instances to keep in the pool",
    )
    parser.add_argument("--port", type=int, default=None, help="Port to bind")
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=None,
        help="Port for HTTP metrics/health server",
    )
    parser.add_argument(
        "--grpc-worker-threads",
        type=int,
        default=None,
        help="gRPC thread pool size (0 or unset uses auto sizing)",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="Maximum concurrent gRPC streaming sessions",
    )
    parser.add_argument(
        "--decode-timeout",
        type=float,
        default=None,
        help="Seconds to wait for a decode task before logging a timeout (<=0 disables)",
    )
    parser.add_argument(
        "--log-metrics",
        dest="log_metrics",
        action="store_true",
        help="Print decode latency and real-time factor for each transcription pass",
    )
    parser.add_argument(
        "--no-log-metrics",
        dest="log_metrics",
        action="store_false",
        help="Disable metric logging (overrides config)",
    )
    parser.add_argument(
        "--log-transcripts",
        dest="log_transcripts",
        action="store_true",
        help="Log transcript text in decode logs (PII risk)",
    )
    parser.add_argument(
        "--no-log-transcripts",
        dest="log_transcripts",
        action="store_false",
        help="Disable transcript logging (overrides config)",
    )
    parser.set_defaults(
        log_metrics=None,
        log_transcripts=None,
        language_fix=None,
        tls_required=None,
        require_api_key=None,
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging level (e.g. DEBUG, INFO); overrides config",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path; overrides config",
    )
    parser.add_argument(
        "--transcript-log-file",
        default=None,
        help="Optional transcript log sink; enables transcript logging",
    )
    parser.add_argument(
        "--transcript-log-retention-days",
        type=int,
        default=None,
        help="Days to retain transcript logs (default: 7)",
    )
    parser.add_argument(
        "--faster-whisper-log-level",
        default=None,
        help="Override faster_whisper logger level (e.g. DEBUG, INFO, WARNING)",
    )
    parser.add_argument(
        "--tls-cert-file",
        default=None,
        help="Path to TLS certificate chain for gRPC server",
    )
    parser.add_argument(
        "--tls-key-file",
        default=None,
        help="Path to TLS private key for gRPC server",
    )
    parser.add_argument(
        "--tls-required",
        dest="tls_required",
        action="store_true",
        help="Require TLS; refuse to start without cert/key",
    )
    parser.add_argument(
        "--no-tls-required",
        dest="tls_required",
        action="store_false",
        help="Allow running gRPC without TLS (overrides config)",
    )
    parser.add_argument(
        "--vad-silence",
        type=float,
        default=None,
        help="Seconds of trailing silence that trigger VAD",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=None,
        help="VAD probability threshold (0-1)",
    )
    parser.add_argument(
        "--speech-threshold",
        type=float,
        default=None,
        help="Minimum RMS required before decoding buffered audio",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Fallback sample rate for streams that omit it",
    )
    parser.add_argument(
        "--require-api-key",
        dest="require_api_key",
        action="store_true",
        help="Require api_key attribute on CreateSession",
    )
    parser.add_argument(
        "--no-require-api-key",
        dest="require_api_key",
        action="store_false",
        help="Disable api_key requirement (overrides config)",
    )
    return parser.parse_args()


def configure_from_args(args: argparse.Namespace) -> ServerConfig:
    """Load config files and apply CLI overrides."""
    config_arg_path = Path(args.config).expanduser() if args.config else None
    model_config_arg_path = (
        Path(args.model_config).expanduser() if args.model_config else None
    )
    effective_config_path = config_arg_path or DEFAULT_CONFIG_PATH
    effective_model_path = model_config_arg_path or DEFAULT_MODEL_CONFIG_PATH
    config = load_config(effective_config_path, effective_model_path)

    if args.model is not None:
        config.model = args.model
    if args.model_backend is not None:
        config.model_backend = args.model_backend
    if args.device is not None:
        config.device = args.device
    if args.compute_type is not None:
        config.compute_type = args.compute_type
    if args.languages:
        config.language = args.languages[-1]
    if args.language_fix is not None:
        config.language_fix = args.language_fix
    if args.task:
        config.task = args.task
    if args.model_pool_size is not None:
        config.model_pool_size = args.model_pool_size
    if args.port is not None:
        config.port = args.port
    if args.max_sessions is not None:
        config.max_sessions = args.max_sessions
    if args.metrics_port is not None:
        config.metrics_port = args.metrics_port
    if args.grpc_worker_threads is not None:
        config.grpc_worker_threads = args.grpc_worker_threads
    if args.decode_timeout is not None:
        config.decode_timeout_sec = args.decode_timeout
    if args.speech_threshold is not None:
        config.speech_rms_threshold = args.speech_threshold
    if args.log_metrics is not None:
        config.log_metrics = args.log_metrics
    if args.log_transcripts is not None:
        config.log_transcripts = args.log_transcripts
    if args.log_level is not None:
        config.log_level = args.log_level
    if args.log_file is not None:
        config.log_file = args.log_file
    if args.transcript_log_file is not None:
        config.transcript_log_file = args.transcript_log_file
    if args.transcript_log_retention_days is not None:
        config.transcript_retention_days = args.transcript_log_retention_days
    if args.faster_whisper_log_level is not None:
        config.faster_whisper_log_level = args.faster_whisper_log_level
    if args.tls_cert_file is not None:
        config.tls_cert_file = args.tls_cert_file
    if args.tls_key_file is not None:
        config.tls_key_file = args.tls_key_file
    if args.tls_required is not None:
        config.tls_required = args.tls_required
    if args.vad_silence is not None:
        config.vad_silence = args.vad_silence
    if args.vad_threshold is not None:
        config.vad_threshold = args.vad_threshold
    if args.sample_rate is not None:
        config.sample_rate = args.sample_rate
    if args.require_api_key is not None:
        config.require_api_key = args.require_api_key

    configure_logging(
        config.log_level,
        config.log_file,
        config.faster_whisper_log_level,
        config.transcript_log_file,
        config.transcript_retention_days,
    )
    if config.log_transcripts and not config.transcript_log_file:
        LOGGER.warning(
            "log_transcripts is enabled but transcript_log_file is not set; "
            "transcript text will not be persisted."
        )
    if effective_config_path.exists():
        LOGGER.info("Loaded server config from %s", effective_config_path)
    else:
        LOGGER.info(
            "Server config file not found at %s; using defaults/CLI overrides",
            effective_config_path,
        )
    if effective_model_path.exists():
        LOGGER.info("Loaded model config from %s", effective_model_path)
    else:
        LOGGER.info(
            "Model config file not found at %s; using defaults/CLI overrides",
            effective_model_path,
        )
    return config


def main() -> None:
    """CLI entrypoint for launching the STT server."""
    args = parse_args()
    config = configure_from_args(args)
    serve(config)


if __name__ == "__main__":
    main()
