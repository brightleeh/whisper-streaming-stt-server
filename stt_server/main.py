import argparse
import os
import signal
import threading
from concurrent import futures
from pathlib import Path

import grpc

from gen.stt.python.v1 import stt_pb2_grpc
from stt_server.backend.transport import (
    ModelRuntimeConfig,
    ServicerConfig,
    StorageRuntimeConfig,
    StreamingRuntimeConfig,
    STTGrpcServicer,
    start_http_server,
)
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
    grpc_executor = futures.ThreadPoolExecutor(max_workers=config.max_sessions)
    server = grpc.server(grpc_executor)
    model_cfg = ModelRuntimeConfig(
        model_size=config.model,
        device=config.device,
        compute_type=config.compute_type,
        log_metrics=config.log_metrics,
        task=config.task,
        language=config.language,
        language_fix=config.language_fix,
        model_pool_size=config.model_pool_size,
        decode_profiles=config.decode_profiles,
        default_decode_profile=config.default_decode_profile,
    )
    streaming_cfg = StreamingRuntimeConfig(
        vad_silence=config.vad_silence,
        vad_threshold=config.vad_threshold,
        speech_rms_threshold=config.speech_rms_threshold,
        session_timeout_sec=config.session_timeout_sec,
        sample_rate=config.sample_rate,
        decode_timeout_sec=config.decode_timeout_sec,
    )
    storage_cfg = StorageRuntimeConfig(
        enabled=config.persist_audio,
        directory=config.audio_storage_dir,
        max_bytes=config.audio_storage_max_bytes,
        max_files=config.audio_storage_max_files,
        max_age_days=config.audio_storage_max_age_days,
    )
    servicer_config = ServicerConfig(
        model=model_cfg, streaming=streaming_cfg, storage=storage_cfg
    )
    servicer = STTGrpcServicer(servicer_config)
    stt_pb2_grpc.add_STTBackendServicer_to_server(servicer, server)  # type: ignore[name-defined]
    server.add_insecure_port(f"[::]:{config.port}")
    http_handle = start_http_server(
        runtime=servicer.runtime,
        server_state=server_state,
        host="0.0.0.0",
        port=config.metrics_port,
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
        "--device", default=None, help="Target device passed to faster-whisper"
    )
    parser.add_argument(
        "--compute-type", default=None, help="faster-whisper compute_type"
    )
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
    parser.set_defaults(log_metrics=None, language_fix=None)
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
        "--faster-whisper-log-level",
        default=None,
        help="Override faster_whisper logger level (e.g. DEBUG, INFO, WARNING)",
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
    return parser.parse_args()


def configure_from_args(args: argparse.Namespace) -> ServerConfig:
    config_arg_path = Path(args.config).expanduser() if args.config else None
    model_config_arg_path = (
        Path(args.model_config).expanduser() if args.model_config else None
    )
    effective_config_path = config_arg_path or DEFAULT_CONFIG_PATH
    effective_model_path = model_config_arg_path or DEFAULT_MODEL_CONFIG_PATH
    config = load_config(effective_config_path, effective_model_path)

    if args.model is not None:
        config.model = args.model
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
    if args.decode_timeout is not None:
        config.decode_timeout_sec = args.decode_timeout
    if args.speech_threshold is not None:
        config.speech_rms_threshold = args.speech_threshold
    if args.log_metrics is not None:
        config.log_metrics = args.log_metrics
    if args.log_level is not None:
        config.log_level = args.log_level
    if args.log_file is not None:
        config.log_file = args.log_file
    if args.faster_whisper_log_level is not None:
        config.faster_whisper_log_level = args.faster_whisper_log_level
    if args.vad_silence is not None:
        config.vad_silence = args.vad_silence
    if args.vad_threshold is not None:
        config.vad_threshold = args.vad_threshold
    if args.sample_rate is not None:
        config.sample_rate = args.sample_rate

    configure_logging(
        config.log_level, config.log_file, config.faster_whisper_log_level
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
    args = parse_args()
    config = configure_from_args(args)
    serve(config)


if __name__ == "__main__":
    main()
