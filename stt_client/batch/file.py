"""Batch client for submitting a single audio file to the STT server."""

import argparse
import hashlib
import hmac
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import grpc
import numpy as np
import soundfile as sf
import yaml

from gen.stt.python.v1 import stt_pb2, stt_pb2_grpc

TASK_CHOICES = ("transcribe", "translate")
PROFILE_CHOICES = ("realtime", "accurate")
DEFAULT_AUDIO_PATH = Path(__file__).resolve().parents[1] / "assets" / "hello.wav"
DEFAULT_AUDIO_DISPLAY = "stt_client/assets/hello.wav"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "file.yaml"
CONFIG_KEYS = {
    "audio_path",
    "server",
    "vad_mode",
    "metrics",
    "grpc_max_receive_message_bytes",
    "grpc_max_send_message_bytes",
    "tls",
    "tls_ca_file",
    "vad_silence",
    "vad_threshold",
    "require_token",
    "language",
    "task",
    "decode_profile",
    "attributes",
    "signed_token_secret",
}


@dataclass(frozen=True)
class ResultDisplay:
    """Display-friendly fields for a recognition result."""

    session_id: str
    text: str
    time: str
    language: str
    language_code: str
    score: float
    recognized_at: str


@dataclass(frozen=True)
class ConnectionConfig:
    """Connection settings for the batch client."""

    target: str
    grpc_max_receive_message_bytes: Optional[int]
    grpc_max_send_message_bytes: Optional[int]
    tls_enabled: bool
    tls_ca_file: Optional[str]


@dataclass(frozen=True)
class SessionConfig:
    """Session settings for the batch client."""

    attributes: Dict[str, str]
    require_token: bool
    signed_token_secret: Optional[str]
    language: str
    task: str
    decode_profile: str
    vad: "VADConfig"


@dataclass(frozen=True)
class VADConfig:
    """VAD settings for the batch client."""

    mode: str
    silence: Optional[float]
    threshold: Optional[float]


@dataclass(frozen=True)
class RunConfig:
    """Immutable configuration for a single batch run."""

    path: str
    connection: ConnectionConfig
    session: SessionConfig
    report_metrics: bool


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load a YAML config file as a mapping."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping at the top level.")
    return data


def task_to_enum(value: str) -> stt_pb2.Task.ValueType:
    """Map task name strings to protobuf enums."""
    return (
        stt_pb2.TASK_TRANSLATE
        if value.lower() == "translate"
        else stt_pb2.TASK_TRANSCRIBE
    )


def profile_to_enum(value: str) -> stt_pb2.DecodeProfile.ValueType:
    """Map decode profile strings to protobuf enums."""
    return (
        stt_pb2.DECODE_PROFILE_ACCURATE
        if value.lower() == "accurate"
        else stt_pb2.DECODE_PROFILE_REALTIME
    )


def load_audio(filepath: str) -> Tuple[np.ndarray, int]:
    """Load an audio file and return PCM16 samples + sample rate."""
    audio, sr = sf.read(filepath)
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = (audio * 32767).astype("int16")
    return audio, sr


def merge_transcript(prefix: str, next_text: str) -> str:
    """Merge partial transcripts without duplicating prefixes."""
    prefix = prefix.strip()
    next_text = next_text.strip()
    if not prefix:
        return next_text
    if not next_text:
        return prefix
    if next_text.startswith(prefix):
        return next_text
    return f"{prefix} {next_text}"


def _create_channel(
    target: str,
    grpc_max_receive_message_bytes: Optional[int],
    grpc_max_send_message_bytes: Optional[int],
    tls_enabled: bool,
    tls_ca_file: Optional[str],
) -> grpc.Channel:
    """Create a gRPC channel with optional message size limits."""
    options = []
    if grpc_max_receive_message_bytes and grpc_max_receive_message_bytes > 0:
        options.append(
            ("grpc.max_receive_message_length", grpc_max_receive_message_bytes)
        )
    if grpc_max_send_message_bytes and grpc_max_send_message_bytes > 0:
        options.append(("grpc.max_send_message_length", grpc_max_send_message_bytes))
    if tls_ca_file:
        tls_enabled = True
        cert_path = Path(tls_ca_file).expanduser()
        if not cert_path.exists():
            raise FileNotFoundError(f"TLS CA file not found: {cert_path}")
        root_certificates = cert_path.read_bytes()
    else:
        root_certificates = None
    if tls_enabled:
        credentials = grpc.ssl_channel_credentials(root_certificates=root_certificates)
        return grpc.secure_channel(target, credentials, options=options)
    if options:
        return grpc.insecure_channel(target, options=options)
    return grpc.insecure_channel(target)


def _format_value(key: str, value: Any) -> str:
    """Format scalar values for display."""
    if isinstance(value, float):
        suffix = "s" if key.endswith("_sec") else ""
        return f"{value:.2f}{suffix}"
    return str(value)


def format_kv_block(title: str, values: Dict[str, Any]) -> str:
    """Format a dict into an aligned key/value block."""
    if not values:
        return f"[{title}]"
    width = max(len(label) for label in values)
    lines = [f"[{title}]"]
    for label, value in values.items():
        lines.append(f"  {label:<{width}} : {_format_value(label, value)}")
    return "\n".join(lines)


def format_output(kind: str, display: ResultDisplay) -> str:
    """Format a recognition result for display."""
    return format_kv_block(kind, asdict(display))


def _ensure_session_id(attributes: Dict[str, str]) -> str:
    """Ensure attributes contain a session_id and return it."""
    session_id = attributes.get("session_id") or str(int(time.time() * 1000))
    attributes["session_id"] = session_id
    return session_id


def _build_signed_token_metadata(
    session_id: str, signed_token_secret: Optional[str]
) -> list[tuple[str, str]]:
    secret = (signed_token_secret or "").strip()
    if not secret:
        return []
    timestamp = str(int(time.time()))
    payload = f"{session_id}:{timestamp}".encode("utf-8")
    signature = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return [("authorization", f"Bearer {signature}"), ("x-stt-auth-ts", timestamp)]


def _build_session_request(
    session: SessionConfig, session_id: str
) -> stt_pb2.SessionRequest:
    """Build a session creation request for the batch run."""
    request = stt_pb2.SessionRequest(
        session_id=session_id,
        attributes=session.attributes,
        vad_mode=(
            stt_pb2.VAD_AUTO_END
            if session.vad.mode.lower() == "auto"
            else stt_pb2.VAD_CONTINUE
        ),
        vad_silence=session.vad.silence if session.vad.silence is not None else 0.0,
        vad_threshold=(
            session.vad.threshold if session.vad.threshold is not None else 0.0
        ),
        require_token=session.require_token,
        language_code=session.language,
        task=task_to_enum(session.task),
        decode_profile=profile_to_enum(session.decode_profile),
    )
    if session.vad.threshold is not None:
        request.vad_threshold_override = session.vad.threshold
    return request


def _print_stream_results(
    responses: Iterator[stt_pb2.STTResult],
    session_id: str,
    stream_start: float,
) -> None:
    """Print stream responses as they arrive."""
    committed_text = ""
    for resp in responses:
        recognized_at = time.perf_counter() - stream_start
        language_name = (resp.language or resp.language_code or "unknown").strip()
        language_code = (resp.language_code or "").strip()
        score = resp.probability
        server_committed = (getattr(resp, "committed_text", "") or "").strip()
        server_unstable = (getattr(resp, "unstable_text", "") or "").strip()
        if server_committed or server_unstable:
            display_text = f"{server_committed} {server_unstable}".strip()
            if server_committed:
                committed_text = server_committed
            elif resp.is_final:
                committed_text = display_text
        elif resp.is_final:
            committed_text = merge_transcript(committed_text, resp.text)
            display_text = committed_text
        else:
            display_text = merge_transcript(committed_text, resp.text)
        if resp.is_final:
            kind = "FINAL"
        else:
            kind = "PARTIAL"
        display = ResultDisplay(
            session_id=session_id,
            text=display_text,
            time=f"{resp.start_sec:.2f}-{resp.end_sec:.2f}s",
            language=language_name,
            language_code=language_code,
            score=score,
            recognized_at=f"{recognized_at:.2f}s",
        )
        print(format_output(kind, display))


def _print_metrics(
    session_id: str,
    audio_len: int,
    sample_rate: int,
    stream_start: float,
) -> None:
    """Print client-side metrics for a completed stream."""
    total_wall = time.perf_counter() - stream_start
    print(
        format_kv_block(
            "METRIC",
            {
                "session_id": session_id,
                "mode": "batch",
                "audio_duration_sec": (
                    float(audio_len) / sample_rate if sample_rate > 0 else 0.0
                ),
                "wall_clock_sec": total_wall,
            },
        )
    )


def single_chunk_iter(
    pcm_bytes: bytes, sample_rate: int, session_id: str, session_token: str
) -> Iterator[stt_pb2.AudioChunk]:
    """Yield a single AudioChunk carrying the entire file buffer."""
    yield stt_pb2.AudioChunk(
        pcm16=pcm_bytes,
        sample_rate=sample_rate,
        is_final=True,
        session_id=session_id,
        session_token=session_token,
    )


def run(config: RunConfig) -> None:
    """Run a batch streaming request against the STT server."""
    channel = _create_channel(
        config.connection.target,
        config.connection.grpc_max_receive_message_bytes,
        config.connection.grpc_max_send_message_bytes,
        config.connection.tls_enabled,
        config.connection.tls_ca_file,
    )
    metrics_ready = False
    audio_len = 0
    sample_rate = 0
    stream_start = 0.0
    try:
        stub = stt_pb2_grpc.STTBackendStub(channel)
        session_id = _ensure_session_id(config.session.attributes)
        request = _build_session_request(config.session, session_id)
        metadata = _build_signed_token_metadata(
            session_id, config.session.signed_token_secret
        )
        if metadata:
            session_resp = stub.CreateSession(request, metadata=metadata)
        else:
            session_resp = stub.CreateSession(request)
        session_token = session_resp.token if session_resp.token_required else ""
        print(
            f"[SESSION] session_id={session_id} created "
            f"(token_required={session_resp.token_required})"
        )

        audio, sample_rate = load_audio(config.path)
        audio_len = len(audio)
        pcm_bytes = audio.tobytes()
        responses = stub.StreamingRecognize(
            single_chunk_iter(pcm_bytes, sample_rate, session_id, session_token),
            metadata=[("session-id", session_id)],
        )
        print(
            f"[STREAM] session_id={session_id} started for path='{config.path}' "
            f"({sample_rate} Hz)"
        )
        stream_start = time.perf_counter()
        metrics_ready = True
        try:
            _print_stream_results(responses, session_id, stream_start)
            print(f"[STREAM] session_id={session_id} completed normally")
        except grpc.RpcError as exc:
            print(
                f"[STREAM] session_id={session_id} terminated by RPC error: {exc}",
                file=sys.stderr,
            )
            raise
        except (RuntimeError, ValueError) as exc:
            print(
                f"[STREAM] session_id={session_id} terminated by client error: {exc}",
                file=sys.stderr,
            )
            raise
        finally:
            if config.report_metrics and metrics_ready:
                _print_metrics(session_id, audio_len, sample_rate, stream_start)
    finally:
        channel.close()


def _build_config_parser() -> argparse.ArgumentParser:
    """Build a parser for the optional YAML config flag."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-c",
        "--config",
        nargs="?",
        const=str(DEFAULT_CONFIG_PATH),
        default=None,
        help=f"Path to YAML config (default: {DEFAULT_CONFIG_PATH})",
    )
    return parser


def _normalize_config_attributes(
    raw_config: Dict[str, Any], parser: argparse.ArgumentParser
) -> None:
    """Normalize the attributes entry into a list of key=value strings."""
    if "attributes" not in raw_config:
        return
    attributes = raw_config["attributes"]
    if attributes is None:
        raw_config["attributes"] = []
    elif isinstance(attributes, dict):
        raw_config["attributes"] = [
            f"{key}={value}" for key, value in attributes.items()
        ]
    elif isinstance(attributes, list):
        raw_config["attributes"] = [str(item) for item in attributes]
    else:
        parser.error(
            "Config 'attributes' must be a mapping or list of KEY=VALUE strings."
        )


def _load_cli_config(
    config_path: Optional[str], parser: argparse.ArgumentParser
) -> Dict[str, Any]:
    """Load CLI config values from YAML when provided."""
    if not config_path:
        return {}
    resolved = Path(config_path).expanduser()
    if not resolved.exists():
        parser.error(f"Config file not found: {resolved}")
    try:
        raw_config = load_yaml_config(resolved)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(f"Failed to load config file: {exc}")
    _normalize_config_attributes(raw_config, parser)
    return {k: v for k, v in raw_config.items() if k in CONFIG_KEYS}


def _build_arg_parser(config_values: Dict[str, Any]) -> argparse.ArgumentParser:
    """Build the CLI parser for batch streaming arguments."""
    parser = argparse.ArgumentParser(description="Batch STT client (single chunk)")
    parser.add_argument(
        "-c",
        "--config",
        nargs="?",
        const=str(DEFAULT_CONFIG_PATH),
        default=None,
        help=f"Path to YAML config (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=str(DEFAULT_AUDIO_PATH),
        metavar="AUDIO",
        help=(
            "Path to a WAV/FLAC file readable by soundfile "
            f"(default: {DEFAULT_AUDIO_DISPLAY})"
        ),
    )
    parser.add_argument(
        "--server",
        default="localhost:50051",
        help="gRPC target in host:port format (default: %(default)s)",
    )
    parser.add_argument(
        "--vad-mode",
        choices=("continue", "auto"),
        default="auto",
        help="VAD mode for the server session (default: %(default)s)",
    )
    parser.add_argument(
        "--require-token",
        action="store_true",
        help="Request a session token and include it with every chunk",
    )
    parser.add_argument(
        "--signed-token-secret",
        default=None,
        help="HMAC secret for CreateSession signed_token metadata",
    )
    parser.add_argument(
        "--language",
        default="",
        help="Desired input language (BCP-47 code, leave blank for auto)",
    )
    parser.add_argument(
        "--task",
        choices=TASK_CHOICES,
        default="transcribe",
        help="Whisper task; default: %(default)s",
    )
    parser.add_argument(
        "--decode-profile",
        choices=PROFILE_CHOICES,
        default="accurate",
        help="Decoding profile to request; default: %(default)s",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Print client-side performance metrics after streaming",
    )
    parser.add_argument(
        "--grpc-max-receive-message-bytes",
        type=int,
        default=None,
        help="Max gRPC receive message size in bytes (default: unset)",
    )
    parser.add_argument(
        "--grpc-max-send-message-bytes",
        type=int,
        default=None,
        help="Max gRPC send message size in bytes (default: unset)",
    )
    parser.add_argument(
        "--tls",
        action="store_true",
        help="Enable TLS for the gRPC channel (uses system trust store)",
    )
    parser.add_argument(
        "--tls-ca-file",
        default=None,
        help="Path to CA certificate for TLS (self-signed certs)",
    )
    parser.add_argument(
        "--attr",
        "--meta",
        dest="attributes",
        action="append",
        metavar="KEY=VALUE",
        default=[],
        help="Session attributes (repeatable). '--meta' kept for compatibility.",
    )
    parser.add_argument(
        "--vad-silence",
        type=float,
        default=None,
        help="Override server VAD silence window in seconds",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=None,
        help="Override server VAD probability threshold (0-1)",
    )
    if config_values:
        parser.set_defaults(**config_values)
    return parser


def _resolve_audio_path(audio_path: str, parser: argparse.ArgumentParser) -> Path:
    """Resolve and validate the audio path argument."""
    resolved = Path(audio_path).expanduser()
    if not resolved.exists():
        parser.error(f"Audio file not found: {resolved}")
    return resolved


def _parse_attributes(
    entries: list[str], parser: argparse.ArgumentParser
) -> Dict[str, str]:
    """Parse --attr entries into a dict."""
    attributes: Dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            parser.error(f"Invalid --attr entry (expected key=value): {entry}")
        key, value = entry.split("=", 1)
        attributes[key] = value
    return attributes


def main() -> None:
    """CLI entrypoint for the batch client."""
    config_parser = _build_config_parser()
    config_args, remaining_argv = config_parser.parse_known_args()
    config_values = _load_cli_config(config_args.config, config_parser)
    parser = _build_arg_parser(config_values)
    args = parser.parse_args(remaining_argv)

    audio_path = _resolve_audio_path(args.audio_path, parser)
    attr_dict = _parse_attributes(args.attributes, parser)
    run(
        RunConfig(
            path=str(audio_path),
            connection=ConnectionConfig(
                target=args.server,
                grpc_max_receive_message_bytes=args.grpc_max_receive_message_bytes,
                grpc_max_send_message_bytes=args.grpc_max_send_message_bytes,
                tls_enabled=args.tls,
                tls_ca_file=args.tls_ca_file,
            ),
            session=SessionConfig(
                attributes=attr_dict,
                require_token=args.require_token,
                signed_token_secret=args.signed_token_secret,
                language=args.language,
                task=args.task,
                decode_profile=args.decode_profile,
                vad=VADConfig(
                    mode=args.vad_mode,
                    silence=args.vad_silence,
                    threshold=args.vad_threshold,
                ),
            ),
            report_metrics=args.metrics,
        )
    )


if __name__ == "__main__":
    main()
