"""Realtime file-streaming client for the STT server."""

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import grpc
import numpy as np
import soundfile as sf
import yaml

from gen.stt.python.v1 import stt_pb2
from stt_client.sdk import StreamingClient

CHUNK_MS = 100  # 100ms
TASK_CHOICES = ("transcribe", "translate")
PROFILE_CHOICES = ("realtime", "accurate")
DEFAULT_AUDIO_PATH = Path(__file__).resolve().parents[1] / "assets" / "hello.wav"
DEFAULT_AUDIO_DISPLAY = "stt_client/assets/hello.wav"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "file.yaml"
CONFIG_KEYS = {
    "audio_path",
    "server",
    "chunk_ms",
    "no_realtime",
    "metrics",
    "grpc_max_receive_message_bytes",
    "grpc_max_send_message_bytes",
    "tls",
    "tls_ca_file",
    "vad_mode",
    "vad_silence",
    "vad_threshold",
    "require_token",
    "language",
    "task",
    "decode_profile",
    "attributes",
    "signed_token_secret",
}


@dataclass
class StreamStats:
    """Accumulates stream counters for chunks and responses."""

    chunks: int = 0
    responses: int = 0


@dataclass(frozen=True)
class ResultDisplay:
    """Display-ready fields for a recognition result."""

    session_id: str
    text: str
    time: str
    language: str
    language_code: str
    score: float
    recognized_at: str


@dataclass(frozen=True)
class MetricSummary:
    """Summary of client-side streaming metrics."""

    session_id: str
    mode: str
    chunks_sent: int
    responses: int
    audio_duration_sec: float
    wall_clock_sec: float
    real_time_factor: float


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file into a dict."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping at the top level.")
    return data


def task_to_enum(value: str) -> stt_pb2.Task.ValueType:
    """Map a task string to the protobuf enum."""
    return (
        stt_pb2.TASK_TRANSLATE
        if value.lower() == "translate"
        else stt_pb2.TASK_TRANSCRIBE
    )


def profile_to_enum(value: str) -> stt_pb2.DecodeProfile.ValueType:
    """Map a decode profile string to the protobuf enum."""
    return (
        stt_pb2.DECODE_PROFILE_ACCURATE
        if value.lower() == "accurate"
        else stt_pb2.DECODE_PROFILE_REALTIME
    )


def load_audio(filepath: str) -> Tuple[np.ndarray, int]:
    """Load an audio file and return PCM16 samples with sample rate."""
    audio, sr = sf.read(filepath)
    if audio.ndim > 1:
        audio = audio[:, 0]  # mono only
    audio = (audio * 32767).astype("int16")
    return audio, sr


def merge_transcript(prefix: str, next_text: str) -> str:
    """Combine partial transcripts while avoiding duplicated prefixes."""
    prefix = prefix.strip()
    next_text = next_text.strip()
    if not prefix:
        return next_text
    if not next_text:
        return prefix
    if next_text.startswith(prefix):
        return next_text
    return f"{prefix} {next_text}"


def _format_value(key: str, value: Any) -> str:
    """Format scalar values for key/value display."""
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


def format_output(
    kind: str,
    text: str,
    start_sec: float,
    end_sec: float,
    language: str,
    language_code: str,
    score: float,
    recognized_at: float,
    session_id: str,
) -> str:
    """Format a recognition response for display."""
    display = ResultDisplay(
        session_id=session_id,
        text=text,
        time=f"{start_sec:.2f}-{end_sec:.2f}s",
        language=language,
        language_code=language_code,
        score=score,
        recognized_at=f"{recognized_at:.2f}s",
    )
    return format_kv_block(kind, asdict(display))


def stream_chunks(
    audio: np.ndarray,
    sr: int,
    chunk_ms: int,
    realtime: bool,
    stats: StreamStats,
    session_id: str,
    session_token: str,
    progress_state: Optional[Dict[str, bool]] = None,
) -> Iterator[stt_pb2.AudioChunk]:
    """Yield PCM chunks as AudioChunk messages, optionally in real time."""
    samples_per_chunk = max(int(sr * (chunk_ms / 1000)), 1)
    idx = 0
    total = len(audio)
    total_bytes = audio.nbytes
    bytes_sent = 0
    sleep_time = chunk_ms / 1000.0

    while idx < total:
        end = min(idx + samples_per_chunk, total)
        pcm = audio[idx:end].tobytes()
        idx = end

        stats.chunks += 1
        bytes_sent += len(pcm)
        print(f"\r[SEND] bytes={bytes_sent}/{total_bytes}\033[K", end="", flush=True)
        if progress_state is not None:
            progress_state["dirty"] = True
        yield stt_pb2.AudioChunk(
            pcm16=pcm,
            sample_rate=sr,
            is_final=False,
            session_id=session_id,
            session_token=session_token,
        )

        if realtime:
            time.sleep(sleep_time)

    if total_bytes:
        print()
        if progress_state is not None:
            progress_state["dirty"] = False

    stats.chunks += 1
    yield stt_pb2.AudioChunk(
        pcm16=b"",
        sample_rate=sr,
        is_final=True,
        session_id=session_id,
        session_token=session_token,
    )


def run(
    path: str,
    target: str,
    chunk_ms: int,
    realtime: bool,
    report_metrics: bool,
    grpc_max_receive_message_bytes: Optional[int],
    grpc_max_send_message_bytes: Optional[int],
    tls_enabled: bool,
    tls_ca_file: Optional[str],
    vad_mode: str,
    attributes: Dict[str, str],
    require_token: bool,
    signed_token_secret: Optional[str],
    language: str,
    task: str,
    decode_profile: str,
    vad_silence: Optional[float],
    vad_threshold: Optional[float],
) -> None:
    """Run a realtime streaming session for a single audio file."""
    client = StreamingClient(
        target,
        grpc_max_receive_message_bytes=grpc_max_receive_message_bytes,
        grpc_max_send_message_bytes=grpc_max_send_message_bytes,
        tls_enabled=tls_enabled,
        tls_ca_file=tls_ca_file,
        signed_token_secret=signed_token_secret,
    )
    session_id = attributes.get("session_id") or str(int(time.time() * 1000))
    attributes["session_id"] = session_id

    request = stt_pb2.SessionRequest(
        session_id=session_id,
        attributes=attributes,
        vad_mode=(stt_pb2.VAD_AUTO_END if vad_mode == "auto" else stt_pb2.VAD_CONTINUE),
        vad_silence=vad_silence or 0.0,
        vad_threshold=vad_threshold or 0.0,
        require_token=require_token,
        language_code=language,
        task=task_to_enum(task),
        decode_profile=profile_to_enum(decode_profile),
    )
    if vad_threshold is not None:
        request.vad_threshold_override = vad_threshold
    metadata = client.build_signed_metadata(session_id)
    session_resp = client.create_session(request, metadata=metadata)
    session_token = session_resp.token if session_resp.token_required else ""
    print(
        f"[SESSION] session_id={session_id} created "
        f"(token_required={session_resp.token_required})"
    )

    audio, sr = load_audio(path)
    audio_duration = len(audio) / float(sr) if sr > 0 else 0.0

    stats = StreamStats()
    stream_start = time.perf_counter()

    progress_state = {"dirty": False}
    responses = client.streaming_recognize(
        stream_chunks(
            audio,
            sr,
            chunk_ms,
            realtime,
            stats,
            session_id,
            session_token,
            progress_state,
        ),
        metadata=[("session-id", session_id)],
    )
    print(
        f"[STREAM] session_id={session_id} started "
        f"(chunk_ms={chunk_ms}, realtime={realtime})"
    )

    committed_text = ""
    try:
        for r in responses:
            stats.responses += 1
            recognized_at = time.perf_counter() - stream_start
            language_name = (r.language or r.language_code or "unknown").strip()
            language_code = (r.language_code or "").strip()
            score = r.probability
            if progress_state["dirty"]:
                print()
                progress_state["dirty"] = False
            server_committed = (getattr(r, "committed_text", "") or "").strip()
            server_unstable = (getattr(r, "unstable_text", "") or "").strip()
            if server_committed or server_unstable:
                display_text = f"{server_committed} {server_unstable}".strip()
                if server_committed:
                    committed_text = server_committed
                elif r.is_final:
                    committed_text = display_text
            elif r.is_final:
                committed_text = merge_transcript(committed_text, r.text)
                display_text = committed_text
            else:
                display_text = merge_transcript(committed_text, r.text)
            if r.is_final:
                print(
                    format_output(
                        "FINAL",
                        display_text,
                        r.start_sec,
                        r.end_sec,
                        language_name,
                        language_code,
                        score,
                        recognized_at,
                        session_id,
                    )
                )
            else:
                print(
                    format_output(
                        "PARTIAL",
                        display_text,
                        r.start_sec,
                        r.end_sec,
                        language_name,
                        language_code,
                        score,
                        recognized_at,
                        session_id,
                    )
                )
        print(f"[STREAM] session_id={session_id} completed normally")
    except grpc.RpcError as exc:
        print(
            f"[STREAM] session_id={session_id} terminated by RPC error: {exc}",
            file=sys.stderr,
        )
        raise
    except (RuntimeError, ValueError, OSError) as exc:
        print(
            f"[STREAM] session_id={session_id} terminated by client error: {exc}",
            file=sys.stderr,
        )
        raise
    finally:
        total_wall = time.perf_counter() - stream_start
        if report_metrics:
            rtf = total_wall / audio_duration if audio_duration > 0 else float("inf")
            mode = "realtime" if realtime else "burst"
            summary = MetricSummary(
                session_id=session_id,
                mode=mode,
                chunks_sent=stats.chunks,
                responses=stats.responses,
                audio_duration_sec=audio_duration,
                wall_clock_sec=total_wall,
                real_time_factor=rtf,
            )
            print(format_kv_block("METRIC", asdict(summary)))
        client.close()


def main() -> None:
    """CLI entrypoint for the realtime file client."""
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "-c",
        "--config",
        nargs="?",
        const=str(DEFAULT_CONFIG_PATH),
        default=None,
        help=f"Path to YAML config (default: {DEFAULT_CONFIG_PATH})",
    )
    config_args, remaining_argv = config_parser.parse_known_args()
    config_values: Dict[str, Any] = {}
    if config_args.config:
        config_path = Path(config_args.config).expanduser()
        if not config_path.exists():
            config_parser.error(f"Config file not found: {config_path}")
        try:
            raw_config = load_yaml_config(config_path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            config_parser.error(f"Failed to load config file: {exc}")
        if "realtime" in raw_config and "no_realtime" not in raw_config:
            raw_config["no_realtime"] = not bool(raw_config["realtime"])
        raw_config.pop("realtime", None)
        if "attributes" in raw_config:
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
                config_parser.error(
                    "Config 'attributes' must be a mapping or list of KEY=VALUE strings."
                )
        config_values = {k: v for k, v in raw_config.items() if k in CONFIG_KEYS}

    parser = argparse.ArgumentParser(description="Streaming STT sample client")
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
        "--chunk-ms",
        type=int,
        default=CHUNK_MS,
        help="Chunk size in milliseconds (default: %(default)s)",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Send audio as fast as possible (disable per-chunk sleep)",
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
        "--vad-mode",
        choices=("continue", "auto"),
        default="continue",
        help="VAD mode (continue or auto); default: %(default)s",
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
        "--attr",
        "--meta",
        dest="attributes",
        action="append",
        metavar="KEY=VALUE",
        default=[],
        help="Session attributes (repeatable). '--meta' is kept for compatibility.",
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
        default="realtime",
        help="Decoding profile to request; default: %(default)s",
    )
    if config_values:
        parser.set_defaults(**config_values)
    args = parser.parse_args(remaining_argv)

    audio_path = Path(args.audio_path).expanduser()
    if not audio_path.exists():
        parser.error(f"Audio file not found: {audio_path}")

    attr_dict: Dict[str, str] = {}
    for entry in args.attributes:
        if "=" not in entry:
            parser.error(f"Invalid --attr entry (expected key=value): {entry}")
        key, value = entry.split("=", 1)
        attr_dict[key] = value
    run(
        str(audio_path),
        target=args.server,
        chunk_ms=args.chunk_ms,
        realtime=not args.no_realtime,
        report_metrics=args.metrics,
        grpc_max_receive_message_bytes=args.grpc_max_receive_message_bytes,
        grpc_max_send_message_bytes=args.grpc_max_send_message_bytes,
        tls_enabled=args.tls,
        tls_ca_file=args.tls_ca_file,
        vad_mode=args.vad_mode,
        attributes=attr_dict,
        require_token=args.require_token,
        signed_token_secret=args.signed_token_secret,
        language=args.language,
        task=args.task,
        decode_profile=args.decode_profile,
        vad_silence=args.vad_silence,
        vad_threshold=args.vad_threshold,
    )


if __name__ == "__main__":
    main()
