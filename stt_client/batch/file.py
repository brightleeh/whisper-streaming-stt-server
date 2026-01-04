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
    "vad_silence",
    "vad_threshold",
    "require_token",
    "language",
    "task",
    "decode_profile",
    "attributes",
}


@dataclass(frozen=True)
class ResultDisplay:
    session_id: str
    text: str
    time: str
    language: str
    language_code: str
    score: float
    recognized_at: str


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping at the top level.")
    return data


def task_to_enum(value: str) -> stt_pb2.Task.ValueType:
    return (
        stt_pb2.TASK_TRANSLATE
        if value.lower() == "translate"
        else stt_pb2.TASK_TRANSCRIBE
    )


def profile_to_enum(value: str) -> stt_pb2.DecodeProfile.ValueType:
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
    if isinstance(value, float):
        suffix = "s" if key.endswith("_sec") else ""
        return f"{value:.2f}{suffix}"
    return str(value)


def format_kv_block(title: str, values: Dict[str, Any]) -> str:
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


def run(
    path: str,
    target: str,
    vad_mode: str,
    attributes: Dict[str, str],
    require_token: bool,
    language: str,
    task: str,
    decode_profile: str,
    report_metrics: bool,
    vad_silence: Optional[float],
    vad_threshold: Optional[float],
) -> None:
    channel = grpc.insecure_channel(target)
    stub = stt_pb2_grpc.STTBackendStub(channel)
    session_id = attributes.get("session_id") or str(int(time.time() * 1000))
    attributes["session_id"] = session_id

    request = stt_pb2.SessionRequest(
        session_id=session_id,
        attributes=attributes,
        vad_mode=(
            stt_pb2.VAD_AUTO_END if vad_mode.lower() == "auto" else stt_pb2.VAD_CONTINUE
        ),
        vad_silence=vad_silence if vad_silence is not None else 0.0,
        vad_threshold=vad_threshold if vad_threshold is not None else 0.0,
        require_token=require_token,
        language_code=language,
        task=task_to_enum(task),
        decode_profile=profile_to_enum(decode_profile),
    )
    session_resp = stub.CreateSession(request)
    session_token = session_resp.token if session_resp.token_required else ""
    print(
        f"[SESSION] session_id={session_id} created "
        f"(token_required={session_resp.token_required})"
    )

    audio, sample_rate = load_audio(path)
    pcm_bytes = audio.tobytes()
    responses = stub.StreamingRecognize(
        single_chunk_iter(pcm_bytes, sample_rate, session_id, session_token),
        metadata=[("session-id", session_id)],
    )
    print(
        f"[STREAM] session_id={session_id} started for path='{path}' "
        f"({sample_rate} Hz)"
    )
    try:
        committed_text = ""
        stream_start = time.perf_counter()
        for resp in responses:
            recognized_at = time.perf_counter() - stream_start
            language_name = (resp.language or resp.language_code or "unknown").strip()
            language_code = (resp.language_code or "").strip()
            score = resp.probability
            if resp.is_final:
                committed_text = merge_transcript(committed_text, resp.text)
                display_text = committed_text
                kind = "FINAL"
            else:
                display_text = merge_transcript(committed_text, resp.text)
                kind = "PARTIAL"
            print(
                format_output(
                    kind,
                    display_text,
                    resp.start_sec,
                    resp.end_sec,
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
    except Exception as exc:
        print(
            f"[STREAM] session_id={session_id} terminated by client error: {exc}",
            file=sys.stderr,
        )
        raise
    finally:
        if report_metrics:
            total_wall = time.perf_counter() - stream_start
            print(
                format_kv_block(
                    "METRIC",
                    {
                        "session_id": session_id,
                        "mode": "batch",
                        "audio_duration_sec": (
                            float(len(audio)) / sample_rate if sample_rate > 0 else 0.0
                        ),
                        "wall_clock_sec": total_wall,
                    },
                )
            )
        channel.close()


def main() -> None:
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
        except Exception as exc:
            config_parser.error(f"Failed to load config file: {exc}")
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
        vad_mode=args.vad_mode,
        attributes=attr_dict,
        require_token=args.require_token,
        language=args.language,
        task=args.task,
        decode_profile=args.decode_profile,
        report_metrics=args.metrics,
        vad_silence=args.vad_silence,
        vad_threshold=args.vad_threshold,
    )


if __name__ == "__main__":
    main()
