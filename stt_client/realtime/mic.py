"""Realtime microphone client for the STT server."""

import argparse
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

import grpc
import sounddevice as sd
import yaml

from gen.stt.python.v1 import stt_pb2, stt_pb2_grpc

TASK_CHOICES = ("transcribe", "translate")
PROFILE_CHOICES = ("realtime", "accurate")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "mic.yaml"
CONFIG_KEYS = {
    "server",
    "chunk_ms",
    "sample_rate",
    "device",
    "metrics",
    "grpc_max_receive_message_bytes",
    "grpc_max_send_message_bytes",
    "vad_mode",
    "require_token",
    "language",
    "task",
    "decode_profile",
    "vad_silence",
    "vad_threshold",
}


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


def _create_channel(
    target: str,
    grpc_max_receive_message_bytes: Optional[int],
    grpc_max_send_message_bytes: Optional[int],
) -> grpc.Channel:
    """Create a gRPC channel with optional message size limits."""
    options = []
    if grpc_max_receive_message_bytes and grpc_max_receive_message_bytes > 0:
        options.append(
            ("grpc.max_receive_message_length", grpc_max_receive_message_bytes)
        )
    if grpc_max_send_message_bytes and grpc_max_send_message_bytes > 0:
        options.append(("grpc.max_send_message_length", grpc_max_send_message_bytes))
    if options:
        return grpc.insecure_channel(target, options=options)
    return grpc.insecure_channel(target)


class MicrophoneStream:
    """Capture PCM16 audio from the default macOS microphone."""

    def __init__(self, sample_rate: int, chunk_ms: int, device: Optional[str] = None):
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.device = device
        self.frames_per_chunk = max(int(sample_rate * (chunk_ms / 1000)), 1)
        self.queue: "queue.Queue[Union[bytes, Exception, None]]" = queue.Queue()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.samples_sent = 0

    def start(self) -> "MicrophoneStream":
        """Start the background audio capture thread."""
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return self

    def _capture_loop(self) -> None:
        """Capture audio frames and enqueue them for streaming."""
        try:
            with sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=self.frames_per_chunk,
                channels=1,
                dtype="int16",
                device=self.device,
            ) as stream:
                while not self.stop_event.is_set():
                    data, overflowed = stream.read(self.frames_per_chunk)
                    if overflowed:
                        print(
                            "[MIC] Input overflow detected; audio may drop.",
                            file=sys.stderr,
                        )
                    self.queue.put(bytes(data))
        except Exception as exc:  # PortAudio errors, etc.
            self.queue.put(exc)
        finally:
            self.queue.put(None)

    def request_stream(
        self, session_id: str, session_token: str
    ) -> Iterator[stt_pb2.AudioChunk]:
        """Yield AudioChunk messages from captured microphone audio."""
        while True:
            item = self.queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            self.samples_sent += len(item) // 2
            yield stt_pb2.AudioChunk(
                pcm16=item,
                sample_rate=self.sample_rate,
                is_final=False,
                session_id=session_id,
                session_token=session_token,
            )

        yield stt_pb2.AudioChunk(
            pcm16=b"",
            sample_rate=self.sample_rate,
            is_final=True,
            session_id=session_id,
            session_token=session_token,
        )

    def stop(self) -> None:
        """Signal the capture thread to stop and drain the queue."""
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        # Ensure the consumer wakes up even if capture thread exited early.
        self.queue.put(None)

    @property
    def duration_seconds(self) -> float:
        """Return total captured audio duration in seconds."""
        if self.sample_rate <= 0:
            return 0.0
        return self.samples_sent / float(self.sample_rate)


def run(
    target: str,
    sample_rate: int,
    chunk_ms: int,
    input_device: Optional[str],
    report_metrics: bool,
    grpc_max_receive_message_bytes: Optional[int],
    grpc_max_send_message_bytes: Optional[int],
    vad_mode: str,
    require_token: bool,
    language: str,
    task: str,
    decode_profile: str,
    vad_silence: Optional[float],
    vad_threshold: Optional[float],
) -> None:
    """Run a realtime microphone streaming session."""

    def merge_transcript(prefix: str, next_text: str) -> str:
        """Combine partial transcripts without duplicating prefixes."""
        prefix = prefix.strip()
        next_text = next_text.strip()
        if not prefix:
            return next_text
        if not next_text:
            return prefix
        if next_text.startswith(prefix):
            return next_text
        return f"{prefix} {next_text}"

    def format_output(
        prefix: str,
        text: str,
        start_sec: float,
        end_sec: float,
        language: str,
        score: float,
        recognized_at: float,
    ) -> str:
        """Format a single recognition response for display."""
        return (
            f"{prefix} TEXT: {text}\n"
            f"   TIME: {start_sec:.2f}-{end_sec:.2f}s\n"
            f"   LANG: {language} | SCORE: {score:.2f} | "
            f"RECOGNIZED_AT: {recognized_at:.2f}s"
        )

    mic = MicrophoneStream(
        sample_rate=sample_rate, chunk_ms=chunk_ms, device=input_device
    ).start()
    channel = _create_channel(
        target, grpc_max_receive_message_bytes, grpc_max_send_message_bytes
    )
    stub = stt_pb2_grpc.STTBackendStub(channel)
    session_id = str(int(time.time() * 1000))
    attributes: Dict[str, str] = {}
    if vad_silence is not None:
        attributes["vad_silence"] = str(vad_silence)
    if vad_threshold is not None:
        attributes["vad_threshold"] = str(vad_threshold)

    request = stt_pb2.SessionRequest(
        session_id=session_id,
        attributes=attributes,
        vad_mode=(
            stt_pb2.VAD_AUTO_END if vad_mode.lower() == "auto" else stt_pb2.VAD_CONTINUE
        ),
        vad_silence=vad_silence or 0.0,
        vad_threshold=vad_threshold or 0.0,
        require_token=require_token,
        language_code=language,
        task=task_to_enum(task),
        decode_profile=profile_to_enum(decode_profile),
    )
    if vad_threshold is not None:
        request.vad_threshold_override = vad_threshold
    session_resp = stub.CreateSession(request)
    session_token = session_resp.token if session_resp.token_required else ""
    print(
        f"[SESSION] session_id={session_id} created "
        f"(token_required={session_resp.token_required})"
    )

    stream_start = time.perf_counter()
    request_iter = mic.request_stream(session_id, session_token)
    responses = stub.StreamingRecognize(
        request_iter,
        metadata=[("session-id", session_id)],
    )
    print(
        f"[STREAM] session_id={session_id} microphone streaming at {sample_rate} Hz "
        f"({chunk_ms} ms chunks). Press Ctrl+C to stop."
    )

    committed_text = ""
    stop_requested = False
    try:
        response_iter = iter(responses)
        while True:
            try:
                resp = next(response_iter)
            except StopIteration:
                break
            except KeyboardInterrupt:
                if not stop_requested:
                    stop_requested = True
                    mic.stop()
                    print(
                        "[STREAM] Stop requested; waiting for final results. "
                        "Press Ctrl+C again to force exit."
                    )
                    continue
                raise

            recognized_at = time.perf_counter() - stream_start
            language = (resp.language or resp.language_code or "unknown").strip()
            score = resp.probability
            if resp.is_final:
                committed_text = merge_transcript(committed_text, resp.text)
                display_text = committed_text
                print(
                    format_output(
                        "✅",
                        display_text,
                        resp.start_sec,
                        resp.end_sec,
                        language,
                        score,
                        recognized_at,
                    )
                )
            else:
                display_text = merge_transcript(committed_text, resp.text)
                print(
                    format_output(
                        "⏳",
                        display_text,
                        resp.start_sec,
                        resp.end_sec,
                        language,
                        score,
                        recognized_at,
                    )
                )
        print(f"[STREAM] session_id={session_id} completed normally")
    except grpc.RpcError as exc:
        print(
            f"[STREAM] session_id={session_id} terminated by RPC error: {exc}",
            file=sys.stderr,
        )
        raise
    except KeyboardInterrupt:
        print(f"[STREAM] session_id={session_id} interrupted by user")
    except Exception as exc:
        print(
            f"[STREAM] session_id={session_id} terminated by client error: {exc}",
            file=sys.stderr,
        )
        raise
    finally:
        mic.stop()
        channel.close()
        total_wall = time.perf_counter() - stream_start
        if report_metrics:
            audio_duration = mic.duration_seconds
            rtf = total_wall / audio_duration if audio_duration > 0 else float("inf")
            print(
                f"[METRIC] audio_duration={audio_duration:.2f}s "
                f"wall_clock={total_wall:.2f}s real_time_factor={rtf:.2f}"
            )


def main() -> None:
    """CLI entrypoint for the realtime microphone client."""
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
        config_values = {k: v for k, v in raw_config.items() if k in CONFIG_KEYS}

    parser = argparse.ArgumentParser(
        description="Streaming STT client (macOS microphone)"
    )
    parser.add_argument(
        "-c",
        "--config",
        nargs="?",
        const=str(DEFAULT_CONFIG_PATH),
        default=None,
        help=f"Path to YAML config (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--server",
        default="localhost:50051",
        help="gRPC target in host:port format (default: %(default)s)",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=100,
        help="Chunk size in milliseconds (default: %(default)s)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Microphone capture sample rate (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="macOS CoreAudio input device name/index (defaults to system mic)",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Print capture duration and real-time factor on exit",
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
        "--vad-mode",
        choices=("continue", "auto"),
        default="continue",
        help="VAD mode (continue or auto); default: %(default)s",
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
        default="realtime",
        help="Decoding profile to request; default: %(default)s",
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

    run(
        target=args.server,
        sample_rate=args.sample_rate,
        chunk_ms=args.chunk_ms,
        input_device=args.device,
        report_metrics=args.metrics,
        grpc_max_receive_message_bytes=args.grpc_max_receive_message_bytes,
        grpc_max_send_message_bytes=args.grpc_max_send_message_bytes,
        vad_mode=args.vad_mode,
        require_token=args.require_token,
        language=args.language,
        task=args.task,
        decode_profile=args.decode_profile,
        vad_silence=args.vad_silence,
        vad_threshold=args.vad_threshold,
    )


if __name__ == "__main__":
    main()
