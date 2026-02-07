import argparse
import csv
import json
import math
import sys
import threading
import time
import uuid
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    TextIO,
    Tuple,
)

import grpc

from gen.stt.python.v1 import stt_pb2, stt_pb2_grpc

BYTES_PER_SAMPLE = 2


def load_wav(path: str) -> Tuple[bytes, int]:
    """Load a mono 16-bit PCM WAV file."""
    with wave.open(path, "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError("Only mono WAV files are supported")
        if wf.getsampwidth() != BYTES_PER_SAMPLE:
            raise ValueError("Only 16-bit PCM WAV files are supported")
        sr = wf.getframerate()
        data = wf.readframes(wf.getnframes())
    return data, sr


def audio_chunks(
    pcm16: bytes,
    sample_rate: int,
    chunk_ms: int,
    realtime: bool,
    speed: float,
    session_id: str,
    session_token: str,
    timing: "StreamTiming | None" = None,
):
    """Yield gRPC audio chunks from PCM16 data."""
    chunk_samples = max(int(sample_rate * (chunk_ms / 1000.0)), 1)
    chunk_bytes = chunk_samples * BYTES_PER_SAMPLE
    sleep_time = chunk_ms / 1000.0

    for idx in range(0, len(pcm16), chunk_bytes):
        if timing is not None:
            now = time.perf_counter()
            if timing.first_chunk_sec is None:
                timing.first_chunk_sec = now
            timing.last_chunk_sec = now
        chunk = pcm16[idx : idx + chunk_bytes]
        yield stt_pb2.AudioChunk(
            pcm16=chunk,
            sample_rate=sample_rate,
            is_final=False,
            session_id=session_id,
            session_token=session_token,
        )
        if realtime:
            pace = sleep_time
            if speed > 0:
                pace = sleep_time / speed
            if pace > 0:
                time.sleep(pace)

    if timing is not None:
        now = time.perf_counter()
        if timing.first_chunk_sec is None:
            timing.first_chunk_sec = now
        timing.last_chunk_sec = now
    yield stt_pb2.AudioChunk(
        pcm16=b"",
        sample_rate=sample_rate,
        is_final=True,
        session_id=session_id,
        session_token=session_token,
    )


def make_channel(
    target: str,
    tls: bool,
    ca_cert: Optional[str],
    server_hostname: Optional[str],
    options: List[Tuple[str, Any]],
) -> grpc.Channel:
    """Create gRPC channel with optional TLS support."""
    if not tls and not ca_cert:
        return grpc.insecure_channel(target, options=options)
    root = None
    if ca_cert:
        root = Path(ca_cert).expanduser().read_bytes()
    if server_hostname:
        options = options + [("grpc.ssl_target_name_override", server_hostname)]
    creds = grpc.ssl_channel_credentials(root_certificates=root)
    return grpc.secure_channel(target, creds, options=options)


def parse_task(value: str) -> stt_pb2.Task.ValueType:
    """Map a task string to the Task enum."""
    if value == "translate":
        return stt_pb2.TASK_TRANSLATE
    return stt_pb2.TASK_TRANSCRIBE


def parse_profile(value: str) -> stt_pb2.DecodeProfile.ValueType:
    """Map a profile string to the DecodeProfile enum."""
    if value == "accurate":
        return stt_pb2.DECODE_PROFILE_ACCURATE
    return stt_pb2.DECODE_PROFILE_REALTIME


def parse_vad_mode(value: str) -> stt_pb2.VADMode.ValueType:
    """Map a VAD mode string to the VADMode enum."""
    if value == "auto":
        return stt_pb2.VAD_AUTO_END
    return stt_pb2.VAD_CONTINUE


def _extract_decode_metrics(
    metadata: Optional[Iterable[Tuple[str, str]]],
) -> Dict[str, float]:
    """Extract decode timing metrics from trailing metadata."""
    if not metadata:
        return {}
    mapping: Dict[str, float] = {}
    for key, value in metadata:
        if isinstance(key, bytes):
            key = key.decode(errors="replace")
        key_lower = key.lower()
        if key_lower not in {
            "stt-decode-buffer-wait-sec",
            "stt-decode-queue-wait-sec",
            "stt-decode-inference-sec",
            "stt-decode-response-emit-sec",
            "stt-decode-total-sec",
            "stt-decode-count",
        }:
            continue
        if isinstance(value, bytes):
            value = value.decode(errors="replace")
        try:
            mapping[key_lower] = float(value)
        except (ValueError, TypeError):
            continue
    return mapping


@dataclass
class BenchStats:
    """Aggregates benchmark metrics across sessions."""

    sessions: int = 0
    failures: int = 0
    responses: int = 0
    total_time_sec: float = 0.0
    latencies: List[float] = field(default_factory=list)
    response_counts: List[int] = field(default_factory=list)
    error_counts: Dict[str, int] = field(default_factory=dict)
    failure_stages: Dict[str, int] = field(default_factory=dict)
    total_audio_sec: float = 0.0
    rtfs: List[float] = field(default_factory=list)
    tail_latencies: List[float] = field(default_factory=list)
    decode_buffer_waits: List[float] = field(default_factory=list)
    decode_queue_waits: List[float] = field(default_factory=list)
    decode_inference: List[float] = field(default_factory=list)
    decode_response_emit: List[float] = field(default_factory=list)
    decode_totals: List[float] = field(default_factory=list)
    decode_counts: List[int] = field(default_factory=list)
    decode_inference_avg: List[float] = field(default_factory=list)
    decode_queue_wait_avg: List[float] = field(default_factory=list)
    decode_buffer_wait_avg: List[float] = field(default_factory=list)
    decode_response_emit_avg: List[float] = field(default_factory=list)
    decode_total_avg: List[float] = field(default_factory=list)
    session_logs: List["SessionLog"] = field(default_factory=list)
    warmup_sessions: int = 0
    warmup_failures: int = 0

    def record(
        self,
        responses: int,
        elapsed: float,
        audio_sec: float,
        rtf: float,
        tail_sec: Optional[float],
        decode_buffer_wait_sec: Optional[float],
        decode_queue_wait_sec: Optional[float],
        decode_inference_sec: Optional[float],
        decode_response_emit_sec: Optional[float],
        decode_total_sec: Optional[float],
        decode_count: Optional[int],
    ) -> None:
        """Record a successful session result."""
        self.sessions += 1
        self.responses += responses
        self.total_time_sec += elapsed
        self.latencies.append(elapsed)
        self.response_counts.append(responses)
        self.total_audio_sec += audio_sec
        if rtf >= 0:
            self.rtfs.append(rtf)
        if tail_sec is not None:
            self.tail_latencies.append(tail_sec)
        if decode_buffer_wait_sec is not None:
            self.decode_buffer_waits.append(decode_buffer_wait_sec)
        if decode_queue_wait_sec is not None:
            self.decode_queue_waits.append(decode_queue_wait_sec)
        if decode_inference_sec is not None:
            self.decode_inference.append(decode_inference_sec)
        if decode_response_emit_sec is not None:
            self.decode_response_emit.append(decode_response_emit_sec)
        if decode_total_sec is not None:
            self.decode_totals.append(decode_total_sec)
        if decode_count is not None and decode_count > 0:
            self.decode_counts.append(decode_count)
            if decode_inference_sec is not None:
                self.decode_inference_avg.append(decode_inference_sec / decode_count)
            if decode_queue_wait_sec is not None:
                self.decode_queue_wait_avg.append(decode_queue_wait_sec / decode_count)
            if decode_buffer_wait_sec is not None:
                self.decode_buffer_wait_avg.append(
                    decode_buffer_wait_sec / decode_count
                )
            if decode_response_emit_sec is not None:
                self.decode_response_emit_avg.append(
                    decode_response_emit_sec / decode_count
                )
            if decode_total_sec is not None:
                self.decode_total_avg.append(decode_total_sec / decode_count)

    def record_failure(self, exc: grpc.RpcError, stage: str) -> None:
        """Record a failed session and error code."""
        self.failures += 1
        try:
            code = exc.code().name
        except (AttributeError, ValueError, TypeError):
            code = "UNKNOWN"
        self.error_counts[code] = self.error_counts.get(code, 0) + 1
        self.failure_stages[stage] = self.failure_stages.get(stage, 0) + 1

    def record_warmup(self, success: bool) -> None:
        """Record warmup success/failure."""
        self.warmup_sessions += 1
        if not success:
            self.warmup_failures += 1

    def maybe_log(self, entry: "SessionLog", max_logs: int) -> None:
        """Store a session log entry if under the limit."""
        if max_logs <= 0:
            return
        if len(self.session_logs) < max_logs:
            self.session_logs.append(entry)


@dataclass(frozen=True)
class SessionLog:
    """Per-session log entry for benchmark reporting."""

    session_id: str
    responses: int
    elapsed_sec: float
    success: bool
    error_code: Optional[str] = None
    send_sec: Optional[float] = None
    first_response_sec: Optional[float] = None
    tail_sec: Optional[float] = None
    total_sec: Optional[float] = None
    audio_sec: Optional[float] = None
    rtf: Optional[float] = None
    decode_buffer_wait_sec: Optional[float] = None
    decode_queue_wait_sec: Optional[float] = None
    decode_inference_sec: Optional[float] = None
    decode_response_emit_sec: Optional[float] = None
    decode_total_sec: Optional[float] = None
    decode_count: Optional[int] = None
    failure_stage: Optional[str] = None


SESSION_LOG_HEADERS = [
    "session_id",
    "responses",
    "success",
    "error_code",
    "failure_stage",
    "send_duration_seconds",
    "first_response_seconds",
    "tail_duration_seconds",
    "total_duration_seconds",
    "audio_seconds",
    "rtf",
    "decode_buffer_wait_seconds",
    "decode_queue_wait_seconds",
    "decode_inference_seconds",
    "decode_response_emit_seconds",
    "decode_total_seconds",
    "decode_count",
]

SESSION_LOG_COLUMN_DESCRIPTIONS = [
    ("session_id", "Client-provided session identifier"),
    ("responses", "Number of responses in the stream"),
    ("success", "True when the stream completed without error"),
    ("error_code", "gRPC error code when success is false"),
    ("failure_stage", "Stage of failure when success is false"),
    ("send_duration_seconds", "Time spent sending audio chunks from the client"),
    ("first_response_seconds", "Time from stream start to first response"),
    (
        "tail_duration_seconds",
        "Time from last audio chunk sent to final response",
    ),
    ("total_duration_seconds", "Total stream duration from start to finish"),
    ("audio_seconds", "Total audio seconds sent by the client"),
    ("rtf", "Real-time factor (audio_sec / total_duration)"),
    (
        "decode_buffer_wait_seconds",
        "Server-side time spent buffering audio before scheduling decode",
    ),
    (
        "decode_queue_wait_seconds",
        "Server-side time waiting for an available model worker",
    ),
    ("decode_inference_seconds", "Server-side model execution time"),
    (
        "decode_response_emit_seconds",
        "Server-side time spent yielding responses to the client",
    ),
    (
        "decode_total_seconds",
        "Sum of buffer wait, queue wait, inference, and response emit",
    ),
    ("decode_count", "Number of decode passes in the stream"),
]


def format_seconds_value(value: Optional[float]) -> Optional[float]:
    """Round a seconds value to 3 decimal places."""
    if value is None:
        return None
    return round(value, 3)


def format_seconds_string(value: Optional[float]) -> str:
    """Format a seconds value as a string for reports."""
    if value is None:
        return ""
    return f"{value:.3f}"


def session_log_payload(entry: SessionLog) -> Dict[str, object]:
    """Build a JSON-friendly payload for a session log entry."""
    return {
        "session_id": entry.session_id,
        "responses": entry.responses,
        "success": entry.success,
        "error_code": entry.error_code,
        "failure_stage": entry.failure_stage,
        "send_duration_seconds": format_seconds_value(entry.send_sec),
        "first_response_seconds": format_seconds_value(entry.first_response_sec),
        "tail_duration_seconds": format_seconds_value(entry.tail_sec),
        "total_duration_seconds": format_seconds_value(entry.total_sec),
        "audio_seconds": format_seconds_value(entry.audio_sec),
        "rtf": format_seconds_value(entry.rtf),
        "decode_buffer_wait_seconds": format_seconds_value(
            entry.decode_buffer_wait_sec
        ),
        "decode_queue_wait_seconds": format_seconds_value(entry.decode_queue_wait_sec),
        "decode_inference_seconds": format_seconds_value(entry.decode_inference_sec),
        "decode_response_emit_seconds": format_seconds_value(
            entry.decode_response_emit_sec
        ),
        "decode_total_seconds": format_seconds_value(entry.decode_total_sec),
        "decode_count": entry.decode_count,
    }


def session_log_row(entry: SessionLog) -> List[str]:
    """Build a CSV/TSV row for a session log entry."""
    return [
        entry.session_id,
        str(entry.responses),
        "true" if entry.success else "false",
        entry.error_code or "",
        entry.failure_stage or "",
        format_seconds_string(entry.send_sec),
        format_seconds_string(entry.first_response_sec),
        format_seconds_string(entry.tail_sec),
        format_seconds_string(entry.total_sec),
        format_seconds_string(entry.audio_sec),
        format_seconds_string(entry.rtf),
        format_seconds_string(entry.decode_buffer_wait_sec),
        format_seconds_string(entry.decode_queue_wait_sec),
        format_seconds_string(entry.decode_inference_sec),
        format_seconds_string(entry.decode_response_emit_sec),
        format_seconds_string(entry.decode_total_sec),
        str(entry.decode_count or ""),
    ]


def write_markdown_legend(handle: TextIO) -> None:
    """Write the column legend for markdown session logs."""
    handle.write("Session log column definitions:\n")
    for name, description in SESSION_LOG_COLUMN_DESCRIPTIONS:
        handle.write(f"- `{name}`: {description}\n")
    handle.write("\n")


def write_markdown_header(
    handle: TextIO, title: str, started_at: str, finished_at: Optional[str]
) -> None:
    """Write the markdown report header."""
    handle.write(f"# {title}\n\n")
    handle.write(f"- Test started at: {started_at}\n")
    if finished_at:
        handle.write(f"- Test finished at: {finished_at}\n")
    handle.write("\n")


class CsvWriter(Protocol):
    """Protocol for minimal CSV-like writers."""

    def writerow(self, row: Iterable[Any], /) -> Any:
        """Write a row to the CSV-like output."""
        ...


@dataclass
class SessionLogWriter:
    """Writes session logs in CSV/TSV/JSONL/Markdown formats."""

    handle: TextIO
    session_log_format: str
    lock: threading.Lock
    title: str
    started_at: str
    writer: Optional[CsvWriter] = None

    def write_header(self) -> None:
        """Write the header for the configured log format."""
        if self.session_log_format == "csv":
            writer = csv.writer(self.handle)
            self.writer = writer
            writer.writerow(SESSION_LOG_HEADERS)
        elif self.session_log_format == "tsv":
            writer = csv.writer(self.handle, delimiter="\t")
            self.writer = writer
            writer.writerow(SESSION_LOG_HEADERS)
        elif self.session_log_format == "markdown":
            write_markdown_header(self.handle, self.title, self.started_at, None)
            write_markdown_legend(self.handle)
            self.handle.write("| " + " | ".join(SESSION_LOG_HEADERS) + " |\n")
            self.handle.write(
                "| " + " | ".join(["---"] * len(SESSION_LOG_HEADERS)) + " |\n"
            )

    def write_entry(self, entry: SessionLog) -> None:
        """Write a single session log entry."""
        with self.lock:
            if self.session_log_format in {"csv", "tsv"}:
                writer = self.writer
                if writer is None:
                    return
                writer.writerow(session_log_row(entry))
                return
            if self.session_log_format == "jsonl":
                self.handle.write(json.dumps(session_log_payload(entry)) + "\n")
                return
            if self.session_log_format == "markdown":
                self.handle.write("| " + " | ".join(session_log_row(entry)) + " |\n")
                return

    def close(self) -> None:
        """Close the underlying log handle."""
        self.handle.close()


@dataclass
class StreamTiming:
    """Tracks timing for a single streaming call."""

    start_sec: float
    first_chunk_sec: Optional[float] = None
    last_chunk_sec: Optional[float] = None


@dataclass
class BenchConfig:
    """Configuration for running the gRPC benchmark."""

    target: str
    channels: int
    iterations: int
    warmup_iterations: int
    audio_path: str
    chunk_ms: int
    realtime: bool
    speed: float
    require_token: bool
    task: str
    profile: str
    vad_mode: str
    log_sessions: bool
    max_session_logs: int
    ramp_steps: int
    ramp_interval_sec: float
    session_log_writer: Optional[SessionLogWriter]
    tls: bool
    ca_cert: Optional[str]
    server_hostname: Optional[str]
    rpc_timeout_sec: Optional[float]
    stream_timeout_sec: Optional[float]
    metadata: List[Tuple[str, str]]


def run_channel(
    index: int,
    pcm16: bytes,
    sample_rate: int,
    config: BenchConfig,
    stats: BenchStats,
    lock: threading.Lock,
) -> None:
    """Run a single channel and record results."""
    channel = make_channel(
        config.target,
        config.tls,
        config.ca_cert,
        config.server_hostname,
        options=[],
    )
    stub = stt_pb2_grpc.STTBackendStub(channel)
    task_enum = parse_task(config.task)
    profile_enum = parse_profile(config.profile)
    vad_mode_enum = parse_vad_mode(config.vad_mode)

    total_iterations = config.warmup_iterations + config.iterations
    channel_width = max(1, len(str(config.channels)))
    iteration_width = max(1, len(str(config.iterations)))
    for i in range(total_iterations):
        session_id = (
            f"bench-{index:0{channel_width}d}-{i:0{iteration_width}d}-"
            f"{uuid.uuid4().hex[:8]}"
        )
        audio_sec = len(pcm16) / (sample_rate * BYTES_PER_SAMPLE)
        try:
            timing = StreamTiming(start_sec=time.perf_counter())
            try:
                response = stub.CreateSession(
                    stt_pb2.SessionRequest(
                        session_id=session_id,
                        require_token=config.require_token,
                        task=task_enum,
                        decode_profile=profile_enum,
                        vad_mode=vad_mode_enum,
                    ),
                    timeout=config.rpc_timeout_sec,
                    metadata=config.metadata,
                )
            except grpc.RpcError as exc:
                session_log_entry = SessionLog(
                    session_id=session_id,
                    responses=0,
                    elapsed_sec=0.0,
                    success=False,
                    error_code=exc.code().name if hasattr(exc, "code") else "UNKNOWN",
                    failure_stage="create_session",
                )
                if i >= config.warmup_iterations and config.session_log_writer:
                    config.session_log_writer.write_entry(session_log_entry)
                with lock:
                    if i < config.warmup_iterations:
                        stats.record_warmup(False)
                    else:
                        stats.record_failure(exc, "create_session")
                        if config.log_sessions:
                            stats.maybe_log(session_log_entry, config.max_session_logs)
                continue
            token = response.token if response.token_required else ""
            responses = 0
            first_response_at: Optional[float] = None
            last_response_at: Optional[float] = None
            call = stub.StreamingRecognize(
                audio_chunks(
                    pcm16,
                    sample_rate,
                    config.chunk_ms,
                    config.realtime,
                    config.speed,
                    session_id,
                    token,
                    timing=timing,
                ),
                timeout=config.stream_timeout_sec,
                metadata=config.metadata,
            )
            try:
                for _ in call:
                    now = time.perf_counter()
                    if first_response_at is None:
                        first_response_at = now
                    last_response_at = now
                    responses += 1
                try:
                    trailing = call.trailing_metadata()
                except grpc.RpcError:
                    trailing = None
            except grpc.RpcError as exc:
                session_log_entry = SessionLog(
                    session_id=session_id,
                    responses=responses,
                    elapsed_sec=0.0,
                    success=False,
                    error_code=exc.code().name if hasattr(exc, "code") else "UNKNOWN",
                    failure_stage="streaming",
                )
                if i >= config.warmup_iterations and config.session_log_writer:
                    config.session_log_writer.write_entry(session_log_entry)
                with lock:
                    if i < config.warmup_iterations:
                        stats.record_warmup(False)
                    else:
                        stats.record_failure(exc, "streaming")
                        if config.log_sessions:
                            stats.maybe_log(session_log_entry, config.max_session_logs)
                continue
            decode_metrics = _extract_decode_metrics(trailing)
            end = time.perf_counter()
            elapsed = end - timing.start_sec
            send_end = timing.last_chunk_sec or timing.start_sec
            send_sec = max(0.0, send_end - timing.start_sec)
            total_sec = max(0.0, elapsed)
            first_sec = (
                max(0.0, first_response_at - timing.start_sec)
                if first_response_at is not None
                else None
            )
            tail_sec = (
                max(0.0, last_response_at - send_end)
                if last_response_at is not None
                else None
            )
            rtf = audio_sec / elapsed if elapsed > 0 else 0.0
            session_log_entry = SessionLog(
                session_id=session_id,
                responses=responses,
                elapsed_sec=elapsed,
                success=True,
                audio_sec=audio_sec,
                rtf=rtf,
                send_sec=send_sec,
                first_response_sec=first_sec,
                tail_sec=tail_sec,
                total_sec=total_sec,
                decode_buffer_wait_sec=decode_metrics.get("stt-decode-buffer-wait-sec"),
                decode_queue_wait_sec=decode_metrics.get("stt-decode-queue-wait-sec"),
                decode_inference_sec=decode_metrics.get("stt-decode-inference-sec"),
                decode_response_emit_sec=decode_metrics.get(
                    "stt-decode-response-emit-sec"
                ),
                decode_total_sec=decode_metrics.get("stt-decode-total-sec"),
                decode_count=(
                    int(decode_metrics["stt-decode-count"])
                    if "stt-decode-count" in decode_metrics
                    else None
                ),
            )
            if i >= config.warmup_iterations and config.session_log_writer:
                config.session_log_writer.write_entry(session_log_entry)
            with lock:
                if i < config.warmup_iterations:
                    stats.record_warmup(True)
                else:
                    stats.record(
                        responses,
                        elapsed,
                        audio_sec,
                        rtf,
                        tail_sec,
                        decode_metrics.get("stt-decode-buffer-wait-sec"),
                        decode_metrics.get("stt-decode-queue-wait-sec"),
                        decode_metrics.get("stt-decode-inference-sec"),
                        decode_metrics.get("stt-decode-response-emit-sec"),
                        decode_metrics.get("stt-decode-total-sec"),
                        (
                            int(decode_metrics["stt-decode-count"])
                            if "stt-decode-count" in decode_metrics
                            else None
                        ),
                    )
                    if config.log_sessions:
                        stats.maybe_log(session_log_entry, config.max_session_logs)
        except grpc.RpcError as exc:
            session_log_entry = SessionLog(
                session_id=session_id,
                responses=0,
                elapsed_sec=0.0,
                success=False,
                error_code=exc.code().name if hasattr(exc, "code") else "UNKNOWN",
                failure_stage="unknown",
            )
            if i >= config.warmup_iterations and config.session_log_writer:
                config.session_log_writer.write_entry(session_log_entry)
            with lock:
                if i < config.warmup_iterations:
                    stats.record_warmup(False)
                else:
                    stats.record_failure(
                        exc, session_log_entry.failure_stage or "unknown"
                    )
                    if config.log_sessions:
                        stats.maybe_log(session_log_entry, config.max_session_logs)

    channel.close()


def main() -> None:
    """CLI entrypoint for the gRPC streaming load test."""
    parser = argparse.ArgumentParser(description="gRPC streaming load test")
    parser.add_argument("--target", default="localhost:50051")
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=1,
        help="Warm-up iterations per channel (not counted in metrics)",
    )
    parser.add_argument(
        "--audio",
        default="stt_client/assets/hello.wav",
        help="Path to mono 16-bit PCM WAV",
    )
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Realtime speed multiplier (2.0 = 2x faster). Used with --realtime.",
    )
    parser.add_argument("--tls", action="store_true", help="Enable TLS")
    parser.add_argument("--ca-cert", default=None, help="CA cert for TLS")
    parser.add_argument(
        "--server-hostname",
        default=None,
        help="Override TLS server hostname (SNI)",
    )
    parser.add_argument("--require-token", action="store_true")
    parser.add_argument("--rpc-timeout-sec", type=float, default=None)
    parser.add_argument("--stream-timeout-sec", type=float, default=None)
    parser.add_argument(
        "--deadline-sec",
        type=float,
        default=None,
        help="Set both rpc and stream timeouts",
    )
    parser.add_argument(
        "--task", choices=("transcribe", "translate"), default="transcribe"
    )
    parser.add_argument(
        "--decode-profile", choices=("realtime", "accurate"), default="realtime"
    )
    parser.add_argument("--vad-mode", choices=("continue", "auto"), default="continue")
    parser.add_argument(
        "--ramp-steps",
        type=int,
        default=5,
        help="Number of ramp-up steps for starting channels",
    )
    parser.add_argument(
        "--ramp-interval-sec",
        type=float,
        default=2.0,
        help="Seconds to wait between ramp-up steps",
    )
    parser.add_argument(
        "--log-sessions",
        action="store_true",
        help="Print per-session logs (limited by --max-session-logs)",
    )
    parser.add_argument(
        "--session-log-format",
        choices=("csv", "tsv", "jsonl", "markdown"),
        default="jsonl",
        help="Format for per-session logs",
    )
    parser.add_argument(
        "--session-log-path",
        default=None,
        help="Write all session logs to this file (uses session-log-format)",
    )
    parser.add_argument(
        "--max-session-logs",
        type=int,
        default=10,
        help="Max number of session logs to keep/print (0 disables)",
    )
    args = parser.parse_args()
    if args.deadline_sec is not None:
        args.rpc_timeout_sec = args.deadline_sec
        args.stream_timeout_sec = args.deadline_sec

    test_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session_log_writer: Optional[SessionLogWriter] = None
    if args.session_log_path:
        session_log_path = Path(args.session_log_path).expanduser()
        if session_log_path.parent:
            session_log_path.parent.mkdir(parents=True, exist_ok=True)
        session_log_handle = session_log_path.open("w", encoding="utf-8")
        session_log_writer = SessionLogWriter(
            handle=session_log_handle,
            session_log_format=args.session_log_format,
            lock=threading.Lock(),
            title="Bench session logs",
            started_at=test_started_at,
        )
        session_log_writer.write_header()

    pcm16, sample_rate = load_wav(args.audio)
    config = BenchConfig(
        target=args.target,
        channels=max(args.channels, 1),
        iterations=max(args.iterations, 1),
        warmup_iterations=max(args.warmup_iterations, 0),
        audio_path=args.audio,
        chunk_ms=max(args.chunk_ms, 1),
        realtime=args.realtime,
        speed=max(args.speed, 0.0),
        require_token=args.require_token,
        task=args.task,
        profile=args.decode_profile,
        vad_mode=args.vad_mode,
        log_sessions=args.log_sessions,
        max_session_logs=max(args.max_session_logs, 0),
        ramp_steps=max(args.ramp_steps, 1),
        ramp_interval_sec=max(args.ramp_interval_sec, 0.0),
        session_log_writer=session_log_writer,
        tls=args.tls or bool(args.ca_cert),
        ca_cert=args.ca_cert,
        server_hostname=args.server_hostname,
        rpc_timeout_sec=args.rpc_timeout_sec,
        stream_timeout_sec=args.stream_timeout_sec,
        metadata=[],
    )

    stats = BenchStats()
    lock = threading.Lock()
    threads = []
    start = time.perf_counter()

    total_threads = config.channels
    step_size = math.ceil(total_threads / config.ramp_steps)
    for idx in range(total_threads):
        thread = threading.Thread(
            target=run_channel,
            args=(idx, pcm16, sample_rate, config, stats, lock),
        )
        threads.append(thread)
        thread.start()
        if (idx + 1) % step_size == 0 and (idx + 1) < total_threads:
            time.sleep(config.ramp_interval_sec)

    for thread in threads:
        thread.join()
    test_finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if session_log_writer and session_log_writer.session_log_format == "markdown":
        session_log_writer.handle.write(f"\nTest finished at: {test_finished_at}\n")
    if session_log_writer:
        session_log_writer.close()

    elapsed = time.perf_counter() - start
    total_sessions = stats.sessions
    avg_time = stats.total_time_sec / total_sessions if total_sessions else 0.0
    throughput = total_sessions / elapsed if elapsed > 0 else 0.0

    latencies = sorted(stats.latencies)
    response_counts = sorted(stats.response_counts)

    def pct(values: Sequence[float], value: float) -> float:
        if not values:
            return 0.0
        rank = max(1, math.ceil((value / 100.0) * len(values))) - 1
        return values[min(rank, len(values) - 1)]

    p50 = pct(latencies, 50)
    p90 = pct(latencies, 90)
    p95 = pct(latencies, 95)
    p99 = pct(latencies, 99)

    r50 = pct(response_counts, 50)
    r90 = pct(response_counts, 90)
    r95 = pct(response_counts, 95)
    r99 = pct(response_counts, 99)

    print("* Info")
    print("    Sessions:", total_sessions)
    print("    Failures:", stats.failures)
    print("    Responses:", stats.responses)
    print("    Wall time:", f"{elapsed:.3f}s")
    print("    Avg session time:", f"{avg_time:.3f}s")
    print("    Sessions/sec:", f"{throughput:.3f}")
    if stats.total_audio_sec > 0 and elapsed > 0:
        print("    Audio-sec/sec:", f"{(stats.total_audio_sec / elapsed):.3f}")
    if stats.warmup_sessions:
        print("* Warmup")
        print("    Warmup sessions:", stats.warmup_sessions)
        print("    Warmup failures:", stats.warmup_failures)
    if stats.responses and elapsed > 0:
        print("* Response Rate")
        print("    per_sec:", f"{(stats.responses / elapsed):.3f}")
    if latencies:
        print("* Latency")
        print("    p50:", f"{p50:.3f}s")
        print("    p90:", f"{p90:.3f}s")
        print("    p95:", f"{p95:.3f}s")
        print("    p99:", f"{p99:.3f}s")
        print("    min:", f"{latencies[0]:.3f}s")
        print("    max:", f"{latencies[-1]:.3f}s")
    if response_counts:
        avg_responses = stats.responses / total_sessions if total_sessions else 0.0
        print("* Responses Per Session")
        print("    avg:", f"{avg_responses:.3f}")
        print("    p50:", f"{r50:.0f}")
        print("    p90:", f"{r90:.0f}")
        print("    p95:", f"{r95:.0f}")
        print("    p99:", f"{r99:.0f}")
        print("    min:", f"{response_counts[0]:.0f}")
        print("    max:", f"{response_counts[-1]:.0f}")
    if stats.rtfs:
        r50 = pct(sorted(stats.rtfs), 50)
        r95 = pct(sorted(stats.rtfs), 95)
        r99 = pct(sorted(stats.rtfs), 99)
        print("* RTF")
        print("    p50:", f"{r50:.3f}")
        print("    p95:", f"{r95:.3f}")
        print("    p99:", f"{r99:.3f}")
    if stats.tail_latencies:
        t50 = pct(sorted(stats.tail_latencies), 50)
        t95 = pct(sorted(stats.tail_latencies), 95)
        t99 = pct(sorted(stats.tail_latencies), 99)
        print("* Tail")
        print("    p50:", f"{t50:.3f}s")
        print("    p95:", f"{t95:.3f}s")
        print("    p99:", f"{t99:.3f}s")
    if stats.decode_buffer_waits:
        b50 = pct(sorted(stats.decode_buffer_waits), 50)
        b95 = pct(sorted(stats.decode_buffer_waits), 95)
        b99 = pct(sorted(stats.decode_buffer_waits), 99)
        print("* Decode Buffer Wait")
        print("    p50:", f"{b50:.3f}s")
        print("    p95:", f"{b95:.3f}s")
        print("    p99:", f"{b99:.3f}s")
    if stats.decode_queue_waits:
        q50 = pct(sorted(stats.decode_queue_waits), 50)
        q95 = pct(sorted(stats.decode_queue_waits), 95)
        q99 = pct(sorted(stats.decode_queue_waits), 99)
        print("* Decode Queue Wait")
        print("    p50:", f"{q50:.3f}s")
        print("    p95:", f"{q95:.3f}s")
        print("    p99:", f"{q99:.3f}s")
    if stats.decode_inference:
        i50 = pct(sorted(stats.decode_inference), 50)
        i95 = pct(sorted(stats.decode_inference), 95)
        i99 = pct(sorted(stats.decode_inference), 99)
        print("* Decode Inference")
        print("    p50:", f"{i50:.3f}s")
        print("    p95:", f"{i95:.3f}s")
        print("    p99:", f"{i99:.3f}s")
    if stats.decode_response_emit:
        e50 = pct(sorted(stats.decode_response_emit), 50)
        e95 = pct(sorted(stats.decode_response_emit), 95)
        e99 = pct(sorted(stats.decode_response_emit), 99)
        print("* Decode Response Emit")
        print("    p50:", f"{e50:.3f}s")
        print("    p95:", f"{e95:.3f}s")
        print("    p99:", f"{e99:.3f}s")
    if stats.decode_totals:
        d50 = pct(sorted(stats.decode_totals), 50)
        d95 = pct(sorted(stats.decode_totals), 95)
        d99 = pct(sorted(stats.decode_totals), 99)
        print("* Decode Total")
        print("    p50:", f"{d50:.3f}s")
        print("    p95:", f"{d95:.3f}s")
        print("    p99:", f"{d99:.3f}s")
        if (
            stats.decode_buffer_waits
            and stats.decode_queue_waits
            and stats.decode_inference
        ):
            b50 = pct(sorted(stats.decode_buffer_waits), 50)
            q50 = pct(sorted(stats.decode_queue_waits), 50)
            i50 = pct(sorted(stats.decode_inference), 50)
            e50 = (
                pct(sorted(stats.decode_response_emit), 50)
                if stats.decode_response_emit
                else 0.0
            )
            total = d50 if d50 > 0 else 0.0
            if total > 0:
                shares = {
                    "queue_wait": q50 / total,
                    "inference": i50 / total,
                    "buffer_wait": b50 / total,
                    "response_emit": e50 / total,
                }
                dominant = max(shares.items(), key=lambda item: item[1])[0]
                print("* Decode Bottleneck")
                print(
                    "    p50 share:",
                    f"queue_wait={shares['queue_wait']:.0%}",
                    f"inference={shares['inference']:.0%}",
                    f"buffer_wait={shares['buffer_wait']:.0%}",
                    f"emit={shares['response_emit']:.0%}",
                )
                print(
                    "    dominant:",
                    f"{dominant} ({shares[dominant]:.0%} of total)",
                )
    if stats.decode_counts:
        c50 = pct(sorted(stats.decode_counts), 50)
        c95 = pct(sorted(stats.decode_counts), 95)
        c99 = pct(sorted(stats.decode_counts), 99)
        print("* Decode Count")
        print("    p50:", f"{c50:.0f}")
        print("    p95:", f"{c95:.0f}")
        print("    p99:", f"{c99:.0f}")
    if stats.decode_total_avg:
        a50 = pct(sorted(stats.decode_total_avg), 50)
        a95 = pct(sorted(stats.decode_total_avg), 95)
        a99 = pct(sorted(stats.decode_total_avg), 99)
        print("* Decode Avg (per pass)")
        print("    p50:", f"{a50:.3f}s")
        print("    p95:", f"{a95:.3f}s")
        print("    p99:", f"{a99:.3f}s")
    if stats.error_counts:
        print("* Error Codes")
        for code, count in sorted(stats.error_counts.items()):
            print(f"  {code}: {count}")
    if stats.failure_stages:
        print("* Failure Stages")
        for stage, count in sorted(stats.failure_stages.items()):
            print(f"  {stage}: {count}")
    if stats.session_logs:
        if args.session_log_format == "csv":
            writer = csv.writer(sys.stdout)
            writer.writerow(SESSION_LOG_HEADERS)
            for entry in stats.session_logs:
                writer.writerow(session_log_row(entry))
        elif args.session_log_format == "tsv":
            writer = csv.writer(sys.stdout, delimiter="\t")
            writer.writerow(SESSION_LOG_HEADERS)
            for entry in stats.session_logs:
                writer.writerow(session_log_row(entry))
        elif args.session_log_format == "jsonl":
            for entry in stats.session_logs:
                sys.stdout.write(json.dumps(session_log_payload(entry)) + "\n")
        elif args.session_log_format == "markdown":
            write_markdown_header(
                sys.stdout, "Bench session logs", test_started_at, test_finished_at
            )
            write_markdown_legend(sys.stdout)
            print("| " + " | ".join(SESSION_LOG_HEADERS) + " |")
            print("| " + " | ".join(["---"] * len(SESSION_LOG_HEADERS)) + " |")
            for entry in stats.session_logs:
                print("| " + " | ".join(session_log_row(entry)) + " |")
        else:
            for entry in stats.session_logs:
                sys.stdout.write(json.dumps(session_log_payload(entry)) + "\n")


if __name__ == "__main__":
    if __package__ is None:
        raise SystemExit(
            "Run as a module from the repo root: python -m tools.bench.grpc_load_test ..."
        )
    main()
