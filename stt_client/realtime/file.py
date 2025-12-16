import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import grpc
import numpy as np
import soundfile as sf

from gen.stt.python.v1 import stt_pb2, stt_pb2_grpc

CHUNK_MS = 100  # 100ms
TASK_CHOICES = ("transcribe", "translate")
PROFILE_CHOICES = ("realtime", "accurate")


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
    audio, sr = sf.read(filepath)
    if audio.ndim > 1:
        audio = audio[:, 0]  # mono only
    audio = (audio * 32767).astype("int16")
    return audio, sr


def stream_chunks(
    audio: np.ndarray,
    sr: int,
    chunk_ms: int,
    realtime: bool,
    stats: Dict[str, int],
    session_id: str,
    session_token: str,
) -> Iterator[stt_pb2.AudioChunk]:
    samples_per_chunk = max(int(sr * (chunk_ms / 1000)), 1)
    idx = 0
    total = len(audio)
    sleep_time = chunk_ms / 1000.0

    while idx < total:
        end = min(idx + samples_per_chunk, total)
        pcm = audio[idx:end].tobytes()
        idx = end

        stats["chunks"] += 1
        print(f"[SEND] [session_id={session_id}] chunk_bytes={len(pcm)}")
        yield stt_pb2.AudioChunk(
            pcm16=pcm,
            sample_rate=sr,
            is_final=False,
            session_id=session_id,
            session_token=session_token,
        )

        if realtime:
            time.sleep(sleep_time)

    stats["chunks"] += 1
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
    epd_mode: str,
    attributes: Dict[str, str],
    require_token: bool,
    language: str,
    task: str,
    decode_profile: str,
    epd_silence: Optional[float],
    epd_threshold: Optional[float],
) -> None:
    channel = grpc.insecure_channel(target)
    stub = stt_pb2_grpc.STTBackendStub(channel)
    session_id = attributes.get("session_id") or str(int(time.time() * 1000))
    attributes["session_id"] = session_id

    session_resp = stub.CreateSession(
        stt_pb2.SessionRequest(
            session_id=session_id,
            attributes=attributes,
            epd_mode=(
                stt_pb2.EPD_AUTO_END if epd_mode == "auto" else stt_pb2.EPD_CONTINUE
            ),
            epd_silence=epd_silence or 0.0,
            epd_threshold=epd_threshold or 0.0,
            require_token=require_token,
            language_code=language,
            task=task_to_enum(task),
            decode_profile=profile_to_enum(decode_profile),
        )
    )
    session_token = session_resp.token if session_resp.token_required else ""
    print(
        f"[SESSION] session_id={session_id} created "
        f"(token_required={session_resp.token_required})"
    )

    audio, sr = load_audio(path)
    audio_duration = len(audio) / float(sr) if sr > 0 else 0.0

    stats = {"chunks": 0, "responses": 0}
    start = time.perf_counter()

    responses = stub.StreamingRecognize(
        stream_chunks(audio, sr, chunk_ms, realtime, stats, session_id, session_token),
        metadata=[("session-id", session_id)],
    )
    print(
        f"[STREAM] session_id={session_id} started "
        f"(chunk_ms={chunk_ms}, realtime={realtime})"
    )

    try:
        for r in responses:
            stats["responses"] += 1
            kind = "[FINAL]" if r.is_final else "[PARTIAL]"
            print(f"{kind} [session_id={session_id}] {r.text}")
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
        total_wall = time.perf_counter() - start
        if report_metrics:
            rtf = total_wall / audio_duration if audio_duration > 0 else float("inf")
            mode = "realtime" if realtime else "burst"
            print(
                f"[METRIC] mode={mode} chunks_sent={stats['chunks']} "
                f"responses={stats['responses']} "
                f"audio_duration={audio_duration:.2f}s wall_clock={total_wall:.2f}s "
                f"real_time_factor={rtf:.2f}"
            )
        channel.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming STT sample client")
    parser.add_argument(
        "audio_path",
        metavar="AUDIO",
        help="Path to a WAV/FLAC file readable by soundfile",
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
        "--epd-mode",
        choices=("continue", "auto"),
        default="continue",
        help="EPD mode (continue or auto); default: %(default)s",
    )
    parser.add_argument(
        "--epd-silence",
        type=float,
        default=None,
        help="Override server EPD silence window in seconds",
    )
    parser.add_argument(
        "--epd-threshold",
        type=float,
        default=None,
        help="Override server EPD RMS threshold",
    )
    parser.add_argument(
        "--require-token",
        action="store_true",
        help="Request a session token and include it with every chunk",
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
    args = parser.parse_args()

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
        epd_mode=args.epd_mode,
        attributes=attr_dict,
        require_token=args.require_token,
        language=args.language,
        task=args.task,
        decode_profile=args.decode_profile,
        epd_silence=args.epd_silence,
        epd_threshold=args.epd_threshold,
    )


if __name__ == "__main__":
    main()
