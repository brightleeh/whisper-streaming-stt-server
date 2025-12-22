import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import grpc
import numpy as np
import soundfile as sf

from gen.stt.python.v1 import stt_pb2, stt_pb2_grpc

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
    """Load an audio file and return PCM16 samples + sample rate."""
    audio, sr = sf.read(filepath)
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = (audio * 32767).astype("int16")
    return audio, sr


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
        for resp in responses:
            kind = "[FINAL]" if resp.is_final else "[PARTIAL]"
            print(f"{kind} [session_id={session_id}] {resp.text}")
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
        channel.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch STT client (single chunk)")
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
        vad_mode=args.vad_mode,
        attributes=attr_dict,
        require_token=args.require_token,
        language=args.language,
        task=args.task,
        decode_profile=args.decode_profile,
        vad_silence=args.vad_silence,
        vad_threshold=args.vad_threshold,
    )


if __name__ == "__main__":
    main()
