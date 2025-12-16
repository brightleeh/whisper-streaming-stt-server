import argparse
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Iterator, Optional, Union

import grpc
import sounddevice as sd

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
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return self

    def _capture_loop(self) -> None:
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
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        # Ensure the consumer wakes up even if capture thread exited early.
        self.queue.put(None)

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self.samples_sent / float(self.sample_rate)


def run(
    target: str,
    sample_rate: int,
    chunk_ms: int,
    input_device: Optional[str],
    report_metrics: bool,
    epd_mode: str,
    require_token: bool,
    language: str,
    task: str,
    decode_profile: str,
    epd_silence: Optional[float],
    epd_threshold: Optional[float],
) -> None:
    mic = MicrophoneStream(
        sample_rate=sample_rate, chunk_ms=chunk_ms, device=input_device
    ).start()
    channel = grpc.insecure_channel(target)
    stub = stt_pb2_grpc.STTBackendStub(channel)
    session_id = str(int(time.time() * 1000))
    attributes: Dict[str, str] = {}
    if epd_silence is not None:
        attributes["epd_silence"] = str(epd_silence)
    if epd_threshold is not None:
        attributes["epd_threshold"] = str(epd_threshold)

    session_resp = stub.CreateSession(
        stt_pb2.SessionRequest(
            session_id=session_id,
            attributes=attributes,
            epd_mode=(
                stt_pb2.EPD_AUTO_END
                if epd_mode.lower() == "auto"
                else stt_pb2.EPD_CONTINUE
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

    start = time.perf_counter()
    request_iter = mic.request_stream(session_id, session_token)
    responses = stub.StreamingRecognize(
        request_iter,
        metadata=[("session-id", session_id)],
    )
    print(
        f"[STREAM] session_id={session_id} microphone streaming at {sample_rate} Hz "
        f"({chunk_ms} ms chunks). Press Ctrl+C to stop."
    )

    try:
        for resp in responses:
            kind = "[FINAL]" if resp.is_final else "[PARTIAL]"
            print(
                f"{kind} [session_id={session_id}] "
                f"[{resp.start_sec}][{resp.end_sec}] {resp.text}"
            )
        print(f"[STREAM] session_id={session_id} completed normally")
    except grpc.RpcError as exc:
        print(
            f"[STREAM] session_id={session_id} terminated by RPC error: {exc}",
            file=sys.stderr,
        )
        raise
    except KeyboardInterrupt:
        print(f"\n[STREAM] session_id={session_id} interrupted by user")
    except Exception as exc:
        print(
            f"[STREAM] session_id={session_id} terminated by client error: {exc}",
            file=sys.stderr,
        )
        raise
    finally:
        mic.stop()
        channel.close()
        total_wall = time.perf_counter() - start
        if report_metrics:
            audio_duration = mic.duration_seconds
            rtf = total_wall / audio_duration if audio_duration > 0 else float("inf")
            print(
                f"[METRIC] audio_duration={audio_duration:.2f}s "
                f"wall_clock={total_wall:.2f}s real_time_factor={rtf:.2f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Streaming STT client (macOS microphone)"
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
        "--epd-mode",
        choices=("continue", "auto"),
        default="continue",
        help="EPD mode (continue or auto); default: %(default)s",
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
    args = parser.parse_args()

    run(
        target=args.server,
        sample_rate=args.sample_rate,
        chunk_ms=args.chunk_ms,
        input_device=args.device,
        report_metrics=args.metrics,
        epd_mode=args.epd_mode,
        require_token=args.require_token,
        language=args.language,
        task=args.task,
        decode_profile=args.decode_profile,
        epd_silence=args.epd_silence,
        epd_threshold=args.epd_threshold,
    )


if __name__ == "__main__":
    main()
