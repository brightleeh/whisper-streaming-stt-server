"""Long-running abuse/profiling helper (manual execution)."""

from __future__ import annotations

import argparse
import os
import random
import time

import grpc
import requests

from gen.stt.python.v1 import stt_pb2, stt_pb2_grpc


def _system_metrics(http_base: str, token: str | None) -> dict:
    headers = {}
    if token:
        headers["authorization"] = f"Bearer {token}"
    response = requests.get(f"{http_base}/system", headers=headers, timeout=5)
    response.raise_for_status()
    return response.json()


def _pcm_silence(sample_rate: int, chunk_ms: int) -> bytes:
    samples = int(sample_rate * (chunk_ms / 1000.0))
    return b"\x00\x00" * samples


def _pcm_noise(sample_rate: int, chunk_ms: int) -> bytes:
    samples = int(sample_rate * (chunk_ms / 1000.0))
    values = [random.randint(-32768, 32767) for _ in range(samples)]
    return b"".join(int(v).to_bytes(2, "little", signed=True) for v in values)


def _run_stream(
    stub: stt_pb2_grpc.STTBackendStub,
    session_id: str,
    sample_rate: int,
    chunk_ms: int,
    duration_sec: int,
    mode: str,
) -> None:
    if mode == "noise":
        payload = _pcm_noise(sample_rate, chunk_ms)
    else:
        payload = _pcm_silence(sample_rate, chunk_ms)

    def chunks():
        end_at = time.time() + duration_sec
        while time.time() < end_at:
            yield stt_pb2.AudioChunk(
                session_id=session_id,
                sample_rate=sample_rate,
                pcm16=payload,
            )
        yield stt_pb2.AudioChunk(session_id=session_id, is_final=True)

    for _ in stub.StreamingRecognize(chunks()):
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Long-running abuse/profiling helper")
    parser.add_argument("--server", default="localhost:50051")
    parser.add_argument("--http", default="http://localhost:8000")
    parser.add_argument("--duration-sec", type=int, default=300)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--mode", choices=("silence", "noise"), default="silence")
    parser.add_argument("--token", default=os.getenv("STT_OBSERVABILITY_TOKEN", ""))
    parser.add_argument("--api-key", default="")
    args = parser.parse_args()

    token = args.token.strip() or None
    session_id = f"longrun-{int(time.time() * 1000)}"

    before = _system_metrics(args.http, token)
    before_threads = before.get("process", {}).get("threads")
    before_rss = before.get("process", {}).get("rss_bytes")

    with grpc.insecure_channel(args.server) as channel:
        stub = stt_pb2_grpc.STTBackendStub(channel)
        attrs = {"api_key": args.api_key} if args.api_key else {}
        stub.CreateSession(
            stt_pb2.SessionRequest(
                session_id=session_id,
                attributes=attrs,
                vad_threshold_override=0.0,
            )
        )
        _run_stream(
            stub,
            session_id,
            args.sample_rate,
            args.chunk_ms,
            args.duration_sec,
            args.mode,
        )

    after = _system_metrics(args.http, token)
    after_threads = after.get("process", {}).get("threads")
    after_rss = after.get("process", {}).get("rss_bytes")

    print("Long-run summary:")
    print(f"  duration_sec: {args.duration_sec}")
    print(f"  mode: {args.mode}")
    if before_threads is not None and after_threads is not None:
        print(f"  threads: {before_threads} -> {after_threads}")
    if before_rss is not None and after_rss is not None:
        delta = after_rss - before_rss
        print(f"  rss_bytes: {before_rss} -> {after_rss} (delta={delta})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
