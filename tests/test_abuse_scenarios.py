"""Optional abuse/load scenario tests (opt-in via STT_RUN_ABUSE_TESTS)."""

import copy
import os
import random
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import grpc
import pytest
import requests
import yaml

from gen.stt.python.v1 import stt_pb2, stt_pb2_grpc

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _pick_port(default: int) -> int:
    for _ in range(40):
        candidate = random.randint(20000, 40000)
        if not _port_in_use(candidate):
            return candidate
    return default


def _start_temp_server(config: dict) -> dict:
    grpc_port = _pick_port(50051)
    http_port = _pick_port(8000)
    if http_port == grpc_port:
        http_port = _pick_port(8000)

    token = "test-observability-token"
    config_copy = copy.deepcopy(config)
    server_cfg = config_copy.setdefault("server", {})
    server_cfg["port"] = grpc_port
    server_cfg["metrics_port"] = http_port

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(config_copy, fh)
        config_path = fh.name

    cmd = [
        sys.executable,
        "-m",
        "stt_server.main",
        "--config",
        config_path,
        "--model",
        "tiny",
        "--device",
        "cpu",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["STT_OBSERVABILITY_TOKEN"] = token

    proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env)

    health_url = f"http://localhost:{http_port}/health"
    health_headers = {"authorization": f"Bearer {token}"}
    for _ in range(60):
        try:
            if (
                requests.get(health_url, headers=health_headers, timeout=1).status_code
                == 200
            ):
                break
        except requests.exceptions.RequestException:
            time.sleep(1)
    else:
        proc.terminate()
        raise RuntimeError("Server failed to start within 60 seconds.")

    return {
        "proc": proc,
        "grpc_target": f"localhost:{grpc_port}",
        "http_base_url": f"http://localhost:{http_port}",
        "token": token,
        "config_path": config_path,
    }


def _stop_temp_server(server: dict) -> None:
    proc = server.get("proc")
    if proc is not None:
        proc.terminate()
        proc.wait()
    config_path = server.get("config_path")
    if config_path:
        try:
            Path(config_path).unlink(missing_ok=True)
        except OSError:
            pass


@pytest.fixture(scope="module")
def abuse_server():
    """Start a temporary server with tighter limits for abuse scenarios."""
    if os.getenv("STT_RUN_ABUSE_TESTS", "").strip().lower() not in {"1", "true", "yes"}:
        pytest.skip("Abuse/load scenario tests disabled (set STT_RUN_ABUSE_TESTS=1).")

    config = {
        "server": {
            "create_session_rps": 5.0,
            "create_session_burst": 5.0,
            "max_sessions_per_api_key": 100,
            "max_sessions_per_ip": 100,
            "max_audio_seconds_per_session": 1.0,
            "max_audio_bytes_per_sec": 8000,
            "max_audio_bytes_per_sec_burst": 8000,
            "http_rate_limit_rps": 50.0,
            "http_rate_limit_burst": 100.0,
        },
        "logging": {"level": "INFO"},
    }

    server = _start_temp_server(config)
    try:
        yield server
    finally:
        _stop_temp_server(server)


@pytest.fixture(scope="module")
def abuse_backpressure_server():
    """Start a temporary server configured to surface backpressure metrics."""
    if os.getenv("STT_RUN_ABUSE_TESTS", "").strip().lower() not in {"1", "true", "yes"}:
        pytest.skip("Abuse/load scenario tests disabled (set STT_RUN_ABUSE_TESTS=1).")

    config = {
        "server": {
            "max_audio_bytes_per_sec": 0,
            "max_audio_bytes_per_sec_burst": 0,
            "max_buffer_sec": 0.5,
            "max_total_buffer_bytes": 4096,
            "max_pending_decodes_per_stream": 1,
            "max_pending_decodes_global": 4,
            "partial_decode_interval_sec": 0.1,
            "partial_decode_window_sec": 0.5,
            "buffer_overlap_sec": 0.0,
            "max_chunk_ms": 500,
        },
        "safety": {"speech_rms_threshold": 0.0},
        "logging": {"level": "INFO"},
    }

    server = _start_temp_server(config)
    try:
        yield server
    finally:
        _stop_temp_server(server)


def _system_metrics(http_base_url: str, token: str) -> dict:
    response = requests.get(
        f"{http_base_url}/system",
        headers={"authorization": f"Bearer {token}"},
        timeout=2,
    )
    assert response.status_code == 200
    return response.json()


def _metrics_snapshot(http_base_url: str, token: str) -> dict:
    response = requests.get(
        f"{http_base_url}/metrics.json",
        headers={"authorization": f"Bearer {token}"},
        timeout=2,
    )
    assert response.status_code == 200
    return response.json()


def test_session_storm_is_rejected_gracefully(abuse_server):
    """CreateSession storm should be rejected without crashing the server."""
    grpc_target = abuse_server["grpc_target"]
    http_base = abuse_server["http_base_url"]
    token = abuse_server["token"]

    before = _system_metrics(http_base, token)

    failures = 0
    with grpc.insecure_channel(grpc_target) as channel:
        stub = stt_pb2_grpc.STTBackendStub(channel)
        for idx in range(20):
            request = stt_pb2.SessionRequest(
                session_id=f"storm-{idx}",
                attributes={"api_key": "storm"},
                vad_threshold_override=0.0,
            )
            try:
                stub.CreateSession(request)
            except grpc.RpcError as exc:
                if exc.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                    failures += 1

    assert failures > 0, "Expected some CreateSession rejections under storm load."

    after = _system_metrics(http_base, token)
    before_threads = before.get("process", {}).get("threads")
    after_threads = after.get("process", {}).get("threads")
    if before_threads is not None and after_threads is not None:
        assert after_threads <= before_threads + 10

    health = requests.get(
        f"{http_base}/health",
        headers={"authorization": f"Bearer {token}"},
        timeout=2,
    )
    assert health.status_code == 200


def test_long_stream_is_rejected_gracefully(abuse_server):
    """Long or fast streams should be rejected with RESOURCE_EXHAUSTED."""
    grpc_target = abuse_server["grpc_target"]

    with grpc.insecure_channel(grpc_target) as channel:
        stub = stt_pb2_grpc.STTBackendStub(channel)
        session_id = f"long-{int(time.time() * 1000)}"
        stub.CreateSession(
            stt_pb2.SessionRequest(session_id=session_id, vad_threshold_override=0.0)
        )

        def chunks():
            sample_rate = 16000
            payload = b"\x00\x00" * sample_rate
            for _ in range(3):
                yield stt_pb2.AudioChunk(
                    session_id=session_id,
                    sample_rate=sample_rate,
                    pcm16=payload,
                )
            yield stt_pb2.AudioChunk(session_id=session_id, is_final=True)

        with pytest.raises(grpc.RpcError) as exc_info:
            list(stub.StreamingRecognize(chunks()))

        assert exc_info.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED


def test_noise_stream_keeps_server_healthy(abuse_server):
    """Noise input should not crash the server or leak threads quickly."""
    grpc_target = abuse_server["grpc_target"]
    http_base = abuse_server["http_base_url"]
    token = abuse_server["token"]

    before = _system_metrics(http_base, token)
    before_threads = before.get("process", {}).get("threads")

    with grpc.insecure_channel(grpc_target) as channel:
        stub = stt_pb2_grpc.STTBackendStub(channel)
        session_id = f"noise-{int(time.time() * 1000)}"
        stub.CreateSession(
            stt_pb2.SessionRequest(session_id=session_id, vad_threshold_override=0.0)
        )

        sample_rate = 8000
        chunk_ms = 100
        samples = int(sample_rate * (chunk_ms / 1000.0))

        def noise_chunk() -> bytes:
            return b"".join(
                random.randint(-32768, 32767).to_bytes(2, "little", signed=True)
                for _ in range(samples)
            )

        def chunks():
            for _ in range(5):
                yield stt_pb2.AudioChunk(
                    session_id=session_id,
                    sample_rate=sample_rate,
                    pcm16=noise_chunk(),
                )
            yield stt_pb2.AudioChunk(session_id=session_id, is_final=True)

        try:
            list(stub.StreamingRecognize(chunks()))
        except grpc.RpcError as exc:
            assert exc.code() in {
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                grpc.StatusCode.DEADLINE_EXCEEDED,
            }

    after = _system_metrics(http_base, token)
    after_threads = after.get("process", {}).get("threads")
    if before_threads is not None and after_threads is not None:
        assert after_threads <= before_threads + 20


def test_backpressure_metrics_are_recorded(abuse_backpressure_server):
    """Backpressure paths should surface metrics under load."""
    grpc_target = abuse_backpressure_server["grpc_target"]
    http_base = abuse_backpressure_server["http_base_url"]
    token = abuse_backpressure_server["token"]

    stop_event = threading.Event()
    maxima = {"buffer_bytes_total": 0.0, "decode_pending": 0.0}

    def poll_metrics() -> None:
        while not stop_event.is_set():
            try:
                metrics = _metrics_snapshot(http_base, token)
            except requests.RequestException:
                time.sleep(0.05)
                continue
            for key in ("buffer_bytes_total", "decode_pending"):
                value = metrics.get(key)
                if value is None:
                    continue
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                maxima[key] = max(maxima[key], value)
            time.sleep(0.05)

    poller = threading.Thread(target=poll_metrics, daemon=True)
    poller.start()

    with grpc.insecure_channel(grpc_target) as channel:
        stub = stt_pb2_grpc.STTBackendStub(channel)
        session_id = f"bp-{int(time.time() * 1000)}"
        stub.CreateSession(
            stt_pb2.SessionRequest(
                session_id=session_id,
                attributes={"partial": "true"},
                vad_threshold_override=0.0,
            )
        )

        sample_rate = 16000
        chunk_ms = 100
        samples = int(sample_rate * (chunk_ms / 1000.0))
        payload = b"\x00\x00" * samples

        def chunks():
            for _ in range(30):
                yield stt_pb2.AudioChunk(
                    session_id=session_id,
                    sample_rate=sample_rate,
                    pcm16=payload,
                )
                time.sleep(0.01)
            yield stt_pb2.AudioChunk(session_id=session_id, is_final=True)

        try:
            list(stub.StreamingRecognize(chunks()))
        except grpc.RpcError:
            pass

    stop_event.set()
    poller.join(timeout=2)

    metrics = _metrics_snapshot(http_base, token)
    assert maxima["buffer_bytes_total"] >= 4096
    assert maxima["decode_pending"] >= 1
    assert metrics.get("partial_drop_count", 0) > 0
