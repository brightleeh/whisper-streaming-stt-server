import os
import random
import socket
import subprocess
import sys
import time

import pytest
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def _start_server(model_backend: str, device: str):
    grpc_port = _pick_port(50051)
    http_port = _pick_port(8000)
    if http_port == grpc_port:
        http_port = _pick_port(8000)
    health_url = f"http://localhost:{http_port}/health"
    cmd = [
        sys.executable,
        "-m",
        "stt_server.main",
        "--model",
        "tiny",
        "--model-backend",
        model_backend,
        "--device",
        device,
        "--port",
        str(grpc_port),
        "--metrics-port",
        str(http_port),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT
    proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env)
    for _ in range(60):
        try:
            if requests.get(health_url, timeout=1).status_code == 200:
                return proc, f"localhost:{grpc_port}"
        except requests.exceptions.RequestException:
            time.sleep(1)
    proc.terminate()
    raise RuntimeError("Backend test server failed to start within 60 seconds.")


def _run_client(grpc_target: str) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "stt_client.realtime.file",
        "-c",
        "stt_client/config/file.yaml",
        "--server",
        grpc_target,
    ]
    return subprocess.run(
        cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60
    )


def _maybe_skip_backend_tests():
    if os.getenv("STT_SKIP_BACKEND_INTEGRATION", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        pytest.skip(
            "Backend integration tests skipped via STT_SKIP_BACKEND_INTEGRATION."
        )


def test_faster_whisper_cpu_integration():
    """Run client against faster_whisper backend on CPU."""
    _maybe_skip_backend_tests()
    proc = None
    try:
        proc, grpc_target = _start_server("faster_whisper", "cpu")
        result = _run_client(grpc_target)
        assert result.returncode == 0, f"Client failed: {result.stderr}"
        assert "안녕" in result.stdout
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait()


def test_torch_whisper_cpu_integration():
    """Run client against torch_whisper backend on CPU."""
    _maybe_skip_backend_tests()
    pytest.importorskip("whisper")
    proc = None
    try:
        proc, grpc_target = _start_server("torch_whisper", "cpu")
        result = _run_client(grpc_target)
        assert result.returncode == 0, f"Client failed: {result.stderr}"
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait()
