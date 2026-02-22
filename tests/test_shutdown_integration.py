import os
import random
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _maybe_skip_shutdown_integration() -> None:
    flag = os.getenv("STT_RUN_SHUTDOWN_INTEGRATION", "").strip().lower()
    if flag not in {"1", "true", "yes", "on"}:
        pytest.skip(
            "Shutdown integration tests are disabled. "
            "Set STT_RUN_SHUTDOWN_INTEGRATION=1 to enable."
        )


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


def _wait_for_health(health_url: str, timeout_sec: float = 90.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            response = requests.get(health_url, timeout=1)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1.0)
    return False


def _start_server() -> tuple[subprocess.Popen, str]:
    grpc_port = _pick_port(50081)
    http_port = _pick_port(18081)
    if http_port == grpc_port:
        http_port = _pick_port(18082)
    health_url = f"http://localhost:{http_port}/health"

    cmd = [
        sys.executable,
        "-m",
        "stt_server.main",
        "--model",
        "tiny",
        "--device",
        "cpu",
        "--port",
        str(grpc_port),
        "--metrics-port",
        str(http_port),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["STT_ALLOW_INSECURE_WS"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if not _wait_for_health(health_url):
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        stdout, stderr = proc.communicate(timeout=1)
        raise RuntimeError(
            "Shutdown integration server failed to start.\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )

    return proc, health_url


def _stop_and_collect(
    proc: subprocess.Popen, timeout_sec: float = 20.0
) -> tuple[int, str, str]:
    try:
        proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    stdout, stderr = proc.communicate(timeout=1)
    return proc.returncode, stdout, stderr


def test_shutdown_integration_single_sigterm() -> None:
    """Real server should stop cleanly on a single SIGTERM."""
    _maybe_skip_shutdown_integration()
    proc, _health = _start_server()
    proc.send_signal(signal.SIGTERM)
    return_code, _stdout, stderr = _stop_and_collect(proc, timeout_sec=25.0)
    assert return_code in {0, -signal.SIGTERM}
    assert "Graceful shutdown started" in stderr


def test_shutdown_integration_double_sigterm_exits_quickly() -> None:
    """Repeated SIGTERM should never hang shutdown."""
    _maybe_skip_shutdown_integration()
    proc, _health = _start_server()

    started = time.monotonic()
    proc.send_signal(signal.SIGTERM)
    time.sleep(0.2)
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)

    return_code, _stdout, stderr = _stop_and_collect(proc, timeout_sec=10.0)
    elapsed = time.monotonic() - started

    assert return_code in {0, -signal.SIGTERM}
    assert elapsed < 10.0
    assert "Received signal" in stderr
