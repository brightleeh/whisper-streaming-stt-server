import os
import subprocess
import sys
import time
from pathlib import Path

import grpc
import pytest
import requests

from gen.stt.python.v1 import stt_pb2, stt_pb2_grpc

# Set project root path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def grpc_server():
    """Ensure the gRPC server is running for tests. Starts a temporary one if needed."""
    health_url = "http://localhost:8000/health"

    # 1. Check if a server is already running
    try:
        if requests.get(health_url, timeout=1).status_code == 200:
            yield  # Use existing server
            return
    except requests.exceptions.RequestException:
        pass  # Server not running, proceed to start

    # 2. Start a temporary server
    print("\nStarting temporary gRPC server (tiny model)...")
    cmd = [
        sys.executable,
        "-m",
        "stt_server.main",
        "--model",
        "tiny",  # Use tiny model for faster load times in tests
        "--device",
        "cpu",
    ]
    # Ensure PYTHONPATH includes the project root
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT

    proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env)

    # 3. Wait for the server to become healthy
    for _ in range(60):  # Wait up to 60 seconds
        try:
            if requests.get(health_url, timeout=1).status_code == 200:
                break
        except requests.exceptions.RequestException:
            time.sleep(1)
    else:
        proc.terminate()
        raise RuntimeError("Server failed to start within 60 seconds.")

    yield proc

    # 4. Cleanup
    print("\nStopping temporary gRPC server...")
    proc.terminate()
    proc.wait()


def test_audio_asset_exists():
    """Check if the sample audio file exists."""
    asset_path = os.path.join(PROJECT_ROOT, "stt_client/assets/hello.wav")
    assert os.path.exists(asset_path), f"File not found: {asset_path}"
    print(f"\n[PASS] Audio asset found: {asset_path}")


def test_grpc_stubs_generated():
    """Check if gRPC stub codes are generated. If not, try to generate them."""
    rel_stubs_path = os.path.join("gen", "stt", "python", "v1")
    abs_stubs_path = os.path.join(PROJECT_ROOT, rel_stubs_path)
    pb2_path = os.path.join(abs_stubs_path, "stt_pb2.py")
    grpc_path = os.path.join(abs_stubs_path, "stt_pb2_grpc.py")

    # 1. Check if files already exist
    if os.path.exists(pb2_path) and os.path.exists(grpc_path):
        print("\n[PASS] gRPC stubs already exist.")
        return  # Stubs are already there, test passes

    # 2. If not, try to generate them based on README.md
    print("\nAttempting to generate missing gRPC stubs...")

    # Ensure directory structure exists
    os.makedirs(abs_stubs_path, exist_ok=True)
    Path(PROJECT_ROOT, "gen", "__init__.py").touch()
    Path(PROJECT_ROOT, "gen", "stt", "__init__.py").touch()
    Path(PROJECT_ROOT, "gen", "stt", "python", "__init__.py").touch()
    Path(abs_stubs_path, "__init__.py").touch()

    # The protoc command from README
    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        "-I",
        "proto",
        f"--python_out={rel_stubs_path}",
        f"--grpc_python_out={rel_stubs_path}",
        f"--mypy_out={rel_stubs_path}",
        "proto/stt.proto",
    ]

    result = subprocess.run(
        cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30
    )

    # 3. Assert that the files exist *after* generation
    assert result.returncode == 0, f"gRPC stub generation failed: {result.stderr}"
    assert os.path.exists(
        pb2_path
    ), "stt_pb2.py is still missing after attempting generation."
    assert os.path.exists(
        grpc_path
    ), "stt_pb2_grpc.py is still missing after attempting generation."
    print("[PASS] gRPC stubs generated successfully.")


def test_server_health_check(grpc_server):
    """Call /health endpoint to check response when server is running."""
    url = "http://localhost:8000/health"
    response = requests.get(url, timeout=2)
    assert response.status_code == 200, f"Server is unhealthy: {response.status_code}"
    print(f"\n[PASS] Server /health check returned status {response.status_code}.")


def test_server_metrics_check(grpc_server):
    """Call /metrics endpoint to check response is valid JSON."""
    url = "http://localhost:8000/metrics"
    response = requests.get(url, timeout=2)
    assert (
        response.status_code == 200
    ), f"Metrics endpoint failed: {response.status_code}"
    assert isinstance(response.json(), dict), "Metrics response is not valid JSON"
    print("\n[PASS] Server /metrics check returned valid JSON.")


def test_client_integration(grpc_server):
    """Run the sample client to verify end-to-end speech recognition."""
    # Server is guaranteed to be running by the fixture

    # Run the client as a subprocess
    # We use the default file client which should process hello.wav
    cmd = [
        sys.executable,
        "-m",
        "stt_client.realtime.file",
        "-c",
        "stt_client/config/file.yaml",
        "--no-realtime",  # Process as fast as possible
    ]

    result = subprocess.run(
        cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30
    )

    assert result.returncode == 0, f"Client failed: {result.stderr}"
    # The sample audio contains "안녕하세요"
    assert "안녕하세요" in result.stdout, "Expected transcript not found in output"
    print("\n[PASS] Client integration test successful. Output contained '안녕하세요'.")
    print(f"Client stdout:\n---\n{result.stdout.strip()}\n---")
