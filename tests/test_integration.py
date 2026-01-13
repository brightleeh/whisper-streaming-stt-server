import os
import subprocess
import sys
import tempfile
import threading
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
    # The sample audio contains "안녕"
    assert "안녕" in result.stdout, "Expected transcript not found in output"
    print("\n[PASS] Client integration test successful. Output contained '안녕'.")
    print(f"Client stdout:\n---\n{result.stdout.strip()}\n---")


def test_error_missing_session_id(grpc_server):
    """Test for ERR1001: CreateSession with a missing session_id."""
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = stt_pb2_grpc.STTBackendStub(channel)
        request = stt_pb2.SessionRequest(session_id="")

        with pytest.raises(grpc.RpcError) as e:
            stub.CreateSession(request)

        assert e.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert "ERR1001" in str(e.value.details())
        print("\n[PASS] ERR1001 (missing session_id) test successful.")


def test_error_duplicate_session_id(grpc_server):
    """Test for ERR1002: CreateSession with a duplicate session_id."""
    session_id = f"test-duplicate-{time.time()}"
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = stt_pb2_grpc.STTBackendStub(channel)

        # First call should succeed
        request1 = stt_pb2.SessionRequest(session_id=session_id)
        stub.CreateSession(request1)

        # Second call with the same ID should fail
        request2 = stt_pb2.SessionRequest(session_id=session_id)
        with pytest.raises(grpc.RpcError) as e:
            stub.CreateSession(request2)

        assert e.value.code() == grpc.StatusCode.ALREADY_EXISTS
        assert "ERR1002" in str(e.value.details())
        print("\n[PASS] ERR1002 (duplicate session_id) test successful.")


def test_error_negative_vad_threshold(grpc_server):
    """Test for ERR1003: CreateSession with a negative vad_threshold."""
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = stt_pb2_grpc.STTBackendStub(channel)
        request = stt_pb2.SessionRequest(
            session_id=f"test-vad-threshold-{time.time()}", vad_threshold=-0.1
        )

        with pytest.raises(grpc.RpcError) as e:
            stub.CreateSession(request)

        assert e.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert "ERR1003" in str(e.value.details())
        print("\n[PASS] ERR1003 (negative vad_threshold) test successful.")


def test_error_unknown_session_id_in_stream(grpc_server):
    """Test for ERR1004: StreamingRecognize with an unknown session_id."""
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = stt_pb2_grpc.STTBackendStub(channel)

        def audio_chunks():
            yield stt_pb2.AudioChunk(
                session_id="non-existent-session-id", pcm16=b"\x00\x01"
            )
            threading.Event().wait()

        with pytest.raises(grpc.RpcError) as e:
            # We need to consume the iterator to trigger the error
            list(stub.StreamingRecognize(audio_chunks()))

        assert e.value.code() == grpc.StatusCode.UNAUTHENTICATED
        assert "ERR1004" in str(e.value.details())
        print("\n[PASS] ERR1004 (unknown session_id) test successful.")


def test_error_invalid_session_token(grpc_server):
    """Test for ERR1005: StreamingRecognize with an invalid session token."""
    session_id = f"test-token-{time.time()}"
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = stt_pb2_grpc.STTBackendStub(channel)

        # Create a session that requires a token
        create_request = stt_pb2.SessionRequest(
            session_id=session_id, require_token=True
        )
        response = stub.CreateSession(create_request)
        assert response.token_required is True
        assert response.token != ""

        # Now, try to stream with an invalid token
        def audio_chunks():
            yield stt_pb2.AudioChunk(
                session_id=session_id, session_token="invalid-token", pcm16=b"\x00\x01"
            )
            threading.Event().wait()

        with pytest.raises(grpc.RpcError) as e:
            list(stub.StreamingRecognize(audio_chunks()))

        assert e.value.code() == grpc.StatusCode.PERMISSION_DENIED
        assert "ERR1005" in str(e.value.details())
        print("\n[PASS] ERR1005 (invalid session token) test successful.")


@pytest.fixture(scope="function")
def short_timeout_grpc_server():
    """Starts a gRPC server with a 2-second session timeout for testing."""
    health_url = "http://localhost:8001/health"
    grpc_port = 50052
    proc = None
    config_path = None

    # Create a temporary config file
    config_content = "server:\n  session_timeout_sec: 2\n"
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".yaml"
    ) as tmp_config:
        config_path = tmp_config.name
        tmp_config.write(config_content)

    try:
        # Start a temporary server with the custom config
        print("\nStarting temporary gRPC server (2s timeout)...")
        cmd = [
            sys.executable,
            "-m",
            "stt_server.main",
            "--model",
            "tiny",
            "--device",
            "cpu",
            "--config",
            config_path,
            "--port",
            str(grpc_port),
            "--metrics-port",
            "8001",
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = PROJECT_ROOT
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env)

        # Wait for the server to become healthy
        for _ in range(60):
            try:
                if requests.get(health_url, timeout=1).status_code == 200:
                    break
            except requests.exceptions.RequestException:
                time.sleep(1)
        else:
            raise RuntimeError("Short-timeout server failed to start.")

        yield grpc_port  # Provide the port to the test

    finally:
        # Cleanup
        if proc:
            print("\nStopping temporary gRPC server (2s timeout)...")
            proc.terminate()
            proc.wait()
        if config_path and os.path.exists(config_path):
            os.unlink(config_path)


def test_error_session_timeout(short_timeout_grpc_server):
    """Test for ERR1006: Session timeout due to inactivity."""
    grpc_port = short_timeout_grpc_server
    session_id = f"test-timeout-{time.time()}"

    with grpc.insecure_channel(f"localhost:{grpc_port}") as channel:
        stub = stt_pb2_grpc.STTBackendStub(channel)

        # Create a session
        stub.CreateSession(stt_pb2.SessionRequest(session_id=session_id))

        def audio_chunks_generator():
            # Send one empty chunk to start the stream and the inactivity timer
            yield stt_pb2.AudioChunk(session_id=session_id)
            # Wait longer than the 2-second timeout
            time.sleep(3)
            # The server should have closed the stream. This next yield will fail.
            yield stt_pb2.AudioChunk(session_id=session_id, pcm16=b"\x00\x01")

        with pytest.raises(grpc.RpcError) as e:
            # Consume the generator to trigger the RPC calls and see the error
            list(stub.StreamingRecognize(audio_chunks_generator()))

        assert e.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
        assert "ERR1006" in str(e.value.details())
        print("\n[PASS] ERR1006 (session timeout) test successful.")
