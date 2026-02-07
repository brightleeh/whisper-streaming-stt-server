import contextlib
import os
import random
import socket
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import grpc
import pytest
import requests

from gen.stt.python.v1 import stt_pb2, stt_pb2_grpc

# Set project root path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _pick_port(env_name: str, default: int) -> int:
    env_value = os.getenv(env_name)
    if env_value:
        try:
            return int(env_value)
        except ValueError:
            pass
    for _ in range(40):
        candidate = random.randint(20000, 40000)
        if not _port_in_use(candidate):
            return candidate
    return default


def _load_wav_pcm16(path: str) -> tuple[bytes, int]:
    with wave.open(path, "rb") as wf:
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
    if sample_width != 2 or channels != 1:
        raise ValueError(f"Unsupported WAV format: {channels}ch {sample_width * 8}-bit")
    return frames, sample_rate


def _chunk_pcm(pcm16: bytes, sample_rate: int, chunk_ms: int) -> list[bytes]:
    chunk_bytes = max(1, int(sample_rate * 2 * (chunk_ms / 1000.0)))
    return [
        pcm16[offset : offset + chunk_bytes]
        for offset in range(0, len(pcm16), chunk_bytes)
        if pcm16[offset : offset + chunk_bytes]
    ]


@contextlib.contextmanager
def _temp_grpc_server(config_content: str):
    if os.getenv("STT_SKIP_INTEGRATION", "").strip().lower() in {"1", "true", "yes"}:
        pytest.skip("Integration tests skipped via STT_SKIP_INTEGRATION.")
    grpc_port = _pick_port("STT_TEST_GRPC_PORT", 50053)
    http_port = _pick_port("STT_TEST_HTTP_PORT", 8002)
    if http_port == grpc_port:
        http_port = _pick_port("STT_TEST_HTTP_PORT", 8002)
    health_url = f"http://localhost:{http_port}/health"
    proc = None
    config_path = None

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".yaml"
    ) as tmp_config:
        config_path = tmp_config.name
        tmp_config.write(config_content)

    try:
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
            str(http_port),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = PROJECT_ROOT
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env)

        for _ in range(60):
            try:
                if requests.get(health_url, timeout=1).status_code == 200:
                    break
            except requests.exceptions.RequestException:
                time.sleep(1)
        else:
            raise RuntimeError("Temporary server failed to start.")

        yield {
            "proc": proc,
            "grpc_target": f"localhost:{grpc_port}",
            "http_base_url": f"http://localhost:{http_port}",
        }
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait()
        if config_path and os.path.exists(config_path):
            os.unlink(config_path)


@pytest.fixture(scope="module")
def grpc_server():
    """Ensure the gRPC server is running for tests. Starts a temporary one if needed."""
    if os.getenv("STT_SKIP_INTEGRATION", "").strip().lower() in {"1", "true", "yes"}:
        pytest.skip("Integration tests skipped via STT_SKIP_INTEGRATION.")
    grpc_port = _pick_port("STT_TEST_GRPC_PORT", 50051)
    http_port = _pick_port("STT_TEST_HTTP_PORT", 8000)
    if http_port == grpc_port:
        http_port = _pick_port("STT_TEST_HTTP_PORT", 8000)
    health_url = f"http://localhost:{http_port}/health"

    # 1. Use existing server if healthy
    try:
        if requests.get(health_url, timeout=1).status_code == 200:
            yield {
                "proc": None,
                "grpc_target": f"localhost:{grpc_port}",
                "http_base_url": f"http://localhost:{http_port}",
            }
            return
    except requests.exceptions.RequestException:
        if os.getenv("STT_REQUIRE_EXISTING", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }:
            pytest.skip("Existing server not reachable; STT_REQUIRE_EXISTING set.")

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
        "--port",
        str(grpc_port),
        "--metrics-port",
        str(http_port),
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

    yield {
        "proc": proc,
        "grpc_target": f"localhost:{grpc_port}",
        "http_base_url": f"http://localhost:{http_port}",
    }

    # 4. Cleanup
    if proc is not None:
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
    url = f"{grpc_server['http_base_url']}/health"
    response = requests.get(url, timeout=2)
    assert response.status_code == 200, f"Server is unhealthy: {response.status_code}"
    print(f"\n[PASS] Server /health check returned status {response.status_code}.")


def test_server_metrics_json_check(grpc_server):
    """Call /metrics.json endpoint to check response is valid JSON."""
    url = f"{grpc_server['http_base_url']}/metrics.json"
    response = requests.get(url, timeout=2)
    assert (
        response.status_code == 200
    ), f"Metrics endpoint failed: {response.status_code}"
    assert isinstance(response.json(), dict), "Metrics response is not valid JSON"
    print("\n[PASS] Server /metrics.json check returned valid JSON.")


def test_server_metrics_text_check(grpc_server):
    """Call /metrics endpoint to check response is Prometheus text."""
    url = f"{grpc_server['http_base_url']}/metrics"
    response = requests.get(url, timeout=2)
    assert (
        response.status_code == 200
    ), f"Metrics endpoint failed: {response.status_code}"
    content_type = response.headers.get("content-type", "")
    assert "text/plain" in content_type
    assert "stt_" in response.text
    print("\n[PASS] Server /metrics check returned Prometheus text.")


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
        "--server",
        grpc_server["grpc_target"],
    ]
    if os.getenv("STT_TEST_NO_REALTIME", "").strip().lower() in {"1", "true", "yes"}:
        cmd.append("--no-realtime")  # Process as fast as possible

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
    with grpc.insecure_channel(grpc_server["grpc_target"]) as channel:
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
    with grpc.insecure_channel(grpc_server["grpc_target"]) as channel:
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
    with grpc.insecure_channel(grpc_server["grpc_target"]) as channel:
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
    with grpc.insecure_channel(grpc_server["grpc_target"]) as channel:
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
    with grpc.insecure_channel(grpc_server["grpc_target"]) as channel:
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
    if os.getenv("STT_SKIP_INTEGRATION", "").strip().lower() in {"1", "true", "yes"}:
        pytest.skip("Integration tests skipped via STT_SKIP_INTEGRATION.")
    grpc_port = _pick_port("STT_TEST_GRPC_PORT", 50052)
    http_port = _pick_port("STT_TEST_HTTP_PORT", 8001)
    if http_port == grpc_port:
        http_port = _pick_port("STT_TEST_HTTP_PORT", 8001)
    health_url = f"http://localhost:{http_port}/health"
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
            str(http_port),
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
        # Send one real chunk to start the stream and the inactivity timer
        yield stt_pb2.AudioChunk(
            session_id=session_id, pcm16=b"\x00\x01", sample_rate=16000
        )
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


def test_streaming_partial_and_final_results():
    """Stream audio and expect partials plus a final result."""
    config_content = (
        "server:\n"
        "  partial_decode_interval_sec: 0.2\n"
        "  partial_decode_window_sec: 2.0\n"
        "  session_timeout_sec: 30\n"
        "  max_pending_decodes_per_stream: 0\n"
        "  max_audio_bytes_per_sec: 0\n"
        "  max_audio_bytes_per_sec_burst: 0\n"
        "vad:\n"
        "  threshold: 0.1\n"
        "safety:\n"
        "  speech_rms_threshold: 0.0\n"
    )
    asset_path = os.path.join(PROJECT_ROOT, "stt_client/assets/hello.wav")
    pcm16, sample_rate = _load_wav_pcm16(asset_path)
    chunks = _chunk_pcm(pcm16, sample_rate, chunk_ms=200)

    with _temp_grpc_server(config_content) as server:
        with grpc.insecure_channel(server["grpc_target"]) as channel:
            stub = stt_pb2_grpc.STTBackendStub(channel)
            session_id = f"test-partial-{time.time()}"
            stub.CreateSession(
                stt_pb2.SessionRequest(
                    session_id=session_id,
                    attributes={"partial": "true"},
                    vad_mode=stt_pb2.VAD_CONTINUE,
                )
            )

            def audio_chunks():
                for idx, chunk in enumerate(chunks):
                    yield stt_pb2.AudioChunk(
                        session_id=session_id,
                        pcm16=chunk,
                        sample_rate=sample_rate,
                        is_final=(idx == len(chunks) - 1),
                    )
                    time.sleep(len(chunk) / (sample_rate * 2))

            results = list(stub.StreamingRecognize(audio_chunks()))
            partials = [result for result in results if not result.is_final]
            finals = [result for result in results if result.is_final]
            assert partials, "Expected at least one partial result"
            assert len(finals) == 1


def test_chunk_too_large_invalid_argument():
    """Send an oversized chunk and expect ERR1007."""
    config_content = "server:\n  max_chunk_ms: 20\n"
    with _temp_grpc_server(config_content) as server:
        with grpc.insecure_channel(server["grpc_target"]) as channel:
            stub = stt_pb2_grpc.STTBackendStub(channel)
            session_id = f"test-chunk-max-{time.time()}"
            stub.CreateSession(stt_pb2.SessionRequest(session_id=session_id))

            def audio_chunks():
                yield stt_pb2.AudioChunk(
                    session_id=session_id,
                    pcm16=b"\x00\x00" * 500,
                    sample_rate=16000,
                    is_final=True,
                )

            with pytest.raises(grpc.RpcError) as e:
                list(stub.StreamingRecognize(audio_chunks()))

            assert e.value.code() == grpc.StatusCode.INVALID_ARGUMENT
            assert "ERR1007" in str(e.value.details())


def test_stream_rate_limit_exceeded():
    """Exceed stream rate limit and expect ERR2003."""
    config_content = (
        "server:\n"
        "  max_audio_bytes_per_sec: 1000\n"
        "  max_audio_bytes_per_sec_burst: 1000\n"
    )
    with _temp_grpc_server(config_content) as server:
        with grpc.insecure_channel(server["grpc_target"]) as channel:
            stub = stt_pb2_grpc.STTBackendStub(channel)
            session_id = f"test-rate-limit-{time.time()}"
            stub.CreateSession(stt_pb2.SessionRequest(session_id=session_id))

            def audio_chunks():
                yield stt_pb2.AudioChunk(
                    session_id=session_id,
                    pcm16=b"\x00\x00" * 800,
                    sample_rate=16000,
                    is_final=True,
                )

            with pytest.raises(grpc.RpcError) as e:
                list(stub.StreamingRecognize(audio_chunks()))

            assert e.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
            assert "ERR2003" in str(e.value.details())


def test_buffer_limit_triggers_partial_drop_metric():
    """Hit buffer limits and expect partial drop metrics to increment."""
    config_content = (
        "server:\n"
        "  max_buffer_sec: 0.5\n"
        "  partial_decode_interval_sec: 0.1\n"
        "  partial_decode_window_sec: 1.0\n"
        "  max_pending_decodes_per_stream: 1\n"
        "  max_audio_bytes_per_sec: 0\n"
        "  max_audio_bytes_per_sec_burst: 0\n"
        "vad:\n"
        "  threshold: 0.1\n"
        "safety:\n"
        "  speech_rms_threshold: 0.0\n"
    )
    asset_path = os.path.join(PROJECT_ROOT, "stt_client/assets/hello.wav")
    pcm16, sample_rate = _load_wav_pcm16(asset_path)
    pcm16 = pcm16 * 6
    chunks = _chunk_pcm(pcm16, sample_rate, chunk_ms=500)

    with _temp_grpc_server(config_content) as server:
        with grpc.insecure_channel(server["grpc_target"]) as channel:
            stub = stt_pb2_grpc.STTBackendStub(channel)
            session_id = f"test-buffer-limit-{time.time()}"
            stub.CreateSession(
                stt_pb2.SessionRequest(
                    session_id=session_id,
                    attributes={"partial": "true"},
                    vad_mode=stt_pb2.VAD_CONTINUE,
                )
            )

            def audio_chunks():
                for idx, chunk in enumerate(chunks):
                    yield stt_pb2.AudioChunk(
                        session_id=session_id,
                        pcm16=chunk,
                        sample_rate=sample_rate,
                        is_final=(idx == len(chunks) - 1),
                    )

            list(stub.StreamingRecognize(audio_chunks()))

        metrics = requests.get(
            f"{server['http_base_url']}/metrics.json", timeout=2
        ).json()
        assert metrics.get("partial_drop_count", 0) > 0


def test_decode_timeout_returns_deadline_exceeded():
    """Force decode timeout and expect ERR2001."""
    config_content = (
        "server:\n"
        "  decode_timeout_sec: 0.001\n"
        "  max_chunk_ms: 10000\n"
        "  max_audio_bytes_per_sec: 0\n"
        "  max_audio_bytes_per_sec_burst: 0\n"
    )
    with _temp_grpc_server(config_content) as server:
        with grpc.insecure_channel(server["grpc_target"]) as channel:
            stub = stt_pb2_grpc.STTBackendStub(channel)
            session_id = f"test-decode-timeout-{time.time()}"
            stub.CreateSession(stt_pb2.SessionRequest(session_id=session_id))

            pcm16 = b"\x00\x00" * 16000 * 3

            def audio_chunks():
                yield stt_pb2.AudioChunk(
                    session_id=session_id,
                    pcm16=pcm16,
                    sample_rate=16000,
                    is_final=True,
                )

            with pytest.raises(grpc.RpcError) as e:
                list(stub.StreamingRecognize(audio_chunks()))

            assert e.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
            assert "ERR2001" in str(e.value.details())


def test_max_sessions_per_ip_exceeded():
    """Exceed per-IP session limit and expect ERR1011."""
    config_content = "server:\n  max_sessions_per_ip: 1\n"
    with _temp_grpc_server(config_content) as server:
        with grpc.insecure_channel(server["grpc_target"]) as channel:
            stub = stt_pb2_grpc.STTBackendStub(channel)
            stub.CreateSession(
                stt_pb2.SessionRequest(session_id=f"test-sess-1-{time.time()}")
            )
            with pytest.raises(grpc.RpcError) as e:
                stub.CreateSession(
                    stt_pb2.SessionRequest(session_id=f"test-sess-2-{time.time()}")
                )
            assert e.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
            assert "ERR1011" in str(e.value.details())
