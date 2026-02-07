import pytest

from stt_server.backend.runtime.metrics import Metrics


def test_record_decode_tracks_buffer_wait_and_response_emit():
    """Test record decode tracks buffer wait and response emit."""
    metrics = Metrics()

    metrics.record_decode(
        inference_sec=1.0,
        real_time_factor=0.5,
        queue_wait_sec=0.2,
        buffer_wait_sec=0.1,
        response_emit_sec=0.05,
    )
    metrics.record_decode(
        inference_sec=2.0,
        real_time_factor=1.5,
        queue_wait_sec=0.4,
        buffer_wait_sec=0.3,
        response_emit_sec=0.25,
    )

    render = metrics.render()
    assert render["decode_buffer_wait_total"] == pytest.approx(0.4)
    assert render["decode_buffer_wait_count"] == 2
    assert render["decode_buffer_wait_max"] == pytest.approx(0.3)
    assert render["decode_response_emit_total"] == pytest.approx(0.3)
    assert render["decode_response_emit_count"] == 2
    assert render["decode_response_emit_max"] == pytest.approx(0.25)

    snapshot = metrics.snapshot()
    assert snapshot["decode_buffer_wait_avg"] == pytest.approx(0.2)
    assert snapshot["decode_response_emit_avg"] == pytest.approx(0.15)


def test_metrics_track_stream_buffers_and_total_bytes():
    """Test stream buffer bytes and global buffer total metrics."""
    metrics = Metrics()

    metrics.set_buffer_total(1024)
    metrics.set_stream_buffer_bytes("session-1", 256)
    metrics.set_stream_buffer_bytes("session-2", 512)
    metrics.clear_stream_buffer("session-1")

    payload = metrics.render()
    assert payload["buffer_bytes_total"] == 1024
    stream_buffers = payload["stream_buffer_bytes"]
    expected_key = metrics._hash_key("session:session-2")
    assert stream_buffers[expected_key] == 512
    assert metrics._hash_key("session:session-1") not in stream_buffers


def test_metrics_rate_limit_blocks_hash_keys():
    """Test rate limit block metrics track hashed keys."""
    metrics = Metrics()

    metrics.record_rate_limit_block("stream", "api:secret")
    metrics.record_rate_limit_block("stream", "api:secret")
    metrics.record_rate_limit_block("http", "1.2.3.4")

    payload = metrics.render()
    assert payload["rate_limit_blocks"]["stream"] == 2
    assert payload["rate_limit_blocks"]["http"] == 1
    hashed = metrics._hash_key("api:secret")
    assert payload["rate_limit_blocks_by_key"][f"stream_{hashed}"] == 2


def test_metrics_decode_pending_tracks_latest_value():
    """Test decode pending metric reflects latest count."""
    metrics = Metrics()
    metrics.set_decode_pending(3)
    assert metrics.render()["decode_pending"] == 3
    metrics.set_decode_pending(1)
    assert metrics.render()["decode_pending"] == 1
