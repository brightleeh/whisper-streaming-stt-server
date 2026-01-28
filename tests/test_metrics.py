import pytest

from stt_server.backend.runtime.metrics import Metrics


def test_record_decode_tracks_buffer_wait_and_response_emit():
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
