from __future__ import annotations

import threading
import time
from collections import deque
from concurrent import futures
from queue import Queue

import pytest

from stt_server.backend.application.model_registry import ModelRegistry, _DecodeTask


def test_model_registry_request_cancel_marks_future_cancelled():
    """request_cancel should signal and skip running tasks."""
    registry = ModelRegistry()
    model_id = "default"
    registry._session_inflight[model_id] = set()
    registry._dispatch_conds[model_id] = threading.Condition()

    future = futures.Future()
    cancel_event = threading.Event()
    registry._register_cancel_event(future, cancel_event)

    task = _DecodeTask(
        pcm=b"abc",
        sample_rate=16000,
        decode_options=None,
        session_id="session",
        is_final=False,
        submitted_at=time.perf_counter(),
        future=future,
        cancel_event=cancel_event,
    )
    queue: Queue = Queue()
    queue.put(task)

    assert registry.request_cancel(future) is True
    assert cancel_event.is_set()

    item = queue.get()
    assert registry._skip_cancelled_task(model_id, item, queue) is True

    with pytest.raises(futures.CancelledError):
        future.result()
    assert future.done()
    assert future not in registry._cancel_events


def test_model_registry_cancel_stale_partials_clears_cancel_map():
    """Cancelling stale partials should drop cancel tokens for cancelled futures."""
    registry = ModelRegistry()

    partial_future = futures.Future()
    partial_event = threading.Event()
    registry._register_cancel_event(partial_future, partial_event)

    final_future = futures.Future()
    final_event = threading.Event()
    registry._register_cancel_event(final_future, final_event)

    queue = deque(
        [
            _DecodeTask(
                pcm=b"partial",
                sample_rate=16000,
                decode_options=None,
                session_id="session",
                is_final=False,
                submitted_at=time.perf_counter(),
                future=partial_future,
                cancel_event=partial_event,
            ),
            _DecodeTask(
                pcm=b"final",
                sample_rate=16000,
                decode_options=None,
                session_id="session",
                is_final=True,
                submitted_at=time.perf_counter(),
                future=final_future,
                cancel_event=final_event,
            ),
        ]
    )

    registry._cancel_stale_partials(queue)

    assert len(queue) == 1
    assert queue[0].is_final is True
    assert partial_future not in registry._cancel_events
    assert final_future in registry._cancel_events
