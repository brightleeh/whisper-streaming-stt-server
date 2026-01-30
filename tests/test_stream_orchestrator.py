import threading
import time
from unittest.mock import MagicMock

import pytest

from gen.stt.python.v1 import stt_pb2
from stt_server.backend.application.session_manager import (
    SessionFacade,
    SessionInfo,
    SessionRegistry,
)
from stt_server.backend.application.stream_orchestrator import (
    StreamOrchestrator,
    StreamOrchestratorConfig,
)
from stt_server.config.languages import SupportedLanguages
from stt_server.errors import ErrorCode, status_for


class FakeContext:
    def __init__(self, metadata=None) -> None:
        self._metadata = metadata or []
        self.trailing_metadata = None
        self.callbacks = []
        self.abort_calls = []

    def invocation_metadata(self):
        return list(self._metadata)

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def is_active(self) -> bool:
        return True

    def set_trailing_metadata(self, metadata):
        self.trailing_metadata = metadata

    def abort(self, code, details):
        self.abort_calls.append((code, details, threading.current_thread()))
        raise RuntimeError(f"abort called: {code} {details}")


class FakeDecodeStream:
    def __init__(self, summary, pending=False, emit_delay=0.0):
        self._summary = summary
        self._pending = pending
        self._emit_delay = emit_delay
        self.session_id = None
        self.scheduled = []

    def set_session_id(self, session_id):
        self.session_id = session_id

    def set_model_id(self, model_id):
        return None

    def cancel_pending(self):
        return 0, 0

    def pending_count(self):
        return 0

    def drop_pending_partials(self, max_drop=None):
        return 0, 0

    def has_pending_results(self):
        return self._pending

    def emit_ready(self, block):
        if block and self._emit_delay > 0:
            time.sleep(self._emit_delay)
        self._pending = False
        return []

    def schedule_decode(self, *args, **kwargs):
        self.scheduled.append((args, kwargs))

    def timing_summary(self):
        return self._summary


def test_stream_orchestrator_sets_decode_trailing_metadata(monkeypatch):
    session_facade = SessionFacade(SessionRegistry())
    model_registry = MagicMock()
    config = StreamOrchestratorConfig(
        vad_threshold=0.5,
        vad_silence=0.2,
        speech_rms_threshold=0.0,
        session_timeout_sec=10.0,
        default_sample_rate=16000,
        decode_timeout_sec=1.0,
        language_lookup=SupportedLanguages(),
        storage_enabled=False,
        storage_directory=".",
        storage_max_bytes=None,
        storage_max_files=None,
        storage_max_age_days=None,
    )

    orchestrator = StreamOrchestrator(session_facade, model_registry, config)
    fake_stream = FakeDecodeStream((0.1, 0.2, 0.3, 0.4, 5))
    monkeypatch.setattr(
        orchestrator.decode_scheduler, "new_stream", lambda: fake_stream
    )

    context = FakeContext()
    results = list(orchestrator.run(iter([]), context))  # type: ignore[arg-type]

    assert results == []
    assert context.trailing_metadata is not None
    metadata = dict(context.trailing_metadata)
    assert metadata["stt-decode-buffer-wait-sec"] == "0.100000"
    assert metadata["stt-decode-queue-wait-sec"] == "0.200000"
    assert metadata["stt-decode-inference-sec"] == "0.300000"
    assert metadata["stt-decode-response-emit-sec"] == "0.400000"
    assert metadata["stt-decode-total-sec"] == "1.000000"
    assert metadata["stt-decode-count"] == "5"


def test_stream_orchestrator_timeout_aborts_in_main_loop(monkeypatch):
    session_facade = SessionFacade(SessionRegistry())
    model_registry = MagicMock()
    config = StreamOrchestratorConfig(
        vad_threshold=0.5,
        vad_silence=0.2,
        speech_rms_threshold=0.0,
        session_timeout_sec=0.0,
        default_sample_rate=16000,
        decode_timeout_sec=1.0,
        language_lookup=SupportedLanguages(),
        storage_enabled=False,
        storage_directory=".",
        storage_max_bytes=None,
        storage_max_files=None,
        storage_max_age_days=None,
    )

    orchestrator = StreamOrchestrator(session_facade, model_registry, config)
    fake_stream = FakeDecodeStream((0.0, 0.0, 0.0, 0.0, 0))
    monkeypatch.setattr(
        orchestrator.decode_scheduler, "new_stream", lambda: fake_stream
    )

    class InlineThread:
        def __init__(self, target=None, daemon=None):
            self._target = target
            self.daemon = daemon

        def start(self):
            if self._target:
                self._target()

    import stt_server.backend.application.stream_orchestrator as stream_orchestrator

    monkeypatch.setattr(stream_orchestrator.threading, "Thread", InlineThread)

    context = FakeContext()
    main_thread = threading.current_thread()
    results = list(orchestrator.run(iter([]), context))  # type: ignore[arg-type]

    assert results == []
    assert len(context.abort_calls) == 1
    code, details, abort_thread = context.abort_calls[0]
    assert code == status_for(ErrorCode.SESSION_TIMEOUT)
    assert "ERR1006" in details
    assert abort_thread is main_thread


def test_stream_orchestrator_enforces_buffer_limit_with_partial_decode(monkeypatch):
    session_registry = SessionRegistry()
    session_facade = SessionFacade(session_registry)
    model_registry = MagicMock()
    config = StreamOrchestratorConfig(
        vad_threshold=0.0,
        vad_silence=0.2,
        speech_rms_threshold=0.0,
        session_timeout_sec=10.0,
        default_sample_rate=16000,
        decode_timeout_sec=1.0,
        language_lookup=SupportedLanguages(),
        max_buffer_sec=1.0,
        storage_enabled=False,
        storage_directory=".",
        storage_max_bytes=None,
        storage_max_files=None,
        storage_max_age_days=None,
    )

    session_id = "session-buffer-limit"
    session_registry.create_session(
        session_id,
        SessionInfo(
            attributes={},
            vad_mode=stt_pb2.VAD_CONTINUE,
            vad_silence=0.2,
            vad_threshold=0.0,
            token="",
            token_required=False,
            api_key="",
            decode_profile="realtime",
            decode_options={},
            language_code="",
            task="transcribe",
        ),
    )

    orchestrator = StreamOrchestrator(session_facade, model_registry, config)
    fake_stream = FakeDecodeStream((0.0, 0.0, 0.0, 0.0, 0))
    monkeypatch.setattr(
        orchestrator.decode_scheduler, "new_stream", lambda: fake_stream
    )

    sample_rate = 16000
    one_sec_pcm16 = b"\x00\x00" * sample_rate
    chunks = [
        stt_pb2.AudioChunk(
            pcm16=one_sec_pcm16, sample_rate=sample_rate, session_id=session_id
        ),
        stt_pb2.AudioChunk(
            pcm16=one_sec_pcm16, sample_rate=sample_rate, session_id=session_id
        ),
    ]

    context = FakeContext()
    results = list(orchestrator.run(iter(chunks), context))  # type: ignore[arg-type]

    assert results == []
    assert len(fake_stream.scheduled) == 1
    args, kwargs = fake_stream.scheduled[0]
    assert args[3] is False  # is_final
    assert kwargs.get("count_vad") is False


def test_stream_orchestrator_rejects_oversized_chunk(monkeypatch):
    session_registry = SessionRegistry()
    session_facade = SessionFacade(session_registry)
    model_registry = MagicMock()
    config = StreamOrchestratorConfig(
        vad_threshold=0.0,
        vad_silence=0.2,
        speech_rms_threshold=0.0,
        session_timeout_sec=10.0,
        default_sample_rate=16000,
        decode_timeout_sec=1.0,
        language_lookup=SupportedLanguages(),
        max_chunk_ms=10,
        storage_enabled=False,
        storage_directory=".",
        storage_max_bytes=None,
        storage_max_files=None,
        storage_max_age_days=None,
    )

    session_id = "session-oversized-chunk"
    session_registry.create_session(
        session_id,
        SessionInfo(
            attributes={},
            vad_mode=stt_pb2.VAD_CONTINUE,
            vad_silence=0.2,
            vad_threshold=0.0,
            token="",
            token_required=False,
            api_key="",
            decode_profile="realtime",
            decode_options={},
            language_code="",
            task="transcribe",
        ),
    )

    orchestrator = StreamOrchestrator(session_facade, model_registry, config)
    fake_stream = FakeDecodeStream((0.0, 0.0, 0.0, 0.0, 0))
    monkeypatch.setattr(
        orchestrator.decode_scheduler, "new_stream", lambda: fake_stream
    )

    sample_rate = 16000
    oversized_pcm16 = b"\x00\x00" * 1000
    chunks = [
        stt_pb2.AudioChunk(
            pcm16=oversized_pcm16, sample_rate=sample_rate, session_id=session_id
        )
    ]

    context = FakeContext()
    with pytest.raises(RuntimeError, match="ERR1007"):
        list(orchestrator.run(iter(chunks), context))  # type: ignore[arg-type]

    assert len(context.abort_calls) == 1
    code, details, _abort_thread = context.abort_calls[0]
    assert code == status_for(ErrorCode.AUDIO_CHUNK_TOO_LARGE)
    assert "ERR1007" in details


def test_stream_orchestrator_drops_partial_when_stream_pending_limit_reached(
    monkeypatch,
):
    session_registry = SessionRegistry()
    session_id = "session-pending-limit"
    session_registry.create_session(
        session_id,
        SessionInfo(
            attributes={},
            vad_mode=stt_pb2.VAD_CONTINUE,
            vad_silence=0.2,
            vad_threshold=0.0,
            token="",
            token_required=False,
            api_key="",
            decode_profile="realtime",
            decode_options={},
            language_code="",
            task="transcribe",
        ),
    )

    session_facade = SessionFacade(session_registry)
    model_registry = MagicMock()
    config = StreamOrchestratorConfig(
        vad_threshold=0.0,
        vad_silence=0.2,
        speech_rms_threshold=0.0,
        session_timeout_sec=10.0,
        default_sample_rate=16000,
        decode_timeout_sec=1.0,
        language_lookup=SupportedLanguages(),
        max_buffer_sec=0.5,
        max_pending_decodes_per_stream=1,
        storage_enabled=False,
        storage_directory=".",
        storage_max_bytes=None,
        storage_max_files=None,
        storage_max_age_days=None,
    )

    orchestrator = StreamOrchestrator(session_facade, model_registry, config)
    fake_stream = FakeDecodeStream((0.0, 0.0, 0.0, 0.0, 0))
    drop_calls: list[int | None] = []

    def drop_partials(max_drop=None):
        drop_calls.append(max_drop)
        return 0, 0

    fake_stream.pending_count = lambda: 1  # type: ignore[assignment]
    fake_stream.drop_pending_partials = drop_partials  # type: ignore[assignment]
    monkeypatch.setattr(
        orchestrator.decode_scheduler, "new_stream", lambda: fake_stream
    )

    sample_rate = 16000
    one_sec_pcm16 = b"\x00\x00" * sample_rate
    chunks = [
        stt_pb2.AudioChunk(
            pcm16=one_sec_pcm16, sample_rate=sample_rate, session_id=session_id
        )
    ]

    context = FakeContext()
    results = list(orchestrator.run(iter(chunks), context))  # type: ignore[arg-type]

    assert results == []
    assert fake_stream.scheduled == []
    assert drop_calls == [1]


def test_stream_orchestrator_aborts_when_global_pending_limit_reached(
    monkeypatch,
):
    session_registry = SessionRegistry()
    session_id = "session-global-pending-limit"
    session_registry.create_session(
        session_id,
        SessionInfo(
            attributes={},
            vad_mode=stt_pb2.VAD_CONTINUE,
            vad_silence=0.2,
            vad_threshold=0.0,
            token="",
            token_required=False,
            api_key="",
            decode_profile="realtime",
            decode_options={},
            language_code="",
            task="transcribe",
        ),
    )

    session_facade = SessionFacade(session_registry)
    model_registry = MagicMock()
    config = StreamOrchestratorConfig(
        vad_threshold=0.0,
        vad_silence=0.2,
        speech_rms_threshold=0.0,
        session_timeout_sec=10.0,
        default_sample_rate=16000,
        decode_timeout_sec=1.0,
        language_lookup=SupportedLanguages(),
        max_pending_decodes_global=1,
        decode_queue_timeout_sec=0.0,
        storage_enabled=False,
        storage_directory=".",
        storage_max_bytes=None,
        storage_max_files=None,
        storage_max_age_days=None,
    )

    orchestrator = StreamOrchestrator(session_facade, model_registry, config)
    fake_stream = FakeDecodeStream((0.0, 0.0, 0.0, 0.0, 0))
    monkeypatch.setattr(
        orchestrator.decode_scheduler, "new_stream", lambda: fake_stream
    )

    acquired = orchestrator.decode_scheduler.acquire_pending_slot(
        block=False, timeout=None
    )
    assert acquired
    try:
        sample_rate = 16000
        pcm16 = b"\x00\x00" * 400
        chunks = [
            stt_pb2.AudioChunk(
                pcm16=pcm16,
                sample_rate=sample_rate,
                session_id=session_id,
                is_final=True,
            )
        ]

        context = FakeContext()
        with pytest.raises(RuntimeError, match="ERR2001"):
            list(orchestrator.run(iter(chunks), context))  # type: ignore[arg-type]

        assert len(context.abort_calls) == 1
        code, details, _abort_thread = context.abort_calls[0]
        assert code == status_for(ErrorCode.DECODE_TIMEOUT)
        assert "ERR2001" in details
    finally:
        orchestrator.decode_scheduler.release_pending_slot()

def test_stream_orchestrator_timeout_ignored_while_pending_decode(monkeypatch):
    session_registry = SessionRegistry()
    session_id = "session-timeout-pending"
    session_registry.create_session(
        session_id,
        SessionInfo(
            attributes={},
            vad_mode=stt_pb2.VAD_CONTINUE,
            vad_silence=0.2,
            vad_threshold=0.0,
            token="",
            token_required=False,
            api_key="",
            decode_profile="realtime",
            decode_options={},
            language_code="",
            task="transcribe",
        ),
    )

    session_facade = SessionFacade(session_registry)
    model_registry = MagicMock()
    config = StreamOrchestratorConfig(
        vad_threshold=0.5,
        vad_silence=0.2,
        speech_rms_threshold=0.0,
        session_timeout_sec=0.02,
        default_sample_rate=16000,
        decode_timeout_sec=1.0,
        language_lookup=SupportedLanguages(),
        storage_enabled=False,
        storage_directory=".",
        storage_max_bytes=None,
        storage_max_files=None,
        storage_max_age_days=None,
    )

    orchestrator = StreamOrchestrator(session_facade, model_registry, config)
    fake_stream = FakeDecodeStream(
        (0.0, 0.0, 0.0, 0.0, 0), pending=True, emit_delay=0.05
    )
    monkeypatch.setattr(
        orchestrator.decode_scheduler, "new_stream", lambda: fake_stream
    )

    reasons = []

    def record_remove(state, reason=""):
        reasons.append(reason)

    monkeypatch.setattr(session_facade, "remove_session", record_remove)

    context = FakeContext(metadata=[("session-id", session_id)])
    results = list(orchestrator.run(iter([]), context))  # type: ignore[arg-type]

    assert results == []
    assert reasons and reasons[0] != "timeout"
