from unittest.mock import MagicMock

from stt_server.backend.application.session_manager import (
    SessionFacade,
    SessionRegistry,
)
from stt_server.backend.application.stream_orchestrator import (
    StreamOrchestrator,
    StreamOrchestratorConfig,
)
from stt_server.config.languages import SupportedLanguages


class FakeContext:
    def __init__(self) -> None:
        self.trailing_metadata = None
        self.callbacks = []

    def invocation_metadata(self):
        return []

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def is_active(self) -> bool:
        return True

    def set_trailing_metadata(self, metadata):
        self.trailing_metadata = metadata

    def abort(self, code, details):
        raise RuntimeError(f"abort called: {code} {details}")


class FakeDecodeStream:
    def __init__(self, summary):
        self._summary = summary
        self.session_id = None

    def set_session_id(self, session_id):
        self.session_id = session_id

    def set_model_id(self, model_id):
        return None

    def cancel_pending(self):
        return 0, 0

    def has_pending_results(self):
        return False

    def emit_ready(self, block):
        return []

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
