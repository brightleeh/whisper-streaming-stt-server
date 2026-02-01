from pathlib import Path

from stt_server.utils import logger as logger_module
from stt_server.utils.logger import (
    LOGGER,
    TRANSCRIPT_LOGGER,
    clear_session_id,
    configure_logging,
    set_session_id,
)


def _stop_logging_listener() -> None:
    """Helper for  stop logging listener."""
    if logger_module.QUEUE_LISTENER:
        logger_module.QUEUE_LISTENER.stop()
        for handler in logger_module.QUEUE_LISTENER.handlers:
            handler.close()
        logger_module.QUEUE_LISTENER = None
    for handler in TRANSCRIPT_LOGGER.handlers:
        handler.close()
    TRANSCRIPT_LOGGER.handlers.clear()


def test_logging_includes_session_id(tmp_path: Path) -> None:
    """Test logging includes session id."""
    log_path = tmp_path / "session.log"
    configure_logging("INFO", str(log_path))
    try:
        set_session_id("session-123")
        LOGGER.info("log-test")
    finally:
        clear_session_id()
        _stop_logging_listener()

    content = log_path.read_text(encoding="utf-8")
    assert "session_id=session-123" in content


def test_transcript_logging_is_opt_in(tmp_path: Path) -> None:
    """Ensure transcripts do not appear in main logs by default."""
    log_path = tmp_path / "server.log"
    configure_logging("INFO", str(log_path))
    try:
        set_session_id("session-456")
        TRANSCRIPT_LOGGER.info("pii-text-1234")
    finally:
        clear_session_id()
        _stop_logging_listener()

    content = log_path.read_text(encoding="utf-8")
    assert "pii-text-1234" not in content


def test_transcript_logging_uses_separate_sink(tmp_path: Path) -> None:
    """Ensure transcripts only appear in the dedicated transcript sink."""
    log_path = tmp_path / "server.log"
    transcript_path = tmp_path / "transcripts.log"
    configure_logging("INFO", str(log_path), transcript_log_file=str(transcript_path))
    try:
        set_session_id("session-789")
        TRANSCRIPT_LOGGER.info("pii-text-5678")
    finally:
        clear_session_id()
        _stop_logging_listener()

    server_content = log_path.read_text(encoding="utf-8")
    transcript_content = transcript_path.read_text(encoding="utf-8")
    assert "pii-text-5678" not in server_content
    assert "pii-text-5678" in transcript_content
