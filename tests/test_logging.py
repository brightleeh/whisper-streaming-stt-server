from pathlib import Path

from stt_server.utils import logger as logger_module
from stt_server.utils.logger import (
    LOGGER,
    clear_session_id,
    configure_logging,
    set_session_id,
)


def _stop_logging_listener() -> None:
    if logger_module.QUEUE_LISTENER:
        logger_module.QUEUE_LISTENER.stop()
        for handler in logger_module.QUEUE_LISTENER.handlers:
            handler.close()
        logger_module.QUEUE_LISTENER = None


def test_logging_includes_session_id(tmp_path: Path) -> None:
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
