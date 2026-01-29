import logging
import logging.handlers
import queue
from contextvars import ContextVar
from pathlib import Path
from typing import List, Optional

# Custom TRACE level below DEBUG.
TRACE_LEVEL_NUM = 5
logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")


def trace(self: logging.Logger, message: str, *args, **kwargs) -> None:
    """Logger helper for TRACE level."""
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kwargs)


logging.Logger.trace = trace  # type: ignore

LOG_QUEUE: "queue.Queue[logging.LogRecord]" = queue.Queue()
QUEUE_LISTENER: Optional[logging.handlers.QueueListener] = None
_SESSION_ID: ContextVar[str] = ContextVar("session_id", default="-")


def set_session_id(session_id: Optional[str]) -> None:
    _SESSION_ID.set(session_id or "-")


def clear_session_id() -> None:
    _SESSION_ID.set("-")


def _get_session_id() -> str:
    return _SESSION_ID.get()


def _resolve_level(value: Optional[str], fallback: int) -> int:
    if not value:
        return fallback
    upper = value.upper()
    if upper == "TRACE":
        return TRACE_LEVEL_NUM
    return getattr(logging, upper, fallback)


def configure_logging(
    level: str, log_file: Optional[str], faster_whisper_level: Optional[str] = None
) -> None:
    """Configure root logging with queue-based handlers."""
    global QUEUE_LISTENER
    numeric_level = _resolve_level(level, logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s [%(filename)s:%(lineno)d] "
        "[session_id=%(session_id)s]: %(message)s"
    )

    class _SessionIdFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.session_id = _get_session_id()
            return True

    if QUEUE_LISTENER:
        QUEUE_LISTENER.stop()
        for handler in QUEUE_LISTENER.handlers:
            handler.close()
        QUEUE_LISTENER = None

    handlers: List[logging.Handler] = []
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    handlers.append(stream_handler)

    if log_file:
        log_path = Path(log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    queue_handler = logging.handlers.QueueHandler(LOG_QUEUE)
    queue_handler.addFilter(_SessionIdFilter())
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.close()
    root_logger.handlers.clear()
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(queue_handler)

    faster_whisper_logger = logging.getLogger("faster_whisper")
    faster_whisper_default_level = logging.WARNING
    faster_whisper_logger.setLevel(
        _resolve_level(faster_whisper_level, faster_whisper_default_level)
    )

    QUEUE_LISTENER = logging.handlers.QueueListener(
        LOG_QUEUE, *handlers, respect_handler_level=True
    )
    QUEUE_LISTENER.start()


LOGGER = logging.getLogger("stt_server")

__all__ = [
    "clear_session_id",
    "configure_logging",
    "LOGGER",
    "set_session_id",
    "TRACE_LEVEL_NUM",
]
