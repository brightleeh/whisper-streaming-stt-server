import logging
import logging.handlers
import queue
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


def configure_logging(level: str, log_file: Optional[str]) -> None:
    """Configure root logging with queue-based handlers."""
    global QUEUE_LISTENER
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    if level.upper() == "TRACE":
        numeric_level = TRACE_LEVEL_NUM

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

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
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(queue_handler)

    if QUEUE_LISTENER:
        QUEUE_LISTENER.stop()
    QUEUE_LISTENER = logging.handlers.QueueListener(
        LOG_QUEUE, *handlers, respect_handler_level=True
    )
    QUEUE_LISTENER.start()


LOGGER = logging.getLogger("stt_server")

__all__ = ["configure_logging", "LOGGER", "TRACE_LEVEL_NUM"]
