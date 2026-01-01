from __future__ import annotations

import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from stt_server.utils.logger import LOGGER


def _sanitize_session_id(value: str) -> str:
    sanitized = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_"):
            sanitized.append(ch)
        else:
            sanitized.append("_")
        if len(sanitized) >= 80:
            break
    result = "".join(sanitized).strip("_")
    return result or "session"


@dataclass
class AudioStorageConfig:
    enabled: bool = False
    directory: Path = Path("data/audio")
    max_bytes: Optional[int] = None
    max_files: Optional[int] = None
    max_age_days: Optional[int] = None


class SessionAudioRecorder:
    """Streams PCM16 audio into a WAV file for a single session."""

    def __init__(self, session_id: str, path: Path, sample_rate: int) -> None:
        self.session_id = session_id
        self.path = path
        self.sample_rate = max(sample_rate, 1)
        self._bytes_written = 0
        self._wave = wave.open(str(self.path), "wb")
        self._wave.setnchannels(1)
        self._wave.setsampwidth(2)
        self._wave.setframerate(self.sample_rate)
        self._closed = False

    def append(self, pcm16: bytes) -> None:
        if self._closed or not pcm16:
            return
        self._wave.writeframes(pcm16)
        self._bytes_written += len(pcm16)

    def finalize(self) -> Optional[Path]:
        if not self._closed:
            self._wave.close()
            self._closed = True
        if self._bytes_written <= 0:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            return None
        return self.path

    @property
    def bytes_written(self) -> int:
        return self._bytes_written


class AudioStorageManager:
    """Persists streamed audio to disk and enforces retention limits."""

    def __init__(self, config: AudioStorageConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._directory = config.directory
        self._directory.mkdir(parents=True, exist_ok=True)
        self._max_bytes = (
            int(config.max_bytes) if config.max_bytes and config.max_bytes > 0 else -1
        )
        self._max_files = (
            int(config.max_files) if config.max_files and config.max_files > 0 else -1
        )
        self._max_age_days = (
            int(config.max_age_days)
            if config.max_age_days and config.max_age_days > 0
            else None
        )

    def start_recording(
        self, session_id: str, sample_rate: int
    ) -> SessionAudioRecorder:
        timestamp = time.strftime("%Y%m%dT%H%M%S")
        safe_session = _sanitize_session_id(session_id)
        filename = f"{timestamp}_{safe_session}.wav"
        path = self._directory / filename
        LOGGER.info(
            "Audio capture started session_id=%s path=%s sample_rate=%d",
            session_id,
            path,
            sample_rate,
        )
        return SessionAudioRecorder(session_id, path, sample_rate)

    def finalize_recording(self, recorder: SessionAudioRecorder, reason: str) -> None:
        path = recorder.finalize()
        if not path:
            LOGGER.info(
                "Audio capture discarded for session_id=%s reason=%s (empty payload)",
                recorder.session_id,
                reason,
            )
            return
        LOGGER.info(
            "Audio capture stored session_id=%s path=%s bytes=%d reason=%s",
            recorder.session_id,
            path,
            recorder.bytes_written,
            reason,
        )
        self._apply_retention()

    def _apply_retention(self) -> None:
        with self._lock:
            entries = self._gather_entries()
            if not entries:
                return
            now = time.time()
            if self._max_age_days:
                cutoff = now - (self._max_age_days * 86400)
                entries = self._remove_older_than(entries, cutoff, "max_age_days")
            if self._max_files > 0:
                entries = self._enforce_file_count(entries)
            if self._max_bytes > 0:
                self._enforce_total_bytes(entries)

    def _gather_entries(self) -> List[Tuple[Path, float, int]]:
        entries: List[Tuple[Path, float, int]] = []
        for path in self._directory.glob("*.wav"):
            try:
                stats = path.stat()
            except FileNotFoundError:
                continue
            entries.append((path, stats.st_mtime, stats.st_size))
        entries.sort(key=lambda item: item[1])
        return entries

    def _remove_older_than(
        self,
        entries: List[Tuple[Path, float, int]],
        cutoff: float,
        reason: str,
    ) -> List[Tuple[Path, float, int]]:
        remaining: List[Tuple[Path, float, int]] = []
        for path, mtime, size in entries:
            if mtime < cutoff:
                self._delete_file(path, reason)
            else:
                remaining.append((path, mtime, size))
        return remaining

    def _enforce_file_count(
        self, entries: List[Tuple[Path, float, int]]
    ) -> List[Tuple[Path, float, int]]:
        while self._max_files > 0 and len(entries) > self._max_files:
            path, _, _ = entries.pop(0)
            self._delete_file(path, "max_files")
        return entries

    def _enforce_total_bytes(self, entries: List[Tuple[Path, float, int]]) -> None:
        if self._max_bytes <= 0:
            return
        total = sum(size for _, _, size in entries)
        while entries and total > self._max_bytes:
            path, _, size = entries.pop(0)
            total -= size
            self._delete_file(path, "max_bytes")

    def _delete_file(self, path: Path, reason: str) -> None:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            size = 0
        try:
            path.unlink()
            LOGGER.info(
                "Deleted stored audio path=%s size=%d reason=%s", path, size, reason
            )
        except FileNotFoundError:
            LOGGER.debug("Audio path already removed: %s", path)
