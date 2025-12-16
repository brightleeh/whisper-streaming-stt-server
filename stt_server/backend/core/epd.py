"""Endpoint detection helpers."""

from __future__ import annotations

from dataclasses import dataclass

from stt_server.utils import audio


@dataclass
class EPDUpdate:
    triggered: bool
    speech_active: bool
    silence_duration: float
    chunk_duration: float
    chunk_rms: float


class EPDState:
    """Tracks silence windows and determines when to trigger EPD."""

    def __init__(self, epd_threshold: float, epd_silence: float) -> None:
        self.epd_threshold = epd_threshold
        self.epd_silence = epd_silence
        self.speech_active = False
        self.silence_duration = 0.0

    def update(self, chunk_bytes: bytes, sample_rate: int) -> EPDUpdate:
        chunk_duration = audio.chunk_duration_seconds(len(chunk_bytes), sample_rate)
        chunk_rms = audio.chunk_rms(chunk_bytes)
        triggered = False

        if chunk_bytes:
            if chunk_rms >= self.epd_threshold:
                self.speech_active = True
                self.silence_duration = 0.0
            else:
                self.silence_duration += chunk_duration
        elif chunk_duration > 0:
            self.silence_duration += chunk_duration

        if self.speech_active and self.silence_duration >= self.epd_silence:
            triggered = True

        return EPDUpdate(
            triggered=triggered,
            speech_active=self.speech_active,
            silence_duration=self.silence_duration,
            chunk_duration=chunk_duration,
            chunk_rms=chunk_rms,
        )

    def reset_after_trigger(self) -> None:
        self.speech_active = False
        self.silence_duration = 0.0


def buffer_is_speech(buffer_bytes: bytes, threshold: float) -> bool:
    """Return True when buffer RMS meets the speech threshold."""
    if threshold <= 0:
        return True
    return audio.chunk_rms(buffer_bytes) >= threshold
