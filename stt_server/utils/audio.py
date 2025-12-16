import librosa
import numpy as np


def pcm16_to_float32(pcm_bytes):
    """PCM16 bytes → float32 numpy array"""
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def ensure_16k(audio, src_rate):
    """Resample input audio to Whisper's required 16 kHz when needed."""
    if src_rate == 16000:
        return audio
    return librosa.resample(audio, orig_sr=src_rate, target_sr=16000)


def chunk_duration_seconds(byte_length: int, sample_rate: int) -> float:
    """Return chunk duration given PCM16 byte length and sample rate."""
    if sample_rate <= 0:
        return 0.0
    bytes_per_sample = 2  # PCM16
    samples = byte_length / bytes_per_sample
    return samples / float(sample_rate)


def chunk_rms(pcm_bytes: bytes) -> float:
    """Compute RMS of PCM16 bytes."""
    if not pcm_bytes:
        return 0.0
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(np.square(audio))))
