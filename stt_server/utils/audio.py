import numpy as np
import torch
import torchaudio


def pcm16_to_float32(pcm_bytes):
    """PCM16 bytes → float32 numpy array"""
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def ensure_16k(audio, src_rate):
    """Resample input audio to Whisper's required 16 kHz when needed."""
    if src_rate == 16000:
        return audio

    # Numpy -> Torch Tensor (add channel dimension: [1, Time])
    waveform = torch.from_numpy(audio).unsqueeze(0)

    # Resample using torchaudio.functional.resample.
    # We use lowpass_filter_width=6, which matches the default value in PyTorch.
    # This provides an optimal trade-off between processing speed and anti-aliasing quality for speech tasks.
    resampled_waveform = torchaudio.functional.resample(
        waveform,
        orig_freq=src_rate,
        new_freq=16000,
        lowpass_filter_width=6,
    )

    # Tensor -> Numpy (remove channel dimension)
    return resampled_waveform.squeeze(0).numpy()


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
