"""Default values and helpers for model-related configuration."""

from typing import Any, Dict

DEFAULT_MODEL_ID = "default"
DEFAULT_MODEL_NAME = "small"
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE_TYPE = "int8"
DEFAULT_TASK = "transcribe"
DEFAULT_LANGUAGE = "ko"
DEFAULT_LANGUAGE_FIX = False
DEFAULT_MODEL_POOL_SIZE = 1

DEFAULT_DECODE_PROFILE: Dict[str, Any] = {
    "beam_size": 1,
    "best_of": 1,
    "patience": 1.0,
    "temperature": 0.0,
    "length_penalty": 1.0,
    "without_timestamps": True,
    "compression_ratio_threshold": 2.4,
    "no_speech_threshold": 0.6,
    "log_prob_threshold": -1.0,
}


def default_decode_profiles() -> Dict[str, Dict[str, Any]]:
    """Return the default decode profile map."""
    return {"realtime": DEFAULT_DECODE_PROFILE.copy()}


MODEL_SECTION_MAP = {
    "name": "model",
    "device": "device",
    "compute_type": "compute_type",
    "language": "language",
    "language_fix": "language_fix",
    "pool_size": "model_pool_size",
    "task": "task",
    "default_decode_profile": "default_decode_profile",
}


__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_DEVICE",
    "DEFAULT_COMPUTE_TYPE",
    "DEFAULT_TASK",
    "DEFAULT_LANGUAGE",
    "DEFAULT_LANGUAGE_FIX",
    "DEFAULT_MODEL_POOL_SIZE",
    "DEFAULT_DECODE_PROFILE",
    "default_decode_profiles",
    "MODEL_SECTION_MAP",
]
