"""Decode profile, language, and task resolution helpers."""

from __future__ import annotations

from typing import Any, Dict, Final, Optional, Tuple

from gen.stt.python.v1 import stt_pb2
from stt_server.config.default.model import (
    ALLOWED_DECODE_OPTION_KEYS,
    default_decode_profiles,
)
from stt_server.config.languages import SupportedLanguages

PROFILE_ENUM_TO_NAME: Final[dict[stt_pb2.DecodeProfile.ValueType, str]] = {
    stt_pb2.DECODE_PROFILE_REALTIME: "realtime",
    stt_pb2.DECODE_PROFILE_ACCURATE: "accurate",
}
PROFILE_NAME_TO_ENUM = {v: k for k, v in PROFILE_ENUM_TO_NAME.items()}
TASK_ENUM_TO_NAME: Final[dict[stt_pb2.Task.ValueType, str]] = {
    stt_pb2.TASK_TRANSLATE: "translate",
    stt_pb2.TASK_TRANSCRIBE: "transcribe",
}
TASK_NAME_TO_ENUM = {v: k for k, v in TASK_ENUM_TO_NAME.items()}


def normalize_decode_profiles(
    raw_profiles: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """Normalize decode profiles and apply defaults when missing."""
    profiles: Dict[str, Dict[str, Any]] = {}
    if raw_profiles:
        for name, options in raw_profiles.items():
            if isinstance(options, dict):
                profiles[name] = dict(options)
    if not profiles:
        profiles.update(default_decode_profiles())
    return profiles


def resolve_decode_profile(
    requested: Optional[str],
    profiles: Dict[str, Dict[str, Any]],
    default_profile: str,
) -> Tuple[str, Dict[str, Any]]:
    """Resolve the requested profile name and options with fallback."""
    if requested and requested in profiles:
        return requested, profiles[requested].copy()
    if requested and requested not in profiles:
        return default_profile, profiles[default_profile].copy()
    return default_profile, profiles[default_profile].copy()


def invalid_decode_options(options: Dict[str, Any]) -> list[str]:
    """Return unsupported decode option keys."""
    return [key for key in options.keys() if key not in ALLOWED_DECODE_OPTION_KEYS]


def resolve_language_code(
    requested: str,
    default_language: str,
    language_fix: bool,
    supported: SupportedLanguages,
) -> str:
    """Resolve a language code based on request and configuration."""
    trimmed = requested.strip().lower() if requested else ""
    codes = supported.get_codes()
    if trimmed:
        if codes is not None and trimmed not in codes:
            return ""
        return trimmed
    if language_fix and default_language:
        if codes is not None and default_language not in codes:
            return ""
        return default_language
    return ""


def resolve_task(requested: stt_pb2.Task.ValueType, default_task: str) -> str:
    """Resolve the task name from enum with default fallback."""
    return TASK_ENUM_TO_NAME.get(requested, default_task)


def task_enum_from_name(name: str) -> stt_pb2.Task.ValueType:
    """Resolve a task enum from a task name."""
    return TASK_NAME_TO_ENUM.get(name or "", stt_pb2.TASK_TRANSCRIBE)


def profile_name_from_enum(
    profile_enum: stt_pb2.DecodeProfile.ValueType,
) -> Optional[str]:
    """Resolve a profile name from the enum value."""
    return PROFILE_ENUM_TO_NAME.get(profile_enum)


def profile_enum_from_name(name: str) -> stt_pb2.DecodeProfile.ValueType:
    """Resolve a profile enum from a profile name."""
    return PROFILE_NAME_TO_ENUM.get(name or "", stt_pb2.DECODE_PROFILE_UNSPECIFIED)
