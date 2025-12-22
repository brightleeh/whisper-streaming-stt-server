from __future__ import annotations

import secrets
from typing import Any, Dict

import grpc

from gen.stt.python.v1 import stt_pb2
from stt_server.backend.core.profile_resolver import (
    profile_enum_from_name,
    profile_name_from_enum,
    resolve_decode_profile,
    resolve_language_code,
    resolve_task,
    task_enum_from_name,
)
from stt_server.backend.core.session_manager import SessionInfo, SessionManager
from stt_server.config.languages import SupportedLanguages
from stt_server.utils.logger import LOGGER


class CreateSessionHandler:
    """Handles CreateSession requests for the STT backend servicer."""

    def __init__(
        self,
        session_manager: SessionManager,
        decode_profiles: Dict[str, Dict[str, Any]],
        default_decode_profile: str,
        default_language: str,
        language_fix: bool,
        default_task: str,
        supported_languages: SupportedLanguages,
        default_vad_silence: float,
        default_vad_threshold: float,
    ) -> None:
        self._session_manager = session_manager
        self._decode_profiles = decode_profiles
        self._default_decode_profile = default_decode_profile
        self._default_language = default_language
        self._language_fix = language_fix
        self._default_task = default_task
        self._supported_languages = supported_languages
        self._default_vad_silence = max(0.0, default_vad_silence)
        self._default_vad_threshold = max(0.0, default_vad_threshold)

    def handle(
        self, request: stt_pb2.SessionRequest, context: grpc.ServicerContext
    ) -> stt_pb2.SessionResponse:
        if not request.session_id:
            LOGGER.error("ERR1001 session_id is required")
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "ERR1001 session_id is required",
            )

        session_id = request.session_id
        vad_mode = (
            request.vad_mode
            if request.vad_mode in (stt_pb2.VAD_CONTINUE, stt_pb2.VAD_AUTO_END)
            else stt_pb2.VAD_CONTINUE
        )
        token_required = bool(request.require_token)
        token = secrets.token_hex(16) if token_required else ""
        api_key = (
            request.attributes.get("api_key") or request.attributes.get("api-key") or ""
        )

        requested_profile = profile_name_from_enum(request.decode_profile)
        if not requested_profile:
            requested_profile = request.attributes.get(
                "decode_profiles"
            ) or request.attributes.get("decode_profile")
        profile_name, profile_options = resolve_decode_profile(
            requested_profile,
            self._decode_profiles,
            self._default_decode_profile,
        )
        language_code = resolve_language_code(
            request.language_code,
            self._default_language,
            self._language_fix,
            self._supported_languages,
        )
        session_task = resolve_task(request.task, self._default_task)

        options = profile_options.copy()
        if session_task:
            options["task"] = session_task
        if language_code:
            options["language"] = language_code

        vad_silence = self._resolve_vad_silence(request.vad_silence, context)
        vad_threshold = self._resolve_vad_threshold(request.vad_threshold, context)
        session_info = SessionInfo(
            attributes=dict(request.attributes),
            vad_mode=vad_mode,
            vad_silence=vad_silence,
            vad_threshold=vad_threshold,
            token=token,
            token_required=token_required,
            api_key=api_key,
            decode_profile=profile_name,
            decode_options=options,
            language_code=language_code,
            task=session_task,
        )
        try:
            self._session_manager.create_session(session_id, session_info)
        except ValueError:
            LOGGER.error("ERR1002 session_id already active")
            context.abort(
                grpc.StatusCode.ALREADY_EXISTS,
                "ERR1002 session_id already active",
            )

        response_attributes = dict(request.attributes)
        response_attributes["decode_profile"] = profile_name
        if language_code:
            response_attributes["language_code"] = language_code

        LOGGER.info(
            "Created session_id=%s vad_mode=%s token_required=%s decode_profile=%s language=%s task=%s vad_silence=%.3f vad_threshold=%.4f attributes=%s",
            session_id,
            "AUTO_END" if vad_mode == stt_pb2.VAD_AUTO_END else "CONTINUE",
            token_required,
            profile_name,
            language_code or "auto",
            session_task,
            vad_silence,
            vad_threshold,
            dict(request.attributes),
        )

        return stt_pb2.SessionResponse(
            attributes=response_attributes,
            vad_mode=vad_mode,
            vad_silence=vad_silence,
            vad_threshold=vad_threshold,
            token=token,
            token_required=token_required,
            language_code=language_code,
            task=task_enum_from_name(session_task),
            decode_profile=profile_enum_from_name(profile_name),
        )

    def _resolve_vad_silence(
        self, value: float, context: grpc.ServicerContext
    ) -> float:
        if value <= 0:
            return self._default_vad_silence
        return value

    def _resolve_vad_threshold(
        self, value: float, context: grpc.ServicerContext
    ) -> float:
        if value < 0:
            LOGGER.error("ERR1003 vad_threshold must be non-negative")
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "ERR1003 vad_threshold must be non-negative",
            )
        if value == 0:
            return self._default_vad_threshold
        return value
