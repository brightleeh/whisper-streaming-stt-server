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
        default_epd_silence: float,
        default_epd_threshold: float,
    ) -> None:
        self._session_manager = session_manager
        self._decode_profiles = decode_profiles
        self._default_decode_profile = default_decode_profile
        self._default_language = default_language
        self._language_fix = language_fix
        self._default_task = default_task
        self._supported_languages = supported_languages
        self._default_epd_silence = max(0.0, default_epd_silence)
        self._default_epd_threshold = max(0.0, default_epd_threshold)

    def handle(
        self, request: stt_pb2.SessionRequest, context: grpc.ServicerContext
    ) -> stt_pb2.SessionResponse:
        if not request.session_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "session_id is required")

        session_id = request.session_id
        epd_mode = (
            request.epd_mode
            if request.epd_mode in (stt_pb2.EPD_CONTINUE, stt_pb2.EPD_AUTO_END)
            else stt_pb2.EPD_CONTINUE
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

        epd_silence = self._resolve_epd_silence(request.epd_silence, context)
        epd_threshold = self._resolve_epd_threshold(request.epd_threshold, context)
        session_info = SessionInfo(
            attributes=dict(request.attributes),
            epd_mode=epd_mode,
            epd_silence=epd_silence,
            epd_threshold=epd_threshold,
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
            context.abort(grpc.StatusCode.ALREADY_EXISTS, "session_id already active")

        response_attributes = dict(request.attributes)
        response_attributes["decode_profile"] = profile_name
        if language_code:
            response_attributes["language_code"] = language_code

        LOGGER.info(
            "Created session_id=%s epd_mode=%s token_required=%s decode_profile=%s language=%s task=%s epd_silence=%.3f epd_threshold=%.4f attributes=%s",
            session_id,
            "AUTO_END" if epd_mode == stt_pb2.EPD_AUTO_END else "CONTINUE",
            token_required,
            profile_name,
            language_code or "auto",
            session_task,
            epd_silence,
            epd_threshold,
            dict(request.attributes),
        )

        return stt_pb2.SessionResponse(
            attributes=response_attributes,
            epd_mode=epd_mode,
            epd_silence=epd_silence,
            epd_threshold=epd_threshold,
            token=token,
            token_required=token_required,
            language_code=language_code,
            task=task_enum_from_name(session_task),
            decode_profile=profile_enum_from_name(profile_name),
        )

    def _resolve_epd_silence(
        self, value: float, context: grpc.ServicerContext
    ) -> float:
        if value <= 0:
            return self._default_epd_silence
        return value

    def _resolve_epd_threshold(
        self, value: float, context: grpc.ServicerContext
    ) -> float:
        if value < 0:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "epd_threshold must be non-negative",
            )
        if value == 0:
            return self._default_epd_threshold
        return value
