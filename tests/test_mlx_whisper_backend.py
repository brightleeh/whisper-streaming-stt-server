"""Unit tests for the mlx_whisper backend."""

from unittest.mock import patch

import pytest

# All tests in this module require mlx_whisper (macOS + Apple Silicon only).
mlx_whisper_mod = pytest.importorskip("mlx_whisper")
mlx_core = pytest.importorskip("mlx.core")

from stt_server.model.backends.base import Segment  # noqa: E402
from stt_server.model.backends.mlx_whisper import (  # noqa: E402
    MODEL_REPO_MAP,
    MlxWhisperBackend,
)


class TestModelRepoMapping:
    def test_standard_sizes_mapped(self):
        for size in (
            "tiny",
            "base",
            "small",
            "medium",
            "large",
            "large-v2",
            "large-v3",
        ):
            assert size in MODEL_REPO_MAP

    def test_small_maps_to_mlx_community(self):
        backend = MlxWhisperBackend("small", "mlx", "float16")
        assert backend.model_repo == "mlx-community/whisper-small-mlx"

    def test_large_maps_to_v3(self):
        backend = MlxWhisperBackend("large", "mlx", "float16")
        assert backend.model_repo == "mlx-community/whisper-large-v3-mlx"

    def test_unknown_size_uses_fallback(self):
        backend = MlxWhisperBackend("distil-large-v3", "mlx", "float16")
        assert backend.model_repo == "mlx-community/whisper-distil-large-v3"

    def test_custom_repo_passthrough(self):
        repo = "my-org/custom-whisper-model"
        backend = MlxWhisperBackend(repo, "mlx", "float16")
        assert backend.model_repo == repo


class TestDtype:
    def test_float16_aliases(self):
        for ct in ("float16", "fp16", "half"):
            backend = MlxWhisperBackend("tiny", "mlx", ct)
            assert backend._dtype == mlx_core.float16

    def test_float32_default(self):
        backend = MlxWhisperBackend("tiny", "mlx", "float32")
        assert backend._dtype == mlx_core.float32


class TestTranscribe:
    def test_parses_segments(self):
        mock_result = {
            "segments": [
                {"start": 0.0, "end": 1.5, "text": "hello"},
                {"start": 1.5, "end": 3.0, "text": " world"},
            ],
            "language": "en",
        }
        backend = MlxWhisperBackend("tiny", "mlx", "float16")
        with patch.object(mlx_whisper_mod, "transcribe", return_value=mock_result):
            segments, info = backend.transcribe(b"\x00" * 100, {})

        assert len(segments) == 2
        assert segments[0] == Segment(0.0, 1.5, "hello")
        assert segments[1] == Segment(1.5, 3.0, " world")
        assert info.language == "en"
        assert info.language_probability == -1.0

    def test_empty_segments(self):
        mock_result = {"segments": [], "language": ""}
        backend = MlxWhisperBackend("tiny", "mlx", "float16")
        with patch.object(mlx_whisper_mod, "transcribe", return_value=mock_result):
            segments, info = backend.transcribe(b"\x00" * 100, {})

        assert segments == []
        assert info.language == ""

    def test_passes_repo_to_transcribe(self):
        mock_result = {"segments": [], "language": ""}
        backend = MlxWhisperBackend("small", "mlx", "float16")
        with patch.object(
            mlx_whisper_mod, "transcribe", return_value=mock_result
        ) as mock_fn:
            backend.transcribe(b"\x00" * 100, {"language": "ko"})

        mock_fn.assert_called_once()
        _, kwargs = mock_fn.call_args
        assert kwargs["path_or_hf_repo"] == "mlx-community/whisper-small-mlx"
        assert kwargs["language"] == "ko"
        assert kwargs["fp16"] is True

    def test_robust_segment_parsing(self):
        mock_result = {
            "segments": [
                {"start": "bad", "end": 1.0, "text": "ok"},
                "not_a_dict",
                {"start": 2.0, "end": 3.0, "text": None},
            ],
            "language": None,
        }
        backend = MlxWhisperBackend("tiny", "mlx", "float32")
        with patch.object(mlx_whisper_mod, "transcribe", return_value=mock_result):
            segments, info = backend.transcribe(b"\x00" * 100, {})

        assert len(segments) == 2
        assert segments[0].start == 0.0  # "bad" -> fallback 0.0
        assert segments[1].text == ""  # None -> ""
        assert info.language == ""  # None -> ""


class TestNormalizeOptions:
    def test_supported_options_pass_through(self):
        backend = MlxWhisperBackend("tiny", "mlx", "float16")
        opts = backend._normalize_options({"language": "en", "temperature": 0.0})
        assert opts == {"language": "en", "temperature": 0.0}

    def test_beam_size_and_patience_dropped(self):
        """beam_size and patience are silently dropped (beam search not implemented)."""
        backend = MlxWhisperBackend("tiny", "mlx", "float16")
        opts = backend._normalize_options(
            {"language": "ko", "beam_size": 5, "patience": 1.0, "best_of": 5}
        )
        assert "beam_size" not in opts
        assert "patience" not in opts
        assert "best_of" in opts  # sampling mode is supported

    def test_unsupported_options_dropped(self):
        backend = MlxWhisperBackend("tiny", "mlx", "float16")
        opts = backend._normalize_options(
            {"language": "en", "vad_filter": True, "batch_size": 4}
        )
        assert "language" in opts
        assert "vad_filter" not in opts
        assert "batch_size" not in opts

    def test_log_prob_threshold_remapped(self):
        backend = MlxWhisperBackend("tiny", "mlx", "float16")
        opts = backend._normalize_options({"log_prob_threshold": -1.0})
        assert "logprob_threshold" in opts
        assert "log_prob_threshold" not in opts

    def test_without_timestamps_remapped(self):
        backend = MlxWhisperBackend("tiny", "mlx", "float16")
        opts = backend._normalize_options({"without_timestamps": True})
        assert "word_timestamps" in opts
        assert opts["word_timestamps"] is False
        assert "without_timestamps" not in opts
