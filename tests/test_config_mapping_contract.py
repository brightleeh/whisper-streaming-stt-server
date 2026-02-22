"""Config mapping contract tests for YAML/CLI -> ServerConfig."""

from dataclasses import fields

import yaml

from stt_server.config.default import MODEL_SECTION_MAP, SERVER_SECTION_MAP
from stt_server.config.loader import ServerConfig, load_config
from stt_server.main import configure_from_args, parse_args


def test_section_maps_target_valid_server_config_fields() -> None:
    """All section-map targets must resolve to real ServerConfig fields."""
    field_names = {f.name for f in fields(ServerConfig)}

    for _section, mapping in SERVER_SECTION_MAP.items():
        for _yaml_key, target_field in mapping.items():
            assert target_field in field_names

    for _yaml_key, target_field in MODEL_SECTION_MAP.items():
        assert target_field in field_names


def test_yaml_and_cli_overrides_map_into_server_config(tmp_path, monkeypatch) -> None:
    """YAML values should load, and CLI flags should override selected fields."""
    server_yaml = tmp_path / "server.yaml"
    model_yaml = tmp_path / "model.yaml"

    server_yaml.write_text(
        yaml.safe_dump(
            {
                "server": {
                    "port": 51001,
                    "metrics_port": 18101,
                    "max_sessions": 3,
                    "max_total_buffer_bytes": 12345678,
                    "sample_rate": 8000,
                },
                "vad": {"silence": 0.95, "threshold": 0.45},
                "safety": {"speech_rms_threshold": 0.03},
                "auth": {"require_api_key": False},
            }
        ),
        encoding="utf-8",
    )
    model_yaml.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "name": "tiny",
                    "backend": "faster_whisper",
                    "language_fix": True,
                }
            }
        ),
        encoding="utf-8",
    )

    loaded = load_config(server_yaml, model_yaml)
    assert loaded.port == 51001
    assert loaded.metrics_port == 18101
    assert loaded.max_sessions == 3
    assert loaded.max_total_buffer_bytes == 12345678
    assert loaded.sample_rate == 8000
    assert loaded.vad_silence == 0.95
    assert loaded.vad_threshold == 0.45
    assert loaded.speech_rms_threshold == 0.03
    assert loaded.require_api_key is False
    assert loaded.model == "tiny"
    assert loaded.language_fix is True

    monkeypatch.setattr(
        "sys.argv",
        [
            "stt_server.main",
            "--config",
            str(server_yaml),
            "--model-config",
            str(model_yaml),
            "--port",
            "52001",
            "--metrics-port",
            "18201",
            "--max-sessions",
            "9",
            "--decode-timeout",
            "12.5",
            "--speech-threshold",
            "0.11",
            "--sample-rate",
            "16000",
            "--require-api-key",
            "--no-language-fix",
        ],
    )
    args = parse_args()
    configured = configure_from_args(args)

    assert configured.port == 52001
    assert configured.metrics_port == 18201
    assert configured.max_sessions == 9
    assert configured.decode_timeout_sec == 12.5
    assert configured.speech_rms_threshold == 0.11
    assert configured.sample_rate == 16000
    assert configured.require_api_key is True
    assert configured.language_fix is False

    # Non-overridden YAML fields should remain intact.
    assert configured.max_total_buffer_bytes == 12345678
    assert configured.vad_silence == 0.95
