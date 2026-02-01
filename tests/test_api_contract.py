import json
import re
from pathlib import Path

from stt_server.errors import ERROR_SPECS, ErrorCode

PROTO_PATH = Path(__file__).resolve().parents[1] / "proto" / "stt.proto"
PROTO_CONTRACT_PATH = (
    Path(__file__).resolve().parent / "compat" / "stt_proto_contract.json"
)
ERROR_CONTRACT_PATH = (
    Path(__file__).resolve().parent / "compat" / "error_code_contract.json"
)


def _strip_comments(line: str) -> str:
    if "//" in line:
        return line.split("//", 1)[0]
    return line


def _parse_proto_fields() -> dict[str, list[dict[str, object]]]:
    messages: dict[str, list[dict[str, object]]] = {}
    current_message: str | None = None
    field_re = re.compile(
        r"^(?:(optional|repeated)\s+)?(map<[^>]+>|\w+)\s+(\w+)\s*=\s*(\d+)\s*;"
    )

    for raw_line in PROTO_PATH.read_text(encoding="utf-8").splitlines():
        line = _strip_comments(raw_line).strip()
        if not line:
            continue
        if line.startswith("message "):
            current_message = line.split()[1].strip()
            if current_message.endswith("{"):
                current_message = current_message[:-1].strip()
            messages[current_message] = []
            continue
        if current_message and line.startswith("}"):
            current_message = None
            continue
        if not current_message:
            continue
        match = field_re.match(line)
        if not match:
            continue
        modifier, type_name, field_name, number = match.groups()
        label = "singular"
        if modifier == "optional":
            label = "optional"
        elif modifier == "repeated":
            label = "repeated"
        if type_name.startswith("map<"):
            label = "map"
            type_name = type_name.replace(" ", "")
        messages[current_message].append(
            {
                "name": field_name,
                "number": int(number),
                "type": type_name,
                "label": label,
            }
        )
    return messages


def test_proto_contract_is_compatible():
    """Ensure proto changes are additive and field numbers are stable."""
    contract = json.loads(PROTO_CONTRACT_PATH.read_text(encoding="utf-8"))
    contract.pop("//", None)
    current = _parse_proto_fields()

    for message, fields in contract.items():
        assert message in current, f"Missing message: {message}"
        current_fields = {field["name"]: field for field in current[message]}
        for expected in fields:
            name = expected["name"]
            assert name in current_fields, f"Missing field {message}.{name}"
            actual = current_fields[name]
            assert actual["number"] == expected["number"], (
                f"{message}.{name} number changed "
                f"{actual['number']} != {expected['number']}"
            )
            assert (
                actual["type"] == expected["type"]
            ), f"{message}.{name} type changed {actual['type']} != {expected['type']}"
            assert (
                actual["label"] == expected["label"]
            ), f"{message}.{name} label changed {actual['label']} != {expected['label']}"

    for message, fields in current.items():
        numbers = [field["number"] for field in fields]
        assert len(numbers) == len(
            set(numbers)
        ), f"Field number reuse detected in {message}"


def test_error_status_contract():
    """Ensure selected error/status mappings remain stable."""
    contract = json.loads(ERROR_CONTRACT_PATH.read_text(encoding="utf-8"))
    contract.pop("//", None)
    for code_name, expected in contract.items():
        code = ErrorCode[code_name]
        spec = ERROR_SPECS[code]
        assert spec.status.name == expected["grpc"]
        assert spec.http_status == expected["http"]
