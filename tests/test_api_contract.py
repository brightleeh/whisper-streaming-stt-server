import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

from stt_server.errors import ERROR_SPECS, ErrorCode

PROTO_PATH = Path(__file__).resolve().parents[1] / "proto" / "stt.proto"
PROTO_CONTRACT_PATH = (
    Path(__file__).resolve().parent / "compat" / "stt_proto_contract.json"
)
ERROR_CONTRACT_PATH = (
    Path(__file__).resolve().parent / "compat" / "error_code_contract.json"
)
PROTO_RESERVED_CONTRACT_PATH = (
    Path(__file__).resolve().parent / "compat" / "proto_reserved_contract.json"
)


class _ProtoField(TypedDict):
    name: str
    number: int
    type: str
    label: str


@dataclass
class _ParsedProtoMessage:
    fields: list[_ProtoField] = field(default_factory=list)
    reserved_numbers: set[int] = field(default_factory=set)
    reserved_names: set[str] = field(default_factory=set)


def _strip_comments(line: str) -> str:
    if "//" in line:
        return line.split("//", 1)[0]
    return line


def _parse_proto_messages() -> dict[str, _ParsedProtoMessage]:
    messages: dict[str, _ParsedProtoMessage] = {}
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
            messages[current_message] = _ParsedProtoMessage()
            continue
        if current_message and line.startswith("}"):
            current_message = None
            continue
        if not current_message:
            continue
        message = messages[current_message]

        if line.startswith("reserved ") and line.endswith(";"):
            clause = line[len("reserved ") : -1].strip()
            if '"' in clause:
                for name in re.findall(r'"([^"]+)"', clause):
                    message.reserved_names.add(name)
            else:
                for token in clause.split(","):
                    part = token.strip()
                    if not part:
                        continue
                    if "to" in part:
                        start_raw, end_raw = part.split("to", 1)
                        start = int(start_raw.strip())
                        end = int(end_raw.strip())
                        if start > end:
                            start, end = end, start
                        message.reserved_numbers.update(range(start, end + 1))
                    else:
                        message.reserved_numbers.add(int(part))
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
        message.fields.append(
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
    current = _parse_proto_messages()

    for message, fields in contract.items():
        assert message in current, f"Missing message: {message}"
        current_fields = {field["name"]: field for field in current[message].fields}
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

    for message, parsed in current.items():
        numbers = [field["number"] for field in parsed.fields]
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


def test_proto_reserved_contract():
    """Removed proto fields must stay reserved by name and number."""
    contract = json.loads(PROTO_RESERVED_CONTRACT_PATH.read_text(encoding="utf-8"))
    contract.pop("//", None)
    entries = contract.get("entries", [])
    assert isinstance(
        entries, list
    ), "proto_reserved_contract.json entries must be a list"

    current = _parse_proto_messages()
    seen: set[tuple[str, str, int]] = set()

    for entry in entries:
        message = entry["message"]
        name = entry["name"]
        number = int(entry["number"])
        key = (message, name, number)
        assert key not in seen, f"Duplicate reserved contract entry: {key}"
        seen.add(key)

        assert message in current, f"Missing message for reserved contract: {message}"
        parsed = current[message]
        active_names = {field["name"] for field in parsed.fields}
        active_numbers = {field["number"] for field in parsed.fields}

        assert (
            name not in active_names
        ), f"{message}.{name} exists as an active field; reserved contract entry is stale."
        assert (
            number not in active_numbers
        ), f"{message} field number {number} is active; reserved contract entry is stale."

        assert (
            name in parsed.reserved_names
        ), f"{message} must reserve removed field name '{name}'"
        assert (
            number in parsed.reserved_numbers
        ), f"{message} must reserve removed field number {number}"
