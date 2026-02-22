#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

GEN_DIR="gen/stt/python/v1"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

mkdir -p "$GEN_DIR"
touch gen/__init__.py
touch gen/stt/__init__.py
touch gen/stt/python/__init__.py
touch "$GEN_DIR/__init__.py"

if ! "$PYTHON_BIN" -c "import grpc_tools.protoc" >/dev/null 2>&1; then
  echo "grpcio-tools is required. Install dev dependencies first." >&2
  echo "Example: pip install -e '.[dev]'" >&2
  exit 1
fi

PROTOC_CMD=(
  "$PYTHON_BIN" -m grpc_tools.protoc
  -I proto
  --python_out="$GEN_DIR"
  --grpc_python_out="$GEN_DIR"
)

MYPY_PLUGIN=""
PYTHON_BIN_DIR="$(dirname "$PYTHON_BIN")"
if [[ -x "$PYTHON_BIN_DIR/protoc-gen-mypy" ]]; then
  MYPY_PLUGIN="$PYTHON_BIN_DIR/protoc-gen-mypy"
elif command -v protoc-gen-mypy >/dev/null 2>&1; then
  MYPY_PLUGIN="$(command -v protoc-gen-mypy)"
fi

if [[ -n "$MYPY_PLUGIN" ]]; then
  PROTOC_CMD+=(--plugin=protoc-gen-mypy="$MYPY_PLUGIN")
  PROTOC_CMD+=(--mypy_out="./$GEN_DIR")
else
  echo "Skipping mypy stub generation (protoc-gen-mypy not found)."
fi

PROTOC_CMD+=(proto/stt.proto)
"${PROTOC_CMD[@]}"

echo "Generated gRPC stubs in $GEN_DIR (python=$PYTHON_BIN)"
