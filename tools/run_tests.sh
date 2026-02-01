#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-unit}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

case "$MODE" in
  unit)
    export STT_SKIP_INTEGRATION=1
    python3 -m pytest -q
    ;;
  integration)
    export STT_REQUIRE_EXISTING=${STT_REQUIRE_EXISTING:-1}
    python3 -m pytest tests/test_integration.py -q
    ;;
  abuse)
    export STT_RUN_ABUSE_TESTS=1
    python3 -m pytest tests/test_abuse_scenarios.py -q
    ;;
  all)
    python3 -m pytest -q
    ;;
  *)
    echo "Usage: $0 {unit|integration|abuse|all}" >&2
    exit 1
    ;;
esac
