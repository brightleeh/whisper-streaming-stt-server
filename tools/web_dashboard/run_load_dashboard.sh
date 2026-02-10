#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if ! command -v pnpm >/dev/null 2>&1; then
  echo "pnpm not found. Install with: npm install -g pnpm" >&2
  exit 1
fi

WEB_DASHBOARD_HOST="${WEB_DASHBOARD_HOST:-0.0.0.0}"
WEB_DASHBOARD_PORT="${WEB_DASHBOARD_PORT:-8010}"
WEB_DASHBOARD_RELOAD="${WEB_DASHBOARD_RELOAD:-1}"
export WEB_DASHBOARD_HOST WEB_DASHBOARD_PORT WEB_DASHBOARD_RELOAD
PUBLIC_HOST="${WEB_DASHBOARD_PUBLIC_HOST:-}"
if [[ -n "${PUBLIC_HOST}" ]]; then
  if [[ "${PUBLIC_HOST}" == http*://* ]]; then
    PUBLIC_ORIGIN="${PUBLIC_HOST}:3000"
    CONTROLLER_BASE="${PUBLIC_HOST}:${WEB_DASHBOARD_PORT}"
  else
    PUBLIC_ORIGIN="http://${PUBLIC_HOST}:3000"
    CONTROLLER_BASE="http://${PUBLIC_HOST}:${WEB_DASHBOARD_PORT}"
  fi
  export NEXT_PUBLIC_CONTROLLER_BASE="${NEXT_PUBLIC_CONTROLLER_BASE:-${CONTROLLER_BASE}}"
  if [[ -z "${WEB_DASHBOARD_CORS_ORIGINS:-}" ]]; then
    export WEB_DASHBOARD_CORS_ORIGINS="${PUBLIC_ORIGIN},http://localhost:3000,http://127.0.0.1:3000"
  fi
else
  export NEXT_PUBLIC_CONTROLLER_BASE="${NEXT_PUBLIC_CONTROLLER_BASE:-http://localhost:${WEB_DASHBOARD_PORT}}"
fi

cd "${REPO_ROOT}"
python -m tools.web_dashboard &
BACK_PID=$!

cd "${SCRIPT_DIR}/frontend"
pnpm dev &
FRONT_PID=$!

cleanup() {
  kill "${BACK_PID}" "${FRONT_PID}" >/dev/null 2>&1 || true
}

trap cleanup EXIT INT TERM
wait "${BACK_PID}" "${FRONT_PID}"
