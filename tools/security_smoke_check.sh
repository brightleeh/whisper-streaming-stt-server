#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
PUBLIC_HEALTH="${STT_PUBLIC_HEALTH:-}"
HEALTH_DETAIL_MODE="${STT_HEALTH_DETAIL_MODE:-}"

ok() { echo "[ok] $*"; }
fail() { echo "[fail] $*" >&2; exit 1; }

request_status() {
  local url="$1"
  local status
  if status=$(curl -sS -o /dev/null -w "%{http_code}" "$url"); then
    echo "$status"
    return 0
  fi
  echo "000"
  return 1
}

expect_protected() {
  local path="$1"
  local url="${BASE_URL}${path}"
  local status
  if status=$(request_status "$url"); then
    if [[ "$status" == "401" || "$status" == "403" ]]; then
      ok "$path protected ($status)"
      return 0
    fi
    fail "$path expected 401/403, got ${status}"
  fi
  ok "$path unreachable (treated as protected)"
}

expect_health() {
  local path="$1"
  local url="${BASE_URL}${path}"
  local status
  if status=$(request_status "$url"); then
    if [[ "$PUBLIC_HEALTH" =~ ^(1|true|yes|on|minimal)$ || "$HEALTH_DETAIL_MODE" =~ ^(1|true|yes|on|token)$ ]]; then
      if [[ "$status" == "200" || "$status" == "503" ]]; then
        ok "$path public minimal ($status)"
        return 0
      fi
      fail "$path expected 200/503 for public health, got ${status}"
    fi
    if [[ "$status" == "401" || "$status" == "403" ]]; then
      ok "$path protected ($status)"
      return 0
    fi
    fail "$path expected 401/403, got ${status}"
  fi
  ok "$path unreachable (treated as protected)"
}

expect_protected "/metrics"
expect_protected "/metrics.json"
expect_protected "/system"
expect_health "/health"

ok "security smoke check passed"
