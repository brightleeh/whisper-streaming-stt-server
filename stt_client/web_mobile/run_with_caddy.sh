#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

: "${MOBILE_WEB_ADDR:=:8443}"
: "${MOBILE_WEB_ROOT:=$SCRIPT_DIR}"
: "${MOBILE_WS_UPSTREAM:=127.0.0.1:8001}"
: "${MOBILE_TLS_CERT:=$SCRIPT_DIR/certs/mobile.crt}"
: "${MOBILE_TLS_KEY:=$SCRIPT_DIR/certs/mobile.key}"

export MOBILE_WEB_ADDR
export MOBILE_WEB_ROOT
export MOBILE_WS_UPSTREAM
export MOBILE_TLS_CERT
export MOBILE_TLS_KEY

echo "[mobile-web] addr=${MOBILE_WEB_ADDR}"
echo "[mobile-web] root=${MOBILE_WEB_ROOT}"
echo "[mobile-web] ws_upstream=${MOBILE_WS_UPSTREAM}"
echo "[mobile-web] tls_cert=${MOBILE_TLS_CERT}"
echo "[mobile-web] tls_key=${MOBILE_TLS_KEY}"
echo "[mobile-web] starting caddy..."

exec caddy run --config "${SCRIPT_DIR}/Caddyfile" --adapter caddyfile
