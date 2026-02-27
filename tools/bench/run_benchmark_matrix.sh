#!/usr/bin/env bash
# =============================================================================
# Whisper Streaming STT Server — Benchmark Matrix Runner
# =============================================================================
#
# Usage:
#   ./tools/bench/run_benchmark_matrix.sh <PROFILE>
#
# Profiles:
#   macos-cpu        macOS ARM64 (M4 Pro, 24GB RAM) — faster_whisper, cpu, int8
#   macos-gpu        macOS ARM64 (M4 Pro, 24GB RAM) — torch_whisper, mps, fp16
#   macos-gpu-mlx    macOS ARM64 (M4 Pro, 24GB RAM) — mlx_whisper, mps, fp16
#   ubuntu-cpu       Ubuntu x86_64 (Ryzen 7 8845HS, RTX 4060 8GB, 32GB RAM) — faster_whisper, cpu, int8
#   ubuntu-gpu       Ubuntu x86_64 (Ryzen 7 8845HS, RTX 4060 8GB, 32GB RAM) — faster_whisper, cuda, fp16
#
# Environment variables (optional):
#   STT_BENCH_TARGET        gRPC target            (default: localhost:50051)
#   STT_BENCH_AUDIO         WAV file path           (default: stt_client/assets/hello.wav)
#   STT_BENCH_ITERATIONS    iterations per channel  (default: 3)
#   STT_BENCH_WARMUP        warmup iterations       (default: 1)
#   STT_BENCH_CHUNK_MS      chunk size in ms        (default: 100)
#   STT_BENCH_OUTDIR        results output dir      (default: bench_results/<profile>/<timestamp>)
#   STT_BENCH_MAX_SESSIONS  server --max-sessions   (default: 50)
#   STT_BENCH_SERVER_WAIT   seconds to wait after server start (default: 120)
#   STT_BENCH_SKIP_SERVER   set to 1 to skip server start (use existing server)
#
# The script will:
#   1. Start the server with the correct backend/device/pool_size
#   2. Wait for /health to return 200
#   3. Run grpc_load_test for each (pool_size, channels) pair
#   4. Collect session logs (JSONL) + summary into $STT_BENCH_OUTDIR
#   5. Stop the server, move to the next pool_size, repeat
#
# Example:
#   ./tools/bench/run_benchmark_matrix.sh macos-gpu
#   ./tools/bench/run_benchmark_matrix.sh macos-gpu-mlx
#   STT_BENCH_SKIP_SERVER=1 ./tools/bench/run_benchmark_matrix.sh ubuntu-gpu
# =============================================================================
set -euo pipefail

# ----------------------------- utilities -------------------------------------
_log()  { echo "[$(date +%H:%M:%S)] $*"; }
_fail() { echo "[$(date +%H:%M:%S)] ERROR: $*" >&2; exit 1; }

# ----------------------------- profile lookup --------------------------------
PROFILE="${1:-}"
if [[ -z "$PROFILE" ]]; then
  echo "Usage: $0 <macos-cpu|macos-gpu|macos-gpu-mlx|ubuntu-cpu|ubuntu-gpu>"
  exit 1
fi

case "$PROFILE" in
  macos-cpu)
    MODEL_BACKEND="faster_whisper"
    DEVICE="cpu"
    COMPUTE_TYPE="int8"
    PROFILE_LABEL="macOS ARM64 (M4 Pro, 24GB RAM) — faster_whisper (CPU, int8)"
    ;;
  macos-gpu)
    MODEL_BACKEND="torch_whisper"
    DEVICE="mps"
    COMPUTE_TYPE="float32"
    PROFILE_LABEL="macOS ARM64 (M4 Pro, 24GB RAM) — torch_whisper (MPS, fp32)"
    ;;
  macos-gpu-mlx)
    MODEL_BACKEND="mlx_whisper"
    DEVICE="mps"
    COMPUTE_TYPE="float16"
    PROFILE_LABEL="macOS ARM64 (M4 Pro, 24GB RAM) — mlx_whisper (MPS, fp16)"
    ;;
  ubuntu-cpu)
    MODEL_BACKEND="faster_whisper"
    DEVICE="cpu"
    COMPUTE_TYPE="int8"
    PROFILE_LABEL="Ubuntu x86_64 (Ryzen 7 8845HS, RTX 4060 8GB, 32GB RAM) — faster_whisper (CPU, int8)"
    ;;
  ubuntu-gpu)
    MODEL_BACKEND="faster_whisper"
    DEVICE="cuda"
    COMPUTE_TYPE="float16"
    PROFILE_LABEL="Ubuntu x86_64 (Ryzen 7 8845HS, RTX 4060 8GB, 32GB RAM) — faster_whisper (CUDA, fp16)"
    ;;
  *)
    echo "Unknown profile: $PROFILE"
    echo "Valid profiles: macos-cpu, macos-gpu, macos-gpu-mlx, ubuntu-cpu, ubuntu-gpu"
    exit 1
    ;;
esac

# ----------------------------- configuration ---------------------------------
MODEL="small"
TARGET="${STT_BENCH_TARGET:-localhost:50051}"
AUDIO="${STT_BENCH_AUDIO:-stt_client/assets/hello.wav}"
ITERATIONS="${STT_BENCH_ITERATIONS:-3}"
WARMUP="${STT_BENCH_WARMUP:-1}"
CHUNK_MS="${STT_BENCH_CHUNK_MS:-100}"
MAX_SESSIONS="${STT_BENCH_MAX_SESSIONS:-50}"
SERVER_WAIT="${STT_BENCH_SERVER_WAIT:-120}"
SKIP_SERVER="${STT_BENCH_SKIP_SERVER:-0}"

# ----------------------------- venv detection --------------------------------
# Resolve python from .venv; fall back to system python3 if not found.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON=""
if [[ -x "${REPO_ROOT}/.venv/bin/python3" ]]; then
  PYTHON="${REPO_ROOT}/.venv/bin/python3"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${REPO_ROOT}/.venv/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python3" ]]; then
  PYTHON="${VIRTUAL_ENV}/bin/python3"
else
  _log "WARNING: .venv not found at ${REPO_ROOT}/.venv — falling back to system python3"
  PYTHON="python3"
fi

_log "Using Python: $PYTHON ($($PYTHON --version 2>&1))"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${STT_BENCH_OUTDIR:-bench_results/${PROFILE}/${TIMESTAMP}}"
mkdir -p "$OUTDIR"

# ----------------------------- test matrix -----------------------------------
# Format: "pool_size:ch1,ch2,ch3,..."
# mlx_whisper is not thread-safe so only pool_size=1 is supported.
if [[ "$MODEL_BACKEND" == "mlx_whisper" ]]; then
  MATRIX=(
    "1:1,3,5,10,15,20,30"
  )
else
  MATRIX=(
    "1:1,3,5,10"
    "2:1,5,10,15"
    "4:5,10,15,20,30"
  )
fi

# ----------------------------- server management -----------------------------
SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    _log "Stopping server (PID $SERVER_PID)..."
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}
trap cleanup EXIT

wait_for_health() {
  local url="http://localhost:8000/health"
  local max_wait="$SERVER_WAIT"
  local elapsed=0
  _log "Waiting for server health at $url (max ${max_wait}s)..."
  while (( elapsed < max_wait )); do
    if curl -sf -o /dev/null "$url" 2>/dev/null; then
      _log "Server healthy after ${elapsed}s."
      return 0
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
  _fail "Server failed to become healthy within ${max_wait}s"
}

start_server() {
  local pool_size="$1"
  _log "Starting server: model=$MODEL device=$DEVICE backend=$MODEL_BACKEND pool=$pool_size"

  STT_ALLOW_INSECURE_WS=1 "$PYTHON" -m stt_server.main \
    --model "$MODEL" \
    --device "$DEVICE" \
    --model-backend "$MODEL_BACKEND" \
    --compute-type "$COMPUTE_TYPE" \
    --model-pool-size "$pool_size" \
    --max-sessions "$MAX_SESSIONS" \
    --log-metrics \
    > "${OUTDIR}/server_pool${pool_size}.log" 2>&1 &

  SERVER_PID=$!
  _log "Server started (PID $SERVER_PID)"
  wait_for_health
}

stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    _log "Stopping server (PID $SERVER_PID)..."
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    sleep 3
  fi
}

run_bench() {
  local pool_size="$1"
  local channels="$2"
  local run_label="pool${pool_size}_ch${channels}"
  local log_path="${OUTDIR}/${run_label}.jsonl"
  local summary_path="${OUTDIR}/${run_label}_summary.txt"

  _log "--- Bench: pool_size=$pool_size channels=$channels ---"

  "$PYTHON" -m tools.bench.grpc_load_test \
    --target "$TARGET" \
    --channels "$channels" \
    --iterations "$ITERATIONS" \
    --warmup-iterations "$WARMUP" \
    --audio "$AUDIO" \
    --chunk-ms "$CHUNK_MS" \
    --realtime \
    --log-sessions \
    --session-log-format jsonl \
    --session-log-path "$log_path" \
    --attr partial=true \
    --attr emit_final_on_vad=true \
    --ramp-steps 3 \
    --ramp-interval-sec 1.0 \
    2>&1 | tee "$summary_path"

  _log "Results: $summary_path"
  _log "Session logs: $log_path"
}

# ----------------------------- metadata file ---------------------------------
cat > "${OUTDIR}/metadata.txt" <<EOF
Benchmark Metadata
==================
Profile:       $PROFILE
Label:         $PROFILE_LABEL
Model:         $MODEL
Backend:       $MODEL_BACKEND
Device:        $DEVICE
Compute Type:  $COMPUTE_TYPE
Audio:         $AUDIO
Iterations:    $ITERATIONS
Warmup:        $WARMUP
Chunk MS:      $CHUNK_MS
Max Sessions:  $MAX_SESSIONS
Timestamp:     $TIMESTAMP
Matrix:
$(for entry in "${MATRIX[@]}"; do echo "  $entry"; done)
EOF

_log "============================================"
_log "Benchmark: $PROFILE_LABEL"
_log "Output:    $OUTDIR"
_log "============================================"

# ----------------------------- main loop -------------------------------------
for entry in "${MATRIX[@]}"; do
  IFS=':' read -r pool_size channels_csv <<< "$entry"
  IFS=',' read -ra channels_arr <<< "$channels_csv"

  if [[ "$SKIP_SERVER" != "1" ]]; then
    stop_server
    start_server "$pool_size"
  else
    _log "Skipping server start (STT_BENCH_SKIP_SERVER=1), pool_size=$pool_size"
    _log "Make sure the server is running with --model-pool-size $pool_size"
    if [[ "${#MATRIX[@]}" -gt 1 ]]; then
      _log "WARNING: With SKIP_SERVER=1 and multiple pool sizes, you must restart"
      _log "         the server manually between pool_size changes."
      if [[ "$pool_size" != "1" ]]; then
        _log "Press Enter when server is ready with pool_size=$pool_size (or Ctrl+C to abort)..."
        read -r
      fi
    fi
  fi

  for ch in "${channels_arr[@]}"; do
    run_bench "$pool_size" "$ch"
    sleep 2
  done
done

if [[ "$SKIP_SERVER" != "1" ]]; then
  stop_server
fi

# ----------------------------- summary report --------------------------------
REPORT="${OUTDIR}/REPORT.md"

# Extract a metric value from a summary file.
# Usage: _extract <file> <section> <key>
# Example: _extract summary.txt "RTF" "p50"  -> "0.893"
_extract() {
  local file="$1" section="$2" key="$3"
  awk -v sect="$section" -v k="$key" '
    /^\* / { cur = substr($0, 3) }
    cur == sect && $0 ~ "^    " k ": " {
      val = $0; sub(/^[[:space:]]*[^:]+:[[:space:]]*/, "", val)
      print val; exit
    }
  ' "$file"
}

# Extract bottleneck dominant field.
# Example: _extract_bottleneck summary.txt -> "buffer_wait (92%)"
_extract_bottleneck() {
  awk '/dominant:/ { sub(/.*dominant:[[:space:]]*/, ""); print; exit }' "$1"
}

cat > "$REPORT" <<EOF
# Benchmark Report: ${PROFILE}

**${PROFILE_LABEL}**
Date: $(date +%Y-%m-%d)
Model: ${MODEL} | Backend: ${MODEL_BACKEND} | Device: ${DEVICE} | Compute: ${COMPUTE_TYPE}

## Results

| pool | ch | RTF P50 | Queue P50 | Queue P95 | Infer P50 | Infer P95 | Total P50 | Total P95 | Bottleneck | Errors |
|------|-----|---------|-----------|-----------|-----------|-----------|-----------|-----------|------------|--------|
EOF

for entry in "${MATRIX[@]}"; do
  IFS=':' read -r pool_size channels_csv <<< "$entry"
  IFS=',' read -ra channels_arr <<< "$channels_csv"
  for ch in "${channels_arr[@]}"; do
    summary="${OUTDIR}/pool${pool_size}_ch${ch}_summary.txt"
    if [[ -f "$summary" ]]; then
      rtf_p50=$(_extract "$summary" "RTF" "p50")
      queue_p50=$(_extract "$summary" "Decode Queue Wait" "p50")
      queue_p95=$(_extract "$summary" "Decode Queue Wait" "p95")
      infer_p50=$(_extract "$summary" "Decode Inference" "p50")
      infer_p95=$(_extract "$summary" "Decode Inference" "p95")
      total_p50=$(_extract "$summary" "Decode Total" "p50")
      total_p95=$(_extract "$summary" "Decode Total" "p95")
      bottleneck=$(_extract_bottleneck "$summary")
      failures=$(_extract "$summary" "Info" "Failures")
      sessions=$(_extract "$summary" "Info" "Sessions")
      if [[ "$failures" == "0" ]]; then
        errors="0/${sessions}"
      else
        errors="**${failures}/${sessions}**"
      fi
      echo "| $pool_size | $ch | $rtf_p50 | $queue_p50 | $queue_p95 | $infer_p50 | $infer_p95 | $total_p50 | $total_p95 | $bottleneck | $errors |" >> "$REPORT"
    else
      echo "| $pool_size | $ch | — | — | — | — | — | — | — | — | ❌ Missing |" >> "$REPORT"
    fi
  done
done

cat >> "$REPORT" <<'EOF'

## Metrics Legend

| Metric | Description |
|--------|-------------|
| **RTF P50** | Median Real-Time Factor (decode time / audio duration). <1.0 = faster than realtime. |
| **Queue P50/P95** | Time waiting for a worker. Spikes when channels exceed pool capacity. |
| **Infer P50/P95** | Model execution time per decode pass. Hardware-bound. |
| **Total P50/P95** | End-to-end decode latency (queue wait + inference + emit). |
| **Bottleneck** | Dominant phase and its share of total decode time. |
| **Errors** | Failures / total sessions. |

Session-level JSONL logs are in `*.jsonl` files for further analysis.
EOF

_log "============================================"
_log "Benchmark complete!"
_log "Report:  $REPORT"
_log "Results: $OUTDIR/"
_log "============================================"