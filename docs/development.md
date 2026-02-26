# Development

## SLO/Performance

See `docs/slo.md` for draft targets and a load-test report template.

## Tests

- Unit tests only (skip integration): `./tools/run_tests.sh unit`
- Integration tests (requires running server): `./tools/run_tests.sh integration`
- Abuse/load smoke tests (starts temp server): `./tools/run_tests.sh abuse`
- Abuse scenarios include a backpressure metrics check (buffer cap + pending drops); enable with `STT_RUN_ABUSE_TESTS=1`.
- Shutdown integration tests (real process + SIGTERM): `STT_RUN_SHUTDOWN_INTEGRATION=1 python3 -m pytest tests/test_shutdown_integration.py -q`
- Config mapping contract tests (YAML/CLI -> `ServerConfig`): `python3 -m pytest tests/test_config_mapping_contract.py -q`
- Full test run: `./tools/run_tests.sh all`

## API compatibility workflow

- `tests/test_api_contract.py` enforces proto and error mapping compatibility.
- If you intentionally retire a proto field:
  1. Add `reserved <number>;` and `reserved "<name>";` to the message in `proto/stt.proto`.
  2. Add an entry to `tests/compat/proto_reserved_contract.json`.
  3. Update `tests/compat/stt_proto_contract.json` as part of the intentional contract change.

## Long-run abuse/profiling

Manual helper for longer-running noise/silence streams (RSS/threads delta summary):

```bash
python3 tools/long_run_abuse.py --server localhost:50051 --http http://localhost:8000 \
  --duration-sec 600 --mode noise --token "$STT_OBSERVABILITY_TOKEN"
```

## Generate gRPC stubs

Use the shared generator script so local/CI generation stays identical:

```bash
./tools/gen_proto.sh
```

Commit/publish the `gen` package (or copy it into each project) so both
sides share the same `stt_pb2.py` and `stt_pb2_grpc.py` files.

## Dependency policy

- Runtime dependencies use bounded ranges (`>=` and `<`) in `pyproject.toml` and `requirements.txt`.
- Prefer raising the lower bound only after verifying compatibility in CI.
- Keep major upgrades explicit (for example `grpcio 1.x -> 2.x`) instead of allowing silent jumps.

## Load testing (bench)

Run the gRPC load-test script from the repo root:

```bash
python -m tools.bench.grpc_load_test --channels 100 --iterations 1 --warmup-iterations 1 --log-sessions
```

Backpressure/drop validation (short run):

```bash
python -m stt_server.main --config config/loadtest/bench_backpressure.yaml \
  --model small --device mps --model-backend torch_whisper --model-pool-size 1 --max-sessions 100

python -m tools.bench.grpc_load_test \
  --channels 50 --iterations 6 --chunk-ms 100 --realtime --speed 2.0 --attr partial=true
```

Expected signals: `metrics.partial_drop_count` increases, `metrics.buffer_bytes_total` plateaus near
`max_total_buffer_bytes`, and `metrics.decode_pending` plateaus near `max_pending_decodes_global`.
To confirm stream rate limits, keep the default `config/server.yaml` and increase load
(for example `--speed 2.5` with 50 channels) until `ERR2003` appears and `metrics.rate_limit_blocks`
increments.

Per-session logs can be emitted in structured formats:

- `--session-log-format` supports `jsonl` (default), `csv`, `tsv`, and `markdown`.
- `--log-sessions` prints per-session logs to stdout (limited by `--max-session-logs`).
- `--session-log-path <path>` writes **all** session logs to a file using the selected format.
- `--warmup-iterations` runs warm-up iterations per channel that are excluded from the stats.
- `--ramp-steps` / `--ramp-interval-sec` ramp channels up in batches instead of firing all at once.

Per-session fields include decode timing breakdown (`decode_buffer_wait_seconds`, `decode_queue_wait_seconds`, `decode_inference_seconds`, `decode_response_emit_seconds`, `decode_total_seconds`), rounded to three decimals.
Markdown includes headers/start/end/columns; CSV/TSV/JSONL are data-only.

### Benchmark matrix

Run a full benchmark matrix across pool sizes and channel counts (starts/stops the server automatically per pool_size):

```bash
./tools/bench/run_benchmark_matrix.sh macos-gpu
./tools/bench/run_benchmark_matrix.sh macos-gpu-mlx
```

To use an already-running server:

```bash
STT_BENCH_SKIP_SERVER=1 ./tools/bench/run_benchmark_matrix.sh ubuntu-gpu
```

Results are written to `bench_results/<profile>/<timestamp>/`.
See the script header for available profiles and environment variables.

## Assets

- `stt_client/assets/hello.wav`: sourced from https://github.com/SkelterLabsInc/stt-dataset-example
