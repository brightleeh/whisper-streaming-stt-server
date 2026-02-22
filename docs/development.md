# Development

## SLO/Performance

See `docs/slo.md` for draft targets and a load-test report template.

## Tests

- Unit tests only (skip integration): `./tools/run_tests.sh unit`
- Integration tests (requires running server): `./tools/run_tests.sh integration`
- Abuse/load smoke tests (starts temp server): `./tools/run_tests.sh abuse`
- Abuse scenarios include a backpressure metrics check (buffer cap + pending drops); enable with `STT_RUN_ABUSE_TESTS=1`.
- Full test run: `./tools/run_tests.sh all`

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

## Assets

- `stt_client/assets/hello.wav`: sourced from https://github.com/SkelterLabsInc/stt-dataset-example
