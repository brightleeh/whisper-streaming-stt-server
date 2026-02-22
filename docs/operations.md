# Operations/Observability

The server also exposes an HTTP control plane (default `0.0.0.0:8000`) serving:

- `GET /metrics`: Prometheus text exposition (flattened counters/gauges).
- `GET /metrics.json`: JSON counters/gauges (active sessions, API-key session counts, decode timing aggregates, RTF stats, VAD trigger totals, active VAD utterances, error counts, buffer bytes, partial drop counts, rate-limit blocks).
- `GET /health`: returns `200` when the gRPC server is running, Whisper models are loaded, and worker pools are healthy; otherwise `500`.
- `GET /system`: JSON process/system metrics (CPU, RAM, thread counts). Uses `psutil` when available; otherwise falls back to basic RSS info. Optional GPU metrics can be enabled with `STT_ENABLE_GPU_METRICS=1` when `pynvml` is installed.

Backpressure health checklist (during load):

- `system.process.rss_bytes`: should stabilize after warmup (no unbounded climb).
- `metrics.decode_pending`: should plateau near the configured pending cap.
- `metrics.buffer_bytes_total`: should plateau near `max_total_buffer_bytes` when buffers are pressured.
- `metrics.partial_drop_count` / `metrics.rate_limit_blocks.*`: should increase when limits are actively shedding load.
- Dashboard and alerting suggestions live in `docs/slo.md`.

## Tuning profiles (standard presets)

Baseline backend guidance:

- macOS (Apple Silicon): `torch_whisper` + `mps`
- Linux + NVIDIA GPU: `faster_whisper` + `cuda`
- CPU only: `faster_whisper` + `cpu` (prefer `int8`)

Model size:

- If unsure, start with `small` on GPU and `tiny` or `small` on CPU.
- Replace the RTF ranges below with your benchmarked numbers.

**Low-latency (UI-first)**

- Goal: fast partials, quick UI updates, tighter buffers.
- Server:
  - `partial_decode_interval_sec: 0.2-0.4`
  - `partial_decode_window_sec: 2-4`
  - `buffer_overlap_sec: 0.2-0.4`
  - `max_buffer_sec: 3-6`
  - `vad_silence: 0.4-0.6`
  - `vad_threshold: 0.5-0.7`
  - `emit_final_on_vad: true`
  - `max_pending_decodes_per_stream: 2-4`
- Client:
  - `attributes.partial=true`
- Recommended hardware: GPU preferred (MPS/CUDA).
- Suggested `max_sessions`: start at `~2-4 x model_pool_size` and tune.

**Balanced (default)**

- Goal: stable partials and reasonable accuracy.
- Server: keep defaults (`interval=1.5`, `window=10`, `overlap=0.5`, `max_buffer=20`, `vad_silence=0.8`, `vad_threshold=0.5`).
- Client:
  - `attributes.partial=true` (optional)
- Recommended hardware: GPU or strong CPU.
- Suggested `max_sessions`: start at `~4-6 x model_pool_size` and tune.

**High-accuracy (final-first)**

- Goal: higher stability and accuracy, less partial churn.
- Server:
  - `partial_decode_interval_sec: 2-3` (or disable partials)
  - `partial_decode_window_sec: 12-20`
  - `buffer_overlap_sec: 0.8-1.2`
  - `max_buffer_sec: 30-60`
  - `vad_silence: 0.8-1.2`
  - `vad_threshold: 0.3-0.5`
  - `emit_final_on_vad: false`
  - `max_pending_decodes_per_stream: 8-12`
- Client:
  - `attributes.partial=false` (recommended)
- Recommended hardware: GPU preferred; CPU only for low concurrency.
- Suggested `max_sessions`: start at `~3-5 x model_pool_size` and tune.

RTF ranges should be filled in after benchmarking in your target environment.

## Adaptive throttling (optional)

Enable adaptive throttling to reduce partial load and shed new sessions when the system is under pressure.
The policy watches pending decode depth, buffer pressure, and orphan rate, then:

- increases `partial_decode_interval_sec` (fewer partials)
- reduces `decode_batch_window_ms` (lower queue latency)
- temporarily rejects new CreateSession requests

Configure in `server:`:

```yaml
adaptive_throttle_enabled: true
adaptive_throttle_interval_sec: 2.0
adaptive_pending_ratio_high: 0.8
adaptive_buffer_ratio_high: 0.85
adaptive_orphan_rate_high: 0.2
adaptive_partial_interval_scale: 2.0
adaptive_partial_interval_max_sec: null
adaptive_batch_window_target_ms: 0
adaptive_create_session_backoff_sec: 2.0
```

`adaptive_batch_window_target_ms` was previously named `adaptive_batch_window_min_ms` and is still accepted for backward compatibility.

Operational guidance (when to enable)

- Enable when `decode_pending` sits near the cap for 1-2 minutes, or
  `buffer_bytes_total` sustains >80% of `max_total_buffer_bytes`, or
  orphan rate spikes.
- Disable after ~10 minutes of stability (pending <50%, buffer <50%, orphan rate low).

Security checks:

- `tools/security_smoke_check.sh http://<host>:<port>` verifies `/metrics*`, `/system`, `/health` are protected (or `/health` is minimal when `STT_PUBLIC_HEALTH=minimal`).
- `tools/check_tls_expiry.py /path/to/cert.pem --warn-days 14` fails when certificates near expiry.

## Terminal dashboard (optional)

Use the terminal dashboard to poll `/metrics.json` and `/system` on a fixed interval:

```bash
python -m tools.dashboard.monitor_dashboard
```

Capture metrics to a file (JSONL or CSV) for later graphing:

```bash
python -m tools.dashboard.metrics_capture --duration-sec 300 --out metrics.jsonl
```

Plot captured metrics (requires `matplotlib`):

```bash
python -m tools.dashboard.plot_metrics metrics.jsonl --relative --output metrics.png
```

Common options:

- `--metrics-url` / `--system-url` to target a remote server.
- `--interval` to control refresh cadence (seconds).
- `--once` to fetch a single snapshot.
- `--no-clear` to avoid clearing the terminal between updates.

## Decode timing breakdown

Decode timing is measured inside the server for every decode task. The timing is split into distinct phases so you can see where time is being spent:

- **buffer wait time**: time spent accumulating audio before the decode is scheduled
- **queue wait time**: time spent waiting for an available model worker after the decode is scheduled
- **inference time**: time spent executing the model
- **response emit time**: time spent yielding results back to the client
- **total decode time**: sum of buffer wait + queue wait + inference + response emit

```mermaid
flowchart LR
  Buffering["Buffering audio until decode is scheduled"]
  QueueWait["Queue wait<br>(waiting for available model worker)"]
  Inference["Inference<br>(model execution)"]
  ResponseEmit["Response emit<br>(yielding results to the client)"]
  Buffering --> QueueWait --> Inference --> ResponseEmit
```

These totals are attached to each `StreamingRecognize` call as trailing metadata:

- `stt-decode-buffer-wait-sec`
- `stt-decode-queue-wait-sec`
- `stt-decode-inference-sec`
- `stt-decode-response-emit-sec`
- `stt-decode-total-sec`
- `stt-decode-count`

The `sec` suffix means seconds. Bench session logs print the same values per stream to help correlate client-observed latency with server-side decode timing.

## Docker

Dockerfiles live in the `docker/` directory (Ubuntu 22.04).

Build:

```bash
docker build -f docker/Dockerfile.ubuntu -t whisper-stt-server:ubuntu .
```

Run:

```bash
docker run --rm -p 50051:50051 -p 8000:8000 whisper-stt-server:ubuntu
```

### Kubernetes

Apply manifests (ConfigMap, Deployment, Service):

```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Storage policy:

- `k8s/deployment.yaml`: uses `emptyDir` for `/app/data/audio` (ephemeral; reset on Pod restart).
- `k8s/deployment.pvc.yaml`: uses `PersistentVolumeClaim` (`stt-audio-pvc`) for durable audio capture.

If you need durable server-side recordings:

```bash
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.pvc.yaml
kubectl apply -f k8s/service.yaml
```

NodePort access (on-premises):

```bash
kubectl get nodes -o wide

# Use the Internal-IP value from `kubectl get nodes -o wide` for <node-internal-ip>.
python -m stt_client.realtime.mic --server <node-internal-ip>:32051
```

Update config and rollout:

```bash
kubectl apply -f k8s/configmap.yaml
kubectl rollout restart deployment/stt-server
```

## Server-side audio capture

Enable `storage.persist_audio: true` to have the backend archive one WAV file per session inside `storage.directory`. Audio chunks are queued to a background writer thread to avoid blocking the streaming loop. Retention is enforced lazily right after each session finishes:

- `max_bytes`: cap total on-disk bytes (oldest files removed first).
- `max_files`: keep at most _N_ WAV files (oldest deleted first).
- `max_age_days`: delete recordings older than _N_ days.

Leave any limit `null` or negative for “no limit”. Audio capture is disabled by default and only activates when configured, so deployments that do not need server-side logging incur no overhead.
