# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-03-12

First stable release. All public APIs (gRPC proto, HTTP endpoints, error codes) are
now under the [API Stability Contract](README.md#api-stability-contract) — additive
changes only, enforced by CI.

### Core

- Bidirectional gRPC `StreamingRecognize` with real-time partial and final results
- Silero VAD endpointing: **Continue** (multi-utterance) and **Auto-End** (single utterance) modes
- `committed_text` / `unstable_text` split output (committed is monotonically growing)
- `emit_final_on_vad`: emit final results on each VAD trigger without closing the session
- Configurable partial decode interval, window size, and buffer overlap
- Per-session language, task, and decode profile overrides (`realtime`, `accurate`)

### Backends

- **faster_whisper** — CPU / CUDA via CTranslate2 (default)
- **torch_whisper** — PyTorch with MPS support for Apple Silicon
- **mlx_whisper** — MLX-native backend for Apple Silicon (`pip install .[mlx]`)
- Multi-model registry with named model load profiles
- Admin hot-swap: `/admin/load_model`, `/admin/unload_model`, `/admin/list_models`

### Traffic Control & Protection

- Concurrent session caps: global, per-IP, per-API-key
- CreateSession token-bucket rate limiter
- Inbound audio byte-rate limiter with separate realtime / batch mode overrides
- Global and per-stream pending decode caps with backpressure (finals block, partials drop)
- Adaptive throttling: auto-widen partial interval and pause session creation under load
- Per-session audio duration hard cap (`max_audio_seconds_per_session`)

### Auth & Security

- API key validation (`auth.require_api_key`)
- HMAC signed-token auth (`auth.create_session_auth_profile: signed_token`) with TTL
- gRPC TLS with optional enforcement (`tls.required`)
- WebSocket public-bind auth enforcement (bypass: `STT_ALLOW_INSECURE_WS=1`)
- Transcript logging to a dedicated opt-in sink; disabled by default (PII-safe)

### Observability

- Prometheus `/metrics`, JSON `/metrics.json`, structured `/health`, `/system`
- Token-protected endpoints (`STT_OBSERVABILITY_TOKEN`), IP allowlist (`STT_HTTP_ALLOWLIST`)
- HTTP rate limiting with trusted-proxy-aware client IP resolution
- Decode latency, RTF, pending, orphan, partial drop, buffer bytes, and VAD pool metrics
- Public minimal health mode (`STT_PUBLIC_HEALTH=minimal`) for unauthenticated probes

### Clients

- Python realtime streaming client (`stt_client.realtime`)
- Python batch file client (`stt_client.batch`)
- Live microphone client (`stt_client.realtime.mic`)
- Python SDK (`stt_client.sdk`)
- Browser/PWA WebSocket client (`stt_client/web_mobile`)

### Infrastructure

- WebSocket bridge (`/ws/stream`) for browser-based streaming
- Ubuntu Dockerfile and Kubernetes manifests (Deployment, Service, HPA, PVC)
- CI pipeline: unit tests, abuse/backpressure tests, proto/HTTP contract checks
- Load test tooling (`tools/bench/grpc_load_test`) with bottleneck analysis
- Web ops dashboard (`tools/web_dashboard`)

### API Stability Contract (declared in this release)

- **gRPC / Proto**: additive only; no field removal, renaming, type change, or number reuse
- **HTTP responses**: additive only; error format `{code, message}` is frozen
- **Error code mappings**: pinned in `tests/compat/error_code_contract.json`
- Enforced by `tests/test_api_contract.py` and golden files in `tests/compat/`

### Known Limitations

- **Integration tests**: gRPC round-trip tests run locally only; CI skips them
  (`STT_SKIP_INTEGRATION=1`) because model weights are not available.
- **torch_whisper on MPS**: forces float32 due to PyTorch fp16 instability (~2x slower
  than MLX).
- **mlx_whisper**: thread-safety constraint enforces `pool_size=1`.
- **GPU pool sizing**: `pool_size=1` is optimal for all GPU backends; pool=4 causes
  3–5x inference degradation from VRAM contention. Scale horizontally instead.
- **CPU extreme load**: pool=4 / 30 channels exceeds `decode_timeout_sec` on tested
  hardware. CPU deployments beyond pool×3 channels require horizontal scaling.

[1.0.0]: https://github.com/brightleeh/whisper-streaming-stt-server/releases/tag/v1.0.0
