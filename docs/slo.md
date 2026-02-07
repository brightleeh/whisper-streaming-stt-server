SLO/Performance Targets (Draft)

Service Level Objectives

- CreateSession P99: <= 300 ms (under steady load at max_sessions)
- Streaming first partial P99: <= 2.5 s (speech start -> first partial)
- Streaming final P99: <= 5.0 s (utterance end -> final)
- gRPC error rate: <= 0.5% (5xx/UNAVAILABLE/RESOURCE_EXHAUSTED excluded from availability SLO but tracked)

Grafana Dashboard (Recommended Panels)

Traffic
- Active sessions: `stt_active_sessions`
- CreateSession RPS: log-based (`"Created session_id="`); add a counter if you need Prometheus-only
- Stream RPS: log-based (`"Streaming started for session_id="`); add a counter if you need Prometheus-only

Performance
- Decode latency avg: `stt_decode_latency_total / stt_decode_latency_count`
- Decode latency max: `stt_decode_latency_max`
- RTF avg: `stt_rtf_total / stt_rtf_count`
- RTF max: `stt_rtf_max`
- P50/P95/P99 decode latency: requires per-request histograms or log-based percentiles

Quality
- Partial drop rate (approx): `rate(stt_partial_drop_count[5m]) / rate(stt_decode_latency_count[5m])`
- Orphan rate: `rate(stt_decode_orphaned[5m]) / rate(stt_decode_orphaned[5m] + stt_decode_cancelled[5m])`
- Partial/final ratio: requires per-result counters or log-based analysis

Resources
- Process RSS: `/system` JSON (scrape via JSON datasource or export with node/process exporters)
- GPU VRAM: `/system` JSON when `STT_ENABLE_GPU_METRICS=1` and `pynvml` installed
- Global buffer bytes: `stt_buffer_bytes_total`
- Per-stream buffer bytes: `stt_stream_buffer_bytes_*` (hashed keys)
- Pending decodes: `stt_decode_pending`

Limits/Protection
- Rate limit blocks: `stt_rate_limit_blocks_*`
- Session limit rejects: log-based `ERR1011` (not exported as a counter today)
- Error rate by gRPC status: `rate(stt_error_counts_*[5m])`

Alert Rules (examples)

- P99 decode latency > X sec for 5m: use histogram percentile if available; otherwise alert on `stt_decode_latency_max > X`
- Orphan rate > Y%: `rate(stt_decode_orphaned[5m]) / rate(stt_decode_orphaned[5m] + stt_decode_cancelled[5m]) > Y`
- Partial drop rate > Z%: `rate(stt_partial_drop_count[5m]) / rate(stt_decode_latency_count[5m]) > Z`
- Global buffer usage > 90% for 5m: `stt_buffer_bytes_total / <max_total_buffer_bytes> > 0.9`
- Session churn spike: `changes(stt_active_sessions[5m]) > N` or log-based CreateSession/stream start counts

Load Test Report Template
Date:
Environment:
Server Version:
Model:
Config (key limits):
Client Mix (realtime/batch):

Workload

- Concurrent sessions:
- Audio duration per session:
- Input type: silence/noise/speech
- Client pacing: realtime / no-realtime
- Duration:

Results

- CreateSession P50/P90/P99:
- First partial P50/P90/P99:
- Final result P50/P90/P99:
- Error rate (by code):
- CPU/RSS/threads delta:

Notes

- Bottlenecks observed:
- Mitigations:
