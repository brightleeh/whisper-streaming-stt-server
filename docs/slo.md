SLO/Performance Targets (Draft)

Service Level Objectives

- CreateSession P99: <= 300 ms (under steady load at max_sessions)
- Streaming first partial P99: <= 2.5 s (speech start -> first partial)
- Streaming final P99: <= 5.0 s (utterance end -> final)
- gRPC error rate: <= 0.5% (5xx/UNAVAILABLE/RESOURCE_EXHAUSTED excluded from availability SLO but tracked)

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
