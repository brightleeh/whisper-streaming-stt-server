# Client Error Handling Guide

This guide maps `ERR####` codes to recommended client actions. Use it to decide when
to retry, back off, or fix client inputs. For the raw list of codes, see
`docs/troubleshooting.md`.

## Retry guidance (high level)

- **Retryable**: transient capacity/availability (`RESOURCE_EXHAUSTED`, `UNAVAILABLE`).
- **Fix input**: invalid request, chunk size, or missing auth (`INVALID_ARGUMENT`,
  `UNAUTHENTICATED`, `PERMISSION_DENIED`).
- **Restart session**: session timeout or decode timeout; create a fresh session.

## Error code mapping

| Code | gRPC status | What happened | Recommended client action |
| --- | --- | --- | --- |
| ERR1001 | INVALID_ARGUMENT | `session_id` missing in `CreateSession` | Fix request (no retry). |
| ERR1002 | ALREADY_EXISTS | `session_id` already active | Use a unique session ID. |
| ERR1003 | INVALID_ARGUMENT | Invalid VAD settings | Fix request (no retry). |
| ERR1004 | UNAUTHENTICATED | Unknown/missing `session_id` | Create a session first. |
| ERR1005 | PERMISSION_DENIED | Invalid session token | Refresh token and retry. |
| ERR1006 | DEADLINE_EXCEEDED | Session timeout (no audio) | Start a new session. |
| ERR1007 | INVALID_ARGUMENT | Audio chunk too large | Reduce `chunk_ms` or sample rate. |
| ERR1008 | RESOURCE_EXHAUSTED | VAD pool exhausted | Back off and retry. |
| ERR1009 | UNAUTHENTICATED | API key missing | Provide API key (no retry). |
| ERR1010 | INVALID_ARGUMENT | Invalid decode option(s) | Fix request (no retry). |
| ERR1011 | RESOURCE_EXHAUSTED | Max sessions reached | Back off or reduce concurrency. |
| ERR1012 | RESOURCE_EXHAUSTED | CreateSession rate limited | Back off with jitter. |
| ERR1013 | UNAVAILABLE | Server shutting down | Retry with backoff or fail over. |
| ERR1014 | UNAUTHENTICATED | CreateSession auth failed | Fix signature/secret. |
| ERR2001 | DEADLINE_EXCEEDED | Decode timed out | Start a new session; consider lowering load. |
| ERR2002 | INTERNAL | Decode task failed | Retry once; if repeated, alert. |
| ERR2003 | RESOURCE_EXHAUSTED | Stream rate limit exceeded | Reduce send rate / chunk cadence. |
| ERR2004 | RESOURCE_EXHAUSTED | Stream audio budget exceeded | Reduce bitrate or duration. |
| ERR3001 | UNKNOWN | Unexpected CreateSession error | Retry with backoff. |
| ERR3002 | UNKNOWN | Unexpected StreamingRecognize error | Retry with backoff. |

## Notes

- **Streaming retries** are only safe if no partials have been consumed. Use
  a restartable audio source when enabling automatic retries.
- **Backoff** should include jitter to avoid thundering herds.
