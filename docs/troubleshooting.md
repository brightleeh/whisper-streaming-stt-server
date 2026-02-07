# Troubleshooting

## Error codes

Errors are tagged in logs and gRPC error messages with `ERR####`. HTTP endpoints (admin/control plane) return a JSON payload with `code` and `message` when they fail. The gRPC status codes are listed below for clarity.

### `ERR1xxx`: request/session validation or authentication failures

- `ERR1001` (INVALID_ARGUMENT): missing `session_id` in `CreateSession`
- `ERR1002` (ALREADY_EXISTS): `session_id` already active
- `ERR1003` (INVALID_ARGUMENT): `vad_threshold` must be non-negative
- `ERR1004` (UNAUTHENTICATED): unknown or missing `session_id`
- `ERR1005` (PERMISSION_DENIED): invalid session token
- `ERR1006` (DEADLINE_EXCEEDED): session timeout (no audio)
- `ERR1007` (INVALID_ARGUMENT): audio chunk exceeds maximum size
- `ERR1008` (RESOURCE_EXHAUSTED): VAD capacity exhausted
- `ERR1009` (UNAUTHENTICATED): API key required but missing
- `ERR1010` (INVALID_ARGUMENT): invalid decode option(s)
- `ERR1011` (RESOURCE_EXHAUSTED): session limit exceeded
- `ERR1012` (RESOURCE_EXHAUSTED): CreateSession rate limited
- `ERR1013` (UNAVAILABLE): server shutting down
- `ERR1014` (UNAUTHENTICATED): CreateSession authentication failed

### `ERR2xxx`: decode pipeline/runtime failures

- `ERR2001` (DEADLINE_EXCEEDED): decode timeout waiting for pending tasks
- `ERR2002` (INTERNAL): decode task failed
- `ERR2003` (RESOURCE_EXHAUSTED): stream rate limit exceeded
- `ERR2004` (RESOURCE_EXHAUSTED): stream audio limit exceeded

### `ERR3xxx`: unexpected server exceptions

- `ERR3001` (UNKNOWN): unexpected `CreateSession` error
- `ERR3002` (UNKNOWN): unexpected `StreamingRecognize` error

### `ERR4xxx`: admin/control-plane HTTP failures

- `ERR4001` (HTTP 501): admin API not enabled
- `ERR4002` (HTTP 409): model already loaded
- `ERR4003` (HTTP 400): model not found or is default (unload failed)
- `ERR4004` (HTTP 401): invalid or missing admin token
- `ERR4005` (HTTP 403): `model_path` not allowed
- `ERR4009` (HTTP 400): unknown model load profile
