# Admin/Control Plane

The server provides an HTTP-based admin control plane for **runtime model management**.
These endpoints allow operators to manage Whisper models without restarting the gRPC server.

## Endpoints

- `POST /admin/load_model`
  Loads a Whisper model into the runtime. This can be used to pre-warm models or dynamically add new model variants.
  Prefer `profile_id` to select a pre-defined load profile from `model_load_profiles` in `config/model.yaml`.
  If no profile is provided and no legacy overrides are sent, the default profile is used.
  `backend` may be specified to select `faster_whisper` or `torch_whisper`.

- `POST /admin/unload_model`
  Unloads a previously loaded model to free CPU/GPU memory. Active sessions using the model must complete or fail gracefully.

- `GET /admin/list_models`
  Returns the list of currently loaded models and their runtime status.

- `GET /admin/load_model_status?model_id=...`
  Returns load job status (`queued`, `running`, `success`, `failed`) and error info when available.

### Load model (profile-based)

Define profiles in `config/model.yaml`:

```yaml
model:
  default_model_load_profile: "default"

model_load_profiles:
  default:
    model_size: "small"
    backend: "faster_whisper"
    device: "cpu"
    compute_type: "int8"
    pool_size: 1
```

Then load by profile:

```bash
curl -X POST "http://localhost:8000/admin/load_model" \
  -H "Authorization: Bearer $STT_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model_id":"small-rt","profile_id":"default"}'
```

Pick an environment-specific profile explicitly (examples):

```bash
# macOS (Apple Silicon)
curl -X POST "http://localhost:8000/admin/load_model" \
  -H "Authorization: Bearer $STT_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model_id":"small-mps","profile_id":"apple_silicon_mps"}'

# Linux + NVIDIA CUDA
curl -X POST "http://localhost:8000/admin/load_model" \
  -H "Authorization: Bearer $STT_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model_id":"small-cuda","profile_id":"nvidia_cuda"}'
```

**Note:** Admin endpoints are intended for operator use only and should be protected or restricted in production environments.

## Admin API security (recommended)

Admin endpoints are **disabled by default**. To enable them, set:

- `STT_ADMIN_ENABLED=true`
- `STT_ADMIN_TOKEN=<token>`

Requests must include an `Authorization: Bearer <token>` header.
Do not rely on IP allowlists alone; treat the admin plane as Internet-facing unless explicitly
firewalled/isolated.

To restrict model loading via `model_path`, set:

- `STT_ADMIN_ALLOW_MODEL_PATH=true` to allow `model_path`
- `STT_ADMIN_MODEL_PATH_ALLOWLIST` (comma-separated prefixes) to whitelist paths/IDs
  - Examples: `/models/`, `openai/`
