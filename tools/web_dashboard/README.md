# Whisper Ops Web Dashboard

This tool provides a FastAPI controller plus a Next.js UI for running gRPC load tests
against the Whisper STT server and streaming KPIs over SSE.

## Run the controller

```bash
pip install -r requirements.txt
python -m tools.web_dashboard
```

Environment variables:

- `WEB_DASHBOARD_HOST` (default: `0.0.0.0`)
- `WEB_DASHBOARD_PORT` (default: `8010`)
- `WEB_DASHBOARD_RELOAD` (default: `1`)

## Run the frontend

```bash
cd tools/web_dashboard/frontend
pnpm install
pnpm dev
```

## Run both (helper script)

```bash
./tools/web_dashboard/run_load_dashboard.sh
```

Remote access:

```bash
WEB_DASHBOARD_PUBLIC_HOST="<server_ip>" ./tools/web_dashboard/run_load_dashboard.sh
```

## Targets

Edit `tools/web_dashboard/targets.json` to define gRPC + HTTP endpoints.

## Notes

- The controller uses `tools/bench/grpc_load_test.py` as a subprocess.
- Session logs are written to `runs/<run_id>/sessions.jsonl`.
- KPI and resource events are streamed via SSE at `/runs/{run_id}/live`.
