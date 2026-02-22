# Whisper Web Client

Browser client for live STT using the existing WebSocket bridge (`/ws/stream`).

## What It Covers

- One-hand friendly UI (large bottom buttons)
- Simple input: microphone or WAV file
- Live status feedback: connection/capture/transfer/result
- Low power/data profile (`Network profile = Low power/data`)
- Theme modes: `System` / `Dark` / `Light`
- PWA install option (manifest + service worker)

## Run

1. Start STT server with WebSocket enabled:

```bash
python -m stt_server.main --model small --device mps --model-backend torch_whisper
```

2. Ensure WebSocket bind host is reachable from your client device.

- Default in `config/server.yaml` is `ws_host: 127.0.0.1` (local-only).
- For LAN/VPN testing, set `ws_host: 0.0.0.0` and restart server.

3. (Quick file-only test) Serve static files:

```bash
cd stt_client/web_mobile
python -m http.server 8082
```

4. Open from device browser:

- `http://<server-ip>:8082`
- Confirm `WebSocket URL` points to `ws://<server-ip>:8001/ws/stream`

## Microphone Requirement (Browser)

Microphone capture in modern browsers needs a secure context:

- Page must be served over `https://`
- WebSocket must be `wss://`

The built-in WS server (`:8001`) is plain WS, so for mic mode in secure browser contexts,
place a TLS reverse proxy in front and proxy `/ws/stream` to `127.0.0.1:8001`.

Default URL behavior in app:

- `http://...` page -> `ws://<host>:8001/ws/stream`
- `https://...` page -> `wss://<same-host>/ws/stream`

## HTTPS + WSS with Caddy

1. Install Caddy (macOS):

```bash
brew install caddy
```

2. Prepare TLS cert/key in `stt_client/web_mobile/certs/`:

Option A (general local dev): `mkcert`

```bash
brew install mkcert nss
mkcert -install
mkcert \
  -cert-file stt_client/web_mobile/certs/mobile.crt \
  -key-file stt_client/web_mobile/certs/mobile.key \
  localhost 127.0.0.1 ::1 <your-hostname>
```

Option B (VPN example): Tailscale cert

```bash
DNS_NAME=$(tailscale status --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))')
tailscale cert \
  --cert-file stt_client/web_mobile/certs/mobile.crt \
  --key-file stt_client/web_mobile/certs/mobile.key \
  "$DNS_NAME"
```

Option C (production): existing cert (Let's Encrypt / internal PKI)

- Copy your cert/key to:
- `stt_client/web_mobile/certs/mobile.crt`
- `stt_client/web_mobile/certs/mobile.key`

3. Start Caddy:

```bash
cd stt_client/web_mobile
./run_with_caddy.sh
```

4. Open from browser:

- `https://<your-hostname>:8443`

The URL hostname must match certificate SAN/CN.

5. Optional quick public HTTPS for testing:

- Use ngrok / Cloudflare Tunnel to expose local `:8443` without network port forwarding.
- Keep the app origin as HTTPS and confirm WS URL is `wss://<same-host>/ws/stream`.

Environment overrides:

- `MOBILE_WEB_ADDR` (default `:8443`)
- `MOBILE_WEB_ROOT` (default `stt_client/web_mobile`)
- `MOBILE_WS_UPSTREAM` (default `127.0.0.1:8001`)
- `MOBILE_TLS_CERT` (default `stt_client/web_mobile/certs/mobile.crt`)
- `MOBILE_TLS_KEY` (default `stt_client/web_mobile/certs/mobile.key`)

## Notes

- HTTPS pages require `wss://` WebSocket URL.
- iOS/Safari requires explicit user gesture before microphone capture.
- File mode expects PCM WAV (or WAV decodable by browser AudioContext).
- Theme can be switched in `Advanced -> Theme` (`System`, `Dark`, `Light`).
