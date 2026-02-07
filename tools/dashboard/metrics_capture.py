"""Capture metrics/system snapshots to a file for graphing."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from typing import Any, Dict, Optional

import requests


def fetch_json(
    url: str, timeout: float, token: Optional[str]
) -> Optional[Dict[str, Any]]:
    if not url:
        return None
    try:
        headers = {}
        if token:
            headers["authorization"] = f"Bearer {token}"
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        return None


def _flatten(value: Any, prefix: str, out: Dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten(item, next_prefix, out)
        return
    if isinstance(value, list):
        for idx, item in enumerate(value):
            next_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            _flatten(item, next_prefix, out)
        return
    if prefix:
        out[prefix] = value


def _flatten_payload(payload: Optional[Dict[str, Any]], prefix: str) -> Dict[str, Any]:
    if payload is None:
        return {}
    out: Dict[str, Any] = {}
    _flatten(payload, prefix, out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture metrics to JSONL/CSV.")
    parser.add_argument("--metrics-url", default="http://localhost:8000/metrics.json")
    parser.add_argument("--system-url", default="http://localhost:8000/system")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--format", choices=("jsonl", "csv"), default="jsonl")
    parser.add_argument("--out", default="metrics_capture.jsonl")
    parser.add_argument("--append", action="store_true")
    parser.add_argument(
        "--token",
        default="",
        help="Observability token (or set STT_OBSERVABILITY_TOKEN).",
    )
    parser.add_argument(
        "--no-flatten",
        action="store_true",
        help="Store nested JSON (jsonl only).",
    )
    args = parser.parse_args()

    mode = "a" if args.append else "w"
    started = time.time()

    token = args.token.strip() or None

    if args.format == "csv":
        import csv

        header_written = False
        fieldnames: list[str] = []
        with open(args.out, mode, encoding="utf-8", newline="") as handle:
            writer = None
            while True:
                now = time.time()
                metrics = fetch_json(args.metrics_url, args.timeout, token)
                system = fetch_json(args.system_url, args.timeout, token)
                row: Dict[str, Any] = {
                    "ts": now,
                    "iso": datetime.fromtimestamp(now).isoformat(),
                }
                row.update(_flatten_payload(metrics, "metrics"))
                row.update(_flatten_payload(system, "system"))
                if not header_written:
                    fieldnames = list(row.keys())
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    header_written = True
                if writer is None:
                    break
                writer.writerow({key: row.get(key) for key in fieldnames})
                handle.flush()
                if args.once:
                    break
                if (
                    args.duration_sec is not None
                    and (time.time() - started) >= args.duration_sec
                ):
                    break
                time.sleep(max(args.interval, 0.1))
        return 0

    with open(args.out, mode, encoding="utf-8") as handle:
        while True:
            now = time.time()
            metrics = fetch_json(args.metrics_url, args.timeout, token)
            system = fetch_json(args.system_url, args.timeout, token)
            if args.no_flatten:
                record = {
                    "ts": now,
                    "iso": datetime.fromtimestamp(now).isoformat(),
                    "metrics": metrics,
                    "system": system,
                }
            else:
                record = {
                    "ts": now,
                    "iso": datetime.fromtimestamp(now).isoformat(),
                }
                record.update(_flatten_payload(metrics, "metrics"))
                record.update(_flatten_payload(system, "system"))
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            handle.flush()
            if args.once:
                break
            if (
                args.duration_sec is not None
                and (time.time() - started) >= args.duration_sec
            ):
                break
            time.sleep(max(args.interval, 0.1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
