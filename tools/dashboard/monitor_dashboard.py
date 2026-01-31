import argparse
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import requests


def fetch_json(
    url: str, timeout: float
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json(), None
    except (requests.RequestException, ValueError) as exc:
        return None, str(exc)


def _format_bytes(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    size = float(value)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _format_float(value: Optional[float], suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}{suffix}"


def _render(metrics: Optional[Dict[str, Any]], system: Optional[Dict[str, Any]]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"STT Dashboard - {now}"]

    if metrics is None:
        lines.append("Metrics: unavailable")
    else:
        decode_total = metrics.get("decode_latency_total")
        decode_count = metrics.get("decode_latency_count")
        decode_max = metrics.get("decode_latency_max")
        queue_total = metrics.get("decode_queue_wait_total")
        queue_count = metrics.get("decode_queue_wait_count")
        queue_max = metrics.get("decode_queue_wait_max")
        decode_avg = (
            (decode_total / decode_count) if decode_total and decode_count else 0.0
        )
        queue_avg = (queue_total / queue_count) if queue_total and queue_count else 0.0
        lines.append(
            "Decode: count={count} avg={avg:.2f}s max={max:.2f}s queue_avg={queue_avg:.2f}s queue_max={queue_max:.2f}s cancelled={cancelled} orphaned={orphaned}".format(
                count=metrics.get("decode_latency_count", 0),
                avg=decode_avg,
                max=decode_max or 0.0,
                queue_avg=queue_avg,
                queue_max=queue_max or 0.0,
                cancelled=metrics.get("decode_cancelled", 0),
                orphaned=metrics.get("decode_orphaned", 0),
            )
        )
        lines.append(
            "RTF: avg={avg} max={max}".format(
                avg=_format_float(metrics.get("rtf_avg")),
                max=_format_float(metrics.get("rtf_max")),
            )
        )
        queue_depth = metrics.get("decode_queue_depth")
        if queue_depth is None:
            queue_depth = "n/a"
        lines.append(
            "Sessions: active={active} queue={queue} vad_triggers={vad} active_vad={active_vad}".format(
                active=metrics.get("active_sessions", 0),
                queue=queue_depth,
                vad=metrics.get("vad_triggers_total", metrics.get("vad_triggers", 0)),
                active_vad=metrics.get("active_vad_utterances", 0),
            )
        )
        error_counts = metrics.get("error_counts", {})
        if error_counts:
            errors = ", ".join(
                f"{code}:{count}" for code, count in sorted(error_counts.items())
            )
            lines.append(f"Errors: {errors}")

    if system is None:
        lines.append("System: unavailable")
    else:
        process = system.get("process", {})
        system_mem = system.get("system", {})
        lines.append(
            "Process: cpu={cpu}% rss={rss} threads={threads}".format(
                cpu=_format_float(process.get("cpu_percent")),
                rss=_format_bytes(process.get("rss_bytes")),
                threads=process.get("threads", "n/a"),
            )
        )
        lines.append(
            "System: mem={mem}% avail={avail} total={total} load={load}".format(
                mem=_format_float(system_mem.get("memory_percent")),
                avail=_format_bytes(system_mem.get("memory_available_bytes")),
                total=_format_bytes(system_mem.get("memory_total_bytes")),
                load=system.get("load_avg") or "n/a",
            )
        )
        gpus = system.get("gpus") or []
        if gpus:
            for gpu in gpus:
                lines.append(
                    "GPU[{index}] {name}: util={util}% mem={mem_used}/{mem_total}MiB temp={temp}C".format(
                        index=gpu.get("index"),
                        name=gpu.get("name"),
                        util=gpu.get("utilization_gpu"),
                        mem_used=gpu.get("memory_used_mib"),
                        mem_total=gpu.get("memory_total_mib"),
                        temp=gpu.get("temperature_gpu"),
                    )
                )
        else:
            lines.append("GPU: n/a")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal dashboard for STT metrics.")
    parser.add_argument("--metrics-url", default="http://localhost:8000/metrics.json")
    parser.add_argument("--system-url", default="http://localhost:8000/system")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--no-clear", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        metrics, metrics_error = fetch_json(args.metrics_url, args.timeout)
        system, system_error = fetch_json(args.system_url, args.timeout)
        if metrics_error:
            metrics = None
        if system_error:
            system = None

        output = _render(metrics, system)
        if not args.no_clear:
            print("\033[2J\033[H", end="")
        print(output)

        if args.once:
            break
        time.sleep(max(args.interval, 0.1))


if __name__ == "__main__":
    if __package__ is None:
        raise SystemExit(
            "Run as a module from the repo root: python -m tools.dashboard.monitor_dashboard"
        )
    main()
