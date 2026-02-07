"""Plot metrics_capture JSONL output."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _load_jsonl(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _default_fields(records: Iterable[Dict[str, object]]) -> List[str]:
    candidates = [
        "system.process.rss_bytes",
        "metrics.buffer_bytes_total",
        "metrics.decode_pending",
        "metrics.partial_drop_count",
        "metrics.rate_limit_blocks.stream",
    ]
    available = set()
    for record in records:
        available.update(record.keys())
    return [field for field in candidates if field in available]


def _series(
    records: List[Dict[str, object]], field: str
) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for record in records:
        ts = record.get("ts")
        if isinstance(ts, (int, float)):
            xs.append(float(ts))
        else:
            continue
        value = record.get(field)
        if isinstance(value, (int, float)):
            ys.append(float(value))
        else:
            ys.append(float("nan"))
    return xs, ys


def _scale_bytes(values: List[float]) -> List[float]:
    return [
        value / (1024.0 * 1024.0) if math.isfinite(value) else value for value in values
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot metrics JSONL output.")
    parser.add_argument("path", help="Path to metrics.jsonl")
    parser.add_argument(
        "--fields",
        default="",
        help="Comma-separated field list (default: common buffer/queue/RSS fields).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Write to image file instead of showing interactively.",
    )
    parser.add_argument(
        "--relative",
        action="store_true",
        help="Plot elapsed seconds on X-axis.",
    )
    parser.add_argument(
        "--title",
        default="STT Metrics",
        help="Chart title.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Downsample to this many points (0 disables).",
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required: python -m pip install matplotlib"
        ) from exc

    records = _load_jsonl(Path(args.path))
    if not records:
        raise SystemExit("No records found in JSONL.")

    if args.fields:
        fields = [field.strip() for field in args.fields.split(",") if field.strip()]
    else:
        fields = _default_fields(records)
    if not fields:
        raise SystemExit("No matching fields found. Use --fields to select.")

    fig, axes = plt.subplots(
        nrows=len(fields),
        ncols=1,
        sharex=True,
        figsize=(12, max(3, len(fields) * 2.2)),
    )
    if len(fields) == 1:
        axes = [axes]

    for axis, field in zip(axes, fields):
        xs, ys = _series(records, field)
        if not xs:
            continue
        if args.max_points and len(xs) > args.max_points:
            step = max(1, len(xs) // args.max_points)
            xs = xs[::step]
            ys = ys[::step]
        if args.relative:
            start = xs[0]
            xs = [value - start for value in xs]
        label = field
        if "bytes" in field:
            ys = _scale_bytes(ys)
            label = f"{field} (MiB)"
        axis.plot(xs, ys, linewidth=1.2)
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)

    axes[-1].set_xlabel("elapsed_sec" if args.relative else "timestamp")
    fig.suptitle(args.title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    if args.output:
        fig.savefig(args.output, dpi=160)
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
