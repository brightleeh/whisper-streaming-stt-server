"""Bench wrapper for the shared monitoring dashboard."""

from tools.dashboard.monitor_dashboard import main

if __name__ == "__main__":
    if __package__ is None:
        raise SystemExit(
            "Run as a module from the repo root: python -m tools.bench.monitor_dashboard"
        )
    main()
