"""Module entry point for the web dashboard controller."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("WEB_DASHBOARD_HOST", "0.0.0.0")
    port = int(os.getenv("WEB_DASHBOARD_PORT", "8010"))
    reload = os.getenv("WEB_DASHBOARD_RELOAD", "1") == "1"
    uvicorn.run(
        "tools.web_dashboard.app:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    main()
