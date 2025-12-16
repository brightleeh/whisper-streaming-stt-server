"""Internal helpers for the stt_server package."""

import sys
from pathlib import Path

# Ensure the project root (parent of this package) is importable when running
# entry points via ``python stt_server/main.py``.
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

__all__ = ["PACKAGE_DIR", "PROJECT_ROOT"]
