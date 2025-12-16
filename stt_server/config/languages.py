"""Helpers for loading the supported language list."""

import csv
import logging
from pathlib import Path
from typing import Dict, Optional, Set

LOGGER = logging.getLogger("stt_server")


class SupportedLanguages:
    """Lazy loader/cache for supported language codes and names."""

    def __init__(self) -> None:
        self._language_map: Dict[str, str] = {}

    def get_codes(self) -> Optional[Set[str]]:
        if not self._language_map:
            self._load()
        return set(self._language_map.keys()) if self._language_map else None

    def get_name(self, code: str) -> str:
        if not code:
            return ""
        if not self._language_map:
            self._load()
        return self._language_map.get(code.lower(), "")

    def _load(self) -> None:
        csv_path = (
            Path(__file__).resolve().parents[2]
            / "config"
            / "data"
            / "supported_languages.csv"
        )
        language_map: Dict[str, str] = {}
        try:
            with csv_path.open("r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    code = row.get("Language Code") or row.get("language_code")
                    name = row.get("Language Name") or row.get("language_name")
                    if not code:
                        continue
                    key = code.strip().lower()
                    language_map[key] = name.strip() if name else ""
        except FileNotFoundError:
            LOGGER.warning(
                "Supported language CSV not found at %s; skipping validation", csv_path
            )
            self._language_map = {}
            return
        self._language_map = language_map


__all__ = ["SupportedLanguages"]
