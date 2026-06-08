"""Small JSON cache for reusable document evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

from src.hash_utils import compute_raw_text_hash


EVIDENCE_CACHE_VERSION = "type-evidence-xxhash-v2"


class EvidenceCache:
    """Persist evidence by content, filename, and extraction settings."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def get(self, *, file_hash: str, filename: str, settings_key: str) -> dict[str, Any] | None:
        path = self._entry_path(file_hash=file_hash, filename=filename, settings_key=settings_key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if (
            str(payload.get("file_hash", "")) != file_hash
            or str(payload.get("filename", "")) != filename
            or str(payload.get("_evidence_cache_settings", "")) != settings_key
        ):
            return None
        return payload

    def save(self, payload: dict[str, Any], *, settings_key: str) -> None:
        file_hash = str(payload.get("file_hash", ""))
        filename = str(payload.get("filename", ""))
        if not file_hash or not filename:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._entry_path(file_hash=file_hash, filename=filename, settings_key=settings_key)
        cached_payload = {**payload, "_evidence_cache_settings": settings_key}
        temporary = path.with_suffix(f".{os.getpid()}.{time.time_ns()}.tmp")
        try:
            temporary.write_text(json.dumps(cached_payload, ensure_ascii=False), encoding="utf-8")
            temporary.replace(path)
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def _entry_path(self, *, file_hash: str, filename: str, settings_key: str) -> Path:
        signature = compute_raw_text_hash(f"{file_hash}|{filename}|{settings_key}")
        return self.directory / f"{signature}.json"
