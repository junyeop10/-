"""SQLite OCR/text extraction cache keyed by file hash and OCR version."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional


class OcrCache:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def get_cached_ocr(self, file_hash: str, ocr_engine: str, ocr_version: str) -> Optional[dict]:
        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT *
                FROM ocr_cache_v2
                WHERE file_hash = ? AND ocr_engine = ? AND ocr_version = ?
                """,
                (file_hash, ocr_engine, ocr_version),
            ).fetchone()
        if row is None:
            return None
        payload = dict(row)
        payload["metadata"] = json.loads(str(payload.pop("metadata_json") or "{}"))
        return payload

    def save_ocr_cache(self, file_hash: str, ocr_engine: str, ocr_version: str, result: dict) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO ocr_cache_v2 (
                    file_hash, ocr_engine, ocr_version, raw_text, cleaned_text, page_count,
                    quality_score, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(file_hash, ocr_engine, ocr_version) DO UPDATE SET
                    raw_text = excluded.raw_text,
                    cleaned_text = excluded.cleaned_text,
                    page_count = excluded.page_count,
                    quality_score = excluded.quality_score,
                    metadata_json = excluded.metadata_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    file_hash,
                    ocr_engine,
                    ocr_version,
                    str(result.get("raw_text", result.get("text", ""))),
                    str(result.get("cleaned_text", result.get("text", ""))),
                    int(result.get("page_count", result.get("pages_scanned", 0)) or 0),
                    float(result.get("quality_score", result.get("confidence", 0.0)) or 0.0),
                    json.dumps(result.get("metadata", {}), ensure_ascii=False),
                ),
            )

    def invalidate_ocr_cache(
        self,
        file_hash: str | None = None,
        engine: str | None = None,
        version: str | None = None,
    ) -> int:
        clauses: list[str] = []
        params: list[str] = []
        if file_hash:
            clauses.append("file_hash = ?")
            params.append(file_hash)
        if engine:
            clauses.append("ocr_engine = ?")
            params.append(engine)
        if version:
            clauses.append("ocr_version = ?")
            params.append(version)
        query = "DELETE FROM ocr_cache_v2"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(query, tuple(params))
            return int(cursor.rowcount)

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ocr_cache_v2 (
                    file_hash TEXT NOT NULL,
                    ocr_engine TEXT NOT NULL,
                    ocr_version TEXT NOT NULL,
                    raw_text TEXT NOT NULL DEFAULT '',
                    cleaned_text TEXT NOT NULL DEFAULT '',
                    page_count INTEGER NOT NULL DEFAULT 0,
                    quality_score REAL NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(file_hash, ocr_engine, ocr_version)
                )
                """
            )
