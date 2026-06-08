"""SQLite embedding cache keyed by cleaned text hash and model version."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np


class EmbeddingCache:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def get_cached_embedding(
        self,
        text_hash: str,
        embedding_model: str,
        embedding_version: str,
    ) -> Optional[np.ndarray]:
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT vector_blob, dtype, dimension
                FROM embedding_cache_v2
                WHERE text_hash = ? AND embedding_model = ? AND embedding_version = ?
                """,
                (text_hash, embedding_model, embedding_version),
            ).fetchone()
            if row is None:
                return None
            blob, dtype, dimension = row
            vector = np.frombuffer(blob, dtype=np.dtype(dtype)).astype(np.float32, copy=False)
            return vector.reshape(int(dimension))

    def save_embedding_cache(
        self,
        text_hash: str,
        embedding_model: str,
        embedding_version: str,
        embedding: np.ndarray,
    ) -> None:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO embedding_cache_v2 (
                    text_hash, embedding_model, embedding_version, dimension, vector_blob, dtype,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'float32', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(text_hash, embedding_model, embedding_version) DO UPDATE SET
                    dimension = excluded.dimension,
                    vector_blob = excluded.vector_blob,
                    dtype = excluded.dtype,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (text_hash, embedding_model, embedding_version, int(vector.shape[0]), vector.tobytes()),
            )

    def invalidate_embedding_cache(
        self,
        text_hash: str | None = None,
        model: str | None = None,
        version: str | None = None,
    ) -> int:
        clauses: list[str] = []
        params: list[str] = []
        if text_hash:
            clauses.append("text_hash = ?")
            params.append(text_hash)
        if model:
            clauses.append("embedding_model = ?")
            params.append(model)
        if version:
            clauses.append("embedding_version = ?")
            params.append(version)
        query = "DELETE FROM embedding_cache_v2"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(query, tuple(params))
            return int(cursor.rowcount)

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache_v2 (
                    text_hash TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_version TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    vector_blob BLOB NOT NULL,
                    dtype TEXT NOT NULL DEFAULT 'float32',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(text_hash, embedding_model, embedding_version)
                )
                """
            )
