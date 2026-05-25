import sqlite3
import time
from pathlib import Path
from typing import Optional

import xxhash

DB_PATH = Path(__file__).resolve().parent.parent / "cache.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS file_cache (
            xxhash TEXT PRIMARY KEY,
            category TEXT,
            confidence REAL,
            classified_at REAL
        )
        """
    )
    conn.commit()
    return conn


def compute_hash(file_bytes: bytes) -> str:
    return xxhash.xxh64(file_bytes).hexdigest()


def get_cached(file_hash: str) -> Optional[dict]:
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT category, confidence, classified_at FROM file_cache WHERE xxhash = ?",
            (file_hash,),
        ).fetchone()
        if row is None:
            return None
        return {
            "category": row[0],
            "confidence": row[1],
            "classified_at": row[2],
        }
    finally:
        conn.close()


def set_cache(file_hash: str, category: str, confidence: float) -> None:
    conn = _get_conn()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO file_cache (xxhash, category, confidence, classified_at)
            VALUES (?, ?, ?, ?)
            """,
            (file_hash, category, confidence, time.time()),
        )
        conn.commit()
    finally:
        conn.close()
