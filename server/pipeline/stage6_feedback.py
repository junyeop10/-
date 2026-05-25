import hashlib
import json
import sqlite3
import time
from pathlib import Path

from models.schemas import Category, ClassifyResult, FeedbackLog, FinalizedDocument

DB_PATH = Path(__file__).resolve().parent.parent / "cache.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            xxhash TEXT,
            embedding TEXT,
            system_category TEXT,
            user_category TEXT,
            corrected INTEGER,
            correction_stage TEXT,
            timestamp REAL
        )
        """
    )
    conn.commit()
    return conn


def save_feedback(log: FeedbackLog) -> None:
    conn = _get_conn()
    try:
        conn.execute(
            """
            INSERT INTO feedback_log
            (xxhash, embedding, system_category, user_category, corrected,
             correction_stage, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                log.xxhash,
                json.dumps(log.embedding),
                log.system_category.value,
                log.user_category.value,
                1 if log.corrected else 0,
                log.correction_stage,
                log.timestamp,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def finalize_document(result: ClassifyResult) -> FinalizedDocument:
    sha256 = ""
    if result.category == Category.FINAL and result.file_path:
        path = Path(result.file_path)
        if path.exists():
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            sha256 = h.hexdigest()

    return FinalizedDocument(
        file_path=result.file_path,
        xxhash=result.xxhash,
        sha256=sha256,
        category=result.category,
        finalized_at=time.time(),
    )


def verify_final_document(doc: FinalizedDocument) -> bool:
    if not doc.sha256:
        return True
    path = Path(doc.file_path)
    if not path.exists():
        return False
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest() == doc.sha256


def get_feedback_embeddings() -> list[dict]:
    conn = _get_conn()
    try:
        rows = conn.execute(
            """
            SELECT embedding, user_category FROM feedback_log
            WHERE corrected = 1
            """
        ).fetchall()
        result = []
        for emb_json, category in rows:
            try:
                embedding = json.loads(emb_json)
            except json.JSONDecodeError:
                continue
            if embedding:
                result.append({"embedding": embedding, "category": category})
        return result
    finally:
        conn.close()
