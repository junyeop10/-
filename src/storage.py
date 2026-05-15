"""SQLite 저장소와 통계 조회를 담당합니다."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class ClassificationRepository:
    """분류 시스템의 모든 DB 작업을 담당합니다."""

    def __init__(self, db_path: str | Path) -> None:
        """DB 경로를 저장하고 부모 폴더를 준비합니다."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        """row_factory가 설정된 SQLite 연결을 반환합니다."""
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize_database(self) -> None:
        """기본 테이블을 생성합니다."""
        schema = """
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL UNIQUE,
            file_name TEXT NOT NULL,
            file_ext TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            xxhash64 TEXT NOT NULL,
            duplicate_of_file_id INTEGER,
            extracted_text TEXT NOT NULL,
            text_length INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (duplicate_of_file_id) REFERENCES files(id)
        );

        CREATE TABLE IF NOT EXISTS classifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER NOT NULL,
            predicted_category TEXT NOT NULL,
            rule_score REAL NOT NULL,
            embedding_score REAL NOT NULL,
            llm_score REAL NOT NULL DEFAULT 0,
            final_score REAL NOT NULL,
            candidate_scores_json TEXT NOT NULL,
            reasoning TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'suggested',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (file_id) REFERENCES files(id)
        );

        CREATE TABLE IF NOT EXISTS feedback_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER NOT NULL,
            classification_id INTEGER NOT NULL,
            predicted_category TEXT NOT NULL,
            final_category TEXT NOT NULL,
            feedback_action TEXT NOT NULL,
            user_note TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (file_id) REFERENCES files(id),
            FOREIGN KEY (classification_id) REFERENCES classifications(id)
        );

        CREATE TABLE IF NOT EXISTS confirmed_examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER NOT NULL,
            category TEXT NOT NULL,
            source_text TEXT NOT NULL,
            embedding_json TEXT NOT NULL,
            source_feedback_log_id INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (file_id) REFERENCES files(id),
            FOREIGN KEY (source_feedback_log_id) REFERENCES feedback_logs(id)
        );

        CREATE TABLE IF NOT EXISTS rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            rule_type TEXT NOT NULL,
            pattern TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0,
            status TEXT NOT NULL DEFAULT 'active',
            source TEXT NOT NULL DEFAULT 'seed',
            evidence_json TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(category, rule_type, pattern)
        );
        """
        with self.connect() as connection:
            connection.executescript(schema)

    def seed_rules_from_categories(self, categories: dict[str, list[str]]) -> None:
        """categories.json 내용을 초기 keyword 규칙으로 저장합니다."""
        with self.connect() as connection:
            connection.execute("DELETE FROM rules WHERE source = 'categories_json'")
            for category, keywords in categories.items():
                for keyword in keywords:
                    connection.execute(
                        """
                        INSERT INTO rules (
                            category, rule_type, pattern, weight, status, source
                        )
                        VALUES (?, 'keyword', ?, 1.0, 'active', 'categories_json')
                        """,
                        (category, keyword),
                    )

    def upsert_file(
        self,
        file_path: str,
        file_name: str,
        file_ext: str,
        file_size: int,
        xxhash64: str,
        duplicate_of_file_id: int | None,
        extracted_text: str,
    ) -> int:
        """파일 메타데이터와 텍스트를 저장하거나 갱신합니다."""
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT id FROM files WHERE file_path = ?",
                (file_path,),
            ).fetchone()

            if existing:
                connection.execute(
                    """
                    UPDATE files
                    SET file_name = ?, file_ext = ?, file_size = ?, xxhash64 = ?,
                        duplicate_of_file_id = ?, extracted_text = ?, text_length = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (
                        file_name,
                        file_ext,
                        file_size,
                        xxhash64,
                        duplicate_of_file_id,
                        extracted_text,
                        len(extracted_text),
                        existing["id"],
                    ),
                )
                return int(existing["id"])

            cursor = connection.execute(
                """
                INSERT INTO files (
                    file_path, file_name, file_ext, file_size, xxhash64,
                    duplicate_of_file_id, extracted_text, text_length
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_path,
                    file_name,
                    file_ext,
                    file_size,
                    xxhash64,
                    duplicate_of_file_id,
                    extracted_text,
                    len(extracted_text),
                ),
            )
            return int(cursor.lastrowid)

    def find_duplicate_file_id(self, xxhash64: str, file_path: str) -> int | None:
        """같은 해시를 가진 이전 파일 id를 찾습니다."""
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT id
                FROM files
                WHERE xxhash64 = ? AND file_path <> ?
                ORDER BY id
                LIMIT 1
                """,
                (xxhash64, file_path),
            ).fetchone()
        return int(row["id"]) if row else None

    def insert_classification(
        self,
        file_id: int,
        predicted_category: str,
        rule_score: float,
        embedding_score: float,
        llm_score: float,
        final_score: float,
        candidate_scores_json: str,
        reasoning: str,
        status: str,
    ) -> int:
        """분류 결과를 classifications에 저장합니다."""
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO classifications (
                    file_id, predicted_category, rule_score, embedding_score, llm_score,
                    final_score, candidate_scores_json, reasoning, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    predicted_category,
                    rule_score,
                    embedding_score,
                    llm_score,
                    final_score,
                    candidate_scores_json,
                    reasoning,
                    status,
                ),
            )
            return int(cursor.lastrowid)

    def save_feedback(
        self,
        file_id: int,
        classification_id: int,
        predicted_category: str,
        final_category: str,
        feedback_action: str,
        user_note: str | None,
    ) -> int:
        """사용자 확정 또는 수정 로그를 저장합니다."""
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO feedback_logs (
                    file_id, classification_id, predicted_category,
                    final_category, feedback_action, user_note
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    classification_id,
                    predicted_category,
                    final_category,
                    feedback_action,
                    user_note,
                ),
            )
            connection.execute(
                "UPDATE classifications SET status = 'reviewed' WHERE id = ?",
                (classification_id,),
            )
            return int(cursor.lastrowid)

    def save_confirmed_example(
        self,
        file_id: int,
        category: str,
        source_text: str,
        embedding: list[float],
        source_feedback_log_id: int,
    ) -> int:
        """확정된 분류를 confirmed_examples에 저장합니다."""
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO confirmed_examples (
                    file_id, category, source_text, embedding_json, source_feedback_log_id
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    category,
                    source_text,
                    json.dumps(embedding, ensure_ascii=False),
                    source_feedback_log_id,
                ),
            )
            return int(cursor.lastrowid)

    def fetch_active_rules(self) -> list[sqlite3.Row]:
        """활성 규칙 목록을 반환합니다."""
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM rules WHERE status = 'active' ORDER BY category, id"
            ).fetchall()

    def fetch_confirmed_examples(self) -> list[sqlite3.Row]:
        """확정 예시와 파일명을 함께 반환합니다."""
        with self.connect() as connection:
            return connection.execute(
                """
                SELECT ce.*, f.file_name
                FROM confirmed_examples ce
                JOIN files f ON f.id = ce.file_id
                ORDER BY ce.id
                """
            ).fetchall()

    def list_categories(self) -> list[str]:
        """rules와 confirmed_examples에서 사용된 카테고리 목록을 반환합니다."""
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT category FROM rules
                UNION
                SELECT category FROM confirmed_examples
                ORDER BY category
                """
            ).fetchall()
        return [str(row["category"]) for row in rows]

    def get_feedback_adjustments(
        self,
        predicted_category: str,
        categories: list[str],
    ) -> dict[str, float]:
        """과거 수정 방향을 다음 분류에 반영할 점수로 계산합니다."""
        adjustments = {category: 0.0 for category in categories}
        if not predicted_category:
            return adjustments

        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT final_category, COUNT(*) AS count
                FROM feedback_logs
                WHERE predicted_category = ?
                GROUP BY final_category
                """,
                (predicted_category,),
            ).fetchall()

        total = sum(int(row["count"]) for row in rows)
        if total == 0:
            return adjustments

        for row in rows:
            adjustments[str(row["final_category"])] = round(int(row["count"]) / total, 4)
        return adjustments

    def get_duplicate_confirmed_category_scores(
        self,
        file_hash: str,
        duplicate_of_file_id: int | None,
        categories: list[str],
    ) -> dict[str, float]:
        """동일 파일이 이미 확정된 적 있으면 보정 점수를 줍니다."""
        scores = {category: 0.0 for category in categories}
        with self.connect() as connection:
            if duplicate_of_file_id is not None:
                rows = connection.execute(
                    "SELECT category FROM confirmed_examples WHERE file_id = ?",
                    (duplicate_of_file_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT ce.category
                    FROM files f
                    JOIN confirmed_examples ce ON ce.file_id = f.id
                    WHERE f.xxhash64 = ?
                    """,
                    (file_hash,),
                ).fetchall()

        for row in rows:
            scores[str(row["category"])] = 1.0
        return scores

    def fetch_correction_examples(self) -> list[sqlite3.Row]:
        """수정된 피드백 사례와 원문 텍스트를 반환합니다."""
        with self.connect() as connection:
            return connection.execute(
                """
                SELECT
                    fl.file_id,
                    fl.predicted_category,
                    fl.final_category,
                    f.file_name,
                    f.extracted_text
                FROM feedback_logs fl
                JOIN files f ON f.id = fl.file_id
                WHERE fl.feedback_action = 'corrected'
                ORDER BY fl.id
                """
            ).fetchall()

    def get_active_rule_tokens_by_category(self) -> dict[str, set[str]]:
        """활성 keyword 규칙을 카테고리별 집합으로 반환합니다."""
        tokens: dict[str, set[str]] = {}
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT category, pattern
                FROM rules
                WHERE status = 'active' AND rule_type = 'keyword'
                """
            ).fetchall()

        for row in rows:
            tokens.setdefault(str(row["category"]), set()).add(str(row["pattern"]).lower())
        return tokens

    def insert_rule_candidate(
        self,
        category: str,
        rule_type: str,
        pattern: str,
        weight: float,
        source: str,
        evidence: dict[str, Any],
        status: str = "candidate",
    ) -> None:
        """규칙 후보를 중복 없이 저장합니다."""
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO rules (
                    category, rule_type, pattern, weight, status, source, evidence_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    category,
                    rule_type,
                    pattern,
                    weight,
                    status,
                    source,
                    json.dumps(evidence, ensure_ascii=False),
                ),
            )

    def fetch_candidate_rules(self) -> list[sqlite3.Row]:
        """candidate 상태의 규칙을 반환합니다."""
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM rules WHERE status = 'candidate' ORDER BY id DESC"
            ).fetchall()

    def get_stats(self) -> dict[str, Any]:
        """CLI stats용 집계 정보와 최근 수정 로그를 반환합니다."""
        with self.connect() as connection:
            files_count = connection.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            classifications_count = connection.execute("SELECT COUNT(*) FROM classifications").fetchone()[0]
            feedback_logs_count = connection.execute("SELECT COUNT(*) FROM feedback_logs").fetchone()[0]
            confirmed_examples_count = connection.execute("SELECT COUNT(*) FROM confirmed_examples").fetchone()[0]
            rules_count = connection.execute("SELECT COUNT(*) FROM rules").fetchone()[0]
            recent_feedback = connection.execute(
                """
                SELECT fl.created_at, f.file_name, fl.predicted_category, fl.final_category, fl.feedback_action
                FROM feedback_logs fl
                JOIN files f ON f.id = fl.file_id
                ORDER BY fl.id DESC
                LIMIT 5
                """
            ).fetchall()

        return {
            "files_count": int(files_count),
            "classifications_count": int(classifications_count),
            "feedback_logs_count": int(feedback_logs_count),
            "confirmed_examples_count": int(confirmed_examples_count),
            "rules_count": int(rules_count),
            "recent_feedback": [dict(row) for row in recent_feedback],
        }
