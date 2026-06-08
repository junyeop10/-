"""SQLite repository with additive enterprise MVP schema support."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from src.category_profiles import (
    DEFAULT_CATEGORY_PROFILES,
    build_category_profile_signature,
    build_synthetic_training_rows,
)
from src.embedding_repository import EmbeddingRepository


SCHEMA_VERSION = "2.0"


class ClassificationRepository:
    """Repository for classification, learning, and move operation state."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_repository: EmbeddingRepository | None = None

    def attach_embedding_repository(self, embedding_repository: EmbeddingRepository | None) -> None:
        self.embedding_repository = embedding_repository

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def initialize_database(self) -> None:
        """Create the base schema and run additive migrations."""
        base_schema = """
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
            connection.executescript(base_schema)
            self._apply_migrations(connection)

    def _apply_migrations(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT NOT NULL UNIQUE,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS move_batches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_mode TEXT NOT NULL,
                status TEXT NOT NULL,
                manifest_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS move_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id INTEGER NOT NULL,
                file_id INTEGER NOT NULL,
                classification_id INTEGER NOT NULL,
                source_path TEXT NOT NULL,
                destination_path TEXT NOT NULL,
                actual_destination_path TEXT,
                restored_path TEXT,
                large_category TEXT NOT NULL,
                middle_category TEXT NOT NULL,
                small_category TEXT,
                confidence REAL NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                error_message TEXT NOT NULL DEFAULT '',
                moved_at TEXT,
                restored_at TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (batch_id) REFERENCES move_batches(id),
                FOREIGN KEY (file_id) REFERENCES files(id),
                FOREIGN KEY (classification_id) REFERENCES classifications(id)
            );

            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id TEXT NOT NULL UNIQUE,
                snapshot_type TEXT NOT NULL,
                manifest_json TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS operation_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_type TEXT NOT NULL,
                batch_id INTEGER,
                status TEXT NOT NULL,
                details_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS adaptive_rule_boosts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                token TEXT NOT NULL,
                boost REAL NOT NULL,
                source TEXT NOT NULL,
                support_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(category, token, source)
            );

            CREATE TABLE IF NOT EXISTS ocr_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash TEXT NOT NULL,
                cache_key TEXT NOT NULL UNIQUE,
                backend TEXT NOT NULL,
                pages_scanned INTEGER NOT NULL DEFAULT 0,
                extracted_text TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS embedding_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cache_key TEXT NOT NULL UNIQUE,
                file_hash TEXT NOT NULL,
                model_name TEXT NOT NULL,
                text_signature TEXT NOT NULL,
                embedding_version TEXT NOT NULL,
                text_kind TEXT NOT NULL DEFAULT 'query',
                embedding_json TEXT NOT NULL,
                vector_dim INTEGER NOT NULL DEFAULT 0,
                hit_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_accessed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS llm_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                classification_id INTEGER,
                provider TEXT NOT NULL,
                model_name TEXT NOT NULL,
                decision_json TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS config_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_name TEXT NOT NULL,
                version TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS document_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                file_hash TEXT NOT NULL,
                extractor_version TEXT NOT NULL,
                filename_features_json TEXT NOT NULL DEFAULT '{}',
                metadata_features_json TEXT NOT NULL DEFAULT '{}',
                structural_features_json TEXT NOT NULL DEFAULT '{}',
                layout_features_json TEXT NOT NULL DEFAULT '{}',
                text_stats_json TEXT NOT NULL DEFAULT '{}',
                compressed_text TEXT NOT NULL DEFAULT '',
                compressed_text_hash TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(file_id, extractor_version),
                FOREIGN KEY (file_id) REFERENCES files(id)
            );

            CREATE TABLE IF NOT EXISTS document_vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                vector_type TEXT NOT NULL,
                vector_key TEXT NOT NULL DEFAULT '',
                vector_json TEXT NOT NULL DEFAULT '',
                model_version TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(file_id, vector_type, model_version),
                FOREIGN KEY (file_id) REFERENCES files(id)
            );

            CREATE TABLE IF NOT EXISTS model_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                trained_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                training_count INTEGER NOT NULL DEFAULT 0,
                metrics_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS category_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                suggested_name TEXT NOT NULL,
                representative_file_ids_json TEXT NOT NULL,
                evidence_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS document_tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                tag TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(file_id, tag, source),
                FOREIGN KEY (file_id) REFERENCES files(id)
            );

            CREATE TABLE IF NOT EXISTS category_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                profile_text TEXT NOT NULL,
                tags_json TEXT NOT NULL DEFAULT '[]',
                profile_signals_json TEXT NOT NULL DEFAULT '{}',
                lexical_profile_json TEXT NOT NULL DEFAULT '{}',
                profile_origin TEXT NOT NULL DEFAULT 'user',
                weight REAL NOT NULL DEFAULT 0.5,
                synthetic_count INTEGER NOT NULL DEFAULT 5,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS document_cache (
                file_hash TEXT PRIMARY KEY,
                file_path TEXT NOT NULL DEFAULT '',
                file_name TEXT NOT NULL DEFAULT '',
                file_size INTEGER NOT NULL DEFAULT 0,
                extension TEXT NOT NULL DEFAULT '',
                raw_text TEXT NOT NULL DEFAULT '',
                cleaned_text TEXT NOT NULL DEFAULT '',
                text_hash TEXT NOT NULL DEFAULT '',
                ocr_engine TEXT NOT NULL DEFAULT '',
                ocr_version TEXT NOT NULL DEFAULT '',
                ocr_quality_score REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS unknown_pool (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash TEXT NOT NULL,
                text_hash TEXT NOT NULL DEFAULT '',
                summary_text TEXT NOT NULL DEFAULT '',
                cleaned_text TEXT NOT NULL DEFAULT '',
                embedding_ref TEXT NOT NULL DEFAULT '',
                nearest_category TEXT NOT NULL DEFAULT '',
                nearest_similarity REAL NOT NULL DEFAULT 0,
                reason TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(file_hash, reason)
            );

            CREATE TABLE IF NOT EXISTS unsupervised_cluster_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                algorithm TEXT NOT NULL,
                parameters_json TEXT NOT NULL DEFAULT '{}',
                metrics_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'completed',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS unsupervised_cluster_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                unknown_pool_id INTEGER NOT NULL,
                cluster_id INTEGER NOT NULL,
                score REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES unsupervised_cluster_runs(id),
                FOREIGN KEY (unknown_pool_id) REFERENCES unknown_pool(id)
            );
            """
        )

        self._ensure_column(connection, "classifications", "large_category", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column(connection, "classifications", "middle_category", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column(connection, "classifications", "small_category", "TEXT")
        self._ensure_column(connection, "classifications", "large_confidence", "REAL NOT NULL DEFAULT 0")
        self._ensure_column(connection, "classifications", "middle_confidence", "REAL NOT NULL DEFAULT 0")
        self._ensure_column(connection, "classifications", "small_confidence", "REAL NOT NULL DEFAULT 0")
        self._ensure_column(connection, "classifications", "source_scores_json", "TEXT NOT NULL DEFAULT '{}'")
        self._ensure_column(connection, "classifications", "explanation_json", "TEXT NOT NULL DEFAULT '{}'")
        self._ensure_column(connection, "classifications", "evidence_json", "TEXT NOT NULL DEFAULT '{}'")
        self._ensure_column(connection, "classifications", "performance_json", "TEXT NOT NULL DEFAULT '{}'")
        self._ensure_column(connection, "classifications", "classifier_version", "TEXT NOT NULL DEFAULT '2.0'")
        self._ensure_column(connection, "classifications", "config_version", "TEXT NOT NULL DEFAULT '2.0'")
        self._ensure_column(connection, "classifications", "predicted_type", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column(connection, "classifications", "type_confidence", "REAL NOT NULL DEFAULT 0")
        self._ensure_column(connection, "classifications", "review_reasons_json", "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column(connection, "classifications", "suggested_tags_json", "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column(connection, "classifications", "cluster_candidate_id", "INTEGER")
        self._ensure_column(connection, "classifications", "ml_evidence_json", "TEXT NOT NULL DEFAULT '{}'")
        self._ensure_column(connection, "classifications", "rule_evidence_json", "TEXT NOT NULL DEFAULT '{}'")
        self._ensure_column(connection, "classifications", "lexical_score", "REAL NOT NULL DEFAULT 0")
        self._ensure_column(connection, "classifications", "layout_score", "REAL NOT NULL DEFAULT 0")
        self._ensure_column(connection, "document_features", "layout_features_json", "TEXT NOT NULL DEFAULT '{}'")
        self._ensure_column(connection, "category_profiles", "profile_signals_json", "TEXT NOT NULL DEFAULT '{}'")
        self._ensure_column(connection, "category_profiles", "lexical_profile_json", "TEXT NOT NULL DEFAULT '{}'")
        self._ensure_column(connection, "category_profiles", "profile_origin", "TEXT NOT NULL DEFAULT 'user'")
        self._ensure_column(connection, "document_cache", "ocr_quality_score", "REAL NOT NULL DEFAULT 0")
        self._ensure_column(connection, "unknown_pool", "embedding_ref", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column(connection, "unknown_pool", "status", "TEXT NOT NULL DEFAULT 'pending'")
        self._ensure_column(connection, "rules", "rule_scope", "TEXT NOT NULL DEFAULT 'content'")
        self._ensure_column(connection, "rules", "negative_weight", "REAL NOT NULL DEFAULT 0")

        self._ensure_column(connection, "feedback_logs", "predicted_large_category", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column(connection, "feedback_logs", "predicted_middle_category", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column(connection, "feedback_logs", "predicted_small_category", "TEXT")
        self._ensure_column(connection, "feedback_logs", "final_large_category", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column(connection, "feedback_logs", "final_middle_category", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column(connection, "feedback_logs", "final_small_category", "TEXT")
        self._ensure_column(connection, "feedback_logs", "evidence_text", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column(connection, "feedback_logs", "metadata_json", "TEXT NOT NULL DEFAULT '{}'")
        self._ensure_column(connection, "feedback_logs", "source_scores_json", "TEXT NOT NULL DEFAULT '{}'")
        self._ensure_column(connection, "feedback_logs", "classifier_version", "TEXT NOT NULL DEFAULT '2.0'")
        self._ensure_column(connection, "feedback_logs", "config_version", "TEXT NOT NULL DEFAULT '2.0'")
        self._ensure_column(connection, "feedback_logs", "ocr_used", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column(connection, "feedback_logs", "llm_used", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column(connection, "feedback_logs", "confirmation_batch_id", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column(connection, "feedback_logs", "confirmation_batch_name", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column(connection, "confirmed_examples", "embedding_key", "TEXT NOT NULL DEFAULT ''")

        existing = connection.execute("SELECT version FROM schema_migrations WHERE version = ?", (SCHEMA_VERSION,)).fetchone()
        if existing is None:
            connection.execute(
                "INSERT INTO schema_migrations (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )

    def _ensure_column(self, connection: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        columns = {
            row["name"]
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def seed_rules_from_categories(self, categories: dict[str, list[str]]) -> None:
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

    def seed_default_category_profiles(self) -> int:
        inserted = 0
        with self.connect() as connection:
            for profile in DEFAULT_CATEGORY_PROFILES:
                profile_signals = profile.get("profile_signals", {})
                lexical_profile = self._build_lexical_profile_payload(profile)
                aliases = profile_signals.get("aliases", []) if isinstance(profile_signals, dict) else []
                existing = connection.execute(
                    "SELECT id, profile_signals_json, profile_origin FROM category_profiles WHERE type = ? LIMIT 1",
                    (str(profile["type"]),),
                ).fetchone()
                if existing is None and aliases:
                    alias_placeholders = ",".join("?" for _ in aliases)
                    existing = connection.execute(
                        f"""
                        SELECT id, profile_signals_json, profile_origin
                        FROM category_profiles
                        WHERE profile_origin = 'default_seed'
                          AND type IN ({alias_placeholders})
                        LIMIT 1
                        """,
                        tuple(str(alias) for alias in aliases),
                    ).fetchone()
                if existing:
                    if str(existing["profile_origin"] or "") == "default_seed":
                        connection.execute(
                            """
                            UPDATE category_profiles
                            SET type = ?,
                                profile_text = ?,
                                tags_json = ?,
                                profile_signals_json = ?,
                                lexical_profile_json = ?,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE id = ?
                            """,
                            (
                                str(profile["type"]),
                                str(profile["profile_text"]),
                                json.dumps(profile.get("tags", []), ensure_ascii=False),
                                json.dumps(profile.get("profile_signals", {}), ensure_ascii=False),
                                json.dumps(lexical_profile, ensure_ascii=False),
                                int(existing["id"]),
                            ),
                        )
                    continue
                connection.execute(
                    """
                    INSERT INTO category_profiles (
                        type, profile_text, tags_json, profile_signals_json, lexical_profile_json, profile_origin, weight, synthetic_count, status
                    ) VALUES (?, ?, ?, ?, ?, 'default_seed', 0.5, 5, 'active')
                    """,
                    (
                        str(profile["type"]),
                        str(profile["profile_text"]),
                        json.dumps(profile.get("tags", []), ensure_ascii=False),
                        json.dumps(profile.get("profile_signals", {}), ensure_ascii=False),
                        json.dumps(lexical_profile, ensure_ascii=False),
                    ),
                )
                inserted += 1
        return inserted

    def _build_lexical_profile_payload(self, profile: dict[str, Any]) -> dict[str, Any]:
        signals = profile.get("profile_signals", {})
        lexical_signals: list[str] = []
        if isinstance(signals, dict):
            for key in (
                "aliases",
                "semantic_signals",
                "ocr_signals",
                "numeric_patterns",
                "document_examples",
                "business_use_cases",
            ):
                values = signals.get(key, [])
                if isinstance(values, list):
                    lexical_signals.extend(str(value) for value in values if str(value).strip())
        lexical_signals.extend(str(tag) for tag in profile.get("tags", []) if str(tag).strip())
        seen: set[str] = set()
        deduped = []
        for item in lexical_signals:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return {"lexical_signals": deduped}

    def backfill_category_profile_signals(self, *, dry_run: bool = False) -> list[dict[str, Any]]:
        """Fill missing profile_signals_json for profiles matching known default seed types."""
        changed: list[dict[str, Any]] = []
        default_by_type = {str(profile["type"]): profile for profile in DEFAULT_CATEGORY_PROFILES}
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, type, profile_signals_json, profile_origin
                FROM category_profiles
                ORDER BY type, id
                """
            ).fetchall()
            for row in rows:
                profile_type = str(row["type"])
                default_profile = default_by_type.get(profile_type)
                if default_profile is None:
                    continue
                current_signals = str(row["profile_signals_json"] or "").strip().lower()
                if current_signals not in {"", "{}", "null"}:
                    continue
                item = {
                    "id": int(row["id"]),
                    "type": profile_type,
                    "profile_origin": str(row["profile_origin"] or "user"),
                }
                changed.append(item)
                if dry_run:
                    continue
                connection.execute(
                    """
                    UPDATE category_profiles
                    SET profile_signals_json = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (
                        json.dumps(default_profile.get("profile_signals", {}), ensure_ascii=False),
                        int(row["id"]),
                    ),
                )
        return changed

    def expand_category_profile_training_data(
        self,
        *,
        synthetic_count: int = 12,
        dry_run: bool = False,
    ) -> list[dict[str, Any]]:
        """Raise synthetic_count for known profile types without touching text or origin."""
        changed: list[dict[str, Any]] = []
        default_types = {str(profile["type"]) for profile in DEFAULT_CATEGORY_PROFILES}
        target_count = max(1, int(synthetic_count))
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, type, profile_origin, synthetic_count
                FROM category_profiles
                ORDER BY type, id
                """
            ).fetchall()
            for row in rows:
                profile_type = str(row["type"])
                current_count = int(row["synthetic_count"] or 0)
                if profile_type not in default_types or current_count >= target_count:
                    continue
                item = {
                    "id": int(row["id"]),
                    "type": profile_type,
                    "profile_origin": str(row["profile_origin"] or "user"),
                    "old_synthetic_count": current_count,
                    "new_synthetic_count": target_count,
                }
                changed.append(item)
                if dry_run:
                    continue
                connection.execute(
                    """
                    UPDATE category_profiles
                    SET synthetic_count = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (target_count, int(row["id"])),
                )
        return changed

    def add_category_profile(
        self,
        *,
        category_type: str,
        profile_text: str,
        tags: list[str] | None = None,
        weight: float = 0.5,
        synthetic_count: int = 5,
        status: str = "active",
        profile_signals: dict[str, Any] | None = None,
        lexical_profile: dict[str, Any] | None = None,
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO category_profiles (
                    type, profile_text, tags_json, profile_signals_json, lexical_profile_json, profile_origin, weight, synthetic_count, status
                ) VALUES (?, ?, ?, ?, ?, 'user', ?, ?, ?)
                """,
                (
                    category_type,
                    profile_text,
                    json.dumps(tags or [], ensure_ascii=False),
                    json.dumps(profile_signals or {}, ensure_ascii=False),
                    json.dumps(lexical_profile or {"lexical_signals": tags or []}, ensure_ascii=False),
                    float(weight),
                    int(synthetic_count),
                    status,
                ),
            )
            return int(cursor.lastrowid)

    def list_category_profiles(self, include_inactive: bool = False) -> list[dict[str, Any]]:
        query = "SELECT * FROM category_profiles"
        if not include_inactive:
            query += " WHERE status = 'active'"
        query += " ORDER BY type, id"
        with self.connect() as connection:
            rows = connection.execute(query).fetchall()
        return [dict(row) for row in rows]

    def deactivate_category_profile(self, profile_id: int) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE category_profiles
                SET status = 'inactive', updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (profile_id,),
            )
            return int(cursor.rowcount)

    def get_category_profile_training_signature(self) -> str:
        return build_category_profile_signature(self.list_category_profiles(include_inactive=False))

    def count_reviewed_feedback_logs(self) -> int:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM feedback_logs
                WHERE feedback_action IN ('confirmed', 'corrected')
                """
            ).fetchone()
        return int(row["count"]) if row else 0

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
        with self.connect() as connection:
            existing = connection.execute("SELECT id FROM files WHERE file_path = ?", (file_path,)).fetchone()
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
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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

    def get_document_features_by_hash(self, file_hash: str, extractor_version: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute(
                """
                SELECT *
                FROM document_features
                WHERE file_hash = ? AND extractor_version = ?
                ORDER BY updated_at DESC, id DESC
                LIMIT 1
                """,
                (file_hash, extractor_version),
            ).fetchone()

    def upsert_document_features(
        self,
        *,
        file_id: int,
        file_hash: str,
        extractor_version: str,
        filename_features: dict[str, Any],
        metadata_features: dict[str, Any],
        structural_features: dict[str, Any],
        text_stats: dict[str, Any],
        compressed_text: str,
        compressed_text_hash: str,
        layout_features: dict[str, Any] | None = None,
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO document_features (
                    file_id, file_hash, extractor_version,
                    filename_features_json, metadata_features_json, structural_features_json,
                    layout_features_json, text_stats_json, compressed_text, compressed_text_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_id, extractor_version) DO UPDATE SET
                    file_hash = excluded.file_hash,
                    filename_features_json = excluded.filename_features_json,
                    metadata_features_json = excluded.metadata_features_json,
                    structural_features_json = excluded.structural_features_json,
                    layout_features_json = excluded.layout_features_json,
                    text_stats_json = excluded.text_stats_json,
                    compressed_text = excluded.compressed_text,
                    compressed_text_hash = excluded.compressed_text_hash,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    file_id,
                    file_hash,
                    extractor_version,
                    json.dumps(filename_features, ensure_ascii=False),
                    json.dumps(metadata_features, ensure_ascii=False),
                    json.dumps(structural_features, ensure_ascii=False),
                    json.dumps(layout_features or {}, ensure_ascii=False),
                    json.dumps(text_stats, ensure_ascii=False),
                    compressed_text,
                    compressed_text_hash,
                ),
            )
            if cursor.lastrowid:
                return int(cursor.lastrowid)
            row = connection.execute(
                "SELECT id FROM document_features WHERE file_id = ? AND extractor_version = ?",
                (file_id, extractor_version),
            ).fetchone()
            return int(row["id"]) if row else 0

    def fetch_type_training_examples(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    fl.final_middle_category AS label,
                    f.file_name,
                    f.extracted_text AS body_text,
                    COALESCE(df.structural_features_json, '{}') AS structural_features_json
                    , COALESCE(df.layout_features_json, '{}') AS layout_features_json
                FROM feedback_logs fl
                JOIN files f ON f.id = fl.file_id
                LEFT JOIN document_features df ON df.file_id = f.id
                WHERE fl.feedback_action IN ('confirmed', 'corrected')

                UNION ALL

                SELECT
                    ce.category AS label,
                    f.file_name,
                    COALESCE(NULLIF(ce.source_text, ''), f.extracted_text) AS body_text,
                    COALESCE(df.structural_features_json, '{}') AS structural_features_json
                    , COALESCE(df.layout_features_json, '{}') AS layout_features_json
                FROM confirmed_examples ce
                JOIN files f ON f.id = ce.file_id
                LEFT JOIN document_features df ON df.file_id = f.id
                ORDER BY file_name
                """
            ).fetchall()
        training_rows = [dict(row) | {"source": "real", "sample_weight": 1.0} for row in rows]
        active_profiles = self.list_category_profiles(include_inactive=False)
        for profile in active_profiles:
            training_rows.extend(build_synthetic_training_rows(profile))
        return training_rows

    def upsert_document_vector(
        self,
        *,
        file_id: int,
        vector_type: str,
        vector_key: str,
        vector_json: str,
        model_version: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO document_vectors (file_id, vector_type, vector_key, vector_json, model_version)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(file_id, vector_type, model_version) DO UPDATE SET
                    vector_key = excluded.vector_key,
                    vector_json = excluded.vector_json
                """,
                (file_id, vector_type, vector_key, vector_json, model_version),
            )

    def insert_model_run(self, model_name: str, model_version: str, training_count: int, metrics: dict[str, Any]) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO model_runs (model_name, model_version, training_count, metrics_json)
                VALUES (?, ?, ?, ?)
                """,
                (model_name, model_version, training_count, json.dumps(metrics, ensure_ascii=False)),
            )
            return int(cursor.lastrowid)

    def upsert_document_tag(self, file_id: int, tag: str, confidence: float, source: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO document_tags (file_id, tag, confidence, source)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(file_id, tag, source) DO UPDATE SET
                    confidence = excluded.confidence
                """,
                (file_id, tag, confidence, source),
            )

    def insert_category_candidate(
        self,
        *,
        source: str,
        suggested_name: str,
        representative_file_ids: list[int],
        evidence: dict[str, Any],
        status: str = "pending",
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO category_candidates (
                    source, suggested_name, representative_file_ids_json, evidence_json, status
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    source,
                    suggested_name,
                    json.dumps(representative_file_ids, ensure_ascii=False),
                    json.dumps(evidence, ensure_ascii=False),
                    status,
                ),
            )
            return int(cursor.lastrowid)

    def fetch_cluster_candidate_rows(self, limit: int = 200) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    f.id AS file_id,
                    f.file_name,
                    f.extracted_text AS text,
                    c.predicted_category,
                    c.predicted_type,
                    c.review_reasons_json,
                    CASE WHEN c.review_reasons_json <> '[]' THEN 1 ELSE 0 END AS review_required,
                    df.compressed_text
                FROM classifications c
                JOIN (
                    SELECT file_id, MAX(id) AS max_id
                    FROM classifications
                    GROUP BY file_id
                ) latest ON latest.max_id = c.id
                JOIN files f ON f.id = c.file_id
                LEFT JOIN document_features df ON df.file_id = f.id
                ORDER BY c.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

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
        large_category: str = "",
        middle_category: str = "",
        small_category: str | None = None,
        large_confidence: float = 0.0,
        middle_confidence: float = 0.0,
        small_confidence: float = 0.0,
        source_scores_json: str = "{}",
        explanation_json: str = "{}",
        evidence_json: str = "{}",
        performance_json: str = "{}",
        classifier_version: str = SCHEMA_VERSION,
        config_version: str = SCHEMA_VERSION,
        predicted_type: str = "",
        type_confidence: float = 0.0,
        review_reasons_json: str = "[]",
        suggested_tags_json: str = "[]",
        cluster_candidate_id: int | None = None,
        ml_evidence_json: str = "{}",
        rule_evidence_json: str = "{}",
        lexical_score: float = 0.0,
        layout_score: float = 0.0,
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO classifications (
                    file_id, predicted_category, rule_score, embedding_score, llm_score,
                    final_score, candidate_scores_json, reasoning, status,
                    large_category, middle_category, small_category,
                    large_confidence, middle_confidence, small_confidence,
                    source_scores_json, explanation_json, evidence_json, performance_json,
                    classifier_version, config_version,
                    predicted_type, type_confidence, review_reasons_json, suggested_tags_json,
                    cluster_candidate_id, ml_evidence_json, rule_evidence_json, lexical_score, layout_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    large_category,
                    middle_category or predicted_category,
                    small_category,
                    large_confidence,
                    middle_confidence,
                    small_confidence,
                    source_scores_json,
                    explanation_json,
                    evidence_json,
                    performance_json,
                    classifier_version,
                    config_version,
                    predicted_type,
                    type_confidence,
                    review_reasons_json,
                    suggested_tags_json,
                    cluster_candidate_id,
                    ml_evidence_json,
                    rule_evidence_json,
                    float(lexical_score),
                    float(layout_score),
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
        predicted_hierarchy: dict[str, Any] | None = None,
        final_hierarchy: dict[str, Any] | None = None,
        evidence_text: str = "",
        metadata: dict[str, Any] | None = None,
        source_scores: dict[str, float] | None = None,
        classifier_version: str = SCHEMA_VERSION,
        config_version: str = SCHEMA_VERSION,
        ocr_used: bool = False,
        llm_used: bool = False,
        confirmation_batch_id: str = "",
        confirmation_batch_name: str = "",
    ) -> int:
        predicted_hierarchy = predicted_hierarchy or {}
        final_hierarchy = final_hierarchy or {}
        metadata = metadata or {}
        source_scores = source_scores or {}
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO feedback_logs (
                    file_id, classification_id, predicted_category, final_category, feedback_action, user_note,
                    predicted_large_category, predicted_middle_category, predicted_small_category,
                    final_large_category, final_middle_category, final_small_category,
                    evidence_text, metadata_json, source_scores_json,
                    classifier_version, config_version, ocr_used, llm_used, confirmation_batch_id, confirmation_batch_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    classification_id,
                    predicted_category,
                    final_category,
                    feedback_action,
                    user_note,
                    str(predicted_hierarchy.get("large_category", "")),
                    str(predicted_hierarchy.get("middle_category", predicted_category)),
                    predicted_hierarchy.get("small_category"),
                    str(final_hierarchy.get("large_category", "")),
                    str(final_hierarchy.get("middle_category", final_category)),
                    final_hierarchy.get("small_category"),
                    evidence_text,
                    json.dumps(metadata, ensure_ascii=False),
                    json.dumps(source_scores, ensure_ascii=False),
                    classifier_version,
                    config_version,
                    int(ocr_used),
                    int(llm_used),
                    confirmation_batch_id,
                    confirmation_batch_name,
                ),
            )
            connection.execute("UPDATE classifications SET status = 'reviewed' WHERE id = ?", (classification_id,))
            return int(cursor.lastrowid)

    def save_confirmed_example(
        self,
        file_id: int,
        category: str,
        source_text: str,
        embedding: list[float],
        source_feedback_log_id: int,
    ) -> int:
        with self.connect() as connection:
            existing = connection.execute(
                """
                SELECT id
                FROM confirmed_examples
                WHERE file_id = ? AND category = ?
                ORDER BY id
                LIMIT 1
                """,
                (file_id, category),
            ).fetchone()
            if existing:
                return int(existing["id"])
            cursor = connection.execute(
                """
                INSERT INTO confirmed_examples (
                    file_id, category, source_text, embedding_json, embedding_key, source_feedback_log_id
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    category,
                    source_text,
                    "" if self.embedding_repository is not None else json.dumps(embedding, ensure_ascii=False),
                    "",
                    source_feedback_log_id,
                ),
            )
            example_id = int(cursor.lastrowid)
            if self.embedding_repository is not None:
                embedding_key = self._build_confirmed_example_embedding_key(example_id)
                self.embedding_repository.save_embedding(
                    embedding_key,
                    embedding,
                    {
                        "storage_type": "confirmed_example",
                        "example_id": example_id,
                        "file_id": file_id,
                        "category": category,
                        "source_feedback_log_id": source_feedback_log_id,
                        "confirmed": True,
                    },
                    overwrite=True,
                )
                connection.execute(
                    "UPDATE confirmed_examples SET embedding_key = ? WHERE id = ?",
                    (embedding_key, example_id),
                )
            return example_id

    def fetch_active_rules(self) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM rules WHERE status = 'active' ORDER BY category, id"
            ).fetchall()

    def fetch_confirmed_examples(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT ce.*, f.file_name
                FROM confirmed_examples ce
                JOIN files f ON f.id = ce.file_id
                ORDER BY ce.id
                """
            ).fetchall()
        resolved_rows: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            embedding = self._resolve_confirmed_example_embedding(item, backfill_to_hdf5=True)
            if embedding is not None:
                item["embedding_json"] = json.dumps(embedding, ensure_ascii=False)
            resolved_rows.append(item)
        return resolved_rows

    def list_categories(self) -> list[str]:
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

    def get_feedback_adjustments(self, predicted_category: str, categories: list[str]) -> dict[str, float]:
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
        for row in self.fetch_adaptive_rule_boosts(predicted_category=None):
            category = str(row["category"])
            if category in adjustments:
                adjustments[category] = round(adjustments.get(category, 0.0) + float(row["boost"]), 4)
        return adjustments

    def get_duplicate_confirmed_category_scores(
        self,
        file_hash: str,
        duplicate_of_file_id: int | None,
        categories: list[str],
    ) -> dict[str, float]:
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
        with self.connect() as connection:
            return connection.execute(
                """
                SELECT fl.file_id, fl.predicted_category, fl.final_category, f.file_name, f.extracted_text
                FROM feedback_logs fl
                JOIN files f ON f.id = fl.file_id
                WHERE fl.feedback_action = 'corrected'
                ORDER BY fl.id
                """
            ).fetchall()

    def get_active_rule_tokens_by_category(self) -> dict[str, set[str]]:
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
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO rules (
                    category, rule_type, pattern, weight, status, source, evidence_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
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
        with self.connect() as connection:
            return connection.execute("SELECT * FROM rules WHERE status = 'candidate' ORDER BY id DESC").fetchall()

    def fetch_feedback_learning_rows(self) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return connection.execute(
                """
                SELECT predicted_middle_category, final_middle_category, evidence_text
                FROM feedback_logs
                WHERE feedback_action = 'corrected'
                ORDER BY id
                """
            ).fetchall()

    def clear_adaptive_rule_boosts(self) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM adaptive_rule_boosts")

    def insert_adaptive_rule_boost(
        self,
        category: str,
        token: str,
        boost: float,
        source: str,
        support_count: int,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO adaptive_rule_boosts (
                    category, token, boost, source, support_count
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (category, token, boost, source, support_count),
            )

    def fetch_adaptive_rule_boosts(self, predicted_category: str | None = None) -> list[sqlite3.Row]:
        with self.connect() as connection:
            if predicted_category is None:
                return connection.execute(
                    "SELECT * FROM adaptive_rule_boosts ORDER BY category, token"
                ).fetchall()
            return connection.execute(
                "SELECT * FROM adaptive_rule_boosts WHERE category = ? ORDER BY token",
                (predicted_category,),
            ).fetchall()

    def record_snapshot(self, snapshot_id: str, snapshot_type: str, manifest: dict[str, Any], reason: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO snapshots (snapshot_id, snapshot_type, manifest_json, reason)
                VALUES (?, ?, ?, ?)
                """,
                (snapshot_id, snapshot_type, json.dumps(manifest, ensure_ascii=False), reason),
            )

    def record_operation(
        self,
        operation_type: str,
        status: str,
        details: dict[str, Any],
        batch_id: int | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO operation_journal (operation_type, batch_id, status, details_json)
                VALUES (?, ?, ?, ?)
                """,
                (operation_type, batch_id, status, json.dumps(details, ensure_ascii=False)),
            )

    def create_move_batch(self, operation_mode: str, status: str, manifest_json: str) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO move_batches (operation_mode, status, manifest_json)
                VALUES (?, ?, ?)
                """,
                (operation_mode, status, manifest_json),
            )
            return int(cursor.lastrowid)

    def add_move_item(self, batch_id: int, item: dict[str, Any], status: str) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO move_items (
                    batch_id, file_id, classification_id, source_path, destination_path,
                    large_category, middle_category, small_category, confidence, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batch_id,
                    int(item["file_id"]),
                    int(item["classification_id"]),
                    str(item["source_path"]),
                    str(item["destination_path"]),
                    str(item["large_category"]),
                    str(item["middle_category"]),
                    item.get("small_category"),
                    float(item["confidence"]),
                    status,
                ),
            )
            return int(cursor.lastrowid)

    def update_move_batch_status(self, batch_id: int, status: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE move_batches SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status, batch_id),
            )

    def update_move_item_status(
        self,
        move_item_id: int,
        status: str,
        actual_destination_path: str | None = None,
        error_message: str = "",
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE move_items
                SET status = ?, actual_destination_path = COALESCE(?, actual_destination_path),
                    error_message = ?, moved_at = CASE WHEN ? = 'committed' THEN CURRENT_TIMESTAMP ELSE moved_at END
                WHERE id = ?
                """,
                (status, actual_destination_path, error_message, status, move_item_id),
            )

    def update_move_item_restore(self, move_item_id: int, status: str, restored_path: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE move_items
                SET status = ?, restored_path = ?, restored_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (status, restored_path, move_item_id),
            )

    def fetch_move_items(self, batch_id: int, status: str | None = None) -> list[sqlite3.Row]:
        query = "SELECT * FROM move_items WHERE batch_id = ?"
        params: list[Any] = [batch_id]
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY id"
        with self.connect() as connection:
            return connection.execute(query, params).fetchall()

    def fetch_move_item(self, move_item_id: int) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute("SELECT * FROM move_items WHERE id = ?", (move_item_id,)).fetchone()

    def fetch_last_committed_move_batch(self) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute(
                """
                SELECT * FROM move_batches
                WHERE status IN ('committed', 'partial_failed')
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()

    def fetch_move_batch(self, batch_id: int) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute("SELECT * FROM move_batches WHERE id = ?", (batch_id,)).fetchone()

    def list_move_history(
        self,
        category: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        batch_id: int | None = None,
    ) -> list[sqlite3.Row]:
        query = """
            SELECT mi.*, mb.created_at AS batch_created_at
            FROM move_items mi
            JOIN move_batches mb ON mb.id = mi.batch_id
            WHERE 1 = 1
        """
        params: list[Any] = []
        if category:
            query += " AND mi.middle_category = ?"
            params.append(category)
        if date_from:
            query += " AND mb.created_at >= ?"
            params.append(date_from)
        if date_to:
            query += " AND mb.created_at <= ?"
            params.append(date_to)
        if batch_id is not None:
            query += " AND mi.batch_id = ?"
            params.append(batch_id)
        query += " ORDER BY mi.id DESC"
        with self.connect() as connection:
            return connection.execute(query, params).fetchall()

    def fetch_latest_classifications(self, limit: int | None = None) -> list[sqlite3.Row]:
        query = """
            SELECT
                c.id AS classification_id,
                c.file_id,
                c.final_score,
                c.large_category,
                c.middle_category,
                c.small_category,
                c.predicted_type,
                c.type_confidence,
                c.review_reasons_json,
                c.suggested_tags_json,
                c.cluster_candidate_id,
                f.file_path,
                f.file_name,
                f.xxhash64,
                f.duplicate_of_file_id
            FROM classifications c
            JOIN (
                SELECT file_id, MAX(id) AS max_id
                FROM classifications
                GROUP BY file_id
            ) latest ON latest.max_id = c.id
            JOIN files f ON f.id = c.file_id
            ORDER BY c.id DESC
        """
        params: list[Any] = []
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with self.connect() as connection:
            return connection.execute(query, params).fetchall()

    def fetch_classifications_by_ids(self, classification_ids: list[int]) -> list[sqlite3.Row]:
        if not classification_ids:
            return []
        placeholders = ",".join("?" for _ in classification_ids)
        query = f"""
            SELECT
                c.id AS classification_id,
                c.file_id,
                c.final_score,
                c.large_category,
                c.middle_category,
                c.small_category,
                c.predicted_type,
                c.type_confidence,
                c.review_reasons_json,
                c.suggested_tags_json,
                c.cluster_candidate_id,
                f.file_path,
                f.file_name,
                f.xxhash64,
                f.duplicate_of_file_id
            FROM classifications c
            JOIN files f ON f.id = c.file_id
            WHERE c.id IN ({placeholders})
            ORDER BY c.id DESC
        """
        with self.connect() as connection:
            return connection.execute(query, [int(value) for value in classification_ids]).fetchall()

    def get_duplicate_group_folder_name(self, xxhash64: str) -> str | None:
        """Return a representative folder name when multiple files share the same content hash."""
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT file_name
                FROM files
                WHERE xxhash64 = ?
                ORDER BY id
                """,
                (xxhash64,),
            ).fetchall()

        if len(rows) <= 1:
            return None

        stem = Path(str(rows[0]["file_name"])).stem.strip()
        return stem or "duplicate_group"

    def list_feedback_logs(
        self,
        category: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        min_confidence: float | None = None,
        file_name_query: str | None = None,
    ) -> list[sqlite3.Row]:
        query = """
            SELECT
                fl.*,
                f.file_name,
                c.final_score
            FROM feedback_logs fl
            JOIN files f ON f.id = fl.file_id
            JOIN classifications c ON c.id = fl.classification_id
            WHERE 1 = 1
        """
        params: list[Any] = []
        if category:
            query += " AND fl.final_middle_category = ?"
            params.append(category)
        if date_from:
            query += " AND fl.created_at >= ?"
            params.append(date_from)
        if date_to:
            query += " AND fl.created_at <= ?"
            params.append(date_to)
        if min_confidence is not None:
            query += " AND c.final_score >= ?"
            params.append(min_confidence)
        if file_name_query:
            query += " AND f.file_name LIKE ?"
            params.append(f"%{file_name_query}%")
        query += " ORDER BY fl.id DESC"
        with self.connect() as connection:
            return connection.execute(query, params).fetchall()

    def list_confirmation_batches(self) -> list[sqlite3.Row]:
        query = """
            SELECT
                CASE
                    WHEN TRIM(COALESCE(fl.confirmation_batch_id, '')) = ''
                    THEN 'legacy-' || fl.id
                    ELSE fl.confirmation_batch_id
                END AS confirmation_batch_id,
                MAX(TRIM(COALESCE(fl.confirmation_batch_name, ''))) AS confirmation_batch_name,
                MIN(fl.created_at) AS created_at,
                MAX(fl.created_at) AS updated_at,
                COUNT(*) AS file_count,
                GROUP_CONCAT(DISTINCT COALESCE(NULLIF(fl.final_middle_category, ''), fl.final_category)) AS categories,
                GROUP_CONCAT(f.file_name, ' | ') AS file_names,
                GROUP_CONCAT(fl.id) AS feedback_log_ids
            FROM feedback_logs fl
            JOIN files f ON f.id = fl.file_id
            GROUP BY
                CASE
                    WHEN TRIM(COALESCE(fl.confirmation_batch_id, '')) = ''
                    THEN 'legacy-' || fl.id
                    ELSE fl.confirmation_batch_id
                END
            ORDER BY MAX(fl.id) DESC
        """
        with self.connect() as connection:
            return connection.execute(query).fetchall()

    def update_confirmation_batch_name(self, confirmation_batch_id: str, confirmation_batch_name: str) -> int:
        confirmation_batch_id = confirmation_batch_id.strip()
        confirmation_batch_name = confirmation_batch_name.strip()
        if not confirmation_batch_id:
            return 0
        if confirmation_batch_id.startswith("legacy-"):
            try:
                feedback_log_id = int(confirmation_batch_id.removeprefix("legacy-"))
            except ValueError:
                return 0
            with self.connect() as connection:
                cursor = connection.execute(
                    "UPDATE feedback_logs SET confirmation_batch_name = ? WHERE id = ?",
                    (confirmation_batch_name, feedback_log_id),
                )
                return int(cursor.rowcount)
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE feedback_logs SET confirmation_batch_name = ? WHERE confirmation_batch_id = ?",
                (confirmation_batch_name, confirmation_batch_id),
            )
            return int(cursor.rowcount)

    def get_feedback_log(self, feedback_log_id: int) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute(
                """
                SELECT fl.*, f.file_name, c.final_score
                FROM feedback_logs fl
                JOIN files f ON f.id = fl.file_id
                JOIN classifications c ON c.id = fl.classification_id
                WHERE fl.id = ?
                """,
                (feedback_log_id,),
            ).fetchone()

    def delete_feedback_log(self, feedback_log_id: int) -> int:
        with self.connect() as connection:
            connection.execute(
                "DELETE FROM confirmed_examples WHERE source_feedback_log_id = ?",
                (feedback_log_id,),
            )
            cursor = connection.execute("DELETE FROM feedback_logs WHERE id = ?", (feedback_log_id,))
            connection.execute("DELETE FROM adaptive_rule_boosts")
            return int(cursor.rowcount)

    def delete_confirmation_batch(self, confirmation_batch_id: str) -> int:
        confirmation_batch_id = confirmation_batch_id.strip()
        if not confirmation_batch_id:
            return 0
        if confirmation_batch_id.startswith("legacy-"):
            try:
                return self.delete_feedback_log(int(confirmation_batch_id.removeprefix("legacy-")))
            except ValueError:
                return 0
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id FROM feedback_logs WHERE confirmation_batch_id = ?",
                (confirmation_batch_id,),
            ).fetchall()
            feedback_ids = [int(row["id"]) for row in rows]
            if not feedback_ids:
                return 0
            placeholders = ",".join("?" for _ in feedback_ids)
            connection.execute(
                f"DELETE FROM confirmed_examples WHERE source_feedback_log_id IN ({placeholders})",
                feedback_ids,
            )
            cursor = connection.execute(
                f"DELETE FROM feedback_logs WHERE id IN ({placeholders})",
                feedback_ids,
            )
            connection.execute("DELETE FROM adaptive_rule_boosts")
            return int(cursor.rowcount)

    def clear_feedback_logs(self) -> int:
        with self.connect() as connection:
            count = connection.execute("SELECT COUNT(*) FROM feedback_logs").fetchone()[0]
            connection.execute("DELETE FROM confirmed_examples")
            connection.execute("DELETE FROM feedback_logs")
            connection.execute("DELETE FROM adaptive_rule_boosts")
            return int(count)

    def export_feedback_logs(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.list_feedback_logs()]

    def save_config_version(self, config_name: str, version: str, payload: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO config_versions (config_name, version, payload_json)
                VALUES (?, ?, ?)
                """,
                (config_name, version, json.dumps(payload, ensure_ascii=False)),
            )

    def cache_ocr_result(
        self,
        file_hash: str,
        cache_key: str,
        backend: str,
        pages_scanned: int,
        extracted_text: str,
        metadata: dict[str, Any],
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO ocr_cache (
                    file_hash, cache_key, backend, pages_scanned, extracted_text, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    file_hash,
                    cache_key,
                    backend,
                    pages_scanned,
                    extracted_text,
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )

    def get_cached_ocr_result(self, cache_key: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute("SELECT * FROM ocr_cache WHERE cache_key = ?", (cache_key,)).fetchone()

    def get_document_cache(self, file_hash: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute("SELECT * FROM document_cache WHERE file_hash = ?", (file_hash,)).fetchone()

    def save_document_cache(
        self,
        *,
        file_hash: str,
        file_path: str,
        file_name: str,
        file_size: int,
        extension: str,
        raw_text: str,
        cleaned_text: str,
        text_hash: str,
        ocr_engine: str = "",
        ocr_version: str = "",
        ocr_quality_score: float = 0.0,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO document_cache (
                    file_hash, file_path, file_name, file_size, extension, raw_text, cleaned_text,
                    text_hash, ocr_engine, ocr_version, ocr_quality_score, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(file_hash) DO UPDATE SET
                    file_path = excluded.file_path,
                    file_name = excluded.file_name,
                    file_size = excluded.file_size,
                    extension = excluded.extension,
                    raw_text = excluded.raw_text,
                    cleaned_text = excluded.cleaned_text,
                    text_hash = excluded.text_hash,
                    ocr_engine = excluded.ocr_engine,
                    ocr_version = excluded.ocr_version,
                    ocr_quality_score = excluded.ocr_quality_score,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    file_hash,
                    file_path,
                    file_name,
                    int(file_size),
                    extension,
                    raw_text,
                    cleaned_text,
                    text_hash,
                    ocr_engine,
                    ocr_version,
                    float(ocr_quality_score),
                ),
            )

    def save_unknown_pool_entry(
        self,
        *,
        file_hash: str,
        text_hash: str,
        cleaned_text: str,
        nearest_category: str,
        nearest_similarity: float,
        reason: str,
        embedding_ref: str = "",
    ) -> int:
        summary_text = cleaned_text[:1200]
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO unknown_pool (
                    file_hash, text_hash, summary_text, cleaned_text, embedding_ref,
                    nearest_category, nearest_similarity, reason, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(file_hash, reason) DO UPDATE SET
                    text_hash = excluded.text_hash,
                    summary_text = excluded.summary_text,
                    cleaned_text = excluded.cleaned_text,
                    embedding_ref = excluded.embedding_ref,
                    nearest_category = excluded.nearest_category,
                    nearest_similarity = excluded.nearest_similarity,
                    status = 'pending',
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    file_hash,
                    text_hash,
                    summary_text,
                    cleaned_text,
                    embedding_ref,
                    nearest_category,
                    float(nearest_similarity),
                    reason,
                ),
            )
            row = connection.execute(
                "SELECT id FROM unknown_pool WHERE file_hash = ? AND reason = ?",
                (file_hash, reason),
            ).fetchone()
            return int(row["id"]) if row else 0

    def list_unknown_pool(self, status: str = "pending", limit: int = 500) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM unknown_pool
                WHERE status = ?
                ORDER BY id
                LIMIT ?
                """,
                (status, int(limit)),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_unsupervised_cluster_run(
        self,
        *,
        algorithm: str,
        parameters: dict[str, Any],
        metrics: dict[str, Any],
        assignments: list[dict[str, Any]],
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO unsupervised_cluster_runs (algorithm, parameters_json, metrics_json, status)
                VALUES (?, ?, ?, 'completed')
                """,
                (
                    algorithm,
                    json.dumps(parameters, ensure_ascii=False),
                    json.dumps(metrics, ensure_ascii=False),
                ),
            )
            run_id = int(cursor.lastrowid)
            for assignment in assignments:
                connection.execute(
                    """
                    INSERT INTO unsupervised_cluster_items (run_id, unknown_pool_id, cluster_id, score)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        int(assignment["unknown_pool_id"]),
                        int(assignment["cluster_id"]),
                        float(assignment.get("score", 0.0)),
                    ),
                )
            return run_id

    def cache_embedding(
        self,
        cache_key: str,
        file_hash: str,
        model_name: str,
        text_signature: str,
        embedding_version: str,
        text_kind: str,
        embedding: list[float],
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO embedding_cache (
                    cache_key, file_hash, model_name, text_signature, embedding_version,
                    text_kind, embedding_json, vector_dim, updated_at, last_accessed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (
                    cache_key,
                    file_hash,
                    model_name,
                    text_signature,
                    embedding_version,
                    text_kind,
                    json.dumps(embedding, ensure_ascii=False),
                    len(embedding),
                ),
            )

    def get_cached_embedding(self, cache_key: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM embedding_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row is not None:
                connection.execute(
                    """
                    UPDATE embedding_cache
                    SET hit_count = hit_count + 1, last_accessed_at = CURRENT_TIMESTAMP
                    WHERE cache_key = ?
                    """,
                    (cache_key,),
                )
                row = connection.execute(
                    "SELECT * FROM embedding_cache WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()
            return row

    def get_embedding_cache_stats(self) -> dict[str, Any]:
        with self.connect() as connection:
            count = int(connection.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0])
            hit_total = int(connection.execute("SELECT COALESCE(SUM(hit_count), 0) FROM embedding_cache").fetchone()[0])
            models = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT model_name, COUNT(*) AS entries, COALESCE(SUM(hit_count), 0) AS hits
                    FROM embedding_cache
                    GROUP BY model_name
                    ORDER BY model_name
                    """
                ).fetchall()
            ]
            kinds = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT text_kind, COUNT(*) AS entries
                    FROM embedding_cache
                    GROUP BY text_kind
                    ORDER BY text_kind
                    """
                ).fetchall()
            ]
        return {
            "entries": count,
            "total_hits": hit_total,
            "models": models,
            "text_kinds": kinds,
        }

    def clear_embedding_cache(self) -> int:
        with self.connect() as connection:
            count = int(connection.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0])
            connection.execute("DELETE FROM embedding_cache")
        return count

    def migrate_confirmed_examples_to_hdf5(
        self,
        *,
        prune_legacy_json: bool = False,
    ) -> dict[str, int]:
        if self.embedding_repository is None:
            return {"migrated": 0, "skipped": 0, "failed": 0}

        migrated = 0
        skipped = 0
        failed = 0
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, file_id, category, source_text, embedding_json, embedding_key, source_feedback_log_id
                FROM confirmed_examples
                ORDER BY id
                """
            ).fetchall()

        for row in rows:
            item = dict(row)
            embedding_key = str(item.get("embedding_key") or "").strip()
            if embedding_key and self.embedding_repository.has_embedding(embedding_key):
                skipped += 1
                continue
            legacy_embedding_json = str(item.get("embedding_json") or "").strip()
            if not legacy_embedding_json:
                skipped += 1
                continue
            try:
                embedding = [float(value) for value in json.loads(legacy_embedding_json)]
                if not embedding_key:
                    embedding_key = self._build_confirmed_example_embedding_key(int(item["id"]))
                self.embedding_repository.save_embedding(
                    embedding_key,
                    embedding,
                    {
                        "storage_type": "confirmed_example",
                        "example_id": int(item["id"]),
                        "file_id": int(item["file_id"]),
                        "category": str(item["category"]),
                        "source_feedback_log_id": int(item["source_feedback_log_id"]),
                        "confirmed": True,
                        "migrated_from": "confirmed_examples.embedding_json",
                    },
                    overwrite=True,
                )
                with self.connect() as connection:
                    if prune_legacy_json:
                        connection.execute(
                            "UPDATE confirmed_examples SET embedding_key = ?, embedding_json = '' WHERE id = ?",
                            (embedding_key, int(item["id"])),
                        )
                    else:
                        connection.execute(
                            "UPDATE confirmed_examples SET embedding_key = ? WHERE id = ?",
                            (embedding_key, int(item["id"])),
                        )
                migrated += 1
            except Exception:
                failed += 1
        return {"migrated": migrated, "skipped": skipped, "failed": failed}

    def get_confirmed_example_embedding_stats(self) -> dict[str, Any]:
        with self.connect() as connection:
            total = int(connection.execute("SELECT COUNT(*) FROM confirmed_examples").fetchone()[0])
            with_key = int(
                connection.execute(
                    "SELECT COUNT(*) FROM confirmed_examples WHERE TRIM(COALESCE(embedding_key, '')) <> ''"
                ).fetchone()[0]
            )
            with_legacy_json = int(
                connection.execute(
                    "SELECT COUNT(*) FROM confirmed_examples WHERE TRIM(COALESCE(embedding_json, '')) <> ''"
                ).fetchone()[0]
            )

        hdf5_available = 0
        missing_hdf5 = 0
        if self.embedding_repository is not None:
            with self.connect() as connection:
                rows = connection.execute(
                    "SELECT embedding_key FROM confirmed_examples WHERE TRIM(COALESCE(embedding_key, '')) <> ''"
                ).fetchall()
            for row in rows:
                key = str(row["embedding_key"])
                if self.embedding_repository.has_embedding(key):
                    hdf5_available += 1
                else:
                    missing_hdf5 += 1

        return {
            "total": total,
            "with_embedding_key": with_key,
            "with_legacy_embedding_json": with_legacy_json,
            "hdf5_available": hdf5_available,
            "missing_hdf5": missing_hdf5,
        }

    def fetch_legacy_embedding_cache_entries(self) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return connection.execute(
                """
                SELECT
                    cache_key,
                    file_hash,
                    model_name,
                    text_signature,
                    embedding_version,
                    text_kind,
                    embedding_json,
                    hit_count
                FROM embedding_cache
                ORDER BY id
                """
            ).fetchall()

    def _build_confirmed_example_embedding_key(self, example_id: int) -> str:
        return f"confirmed_example_{example_id}"

    def _resolve_confirmed_example_embedding(
        self,
        item: dict[str, Any],
        *,
        backfill_to_hdf5: bool,
    ) -> list[float] | None:
        embedding_key = str(item.get("embedding_key") or "").strip()
        if self.embedding_repository is not None and embedding_key:
            vector = self.embedding_repository.get_embedding(embedding_key)
            if vector is not None:
                return [float(value) for value in vector.tolist()]

        legacy_embedding_json = str(item.get("embedding_json") or "").strip()
        if not legacy_embedding_json:
            return None

        try:
            embedding = [float(value) for value in json.loads(legacy_embedding_json)]
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

        if self.embedding_repository is not None and backfill_to_hdf5:
            try:
                if not embedding_key:
                    embedding_key = self._build_confirmed_example_embedding_key(int(item["id"]))
                self.embedding_repository.save_embedding(
                    embedding_key,
                    embedding,
                    {
                        "storage_type": "confirmed_example",
                        "example_id": int(item["id"]),
                        "file_id": int(item["file_id"]),
                        "category": str(item["category"]),
                        "source_feedback_log_id": int(item["source_feedback_log_id"]),
                        "confirmed": True,
                        "backfilled_from": "confirmed_examples.embedding_json",
                    },
                    overwrite=True,
                )
                if not str(item.get("embedding_key") or "").strip():
                    with self.connect() as connection:
                        connection.execute(
                            "UPDATE confirmed_examples SET embedding_key = ? WHERE id = ?",
                            (embedding_key, int(item["id"])),
                        )
                    item["embedding_key"] = embedding_key
            except Exception:
                pass
        return embedding

    def fetch_embedding_rebuild_sources(self) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return connection.execute(
                """
                SELECT
                    'document' AS text_kind,
                    f.xxhash64 AS file_hash,
                    f.extracted_text AS text_value
                FROM files f
                WHERE TRIM(f.extracted_text) <> ''
                UNION ALL
                SELECT
                    'evidence' AS text_kind,
                    '' AS file_hash,
                    ce.source_text AS text_value
                FROM confirmed_examples ce
                WHERE TRIM(ce.source_text) <> ''
                """
            ).fetchall()

    def insert_llm_audit(
        self,
        classification_id: int | None,
        provider: str,
        model_name: str,
        decision: dict[str, Any],
        status: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO llm_audit (classification_id, provider, model_name, decision_json, status)
                VALUES (?, ?, ?, ?, ?)
                """,
                (classification_id, provider, model_name, json.dumps(decision, ensure_ascii=False), status),
            )

    def get_stats(self) -> dict[str, Any]:
        with self.connect() as connection:
            files_count = connection.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            classifications_count = connection.execute("SELECT COUNT(*) FROM classifications").fetchone()[0]
            feedback_logs_count = connection.execute("SELECT COUNT(*) FROM feedback_logs").fetchone()[0]
            confirmed_examples_count = connection.execute("SELECT COUNT(*) FROM confirmed_examples").fetchone()[0]
            rules_count = connection.execute("SELECT COUNT(*) FROM rules").fetchone()[0]
            move_batches_count = connection.execute("SELECT COUNT(*) FROM move_batches").fetchone()[0]
            snapshots_count = connection.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
            adaptive_rules_count = connection.execute("SELECT COUNT(*) FROM adaptive_rule_boosts").fetchone()[0]
            embedding_cache_count = connection.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0]
            document_features_count = connection.execute("SELECT COUNT(*) FROM document_features").fetchone()[0]
            category_candidates_count = connection.execute("SELECT COUNT(*) FROM category_candidates").fetchone()[0]
            document_tags_count = connection.execute("SELECT COUNT(*) FROM document_tags").fetchone()[0]
            category_profiles_count = connection.execute("SELECT COUNT(*) FROM category_profiles").fetchone()[0]
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
            "schema_version": SCHEMA_VERSION,
            "files_count": int(files_count),
            "classifications_count": int(classifications_count),
            "feedback_logs_count": int(feedback_logs_count),
            "confirmed_examples_count": int(confirmed_examples_count),
            "rules_count": int(rules_count),
            "move_batches_count": int(move_batches_count),
            "snapshots_count": int(snapshots_count),
            "adaptive_rules_count": int(adaptive_rules_count),
            "embedding_cache_count": int(embedding_cache_count),
            "document_features_count": int(document_features_count),
            "category_candidates_count": int(category_candidates_count),
            "document_tags_count": int(document_tags_count),
            "category_profiles_count": int(category_profiles_count),
            "recent_feedback": [dict(row) for row in recent_feedback],
        }

    def retrying_execute(
        self,
        query: str,
        params: tuple[Any, ...] = (),
        retries: int = 3,
        sleep_seconds: float = 0.1,
    ) -> None:
        last_error: Exception | None = None
        for _ in range(retries):
            try:
                with self.connect() as connection:
                    connection.execute(query, params)
                return
            except sqlite3.OperationalError as error:
                last_error = error
                time.sleep(sleep_seconds)
        if last_error is not None:
            raise last_error
