from __future__ import annotations

import unittest
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

import numpy as np

from src.embedding_repository import EmbeddingRepository, migrate_sqlite_embedding_cache_to_hdf5
from src.storage import ClassificationRepository
from src.vectorizer import SentenceTransformerEmbedder


class _FakeVector:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return list(self._values)


class _FakeModel:
    def __init__(self, marker: float) -> None:
        self.marker = marker
        self.calls = 0

    def encode(self, texts, normalize_embeddings: bool = True):
        del normalize_embeddings
        self.calls += 1
        if isinstance(texts, list):
            return [_FakeVector([self.marker, float(len(text))]) for text in texts]
        return _FakeVector([self.marker, float(len(texts))])


class _TestEmbedder(SentenceTransformerEmbedder):
    def __init__(self, model_name: str, fake_model: _FakeModel, embedding_repository: EmbeddingRepository) -> None:
        super().__init__(model_name=model_name, embedding_repository=embedding_repository)
        self._fake_model = fake_model

    def _load_model(self):
        return self._fake_model


class EmbeddingRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.base_dir = Path("tests_runtime") / f"embedding_repo_{uuid4().hex}"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.hdf5_path = self.base_dir / "embeddings.h5"
        self.db_path = self.base_dir / "test.db"
        self.embedding_repository = EmbeddingRepository(self.hdf5_path)
        self.sqlite_repository = ClassificationRepository(self.db_path)
        self.sqlite_repository.attach_embedding_repository(self.embedding_repository)
        self.sqlite_repository.initialize_database()

    def tearDown(self) -> None:
        if self.base_dir.exists():
            rmtree(self.base_dir, ignore_errors=True)

    def test_save_get_and_metadata_roundtrip(self) -> None:
        self.embedding_repository.save_embedding(
            "abc123",
            np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            {"category": "계약서", "confirmed": True},
        )
        vector = self.embedding_repository.get_embedding("abc123")
        metadata = self.embedding_repository.get_metadata("abc123")

        self.assertIsNotNone(vector)
        self.assertEqual(vector.dtype, np.float32)
        self.assertAlmostEqual(float(vector[0]), 0.1, places=6)
        self.assertAlmostEqual(float(vector[1]), 0.2, places=6)
        self.assertAlmostEqual(float(vector[2]), 0.3, places=6)
        self.assertEqual(metadata["category"], "계약서")
        self.assertTrue(metadata["confirmed"])

    def test_dimension_validation_rejects_mismatch(self) -> None:
        self.embedding_repository.save_embedding(
            "abc123",
            np.asarray([0.1, 0.2], dtype=np.float32),
            {"category": "report"},
        )
        with self.assertRaises(ValueError):
            self.embedding_repository.save_embedding(
                "abc456",
                np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
                {"category": "report"},
            )

    def test_delete_and_list_ids(self) -> None:
        self.embedding_repository.save_embedding("one", np.asarray([1.0, 2.0], dtype=np.float32), {})
        self.embedding_repository.save_embedding("two", np.asarray([3.0, 4.0], dtype=np.float32), {})
        self.assertEqual(self.embedding_repository.list_ids(), ["one", "two"])
        self.assertTrue(self.embedding_repository.delete_embedding("one"))
        self.assertFalse(self.embedding_repository.has_embedding("one"))
        self.assertEqual(self.embedding_repository.list_ids(), ["two"])

    def test_hdf5_cache_reuse_across_embedder_instances(self) -> None:
        model1 = _FakeModel(marker=1.0)
        embedder1 = _TestEmbedder("model-a", model1, self.embedding_repository)
        first = embedder1.encode(
            "report analysis summary",
            repository=self.sqlite_repository,
            file_hash="hash-a",
            text_kind="query",
        )
        self.assertEqual(model1.calls, 1)

        model2 = _FakeModel(marker=2.0)
        embedder2 = _TestEmbedder("model-a", model2, self.embedding_repository)
        second = embedder2.encode(
            "report analysis summary",
            repository=self.sqlite_repository,
            file_hash="hash-a",
            text_kind="query",
        )
        self.assertEqual(model2.calls, 0)
        self.assertEqual(first, second)
        self.assertEqual(embedder2.get_last_encode_meta()["cache_backend"], "hdf5")

    def test_migrate_legacy_sqlite_cache_to_hdf5(self) -> None:
        self.sqlite_repository.cache_embedding(
            cache_key="hash-a|model-a|sig|1|query",
            file_hash="hash-a",
            model_name="model-a",
            text_signature="sig",
            embedding_version="1",
            text_kind="query",
            embedding=[0.4, 0.5],
        )
        summary = migrate_sqlite_embedding_cache_to_hdf5(
            self.sqlite_repository,
            self.embedding_repository,
        )
        self.assertEqual(summary["migrated"], 1)
        migrated = self.embedding_repository.get_embedding("hash-a")
        self.assertIsNotNone(migrated)
        self.assertAlmostEqual(float(migrated[0]), 0.4, places=6)
        self.assertAlmostEqual(float(migrated[1]), 0.5, places=6)

    def test_confirmed_example_is_saved_in_hdf5_with_lookup_key(self) -> None:
        file_id = self.sqlite_repository.upsert_file(
            file_path="example.txt",
            file_name="example.txt",
            file_ext=".txt",
            file_size=10,
            xxhash64="hash-example",
            duplicate_of_file_id=None,
            extracted_text="example text",
        )
        classification_id = self.sqlite_repository.insert_classification(
            file_id=file_id,
            predicted_category="report",
            rule_score=0.8,
            embedding_score=0.0,
            llm_score=0.0,
            final_score=0.8,
            candidate_scores_json='{"report":0.8}',
            reasoning="test",
            status="suggested",
            large_category="reporting",
            middle_category="report",
            large_confidence=0.8,
            middle_confidence=0.8,
        )
        feedback_id = self.sqlite_repository.save_feedback(
            file_id=file_id,
            classification_id=classification_id,
            predicted_category="report",
            final_category="report",
            feedback_action="confirmed",
            user_note=None,
        )
        example_id = self.sqlite_repository.save_confirmed_example(
            file_id=file_id,
            category="report",
            source_text="example text",
            embedding=[0.1, 0.2],
            source_feedback_log_id=feedback_id,
        )
        rows = self.sqlite_repository.fetch_confirmed_examples()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["id"], example_id)
        self.assertEqual(row["embedding_key"], f"confirmed_example_{example_id}")
        stored_vector = self.embedding_repository.get_embedding(row["embedding_key"])
        self.assertIsNotNone(stored_vector)
        self.assertAlmostEqual(float(stored_vector[0]), 0.1, places=6)
        self.assertEqual(self.sqlite_repository.get_confirmed_example_embedding_stats()["with_embedding_key"], 1)

    def test_confirmed_example_legacy_json_fallback_backfills_hdf5(self) -> None:
        file_id = self.sqlite_repository.upsert_file(
            file_path="legacy.txt",
            file_name="legacy.txt",
            file_ext=".txt",
            file_size=10,
            xxhash64="hash-legacy",
            duplicate_of_file_id=None,
            extracted_text="legacy text",
        )
        classification_id = self.sqlite_repository.insert_classification(
            file_id=file_id,
            predicted_category="report",
            rule_score=0.8,
            embedding_score=0.0,
            llm_score=0.0,
            final_score=0.8,
            candidate_scores_json='{"report":0.8}',
            reasoning="test",
            status="suggested",
            large_category="reporting",
            middle_category="report",
            large_confidence=0.8,
            middle_confidence=0.8,
        )
        feedback_id = self.sqlite_repository.save_feedback(
            file_id=file_id,
            classification_id=classification_id,
            predicted_category="report",
            final_category="report",
            feedback_action="confirmed",
            user_note=None,
        )
        with self.sqlite_repository.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO confirmed_examples (
                    file_id, category, source_text, embedding_json, embedding_key, source_feedback_log_id
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    "report",
                    "legacy text",
                    "[0.3, 0.4]",
                    "",
                    feedback_id,
                ),
            )
            example_id = int(cursor.lastrowid)

        rows = self.sqlite_repository.fetch_confirmed_examples()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["embedding_key"], f"confirmed_example_{example_id}")
        stored_vector = self.embedding_repository.get_embedding(row["embedding_key"])
        self.assertIsNotNone(stored_vector)
        self.assertAlmostEqual(float(stored_vector[0]), 0.3, places=6)


if __name__ == "__main__":
    unittest.main()
