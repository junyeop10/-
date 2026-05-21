from __future__ import annotations

import time
import unittest
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

from src.storage import ClassificationRepository
from src.vectorizer import SentenceTransformerEmbedder


class _FakeVector:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return list(self._values)


class _FakeModel:
    def __init__(self, marker: float, delay: float = 0.0) -> None:
        self.marker = marker
        self.delay = delay
        self.calls = 0

    def encode(self, texts, normalize_embeddings: bool = True):
        del normalize_embeddings
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if isinstance(texts, list):
            return [_FakeVector([self.marker, float(len(text))]) for text in texts]
        return _FakeVector([self.marker, float(len(texts))])


class _TestEmbedder(SentenceTransformerEmbedder):
    def __init__(self, model_name: str, fake_model: _FakeModel) -> None:
        super().__init__(model_name=model_name)
        self._fake_model = fake_model

    def _load_model(self):
        return self._fake_model


class EmbeddingCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.base_dir = Path("tests_runtime") / f"embedding_cache_{uuid4().hex}"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.base_dir / "test.db"
        self.repository = ClassificationRepository(self.db_path)
        self.repository.initialize_database()

    def tearDown(self) -> None:
        if self.base_dir.exists():
            rmtree(self.base_dir, ignore_errors=True)

    def test_cache_reuse_across_embedder_instances(self) -> None:
        model1 = _FakeModel(marker=1.0)
        embedder1 = _TestEmbedder("model-a", model1)
        first = embedder1.encode("report analysis summary", repository=self.repository, file_hash="hash-a", text_kind="query")
        self.assertEqual(model1.calls, 1)

        model2 = _FakeModel(marker=2.0)
        embedder2 = _TestEmbedder("model-a", model2)
        second = embedder2.encode("report analysis summary", repository=self.repository, file_hash="hash-a", text_kind="query")
        self.assertEqual(model2.calls, 0)
        self.assertEqual(first, second)

    def test_cache_invalidation_when_text_changes(self) -> None:
        model1 = _FakeModel(marker=1.0)
        embedder1 = _TestEmbedder("model-a", model1)
        embedder1.encode("report analysis summary", repository=self.repository, file_hash="hash-a", text_kind="query")

        model2 = _FakeModel(marker=2.0)
        embedder2 = _TestEmbedder("model-a", model2)
        changed = embedder2.encode("report analysis revised", repository=self.repository, file_hash="hash-a", text_kind="query")
        self.assertEqual(model2.calls, 1)
        self.assertEqual(changed[0], 2.0)

    def test_model_switch_invalidation(self) -> None:
        model1 = _FakeModel(marker=1.0)
        embedder1 = _TestEmbedder("model-a", model1)
        embedder1.encode("invoice amount due", repository=self.repository, file_hash="hash-b", text_kind="query")

        model2 = _FakeModel(marker=2.0)
        embedder2 = _TestEmbedder("model-b", model2)
        result = embedder2.encode("invoice amount due", repository=self.repository, file_hash="hash-b", text_kind="query")
        self.assertEqual(model2.calls, 1)
        self.assertEqual(result[0], 2.0)

    def test_repeated_cache_hit_is_faster_than_first_generation(self) -> None:
        slow_model = _FakeModel(marker=3.0, delay=0.05)
        embedder = _TestEmbedder("model-slow", slow_model)

        first_start = time.perf_counter()
        first = embedder.encode("business plan target market", repository=self.repository, file_hash="hash-c", text_kind="query")
        first_elapsed = time.perf_counter() - first_start

        second_start = time.perf_counter()
        second = embedder.encode("business plan target market", repository=self.repository, file_hash="hash-c", text_kind="query")
        second_elapsed = time.perf_counter() - second_start

        self.assertEqual(first, second)
        self.assertLess(second_elapsed, first_elapsed)
        self.assertTrue(embedder.get_last_encode_meta().get("cache_hit"))

    def test_cache_stats_and_clear(self) -> None:
        model = _FakeModel(marker=1.0)
        embedder = _TestEmbedder("model-a", model)
        embedder.encode("statement of work deliverable", repository=self.repository, file_hash="hash-d", text_kind="document")
        embedder.encode("statement of work deliverable", repository=self.repository, file_hash="hash-d", text_kind="document")

        stats = self.repository.get_embedding_cache_stats()
        self.assertEqual(stats["entries"], 1)
        self.assertGreaterEqual(stats["total_hits"], 1)

        deleted = self.repository.clear_embedding_cache()
        self.assertEqual(deleted, 1)
        cleared_stats = self.repository.get_embedding_cache_stats()
        self.assertEqual(cleared_stats["entries"], 0)

    def test_last_encode_meta_reports_cache_hit(self) -> None:
        model = _FakeModel(marker=1.0)
        embedder = _TestEmbedder("model-a", model)
        embedder.encode("policy handbook", repository=self.repository, file_hash="hash-e", text_kind="query")
        first_meta = embedder.get_last_encode_meta()
        self.assertFalse(first_meta["cache_hit"])
        embedder.encode("policy handbook", repository=self.repository, file_hash="hash-e", text_kind="query")
        second_meta = embedder.get_last_encode_meta()
        self.assertTrue(second_meta["cache_hit"])

    def test_compressed_text_kind_uses_distinct_cache_entry(self) -> None:
        model = _FakeModel(marker=1.0)
        embedder = _TestEmbedder("model-a", model)
        regular = embedder.encode("full body text", repository=self.repository, file_hash="hash-f", text_kind="query")
        compressed = embedder.encode(
            "filename title summary",
            repository=self.repository,
            file_hash="hash-f",
            text_kind="compressed_query",
            embedding_version="2.1-compressed",
        )

        self.assertNotEqual(regular, compressed)
        self.assertEqual(model.calls, 2)


if __name__ == "__main__":
    unittest.main()
