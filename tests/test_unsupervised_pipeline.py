from __future__ import annotations

import unittest
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

from src.storage import ClassificationRepository
from src.unknown_pool import decide_unknown_pool_entry
from src.unsupervised_clustering import ClusterInput, SklearnTextClusterer


class _FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def encode_many(self, texts, repository=None, file_hashes=None, text_kind="query", embedding_version="1"):
        self.calls.append(
            {
                "texts": list(texts),
                "repository": repository,
                "file_hashes": list(file_hashes or []),
                "text_kind": text_kind,
                "embedding_version": embedding_version,
            }
        )
        return [
            [1.0, 0.0],
            [0.98, 0.02],
            [0.0, 1.0],
            [0.02, 0.98],
        ]


class UnsupervisedPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.base_dir = Path("tests_runtime") / f"unsupervised_{uuid4().hex}"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.repository = ClassificationRepository(self.base_dir / "test.db")
        self.repository.initialize_database()

    def tearDown(self) -> None:
        rmtree(self.base_dir, ignore_errors=True)

    def test_unknown_decision_uses_low_confidence_and_margin(self) -> None:
        decision = decide_unknown_pool_entry(
            confidence=0.41,
            candidate_scores={"계약서": 0.41, "보고서": 0.39},
            review_required=True,
            text="짧은 문서",
        )

        self.assertTrue(decision.should_store)
        self.assertIn("low_confidence", decision.reasons)
        self.assertIn("small_margin", decision.reasons)

    def test_unknown_pool_persists_rows(self) -> None:
        row_id = self.repository.save_unknown_pool_entry(
            file_hash="hash-a",
            text_hash="text-a",
            cleaned_text="계약 금액 지급 기한 관련 미확정 문서",
            nearest_category="계약서",
            nearest_similarity=0.44,
            reason="low_confidence",
        )

        rows = self.repository.list_unknown_pool()

        self.assertGreater(row_id, 0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["nearest_category"], "계약서")

    def test_batch_clusterer_runs_without_realtime_classification(self) -> None:
        items = [
            ClusterInput(1, "a", "계약 금액 지급 기한 계약기간"),
            ClusterInput(2, "b", "계약서 손해배상 지급 기한"),
            ClusterInput(3, "c", "영수증 카드 승인번호 합계금액"),
            ClusterInput(4, "d", "영수증 공급가액 부가세 합계금액"),
        ]
        clusterer = SklearnTextClusterer(eps=0.9, min_samples=2)

        result = clusterer.fit_predict(items)

        self.assertEqual(result.metrics["status"], "ok")
        self.assertEqual(len(result.assignments), 4)
        self.assertIn("cluster_count", result.metrics)

    def test_batch_clusterer_generates_embeddings_before_clustering(self) -> None:
        items = [
            ClusterInput(1, "a", "contract payment deadline"),
            ClusterInput(2, "b", "contract renewal deadline"),
            ClusterInput(3, "c", "receipt card approval"),
            ClusterInput(4, "d", "receipt supply total"),
        ]
        embedder = _FakeEmbedder()
        clusterer = SklearnTextClusterer(min_cluster_size=2, min_samples=2, embedder=embedder)

        result = clusterer.fit_predict(items)

        self.assertEqual(len(embedder.calls), 1)
        self.assertEqual(embedder.calls[0]["text_kind"], "unknown_pool")
        self.assertEqual(result.metrics["embedding_stage"], "sentence_transformer")
        self.assertIn(result.metrics["cluster_stage"], {"hdbscan", "dbscan_after_embedding"})


if __name__ == "__main__":
    unittest.main()
