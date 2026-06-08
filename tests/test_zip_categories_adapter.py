from __future__ import annotations

import json
import unittest
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

from src.zip_categories_adapter import (
    ZIP_PIPELINE_VERSION,
    ZIP_READER_ORIGINAL,
    ZIP_UNCLASSIFIED,
    build_zip_original_evidence,
    classification_result_from_zip,
    normalize_category_name,
    run_zip_categories_pipeline,
)


class _Fake384Embedder:
    def encode_many(self, texts, repository=None, file_hashes=None, text_kind="query", embedding_version="1"):
        del repository, file_hashes, text_kind, embedding_version
        results = []
        for text in texts:
            vector = [0.0] * 384
            lowered = str(text).lower()
            vector[0 if "계약" in lowered or "견적" in lowered else 1] = 1.0
            results.append(vector)
        return results


class _Zero384Embedder:
    def encode_many(self, texts, repository=None, file_hashes=None, text_kind="query", embedding_version="1"):
        del repository, file_hashes, text_kind, embedding_version
        return [[0.0] * 384 for _text in texts]


class _Spread384Embedder:
    def encode_many(self, texts, repository=None, file_hashes=None, text_kind="query", embedding_version="1"):
        del repository, file_hashes, text_kind, embedding_version
        results = []
        for index, _text in enumerate(texts):
            vector = [0.0] * 384
            vector[index % 6] = 1.0
            results.append(vector)
        return results


class _FeedbackRepository:
    def fetch_confirmed_examples(self):
        vector = [0.0] * 384
        vector[0] = 1.0
        return [
            {
                "category": "6. 견적_계약_정산",
                "embedding_json": json.dumps(vector),
                "source_text": "사용자 확정 견적 계약 예시",
            }
        ]


class _RecordingEmbedder:
    def __init__(self) -> None:
        self.calls = []

    def encode_many(self, texts, repository=None, file_hashes=None, text_kind="query", embedding_version="1"):
        del repository, file_hashes, embedding_version
        self.calls.append({"text_kind": text_kind, "texts": list(texts)})
        return [[1.0] + [0.0] * 383 for _text in texts]


class ZipCategoriesAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.base_dir = Path("tests_runtime") / f"zip_adapter_{uuid4().hex}"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        rmtree(self.base_dir, ignore_errors=True)

    def test_original_reader_uses_vendor_filename_fallback(self) -> None:
        path = self.base_dir / "계약서_샘플.hwp"
        path.write_bytes(b"binary")

        evidence = build_zip_original_evidence(path)

        self.assertEqual(evidence["reader_mode"], ZIP_READER_ORIGINAL)
        self.assertEqual(evidence["read_method"], "filename")
        self.assertIn("계약서", evidence["sampled_text"])

    def test_rule_stage_and_tiny_batch_cluster_fallback(self) -> None:
        path = self.base_dir / "서비스_계약서.txt"
        path.write_text("용역 범위와 지급 조건", encoding="utf-8")
        evidence = build_zip_original_evidence(path)

        result = run_zip_categories_pipeline(
            [{"file_path": str(path), "file_hash": evidence["file_hash"], "evidence": evidence}],
            embedder=_Zero384Embedder(),
        )
        item = result["documents"][0]

        self.assertEqual(result["pipeline_version"], ZIP_PIPELINE_VERSION)
        self.assertEqual(item["category"], "견적_계약_정산")
        self.assertTrue(item["rule_confirmed"])
        self.assertEqual(item["cluster_id"], -1)
        self.assertEqual(result["cluster_result"]["status"], "disabled")
        self.assertFalse(result["cluster_result"]["enabled"])

    def test_semantic_low_confidence_is_sent_to_review(self) -> None:
        path = self.base_dir / "메모.txt"
        path.write_text("아무 내용 없는 개인 메모", encoding="utf-8")
        evidence = build_zip_original_evidence(path)

        result = run_zip_categories_pipeline(
            [{"file_path": str(path), "file_hash": evidence["file_hash"], "evidence": evidence}],
            embedder=_Zero384Embedder(),
        )
        item = result["documents"][0]
        classification = classification_result_from_zip(item)

        self.assertEqual(item["category"], ZIP_UNCLASSIFIED)
        self.assertTrue(classification.review_required)
        self.assertEqual(classification.middle_category, ZIP_UNCLASSIFIED)

    def test_six_documents_skip_temporarily_disconnected_umap_and_hdbscan(self) -> None:
        documents = []
        for index in range(6):
            path = self.base_dir / f"메모_{index}.txt"
            path.write_text(f"일반 메모 본문 {index}", encoding="utf-8")
            evidence = build_zip_original_evidence(path)
            documents.append({"file_path": str(path), "file_hash": evidence["file_hash"], "evidence": evidence})

        result = run_zip_categories_pipeline(documents, embedder=_Spread384Embedder())

        self.assertEqual(result["cluster_result"]["status"], "disabled")
        self.assertEqual(result["cluster_result"]["reducer"], "disabled")
        self.assertFalse(result["cluster_result"]["enabled"])
        self.assertEqual(len(result["cluster_result"]["cluster_ids"]), 6)
        self.assertEqual(result["cluster_result"]["cluster_ids"], [-1] * 6)

    def test_confirmed_feedback_is_loaded_into_semantic_store(self) -> None:
        path = self.base_dir / "메모.txt"
        path.write_text("일반 본문", encoding="utf-8")
        evidence = build_zip_original_evidence(path)

        result = run_zip_categories_pipeline(
            [{"file_path": str(path), "file_hash": evidence["file_hash"], "evidence": evidence}],
            embedder=_Fake384Embedder(),
            repository=_FeedbackRepository(),
        )

        self.assertEqual(result["feedback_examples_used"], 1)
        self.assertEqual(result["profile_enhancements"]["confirmed_examples_seen"], 1)
        self.assertEqual(result["profile_enhancements"]["enhanced_category_count"], 1)

    def test_confirmed_feedback_enhances_category_seed_text(self) -> None:
        path = self.base_dir / "메모.txt"
        path.write_text("일반 본문", encoding="utf-8")
        evidence = build_zip_original_evidence(path)
        embedder = _RecordingEmbedder()

        result = run_zip_categories_pipeline(
            [{"file_path": str(path), "file_hash": evidence["file_hash"], "evidence": evidence}],
            embedder=embedder,
            repository=_FeedbackRepository(),
        )

        seed_call = next(call for call in embedder.calls if call["text_kind"] == "categories_zip_description_seed")
        enhanced_seed = "\n".join(seed_call["texts"])
        self.assertIn("confirmed_examples_count: 1", enhanced_seed)
        self.assertIn("사용자", enhanced_seed)
        self.assertEqual(result["profile_enhancements"]["enhanced_category_count"], 1)

    def test_category_display_order_prefix_is_removed(self) -> None:
        self.assertEqual(normalize_category_name("1. 공고_지침_양식"), "공고_지침_양식")
        self.assertEqual(normalize_category_name("  7. 기업 인증서"), "기업 인증서")
        self.assertEqual(normalize_category_name("기타"), "기타")


if __name__ == "__main__":
    unittest.main()
