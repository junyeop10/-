from __future__ import annotations

import json
import unittest
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

from src.api_category_labeler import call_category_labeling_api
from src.categories_cluster_pipeline import (
    CATEGORIES_PIPELINE_VERSION,
    build_type_clustering_vectors,
    embed_documents_three_sections,
    filter_low_probability_documents,
    merge_nearby_clusters,
    run_categories_clustering,
    split_three_sections,
)
from src.cluster_projection import build_cluster_projection, render_cluster_projection_html
from src.clustering_support import build_cluster_summaries, build_clustering_vector, build_parent_cluster_groups, cluster_embeddings
from src.embedding_support import build_embedding_text
from src.evidence_pipeline import (
    API_READER_REQUIRED,
    OCR_USED,
    TEXT_OK,
    build_document_evidence,
)
from src.feature_vector_builder import (
    build_optional_layout_vector,
    build_pattern_vector,
    get_layout_confidence,
)
from src.ocr_support import resize_ocr_image
from src.type_embedding_builder import build_type_embedding_text, suppress_noise_terms
from tools.inspect_cluster_pipeline import run_pipeline


class _FakeEmbedder:
    def encode_many(self, texts, repository=None, file_hashes=None, text_kind="query", embedding_version="1"):
        del repository, file_hashes, text_kind, embedding_version
        embeddings = []
        for text in texts:
            lowered = str(text).lower()
            if "contract" in lowered or "계약" in lowered:
                embeddings.append([1.0, 0.0, 0.0])
            elif "receipt" in lowered or "영수" in lowered:
                embeddings.append([0.0, 1.0, 0.0])
            else:
                embeddings.append([0.0, 0.0, 1.0])
        return embeddings


class _SegmentEmbedder:
    def encode_many(self, texts, repository=None, file_hashes=None, text_kind="query", embedding_version="1"):
        del repository, file_hashes, text_kind, embedding_version
        mapping = {
            "aaa": [1.0, 0.0, 0.0],
            "bbb": [0.0, 1.0, 0.0],
            "ccc": [0.0, 0.0, 1.0],
        }
        return [mapping[str(text)] for text in texts]


class _EmptySegmentEmbedder:
    def encode_many(self, texts, repository=None, file_hashes=None, text_kind="query", embedding_version="1"):
        del repository, file_hashes, text_kind, embedding_version
        return [[1.0, 1.0] if not str(text) else [1.0, 0.0] for text in texts]


class ClusterPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.base_dir = Path("tests_runtime") / f"cluster_pipeline_{uuid4().hex}"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        rmtree(self.base_dir, ignore_errors=True)

    def test_text_ok_evidence_contains_structure_and_embedding_text(self) -> None:
        path = self.base_dir / "contract.txt"
        path.write_text("계약서 contract 갑 을 계약기간 지급 조건 " * 10, encoding="utf-8")

        evidence = build_document_evidence(path, min_text_chars=20)
        embedding_text = build_embedding_text(evidence)

        self.assertEqual(evidence["extraction_status"], TEXT_OK)
        self.assertIn("structural_features", evidence)
        self.assertTrue(embedding_text.strip())
        self.assertIn("structure:", embedding_text)

    def test_evidence_cache_reuses_unchanged_file(self) -> None:
        path = self.base_dir / "cached.txt"
        cache_dir = self.base_dir / "evidence_cache"
        path.write_text("계약서 contract 갑 을 계약기간 지급 조건 " * 10, encoding="utf-8")

        first = build_document_evidence(
            path,
            min_text_chars=20,
            evidence_cache_dir=cache_dir,
            evidence_cache_enabled=True,
        )
        second = build_document_evidence(
            path,
            min_text_chars=20,
            evidence_cache_dir=cache_dir,
            evidence_cache_enabled=True,
        )

        self.assertFalse(first["evidence_cache_hit"])
        self.assertTrue(second["evidence_cache_hit"])
        self.assertEqual(first["sampled_text"], second["sampled_text"])

    def test_old_rule_signals_do_not_expose_category_labels(self) -> None:
        path = self.base_dir / "contract.txt"
        path.write_text("계약서 contract 갑 을 계약기간 지급 조건 " * 10, encoding="utf-8")
        rules = [{"category": "계약서", "rule_type": "keyword", "pattern": "계약", "weight": 1.0}]

        evidence = build_document_evidence(path, rules=rules, min_text_chars=20)

        signals = evidence["old_rule_signals"]["keyword_signals"]
        self.assertTrue(signals)
        self.assertNotIn("source_category", signals[0])
        self.assertEqual(signals[0]["signal"], "계약")

    def test_short_pdf_uses_ocr_fallback_when_ocr_has_evidence(self) -> None:
        path = self.base_dir / "scan.pdf"
        path.write_bytes(b"not a real pdf")

        evidence = build_document_evidence(
            path,
            min_text_chars=20,
            text_extractor=lambda *_args, **_kwargs: "",
            ocr_extractor=lambda *_args, **_kwargs: {
                "ok": True,
                "text": "OCR contract text payment deadline " * 5,
                "pages_scanned": 1,
                "error": "",
            },
        )

        self.assertEqual(evidence["extraction_status"], OCR_USED)
        self.assertIn(OCR_USED, evidence["status_flags"])

    def test_ocr_insufficient_marks_api_reader_required(self) -> None:
        path = self.base_dir / "scan.pdf"
        path.write_bytes(b"not a real pdf")

        evidence = build_document_evidence(
            path,
            min_text_chars=50,
            text_extractor=lambda *_args, **_kwargs: "",
            ocr_extractor=lambda *_args, **_kwargs: {"ok": True, "text": "x", "pages_scanned": 1, "error": ""},
        )

        self.assertIn(API_READER_REQUIRED, evidence["status_flags"])
        self.assertTrue(evidence["api_reader"]["required"])

    def test_cluster_ids_noise_and_payload_representatives_are_built(self) -> None:
        documents = [
            {"evidence": {"filename": "a.txt", "top_tokens": [], "sampled_text": "a", "structural_features": {}, "text_stats": {}}},
            {"evidence": {"filename": "b.txt", "top_tokens": [], "sampled_text": "b", "structural_features": {}, "text_stats": {}}},
            {"evidence": {"filename": "c.txt", "top_tokens": [], "sampled_text": "c", "structural_features": {}, "text_stats": {}}},
        ]
        embeddings = [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]
        cluster_ids = [0, 0, -1]

        summaries, noise = build_cluster_summaries(documents, embeddings, cluster_ids, representative_top_k=2)

        self.assertEqual(documents[0]["cluster_id"], 0)
        self.assertEqual(noise[0]["cluster_id"], -1)
        self.assertTrue(summaries[0]["representative_documents"])

    def test_cluster_embeddings_assigns_id_to_each_document(self) -> None:
        result = cluster_embeddings([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], min_cluster_size=2)

        self.assertEqual(len(result["cluster_ids"]), 3)

    def test_parent_type_candidates_group_similar_fine_clusters_without_absorbing_noise(self) -> None:
        documents = [
            {"evidence": {"filename": "a.txt", "top_tokens": []}},
            {"evidence": {"filename": "b.txt", "top_tokens": []}},
            {"evidence": {"filename": "c.txt", "top_tokens": []}},
            {"evidence": {"filename": "noise.txt", "top_tokens": []}},
        ]
        embeddings = [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [-1.0, 0.0]]

        summaries, mapping = build_parent_cluster_groups(documents, embeddings, [0, 1, 2, -1])

        self.assertEqual(mapping[0], mapping[1])
        self.assertNotEqual(mapping[0], mapping[2])
        self.assertEqual(documents[3]["parent_cluster_id"], -1)
        self.assertEqual(sum(summary["document_count"] for summary in summaries), 3)

    def test_categories_pipeline_splits_and_combines_three_sections(self) -> None:
        documents = [{"file_hash": "hash", "evidence": {"sampled_text": "aaabbbccc"}}]

        embeddings = embed_documents_three_sections(documents, embedder=_SegmentEmbedder())

        self.assertEqual(split_three_sections("aaabbbccc"), ("aaa", "bbb", "ccc"))
        self.assertEqual(documents[0]["embedding_segments"]["weights"], [0.5, 0.25, 0.25])
        self.assertAlmostEqual(sum(value * value for value in embeddings[0]), 1.0, places=5)
        self.assertGreater(embeddings[0][0], embeddings[0][1])
        self.assertEqual(embeddings[0][1], embeddings[0][2])

    def test_categories_pipeline_skips_umap_for_tiny_batches(self) -> None:
        documents = [
            {"file_hash": "a", "evidence": {"sampled_text": "contract agreement payment"}},
            {"file_hash": "b", "evidence": {"sampled_text": "contract agreement renewal"}},
        ]

        result = run_categories_clustering(documents, embedder=_FakeEmbedder(), min_cluster_size=3)

        self.assertEqual(result["clustering_vector_version"], CATEGORIES_PIPELINE_VERSION)
        self.assertEqual(result["reducer_result"]["status"], "not_enough_documents")
        self.assertEqual(result["cluster_result"]["cluster_ids"], [-1, -1])

    def test_categories_pipeline_zeroes_empty_sections(self) -> None:
        documents = [{"file_hash": "short", "evidence": {"sampled_text": "a"}}]

        embeddings = embed_documents_three_sections(documents, embedder=_EmptySegmentEmbedder())

        self.assertEqual(embeddings[0], [1.0, 0.0])

    def test_type_clustering_vectors_append_pattern_signal(self) -> None:
        documents = [{"pattern_vector": [1.0, 0.0]}]

        vectors = build_type_clustering_vectors(documents, [[0.0, 1.0]])

        self.assertEqual(len(vectors[0]), 4)
        self.assertAlmostEqual(sum(value * value for value in vectors[0]), 1.0, places=5)
        self.assertGreater(vectors[0][1], vectors[0][2])

    def test_nearby_clusters_are_conservatively_merged(self) -> None:
        labels, summary = merge_nearby_clusters(
            [0, 0, 1, 1, 2, 2],
            [
                [1.0, 0.0],
                [0.999, 0.001],
                [0.998, 0.002],
                [0.997, 0.003],
                [0.0, 1.0],
                [0.001, 0.999],
            ],
        )

        self.assertEqual(labels[:4], [0, 0, 0, 0])
        self.assertEqual(labels[4:], [1, 1])
        self.assertEqual(summary["cluster_count_before"], 3)
        self.assertEqual(summary["cluster_count_after"], 2)

    def test_low_probability_documents_move_to_noise(self) -> None:
        labels, summary = filter_low_probability_documents(
            [0, 0, 1, -1],
            [1.0, 0.2, 0.8, 0.0],
            minimum_probability=0.5,
        )

        self.assertEqual(labels, [0, -1, 1, -1])
        self.assertEqual(summary["moved_to_noise"], 1)

    def test_cluster_projection_contains_coordinates_and_neighbors(self) -> None:
        documents = [
            {"filename": "a.txt", "evidence": {"filename": "a.txt", "top_tokens": []}, "cluster_id": 0},
            {"filename": "b.txt", "evidence": {"filename": "b.txt", "top_tokens": []}, "cluster_id": 0},
            {"filename": "c.txt", "evidence": {"filename": "c.txt", "top_tokens": []}, "cluster_id": -1},
        ]
        projection = build_cluster_projection(
            documents,
            [[1.0, 0.0, 0.0], [0.99, 0.01, 0.0], [0.0, 1.0, 0.0]],
            [0, 0, -1],
            probabilities=[0.9, 0.8, 0.0],
        )
        html = render_cluster_projection_html(projection)

        self.assertEqual(len(projection["points"]), 3)
        self.assertIn("nearest_neighbors", projection["points"][0])
        self.assertIn("Cluster Projection", html)

    def test_type_embedding_text_preserves_type_terms_and_suppresses_noise(self) -> None:
        evidence = {
            "filename": "주식회사 샘플 계약서.pdf",
            "filename_tokens": ["주식회사", "샘플", "계약서"],
            "top_tokens": [{"token": "계약", "count": 3}, {"token": "샘플회사", "count": 2}],
            "sampled_text": "주식회사 샘플\n서울시 강남구 테스트로 12\n계약 금액과 서명 날짜를 확인한다.",
            "compressed_preview": "title: 용역 계약서\nheadings: 제1조 목적 | 제2조 금액",
            "structural_features": {"clause_pattern_score": 0.8, "legal_term_density": 0.2},
            "layout_features": {},
            "text_stats": {},
        }

        type_text = build_type_embedding_text(evidence)
        suppressed = suppress_noise_terms(evidence["sampled_text"])

        self.assertIn("contract", type_text)
        self.assertIn("계약", type_text)
        self.assertIn("[company_noise]", suppressed)
        self.assertIn("[address_noise]", suppressed)

    def test_pattern_and_layout_vectors_work_without_layout(self) -> None:
        evidence = {
            "filename": "견적서.txt",
            "sampled_text": "견적서 합계 1,200,000원 작성일 2026-01-03",
            "compressed_preview": "",
            "structural_features": {"legal_term_density": 0.0, "clause_pattern_score": 0.0},
            "layout_features": {},
        }

        pattern_vector = build_pattern_vector(evidence)
        layout_vector = build_optional_layout_vector(evidence)
        layout_confidence = get_layout_confidence(evidence)
        clustering_vector = build_clustering_vector([1.0, 0.0, 0.0], pattern_vector, layout_vector, layout_confidence=layout_confidence)

        self.assertEqual(len(pattern_vector), 12)
        self.assertEqual(layout_confidence, 0.0)
        self.assertTrue(clustering_vector)

    def test_api_stub_does_not_call_real_api(self) -> None:
        response = call_category_labeling_api({"cluster_id": 1})

        self.assertEqual(response["status"], "api_not_configured")
        self.assertIsNone(response["category"])

    def test_ocr_resize_preserves_aspect_ratio(self) -> None:
        from PIL import Image

        image = Image.new("RGB", (2400, 1200), "white")
        resized = resize_ocr_image(image, max_edge=1200, min_edge=500)

        self.assertEqual(resized.size, (1200, 600))

    def test_tool_pipeline_writes_payloads_without_api_call(self) -> None:
        input_dir = self.base_dir / "input"
        output_dir = self.base_dir / "outputs"
        input_dir.mkdir()
        (input_dir / "contract_a.txt").write_text("contract agreement payment deadline " * 8, encoding="utf-8")
        (input_dir / "contract_b.txt").write_text("contract renewal payment terms " * 8, encoding="utf-8")

        result = run_pipeline(
            input_path=input_dir,
            output_dir=output_dir,
            db_path=str(self.base_dir / "test.db"),
            categories_path="data/categories.json",
            config_path=Path("data/app_config.json"),
            min_cluster_size=2,
            min_samples=1,
            representative_top_k=2,
            reducer=None,
            ocr_enabled=False,
            embedder=_FakeEmbedder(),
        )

        self.assertFalse(result["api_call_performed"])
        payloads = json.loads((output_dir / "cluster_payloads.json").read_text(encoding="utf-8"))
        evidence = json.loads((output_dir / "cluster_evidence.json").read_text(encoding="utf-8"))
        summary = json.loads((output_dir / "cluster_run_summary.json").read_text(encoding="utf-8"))
        projection = json.loads((output_dir / "cluster_projection.json").read_text(encoding="utf-8"))
        self.assertTrue(evidence)
        self.assertEqual(summary["clustering_vector_version"], CATEGORIES_PIPELINE_VERSION)
        self.assertTrue((output_dir / "cluster_projection.html").exists())
        self.assertIn("points", projection)
        self.assertIn("type_embedding_text", evidence[0])
        self.assertIn("pattern_vector", evidence[0])
        self.assertIn("semantic_embedding", evidence[0])
        self.assertIn("clustering_input_vector", evidence[0])
        self.assertIn("reduced_vector", evidence[0])
        self.assertIn("clustering_vector", evidence[0])
        if payloads:
            self.assertIn("representative_documents", payloads[0])


if __name__ == "__main__":
    unittest.main()
