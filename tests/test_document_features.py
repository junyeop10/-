from __future__ import annotations

import unittest
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

from src.document_features import DocumentFeatureExtractor
from src.storage import ClassificationRepository


class DocumentFeatureExtractorTest(unittest.TestCase):
    def test_extracts_filename_and_structure_signals(self) -> None:
        extractor = DocumentFeatureExtractor()
        bundle = extractor.extract(
            file_name="Transformer_MRI_Review_Paper.pdf",
            file_ext=".pdf",
            file_size=123,
            text="Abstract\nThis paper studies MRI image segmentation. doi:10.1000/test [1]\nReferences\n[1] et al.",
        )

        self.assertTrue(bundle.filename_features["has_paper_hint"])
        self.assertTrue(bundle.structural_features["has_abstract"])
        self.assertTrue(bundle.structural_features["has_references"])
        self.assertTrue(bundle.structural_features["has_doi"])
        self.assertGreaterEqual(bundle.structural_features["citation_count"], 2)
        self.assertIn("Transformer_MRI_Review_Paper.pdf".lower(), bundle.compressed_text)

    def test_feature_cache_can_reuse_file_hash_and_version(self) -> None:
        base_dir = Path("tests_runtime") / f"features_{uuid4().hex}"
        base_dir.mkdir(parents=True, exist_ok=True)
        try:
            repository = ClassificationRepository(base_dir / "test.db")
            repository.initialize_database()
            file_id = repository.upsert_file(
                file_path="a.pdf",
                file_name="a.pdf",
                file_ext=".pdf",
                file_size=10,
                xxhash64="hash-a",
                duplicate_of_file_id=None,
                extracted_text="Abstract and References",
            )
            bundle = DocumentFeatureExtractor().extract(file_name="a.pdf", file_ext=".pdf", text="Abstract")
            repository.upsert_document_features(
                file_id=file_id,
                file_hash="hash-a",
                extractor_version=bundle.feature_version,
                filename_features=bundle.filename_features,
                metadata_features=bundle.metadata_features,
                structural_features=bundle.structural_features,
                text_stats=bundle.text_stats,
                compressed_text=bundle.compressed_text,
                compressed_text_hash=bundle.compressed_text_hash,
            )
            cached = repository.get_document_features_by_hash("hash-a", bundle.feature_version)
            self.assertIsNotNone(cached)
            self.assertEqual(cached["compressed_text_hash"], bundle.compressed_text_hash)
        finally:
            rmtree(base_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
