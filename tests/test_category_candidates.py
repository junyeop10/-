from __future__ import annotations

import unittest
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

from src.cluster_candidates import ClusterCandidateFinder
from src.storage import ClassificationRepository


class CategoryCandidateFinderTest(unittest.TestCase):
    def test_min_cluster_size_is_required(self) -> None:
        finder = ClusterCandidateFinder(min_cluster_size=3)
        candidates = finder.find_candidates(
            [
                {"file_id": 1, "file_name": "medical_ai_1.pdf", "review_required": True, "compressed_text": "medical ai vision"},
                {"file_id": 2, "file_name": "medical_ai_2.pdf", "review_required": True, "compressed_text": "medical ai mri"},
            ]
        )
        self.assertEqual(candidates, [])

    def test_pending_candidate_can_be_saved(self) -> None:
        rows = [
            {"file_id": 1, "file_name": "medical_ai_1.pdf", "review_required": True, "compressed_text": "medical ai vision"},
            {"file_id": 2, "file_name": "medical_ai_2.pdf", "review_required": True, "compressed_text": "medical ai mri"},
            {"file_id": 3, "file_name": "medical_ai_3.pdf", "review_required": True, "compressed_text": "medical ai diagnosis"},
        ]
        candidates = ClusterCandidateFinder(min_cluster_size=3).find_candidates(rows)
        self.assertTrue(candidates)

        base_dir = Path("tests_runtime") / f"candidate_{uuid4().hex}"
        base_dir.mkdir(parents=True, exist_ok=True)
        try:
            repository = ClassificationRepository(base_dir / "test.db")
            repository.initialize_database()
            candidate_id = repository.insert_category_candidate(
                source="test",
                suggested_name=candidates[0].suggested_name,
                representative_file_ids=candidates[0].representative_file_ids,
                evidence=candidates[0].evidence,
            )
            self.assertGreater(candidate_id, 0)
            stats = repository.get_stats()
            self.assertEqual(stats["category_candidates_count"], 1)
        finally:
            rmtree(base_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
