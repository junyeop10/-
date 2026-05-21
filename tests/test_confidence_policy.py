from __future__ import annotations

import unittest

from src.confidence import ConfidencePolicy


class ConfidencePolicyTest(unittest.TestCase):
    def test_review_reasons_cover_low_confidence_margin_and_conflicts(self) -> None:
        decision = ConfidencePolicy(threshold=0.7, margin=0.1).evaluate(
            confidence=0.6,
            candidate_scores={"논문": 0.6, "보고서": 0.55},
            rule_prediction="보고서",
            ml_prediction="논문",
            ml_available=True,
            embedding_prediction="계약서",
            embedding_available=True,
        )

        self.assertTrue(decision.review_required)
        self.assertIn("low_confidence", decision.review_reasons)
        self.assertIn("small_margin", decision.review_reasons)
        self.assertIn("rule_ml_conflict", decision.review_reasons)
        self.assertIn("embedding_ml_conflict", decision.review_reasons)

    def test_low_similarity_cluster_reason(self) -> None:
        decision = ConfidencePolicy().evaluate(
            confidence=0.9,
            candidate_scores={"기타": 0.9, "보고서": 0.1},
            rule_prediction="기타",
            ml_prediction="기타",
            ml_available=True,
            embedding_prediction="기타",
            embedding_available=True,
            cluster_similarity=0.1,
            in_new_cluster=True,
        )

        self.assertTrue(decision.review_required)
        self.assertIn("low_similarity_new_cluster", decision.review_reasons)


if __name__ == "__main__":
    unittest.main()
