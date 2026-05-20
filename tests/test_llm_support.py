from __future__ import annotations

import unittest
from unittest.mock import patch

from src.classifier import ClassificationResult
from src.cli import maybe_refine_with_llm
from src.llm_support import LLMDecision, aggregate_category_scores, should_use_llm


class LocalLLMSupportTest(unittest.TestCase):
    def test_should_use_llm_thresholds(self) -> None:
        self.assertFalse(should_use_llm(0.8))
        self.assertFalse(should_use_llm(0.95))
        self.assertTrue(should_use_llm(0.79))
        self.assertTrue(should_use_llm(0.4))
        self.assertTrue(should_use_llm(0.39))

    def test_aggregate_category_scores_maps_base_categories(self) -> None:
        aggregated = aggregate_category_scores(
            {
                "계약서": 0.72,
                "영수증": 0.64,
                "발표자료": 0.51,
                "검토필요": 0.10,
            }
        )
        self.assertEqual(aggregated["legal/계약서"], 0.72)
        self.assertEqual(aggregated["reporting/발표자료"], 0.51)
        self.assertEqual(aggregated["finance/청구서"], 0.64)
        self.assertEqual(aggregated["miscellaneous/검토필요"], 0.10)

    @patch("src.cli.classify_with_ollama")
    def test_maybe_refine_with_llm_updates_ambiguous_result(self, mock_classify_with_ollama) -> None:
        mock_classify_with_ollama.return_value = LLMDecision(
            recommended_category="reporting/보고서",
            confidence=0.83,
            reason="회의 결과와 보고 성격이 강함",
        )
        result = ClassificationResult(
            predicted_category="보고서",
            confidence=0.62,
            final_score=0.62,
            rule_score=0.60,
            embedding_score=0.41,
            llm_score=0.0,
            feedback_score=0.0,
            duplicate_score=0.0,
            similarity_score=0.41,
            embedding_used=True,
            review_required=True,
            matched_rules=["보고서", "분석"],
            candidate_scores={"보고서": 0.62},
            reasoning="recommend=reporting/보고서",
            query_embedding=[],
            large_category="reporting",
            middle_category="보고서",
        )

        refined = maybe_refine_with_llm(
            result=result,
            evidence_text="회의 결과 및 보고 내용을 정리했다.",
            use_llm=True,
            llm_model="qwen2.5:3b",
            llm_runtime={"available": True},
        )

        self.assertEqual(refined.predicted_category, "보고서")
        self.assertEqual(refined.large_category, "reporting")
        self.assertEqual(refined.llm_score, 0.83)
        self.assertFalse(refined.review_required)
        self.assertEqual(refined.explanation.get("llm_status"), "applied")

    @patch("src.cli.classify_with_ollama", side_effect=RuntimeError("ollama unavailable"))
    def test_maybe_refine_with_llm_keeps_existing_result_on_failure(self, _mock_classify_with_ollama) -> None:
        result = ClassificationResult(
            predicted_category="공지",
            confidence=0.55,
            final_score=0.55,
            rule_score=0.55,
            embedding_score=0.0,
            llm_score=0.0,
            feedback_score=0.0,
            duplicate_score=0.0,
            similarity_score=0.0,
            embedding_used=False,
            review_required=True,
            matched_rules=["공지"],
            candidate_scores={"공지": 0.55},
            reasoning="recommend=operations/공지",
            query_embedding=[],
            large_category="operations",
            middle_category="공지",
        )

        refined = maybe_refine_with_llm(
            result=result,
            evidence_text="공고 제출 서류를 확인했다.",
            use_llm=True,
            llm_model="qwen2.5:3b",
            llm_runtime={"available": True},
        )

        self.assertEqual(refined.predicted_category, "공지")
        self.assertEqual(refined.llm_score, 0.0)
        self.assertIn("error:", str(refined.explanation.get("llm_status", "")))


if __name__ == "__main__":
    unittest.main()
