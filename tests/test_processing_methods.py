from __future__ import annotations

import unittest

from src.classifier import (
    ClassificationResult,
    get_primary_processing_method,
    get_processing_method_label,
    get_processing_trace_text,
)


def make_result(
    *,
    embedding_used: bool = False,
    llm_used: bool = False,
    ocr_used: bool = False,
) -> ClassificationResult:
    return ClassificationResult(
        predicted_category="보고서",
        confidence=0.9,
        final_score=0.9,
        rule_score=0.9,
        embedding_score=0.4 if embedding_used else 0.0,
        llm_score=0.8 if llm_used else 0.0,
        feedback_score=0.0,
        duplicate_score=0.0,
        similarity_score=0.4 if embedding_used else 0.0,
        embedding_used=embedding_used,
        review_required=False,
        matched_rules=[],
        candidate_scores={"보고서": 0.9},
        reasoning="test",
        query_embedding=[],
        llm_used=llm_used,
        ocr_used=ocr_used,
    )


class ProcessingMethodTest(unittest.TestCase):
    def test_rule_processing_label(self) -> None:
        result = make_result()
        self.assertEqual(get_primary_processing_method(result), "rule")
        self.assertEqual(get_processing_method_label(result), "룰 기반")
        self.assertEqual(get_processing_trace_text(result), "룰 기반")

    def test_embedding_processing_label(self) -> None:
        result = make_result(embedding_used=True)
        self.assertEqual(get_primary_processing_method(result), "embedding")
        self.assertEqual(get_processing_method_label(result), "임베딩 보조판단")
        self.assertEqual(get_processing_trace_text(result), "임베딩 보조판단")

    def test_llm_processing_trace_includes_embedding_and_ocr(self) -> None:
        result = make_result(embedding_used=True, llm_used=True, ocr_used=True)
        self.assertEqual(get_primary_processing_method(result), "llm")
        self.assertEqual(
            get_processing_trace_text(result),
            "LLM 보조판단 -> OCR 텍스트보강 -> 임베딩 선판단",
        )


if __name__ == "__main__":
    unittest.main()
