from __future__ import annotations

import unittest
from unittest.mock import patch

from src.cli import apply_ocr_plan, apply_ocr_reasoning, merge_ocr_result
from src.classifier import ClassificationResult
from src.ocr_support import (
    build_filename_hint_evidence,
    detect_filename_classification_hint,
    get_ocr_engine,
    should_run_ocr,
)


class OCRFallbackTest(unittest.TestCase):
    def test_should_skip_ocr_for_strong_filename_hint(self) -> None:
        hint = detect_filename_classification_hint("사업자등록증_테스트.pdf")
        self.assertEqual(hint, "사업자등록증")
        self.assertFalse(should_run_ocr("사업자등록증_테스트.pdf", "", classification_hint=hint))

    def test_should_skip_ocr_for_certificate_style_titles(self) -> None:
        self.assertEqual(
            detect_filename_classification_hint("8. 벤처기업인증서_(주)커넥트스토리.pdf"),
            "벤처기업인증서",
        )
        self.assertEqual(
            detect_filename_classification_hint("지방세완납증명서(6.10).pdf"),
            "지방세완납증명서",
        )
        self.assertEqual(
            detect_filename_classification_hint("중소기업확인서.pdf"),
            "중소기업확인서",
        )
        self.assertEqual(
            detect_filename_classification_hint("표준재무제표증명 2024년도.pdf"),
            "재무제표증명",
        )

    def test_should_skip_ocr_when_text_is_long_enough(self) -> None:
        extracted_text = "가" * 100
        self.assertFalse(should_run_ocr("sample.pdf", extracted_text, min_text_length=100))
        self.assertTrue(should_run_ocr("sample.pdf", "짧은 텍스트", min_text_length=100))

    def test_apply_ocr_plan_keeps_filename_hint_as_evidence(self) -> None:
        record = {
            "file_path": "사업자등록증_테스트.pdf",
            "file_name": "사업자등록증_테스트.pdf",
            "file_ext": ".pdf",
            "evidence_text": "",
        }

        apply_ocr_plan(record, ocr_min_chars=100)

        self.assertFalse(record["pending_ocr"])
        self.assertEqual(record["ocr_status"], "skipped")
        self.assertIn("filename_hint:사업자등록증", record["ocr_reason"])
        self.assertIn("파일명근거 사업자등록증", record["evidence_text"])

    def test_merge_ocr_result_marks_record_and_refreshes_scores(self) -> None:
        record = {
            "file_path": "사업자등록증_테스트.pdf",
            "file_name": "사업자등록증_테스트.pdf",
            "file_ext": ".pdf",
            "evidence_text": "",
            "rule_breakdown": {"scores": {}, "matches": {}},
            "ocr_used": False,
            "ocr_pages": 0,
            "ocr_error": "",
            "ocr_status": "queued",
            "ocr_reason": "text_empty",
            "pending_ocr": True,
            "timings": {"ocr_time": 0.0, "rule_time": 0.0, "worker_time": 0.0},
        }
        rules = [
            {
                "category": "사업자등록증",
                "rule_type": "keyword",
                "pattern": "사업자등록증",
                "weight": 1.0,
            }
        ]

        merge_ocr_result(
            record,
            {
                "ok": True,
                "text": "사업자등록증 등록번호 개업연월일",
                "pages_scanned": 1,
                "elapsed": 0.25,
                "error": "",
            },
            rules,
        )

        self.assertTrue(record["ocr_used"])
        self.assertEqual(record["ocr_pages"], 1)
        self.assertEqual(record["ocr_status"], "used")
        self.assertFalse(record["pending_ocr"])
        self.assertGreater(record["rule_breakdown"]["scores"]["사업자등록증"], 0.0)

    def test_apply_ocr_reasoning_appends_visible_marker(self) -> None:
        result = ClassificationResult(
            predicted_category="사업자등록증",
            confidence=0.9,
            final_score=0.9,
            rule_score=0.9,
            embedding_score=0.0,
            llm_score=0.0,
            feedback_score=0.0,
            duplicate_score=0.0,
            similarity_score=0.0,
            embedding_used=False,
            review_required=False,
            matched_rules=["사업자등록증"],
            candidate_scores={"사업자등록증": 0.9},
            reasoning="recommend=사업자등록증 | rules=사업자등록증",
            query_embedding=[],
        )

        updated = apply_ocr_reasoning(result, ocr_used=True, ocr_pages=3)
        self.assertIn("ocr=used(pages=3)", updated.reasoning)
        self.assertEqual(updated.predicted_category, result.predicted_category)

    def test_ocr_engine_is_reused_within_process(self) -> None:
        class FakeRapidOCR:
            def __init__(self) -> None:
                self.marker = object()

        with patch("src.ocr_support.RapidOCR", FakeRapidOCR), patch("src.ocr_support._OCR_ENGINE", None):
            first = get_ocr_engine()
            second = get_ocr_engine()

        self.assertIs(first, second)

    def test_build_filename_hint_evidence_contains_normalized_name(self) -> None:
        evidence = build_filename_hint_evidence("14. 법인등기부등본_(주)커넥트스토리.pdf")
        self.assertIn("파일명근거 법인등기부등본", evidence)


if __name__ == "__main__":
    unittest.main()
