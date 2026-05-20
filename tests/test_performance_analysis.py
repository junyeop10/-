from __future__ import annotations

import unittest

from src.classifier import ClassificationResult
from src.performance import build_file_latency_analysis, summarize_payload_profiles


class PerformanceAnalysisTest(unittest.TestCase):
    def test_latency_analysis_marks_ocr_as_bottleneck(self) -> None:
        analysis = build_file_latency_analysis(
            {"read_extract": 0.2, "ocr": 2.4, "classification": 0.3, "total": 2.9},
            text_length=120,
            file_size=9_000_000,
            ocr_used=True,
            ocr_pages=3,
            embedding_used=False,
            strong_rule_match=False,
            review_required=True,
            matched_rules_count=0,
        )
        self.assertEqual(analysis["dominant_stage"], "ocr")
        self.assertEqual(analysis["speed_band"], "slow")
        self.assertTrue(any("OCR" in reason for reason in analysis["reasons"]))

    def test_summary_collects_startup_and_slowest_files(self) -> None:
        result = ClassificationResult(
            predicted_category="report",
            confidence=0.9,
            final_score=0.9,
            rule_score=0.8,
            embedding_score=0.1,
            llm_score=0.0,
            feedback_score=0.0,
            duplicate_score=0.0,
            similarity_score=0.1,
            embedding_used=True,
            review_required=False,
            matched_rules=["report"],
            candidate_scores={"report": 0.9},
            reasoning="test",
            query_embedding=[],
        )
        payloads = [
            {
                "file_name": "a.pdf",
                "result": result,
                "performance": {
                    "stage_timings": {"read_extract": 0.3, "classification": 0.5, "total": 0.8},
                    "analysis": {"total_time": 0.8, "dominant_stage": "classification", "summary": "slow-ish", "reasons": []},
                },
            },
            {
                "file_name": "b.pdf",
                "result": result,
                "performance": {
                    "stage_timings": {"read_extract": 0.1, "classification": 0.2, "total": 0.3},
                    "analysis": {"total_time": 0.3, "dominant_stage": "classification", "summary": "fast", "reasons": []},
                },
            },
        ]
        summary = summarize_payload_profiles(
            payloads,
            startup_profile={"stages": {"config_load": 0.1, "db_init": 0.2}, "startup_ready_total": 0.5},
            run_profile={"elapsed": 1.2},
        )
        self.assertEqual(summary["classified_files"], 2)
        self.assertAlmostEqual(summary["startup_total"], 0.5)
        self.assertAlmostEqual(summary["run_elapsed"], 1.2)
        self.assertEqual(summary["slowest_files"][0]["file_name"], "a.pdf")


if __name__ == "__main__":
    unittest.main()
