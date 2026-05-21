from __future__ import annotations

import unittest

from src.classifier import ClassificationResult
from src.gui import (
    build_debug_detail,
    build_user_rationale_summary,
    can_drag_tree_meta,
    drop_target_category_from_meta,
    group_payloads_by_category,
    summarize_processing_methods,
    upsert_payload_by_file_path,
)


class GuiGroupingTest(unittest.TestCase):
    def _payload(
        self,
        file_name: str,
        category: str,
        *,
        embedding_used: bool = False,
        llm_used: bool = False,
    ) -> dict[str, object]:
        return {
            "file_name": file_name,
            "result": ClassificationResult(
                predicted_category=category,
                confidence=0.9,
                final_score=0.9,
                rule_score=0.9,
                embedding_score=0.5 if embedding_used else 0.0,
                llm_score=0.8 if llm_used else 0.0,
                feedback_score=0.0,
                duplicate_score=0.0,
                similarity_score=0.5 if embedding_used else 0.0,
                embedding_used=embedding_used,
                review_required=False,
                matched_rules=[],
                candidate_scores={category: 0.9},
                reasoning="test",
                query_embedding=[],
                llm_used=llm_used,
            ),
        }

    def test_group_payloads_by_category_builds_folder_like_groups(self) -> None:
        payloads = [
            self._payload("a.txt", "보고서"),
            self._payload("b.txt", "계약서"),
            self._payload("c.txt", "보고서"),
        ]
        grouped = group_payloads_by_category(payloads)
        self.assertEqual(list(grouped), ["계약서", "보고서"])
        self.assertEqual(len(grouped["보고서"]), 2)
        self.assertEqual(len(grouped["계약서"]), 1)

    def test_group_payloads_by_category_applies_query_and_filter(self) -> None:
        payloads = [
            self._payload("plan_a.txt", "사업계획서"),
            self._payload("plan_b.txt", "사업계획서"),
            self._payload("report.txt", "보고서"),
        ]
        grouped = group_payloads_by_category(payloads, query="plan", category_filter="사업계획서")
        self.assertEqual(list(grouped), ["사업계획서"])
        self.assertEqual(len(grouped["사업계획서"]), 2)

    def test_drag_helpers_identify_file_drag_and_category_drop(self) -> None:
        file_payload = self._payload("report.txt", "보고서")
        file_meta = {"kind": "file", "payload": file_payload}
        category_meta = {"kind": "category", "category": "계약서", "payloads": []}

        self.assertTrue(can_drag_tree_meta(file_meta))
        self.assertFalse(can_drag_tree_meta(category_meta))
        self.assertEqual(drop_target_category_from_meta(category_meta), "계약서")
        self.assertEqual(drop_target_category_from_meta(file_meta), "보고서")

    def test_summarize_processing_methods_counts_rule_embedding_and_llm(self) -> None:
        payloads = [
            self._payload("rule.txt", "보고서"),
            self._payload("embedding.txt", "보고서", embedding_used=True),
            self._payload("llm.txt", "보고서", embedding_used=True, llm_used=True),
        ]
        counts = summarize_processing_methods(payloads)
        self.assertEqual(counts, {"rule": 1, "embedding": 1, "llm": 1})

    def test_upsert_payload_by_file_path_replaces_existing_file_entry(self) -> None:
        original = self._payload("report.txt", "보고서")
        original["file_path"] = "C:/tmp/report.txt"
        updated = self._payload("report.txt", "계약서")
        updated["file_path"] = "C:/tmp/report.txt"
        payloads = upsert_payload_by_file_path([original], updated)
        self.assertEqual(len(payloads), 1)
        result = payloads[0]["result"]
        assert isinstance(result, ClassificationResult)
        self.assertEqual(result.predicted_category, "계약서")


    def test_user_rationale_summary_hides_raw_debug_json(self) -> None:
        payload = self._payload("paper.pdf", "논문")
        result = payload["result"]
        assert isinstance(result, ClassificationResult)
        result.review_required = True
        result.review_reasons = ["rule_ml_conflict", "small_margin"]
        result.predicted_type = "논문"
        result.type_confidence = 0.72
        result.suggested_tags = [{"tag": "AI", "confidence": 0.75, "source": "feature_rule"}]

        summary = build_user_rationale_summary(result, payload)
        debug = build_debug_detail(result, payload, {"stage_timings": {}, "analysis": {}})

        self.assertIn("이 문서는 '논문' 유형으로 판단했습니다.", summary)
        self.assertIn("규칙 판단과 ML 판단이 다름", summary)
        self.assertIn("더보기", summary)
        self.assertNotIn("candidate_scores", summary)
        self.assertIn("후보 점수", debug)
        self.assertIn("ml_evidence", debug)


if __name__ == "__main__":
    unittest.main()
