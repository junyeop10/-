from __future__ import annotations

import unittest

from src.document_patterns import build_evidence_groups, get_default_document_patterns, get_pattern_for_type


class DocumentPatternLibraryTest(unittest.TestCase):
    def test_default_patterns_cover_enterprise_types(self) -> None:
        patterns = get_default_document_patterns()
        labels = {pattern["type"] for pattern in patterns}

        self.assertGreaterEqual(len(patterns), 8)
        self.assertIn("계약서", labels)
        self.assertIn("영수증", labels)
        self.assertIn("구매발주서", labels)
        self.assertIn("일반 보고서", labels)
        for pattern in patterns:
            signals = pattern["profile_signals"]
            self.assertTrue(signals["semantic_signals"])
            self.assertTrue(signals["layout_signals"])
            self.assertTrue(signals["structural_signals"])
            self.assertTrue(signals["ocr_signals"])
            self.assertTrue(signals["numeric_patterns"])

    def test_alias_lookup_resolves_to_canonical_type(self) -> None:
        pattern = get_pattern_for_type("Receipt")

        self.assertIsNotNone(pattern)
        assert pattern is not None
        self.assertEqual(pattern["type"], "영수증")

    def test_evidence_groups_are_separated(self) -> None:
        groups = build_evidence_groups(
            predicted_type="계약서",
            text="제1조 갑 을 비밀유지 계약기간 서명",
            structural_features={"clause_pattern_score": 0.7, "legal_term_density": 0.4},
            layout_features={"dense_text_score": 0.8, "signature_area_score": 0.5},
            text_stats={},
        )

        self.assertTrue(groups["semantic"])
        self.assertTrue(groups["layout"])
        self.assertTrue(groups["structure"])
        self.assertTrue(groups["ocr"])


if __name__ == "__main__":
    unittest.main()
