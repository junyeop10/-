from __future__ import annotations

import unittest

from src.classifier import HybridClassifier
from src.document_features import DocumentFeatureBundle
from src.lexical_features import compute_lexical_scores, flatten_lexical_scores


class LexicalFeaturesTest(unittest.TestCase):
    def test_empty_text_returns_zero_scores(self) -> None:
        profiles = {"계약서": {"profile_text": "계약 금액 지급 기한"}}

        result = compute_lexical_scores("", profiles)

        self.assertEqual(result["계약서"]["lexical_score"], 0.0)

    def test_contract_document_scores_highest_against_contract_profile(self) -> None:
        profiles = {
            "계약서": {"profile_text": "계약 금액 지급 기한 계약기간 손해배상"},
            "영수증": {"profile_text": "영수증 카드 승인번호 합계금액 부가세"},
        }

        result = compute_lexical_scores("본 계약의 계약 금액과 지급 기한 및 손해배상을 정한다.", profiles)
        scores = flatten_lexical_scores(result, list(profiles))

        self.assertGreater(scores["계약서"], scores["영수증"])
        self.assertGreater(scores["계약서"], 0.2)

    def test_receipt_document_scores_highest_against_receipt_profile(self) -> None:
        profiles = {
            "계약서": {"profile_text": "계약 금액 지급 기한 계약기간"},
            "영수증": {"profile_text": "영수증 카드 승인번호 합계금액 부가세 공급가액"},
        }

        result = compute_lexical_scores("카드 승인번호와 합계금액, 부가세가 표시된 영수증입니다.", profiles)
        scores = flatten_lexical_scores(result, list(profiles))

        self.assertGreater(scores["영수증"], scores["계약서"])
        self.assertGreater(scores["영수증"], 0.15)

    def test_word_ngram_keeps_contract_expressions(self) -> None:
        profiles = {"계약서": {"profile_text": "계약 금액 지급 기한"}}

        result = compute_lexical_scores("계약 금액 및 지급 기한을 확인한다.", profiles)

        self.assertGreater(result["계약서"]["tfidf_score"], 0.0)
        self.assertIn("계약", result["계약서"]["top_terms"])
        self.assertIn("지급", result["계약서"]["top_terms"])

    def test_generic_filename_is_excluded(self) -> None:
        classifier = HybridClassifier.__new__(HybridClassifier)

        self.assertTrue(classifier._is_generic_filename("scan001.pdf"))
        self.assertTrue(classifier._is_generic_filename("document.pdf"))
        self.assertTrue(classifier._is_generic_filename("KakaoTalk_20260526.jpg"))
        self.assertFalse(classifier._is_generic_filename("계약서_용역_2026.pdf"))

    def test_score_weights_are_normalized(self) -> None:
        classifier = HybridClassifier.__new__(HybridClassifier)
        classifier.rule_skip_embedding_threshold = 0.85
        classifier.min_rule_matches_for_skip = 3
        bundle = DocumentFeatureBundle(
            feature_version="test",
            filename_features={},
            metadata_features={},
            structural_features={},
            layout_features={},
            text_stats={"char_count": 1200, "low_quality_scan_score": 0.0, "unreadable_ratio": 0.0},
            compressed_text="계약 금액 지급 기한",
            compressed_text_hash="hash",
        )

        weights = classifier._build_score_weights(
            file_name="scan001.pdf",
            feature_bundle=bundle,
            top_rule_score=0.1,
            top_rule_match_count=0,
            feedback_scores={"계약서": 0.0},
        )

        self.assertAlmostEqual(sum(weights.values()), 1.0, places=3)
        self.assertEqual(weights["filename"], 0.0)


if __name__ == "__main__":
    unittest.main()
