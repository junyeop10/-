from __future__ import annotations

import unittest

from src.type_classifier import TypeClassifier


class TypeClassifierTest(unittest.TestCase):
    def test_unavailable_when_training_data_is_too_small(self) -> None:
        classifier = TypeClassifier(min_examples=4)
        prediction = classifier.predict(
            training_rows=[{"label": "논문", "file_name": "paper.pdf", "body_text": "abstract references"}],
            file_name="new.pdf",
            body_text="abstract references",
            structural_features={},
            fallback_type="기타",
        )
        self.assertFalse(prediction.available)
        self.assertEqual(prediction.predicted_type, "기타")

    def test_filename_signal_contributes_to_prediction_when_sklearn_available(self) -> None:
        rows = [
            {"label": "논문", "file_name": "ai_review_paper_1.pdf", "body_text": "abstract references doi"},
            {"label": "논문", "file_name": "medical_review_paper_2.pdf", "body_text": "abstract references et al"},
            {"label": "계약서", "file_name": "근로계약서_a.pdf", "body_text": "계약 갑 을 계약기간"},
            {"label": "계약서", "file_name": "용역계약서_b.pdf", "body_text": "계약 갑 을 용역"},
        ]
        classifier = TypeClassifier(min_examples=4)
        prediction = classifier.predict(
            training_rows=rows,
            file_name="Transformer_MRI_Review_Paper.pdf",
            body_text="abstract references doi image segmentation",
            structural_features={"has_abstract": True, "has_references": True, "citation_count": 3},
            fallback_type="기타",
        )
        if prediction.evidence.get("status") == "unavailable":
            self.skipTest(str(prediction.evidence.get("reason")))
        self.assertTrue(prediction.available)
        self.assertEqual(prediction.predicted_type, "논문")
        self.assertGreater(prediction.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
