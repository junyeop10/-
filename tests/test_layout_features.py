from __future__ import annotations

import unittest

import numpy as np

from src.document_features import DocumentFeatureExtractor
from src.layout_features import DocumentLayoutFeatureExtractor


def _white(height: int, width: int) -> np.ndarray:
    return np.full((height, width), 255, dtype=np.uint8)


class LayoutFeatureExtractorTest(unittest.TestCase):
    def test_receipt_like_layout_scores_from_narrow_page_and_prices(self) -> None:
        image = _white(520, 160)
        for y in range(40, 460, 28):
            image[y : y + 3, 18:142] = 0
        extractor = DocumentLayoutFeatureExtractor()
        result = extractor.extract_from_images(
            [image],
            ocr_text="item 1 1,000원\nitem 2 2,000원\n합계 3,000원\ntotal 3,000",
        )

        self.assertEqual(result.features["portrait_or_landscape"], "portrait")
        self.assertGreater(result.features["narrow_width_score"], 0.4)
        self.assertGreater(result.features["receipt_pattern_score"], 0.45)
        self.assertTrue(result.features["total_keyword_exists"])
        self.assertGreater(result.features["numeric_line_density"], 0.4)
        self.assertIn("repeated_line_pattern_score", result.features)
        self.assertGreaterEqual(result.features["numeric_column_score"], 0.0)

    def test_certificate_like_layout_detects_centered_header_and_whitespace(self) -> None:
        image = _white(700, 500)
        image[70:95, 145:355] = 0
        image[220:224, 90:410] = 0
        image[440:470, 330:410] = 0
        result = DocumentLayoutFeatureExtractor().extract_from_images([image], ocr_text="증명서\n대표자\n확인")

        self.assertGreater(result.features["centered_title_score"], 0.4)
        self.assertGreater(result.features["certificate_pattern_score"], 0.35)
        self.assertGreater(result.features["whitespace_ratio"], 0.85)

    def test_slide_like_layout_scores_landscape_bullets_and_image_area(self) -> None:
        image = _white(260, 480)
        image[30:48, 40:280] = 0
        image[80:190, 300:440] = 80
        for y in (90, 120, 150):
            image[y : y + 4, 55:240] = 0
        result = DocumentLayoutFeatureExtractor().extract_from_images(
            [image],
            ocr_text="Project Update\n- goal\n- result\n- next step",
        )

        self.assertEqual(result.features["portrait_or_landscape"], "landscape")
        self.assertGreater(result.features["slide_like_layout_score"], 0.45)
        self.assertGreater(result.features["bullet_density"], 0.4)
        self.assertIn("chart_presence_score", result.features)

    def test_paper_like_layout_scores_dense_two_column_page(self) -> None:
        image = _white(700, 500)
        for y in range(60, 640, 16):
            image[y : y + 2, 40:220] = 0
            image[y : y + 2, 280:460] = 0
        result = DocumentLayoutFeatureExtractor().extract_from_images(
            [image],
            ocr_text="Abstract\nThis paper cites [1] and doi:10.1000/test.\nReferences\n[1] et al.",
        )

        self.assertGreater(result.features["two_column_score"], 0.3)
        self.assertGreater(result.features["dense_text_score"], 0.2)
        self.assertGreater(result.features["citation_pattern_density"], 0.2)
        self.assertGreater(result.features["references_last_page_score"], 0.4)

    def test_document_feature_bundle_includes_layout_features(self) -> None:
        bundle = DocumentFeatureExtractor().extract(
            file_name="receipt.txt",
            file_ext=".txt",
            text="합계 3,000원\nitem 1,000원",
        )

        self.assertIn("layout_features", bundle.to_storage_dict())
        self.assertIn("receipt_pattern_score", bundle.layout_features)


if __name__ == "__main__":
    unittest.main()
