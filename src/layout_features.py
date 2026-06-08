"""Lightweight CPU document layout feature extraction."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.text_cleaner import normalize_text


LAYOUT_EXTRACTOR_VERSION = "1.0"


@dataclass
class LayoutFeatureResult:
    version: str
    features: dict[str, Any]


class DocumentLayoutFeatureExtractor:
    """Extract image/layout features without deep learning dependencies."""

    def __init__(self, version: str = LAYOUT_EXTRACTOR_VERSION, max_pages: int = 3, render_dpi: int = 110) -> None:
        self.version = version
        self.max_pages = max_pages
        self.render_dpi = render_dpi

    def extract_from_file(self, path: str | Path, *, ocr_text: str = "") -> LayoutFeatureResult:
        file_path = Path(path)
        images = self._load_sample_images(file_path)
        if not images:
            return LayoutFeatureResult(self.version, self._empty_features(ocr_text=ocr_text))
        return self.extract_from_images(images, ocr_text=ocr_text)

    def extract_from_images(self, images: list[Any], *, ocr_text: str = "") -> LayoutFeatureResult:
        if not images:
            return LayoutFeatureResult(self.version, self._empty_features(ocr_text=ocr_text))
        page_features = [self._extract_page_features(self._to_grayscale_array(image), ocr_text=ocr_text) for image in images]
        features = self._aggregate_page_features(page_features, ocr_text=ocr_text)
        features["layout_extractor_version"] = self.version
        features["sampled_page_count"] = len(page_features)
        return LayoutFeatureResult(self.version, features)

    def _load_sample_images(self, path: Path) -> list[np.ndarray]:
        if not path.exists():
            return []
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._render_pdf_samples(path)
        if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            try:
                from PIL import Image

                return [np.asarray(Image.open(path).convert("L"))]
            except Exception:
                return []
        return []

    def _render_pdf_samples(self, path: Path) -> list[np.ndarray]:
        try:
            import fitz
        except Exception:
            return []
        images: list[np.ndarray] = []
        with fitz.open(path) as document:
            if len(document) == 0:
                return []
            page_indexes = self._sample_page_indexes(len(document))
            matrix = fitz.Matrix(self.render_dpi / 72.0, self.render_dpi / 72.0)
            for page_index in page_indexes:
                page = document.load_page(page_index)
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
                if pixmap.n >= 3:
                    gray = (array[:, :, 0] * 0.299 + array[:, :, 1] * 0.587 + array[:, :, 2] * 0.114).astype(np.uint8)
                else:
                    gray = array[:, :, 0]
                images.append(gray)
        return images

    def _sample_page_indexes(self, page_count: int) -> list[int]:
        candidates = [0, page_count // 2, page_count - 1]
        selected: list[int] = []
        for item in candidates:
            if 0 <= item < page_count and item not in selected:
                selected.append(item)
        return selected[: self.max_pages]

    def _to_grayscale_array(self, image: Any) -> np.ndarray:
        if isinstance(image, np.ndarray):
            array = image
        else:
            try:
                from PIL import Image

                array = np.asarray(image.convert("L") if isinstance(image, Image.Image) else image)
            except Exception:
                array = np.asarray(image)
        if array.ndim == 3:
            array = (array[:, :, 0] * 0.299 + array[:, :, 1] * 0.587 + array[:, :, 2] * 0.114).astype(np.uint8)
        return array.astype(np.uint8, copy=False)

    def _extract_page_features(self, gray: np.ndarray, *, ocr_text: str) -> dict[str, Any]:
        height, width = gray.shape[:2]
        dark_mask = gray <= self._threshold(gray)
        rows = dark_mask.mean(axis=1)
        cols = dark_mask.mean(axis=0)
        line_segments = self._segments(rows, threshold=0.012, min_size=max(2, height // 250))
        col_segments = self._segments(cols, threshold=0.01, min_size=max(2, width // 250))
        block_count, image_area_ratio = self._estimate_blocks(gray, dark_mask)
        centered_title_score = self._centered_title_score(dark_mask, line_segments)
        large_header_score = self._large_header_score(line_segments, height)
        two_column_score = self._two_column_score(cols)
        header_block_score = self._band_density_score(dark_mask, start_ratio=0.0, end_ratio=0.16)
        footer_pattern_score = self._band_density_score(dark_mask, start_ratio=0.84, end_ratio=1.0)
        section_divider_score = self._section_divider_score(dark_mask)
        numeric_column_score = self._numeric_column_score(cols, ocr_text=ocr_text)
        repeated_line_pattern_score = self._repeated_line_pattern_score(line_segments)
        approval_block_score = self._approval_block_score(ocr_text=ocr_text, footer_score=footer_pattern_score)
        chart_presence_score = self._chart_presence_score(image_area_ratio=image_area_ratio, block_count=block_count)
        whitespace_ratio = 1.0 - float(dark_mask.mean())
        text_density = float(dark_mask.mean())
        vertical_flow_score = self._vertical_flow_score(line_segments, height)
        multi_column_score = max(two_column_score, min(1.0, max(len(col_segments) - 1, 0) / 3.0))
        text_metrics = self._text_metrics(ocr_text)

        base = {
            "page_width": int(width),
            "page_height": int(height),
            "aspect_ratio": round(width / max(height, 1), 4),
            "portrait_or_landscape": "landscape" if width > height else "portrait",
            "grayscale_mean": round(float(gray.mean()), 4),
            "grayscale_std": round(float(gray.std()), 4),
            "text_density": round(text_density, 4),
            "whitespace_ratio": round(whitespace_ratio, 4),
            "estimated_line_count": len(line_segments),
            "avg_line_length": round(self._avg_line_length(dark_mask, line_segments), 4),
            "text_block_count": block_count,
            "paragraph_block_count": block_count,
            "vertical_flow_score": round(vertical_flow_score, 4),
            "multi_column_score": round(multi_column_score, 4),
            "centered_title_score": round(centered_title_score, 4),
            "large_header_score": round(large_header_score, 4),
            "image_area_ratio": round(image_area_ratio, 4),
            "two_column_score": round(two_column_score, 4),
            "header_block_score": round(header_block_score, 4),
            "footer_pattern_score": round(footer_pattern_score, 4),
            "section_divider_score": round(section_divider_score, 4),
            "numeric_column_score": round(numeric_column_score, 4),
            "approval_block_score": round(approval_block_score, 4),
            "repeated_line_pattern_score": round(repeated_line_pattern_score, 4),
            "chart_presence_score": round(chart_presence_score, 4),
        }
        base.update(text_metrics)
        base.update(self._document_type_scores(base))
        return base

    def _aggregate_page_features(self, page_features: list[dict[str, Any]], *, ocr_text: str) -> dict[str, Any]:
        numeric_keys = {
            key
            for item in page_features
            for key, value in item.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        aggregated: dict[str, Any] = {}
        first = page_features[0]
        last = page_features[-1]
        for key in numeric_keys:
            values = [float(item.get(key, 0.0)) for item in page_features]
            aggregated[key] = round(sum(values) / max(len(values), 1), 4)
            aggregated[f"first_page_{key}"] = first.get(key, 0.0)
            aggregated[f"last_page_{key}"] = last.get(key, 0.0)
        aggregated["page_width"] = first.get("page_width", 0)
        aggregated["page_height"] = first.get("page_height", 0)
        aggregated["aspect_ratio"] = first.get("aspect_ratio", 0)
        aggregated["portrait_or_landscape"] = first.get("portrait_or_landscape", "unknown")
        aggregated["total_keyword_exists"] = any(bool(item.get("total_keyword_exists")) for item in page_features)
        aggregated["centered_header_exists"] = any(bool(item.get("centered_header_exists")) for item in page_features)
        aggregated["references_last_page_score"] = self._references_last_page_score(ocr_text, last)
        return aggregated

    def _text_metrics(self, ocr_text: str) -> dict[str, Any]:
        text = normalize_text(ocr_text or "")
        chars = [char for char in text if not char.isspace()]
        numeric = sum(char.isdigit() for char in chars)
        currency = sum(char in "$₩￦€¥원" for char in chars)
        colon = text.count(":")
        dash = text.count("-") + text.count("_")
        lines = [line.strip() for line in re.split(r"[\r\n]+", ocr_text or "") if line.strip()]
        price_lines = [
            line for line in lines if re.search(r"([$₩￦€¥]?\s*\d{1,3}(,\d{3})+|\d+\s*원)", line)
        ]
        bullet_lines = [line for line in lines if re.match(r"^\s*([-*•]|\d+[.)])\s+", line)]
        return {
            "numeric_ratio": round(numeric / max(len(chars), 1), 4),
            "currency_symbol_ratio": round(currency / max(len(chars), 1), 4),
            "colon_ratio": round(colon / max(len(chars), 1), 4),
            "dash_ratio": round(dash / max(len(chars), 1), 4),
            "repeated_price_pattern_score": round(min(1.0, len(price_lines) / 5.0), 4),
            "total_keyword_exists": any(word in text for word in ("total", "합계", "총액", "결제", "금액")),
            "numeric_line_density": round(len(price_lines) / max(len(lines), 1), 4),
            "bullet_density": round(len(bullet_lines) / max(len(lines), 1), 4),
            "citation_pattern_density": round(
                len(re.findall(r"(\[[0-9]{1,3}\]|\bet al\.|\bdoi\b)", text, flags=re.IGNORECASE))
                / max(len(lines), 1),
                4,
            ),
        }

    def _document_type_scores(self, features: dict[str, Any]) -> dict[str, Any]:
        aspect_ratio = float(features.get("aspect_ratio", 0.0))
        text_density = float(features.get("text_density", 0.0))
        whitespace_ratio = float(features.get("whitespace_ratio", 0.0))
        numeric_line_density = float(features.get("numeric_line_density", 0.0))
        repeated_price = float(features.get("repeated_price_pattern_score", 0.0))
        narrow_width = 1.0 if aspect_ratio < 0.58 else max(0.0, 0.75 - aspect_ratio)
        slide_like = 0.0
        if features.get("portrait_or_landscape") == "landscape":
            slide_like += 0.45
        slide_like += min(0.35, float(features.get("image_area_ratio", 0.0)))
        slide_like += min(0.2, float(features.get("bullet_density", 0.0)) * 2)
        receipt_score = min(
            1.0,
            narrow_width * 0.35
            + numeric_line_density * 0.3
            + repeated_price * 0.25
            + (0.1 if features.get("total_keyword_exists") else 0.0),
        )
        certificate_score = min(
            1.0,
            float(features.get("centered_title_score", 0.0)) * 0.35
            + whitespace_ratio * 0.25
            + float(features.get("large_header_score", 0.0)) * 0.25
            + float(features.get("seal_or_signature_area_score", 0.0)) * 0.15,
        )
        dense_text = min(1.0, text_density * 4.0 + float(features.get("two_column_score", 0.0)) * 0.25)
        return {
            "narrow_width_score": round(narrow_width, 4),
            "receipt_pattern_score": round(receipt_score, 4),
            "certificate_pattern_score": round(certificate_score, 4),
            "centered_header_exists": float(features.get("centered_title_score", 0.0)) > 0.45,
            "seal_or_signature_area_score": round(self._seal_area_score(features), 4),
            "signature_area_score": round(self._signature_area_score(features), 4),
            "whitespace_balance_score": round(min(1.0, 1.0 - abs(0.65 - whitespace_ratio)), 4),
            "slide_like_layout_score": round(min(1.0, slide_like), 4),
            "title_text_ratio": round(min(1.0, float(features.get("large_header_score", 0.0))), 4),
            "dense_text_score": round(dense_text, 4),
        }

    def _empty_features(self, *, ocr_text: str) -> dict[str, Any]:
        metrics = self._text_metrics(ocr_text)
        return {
            "page_width": 0,
            "page_height": 0,
            "aspect_ratio": 0.0,
            "portrait_or_landscape": "unknown",
            "grayscale_mean": 0.0,
            "grayscale_std": 0.0,
            "text_density": 0.0,
            "whitespace_ratio": 0.0,
            "estimated_line_count": 0,
            "avg_line_length": 0.0,
            "text_block_count": 0,
            "paragraph_block_count": 0,
            "vertical_flow_score": 0.0,
            "multi_column_score": 0.0,
            "centered_title_score": 0.0,
            "large_header_score": 0.0,
            "header_block_score": 0.0,
            "footer_pattern_score": 0.0,
            "signature_area_score": 0.0,
            "chart_presence_score": 0.0,
            "section_divider_score": 0.0,
            "numeric_column_score": 0.0,
            "approval_block_score": 0.0,
            "repeated_line_pattern_score": 0.0,
            "layout_extractor_version": self.version,
            "sampled_page_count": 0,
            **metrics,
            **self._document_type_scores(metrics),
        }

    def _threshold(self, gray: np.ndarray) -> float:
        try:
            import cv2

            value, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return 128.0 if value <= 0 else float(value)
        except Exception:
            return float(max(80, min(220, gray.mean() - (gray.std() * 0.25))))

    def _estimate_blocks(self, gray: np.ndarray, dark_mask: np.ndarray) -> tuple[int, float]:
        try:
            import cv2

            binary = (dark_mask.astype(np.uint8) * 255)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, gray.shape[1] // 80), max(3, gray.shape[0] // 180)))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            areas = [cv2.contourArea(contour) for contour in contours]
            min_area = max(20.0, gray.size * 0.00008)
            text_blocks = [area for area in areas if area >= min_area]
            large_non_text = [area for area in areas if area >= gray.size * 0.04]
            image_area_ratio = sum(large_non_text) / max(float(gray.size), 1.0)
            return len(text_blocks), min(1.0, image_area_ratio)
        except Exception:
            rows = dark_mask.mean(axis=1)
            segments = self._segments(rows, threshold=0.012, min_size=max(2, gray.shape[0] // 250))
            return max(1, len(segments) // 4) if segments else 0, 0.0

    def _segments(self, values: np.ndarray, *, threshold: float, min_size: int) -> list[tuple[int, int]]:
        active = values > threshold
        segments: list[tuple[int, int]] = []
        start: int | None = None
        for index, is_active in enumerate(active):
            if is_active and start is None:
                start = index
            elif not is_active and start is not None:
                if index - start >= min_size:
                    segments.append((start, index))
                start = None
        if start is not None and len(active) - start >= min_size:
            segments.append((start, len(active)))
        return segments

    def _avg_line_length(self, dark_mask: np.ndarray, line_segments: list[tuple[int, int]]) -> float:
        if not line_segments:
            return 0.0
        lengths = []
        for start, end in line_segments:
            cols = np.where(dark_mask[start:end, :].any(axis=0))[0]
            if cols.size:
                lengths.append((cols[-1] - cols[0] + 1) / max(dark_mask.shape[1], 1))
        return sum(lengths) / max(len(lengths), 1)

    def _centered_title_score(self, dark_mask: np.ndarray, line_segments: list[tuple[int, int]]) -> float:
        height, width = dark_mask.shape[:2]
        candidates = [segment for segment in line_segments if segment[0] < height * 0.25]
        if not candidates:
            return 0.0
        best = 0.0
        for start, end in candidates[:5]:
            cols = np.where(dark_mask[start:end, :].any(axis=0))[0]
            if cols.size == 0:
                continue
            center = (cols[0] + cols[-1]) / 2.0
            centered = 1.0 - min(1.0, abs(center - (width / 2.0)) / (width / 2.0))
            relative_width = (cols[-1] - cols[0] + 1) / width
            best = max(best, centered * (1.0 - abs(0.45 - relative_width)))
        return max(0.0, min(1.0, best))

    def _large_header_score(self, line_segments: list[tuple[int, int]], height: int) -> float:
        if not line_segments:
            return 0.0
        top_segments = [end - start for start, end in line_segments if start < height * 0.25]
        if not top_segments:
            return 0.0
        median = sorted(end - start for start, end in line_segments)[len(line_segments) // 2]
        return min(1.0, max(top_segments) / max(median * 2.5, 1))

    def _two_column_score(self, col_density: np.ndarray) -> float:
        width = len(col_density)
        if width < 10:
            return 0.0
        middle = col_density[int(width * 0.43) : int(width * 0.57)]
        left = col_density[int(width * 0.08) : int(width * 0.42)]
        right = col_density[int(width * 0.58) : int(width * 0.92)]
        if left.size == 0 or right.size == 0 or middle.size == 0:
            return 0.0
        side_density = (float(left.mean()) + float(right.mean())) / 2.0
        gutter = float(middle.mean())
        if side_density <= 0:
            return 0.0
        return max(0.0, min(1.0, (side_density - gutter) / side_density))

    def _vertical_flow_score(self, line_segments: list[tuple[int, int]], height: int) -> float:
        if len(line_segments) < 2:
            return 0.0
        centers = [(start + end) / 2.0 for start, end in line_segments]
        gaps = [centers[index + 1] - centers[index] for index in range(len(centers) - 1)]
        if not gaps:
            return 0.0
        avg_gap = sum(gaps) / len(gaps)
        variance = sum((gap - avg_gap) ** 2 for gap in gaps) / len(gaps)
        regularity = 1.0 / (1.0 + math.sqrt(variance) / max(avg_gap, 1.0))
        coverage = min(1.0, (centers[-1] - centers[0]) / max(height, 1))
        return regularity * coverage

    def _band_density_score(self, dark_mask: np.ndarray, *, start_ratio: float, end_ratio: float) -> float:
        height = dark_mask.shape[0]
        start = max(0, min(height, int(height * start_ratio)))
        end = max(start + 1, min(height, int(height * end_ratio)))
        band_density = float(dark_mask[start:end, :].mean())
        whole_density = float(dark_mask.mean())
        if whole_density <= 0:
            return 0.0
        return max(0.0, min(1.0, band_density / max(whole_density * 1.8, 0.0001)))

    def _section_divider_score(self, dark_mask: np.ndarray) -> float:
        height, width = dark_mask.shape[:2]
        rows = dark_mask.mean(axis=1)
        long_rules = 0
        for index, density in enumerate(rows):
            if density < 0.015:
                continue
            cols = np.where(dark_mask[index : index + 1, :].any(axis=0))[0]
            if cols.size and (cols[-1] - cols[0] + 1) / max(width, 1) > 0.55:
                long_rules += 1
        return min(1.0, long_rules / max(height * 0.015, 1.0))

    def _numeric_column_score(self, col_density: np.ndarray, *, ocr_text: str) -> float:
        lines = [line.strip() for line in re.split(r"[\r\n]+", ocr_text or "") if line.strip()]
        numeric_lines = sum(1 for line in lines if len(re.findall(r"\d", line)) >= 3)
        numeric_text_score = min(1.0, numeric_lines / max(len(lines), 1))
        if col_density.size == 0:
            return numeric_text_score
        high_density_cols = float((col_density > max(float(col_density.mean()) * 1.4, 0.01)).mean())
        return min(1.0, numeric_text_score * 0.65 + high_density_cols * 0.35)

    def _approval_block_score(self, *, ocr_text: str, footer_score: float) -> float:
        text = normalize_text(ocr_text or "")
        approval_hits = sum(1 for word in ("approval", "approved", "signature", "vendor", "buyer", "승인", "결재", "서명") if word in text)
        return min(1.0, footer_score * 0.35 + min(1.0, approval_hits / 3.0) * 0.65)

    def _repeated_line_pattern_score(self, line_segments: list[tuple[int, int]]) -> float:
        if len(line_segments) < 4:
            return 0.0
        centers = [(start + end) / 2.0 for start, end in line_segments]
        gaps = [centers[index + 1] - centers[index] for index in range(len(centers) - 1)]
        if not gaps:
            return 0.0
        avg_gap = sum(gaps) / len(gaps)
        variance = sum((gap - avg_gap) ** 2 for gap in gaps) / len(gaps)
        regularity = 1.0 / (1.0 + math.sqrt(variance) / max(avg_gap, 1.0))
        return min(1.0, regularity * min(1.0, len(line_segments) / 18.0))

    def _chart_presence_score(self, *, image_area_ratio: float, block_count: int) -> float:
        if image_area_ratio <= 0:
            return 0.0
        block_factor = 1.0 if block_count >= 3 else 0.55
        return min(1.0, image_area_ratio * 2.0 * block_factor)

    def _seal_area_score(self, features: dict[str, Any]) -> float:
        whitespace = float(features.get("whitespace_ratio", 0.0))
        centered = float(features.get("centered_title_score", 0.0))
        return min(1.0, whitespace * 0.5 + centered * 0.3)

    def _signature_area_score(self, features: dict[str, Any]) -> float:
        footer = float(features.get("footer_pattern_score", 0.0))
        approval = float(features.get("approval_block_score", 0.0))
        whitespace = float(features.get("whitespace_ratio", 0.0))
        return min(1.0, footer * 0.35 + approval * 0.45 + whitespace * 0.2)

    def _references_last_page_score(self, ocr_text: str, last_page_features: dict[str, Any]) -> float:
        text = normalize_text(ocr_text or "")
        references_hint = 1.0 if any(word in text for word in ("references", "참고문헌", "bibliography")) else 0.0
        dense = float(last_page_features.get("dense_text_score", 0.0))
        citation = float(last_page_features.get("citation_pattern_density", 0.0))
        return round(min(1.0, references_hint * 0.55 + dense * 0.25 + citation * 0.2), 4)
