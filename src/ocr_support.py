"""OCR helpers for scanned PDF fallback."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import fitz
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

from src.text_cleaner import build_sampled_text, normalize_text


OCR_MAX_PAGES = 2
OCR_RENDER_SCALE = 2.0
OCR_MAX_IMAGE_EDGE = 1600
OCR_MIN_IMAGE_EDGE = 900
DEFAULT_OCR_MIN_CHARS = 100
_OCR_ENGINE: RapidOCR | None = None

FILENAME_HINT_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("지방세완납증명서", ("지방세완납증명서", "지방세 완납증명서", "완납증명서")),
    ("벤처기업인증서", ("벤처기업인증서", "벤처기업 인증서", "벤처기업확인서", "벤처기업 확인서")),
    ("중소기업확인서", ("중소기업확인서", "중소기업 확인서")),
    ("재무제표증명", ("표준재무제표증명", "재무제표증명", "재무제표 증명")),
    ("법인등기부등본", ("법인등기부등본", "등기부등본", "등기부")),
    ("사업자등록증", ("사업자등록증", "사업자 등록증", "사업자", "등록증")),
]


def get_ocr_engine() -> RapidOCR:
    """Create the OCR engine once per process and then reuse it."""
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE


def detect_filename_classification_hint(file_path: str | Path) -> str | None:
    """Return a strong category hint inferred from the file name."""
    normalized_name = normalize_text(Path(file_path).stem)
    for category, keywords in FILENAME_HINT_RULES:
        if any(keyword in normalized_name for keyword in keywords):
            return category
    return None


def build_filename_hint_evidence(
    file_path: str | Path,
    classification_hint: str | None = None,
) -> str:
    """Build synthetic evidence text from a strong file-name hint."""
    hint = classification_hint or detect_filename_classification_hint(file_path)
    normalized_name = normalize_text(Path(file_path).stem)
    if not hint:
        return normalized_name
    return f"파일명근거 {hint} {normalized_name}".strip()


def should_run_ocr(
    file_path: str | Path,
    extracted_text: str,
    classification_hint: str | None = None,
    min_text_length: int = DEFAULT_OCR_MIN_CHARS,
) -> bool:
    """Return True only when OCR fallback is still necessary."""
    file_ext = Path(file_path).suffix.lower()
    normalized_text = normalize_text(extracted_text)
    if file_ext != ".pdf":
        return False
    if classification_hint:
        return False
    return len(normalized_text) < max(0, min_text_length)


def explain_ocr_decision(
    file_path: str | Path,
    extracted_text: str,
    classification_hint: str | None = None,
    min_text_length: int = DEFAULT_OCR_MIN_CHARS,
) -> dict[str, Any]:
    """Describe whether OCR should run and why."""
    normalized_text = normalize_text(extracted_text)
    text_length = len(normalized_text)
    file_ext = Path(file_path).suffix.lower()
    hint = classification_hint or detect_filename_classification_hint(file_path)
    run_ocr = should_run_ocr(
        file_path=file_path,
        extracted_text=normalized_text,
        classification_hint=hint,
        min_text_length=min_text_length,
    )

    if file_ext != ".pdf":
        reason = "non_pdf"
    elif hint:
        reason = f"filename_hint:{hint}"
    elif text_length >= min_text_length:
        reason = f"text_ok:{text_length}"
    elif text_length == 0:
        reason = "text_empty"
    else:
        reason = f"text_short:{text_length}"

    return {
        "run_ocr": run_ocr,
        "reason": reason,
        "classification_hint": hint,
        "text_length": text_length,
        "hint_evidence": build_filename_hint_evidence(file_path, hint) if hint else "",
    }


def ocr_pdf_file(
    path: str | Path,
    max_pages: int = OCR_MAX_PAGES,
    total_limit: int = 4500,
    part_limit: int = 1500,
) -> dict[str, Any]:
    """OCR a scanned PDF and return sampled text plus metadata."""
    file_path = Path(path)
    if not file_path.exists():
        return {
            "ok": False,
            "file_path": str(file_path),
            "text": "",
            "pages_scanned": 0,
            "elapsed": 0.0,
            "error": f"PDF not found: {file_path}",
        }

    start = time.perf_counter()
    engine = get_ocr_engine()
    page_texts: list[str] = []
    scanned_pages = 0

    try:
        with fitz.open(file_path) as document:
            page_limit = min(len(document), max_pages)
            for page_index in range(page_limit):
                page = document.load_page(page_index)
                pixmap = page.get_pixmap(matrix=fitz.Matrix(OCR_RENDER_SCALE, OCR_RENDER_SCALE), alpha=False)
                image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
                image = resize_ocr_image(image)
                image_array = np.array(image)
                ocr_result, _ = engine(image_array)
                page_text = _flatten_ocr_result(ocr_result)
                if page_text:
                    page_texts.append(page_text)
                scanned_pages += 1
    except Exception as error:
        return {
            "ok": False,
            "file_path": str(file_path),
            "text": "",
            "pages_scanned": scanned_pages,
            "elapsed": time.perf_counter() - start,
            "error": str(error),
        }

    sampled_text = build_sampled_text(
        "\n".join(page_texts),
        total_limit=total_limit,
        part_limit=part_limit,
    )
    return {
        "ok": True,
        "file_path": str(file_path),
        "text": sampled_text,
        "pages_scanned": scanned_pages,
        "elapsed": time.perf_counter() - start,
        "error": "",
    }


def resize_ocr_image(
    image: Image.Image,
    *,
    max_edge: int = OCR_MAX_IMAGE_EDGE,
    min_edge: int = OCR_MIN_IMAGE_EDGE,
) -> Image.Image:
    """Downscale OCR input while preserving aspect ratio and readable text size."""
    width, height = image.size
    longest = max(width, height)
    shortest = min(width, height)
    if longest <= max_edge:
        return image

    scale = max_edge / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    if min(new_width, new_height) < min_edge and shortest > 0:
        min_scale = min_edge / float(shortest)
        if min_scale < 1.0:
            new_width = max(1, int(round(width * min_scale)))
            new_height = max(1, int(round(height * min_scale)))

    if (new_width, new_height) == image.size:
        return image
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def _flatten_ocr_result(ocr_result: Any) -> str:
    """Convert RapidOCR output into plain text lines."""
    if not ocr_result:
        return ""

    lines: list[str] = []
    for item in ocr_result:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        text_part = item[1]
        if isinstance(text_part, (list, tuple)) and text_part:
            text_value = str(text_part[0]).strip()
        else:
            text_value = str(text_part).strip()
        if text_value:
            lines.append(text_value)
    return "\n".join(lines)
