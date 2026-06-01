"""OCR helpers for scanned PDF fallback."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import fitz
import numpy as np
from PIL import Image
import easyocr

from src.text_cleaner import build_sampled_text, normalize_text


# ---------------------------------------------------------------------------
# Korean OCR settings
# ---------------------------------------------------------------------------

OCR_MAX_PAGES = 2
OCR_RENDER_SCALE = 2.0
DEFAULT_OCR_MIN_CHARS = 100
OCR_EARLY_STOP_CHARS = 1000

# EasyOCR Korean + English
# gpu=False: 일반 노트북/CPU 기준 안정 실행
# gpu=True: NVIDIA GPU/CUDA 환경이 제대로 있을 때만 사용
OCR_LANGUAGES = ["ko", "en"]
OCR_GPU = False

_OCR_ENGINE: easyocr.Reader | None = None


FILENAME_HINT_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("지방세완납증명서", ("지방세완납증명서", "지방세 완납증명서", "완납증명서")),
    ("벤처기업인증서", ("벤처기업인증서", "벤처기업 인증서", "벤처기업확인서", "벤처기업 확인서")),
    ("중소기업확인서", ("중소기업확인서", "중소기업 확인서")),
    ("재무제표증명", ("표준재무제표증명", "재무제표증명", "재무제표 증명")),
    ("법인등기부등본", ("법인등기부등본", "등기부등본", "등기부")),
    ("사업자등록증", ("사업자등록증", "사업자 등록증", "사업자", "등록증")),
]


NORMALIZED_FILENAME_HINT_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        category,
        tuple(
            normalized_keyword
            for keyword in keywords
            if (normalized_keyword := normalize_text(keyword))
        ),
    )
    for category, keywords in FILENAME_HINT_RULES
]


def get_ocr_engine() -> easyocr.Reader:
    """
    Create the EasyOCR Korean engine once per process and reuse it.

    첫 실행 때 모델 다운로드/로딩 때문에 시간이 오래 걸릴 수 있다.
    이후 같은 프로세스 안에서는 재사용된다.
    """

    global _OCR_ENGINE

    if _OCR_ENGINE is None:
        _OCR_ENGINE = easyocr.Reader(OCR_LANGUAGES, gpu=OCR_GPU)

    return _OCR_ENGINE


def detect_filename_classification_hint(file_path: str | Path) -> str | None:
    """
    Return a strong category hint inferred from the file name.
    """

    normalized_name = normalize_text(Path(file_path).stem)

    if not normalized_name:
        return None

    for category, keywords in NORMALIZED_FILENAME_HINT_RULES:
        if any(keyword in normalized_name for keyword in keywords):
            return category

    return None


def build_filename_hint_evidence(
    file_path: str | Path,
    classification_hint: str | None = None,
) -> str:
    """
    Build synthetic evidence text from a strong file-name hint.
    """

    hint = classification_hint or detect_filename_classification_hint(file_path)
    normalized_name = normalize_text(Path(file_path).stem)

    if not hint:
        return normalized_name

    return f"파일명근거 {hint} {normalized_name}".strip()


def should_run_ocr(
    file_path: str | Path,
    normalized_text: str,
    classification_hint: str | None = None,
    extraction_failed: bool = False,
    min_text_length: int = DEFAULT_OCR_MIN_CHARS,
    skip_ocr_if_filename_hint: bool = True,
) -> bool:
    """
    Return True only when OCR fallback is necessary.
    """

    file_ext = Path(file_path).suffix.lower()

    if file_ext != ".pdf":
        return False

    if skip_ocr_if_filename_hint and classification_hint:
        return False

    if extraction_failed:
        return True

    return len(normalized_text) < max(0, min_text_length)


def explain_ocr_decision(
    file_path: str | Path,
    extracted_text: str,
    classification_hint: str | None = None,
    extraction_failed: bool = False,
    min_text_length: int = DEFAULT_OCR_MIN_CHARS,
    skip_ocr_if_filename_hint: bool = True,
) -> dict[str, Any]:
    """
    Describe whether OCR should run and why.
    """

    normalized_text = normalize_text(extracted_text)
    text_length = len(normalized_text)
    file_ext = Path(file_path).suffix.lower()
    hint = classification_hint or detect_filename_classification_hint(file_path)

    run_ocr = should_run_ocr(
        file_path=file_path,
        normalized_text=normalized_text,
        classification_hint=hint,
        extraction_failed=extraction_failed,
        min_text_length=min_text_length,
        skip_ocr_if_filename_hint=skip_ocr_if_filename_hint,
    )

    if file_ext != ".pdf":
        reason = "non_pdf"
    elif skip_ocr_if_filename_hint and hint:
        reason = f"filename_hint_skip_ocr:{hint}"
    elif extraction_failed:
        reason = "text_extraction_failed"
    elif text_length == 0:
        reason = "text_empty"
    elif text_length < min_text_length:
        reason = f"text_short:{text_length}"
    else:
        reason = f"text_ok:{text_length}"

    return {
        "run_ocr": run_ocr,
        "reason": reason,
        "classification_hint": hint,
        "text_length": text_length,
        "file_ext": file_ext,
        "extraction_failed": extraction_failed,
        "skip_ocr_if_filename_hint": skip_ocr_if_filename_hint,
        "hint_evidence": build_filename_hint_evidence(file_path, hint) if hint else "",
    }


def ocr_pdf_file(
    path: str | Path,
    max_pages: int = OCR_MAX_PAGES,
    total_limit: int = 3000,
    part_limit: int = 1000,
    early_stop_chars: int = OCR_EARLY_STOP_CHARS,
    render_scale: float = OCR_RENDER_SCALE,
) -> dict[str, Any]:
    """
    OCR a scanned PDF with EasyOCR Korean model and return sampled text.

    Korean-focused version:
    - Render PDF pages at higher resolution.
    - Preprocess image for readability.
    - Use EasyOCR with ["ko", "en"].
    - Stop early once enough text is collected.
    """

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

    if file_path.suffix.lower() != ".pdf":
        return {
            "ok": False,
            "file_path": str(file_path),
            "text": "",
            "pages_scanned": 0,
            "elapsed": 0.0,
            "error": f"OCR target is not a PDF: {file_path.suffix}",
        }

    start = time.perf_counter()
    engine = get_ocr_engine()
    page_texts: list[str] = []
    scanned_pages = 0
    collected_chars = 0

    try:
        with fitz.open(file_path) as document:
            page_limit = min(len(document), max(0, max_pages))

            for page_index in range(page_limit):
                page = document.load_page(page_index)

                matrix = fitz.Matrix(render_scale, render_scale)

                pixmap = page.get_pixmap(
                    matrix=matrix,
                    alpha=False,
                    colorspace=fitz.csRGB,
                )

                image = Image.frombytes(
                    "RGB",
                    [pixmap.width, pixmap.height],
                    pixmap.samples,
                )

                image_array = preprocess_ocr_image(image)

                ocr_result = engine.readtext(
                    image_array,
                    detail=1,
                    paragraph=False,
                    decoder="greedy",
                )

                page_text = _flatten_easyocr_result(ocr_result)

                if page_text:
                    page_texts.append(page_text)
                    collected_chars += len(normalize_text(page_text))

                scanned_pages += 1

                if collected_chars >= early_stop_chars:
                    break

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


def preprocess_ocr_image(image: Image.Image) -> np.ndarray:
    """
    Improve OCR readability before sending the image to EasyOCR.

    EasyOCR는 RGB numpy array 입력을 받을 수 있다.
    """

    gray = image.convert("L")
    array = np.array(gray).astype(np.float32)

    min_value = float(array.min())
    max_value = float(array.max())

    if max_value - min_value < 5:
        output = np.clip(array, 0, 255).astype(np.uint8)
        return np.stack([output, output, output], axis=-1)

    normalized = (array - min_value) / (max_value - min_value)
    normalized = normalized * 255.0

    mean_value = normalized.mean()
    contrast = (normalized - mean_value) * 1.25 + mean_value

    enhanced = np.where(
        contrast < 180,
        contrast * 0.88,
        np.minimum(contrast * 1.03, 255),
    )

    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    return np.stack([enhanced, enhanced, enhanced], axis=-1)


def _flatten_easyocr_result(ocr_result: Any) -> str:
    """
    Convert EasyOCR result into plain text lines.

    EasyOCR result format:
    [
        (bbox, text, confidence),
        ...
    ]
    """

    if not ocr_result:
        return ""

    lines: list[str] = []

    for item in ocr_result:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue

        text_value = str(item[1]).strip()

        if text_value:
            lines.append(text_value)

    return "\n".join(lines)