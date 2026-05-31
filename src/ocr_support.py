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


# ---------------------------------------------------------------------------
# Speed-first OCR settings
# ---------------------------------------------------------------------------

# 기존 5페이지보다 빠르게 처리하기 위해 최대 2페이지만 OCR
OCR_MAX_PAGES = 2

# 기존 2.0보다 낮은 해상도로 렌더링하여 OCR 속도 향상
OCR_RENDER_SCALE = 1.25

# 텍스트 추출 결과가 이보다 짧으면 OCR 후보
DEFAULT_OCR_MIN_CHARS = 100

# OCR 중 충분한 텍스트가 모이면 더 이상 페이지를 읽지 않고 중단
OCR_EARLY_STOP_CHARS = 800

_OCR_ENGINE: RapidOCR | None = None


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


def get_ocr_engine() -> RapidOCR:
    """
    Create the OCR engine once per process and then reuse it.

    OCR 엔진 생성은 비용이 크므로 파일마다 새로 만들지 않는다.
    """

    global _OCR_ENGINE

    if _OCR_ENGINE is None:
        _OCR_ENGINE = RapidOCR()

    return _OCR_ENGINE


def detect_filename_classification_hint(file_path: str | Path) -> str | None:
    """
    Return a strong category hint inferred from the file name.

    예:
    - 사업자등록증_사본.pdf -> 사업자등록증
    - 법인등기부등본.pdf -> 법인등기부등본
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

    파일명만으로 강한 분류 힌트가 있을 때,
    이후 분류 단계에서 근거 텍스트처럼 사용할 수 있다.
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

    속도 최우선 판단:
    1. PDF가 아니면 OCR 안 함
    2. 파일명 힌트가 강하면 OCR 생략 가능
    3. 텍스트 추출 실패면 OCR 실행
    4. 추출 텍스트가 기준치보다 짧으면 OCR 실행

    skip_ocr_if_filename_hint=True:
        속도 우선. 파일명 힌트가 있으면 OCR 생략.

    skip_ocr_if_filename_hint=False:
        정확도 우선. 파일명 힌트가 있어도 텍스트가 부족하면 OCR 실행.
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

    반환 예:
    {
        "run_ocr": True,
        "reason": "text_empty",
        "classification_hint": None,
        "text_length": 0,
        "file_ext": ".pdf",
        "extraction_failed": False,
        "skip_ocr_if_filename_hint": True,
        "hint_evidence": ""
    }
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
) -> dict[str, Any]:
    """
    OCR a scanned PDF and return sampled text plus metadata.

    Speed-first version:
    - 최대 OCR 페이지 수를 줄인다.
    - PDF 렌더링 해상도를 낮춘다.
    - 충분한 OCR 텍스트가 모이면 조기 종료한다.
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

                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(OCR_RENDER_SCALE, OCR_RENDER_SCALE),
                    alpha=False,
                )

                image = Image.frombytes(
                    "RGB",
                    [pixmap.width, pixmap.height],
                    pixmap.samples,
                )

                image_array = np.array(image)

                ocr_result, _ = engine(image_array)
                page_text = _flatten_ocr_result(ocr_result)

                if page_text:
                    page_texts.append(page_text)
                    collected_chars += len(normalize_text(page_text))

                scanned_pages += 1

                # 속도 최우선:
                # 충분한 텍스트가 모이면 남은 페이지는 OCR하지 않는다.
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


def _flatten_ocr_result(ocr_result: Any) -> str:
    """
    Convert RapidOCR output into plain text lines.

    RapidOCR 결과는 보통 다음 형태에 가깝다.
    [
        [box, text, score],
        [box, text, score],
        ...
    ]

    버전에 따라 text 부분이 tuple/list 형태일 수 있어 방어적으로 처리한다.
    """

    if not ocr_result:
        return ""

    lines: list[str] = []

    for item in ocr_result:
        text_value = _extract_text_from_ocr_item(item)

        if text_value:
            lines.append(text_value)

    return "\n".join(lines)


def _extract_text_from_ocr_item(item: Any) -> str:
    """
    Extract text safely from one OCR result item.
    """

    if not isinstance(item, (list, tuple)):
        return ""

    if len(item) < 2:
        return ""

    text_part = item[1]

    if isinstance(text_part, str):
        return text_part.strip()

    if isinstance(text_part, (list, tuple)) and text_part:
        return str(text_part[0]).strip()

    return str(text_part).strip()