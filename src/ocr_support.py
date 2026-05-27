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


OCR_MAX_PAGES = 5
OCR_RENDER_SCALE = 2.0
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
    """
    OCR 엔진을 한 번만 생성하고 재사용한다.

    OCR 엔진 생성은 시간이 걸릴 수 있으므로,
    파일마다 새로 만들지 않고 프로세스 안에서 재사용한다.
    """

    global _OCR_ENGINE

    if _OCR_ENGINE is None:
        _OCR_ENGINE = RapidOCR()

    return _OCR_ENGINE


def detect_filename_classification_hint(file_path: str | Path) -> str | None:
    """
    파일명에서 강한 분류 힌트를 찾는다.

    예:
    - 사업자등록증_2024.pdf -> 사업자등록증
    - 법인등기부등본.pdf -> 법인등기부등본
    """

    normalized_name = normalize_text(Path(file_path).stem)

    for category, keywords in FILENAME_HINT_RULES:
        for keyword in keywords:
            normalized_keyword = normalize_text(keyword)

            if normalized_keyword and normalized_keyword in normalized_name:
                return category

    return None


def build_filename_hint_evidence(
    file_path: str | Path,
    classification_hint: str | None = None,
) -> str:
    """
    파일명 힌트를 근거 텍스트 형태로 만든다.

    OCR 없이도 파일명만으로 강한 근거가 있을 때,
    이후 분류 단계에서 사용할 수 있는 보조 텍스트를 만든다.
    """

    hint = classification_hint or detect_filename_classification_hint(file_path)
    normalized_name = normalize_text(Path(file_path).stem)

    if not hint:
        return normalized_name

    return f"파일명근거 {hint} {normalized_name}".strip()


def should_run_ocr(
    file_path: str | Path,
    extracted_text: str | None,
    classification_hint: str | None = None,
    extraction_failed: bool = False,
    min_text_length: int = DEFAULT_OCR_MIN_CHARS,
) -> bool:
    """
    OCR 실행 여부를 판단한다.

    OCR 실행 조건:
    1. PDF 파일이어야 한다.
    2. 일반 텍스트 추출에 실패했거나, 추출된 텍스트가 너무 짧아야 한다.
    3. 파일명만으로 강한 분류 힌트가 있으면 OCR을 생략할 수 있다.

    주의:
    - OCR은 비용이 큰 작업이므로 모든 PDF에 무조건 실행하지 않는다.
    - txt, docx, xlsx, pptx 등 PDF가 아닌 파일은 여기서 OCR 대상이 아니다.
    """

    path = Path(file_path)
    file_ext = path.suffix.lower()
    normalized_text = normalize_text(extracted_text or "")
    hint = classification_hint or detect_filename_classification_hint(path)

    if file_ext != ".pdf":
        return False

    if hint:
        return False

    if extraction_failed:
        return True

    return len(normalized_text) < max(0, min_text_length)


def explain_ocr_decision(
    file_path: str | Path,
    extracted_text: str | None,
    classification_hint: str | None = None,
    extraction_failed: bool = False,
    min_text_length: int = DEFAULT_OCR_MIN_CHARS,
) -> dict[str, Any]:
    """
    OCR 실행 여부와 그 이유를 설명한다.

    반환값 예:
    {
        "run_ocr": True,
        "reason": "text_extraction_failed",
        "classification_hint": None,
        "text_length": 0,
        "file_ext": ".pdf",
        "extraction_failed": True,
        "hint_evidence": ""
    }
    """

    path = Path(file_path)
    file_ext = path.suffix.lower()
    normalized_text = normalize_text(extracted_text or "")
    text_length = len(normalized_text)
    hint = classification_hint or detect_filename_classification_hint(path)

    run_ocr = should_run_ocr(
        file_path=path,
        extracted_text=normalized_text,
        classification_hint=hint,
        extraction_failed=extraction_failed,
        min_text_length=min_text_length,
    )

    if file_ext != ".pdf":
        reason = "non_pdf"
    elif hint:
        reason = f"filename_hint:{hint}"
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
        "hint_evidence": build_filename_hint_evidence(path, hint) if hint else "",
    }


def ocr_pdf_file(
    path: str | Path,
    max_pages: int = OCR_MAX_PAGES,
    total_limit: int = 4500,
    part_limit: int = 1500,
) -> dict[str, Any]:
    """
    스캔 PDF에 OCR을 적용하고 샘플링된 텍스트와 메타데이터를 반환한다.

    처리 흐름:
    1. PDF 파일 존재 여부 확인
    2. PDF 페이지를 이미지로 렌더링
    3. RapidOCR 실행
    4. OCR 결과를 일반 텍스트로 변환
    5. 시작/중간/끝 기준으로 샘플링 텍스트 생성
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
    page_texts: list[str] = []
    scanned_pages = 0

    try:
        engine = get_ocr_engine()

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

    raw_ocr_text = "\n".join(page_texts)

    sampled_text = build_sampled_text(
        raw_ocr_text,
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
    RapidOCR 결과를 일반 텍스트 줄로 변환한다.

    RapidOCR 결과는 보통 다음 형태에 가깝다.
    [
        [box, text, score],
        [box, text, score],
        ...
    ]

    버전에 따라 text 부분이 tuple/list 형태일 수도 있어서 방어적으로 처리한다.
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
    OCR 결과 item 하나에서 텍스트만 안전하게 꺼낸다.
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