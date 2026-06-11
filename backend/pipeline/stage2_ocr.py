"""
stage2_ocr.py — Stage 2: OCR 폴백

[역할] stage0_extract 가 실패한 이미지·스캔 PDF 에 OCR을 시도합니다.
[입력] file_bytes, filename, ext, extract_result (이전 단계 결과)
[출력] extract_result 와 동일 형식 (성공 시 status=ocr_fallback)
[현재] OCR 엔진 미연동 — 입력을 그대로 통과 (검토 큐로 보냄)
[담당] 정건우 (feature/stage2-ocr)
"""

_IMAGE_EXT = {".jpg", ".jpeg", ".png"}


def run(file_bytes: bytes, filename: str, ext: str, extract_result: dict) -> dict:
    """
    텍스트 추출 실패 시 OCR을 시도합니다.

    현재 MVP: OCR 엔진 미연동 시 extract_result 를 그대로 반환합니다.
    정건우 담당 브랜치에서 pytesseract 등으로 확장 예정.
    """
    try:
        if extract_result.get("status") != "failed":
            return extract_result

        ext = ext.lower() if ext else ""
        if ext not in _IMAGE_EXT and ext != ".pdf":
            return extract_result

        return {
            "status": "failed",
            "front": "",
            "middle": "",
            "rear": "",
            "method": "ocr_pending",
            "reason": "OCR 모듈 미구현 — 검토 큐 또는 담당자 구현 대기",
        }
    except Exception as e:
        return {
            "status": "failed",
            "front": "",
            "middle": "",
            "rear": "",
            "method": "ocr",
            "reason": str(e),
        }
