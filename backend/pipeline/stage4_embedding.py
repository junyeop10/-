"""
Stage 4 — 증거패키지 구성 (플로우차트 최종).

텍스트 추출·OCR 결과를 받아 임베딩·의미신호·의미 코어를 담은 EvidencePackage 를 만듭니다.
파일명 룰은 main.py 에서 선행 처리됩니다.
"""

from models.schemas import EvidencePackage
from pipeline import stage1_evidence


def run(
    file_bytes: bytes,
    filename: str,
    ext: str,
    size_kb: float,
    modified_at: float,
    xxhash: str,
    extract_result: dict,
) -> EvidencePackage:
    """
    추출 결과로부터 임베딩과 EvidencePackage를 만듭니다.

    천승원 담당 브랜치에서 SBERT·차원축소 등으로 교체 예정.
    """
    try:
        return stage1_evidence.run(
            file_bytes, filename, ext, size_kb, modified_at, xxhash, extract_result
        )
    except Exception:
        return EvidencePackage(
            xxhash=xxhash,
            filename=filename,
            ext=ext,
            size_kb=size_kb,
            modified_at=modified_at,
            text_front="",
            text_middle="",
            text_rear="",
            trigger_chunks=[],
            keyword_hits=[],
            pattern_flags={},
            version_hint="",
            embedding=[],
            extract_method="",
            extract_status="failed",
        )
