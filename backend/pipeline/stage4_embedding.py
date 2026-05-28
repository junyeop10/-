"""Stage 4 — 임베딩·EvidencePackage 구성."""

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
