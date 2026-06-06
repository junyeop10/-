"""
stage4_embedding.py — Stage 4: EvidencePackage 조립 (진입점)

[역할] stage1_evidence 를 호출해 LLM 분류에 쓸 증거 패키지를 만듭니다.
       main.py 에서 룰 분류(stage3)에 걸리지 않은 파일만 여기로 옵니다.
[입력] file_bytes, filename, ext, size_kb, modified_at, xxhash, extract_result
[출력] EvidencePackage
[다음] stage5_classify.run(evidence, ...)
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
