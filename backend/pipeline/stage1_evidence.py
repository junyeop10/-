"""
stage1_evidence.py — 증거패키지 내부 구성 (의미신호 + 임베딩)

[역할] 추출된 본문에서 키워드·패턴·버전 힌트를 뽑고 SBERT 임베딩 벡터를 만듭니다.
       stage4_embedding.run() 이 이 모듈을 호출합니다.
[입력] file_bytes, filename, ext, size_kb, modified_at, xxhash, extract_result
[출력] EvidencePackage (LLM·RAG·임베딩 유사도의 공통 입력)
[설정] config/keywords.json
[담당] 천승원 (feature/stage4-embedding)
"""

import re

from config.loader import BASE_KEYWORDS
from models.schemas import EvidencePackage

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _model


def _detect_version_hint(filename: str) -> str:
    lower = filename.lower()
    if re.search(r"최종|final|확정", lower, re.IGNORECASE):
        return "최종"
    if re.search(r"draft|임시", lower, re.IGNORECASE):
        return "draft"
    if re.search(r"rev\d+|v\d+", lower, re.IGNORECASE):
        return "rev"
    return ""


def _match_keywords(text: str) -> list[str]:
    lower = text.lower()
    hits = []
    for _category, words in BASE_KEYWORDS.items():
        for word in words:
            if word.lower() in lower:
                hits.append(word)
    return hits


def run(
    file_bytes: bytes,
    filename: str,
    ext: str,
    size_kb: float,
    modified_at: float,
    xxhash: str,
    extract_result: dict,
) -> EvidencePackage:
    text_front = extract_result.get("front", "")
    text_middle = extract_result.get("middle", "")
    text_rear = extract_result.get("rear", "")
    combined = text_front + text_middle + text_rear

    keyword_hits = _match_keywords(combined)
    version_hint = _detect_version_hint(filename)

    pattern_flags = {
        "is_notice_form": any(
            w.lower() in combined.lower()
            for w in BASE_KEYWORDS.get("공고_지침_양식", [])
        ),
        "is_contract_settlement": any(
            w.lower() in combined.lower()
            for w in BASE_KEYWORDS.get("견적_계약_정산", [])
        ),
        "is_certificate": any(
            w.lower() in combined.lower()
            for w in BASE_KEYWORDS.get("기업 인증서", [])
        ),
        "is_draft": version_hint == "draft",
    }

    embed_text = text_front + text_middle
    embedding: list[float] = []
    if embed_text.strip():
        try:
            embedding = _get_model().encode(embed_text).tolist()
        except Exception:
            embedding = []

    return EvidencePackage(
        xxhash=xxhash,
        filename=filename,
        ext=ext,
        size_kb=size_kb,
        modified_at=modified_at,
        text_front=text_front,
        text_middle=text_middle,
        text_rear=text_rear,
        trigger_chunks=[],
        keyword_hits=keyword_hits,
        pattern_flags=pattern_flags,
        version_hint=version_hint,
        embedding=embedding,
        extract_method=extract_result.get("method", ""),
        extract_status=extract_result.get("status", "failed"),
    )
