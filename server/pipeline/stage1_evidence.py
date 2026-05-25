import re

from models.schemas import EvidencePackage

BASE_KEYWORDS = {
    "최종본": ["최종", "final", "확정", "complete"],
    "발표자료": ["발표", "presentation", "슬라이드", "ppt"],
    "보고서": ["보고서", "report", "분석", "결과"],
    "데이터": ["데이터", "data", "통계", "수치"],
    "참고자료": ["참고", "reference", "논문", "조사"],
    "작업중": ["draft", "임시", "temp", "wip", "작업중"],
    "공고": ["공고", "모집", "접수", "신청"],
    "계약": ["계약", "협약", "을", "갑", "대금"],
}

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
        "is_announcement": any(
            w.lower() in combined.lower() for w in BASE_KEYWORDS["공고"]
        ),
        "is_contract": any(
            w.lower() in combined.lower() for w in BASE_KEYWORDS["계약"]
        ),
        "is_draft": version_hint == "draft"
        or any(w.lower() in combined.lower() for w in BASE_KEYWORDS["작업중"]),
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
