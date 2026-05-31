"""Stage 3 — 키워드·룰 1차 분류 (파일명 키워드만)."""

import os
from pathlib import Path

from config.loader import BASE_KEYWORDS
from models.schemas import Category, ClassifyResult

CATEGORY_NAME_MAP = {
    "최종본": Category.FINAL,
    "발표자료": Category.PRESENTATION,
    "보고서": Category.REPORT,
    "데이터": Category.DATA,
    "참고자료": Category.REFERENCE,
    "작업중": Category.DRAFT,
}

# 파일명 1~2개 키워드만으로도 1차 확정이 가능하도록 기본 임계값을 낮춤.
# 필요 시 환경변수로 조정 가능: RULE_MIN_CONFIDENCE=0.25
RULE_MIN_CONFIDENCE = float(os.getenv("RULE_MIN_CONFIDENCE", "0.25"))
PPT_EXTENSIONS = {".ppt", ".pptx"}


def _match_filename_keywords(filename: str) -> list[str]:
    """파일명( stem )에서만 키워드를 찾습니다."""
    stem = Path(filename).stem.lower()
    hits: list[str] = []
    for _cat, words in BASE_KEYWORDS.items():
        for word in words:
            if word.lower() in stem:
                hits.append(word)
    return hits


def run(
    filename: str,
    ext: str,
    xxhash: str,
    version_hint: str = "",
) -> ClassifyResult | None:
    """
    파일명 기반 룰 분류. ppt/pptx 는 발표자료 후보로 우선 처리합니다.

    신뢰도 임계값 이상이면 ClassifyResult, 아니면 None (다음 Stage로).
    """
    try:
        ext = ext.lower()
        keyword_hits = _match_filename_keywords(filename)

        if ext in PPT_EXTENSIONS:
            return ClassifyResult(
                filename=filename,
                file_path="",
                xxhash=xxhash,
                category=Category.PRESENTATION,
                confidence=0.85,
                reason="ppt/pptx 확장자 룰",
                keywords=keyword_hits,
                classify_method="rule",
                version_hint=version_hint,
            )

        if not keyword_hits:
            return None

        counts: dict[Category, int] = {}
        for cat_name, words in BASE_KEYWORDS.items():
            if cat_name not in CATEGORY_NAME_MAP:
                continue
            cat = CATEGORY_NAME_MAP[cat_name]
            for hit in keyword_hits:
                if hit in words or hit.lower() in [w.lower() for w in words]:
                    counts[cat] = counts.get(cat, 0) + 1

        if not counts:
            return None

        best_cat = max(counts, key=counts.get)
        match_count = counts[best_cat]
        # 기존 global 분모(전체 키워드 수) 대신, 카테고리별 분모를 사용해
        # 파일명 키워드 기반 confidence가 과도하게 낮아지는 문제를 완화.
        best_cat_name = next(
            (name for name, cat in CATEGORY_NAME_MAP.items() if cat == best_cat),
            "",
        )
        best_cat_keywords = BASE_KEYWORDS.get(best_cat_name, [])
        category_total = len(best_cat_keywords)
        confidence = match_count / category_total if category_total else 0.0

        if confidence < RULE_MIN_CONFIDENCE:
            return None

        return ClassifyResult(
            filename=filename,
            file_path="",
            xxhash=xxhash,
            category=best_cat,
            confidence=confidence,
            reason=f"파일명 키워드 매칭 {match_count}건",
            keywords=keyword_hits,
            classify_method="rule",
            version_hint=version_hint,
        )
    except Exception:
        return None
