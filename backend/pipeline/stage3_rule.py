"""
stage3_rule.py — Stage 3: 파일명 룰 기반 1차 분류

[역할] 파일명(stem)에 keywords.json 키워드가 있으면 카테고리를 빠르게 확정합니다.
       ppt/pptx 는 발표자료로 우선 처리합니다.
[입력] filename, ext, xxhash
[출력] ClassifyResult (확정) 또는 None (다음 Stage로)
[설정] config/keywords.json, 환경변수 RULE_MIN_CONFIDENCE
[담당] 정건우 (feature/stage3-rule)
"""

import os
from pathlib import Path

from config.loader import BASE_KEYWORDS
from models.schemas import Category, ClassifyResult

CATEGORY_NAME_MAP = {
    "공고_지침_양식": Category.NOTICE_FORM,
    "사업계획서 수행계획서": Category.BUSINESS_PLAN,
    "조사_참고자료": Category.RESEARCH_REFERENCE,
    "중간_최종 결과물 및 보고서": Category.DELIVERABLE_REPORT,
    "발표자료": Category.PRESENTATION,
    "견적_계약_정산": Category.ESTIMATE_CONTRACT,
    "기업 인증서": Category.CERTIFICATE,
    "기타": Category.OTHER,
}

RULE_MIN_CONFIDENCE = float(os.getenv("RULE_MIN_CONFIDENCE", "0.25"))
PPT_EXTENSIONS = {".ppt", ".pptx"}


def _match_filename_keywords(filename: str) -> list[str]:
    """파일명(stem)에서만 키워드를 찾습니다."""
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
