"""
stage5_rag.py — Stage 5: RAG 유사 카테고리 힌트 검색

[역할] 저신뢰·기타·1차 실패 시, 기존 카테고리 중 무엇과 가까운지 힌트를 찾습니다.
       2차 Claude 프롬프트에 넣어 "기존에 넣을 곳이 있는지" 판단하게 합니다.
[현재] keywords.json + categories.json 기반 키워드·파일명 매칭 (로컬 스텁)
[추후] DB·벡터 검색 API 가 준비되면 fetch_category_hints() 내부만 교체
[설정] config/categories.json (설명·예시), config/keywords.json
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.loader import BASE_KEYWORDS, CATEGORY_CATALOG
from models.schemas import EvidencePackage

TOP_K = 3


@dataclass
class CategoryHint:
    name: str
    description: str
    score: float
    sample_filenames: list[str]


def _score_category(pkg: EvidencePackage, name: str, meta: dict) -> float:
    """키워드·파일명 겹침으로 유사 카테고리 점수 (DB/RAG 전 임시 구현)."""
    score = 0.0
    filename_lower = pkg.filename.lower()
    hits_lower = {h.lower() for h in pkg.keyword_hits}

    for word in BASE_KEYWORDS.get(name, []):
        w = word.lower()
        if w in filename_lower:
            score += 2.0
        if w in hits_lower:
            score += 1.5

    for word in meta.get("sample_keywords", []):
        w = word.lower()
        if w in filename_lower:
            score += 1.0
        if w in hits_lower:
            score += 0.8

    for sample in meta.get("sample_filenames", []):
        stem = Path(sample).stem.lower()
        if stem and stem in filename_lower:
            score += 1.2

    return score


def fetch_category_hints(pkg: EvidencePackage, top_k: int = TOP_K) -> list[CategoryHint]:
    """
    유사 카테고리 힌트를 반환합니다.

    추후 DB·벡터 검색 API로 교체할 인터페이스입니다.
    """
    catalog = CATEGORY_CATALOG or {
        name: {"description": "", "sample_keywords": words, "sample_filenames": []}
        for name, words in BASE_KEYWORDS.items()
    }

    ranked: list[CategoryHint] = []
    for name, meta in catalog.items():
        if name == "기타":
            continue
        score = _score_category(pkg, name, meta)
        if score <= 0:
            continue
        ranked.append(
            CategoryHint(
                name=name,
                description=meta.get("description", ""),
                score=score,
                sample_filenames=list(meta.get("sample_filenames", [])),
            )
        )

    ranked.sort(key=lambda h: h.score, reverse=True)
    return ranked[:top_k]
