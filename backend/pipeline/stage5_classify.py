"""
Stage 5 — Claude API 카테고리 분류 (플로우차트 최종).

피드백 임베딩 유사도로 선분류 후, 미확정 건만 stage5_claude 를 호출합니다.
파일명 룰 확정은 main.py 의 stage3_rule 에서 선행합니다.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

from models.schemas import Category, ClassifyResult, EvidencePackage
from pipeline.stage5_claude import classify_with_claude
from pipeline.stage5_common import is_llm_failure_reason

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

LLM_MIN_CONFIDENCE = float(os.getenv("LLM_MIN_CONFIDENCE", "0.60"))
EMBEDDING_MIN_SIMILARITY = float(os.getenv("EMBEDDING_MIN_SIMILARITY", "0.75"))

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def _embedding_classify(
    evidence: EvidencePackage, feedback_embeddings: list[dict]
) -> ClassifyResult | None:
    """과거 사용자 수정 임베딩과 유사하면 분류 (Stage 5 이전 단계 보조)."""
    if not feedback_embeddings or not evidence.embedding:
        return None

    best_sim = 0.0
    best_category: str | None = None
    for item in feedback_embeddings:
        emb = item.get("embedding", [])
        cat = item.get("category")
        sim = _cosine_similarity(evidence.embedding, emb)
        if sim > best_sim:
            best_sim = sim
            best_category = cat

    if best_sim < EMBEDDING_MIN_SIMILARITY or best_category is None:
        return None

    try:
        category = Category(best_category)
    except ValueError:
        category = Category.UNCLASSIFIED

    return ClassifyResult(
        filename=evidence.filename,
        file_path="",
        xxhash=evidence.xxhash,
        category=category,
        confidence=best_sim,
        reason=f"임베딩 유사도 {best_sim:.2f}",
        keywords=evidence.keyword_hits,
        classify_method="embedding",
        version_hint=evidence.version_hint,
    )


def _category_from_llm_dict(data: dict) -> Category | None:
    cat_str = str(data.get("category", ""))
    try:
        return Category(cat_str)
    except ValueError:
        for cat in Category:
            if cat.value == cat_str:
                return cat
    return None


def _result_from_claude(
    evidence: EvidencePackage, data: dict
) -> ClassifyResult | None:
    category = _category_from_llm_dict(data)
    if category is None:
        return None
    keywords = data.get("keywords", evidence.keyword_hits)
    if not isinstance(keywords, list):
        keywords = evidence.keyword_hits
    return ClassifyResult(
        filename=evidence.filename,
        file_path="",
        xxhash=evidence.xxhash,
        category=category,
        confidence=float(data.get("confidence", 0)),
        reason=str(data.get("reason", "")),
        keywords=[str(k) for k in keywords],
        classify_method="claude_api",
        version_hint=evidence.version_hint,
    )


def _review_queue_result(
    evidence: EvidencePackage, reason: str = ""
) -> ClassifyResult:
    return ClassifyResult(
        filename=evidence.filename,
        file_path="",
        xxhash=evidence.xxhash,
        category=Category.UNCLASSIFIED,
        confidence=0.0,
        reason=reason or "검토 큐 이동",
        keywords=evidence.keyword_hits,
        classify_method="review_queue",
        version_hint=evidence.version_hint,
        review_reason=reason,
    )


async def _classify_with_claude(evidence: EvidencePackage) -> Optional[dict]:
    """Claude API 호출. 실패·무효 응답 시 None."""
    try:
        data = await classify_with_claude(evidence)
    except Exception:
        return None
    if not data or is_llm_failure_reason(str(data.get("reason", ""))):
        return None
    if data.get("category") == Category.UNCLASSIFIED.value and float(
        data.get("confidence", 0)
    ) <= 0:
        return None
    return data


async def run(
    evidence: EvidencePackage, feedback_embeddings: list[dict]
) -> ClassifyResult:
    """
    Stage 5 메인 진입점.

    1) 피드백 임베딩 유사도  2) Claude API  3) 실패 시 검토 큐
    """
    try:
        emb_result = _embedding_classify(evidence, feedback_embeddings)
        if emb_result is not None:
            return emb_result

        claude_data = await _classify_with_claude(evidence)
        if claude_data and float(claude_data.get("confidence", 0)) >= LLM_MIN_CONFIDENCE:
            built = _result_from_claude(evidence, claude_data)
            if built is not None:
                return built

        return _review_queue_result(
            evidence, "Claude API 분류 실패 또는 신뢰도 부족"
        )
    except Exception as exc:
        return _review_queue_result(evidence, str(exc))
