"""
stage5_classify.py — Stage 5: 분류 오케스트레이션 (메인 진입점)

[역할] 한 파일의 최종 카테고리를 결정합니다. main.py 가 이 모듈의 run() 만 호출합니다.
[흐름]
  1. 피드백 임베딩 유사도 (과거 사용자 수정과 비슷하면 즉시 확정)
  2. Claude 1차 분류
  3. 저신뢰·기타·실패 → RAG 힌트 → Claude 2차
  4. 새 카테고리 제안(is_new_category) → 검토 큐
  5. 그 외 실패 → 검토 큐 (classify_method=review_queue)
[입력] EvidencePackage, feedback_embeddings
[출력] ClassifyResult (classify_method: embedding|claude_api|claude_rag|review_queue)
[환경변수] LLM_MIN_CONFIDENCE, EMBEDDING_MIN_SIMILARITY
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

from models.schemas import Category, ClassifyResult, EvidencePackage
from pipeline.stage5_claude import classify_with_claude, classify_with_claude_rag
from pipeline.stage5_common import is_llm_failure_reason
from pipeline.stage5_rag import fetch_category_hints

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
    if cat_str == Category.UNCLASSIFIED.value:
        return None
    try:
        return Category(cat_str)
    except ValueError:
        for cat in Category:
            if cat.value == cat_str:
                return cat
    return None


def _result_from_llm_dict(
    evidence: EvidencePackage,
    data: dict,
    *,
    classify_method: str,
) -> ClassifyResult | None:
    category = _category_from_llm_dict(data)
    if category is None:
        return None
    keywords = data.get("keywords", evidence.keyword_hits)
    if not isinstance(keywords, list):
        keywords = evidence.keyword_hits

    suggested = data.get("suggested_category")
    if suggested is not None and not isinstance(suggested, dict):
        suggested = None

    return ClassifyResult(
        filename=evidence.filename,
        file_path="",
        xxhash=evidence.xxhash,
        category=category,
        confidence=float(data.get("confidence", 0)),
        reason=str(data.get("reason", "")),
        keywords=[str(k) for k in keywords],
        classify_method=classify_method,
        version_hint=evidence.version_hint,
        is_new_category=bool(data.get("is_new_category", False)),
        suggested_category=suggested,
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


def _is_valid_llm_data(data: Optional[dict]) -> bool:
    if not data or is_llm_failure_reason(str(data.get("reason", ""))):
        return False
    if data.get("category") == Category.UNCLASSIFIED.value:
        return False
    return True


def _needs_rag_retry(data: Optional[dict]) -> bool:
    if not _is_valid_llm_data(data):
        return True
    if float(data.get("confidence", 0)) < LLM_MIN_CONFIDENCE:
        return True
    if data.get("category") == Category.OTHER.value:
        return True
    return False


def _new_category_review_reason(data: dict) -> str:
    suggested = data.get("suggested_category") or {}
    name = suggested.get("name", "(이름 없음)")
    description = suggested.get("description", "")
    return f"새 카테고리 제안: {name} — {description}".strip(" —")


async def run(
    evidence: EvidencePackage, feedback_embeddings: list[dict]
) -> ClassifyResult:
    """
    Stage 5 메인 진입점.

    1) 피드백 임베딩  2) Claude 1차  3) RAG+Claude 2차  4) 검토 큐
    """
    try:
        emb_result = _embedding_classify(evidence, feedback_embeddings)
        if emb_result is not None:
            return emb_result

        first_pass = await classify_with_claude(evidence)
        if _is_valid_llm_data(first_pass) and not _needs_rag_retry(first_pass):
            built = _result_from_llm_dict(
                evidence, first_pass, classify_method="claude_api"
            )
            if built is not None:
                return built

        rag_hints = fetch_category_hints(evidence)
        rag_pass = await classify_with_claude_rag(evidence, rag_hints)

        if not _is_valid_llm_data(rag_pass):
            first_reason = (
                str(first_pass.get("reason", ""))
                if first_pass
                else "1차 분류 실패"
            )
            return _review_queue_result(
                evidence,
                f"Claude API 분류 실패 또는 신뢰도 부족 ({first_reason})",
            )

        if rag_pass.get("is_new_category"):
            return _review_queue_result(evidence, _new_category_review_reason(rag_pass))

        if float(rag_pass.get("confidence", 0)) < LLM_MIN_CONFIDENCE:
            return _review_queue_result(
                evidence,
                f"RAG 분류 신뢰도 부족 ({rag_pass.get('confidence', 0):.2f})",
            )

        built = _result_from_llm_dict(
            evidence, rag_pass, classify_method="claude_rag"
        )
        if built is not None:
            return built

        return _review_queue_result(
            evidence, "Claude API 분류 실패 또는 신뢰도 부족"
        )
    except Exception as exc:
        return _review_queue_result(evidence, str(exc))
