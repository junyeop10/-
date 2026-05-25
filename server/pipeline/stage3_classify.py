import asyncio
import json
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from models.schemas import Category, ClassifyResult, EvidencePackage
from config.loader import BASE_KEYWORDS

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

CATEGORY_NAME_MAP = {
    "최종본": Category.FINAL,
    "발표자료": Category.PRESENTATION,
    "보고서": Category.REPORT,
    "데이터": Category.DATA,
    "참고자료": Category.REFERENCE,
    "작업중": Category.DRAFT,
}

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MAX_CONCURRENT_LLM = int(os.getenv("MAX_CONCURRENT_LLM", "5"))

_llm_semaphore: asyncio.Semaphore | None = None

KEYWORD_TO_CATEGORY = {
    "최종": Category.FINAL,
    "final": Category.FINAL,
    "확정": Category.FINAL,
    "complete": Category.FINAL,
    "발표": Category.PRESENTATION,
    "presentation": Category.PRESENTATION,
    "슬라이드": Category.PRESENTATION,
    "ppt": Category.PRESENTATION,
    "보고서": Category.REPORT,
    "report": Category.REPORT,
    "분석": Category.REPORT,
    "결과": Category.REPORT,
    "데이터": Category.DATA,
    "data": Category.DATA,
    "통계": Category.DATA,
    "수치": Category.DATA,
    "참고": Category.REFERENCE,
    "reference": Category.REFERENCE,
    "논문": Category.REFERENCE,
    "조사": Category.REFERENCE,
    "draft": Category.DRAFT,
    "임시": Category.DRAFT,
    "temp": Category.DRAFT,
    "wip": Category.DRAFT,
    "작업중": Category.DRAFT,
}

CATEGORY_KEYWORD_GROUPS = {
    Category.FINAL: ["최종", "final", "확정", "complete"],
    Category.PRESENTATION: ["발표", "presentation", "슬라이드", "ppt"],
    Category.REPORT: ["보고서", "report", "분석", "결과"],
    Category.DATA: ["데이터", "data", "통계", "수치"],
    Category.REFERENCE: ["참고", "reference", "논문", "조사"],
    Category.DRAFT: ["draft", "임시", "temp", "wip", "작업중"],
}


def _get_semaphore() -> asyncio.Semaphore:
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM)
    return _llm_semaphore


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


def _rule_classify(evidence: EvidencePackage) -> ClassifyResult | None:
    if not evidence.keyword_hits:
        return None

    counts: dict[Category, int] = {}
    for cat_name, words in BASE_KEYWORDS.items():
        if cat_name not in CATEGORY_NAME_MAP:
            continue
        cat = CATEGORY_NAME_MAP[cat_name]
        for hit in evidence.keyword_hits:
            if hit in words or hit.lower() in [w.lower() for w in words]:
                counts[cat] = counts.get(cat, 0) + 1

    if not counts:
        return None

    best_cat = max(counts, key=counts.get)
    match_count = counts[best_cat]
    total_keywords = sum(
        len(v) for k, v in BASE_KEYWORDS.items() if k in CATEGORY_NAME_MAP
    )
    confidence = match_count / total_keywords

    if confidence < 0.80:
        return None

    return ClassifyResult(
        filename=evidence.filename,
        file_path="",
        xxhash=evidence.xxhash,
        category=best_cat,
        confidence=confidence,
        reason=f"키워드 매칭 {match_count}건",
        keywords=evidence.keyword_hits,
        classify_method="rule",
        version_hint=evidence.version_hint,
    )


def _embedding_classify(
    evidence: EvidencePackage, feedback_embeddings: list[dict]
) -> ClassifyResult | None:
    if not feedback_embeddings or not evidence.embedding:
        return None

    best_sim = 0.0
    best_category = None
    for item in feedback_embeddings:
        emb = item.get("embedding", [])
        cat = item.get("category")
        sim = _cosine_similarity(evidence.embedding, emb)
        if sim > best_sim:
            best_sim = sim
            best_category = cat

    if best_sim < 0.75 or best_category is None:
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


async def _call_llm(evidence: EvidencePackage) -> dict | None:
    if not ANTHROPIC_API_KEY:
        return None

    categories = [c.value for c in Category if c != Category.UNCLASSIFIED]
    prompt = f"""다음 파일을 분류하세요. JSON만 반환하세요.

파일명: {evidence.filename}
확장자: {evidence.ext}
text_front: {evidence.text_front[:500]}
text_middle: {evidence.text_middle[:500]}
text_rear: {evidence.text_rear[:500]}
keyword_hits: {evidence.keyword_hits}
카테고리 목록: {categories}

응답 형식: {{"category": str, "confidence": float, "reason": str, "keywords": list}}"""

    import anthropic

    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    async def _request() -> dict | None:
        async with _get_semaphore():
            try:
                message = await client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = message.content[0].text
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if not match:
                    return None
                return json.loads(match.group())
            except Exception:
                return None

    result = await _request()
    if result is None:
        result = await _request()
    return result


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


async def run(
    evidence: EvidencePackage, feedback_embeddings: list[dict]
) -> ClassifyResult:
    try:
        rule_result = _rule_classify(evidence)
        if rule_result is not None:
            return rule_result

        emb_result = _embedding_classify(evidence, feedback_embeddings)
        if emb_result is not None:
            return emb_result

        llm_data = await _call_llm(evidence)
        if llm_data is None:
            return _review_queue_result(evidence, "LLM 분류 실패")

        confidence = float(llm_data.get("confidence", 0))
        if confidence < 0.60:
            return _review_queue_result(evidence, "LLM 신뢰도 부족")

        cat_str = llm_data.get("category", "")
        try:
            category = Category(cat_str)
        except ValueError:
            for c in Category:
                if c.value == cat_str:
                    category = c
                    break
            else:
                return _review_queue_result(evidence, "알 수 없는 카테고리")

        return ClassifyResult(
            filename=evidence.filename,
            file_path="",
            xxhash=evidence.xxhash,
            category=category,
            confidence=confidence,
            reason=llm_data.get("reason", ""),
            keywords=llm_data.get("keywords", evidence.keyword_hits),
            classify_method="llm",
            version_hint=evidence.version_hint,
        )
    except Exception as e:
        return _review_queue_result(evidence, str(e))
