import asyncio
import json
import os
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
from dotenv import load_dotenv

from config.loader import BASE_KEYWORDS
from models.schemas import Category, ClassifyResult, EvidencePackage
from pipeline.stage5_llm_common import parse_response_text

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
MAX_CONCURRENT_LLM = int(os.getenv("MAX_CONCURRENT_LLM", "5"))
LLM_MIN_CONFIDENCE = float(os.getenv("LLM_MIN_CONFIDENCE", "0.60"))

_ollama_available: bool = False
_semaphore: asyncio.Semaphore | None = None

CATEGORY_NAME_MAP = {
    "최종본": Category.FINAL,
    "발표자료": Category.PRESENTATION,
    "보고서": Category.REPORT,
    "데이터": Category.DATA,
    "참고자료": Category.REFERENCE,
    "작업중": Category.DRAFT,
}

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

_LLM_FAILURE_REASONS = frozenset(
    {
        "API 오류",
        "JSON 파싱 실패",
        "Ollama 미연결",
        "Ollama 서버 오류",
        "타임아웃 초과",
        "메모리 부족 — qwen2.5:0.5b 등 더 작은 모델 권장",
    }
)


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM)
    return _semaphore


async def check_ollama() -> bool:
    """서버 시작 시 Ollama 실행 여부 확인. 응답 없으면 False."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False


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


def _category_from_llm_dict(data: dict) -> Category | None:
    cat_str = str(data.get("category", ""))
    try:
        return Category(cat_str)
    except ValueError:
        for c in Category:
            if c.value == cat_str:
                return c
    return None


def _result_from_llm_dict(
    evidence: EvidencePackage,
    data: dict,
    classify_method: str,
) -> ClassifyResult | None:
    category = _category_from_llm_dict(data)
    if category is None:
        return None
    confidence = float(data.get("confidence", 0))
    keywords = data.get("keywords", evidence.keyword_hits)
    if not isinstance(keywords, list):
        keywords = evidence.keyword_hits
    return ClassifyResult(
        filename=evidence.filename,
        file_path="",
        xxhash=evidence.xxhash,
        category=category,
        confidence=confidence,
        reason=str(data.get("reason", "")),
        keywords=[str(k) for k in keywords],
        classify_method=classify_method,
        version_hint=evidence.version_hint,
    )


async def _call_local_llm(evidence: EvidencePackage) -> Optional[dict]:
    """
    Ollama로 로컬 LLM 호출.
    실패(연결 오류, 타임아웃, JSON 파싱 실패) 시 None 반환.
    """
    prompt = f"""
파일명: {evidence.filename}
확장자: {evidence.ext}
텍스트 앞부분: {evidence.text_front}
텍스트 중간: {evidence.text_middle}
텍스트 뒷부분: {evidence.text_rear}
키워드 매칭: {evidence.keyword_hits}

카테고리 목록: {[c.value for c in Category]}
반드시 아래 JSON만 반환. 다른 텍스트 금지.
{{"category": "...", "confidence": 0.0, "reason": "...", "keywords": []}}
"""
    try:
        async with _get_semaphore():
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": "qwen2.5:3b",
                        "prompt": prompt,
                        "stream": False,
                    },
                )
                resp.raise_for_status()
                result_text = resp.json().get("response", "")
                parsed = parse_response_text(result_text)
                if parsed is None:
                    try:
                        return json.loads(result_text)
                    except json.JSONDecodeError:
                        return None
                return parsed
    except Exception:
        return None


async def _call_claude_api(evidence: EvidencePackage) -> Optional[dict]:
    """Claude API 폴백 (stage5_llm_claude 위임). 실패 시 None."""
    from pipeline.stage5_llm_claude import classify_with_claude

    try:
        data = await classify_with_claude(evidence)
        if not data or data.get("reason") in _LLM_FAILURE_REASONS:
            return None
        if data.get("category") == Category.UNCLASSIFIED.value and float(
            data.get("confidence", 0)
        ) <= 0:
            return None
        return data
    except Exception:
        return None


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
    """Stage 5 — 임베딩 유사도(피드백) 후 로컬 LLM → Claude 폴백. 룰은 main에서 stage3_rule."""
    try:
        emb_result = _embedding_classify(evidence, feedback_embeddings)
        if emb_result is not None:
            return emb_result

        # 3단계 — 로컬 LLM (Ollama 사용 가능할 때만)
        if _ollama_available:
            local_result = await _call_local_llm(evidence)
            if local_result and float(local_result.get("confidence", 0)) >= LLM_MIN_CONFIDENCE:
                built = _result_from_llm_dict(evidence, local_result, "local_llm")
                if built is not None:
                    return built

        # 4단계 — Claude API 폴백
        claude_result = await _call_claude_api(evidence)
        if claude_result and float(claude_result.get("confidence", 0)) >= LLM_MIN_CONFIDENCE:
            built = _result_from_llm_dict(evidence, claude_result, "claude_api")
            if built is not None:
                return built

        # 전부 실패 → 검토 큐
        return ClassifyResult(
            filename=evidence.filename,
            file_path="",
            xxhash=evidence.xxhash,
            category=Category.UNCLASSIFIED,
            confidence=0.0,
            reason="모든 분류 단계 실패",
            keywords=[],
            classify_method="review_queue",
            version_hint=evidence.version_hint,
            review_reason="모든 분류 단계 실패",
        )
    except Exception as e:
        return _review_queue_result(evidence, str(e))
