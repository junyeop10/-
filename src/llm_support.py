"""Local LLM helpers for ambiguous document classification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import error, request


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
LLM_SKIP_CONFIDENCE = 0.8
LLM_CATEGORIES = [
    "공고_양식",
    "사업계획_기획",
    "조사_리서치",
    "발표_제안",
    "계약_정산",
    "보고_회의",
    "운영_매뉴얼",
    "기타_검토필요",
]

BASE_CATEGORY_TO_LLM_CATEGORY = {
    "공고": "공고_양식",
    "사업계획서": "사업계획_기획",
    "데이터": "조사_리서치",
    "보고서": "보고_회의",
    "발표자료": "발표_제안",
    "계약서": "계약_정산",
    "청구서": "계약_정산",
    "영수증": "계약_정산",
    "과업지시서": "운영_매뉴얼",
    "기타_검토필요": "기타_검토필요",
    "검토필요": "기타_검토필요",
}


@dataclass
class LLMDecision:
    """Validated local LLM output."""

    recommended_category: str
    confidence: float
    reason: str


def should_use_llm(confidence: float) -> bool:
    """Use the local LLM only for ambiguous or very low-confidence cases."""
    return confidence < LLM_SKIP_CONFIDENCE


def aggregate_category_scores(category_scores: dict[str, float]) -> dict[str, float]:
    """Map current classifier categories into the LLM's target taxonomy."""
    aggregated = {category: 0.0 for category in LLM_CATEGORIES}
    for category, score in category_scores.items():
        mapped_category = BASE_CATEGORY_TO_LLM_CATEGORY.get(category, "기타_검토필요")
        aggregated[mapped_category] = max(aggregated[mapped_category], float(score))
    return aggregated


def classify_with_ollama(
    evidence_text: str,
    category_scores: dict[str, float],
    matched_keywords: list[str],
    model: str = DEFAULT_OLLAMA_MODEL,
    timeout_seconds: int = 45,
) -> LLMDecision:
    """Call Ollama and return one validated JSON decision."""
    payload = {
        "model": model,
        "prompt": _build_prompt(
            evidence_text=evidence_text,
            category_scores=category_scores,
            matched_keywords=matched_keywords,
        ),
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
        },
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    http_request = request.Request(
        OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            raw_payload = json.loads(response.read().decode("utf-8"))
    except error.URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
    except TimeoutError as exc:
        raise RuntimeError("Ollama request timed out.") from exc

    raw_response = str(raw_payload.get("response", "")).strip()
    if not raw_response:
        raise RuntimeError("Ollama returned an empty response.")

    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama returned invalid JSON: {raw_response}") from exc

    return _validate_llm_response(parsed)


def _build_prompt(evidence_text: str, category_scores: dict[str, float], matched_keywords: list[str]) -> str:
    """Build a constrained JSON-only prompt for Ollama."""
    return (
        "너는 문서 분류 보조 모델이다.\n"
        "반드시 아래 카테고리 중 하나만 선택하고 JSON 객체만 출력해라.\n"
        f"카테고리 목록: {', '.join(LLM_CATEGORIES)}\n\n"
        "출력 형식:\n"
        '{"recommended_category":"카테고리명","confidence":0.0,"reason":"짧은 근거"}\n\n'
        "규칙:\n"
        "- recommended_category는 카테고리 목록 중 하나만 사용\n"
        "- confidence는 0.0 이상 1.0 이하 숫자\n"
        "- reason은 한두 문장으로 짧게 작성\n"
        "- JSON 외 텍스트 금지\n\n"
        f"evidence_text:\n{evidence_text}\n\n"
        f"category_scores:\n{json.dumps(category_scores, ensure_ascii=False)}\n\n"
        f"matched_keywords:\n{json.dumps(matched_keywords, ensure_ascii=False)}\n"
    )


def _validate_llm_response(payload: dict[str, Any]) -> LLMDecision:
    """Validate the model output against the expected schema."""
    recommended_category = str(payload.get("recommended_category", "")).strip()
    reason = str(payload.get("reason", "")).strip()
    raw_confidence = payload.get("confidence", 0.0)

    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid LLM confidence: {raw_confidence}") from exc

    if recommended_category not in LLM_CATEGORIES:
        raise RuntimeError(f"Invalid LLM category: {recommended_category}")

    confidence = max(0.0, min(confidence, 1.0))
    if not reason:
        reason = "로컬 LLM 분류 결과"

    return LLMDecision(
        recommended_category=recommended_category,
        confidence=confidence,
        reason=reason,
    )
