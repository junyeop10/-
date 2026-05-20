"""LLM provider helpers with backward-compatible Ollama support."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import error, request


DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
LLM_SKIP_CONFIDENCE = 0.8
MAX_LLM_EVIDENCE_CHARS = 1200
MAX_LLM_KEYWORDS = 12
MAX_LLM_SCORES = 6
DEFAULT_LLM_CATEGORIES = [
    "legal/계약서",
    "planning/사업계획서",
    "data/데이터",
    "reporting/발표자료",
    "finance/청구서",
    "reporting/보고서",
    "operations/과업지시서",
    "miscellaneous/검토필요",
]


BASE_CATEGORY_TO_HIERARCHY = {
    "계약서": "legal/계약서",
    "사업계획서": "planning/사업계획서",
    "데이터": "data/데이터",
    "보고서": "reporting/보고서",
    "발표자료": "reporting/발표자료",
    "청구서": "finance/청구서",
    "영수증": "finance/청구서",
    "과업지시서": "operations/과업지시서",
    "검토필요": "miscellaneous/검토필요",
}


@dataclass
class LLMDecision:
    recommended_category: str
    confidence: float
    reason: str
    evidence: list[str] | None = None
    abstain: bool = False


def should_use_llm(confidence: float) -> bool:
    return confidence < LLM_SKIP_CONFIDENCE


def aggregate_category_scores(category_scores: dict[str, float]) -> dict[str, float]:
    aggregated = {category: 0.0 for category in DEFAULT_LLM_CATEGORIES}
    for category, score in category_scores.items():
        mapped_category = BASE_CATEGORY_TO_HIERARCHY.get(category, "miscellaneous/검토필요")
        aggregated[mapped_category] = max(aggregated[mapped_category], float(score))
    return aggregated


def _trim_evidence_text(evidence_text: str, max_chars: int = MAX_LLM_EVIDENCE_CHARS) -> str:
    text = evidence_text.strip()
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2].strip()
    tail = text[-(max_chars // 3) :].strip()
    return f"{head}\n...\n{tail}"


def _compact_category_scores(category_scores: dict[str, float]) -> dict[str, float]:
    ranked = sorted(
        ((str(category), float(score)) for category, score in category_scores.items() if float(score) > 0),
        key=lambda item: item[1],
        reverse=True,
    )
    compact = ranked[:MAX_LLM_SCORES]
    if not compact:
        compact = sorted(
            ((str(category), float(score)) for category, score in category_scores.items()),
            key=lambda item: item[1],
            reverse=True,
        )[:MAX_LLM_SCORES]
    return {category: round(score, 4) for category, score in compact}


def _compact_keywords(matched_keywords: list[str]) -> list[str]:
    return [str(keyword) for keyword in matched_keywords[:MAX_LLM_KEYWORDS]]


class BaseLLMProvider:
    provider_name = "base"

    def classify(
        self,
        evidence_text: str,
        category_scores: dict[str, float],
        matched_keywords: list[str],
        model: str,
        timeout_seconds: int,
    ) -> LLMDecision:
        raise NotImplementedError


class OllamaProvider(BaseLLMProvider):
    provider_name = "ollama"

    def __init__(self, base_url: str = "http://127.0.0.1:11434") -> None:
        self.base_url = base_url.rstrip("/")

    def classify(
        self,
        evidence_text: str,
        category_scores: dict[str, float],
        matched_keywords: list[str],
        model: str,
        timeout_seconds: int,
    ) -> LLMDecision:
        payload = {
            "model": model,
            "prompt": _build_prompt(evidence_text, category_scores, matched_keywords),
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        http_request = request.Request(
            f"{self.base_url}/api/generate",
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


class OpenAICompatibleProvider(BaseLLMProvider):
    provider_name = "openai-compatible"

    def __init__(self, base_url: str, api_key_env: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env

    def classify(
        self,
        evidence_text: str,
        category_scores: dict[str, float],
        matched_keywords: list[str],
        model: str,
        timeout_seconds: int,
    ) -> LLMDecision:
        api_key = os.getenv(self.api_key_env) if self.api_key_env else ""
        if not api_key:
            raise RuntimeError("Missing API key for openai-compatible provider.")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": _build_prompt(evidence_text, category_scores, matched_keywords)},
            ],
            "temperature": 0.1,
        }
        http_request = request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=timeout_seconds) as response:
                raw_payload = json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            raise RuntimeError(f"openai-compatible request failed: {exc}") from exc
        content = raw_payload["choices"][0]["message"]["content"]
        try:
            return _validate_llm_response(json.loads(content))
        except json.JSONDecodeError as exc:
            raise RuntimeError("openai-compatible provider returned invalid JSON.") from exc


class HuggingFaceInferenceProvider(BaseLLMProvider):
    provider_name = "huggingface"

    def classify(
        self,
        evidence_text: str,
        category_scores: dict[str, float],
        matched_keywords: list[str],
        model: str,
        timeout_seconds: int,
    ) -> LLMDecision:
        raise RuntimeError("HuggingFace inference provider is not configured in this environment.")


def get_llm_provider(provider_name: str, base_url: str = "", api_key_env: str = "") -> BaseLLMProvider:
    if provider_name == "ollama":
        return OllamaProvider(base_url or "http://127.0.0.1:11434")
    if provider_name == "openai-compatible":
        return OpenAICompatibleProvider(base_url=base_url, api_key_env=api_key_env)
    if provider_name == "huggingface":
        return HuggingFaceInferenceProvider()
    raise RuntimeError(f"Unsupported LLM provider: {provider_name}")


def classify_with_provider(
    provider_name: str,
    evidence_text: str,
    category_scores: dict[str, float],
    matched_keywords: list[str],
    model: str = DEFAULT_OLLAMA_MODEL,
    timeout_seconds: int = 45,
    base_url: str = "",
    api_key_env: str = "",
) -> LLMDecision:
    provider = get_llm_provider(provider_name=provider_name, base_url=base_url, api_key_env=api_key_env)
    return provider.classify(
        evidence_text=evidence_text,
        category_scores=category_scores,
        matched_keywords=matched_keywords,
        model=model,
        timeout_seconds=timeout_seconds,
    )


def classify_with_ollama(
    evidence_text: str,
    category_scores: dict[str, float],
    matched_keywords: list[str],
    model: str = DEFAULT_OLLAMA_MODEL,
    timeout_seconds: int = 45,
) -> LLMDecision:
    return classify_with_provider(
        provider_name="ollama",
        evidence_text=evidence_text,
        category_scores=category_scores,
        matched_keywords=matched_keywords,
        model=model,
        timeout_seconds=timeout_seconds,
    )


def _build_prompt(evidence_text: str, category_scores: dict[str, float], matched_keywords: list[str]) -> str:
    compact_scores = _compact_category_scores(category_scores)
    compact_keywords = _compact_keywords(matched_keywords)
    compact_evidence = _trim_evidence_text(evidence_text)
    return (
        "Classify the document and return JSON only.\n"
        f"Allowed categories: {', '.join(DEFAULT_LLM_CATEGORIES)}\n"
        '{"recommended_category":"large/middle","confidence":0.0,"reason":"short reason","evidence":["snippet"],"abstain":false}\n'
        "Use one allowed category only.\n"
        "Keep the reason short.\n"
        "If uncertain, set abstain to true.\n\n"
        f"evidence_text:\n{compact_evidence}\n\n"
        f"category_scores:\n{json.dumps(compact_scores, ensure_ascii=False)}\n\n"
        f"matched_keywords:\n{json.dumps(compact_keywords, ensure_ascii=False)}\n"
    )


def _validate_llm_response(payload: dict[str, Any]) -> LLMDecision:
    recommended_category = str(payload.get("recommended_category", "")).strip()
    reason = str(payload.get("reason", "")).strip()
    raw_confidence = payload.get("confidence", 0.0)
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid LLM confidence: {raw_confidence}") from exc
    if recommended_category not in DEFAULT_LLM_CATEGORIES:
        raise RuntimeError(f"Invalid LLM category: {recommended_category}")
    confidence = max(0.0, min(confidence, 1.0))
    evidence = payload.get("evidence")
    if not isinstance(evidence, list):
        evidence = []
    abstain = bool(payload.get("abstain", False))
    if not reason:
        reason = "LLM classification result"
    return LLMDecision(
        recommended_category=recommended_category,
        confidence=confidence,
        reason=reason,
        evidence=[str(item) for item in evidence[:3]],
        abstain=abstain,
    )
