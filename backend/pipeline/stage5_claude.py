"""Stage 5 — Claude API 카테고리 분류 (플로우차트 최종)."""

import asyncio
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from models.schemas import EvidencePackage
from pipeline.stage5_common import (
    RAG_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_rag_user_prompt,
    build_user_prompt,
    failure_result,
    parse_rag_response_text,
    parse_response_text,
)
from pipeline.stage5_rag import CategoryHint

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL_ID = "claude-sonnet-4-20250514"
MAX_TOKENS = 400
MAX_CONCURRENT_LLM = int(os.getenv("MAX_CONCURRENT_LLM", "5"))

_semaphore: asyncio.Semaphore | None = None


def _api_error_reason(exc: Exception) -> str:
    """Anthropic 예외를 테스트·로그용 한글 reason으로 변환."""
    msg = str(exc)
    lower = msg.lower()

    if isinstance(exc, anthropic.AuthenticationError):
        return "API 키 오류"
    if isinstance(exc, anthropic.RateLimitError):
        return "요청 한도 초과"
    if "credit balance" in lower or "too low" in lower:
        return "크레딧 부족"
    if "invalid x-api-key" in lower or "authentication" in lower:
        return "API 키 오류"

    detail = msg if len(msg) <= 200 else msg[:200] + "..."
    return f"API 오류: {detail}"


def _get_semaphore() -> asyncio.Semaphore:
    """동시 Claude API 호출 수를 제한하는 Semaphore (lazy 초기화)."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM)
    return _semaphore


async def _call_api_once(
    client: anthropic.AsyncAnthropic,
    user_prompt: str,
    *,
    system_prompt: str = SYSTEM_PROMPT,
    parser=parse_response_text,
) -> dict | None:
    """Semaphore 안에서 Claude API를 1회 호출합니다."""
    async with _get_semaphore():
        message = await client.messages.create(
            model=MODEL_ID,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = message.content[0].text
        return parser(text)


async def _classify(
    pkg: EvidencePackage,
    user_prompt: str,
    *,
    system_prompt: str = SYSTEM_PROMPT,
    parser=parse_response_text,
) -> dict:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return failure_result("API 키 미설정")

    client = anthropic.AsyncAnthropic(api_key=api_key)

    for attempt in range(2):
        try:
            parsed = await _call_api_once(
                client,
                user_prompt,
                system_prompt=system_prompt,
                parser=parser,
            )
            if parsed is not None:
                return parsed
        except Exception as exc:
            if attempt == 1:
                return failure_result(_api_error_reason(exc))
            continue

    return failure_result("JSON 파싱 실패")


async def classify_with_claude(pkg: EvidencePackage) -> dict:
    """
    EvidencePackage를 입력받아 Claude로 1차 카테고리 분류합니다.

    성공 시 category/confidence/reason/keywords 를 반환합니다.
    API 오류·JSON 파싱 실패 시 분류불가·confidence 0.0 (1회 재시도).
    """
    return await _classify(pkg, build_user_prompt(pkg))


async def classify_with_claude_rag(
    pkg: EvidencePackage, rag_hints: list[CategoryHint]
) -> dict:
    """
    RAG 힌트를 포함한 2차 분류.

    is_new_category·suggested_category 필드를 추가로 반환할 수 있습니다.
    """
    return await _classify(
        pkg,
        build_rag_user_prompt(pkg, rag_hints),
        system_prompt=RAG_SYSTEM_PROMPT,
        parser=parse_rag_response_text,
    )
