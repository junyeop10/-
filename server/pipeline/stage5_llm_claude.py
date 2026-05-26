"""Stage 5 LLM 분류 — Claude API 연동 모듈 (플로우차트 v2)."""

import asyncio
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from models.schemas import EvidencePackage
from pipeline.stage5_llm_common import (
    SYSTEM_PROMPT,
    build_user_prompt,
    failure_result,
    parse_response_text,
)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL_ID = "claude-sonnet-4-20250514"
MAX_TOKENS = 300
MAX_CONCURRENT_LLM = int(os.getenv("MAX_CONCURRENT_LLM", "5"))

_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    """동시 LLM 호출 수를 제한하는 Semaphore (lazy 초기화)."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM)
    return _semaphore


async def _call_api_once(client: anthropic.AsyncAnthropic, user_prompt: str) -> dict | None:
    """Semaphore 안에서 Claude API를 1회 호출합니다."""
    async with _get_semaphore():
        message = await client.messages.create(
            model=MODEL_ID,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = message.content[0].text
        return parse_response_text(text)


async def classify_with_claude(pkg: EvidencePackage) -> dict:
    """
    EvidencePackage를 입력받아 Claude로 카테고리를 분류합니다.

    성공 시 {"category", "confidence", "reason", "keywords"} 를 반환합니다.
    API 오류·JSON 파싱 실패 시 분류불가·confidence 0.0 을 반환합니다 (1회 재시도).
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return failure_result("API 오류")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    user_prompt = build_user_prompt(pkg)

    for attempt in range(2):
        try:
            parsed = await _call_api_once(client, user_prompt)
            if parsed is not None:
                return parsed
        except Exception:
            if attempt == 1:
                return failure_result("API 오류")
            continue

    return failure_result("JSON 파싱 실패")
