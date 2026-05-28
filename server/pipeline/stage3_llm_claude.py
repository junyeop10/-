"""
Stage 3 LLM — Claude API 동기 백엔드 (라우터·동기 호출용).

비동기 파이프라인에서는 stage5_llm_claude.classify_with_claude 를 사용할 수 있습니다.
"""

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


def classify_with_claude(pkg: EvidencePackage) -> dict:
    """
    EvidencePackage를 Claude API로 분류합니다 (동기).

    성공 시 category·confidence·reason·keywords dict 를 반환합니다.
    API·파싱 실패 시 분류불가·confidence 0.0 을 반환합니다.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return failure_result("API 오류")

    client = anthropic.Anthropic(api_key=api_key)
    user_prompt = build_user_prompt(pkg)

    for attempt in range(2):
        try:
            message = client.messages.create(
                model=MODEL_ID,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = message.content[0].text
            parsed = parse_response_text(text)
            if parsed is not None:
                return parsed
        except Exception:
            if attempt == 1:
                return failure_result("API 오류")
            continue

    return failure_result("JSON 파싱 실패")
