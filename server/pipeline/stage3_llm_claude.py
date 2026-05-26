"""Stage 3 LLM 분류 — Claude API 연동 모듈."""

import asyncio
import json
import os
import re
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from models.schemas import EvidencePackage

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL_ID = "claude-sonnet-4-20250514"
MAX_TOKENS = 300
MAX_CONCURRENT_LLM = int(os.getenv("MAX_CONCURRENT_LLM", "5"))

CATEGORIES = [
    "최종본",
    "발표자료",
    "보고서",
    "데이터",
    "참고자료",
    "작업중",
    "분류불가",
]

CATEGORY_DESCRIPTIONS = """
- 최종본: 확정·완료된 최종 산출물 (final, 확정, complete 등)
- 발표자료: 발표·슬라이드·PPT 중심 자료
- 보고서: 분석·결과·현황을 담은 보고 문서
- 데이터: 통계·수치·표·데이터셋 중심 파일
- 참고자료: 참고·논문·조사·배경 자료
- 작업중: draft, 임시, WIP 등 미완성 초안
- 분류불가: 위 카테고리에 해당하지 않거나 판단 불가
""".strip()

SYSTEM_PROMPT = f"""당신은 기업 문서 파일 분류 전문가입니다.
아래 카테고리 중 하나만 선택하세요.

{CATEGORY_DESCRIPTIONS}

반드시 아래 JSON 형식만 출력하세요. 다른 설명·마크다운·코드블록 금지.
{{"category": str, "confidence": float, "reason": str, "keywords": list[str]}}

category는 반드시 다음 중 하나: {", ".join(CATEGORIES)}
confidence는 0.0~1.0 사이 숫자입니다.
"""

_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    """동시 LLM 호출 수를 제한하는 Semaphore (lazy 초기화)."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM)
    return _semaphore


def _failure_result(reason: str) -> dict:
    """API·파싱 실패 시 반환 형식."""
    return {
        "category": "분류불가",
        "confidence": 0.0,
        "reason": reason,
        "keywords": [],
    }


def _build_user_prompt(pkg: EvidencePackage) -> str:
    """EvidencePackage를 유저 프롬프트 문자열로 변환 (텍스트 4500자 예산)."""
    return f"""다음 파일을 분류하세요.

파일명: {pkg.filename}
확장자: {pkg.ext}
파일 크기(KB): {pkg.size_kb:.1f}
추출 상태: {pkg.extract_status}
버전 힌트: {pkg.version_hint or "(없음)"}

keyword_hits: {pkg.keyword_hits}
pattern_flags: {pkg.pattern_flags}

text_front:
{pkg.text_front}

text_middle:
{pkg.text_middle}

text_rear:
{pkg.text_rear}
"""


def _parse_response_text(text: str) -> dict | None:
    """모델 응답 텍스트에서 JSON 객체를 추출·파싱합니다."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    category = str(data.get("category", "분류불가"))
    if category not in CATEGORIES:
        category = "분류불가"

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    reason = str(data.get("reason", ""))
    keywords = data.get("keywords", [])
    if not isinstance(keywords, list):
        keywords = []
    keywords = [str(k) for k in keywords]

    return {
        "category": category,
        "confidence": confidence,
        "reason": reason,
        "keywords": keywords,
    }


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
        return _parse_response_text(text)


async def classify_with_claude(pkg: EvidencePackage) -> dict:
    """
    EvidencePackage를 입력받아 Claude로 카테고리를 분류합니다.

    성공 시 {"category", "confidence", "reason", "keywords"} 를 반환합니다.
    API 오류·JSON 파싱 실패 시 분류불가·confidence 0.0 을 반환합니다 (1회 재시도).
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return _failure_result("API 오류")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    user_prompt = _build_user_prompt(pkg)

    for attempt in range(2):
        try:
            parsed = await _call_api_once(client, user_prompt)
            if parsed is not None:
                return parsed
        except Exception:
            if attempt == 1:
                return _failure_result("API 오류")
            continue

    return _failure_result("JSON 파싱 실패")
