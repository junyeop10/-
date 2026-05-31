"""Stage 5 LLM 공통 프롬프트·JSON 파싱 (Claude / Ollama 공유)."""

import json
import re

from models.schemas import EvidencePackage

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
reason과 keywords는 반드시 한국어로만 작성하세요. 중국어·영어 사용 금지.
"""


def failure_result(reason: str) -> dict:
    """API·파싱 실패 시 반환 형식."""
    return {
        "category": "분류불가",
        "confidence": 0.0,
        "reason": reason,
        "keywords": [],
    }


MAX_LLM_PROMPT_CHARS = 4000


def build_user_prompt(pkg: EvidencePackage) -> str:
    """EvidencePackage를 유저 프롬프트 문자열로 변환."""
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


def build_generate_prompt(
    pkg: EvidencePackage, max_total_chars: int = MAX_LLM_PROMPT_CHARS
) -> str:
    """
    Ollama /api/generate용 단일 프롬프트 (시스템+유저 합쳐 max_total_chars 이하).

    Claude와 동일한 시스템·유저 구조이며, CPU 환경 토큰 예산(약 4000자)을 넘지 않도록 잘립니다.
    """
    user = build_user_prompt(pkg)
    overhead = len(SYSTEM_PROMPT) + len("\n\n") + 16
    user_max = max(0, max_total_chars - overhead)
    if len(user) > user_max:
        user = user[:user_max] + "\n...(생략)"
    return f"{SYSTEM_PROMPT}\n\n{user}"


def parse_response_text(text: str) -> dict | None:
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
