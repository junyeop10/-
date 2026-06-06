"""
stage5_common.py — Stage 5 공통: 프롬프트·JSON 파싱·실패 판별

[역할] Claude 에 보낼 시스템/유저 프롬프트와 응답 JSON 파싱을 담당합니다.
       stage5_claude.py 와 stage5_classify.py 가 공통으로 사용합니다.
[주요 함수]
  - build_user_prompt / build_rag_user_prompt : 프롬프트 생성
  - parse_response_text / parse_rag_response_text : 1차·2차 응답 파싱
  - failure_result / is_llm_failure_reason : API 실패 처리
[카테고리] 팀 폴더 기준 8개 (기타 포함). 분류불가는 시스템 전용.
[담당] 이세연 (feature/stage5-claude)
"""

import json
import re

from models.schemas import CLASSIFICATION_CATEGORIES, EvidencePackage
from pipeline.stage5_rag import CategoryHint

CATEGORIES = list(CLASSIFICATION_CATEGORIES)

CATEGORY_DESCRIPTIONS = """
- 공고_지침_양식: 공고문, 사업 지침, 신청·접수 양식, 모집 안내
- 사업계획서 수행계획서: 사업계획서, 수행계획서, 제안서, 직무수행계획서
- 조사_참고자료: 시장·기술 조사, 참고 문헌, 사전 검토·배경 자료
- 중간_최종 결과물 및 보고서: 중간·최종 보고서, 결과물, 성과 보고
- 발표자료: 발표 슬라이드, PT, 시연·데모 발표 자료
- 견적_계약_정산: 견적서, 계약서, 협약서, 정산·지급 문서
- 기업 인증서: 기업·제품 인증서, 등록증, 면허, 특허, ISO 등
- 기타: 위 7개에 명확히 속하지 않는 일반 문서 (새 유형은 suggested_category로 제안)
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

RAG_SYSTEM_PROMPT = f"""당신은 기업 문서 파일 분류 전문가입니다.
유사 카테고리 힌트(RAG)와 파일 정보를 보고 최종 분류하세요.

{CATEGORY_DESCRIPTIONS}

판단 규칙:
1. 힌트와 본문이 기존 7개 카테고리 중 하나에 맞으면 해당 category를 선택하세요.
2. 7개에 맞지 않지만 문서 유형이 분명하면 category를 "기타"로 두고 is_new_category=true, suggested_category에 새 카테고리명·설명을 제안하세요.
3. 정말 판단 불가하면 category="기타", is_new_category=false 로 두세요.

반드시 아래 JSON만 출력하세요.
{{"category": str, "confidence": float, "reason": str, "keywords": list[str], "is_new_category": bool, "suggested_category": {{"name": str, "description": str}} | null}}

category는 반드시 다음 중 하나: {", ".join(CATEGORIES)}
confidence는 0.0~1.0 사이 숫자입니다.
reason, keywords, suggested_category.description은 한국어로만 작성하세요.
"""


_LLM_FAILURE_REASONS_EXACT = frozenset(
    {
        "JSON 파싱 실패",
        "API 키 미설정",
        "크레딧 부족",
        "API 키 오류",
        "요청 한도 초과",
    }
)


def is_llm_failure_reason(reason: str) -> bool:
    """Claude API·파싱 실패 reason 여부."""
    if reason in _LLM_FAILURE_REASONS_EXACT:
        return True
    return reason.startswith("API 오류:")


def failure_result(reason: str) -> dict:
    """API·파싱 실패 시 반환 형식 (시스템용 분류불가)."""
    return {
        "category": "분류불가",
        "confidence": 0.0,
        "reason": reason,
        "keywords": [],
        "is_new_category": False,
        "suggested_category": None,
    }


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


def build_rag_user_prompt(pkg: EvidencePackage, hints: list[CategoryHint]) -> str:
    """RAG 힌트를 포함한 2차 분류 프롬프트."""
    base = build_user_prompt(pkg)
    if not hints:
        hint_block = "(유사 카테고리 힌트 없음)"
    else:
        lines = []
        for i, hint in enumerate(hints, start=1):
            samples = ", ".join(hint.sample_filenames) or "(없음)"
            lines.append(
                f"{i}. {hint.name} (score={hint.score:.1f})\n"
                f"   설명: {hint.description}\n"
                f"   예시 파일명: {samples}"
            )
        hint_block = "\n".join(lines)

    return f"""{base}

--- RAG 유사 카테고리 힌트 ---
{hint_block}

위 힌트를 참고해 기존 카테고리에 넣을 수 있는지 판단하세요.
넣을 곳이 없으면 is_new_category와 suggested_category를 사용하세요.
"""


def _normalize_keywords(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(k) for k in raw]


def _normalize_suggested(raw: object) -> dict | None:
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name", "")).strip()
    description = str(raw.get("description", "")).strip()
    if not name:
        return None
    return {"name": name, "description": description}


def parse_response_text(text: str) -> dict | None:
    """모델 응답 텍스트에서 JSON 객체를 추출·파싱합니다."""
    return _parse_json_response(text, rag_mode=False)


def parse_rag_response_text(text: str) -> dict | None:
    """RAG 2차 분류 응답 파싱."""
    return _parse_json_response(text, rag_mode=True)


def _parse_json_response(text: str, rag_mode: bool) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    category = str(data.get("category", "기타"))
    if category not in CATEGORIES:
        category = "기타"

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    result = {
        "category": category,
        "confidence": confidence,
        "reason": str(data.get("reason", "")),
        "keywords": _normalize_keywords(data.get("keywords", [])),
        "is_new_category": False,
        "suggested_category": None,
    }

    if rag_mode:
        result["is_new_category"] = bool(data.get("is_new_category", False))
        if result["is_new_category"]:
            result["suggested_category"] = _normalize_suggested(
                data.get("suggested_category")
            )

    return result
