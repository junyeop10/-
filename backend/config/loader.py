"""
loader.py — 팀 설정 JSON 로더

[역할] 서버 시작 시 keywords.json·categories.json 을 읽어 전역 변수로 제공합니다.
[전역 변수]
  - BASE_KEYWORDS    Stage1·Stage3 키워드 매칭 (keywords.json)
  - CATEGORY_CATALOG RAG·LLM 카테고리 설명·예시 (categories.json)
[수정] JSON 저장 후 uvicorn 재시작 (또는 reload_keywords / reload_categories)
"""

import json
from pathlib import Path

_KEYWORDS_PATH = Path(__file__).resolve().parent / "keywords.json"
_CATEGORIES_PATH = Path(__file__).resolve().parent / "categories.json"

_DEFAULT_KEYWORDS: dict[str, list[str]] = {
    "공고_지침_양식": ["공고", "지침", "양식", "모집", "접수", "신청"],
    "사업계획서 수행계획서": ["사업계획", "수행계획", "제안서", "직무수행"],
    "조사_참고자료": ["조사", "참고", "논문", "리서치"],
    "중간_최종 결과물 및 보고서": ["보고서", "결과보고", "최종보고", "중간보고"],
    "발표자료": ["발표", "presentation", "슬라이드", "ppt"],
    "견적_계약_정산": ["견적", "계약", "협약", "정산", "대금"],
    "기업 인증서": ["인증서", "등록증", "면허", "특허", "iso"],
    "기타": ["기타"],
}


def load_keywords() -> dict[str, list[str]]:
    """팀이 수정하는 keywords.json을 읽습니다. 없거나 깨지면 기본값 사용."""
    if not _KEYWORDS_PATH.exists():
        return dict(_DEFAULT_KEYWORDS)

    try:
        with open(_KEYWORDS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return dict(_DEFAULT_KEYWORDS)
        return {
            str(category): [str(w) for w in words]
            for category, words in data.items()
            if isinstance(words, list)
        }
    except (json.JSONDecodeError, OSError):
        return dict(_DEFAULT_KEYWORDS)


def load_categories() -> dict[str, dict]:
    """RAG·LLM용 카테고리 설명·예시 (categories.json)."""
    if not _CATEGORIES_PATH.exists():
        return {}

    try:
        with open(_CATEGORIES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return {
            str(name): {
                "description": str(meta.get("description", "")),
                "sample_keywords": [
                    str(k) for k in meta.get("sample_keywords", []) if k
                ],
                "sample_filenames": [
                    str(f) for f in meta.get("sample_filenames", []) if f
                ],
            }
            for name, meta in data.items()
            if isinstance(meta, dict)
        }
    except (json.JSONDecodeError, OSError):
        return {}


def reload_keywords() -> dict[str, list[str]]:
    """JSON 수정 후 서버 재시작 없이 다시 읽을 때 (선택)."""
    global BASE_KEYWORDS
    BASE_KEYWORDS = load_keywords()
    return BASE_KEYWORDS


def reload_categories() -> dict[str, dict]:
    global CATEGORY_CATALOG
    CATEGORY_CATALOG = load_categories()
    return CATEGORY_CATALOG


BASE_KEYWORDS = load_keywords()
CATEGORY_CATALOG = load_categories()
