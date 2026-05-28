import json
from pathlib import Path

_KEYWORDS_PATH = Path(__file__).resolve().parent / "keywords.json"

_DEFAULT_KEYWORDS: dict[str, list[str]] = {
    "최종본": ["최종", "final", "확정", "complete"],
    "발표자료": ["발표", "presentation", "슬라이드", "ppt"],
    "보고서": ["보고서", "report", "분석", "결과"],
    "데이터": ["데이터", "data", "통계", "수치"],
    "참고자료": ["참고", "reference", "논문", "조사"],
    "작업중": ["draft", "임시", "temp", "wip", "작업중"],
    "공고": ["공고", "모집", "접수", "신청"],
    "계약": ["계약", "협약", "을", "갑", "대금"],
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


def reload_keywords() -> dict[str, list[str]]:
    """JSON 수정 후 서버 재시작 없이 다시 읽을 때 (선택)."""
    global BASE_KEYWORDS
    BASE_KEYWORDS = load_keywords()
    return BASE_KEYWORDS


# 서버 import 시 1회 로드 → Stage1·Stage3가 이 dict 사용
BASE_KEYWORDS = load_keywords()
