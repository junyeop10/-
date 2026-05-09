"""Rule-based classification helpers."""

from __future__ import annotations

import re
from typing import Any

from src.storage import ClassificationRepository
from src.text_cleaner import normalize_text


WEAK_KEYWORDS: dict[str, set[str]] = {
    "계약": {"갑", "을", "용역"},
    "계약서": {"갑", "을", "용역"},
    "보고서": {"결과", "분석", "현황", "성과", "요약"},
    "데이터": {"행", "열"},
}

CONTEXT_RULES: list[dict[str, Any]] = [
    {
        "category": "공고",
        "label": "문맥: 모집+신청+접수",
        "required": ["모집", "신청", "접수"],
        "weight": 4.0,
    },
    {
        "category": "공고문",
        "label": "문맥: 모집+신청+접수",
        "required": ["모집", "신청", "접수"],
        "weight": 4.0,
    },
    {
        "category": "공고",
        "label": "문맥: 공고+지원+제출서류",
        "required": ["공고", "지원", "제출서류"],
        "weight": 4.0,
    },
    {
        "category": "공고문",
        "label": "문맥: 공고+지원+제출서류",
        "required": ["공고", "지원", "제출서류"],
        "weight": 4.0,
    },
    {
        "category": "계약서",
        "label": "문맥: 갑+을+계약기간",
        "required": ["갑", "을", "계약기간"],
        "weight": 4.0,
    },
    {
        "category": "계약서",
        "label": "문맥: 계약금+계약기간",
        "required": ["계약금", "계약기간"],
        "weight": 4.0,
    },
    {
        "category": "과업지시서",
        "label": "문맥: 과업내용+용역목적",
        "required": ["과업내용", "용역목적"],
        "weight": 4.0,
    },
    {
        "category": "과업지시서",
        "label": "문맥: 수행일정+결과물 제출",
        "required": ["수행일정", "결과물 제출"],
        "weight": 4.0,
    },
    {
        "category": "보고서",
        "label": "문맥: 분석+결과+결론",
        "required": ["분석", "결과", "결론"],
        "weight": 3.5,
    },
    {
        "category": "보고서",
        "label": "문맥: 성과+요약+현황",
        "required": ["성과", "요약", "현황"],
        "weight": 3.0,
    },
    {
        "category": "발표자료",
        "label": "문맥: 슬라이드+발표+목차",
        "required": ["슬라이드", "발표", "목차"],
        "weight": 4.0,
    },
    {
        "category": "영수증",
        "label": "문맥: 승인번호+결제금액+가맹점",
        "required": ["승인번호", "결제금액", "가맹점"],
        "weight": 4.0,
    },
    {
        "category": "청구서",
        "label": "문맥: 세금계산서+공급가액+합계금액",
        "required": ["세금계산서", "공급가액", "합계금액"],
        "weight": 4.0,
    },
    {
        "category": "데이터",
        "label": "문맥: csv+데이터셋+레코드",
        "required": ["csv", "데이터셋", "레코드"],
        "weight": 4.0,
    },
]


class RuleBasedClassifier:
    """Scores text with active rules loaded from SQLite."""

    def __init__(self, repository: ClassificationRepository) -> None:
        """Connect the repository."""
        self.repository = repository

    def normalize_text(self, text: str) -> str:
        """Normalize text before rule matching."""
        return normalize_text(text)

    def score_text(self, text: str) -> dict[str, Any]:
        """Return rule matches and scores by category."""
        rules = [
            {
                "category": str(rule["category"]),
                "rule_type": str(rule["rule_type"]),
                "pattern": str(rule["pattern"]),
                "weight": float(rule["weight"]),
            }
            for rule in self.repository.fetch_active_rules()
        ]
        return score_text_with_rules(text, rules)


def score_text_with_rules(text: str, rules: list[dict[str, Any]]) -> dict[str, Any]:
    """Score normalized text with serializable rule dictionaries."""
    scores: dict[str, float] = {}
    matches: dict[str, list[str]] = {}

    for rule in rules:
        category = str(rule["category"])
        rule_type = str(rule["rule_type"])
        pattern = str(rule["pattern"])
        scores.setdefault(category, 0.0)
        matches.setdefault(category, [])

        if _is_match(text=text, rule_type=rule_type, pattern=pattern):
            scores[category] += _rule_weight(category, pattern, float(rule.get("weight", 1.0)))
            matches[category].append(pattern)

    _apply_context_rules(text=text, scores=scores, matches=matches)
    return {"scores": scores, "matches": matches}


def _rule_weight(category: str, pattern: str, base_weight: float) -> float:
    """Lower the impact of weak standalone words."""
    normalized_pattern = pattern.lower().strip()
    if normalized_pattern in WEAK_KEYWORDS.get(category, set()):
        return min(base_weight, 0.25)
    return base_weight


def _apply_context_rules(text: str, scores: dict[str, float], matches: dict[str, list[str]]) -> None:
    """Add stronger scores when several related clues appear together."""
    for rule in CONTEXT_RULES:
        category = str(rule["category"])
        required = [str(keyword).lower() for keyword in rule["required"]]
        if all(keyword in text for keyword in required):
            scores[category] = scores.get(category, 0.0) + float(rule["weight"])
            matches.setdefault(category, []).append(str(rule["label"]))


def _is_match(text: str, rule_type: str, pattern: str) -> bool:
    """Check keyword or regex rule matching."""
    if rule_type == "keyword":
        return pattern.lower() in text
    if rule_type == "regex":
        return re.search(pattern, text, re.IGNORECASE) is not None
    return False
