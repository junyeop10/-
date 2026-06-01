"""Rule-based classification helpers."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.storage import ClassificationRepository
from src.text_cleaner import normalize_text


WEAK_KEYWORDS: dict[str, set[str]] = {
    "계약서": {"갑", "을", "용역", "계약"},
    "보고서": {"결과", "분석", "현황", "성과", "요약"},
    "데이터": {"행", "열"},
    "공고": {"공고", "모집"},
    "사업자등록증": {"상호", "법인명", "대표자"},
    "법인등기부등본": {"상호", "본점", "목적"},
}

CONTEXT_RULES: list[dict[str, Any]] = [
    {"category": "공고", "label": "문맥: 모집+신청+접수", "required": ["모집", "신청", "접수"], "weight": 4.0},
    {"category": "공고", "label": "문맥: 공고+지원+제출서류", "required": ["공고", "지원", "제출서류"], "weight": 4.0},
    {"category": "계약서", "label": "문맥: 갑+을+계약기간", "required": ["갑", "을", "계약기간"], "weight": 4.0},
    {"category": "계약서", "label": "문맥: 계약금+계약기간", "required": ["계약금", "계약기간"], "weight": 4.0},
    {"category": "과업지시서", "label": "문맥: 과업내용+용역목적", "required": ["과업내용", "용역목적"], "weight": 4.0},
    {"category": "과업지시서", "label": "문맥: 수행일정+결과물제출", "required": ["수행일정", "결과물 제출"], "weight": 4.0},
    {"category": "보고서", "label": "문맥: 분석+결과+결론", "required": ["분석", "결과", "결론"], "weight": 3.5},
    {"category": "보고서", "label": "문맥: 성과+요약+현황", "required": ["성과", "요약", "현황"], "weight": 3.0},
    {"category": "발표자료", "label": "문맥: 슬라이드+발표+목차", "required": ["슬라이드", "발표", "목차"], "weight": 4.0},
    {"category": "영수증", "label": "문맥: 승인번호+결제금액+가맹점", "required": ["승인번호", "결제금액", "가맹점"], "weight": 4.0},
    {"category": "청구서", "label": "문맥: 세금계산서+공급가액+합계금액", "required": ["세금계산서", "공급가액", "합계금액"], "weight": 4.0},
    {"category": "사업자등록증", "label": "문맥: 사업자등록번호+개업연월일+대표자", "required": ["사업자등록번호", "개업연월일", "대표자"], "weight": 4.0},
    {"category": "법인등기부등본", "label": "문맥: 상호+본점+회사성립연월일", "required": ["상호", "본점", "회사성립연월일"], "weight": 4.0},
    {"category": "법인등기부등본", "label": "문맥: 등기기록+상호+목적", "required": ["등기기록", "상호", "목적"], "weight": 4.0},
    {"category": "데이터", "label": "문맥: csv+데이터셋+레코드", "required": ["csv", "데이터셋", "레코드"], "weight": 4.0},
    {"category": "사업계획서", "label": "문맥: 사업계획+추진전략+목표시장", "required": ["사업계획", "추진전략", "목표시장"], "weight": 4.0},
]


FILENAME_ONLY_RULES: dict[str, tuple[str, ...]] = {
    "계약서": ("계약서", "계약", "협약서", "협약", "contract", "agreement"),
    "보고서": ("보고서", "결과보고", "최종보고", "분석보고", "성과보고", "report"),
    "발표자료": ("발표", "발표자료", "슬라이드", "제안서", "ppt", "pptx", "presentation"),
    "회의록": ("회의록", "회의", "미팅", "meeting", "minutes"),
    "공고": ("공고", "모집", "지원사업", "사업공고"),
    "과업지시서": ("과업지시서", "과업", "용역과업", "수행일정"),
    "영수증": ("영수증", "결제", "카드", "receipt"),
    "청구서": ("청구서", "세금계산서", "invoice", "bill"),
    "사업자등록증": ("사업자등록증", "사업자", "등록증"),
    "법인등기부등본": ("법인등기부등본", "등기부등본", "등기부"),
    "재무제표증명": ("재무제표", "표준재무제표", "재무제표증명"),
    "벤처기업인증서": ("벤처기업인증서", "벤처기업확인서", "벤처기업", "인증서", "확인서"),
    "중소기업확인서": ("중소기업확인서", "중소기업", "확인서"),
    "지방세완납증명서": ("지방세완납증명서", "지방세", "완납증명서"),
    "데이터": ("데이터", "dataset", "data", "raw", "원본", "csv", "xlsx"),
}

RULE_BASED_FORCE_EXTENSIONS = {".ppt", ".pptx"}


def normalize_filename_text(text: str) -> str:
    """Normalize file name text for filename-only matching."""
    return normalize_text(text).replace(" ", "")


_NORMALIZED_WEAK_KEYWORDS: dict[str, set[str]] = {
    category: {normalize_text(keyword) for keyword in keywords}
    for category, keywords in WEAK_KEYWORDS.items()
}

_NORMALIZED_CONTEXT_RULES: list[dict[str, Any]] = [
    {
        "category": str(rule["category"]),
        "label": str(rule["label"]),
        "required": [normalize_text(str(keyword)) for keyword in rule["required"]],
        "weight": float(rule["weight"]),
    }
    for rule in CONTEXT_RULES
]

_NORMALIZED_FILENAME_ONLY_RULES: dict[str, tuple[str, ...]] = {
    category: tuple(
        normalized_keyword
        for keyword in keywords
        if (normalized_keyword := normalize_filename_text(keyword))
    )
    for category, keywords in FILENAME_ONLY_RULES.items()
}


@lru_cache(maxsize=1024)
def _get_compiled_regex(pattern: str) -> re.Pattern[str] | None:
    """Cache compiled regex objects. Invalid regex returns None instead of crashing."""
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error:
        return None


class RuleBasedClassifier:
    """Scores text with active rules loaded from SQLite."""

    def __init__(self, repository: ClassificationRepository) -> None:
        self.repository = repository
        self._cached_rules: list[dict[str, Any]] | None = None

    def normalize_text(self, text: str) -> str:
        return normalize_text(text)

    def _get_active_rules(self) -> list[dict[str, Any]]:
        """Fetch and cache active DB rules."""
        if self._cached_rules is None:
            self._cached_rules = [
                {
                    "category": str(rule["category"]),
                    "rule_type": str(rule.get("rule_type", "keyword")),
                    "pattern": str(rule["pattern"]),
                    "weight": float(rule.get("weight", 1.0)),
                    "rule_scope": str(rule.get("rule_scope", "content")),
                    "negative_weight": float(rule.get("negative_weight", 0.0)),
                }
                for rule in self.repository.fetch_active_rules()
            ]
        return self._cached_rules

    def clear_cache(self) -> None:
        self._cached_rules = None

    def score_text(self, text: str) -> dict[str, Any]:
        return score_text_with_rules(
            text=text,
            rules=self._get_active_rules(),
            file_name=None,
            apply_context=True,
        )

    def classify_file(self, file_path: str | Path, extracted_text: str = "") -> dict[str, Any]:
        """Classify a file with rule-based logic."""
        path = Path(file_path)

        if should_send_to_filename_rule_based(path):
            filename_result = classify_by_filename_only(path)

            if filename_result["category"] != "미분류":
                return {
                    "file_path": str(path),
                    "used_rule_based": True,
                    "rule_input_text": build_filename_rule_input_text(path),
                    **filename_result,
                }

        db_result = score_text_with_rules(
            text=extracted_text,
            rules=self._get_active_rules(),
            file_name=path.name,
            apply_context=True,
        )

        return {
            "file_path": str(path),
            "used_rule_based": True,
            "forced_by_extension": False,
            "matched_filename_keyword": False,
            "method": "db_rule",
            **db_result,
        }


def score_text_with_rules(
    text: str,
    rules: list[dict[str, Any]],
    file_name: str | None = None,
    apply_context: bool = True,
) -> dict[str, Any]:
    """Score normalized text and filename with serializable rule dictionaries."""
    normalized_text = normalize_text(text)
    compact_text = normalized_text.replace(" ", "")

    normalized_filename = ""
    compact_filename = ""

    if file_name:
        stem = Path(file_name).stem
        normalized_filename = normalize_text(stem)
        compact_filename = normalized_filename.replace(" ", "")

    scores: dict[str, float] = {}
    matches: dict[str, list[str]] = {}
    negative_matches: dict[str, list[str]] = {}

    for rule in rules:
        category = str(rule["category"])
        rule_type = str(rule.get("rule_type", "keyword"))
        pattern = str(rule["pattern"])
        rule_scope = str(rule.get("rule_scope", "content"))
        weight = float(rule.get("weight", 1.0))
        negative_weight = float(rule.get("negative_weight", 0.0))

        scores.setdefault(category, 0.0)
        matches.setdefault(category, [])
        negative_matches.setdefault(category, [])

        if _is_match(
            text=normalized_text,
            compact_text=compact_text,
            filename=normalized_filename,
            compact_filename=compact_filename,
            rule_type=rule_type,
            pattern=pattern,
            rule_scope=rule_scope,
        ):
            if rule_type in {"negative_keyword", "exclusion"} or negative_weight < 0:
                scores[category] -= max(abs(negative_weight), abs(weight))
                negative_matches[category].append(pattern)
                continue

            scores[category] += _rule_weight(category, pattern, weight)
            matches[category].append(pattern)

    if apply_context:
        _apply_context_rules(
            text=normalized_text,
            compact_text=compact_text,
            scores=scores,
            matches=matches,
        )

    return {
        "scores": scores,
        "matches": matches,
        "negative_matches": negative_matches,
    }


def build_rule_input_text(text: str, file_name: str | None = None) -> str:
    """Keep backward compatibility with older code."""
    normalized_text = normalize_text(text)

    if not file_name:
        return normalized_text

    normalized_name = normalize_text(Path(file_name).stem)

    if not normalized_name:
        return normalized_text

    if normalized_text:
        return f"{normalized_name} {normalized_text}"

    return normalized_name


def should_send_to_filename_rule_based(file_path: str | Path) -> bool:
    """Decide whether a file should go to filename-only rule-based classification."""
    path = Path(file_path)
    file_ext = path.suffix.lower()
    normalized_name = normalize_filename_text(path.stem)

    if file_ext in RULE_BASED_FORCE_EXTENSIONS:
        return True

    if not normalized_name:
        return False

    for keywords in _NORMALIZED_FILENAME_ONLY_RULES.values():
        if any(keyword in normalized_name for keyword in keywords):
            return True

    return False


def classify_by_filename_only(file_path: str | Path) -> dict[str, Any]:
    """Classify using only filename keywords and extension."""
    path = Path(file_path)
    file_ext = path.suffix.lower()
    normalized_name = normalize_filename_text(path.stem)

    scores: dict[str, float] = {}
    matches: dict[str, list[str]] = {}

    forced_by_extension = file_ext in RULE_BASED_FORCE_EXTENSIONS
    matched_filename_keyword = False

    if forced_by_extension:
        scores["발표자료"] = scores.get("발표자료", 0.0) + 3.0
        matches.setdefault("발표자료", []).append(file_ext)

    if normalized_name:
        for category, keywords in _NORMALIZED_FILENAME_ONLY_RULES.items():
            for keyword in keywords:
                if keyword and keyword in normalized_name:
                    matched_filename_keyword = True
                    scores[category] = scores.get(category, 0.0) + 1.0
                    matches.setdefault(category, []).append(keyword)

    if not scores:
        return {
            "category": "미분류",
            "confidence": 0.0,
            "method": "filename_only_rule",
            "scores": {},
            "matches": {},
            "negative_matches": {},
            "forced_by_extension": False,
            "matched_filename_keyword": False,
            "reason": "filename_keyword_not_found",
        }

    best_category = max(scores, key=scores.get)
    best_score = scores[best_category]
    confidence = min(1.0, best_score / 3.0)

    if forced_by_extension and matched_filename_keyword:
        reason = "forced_by_extension_and_filename_keyword_matched"
    elif forced_by_extension:
        reason = "forced_by_extension"
    else:
        reason = "filename_keyword_matched"

    return {
        "category": best_category,
        "confidence": confidence,
        "method": "filename_only_rule",
        "scores": scores,
        "matches": matches,
        "negative_matches": {},
        "forced_by_extension": forced_by_extension,
        "matched_filename_keyword": matched_filename_keyword,
        "reason": reason,
    }


def build_filename_rule_input_text(file_path: str | Path) -> str:
    """
    Build filename-only rule input text.

    File content is never included.
    Only filename and forced extension evidence are included.
    """
    path = Path(file_path)
    normalized_name = normalize_filename_text(path.stem)
    file_ext = path.suffix.lower()

    if file_ext in RULE_BASED_FORCE_EXTENSIONS:
        return f"{normalized_name} {file_ext}".strip()

    return normalized_name


def run_filename_rule_based_stage(file_path: str | Path) -> dict[str, Any]:
    """Run stage 3 filename-only rule-based classification."""
    path = Path(file_path)

    if not should_send_to_filename_rule_based(path):
        return {
            "file_path": str(path),
            "used_rule_based": False,
            "category": "미분류",
            "confidence": 0.0,
            "method": "filename_only_rule",
            "rule_input_text": "",
            "reason": "not_filename_rule_target",
        }

    result = classify_by_filename_only(path)

    return {
        "file_path": str(path),
        "used_rule_based": True,
        "rule_input_text": build_filename_rule_input_text(path),
        **result,
    }


def _rule_weight(category: str, pattern: str, base_weight: float) -> float:
    normalized_pattern = normalize_text(pattern)

    if normalized_pattern in _NORMALIZED_WEAK_KEYWORDS.get(category, set()):
        return min(base_weight, 0.25)

    return base_weight


def _apply_context_rules(
    text: str,
    compact_text: str,
    scores: dict[str, float],
    matches: dict[str, list[str]],
) -> None:
    for rule in _NORMALIZED_CONTEXT_RULES:
        category = str(rule["category"])
        required = rule["required"]

        if all(
            _contains_keyword(
                text=text,
                compact_text=compact_text,
                pattern=str(keyword),
            )
            for keyword in required
        ):
            scores[category] = scores.get(category, 0.0) + float(rule["weight"])
            matches.setdefault(category, []).append(str(rule["label"]))


def _contains_keyword(text: str, compact_text: str, pattern: str) -> bool:
    if not pattern:
        return False

    if pattern in text:
        return True

    compact_pattern = pattern.replace(" ", "")

    return bool(compact_pattern) and compact_pattern in compact_text


def _is_match(
    text: str,
    compact_text: str,
    filename: str,
    compact_filename: str,
    rule_type: str,
    pattern: str,
    rule_scope: str = "content",
) -> bool:
    normalized_pattern = normalize_text(pattern)

    if rule_scope == "filename" or rule_type == "filename_regex":
        target_text = filename
        target_compact = compact_filename
    elif rule_scope == "both":
        target_text = f"{filename} {text}".strip()
        target_compact = f"{compact_filename} {compact_text}".strip()
    else:
        target_text = text
        target_compact = compact_text

    if rule_type in {"keyword", "positive_keyword", "negative_keyword", "exclusion"}:
        return _contains_keyword(
            text=target_text,
            compact_text=target_compact,
            pattern=normalized_pattern,
        )

    if rule_type in {"regex", "filename_regex", "metadata_regex"}:
        compiled_regex = _get_compiled_regex(pattern)

        if compiled_regex is None:
            return False

        return compiled_regex.search(target_text) is not None

    if rule_type == "token_set":
        tokens = [
            normalize_text(part)
            for part in pattern.split("|")
            if normalize_text(part)
        ]

        return bool(tokens) and all(
            _contains_keyword(
                text=target_text,
                compact_text=target_compact,
                pattern=token,
            )
            for token in tokens
        )

    return False