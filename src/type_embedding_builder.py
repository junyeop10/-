"""Type-focused embedding text builder for document clustering."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from src.text_cleaner import build_sampled_text, normalize_text, tokenize_text


PROTECTED_TYPE_TERMS = {
    "계약",
    "계약서",
    "견적",
    "견적서",
    "영수",
    "영수증",
    "회의",
    "회의록",
    "공문",
    "신청",
    "신청서",
    "증명",
    "증명서",
    "보고",
    "보고서",
    "발주",
    "발주서",
    "청구",
    "청구서",
    "금액",
    "날짜",
    "승인",
    "서명",
    "contract",
    "agreement",
    "estimate",
    "quotation",
    "receipt",
    "invoice",
    "meeting",
    "minutes",
    "application",
    "certificate",
    "report",
    "purchase",
    "order",
    "signature",
}

TYPE_KEYWORD_GROUPS: dict[str, tuple[str, ...]] = {
    "contract": ("계약", "계약서", "협약", "약정", "갑", "을", "제1조", "contract", "agreement"),
    "estimate": ("견적", "견적서", "quotation", "estimate", "quote"),
    "receipt": ("영수", "영수증", "승인", "합계", "부가세", "receipt"),
    "invoice": ("청구", "청구서", "세금계산서", "계산서", "invoice", "tax"),
    "meeting": ("회의", "회의록", "안건", "참석자", "minutes", "meeting"),
    "official_letter": ("공문", "수신", "참조", "제목", "붙임", "시행"),
    "application": ("신청", "신청서", "지원", "접수", "application", "apply"),
    "certificate": ("증명", "증명서", "확인서", "등록증", "발급", "certificate"),
    "report": ("보고", "보고서", "결과", "최종", "성과", "report"),
    "purchase_order": ("발주", "발주서", "구매", "purchase", "order", "po"),
    "approval": ("승인", "결재", "서명", "직인", "날인", "signature", "approval"),
}

NOISE_LINE_PATTERNS = (
    re.compile(r"^\s*(page|p)\s*\d+\s*(/|of)?\s*\d*\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),
    re.compile(r"^\s*-?\s*\d+\s*-?\s*$"),
)


def build_type_embedding_text(evidence: dict[str, Any]) -> str:
    """Build a type-focused embedding text without replacing legacy embedding text."""
    filename = str(evidence.get("filename", ""))
    compressed_preview = str(evidence.get("compressed_preview", ""))
    sampled_text = suppress_noise_terms(str(evidence.get("sampled_text", "")))
    title, headings = _extract_title_and_headings(compressed_preview)
    type_keywords = extract_type_keywords(evidence)
    pattern_signals = build_pattern_text_signals(evidence)

    return "\n".join(
        line
        for line in (
            f"filename: {filename}",
            f"filename_type_keywords: {', '.join(_filename_type_keywords(filename))}",
            f"title: {title}" if title else "",
            f"headings: {' | '.join(headings[:8])}" if headings else "",
            f"type_keywords: {', '.join(type_keywords)}",
            f"pattern_signals: {', '.join(pattern_signals)}",
            f"cleaned_sampled_text: {build_sampled_text(sampled_text, total_limit=1200, part_limit=400)}",
        )
        if line.strip()
    ).strip()


def extract_type_keywords(evidence: dict[str, Any]) -> list[str]:
    """Return document-type keywords from filename and top tokens."""
    candidates: list[str] = []
    candidates.extend(_filename_type_keywords(str(evidence.get("filename", ""))))
    for token in evidence.get("filename_tokens", []):
        candidates.extend(_match_type_keywords(str(token)))
    for item in evidence.get("top_tokens", []):
        if not isinstance(item, dict):
            continue
        token = str(item.get("token", ""))
        candidates.extend(_match_type_keywords(token))
    candidates.extend(build_pattern_text_signals(evidence))
    return _unique_preserve_order(candidates)[:30]


def build_pattern_text_signals(evidence: dict[str, Any]) -> list[str]:
    """Build lightweight type-oriented text signals from existing evidence."""
    text = normalize_text(
        " ".join(
            [
                str(evidence.get("filename", "")),
                str(evidence.get("sampled_text", "")),
                str(evidence.get("compressed_preview", ""))[:1200],
            ]
        )
    )
    structural = evidence.get("structural_features") or {}
    layout = evidence.get("layout_features") or {}
    signals: list[str] = []
    for label, terms in TYPE_KEYWORD_GROUPS.items():
        if any(term.lower() in text.lower() for term in terms):
            signals.append(label)
    if float(structural.get("clause_pattern_score", 0.0) or 0.0) > 0:
        signals.append("contract_clause")
    if float(structural.get("legal_term_density", 0.0) or 0.0) > 0:
        signals.append("legal_terms")
    if float(layout.get("receipt_pattern_score", 0.0) or 0.0) > 0:
        signals.append("receipt_layout")
    if float(layout.get("certificate_pattern_score", 0.0) or 0.0) > 0:
        signals.append("certificate_layout")
    if re.search(r"\d{3}-\d{2}-\d{5}", text):
        signals.append("business_number")
    if re.search(r"(\d{1,3}(,\d{3})+|\d+)\s*(원|krw|￦|₩)", text, re.IGNORECASE):
        signals.append("money_amount")
    if re.search(r"\d{4}[./-]\d{1,2}[./-]\d{1,2}", text):
        signals.append("date")
    return _unique_preserve_order(signals)


def suppress_noise_terms(text: str, corpus_stats: dict[str, Any] | None = None) -> str:
    """Attenuate common organization/project/contact noise while preserving type terms."""
    del corpus_stats
    normalized = normalize_text(text or "")
    if not normalized:
        return ""
    lines = [line.strip() for line in re.split(r"[\r\n]+", normalized) if line.strip()]
    line_counts = Counter(lines)
    cleaned_lines: list[str] = []
    for line in lines:
        if _contains_protected_term(line):
            cleaned_lines.append(_mask_inline_noise(line))
            continue
        if line_counts[line] >= 3 or any(pattern.search(line) for pattern in NOISE_LINE_PATTERNS):
            cleaned_lines.append("[repeated_or_page_noise]")
            continue
        cleaned_lines.append(_mask_inline_noise(line))
    return "\n".join(cleaned_lines)


def _extract_title_and_headings(compressed_preview: str) -> tuple[str, list[str]]:
    title = ""
    headings: list[str] = []
    for raw_line in compressed_preview.splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        if lowered.startswith("title:"):
            title = line.split(":", 1)[1].strip()
        elif lowered.startswith("headings:"):
            headings = [part.strip() for part in line.split(":", 1)[1].split("|") if part.strip()]
    return suppress_noise_terms(title), [suppress_noise_terms(item) for item in headings if item]


def _filename_type_keywords(filename: str) -> list[str]:
    return _match_type_keywords(normalize_text(filename))


def _match_type_keywords(text: str) -> list[str]:
    lowered = normalize_text(text).lower()
    matches = []
    for label, terms in TYPE_KEYWORD_GROUPS.items():
        if any(term.lower() in lowered for term in terms):
            matches.append(label)
    for term in sorted(PROTECTED_TYPE_TERMS):
        if term.lower() in lowered:
            matches.append(term)
    return matches


def _contains_protected_term(text: str) -> bool:
    lowered = normalize_text(text).lower()
    return any(term.lower() in lowered for term in sorted(PROTECTED_TYPE_TERMS))


def _mask_inline_noise(text: str) -> str:
    masked = text
    masked = re.sub(r"[\w.+-]+@[\w.-]+\.\w+", "[email_noise]", masked)
    masked = re.sub(r"(\+?\d[\d\s-]{7,}\d)", "[phone_or_id_noise]", masked)
    masked = re.sub(r"([가-힣A-Za-z0-9]+(시|군|구)\s+[가-힣A-Za-z0-9\s-]+(로|길)\s*\d*)", "[address_noise]", masked)
    masked = re.sub(r"\b(page|p)\s*\d+\s*(/|of)?\s*\d*\b", "[page_noise]", masked, flags=re.IGNORECASE)
    masked = re.sub(r"\b[A-Z][A-Za-z0-9&.,\-\s]{2,}\s+(Inc|Co|Ltd|LLC|Corp)\b", "[company_noise]", masked)
    masked = re.sub(r"\b(주식회사|\(주\)|㈜)\s*[가-힣A-Za-z0-9._-]+", "[company_noise]", masked)
    return masked


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result
