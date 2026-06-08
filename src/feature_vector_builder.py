"""Numeric feature vectors for type-focused document clustering."""

from __future__ import annotations

import math
import re
from typing import Any

from src.text_cleaner import normalize_text, tokenize_text


CLUSTERING_VECTOR_VERSION = "type-v1"
PATTERN_VECTOR_KEYS = (
    "money_density",
    "date_density",
    "business_number_count",
    "legal_term_density",
    "contract_clause_score",
    "receipt_term_score",
    "invoice_term_score",
    "meeting_term_score",
    "approval_term_score",
    "application_term_score",
    "certificate_term_score",
    "order_purchase_term_score",
)
LAYOUT_VECTOR_KEYS = (
    "table_count",
    "image_count",
    "bullet_ratio",
    "header_block_score",
    "footer_pattern_score",
    "signature_area_score",
    "approval_block_score",
    "chart_presence_score",
)


def build_pattern_vector(evidence: dict[str, Any]) -> list[float]:
    """Build a lightweight pattern vector using existing evidence and cheap regex checks."""
    text = normalize_text(
        " ".join(
            [
                str(evidence.get("filename", "")),
                str(evidence.get("sampled_text", "")),
                str(evidence.get("compressed_preview", ""))[:1200],
            ]
        )
    )
    lowered = text.lower()
    token_count = max(len(tokenize_text(text)), 1)
    structural = evidence.get("structural_features") or {}

    money_count = len(re.findall(r"(\d{1,3}(,\d{3})+|\d+)\s*(원|krw|￦|₩|vat|부가세|합계)", text, re.IGNORECASE))
    date_count = len(re.findall(r"(\d{4}[./-]\d{1,2}[./-]\d{1,2}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", text))
    business_number_count = len(re.findall(r"\d{3}-\d{2}-\d{5}", text))

    values = [
        min(1.0, money_count / max(token_count / 80.0, 1.0)),
        min(1.0, date_count / max(token_count / 100.0, 1.0)),
        min(1.0, business_number_count / 3.0),
        _feature_float(structural, "legal_term_density"),
        max(_feature_float(structural, "clause_pattern_score"), _term_score(lowered, ("제1조", "제 1 조", "갑", "을", "계약", "agreement"))),
        max(_feature_float(structural, "receipt_terms_count") / 5.0, _term_score(lowered, ("영수", "승인", "합계", "부가세", "receipt"))),
        _term_score(lowered, ("청구", "계산서", "세금계산서", "공급가액", "invoice", "tax")),
        _term_score(lowered, ("회의", "회의록", "안건", "참석자", "minutes", "meeting")),
        _term_score(lowered, ("승인", "결재", "서명", "직인", "날인", "approval", "signature")),
        _term_score(lowered, ("신청", "신청서", "접수", "지원", "application")),
        _term_score(lowered, ("증명", "증명서", "확인서", "등록증", "발급", "certificate")),
        _term_score(lowered, ("발주", "발주서", "구매", "purchase order", "po number")),
    ]
    return [round(max(0.0, min(float(value), 1.0)), 6) for value in values]


def build_optional_layout_vector(evidence: dict[str, Any]) -> list[float]:
    """Return layout vector from existing evidence only; never triggers layout extraction."""
    layout = evidence.get("layout_features") or {}
    structural = evidence.get("structural_features") or {}
    values = []
    for key in LAYOUT_VECTOR_KEYS:
        source = structural if key in {"table_count", "image_count", "bullet_ratio"} else layout
        value = _feature_float(source, key)
        if key in {"table_count", "image_count"}:
            value = min(1.0, value / 5.0)
        values.append(round(max(0.0, min(float(value), 1.0)), 6))
    return values


def get_layout_confidence(evidence: dict[str, Any]) -> float:
    """Estimate whether existing layout evidence is useful enough to receive weight."""
    layout = evidence.get("layout_features") or {}
    if not layout:
        return 0.0
    sampled_pages = _feature_float(layout, "sampled_page_count")
    meaningful_scores = [
        _feature_float(layout, key)
        for key in (
            "header_block_score",
            "footer_pattern_score",
            "signature_area_score",
            "approval_block_score",
            "chart_presence_score",
            "receipt_pattern_score",
            "certificate_pattern_score",
            "dense_text_score",
        )
    ]
    if sampled_pages <= 0 and max(meaningful_scores, default=0.0) <= 0:
        return 0.0
    return round(min(1.0, 0.35 * min(1.0, sampled_pages / 3.0) + 0.65 * max(meaningful_scores, default=0.0)), 6)


def normalize_vector(values: list[float]) -> list[float]:
    """L2 normalize a vector; zero vectors are returned unchanged."""
    norm = math.sqrt(sum(float(value) * float(value) for value in values))
    if norm <= 0:
        return [0.0 for _value in values]
    return [float(value) / norm for value in values]


def _feature_float(features: dict[str, Any], key: str) -> float:
    try:
        return float(features.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _term_score(text: str, terms: tuple[str, ...]) -> float:
    hits = sum(1 for term in terms if term.lower() in text)
    return min(1.0, hits / max(len(terms) / 2.0, 1.0))
