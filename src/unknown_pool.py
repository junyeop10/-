"""Unknown/review pool helpers for batch unsupervised clustering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


UNKNOWN_REASONS = {
    "low_confidence",
    "small_margin",
    "low_text_quality",
    "short_text",
    "far_from_profiles",
    "review_required",
}


@dataclass
class UnknownDecision:
    should_store: bool
    reasons: list[str]
    nearest_category: str = ""
    nearest_similarity: float = 0.0


def decide_unknown_pool_entry(
    *,
    confidence: float,
    candidate_scores: dict[str, float],
    review_required: bool,
    text: str,
    text_quality_factor: float = 1.0,
    min_confidence: float = 0.58,
    min_margin: float = 0.08,
    min_text_length: int = 80,
) -> UnknownDecision:
    """Decide whether a document should be retained for offline unsupervised analysis."""
    reasons: list[str] = []
    ranked = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
    nearest_category = ranked[0][0] if ranked else ""
    nearest_similarity = float(ranked[0][1]) if ranked else 0.0
    margin = float(ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else nearest_similarity

    if review_required:
        reasons.append("review_required")
    if confidence < min_confidence:
        reasons.append("low_confidence")
    if margin < min_margin:
        reasons.append("small_margin")
    if text_quality_factor < 0.45:
        reasons.append("low_text_quality")
    if len((text or "").strip()) < min_text_length:
        reasons.append("short_text")

    return UnknownDecision(
        should_store=bool(reasons),
        reasons=reasons,
        nearest_category=nearest_category,
        nearest_similarity=round(nearest_similarity, 4),
    )


def summarize_unknown_row(row: dict[str, Any], limit: int = 500) -> str:
    """Return a compact text snippet suitable for clustering and AI category proposals."""
    text = str(row.get("cleaned_text") or row.get("summary_text") or "")
    return text[:limit].strip()
