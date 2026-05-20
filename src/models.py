"""Shared classification models for the enterprise MVP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HierarchyPrediction:
    large_category: str
    middle_category: str
    small_category: str | None = None
    large_confidence: float = 0.0
    middle_confidence: float = 0.0
    small_confidence: float = 0.0


@dataclass
class ClassificationExplanation:
    summary: str
    matched_rules: list[str] = field(default_factory=list)
    source_scores: dict[str, float] = field(default_factory=dict)
    evidence_snippets: list[str] = field(default_factory=list)
    metadata_signals: dict[str, str] = field(default_factory=dict)
    classifier_contributions: dict[str, float] = field(default_factory=dict)
    used_ocr: bool = False
    used_llm: bool = False
    llm_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "matched_rules": self.matched_rules,
            "source_scores": self.source_scores,
            "evidence_snippets": self.evidence_snippets,
            "metadata_signals": self.metadata_signals,
            "classifier_contributions": self.classifier_contributions,
            "used_ocr": self.used_ocr,
            "used_llm": self.used_llm,
            "llm_reason": self.llm_reason,
        }
