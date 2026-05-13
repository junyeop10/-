"""Hybrid rule and embedding classifier."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.feedback import build_rule_suggestions
from src.rule_classifier import RuleBasedClassifier
from src.storage import ClassificationRepository
from src.vectorizer import SentenceTransformerEmbedder


@dataclass
class ClassificationResult:
    """Classification result and score breakdown for one file."""

    predicted_category: str
    confidence: float
    final_score: float
    rule_score: float
    embedding_score: float
    feedback_score: float
    duplicate_score: float
    similarity_score: float
    embedding_used: bool
    review_required: bool
    matched_rules: list[str]
    candidate_scores: dict[str, float]
    reasoning: str
    query_embedding: list[float]


class HybridClassifier:
    """Combines rules, optional embedding similarity, feedback, and duplicate hints."""

    def __init__(
        self,
        repository: ClassificationRepository,
        embedder: SentenceTransformerEmbedder,
        rule_classifier: RuleBasedClassifier,
        rule_skip_embedding_threshold: float = 0.85,
        min_rule_matches_for_skip: int = 3,
        use_embedding_for_no_rule: bool = True,
        low_rule_confidence_threshold: float = 0.20,
        review_threshold: float = 0.65,
    ) -> None:
        """Connect classifier components and thresholds."""
        self.repository = repository
        self.embedder = embedder
        self.rule_classifier = rule_classifier
        self.rule_skip_embedding_threshold = rule_skip_embedding_threshold
        self.min_rule_matches_for_skip = min_rule_matches_for_skip
        self.use_embedding_for_no_rule = use_embedding_for_no_rule
        self.low_rule_confidence_threshold = low_rule_confidence_threshold
        self.review_threshold = review_threshold

    def classify_file(
        self,
        file_id: int,
        file_hash: str,
        text: str,
        duplicate_of_file_id: int | None,
    ) -> ClassificationResult:
        """Classify evidence text and return final recommendation."""
        del file_id
        categories = self.repository.list_categories()
        if not categories:
            raise ValueError("No categories are available.")

        normalized_text = self.rule_classifier.normalize_text(text)
        rule_breakdown = self.rule_classifier.score_text(normalized_text)
        return self.classify_with_rule_breakdown(
            file_hash=file_hash,
            text=normalized_text,
            duplicate_of_file_id=duplicate_of_file_id,
            rule_breakdown=rule_breakdown,
            categories=categories,
        )

    def classify_with_rule_breakdown(
        self,
        file_hash: str,
        text: str,
        duplicate_of_file_id: int | None,
        rule_breakdown: dict[str, Any],
        categories: list[str] | None = None,
        precomputed_query_embedding: list[float] | None = None,
    ) -> ClassificationResult:
        """Classify text using rule scores already computed by the main or worker process."""
        if categories is None:
            categories = self.repository.list_categories()
        if not categories:
            raise ValueError("No categories are available.")

        normalized_text = self.rule_classifier.normalize_text(text)
        rule_scores = self._normalize_scores(rule_breakdown["scores"], categories)

        top_rule_category = self._pick_top_category(rule_scores, categories)
        top_rule_score = rule_scores.get(top_rule_category, 0.0)
        confirmed_examples = self.repository.fetch_confirmed_examples()
        top_rule_match_count = len(rule_breakdown["matches"].get(top_rule_category, []))
        strong_rule_match = (
            top_rule_score >= self.rule_skip_embedding_threshold
            and top_rule_match_count >= self.min_rule_matches_for_skip
        )
        should_use_embedding = (
            bool(confirmed_examples)
            and not strong_rule_match
            and self.use_embedding_for_no_rule
        )

        query_embedding: list[float] = []
        embedding_breakdown: dict[str, Any] = {"scores": {category: 0.0 for category in categories}, "top_examples": {}}

        if should_use_embedding:
            if precomputed_query_embedding is not None:
                query_embedding = precomputed_query_embedding
            else:
                try:
                    query_embedding = self.embedder.encode(normalized_text)
                except Exception as error:
                    raise RuntimeError(
                        "Embedding model failed to load. Check requirements and network access."
                    ) from error
            embedding_breakdown = self.embedder.score_against_examples(
                query_embedding=query_embedding,
                examples=confirmed_examples,
                categories=categories,
            )

        embedding_scores = embedding_breakdown["scores"]
        feedback_scores = self.repository.get_feedback_adjustments(
            predicted_category=top_rule_category,
            categories=categories,
        )
        duplicate_scores = self.repository.get_duplicate_confirmed_category_scores(
            file_hash=file_hash,
            duplicate_of_file_id=duplicate_of_file_id,
            categories=categories,
        )

        candidate_scores = self._build_candidate_scores(
            categories=categories,
            rule_scores=rule_scores,
            embedding_scores=embedding_scores,
            feedback_scores=feedback_scores,
            duplicate_scores=duplicate_scores,
        )
        weak_unverified_match = top_rule_score < self.low_rule_confidence_threshold and not should_use_embedding
        if (top_rule_match_count == 0 or weak_unverified_match) and max(embedding_scores.values(), default=0.0) <= 0:
            predicted_category = "검토필요"
            candidate_scores[predicted_category] = 0.0
        else:
            predicted_category = self._pick_top_category(candidate_scores, categories)
        confidence = max(candidate_scores[predicted_category], rule_scores.get(predicted_category, 0.0))
        similarity_score = embedding_scores.get(predicted_category, 0.0)
        review_required = confidence < self.review_threshold

        reasoning = self._build_reasoning(
            predicted_category=predicted_category,
            rule_breakdown=rule_breakdown,
            embedding_breakdown=embedding_breakdown,
            feedback_scores=feedback_scores,
            duplicate_scores=duplicate_scores,
            embedding_used=should_use_embedding,
        )

        return ClassificationResult(
            predicted_category=predicted_category,
            confidence=confidence,
            final_score=confidence,
            rule_score=rule_scores.get(predicted_category, 0.0),
            embedding_score=similarity_score,
            feedback_score=feedback_scores.get(predicted_category, 0.0),
            duplicate_score=duplicate_scores.get(predicted_category, 0.0),
            similarity_score=similarity_score,
            embedding_used=should_use_embedding,
            review_required=review_required,
            matched_rules=rule_breakdown["matches"].get(predicted_category, []),
            candidate_scores=candidate_scores,
            reasoning=reasoning,
            query_embedding=query_embedding,
        )

    def persist_classification(self, file_id: int, result: ClassificationResult) -> int:
        """Save the classification result."""
        return self.repository.insert_classification(
            file_id=file_id,
            predicted_category=result.predicted_category,
            rule_score=result.rule_score,
            embedding_score=result.embedding_score,
            llm_score=0.0,
            final_score=result.final_score,
            candidate_scores_json=json.dumps(result.candidate_scores, ensure_ascii=False),
            reasoning=result.reasoning,
            status="suggested",
        )

    def suggest_rules(self, min_occurrences: int = 2) -> list[dict[str, Any]]:
        """Build candidate rules from feedback patterns."""
        return build_rule_suggestions(self.repository, min_occurrences=min_occurrences)

    def _build_candidate_scores(
        self,
        categories: list[str],
        rule_scores: dict[str, float],
        embedding_scores: dict[str, float],
        feedback_scores: dict[str, float],
        duplicate_scores: dict[str, float],
    ) -> dict[str, float]:
        """Combine score sources into final per-category confidence."""
        candidate_scores: dict[str, float] = {}
        for category in categories:
            candidate_scores[category] = round(
                (rule_scores.get(category, 0.0) * 0.55)
                + (embedding_scores.get(category, 0.0) * 0.25)
                + (feedback_scores.get(category, 0.0) * 0.10)
                + (duplicate_scores.get(category, 0.0) * 0.10),
                4,
            )
        return candidate_scores

    def _normalize_scores(
        self,
        raw_scores: dict[str, float],
        categories: list[str],
    ) -> dict[str, float]:
        """Normalize raw rule scores to 0..1."""
        strong_rule_score = 4.0
        return {
            category: round(min(raw_scores.get(category, 0.0) / strong_rule_score, 1.0), 4)
            for category in categories
        }

    def _pick_top_category(self, scores: dict[str, float], categories: list[str]) -> str:
        """Return the highest scoring category."""
        ranked = sorted(categories, key=lambda category: (-scores.get(category, 0.0), category))
        return ranked[0] if ranked else "미분류"

    def _build_reasoning(
        self,
        predicted_category: str,
        rule_breakdown: dict[str, Any],
        embedding_breakdown: dict[str, Any],
        feedback_scores: dict[str, float],
        duplicate_scores: dict[str, float],
        embedding_used: bool,
    ) -> str:
        """Build a short reason string."""
        matched = rule_breakdown["matches"].get(predicted_category, [])
        top_example = embedding_breakdown["top_examples"].get(predicted_category)
        parts = [f"recommend={predicted_category}"]
        parts.append(f"rules={', '.join(matched[:5]) if matched else 'none'}")
        parts.append(f"embedding={'used' if embedding_used else 'skipped'}")

        if top_example:
            parts.append(
                f"similar={top_example['file_name']} "
                f"(category={top_example['category']}, score={top_example['similarity']:.3f})"
            )

        feedback_value = feedback_scores.get(predicted_category, 0.0)
        if feedback_value > 0:
            parts.append(f"feedback={feedback_value:.3f}")

        duplicate_value = duplicate_scores.get(predicted_category, 0.0)
        if duplicate_value > 0:
            parts.append(f"duplicate={duplicate_value:.3f}")

        return " | ".join(parts)
