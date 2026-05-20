"""Hybrid hierarchical classifier with explainable score breakdowns."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from src.feedback import build_rule_suggestions
from src.models import ClassificationExplanation, HierarchyPrediction
from src.performance import build_file_latency_analysis, normalize_stage_timings
from src.rule_classifier import RuleBasedClassifier, build_rule_input_text
from src.storage import ClassificationRepository
from src.taxonomy import Taxonomy, UNCATEGORIZED
from src.vectorizer import SentenceTransformerEmbedder


CLASSIFIER_VERSION = "2.0"


@dataclass
class ClassificationResult:
    predicted_category: str
    confidence: float
    final_score: float
    rule_score: float
    embedding_score: float
    llm_score: float
    feedback_score: float
    duplicate_score: float
    similarity_score: float
    embedding_used: bool
    review_required: bool
    matched_rules: list[str]
    candidate_scores: dict[str, float]
    reasoning: str
    query_embedding: list[float]
    large_category: str = ""
    middle_category: str = ""
    small_category: str | None = None
    large_confidence: float = 0.0
    middle_confidence: float = 0.0
    small_confidence: float = 0.0
    source_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    evidence_snippets: list[str] = field(default_factory=list)
    metadata_signals: dict[str, str] = field(default_factory=dict)
    classifier_contributions: dict[str, float] = field(default_factory=dict)
    explanation: dict[str, Any] = field(default_factory=dict)
    llm_used: bool = False
    ocr_used: bool = False
    processing_profile: dict[str, Any] = field(default_factory=dict)


def get_primary_processing_method(result: ClassificationResult) -> str:
    """Return the dominant classifier path for a result."""
    if result.llm_used:
        return "llm"
    if result.embedding_used:
        return "embedding"
    return "rule"


def get_processing_method_label(result: ClassificationResult) -> str:
    """Return a user-facing Korean label for the dominant classifier path."""
    primary = get_primary_processing_method(result)
    if primary == "llm":
        return "LLM 보조판단"
    if primary == "embedding":
        return "임베딩 보조판단"
    return "룰 기반"


def get_processing_trace_labels(result: ClassificationResult) -> list[str]:
    """Return all processing components that affected the visible result."""
    labels = [get_processing_method_label(result)]
    if result.ocr_used:
        labels.append("OCR 텍스트보강")
    if result.llm_used and result.embedding_used:
        labels.append("임베딩 선판단")
    return labels


def get_processing_trace_text(result: ClassificationResult) -> str:
    """Return a concise processing trace for CLI and GUI displays."""
    return " -> ".join(get_processing_trace_labels(result))


class HybridClassifier:
    """Combines rules, embeddings, feedback, duplicate hints, and taxonomy mapping."""

    def __init__(
        self,
        repository: ClassificationRepository,
        embedder: SentenceTransformerEmbedder | None,
        rule_classifier: RuleBasedClassifier,
        taxonomy: Taxonomy | None = None,
        rule_skip_embedding_threshold: float = 0.85,
        min_rule_matches_for_skip: int = 3,
        use_embedding_for_no_rule: bool = True,
        low_rule_confidence_threshold: float = 0.20,
        review_threshold: float = 0.65,
    ) -> None:
        self.repository = repository
        self.embedder = embedder
        self.rule_classifier = rule_classifier
        self.taxonomy = taxonomy
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
        file_name: str | None = None,
    ) -> ClassificationResult:
        del file_id
        classify_start = time.perf_counter()
        categories = self.repository.list_categories()
        if not categories:
            raise ValueError("No categories are available.")
        normalize_start = time.perf_counter()
        normalized_text = self.rule_classifier.normalize_text(text)
        normalize_time = time.perf_counter() - normalize_start
        rule_start = time.perf_counter()
        rule_breakdown = self.rule_classifier.score_text(build_rule_input_text(normalized_text, file_name))
        rule_time = time.perf_counter() - rule_start
        return self.classify_with_rule_breakdown(
            file_hash=file_hash,
            text=normalized_text,
            duplicate_of_file_id=duplicate_of_file_id,
            rule_breakdown=rule_breakdown,
            file_name=file_name,
            categories=categories,
            initial_stage_timings={
                "normalize": normalize_time,
                "rule": rule_time,
                "classification_total": time.perf_counter() - classify_start,
            },
        )

    def classify_with_rule_breakdown(
        self,
        file_hash: str,
        text: str,
        duplicate_of_file_id: int | None,
        rule_breakdown: dict[str, Any],
        categories: list[str] | None = None,
        precomputed_query_embedding: list[float] | None = None,
        precomputed_embedding_meta: dict[str, Any] | None = None,
        file_name: str | None = None,
        initial_stage_timings: dict[str, float] | None = None,
    ) -> ClassificationResult:
        classify_start = time.perf_counter()
        if categories is None:
            categories = self.repository.list_categories()
        if not categories:
            raise ValueError("No categories are available.")

        normalized_text = self.rule_classifier.normalize_text(text)
        rule_scores = self._normalize_scores(rule_breakdown["scores"], categories)
        top_rule_category = self._pick_top_category(rule_scores, categories)
        top_rule_score = rule_scores.get(top_rule_category, 0.0)
        example_start = time.perf_counter()
        confirmed_examples = self.repository.fetch_confirmed_examples()
        example_lookup_time = time.perf_counter() - example_start
        top_rule_match_count = len(rule_breakdown["matches"].get(top_rule_category, []))
        strong_rule_match = (
            top_rule_score >= self.rule_skip_embedding_threshold
            and top_rule_match_count >= self.min_rule_matches_for_skip
        )
        should_use_embedding = bool(self.embedder and confirmed_examples and not strong_rule_match and self.use_embedding_for_no_rule)

        query_embedding: list[float] = []
        embedding_meta: dict[str, Any] = {}
        embedding_time = 0.0
        embedding_breakdown: dict[str, Any] = {
            "scores": {category: 0.0 for category in categories},
            "top_examples": {},
        }
        if should_use_embedding and self.embedder is not None:
            embedding_start = time.perf_counter()
            if precomputed_query_embedding is not None:
                query_embedding = precomputed_query_embedding
                embedding_meta = dict(precomputed_embedding_meta or {})
            else:
                if isinstance(self.embedder, SentenceTransformerEmbedder):
                    query_embedding = self.embedder.encode(
                        normalized_text,
                        repository=self.repository,
                        file_hash=file_hash,
                        text_kind="query",
                    )
                    embedding_meta = self.embedder.get_last_encode_meta()
                else:
                    query_embedding = self.embedder.encode(normalized_text)
                    embedding_meta = {"cache_hit": None, "elapsed": 0.0, "model_name": "custom"}
            embedding_time = time.perf_counter() - embedding_start
            embedding_breakdown = self.embedder.score_against_examples(
                query_embedding=query_embedding,
                examples=confirmed_examples,
                categories=categories,
            )

        embedding_scores = embedding_breakdown["scores"]
        adjustment_start = time.perf_counter()
        feedback_scores = self.repository.get_feedback_adjustments(predicted_category=top_rule_category, categories=categories)
        duplicate_scores = self.repository.get_duplicate_confirmed_category_scores(
            file_hash=file_hash,
            duplicate_of_file_id=duplicate_of_file_id,
            categories=categories,
        )
        filename_scores = self._build_filename_scores(file_name=file_name, categories=categories)
        metadata_scores = self._build_metadata_scores(file_name=file_name, text=normalized_text, categories=categories)
        adjustment_time = time.perf_counter() - adjustment_start
        combine_start = time.perf_counter()
        candidate_scores = self._build_candidate_scores(
            categories=categories,
            rule_scores=rule_scores,
            embedding_scores=embedding_scores,
            feedback_scores=feedback_scores,
            duplicate_scores=duplicate_scores,
            metadata_scores=metadata_scores,
            filename_scores=filename_scores,
        )
        combine_time = time.perf_counter() - combine_start

        weak_unverified_match = top_rule_score < self.low_rule_confidence_threshold and not should_use_embedding
        if (top_rule_match_count == 0 or weak_unverified_match) and max(embedding_scores.values(), default=0.0) <= 0:
            predicted_category = UNCATEGORIZED
            candidate_scores[predicted_category] = 0.0
        else:
            predicted_category = self._pick_top_category(candidate_scores, categories)

        confidence = max(candidate_scores.get(predicted_category, 0.0), rule_scores.get(predicted_category, 0.0))
        review_required = self._needs_review(candidate_scores, confidence, top_rule_category, embedding_scores)
        hierarchy = self._resolve_hierarchy(predicted_category, confidence)
        explanation_start = time.perf_counter()
        explanation_obj = self._build_explanation(
            predicted_category=predicted_category,
            hierarchy=hierarchy,
            rule_breakdown=rule_breakdown,
            embedding_breakdown=embedding_breakdown,
            feedback_scores=feedback_scores,
            duplicate_scores=duplicate_scores,
            metadata_scores=metadata_scores,
            filename_scores=filename_scores,
            embedding_used=should_use_embedding,
            text=normalized_text,
        )
        explanation_time = time.perf_counter() - explanation_start

        predicted_middle_category = hierarchy.middle_category or predicted_category
        stage_timings = normalize_stage_timings(initial_stage_timings)
        stage_timings.update(
            {
                "confirmed_example_lookup": example_lookup_time,
                "embedding": embedding_time,
                "adjustments": adjustment_time,
                "combine": combine_time,
                "explanation": explanation_time,
                "classification": time.perf_counter() - classify_start,
            }
        )
        stage_timings["total"] = sum(
            value for key, value in stage_timings.items() if key not in {"classification_total", "total"}
        )
        processing_profile = {
            "scope": "classifier",
            "stage_timings": stage_timings,
            "embedding_meta": embedding_meta,
            "strong_rule_match": strong_rule_match,
            "top_rule_match_count": top_rule_match_count,
            "confirmed_examples_count": len(confirmed_examples),
            "analysis": build_file_latency_analysis(
                stage_timings,
                text_length=len(normalized_text),
                embedding_used=should_use_embedding,
                embedding_cache_hit=embedding_meta.get("cache_hit") if embedding_meta else None,
                strong_rule_match=strong_rule_match,
                review_required=review_required,
                matched_rules_count=len(rule_breakdown["matches"].get(predicted_middle_category, [])),
                duplicate_detected=duplicate_of_file_id is not None,
            ),
        }
        return ClassificationResult(
            predicted_category=predicted_middle_category,
            confidence=confidence,
            final_score=confidence,
            rule_score=rule_scores.get(predicted_middle_category, 0.0),
            embedding_score=embedding_scores.get(predicted_middle_category, 0.0),
            llm_score=0.0,
            feedback_score=feedback_scores.get(predicted_middle_category, 0.0),
            duplicate_score=duplicate_scores.get(predicted_middle_category, 0.0),
            similarity_score=embedding_scores.get(predicted_middle_category, 0.0),
            embedding_used=should_use_embedding,
            review_required=review_required,
            matched_rules=rule_breakdown["matches"].get(predicted_middle_category, []),
            candidate_scores=candidate_scores,
            reasoning=explanation_obj.summary,
            query_embedding=query_embedding,
            large_category=hierarchy.large_category,
            middle_category=hierarchy.middle_category,
            small_category=hierarchy.small_category,
            large_confidence=hierarchy.large_confidence,
            middle_confidence=hierarchy.middle_confidence,
            small_confidence=hierarchy.small_confidence,
            source_scores={
                "rule": rule_scores,
                "embedding": embedding_scores,
                "feedback": feedback_scores,
                "duplicate": duplicate_scores,
                "metadata": metadata_scores,
                "filename": filename_scores,
            },
            evidence_snippets=explanation_obj.evidence_snippets,
            metadata_signals=explanation_obj.metadata_signals,
            classifier_contributions=explanation_obj.classifier_contributions,
            explanation=explanation_obj.to_dict(),
            processing_profile=processing_profile,
        )

    def persist_classification(self, file_id: int, result: ClassificationResult) -> int:
        return self.repository.insert_classification(
            file_id=file_id,
            predicted_category=result.predicted_category,
            rule_score=result.rule_score,
            embedding_score=result.embedding_score,
            llm_score=result.llm_score,
            final_score=result.final_score,
            candidate_scores_json=json.dumps(result.candidate_scores, ensure_ascii=False),
            reasoning=result.reasoning,
            status="suggested",
            large_category=result.large_category,
            middle_category=result.middle_category or result.predicted_category,
            small_category=result.small_category,
            large_confidence=result.large_confidence,
            middle_confidence=result.middle_confidence,
            small_confidence=result.small_confidence,
            source_scores_json=json.dumps(result.source_scores, ensure_ascii=False),
            explanation_json=json.dumps(result.explanation, ensure_ascii=False),
            evidence_json=json.dumps(
                {
                    "evidence_snippets": result.evidence_snippets,
                    "metadata_signals": result.metadata_signals,
                    "ocr_used": result.ocr_used,
                    "llm_used": result.llm_used,
                },
                ensure_ascii=False,
            ),
            performance_json=json.dumps(result.processing_profile, ensure_ascii=False),
            classifier_version=CLASSIFIER_VERSION,
            config_version=CLASSIFIER_VERSION,
        )

    def suggest_rules(self, min_occurrences: int = 2) -> list[dict[str, Any]]:
        return build_rule_suggestions(self.repository, min_occurrences=min_occurrences)

    def _build_candidate_scores(
        self,
        categories: list[str],
        rule_scores: dict[str, float],
        embedding_scores: dict[str, float],
        feedback_scores: dict[str, float],
        duplicate_scores: dict[str, float],
        metadata_scores: dict[str, float],
        filename_scores: dict[str, float],
    ) -> dict[str, float]:
        candidate_scores: dict[str, float] = {}
        for category in categories:
            candidate_scores[category] = round(
                (rule_scores.get(category, 0.0) * 0.45)
                + (embedding_scores.get(category, 0.0) * 0.2)
                + (feedback_scores.get(category, 0.0) * 0.1)
                + (duplicate_scores.get(category, 0.0) * 0.05)
                + (metadata_scores.get(category, 0.0) * 0.1)
                + (filename_scores.get(category, 0.0) * 0.1),
                4,
            )
        return candidate_scores

    def _normalize_scores(self, raw_scores: dict[str, float], categories: list[str]) -> dict[str, float]:
        strong_rule_score = 4.0
        return {
            category: round(min(raw_scores.get(category, 0.0) / strong_rule_score, 1.0), 4)
            for category in categories
        }

    def _pick_top_category(self, scores: dict[str, float], categories: list[str]) -> str:
        ranked = sorted(categories, key=lambda category: (-scores.get(category, 0.0), category))
        return ranked[0] if ranked else UNCATEGORIZED

    def _build_filename_scores(self, file_name: str | None, categories: list[str]) -> dict[str, float]:
        scores = {category: 0.0 for category in categories}
        if not file_name or self.taxonomy is None:
            return scores
        lowered = file_name.lower()
        for entry in self.taxonomy.entries:
            if entry.middle_category in scores and any(alias.lower() in lowered for alias in entry.aliases + [entry.flat_label]):
                scores[entry.middle_category] = 0.8
        return scores

    def _build_metadata_scores(self, file_name: str | None, text: str, categories: list[str]) -> dict[str, float]:
        scores = {category: 0.0 for category in categories}
        combined = f"{file_name or ''} {text}".lower()
        extension = ""
        if file_name and "." in file_name:
            extension = file_name.rsplit(".", 1)[-1].lower()
        if extension == "xlsx" and "데이터" in scores:
            scores["데이터"] = 0.35
        if extension == "pptx" and "발표자료" in scores:
            scores["발표자료"] = 0.35
        if "invoice" in combined and "청구서" in scores:
            scores["청구서"] = max(scores["청구서"], 0.4)
        return scores

    def _needs_review(
        self,
        candidate_scores: dict[str, float],
        confidence: float,
        top_rule_category: str,
        embedding_scores: dict[str, float],
    ) -> bool:
        ranked = sorted(candidate_scores.values(), reverse=True)
        margin = ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0]
        top_embedding_category = self._pick_top_category(embedding_scores, list(candidate_scores))
        has_conflict = bool(embedding_scores) and top_embedding_category != top_rule_category and max(embedding_scores.values(), default=0.0) > 0
        return confidence < self.review_threshold or margin < 0.08 or has_conflict

    def _resolve_hierarchy(self, predicted_category: str, confidence: float) -> HierarchyPrediction:
        if self.taxonomy is None:
            return HierarchyPrediction(
                large_category="miscellaneous",
                middle_category=predicted_category,
                large_confidence=confidence,
                middle_confidence=confidence,
            )
        entry = self.taxonomy.resolve(predicted_category)
        return HierarchyPrediction(
            large_category=entry.large_category,
            middle_category=entry.middle_category,
            small_category=entry.small_category,
            large_confidence=confidence,
            middle_confidence=confidence,
            small_confidence=confidence if entry.small_category else 0.0,
        )

    def _build_explanation(
        self,
        predicted_category: str,
        hierarchy: HierarchyPrediction,
        rule_breakdown: dict[str, Any],
        embedding_breakdown: dict[str, Any],
        feedback_scores: dict[str, float],
        duplicate_scores: dict[str, float],
        metadata_scores: dict[str, float],
        filename_scores: dict[str, float],
        embedding_used: bool,
        text: str,
    ) -> ClassificationExplanation:
        matched = rule_breakdown["matches"].get(predicted_category, [])
        top_example = embedding_breakdown["top_examples"].get(predicted_category)
        summary_parts = [
            f"recommend={hierarchy.large_category}/{hierarchy.middle_category}",
            f"rules={', '.join(matched[:5]) if matched else 'none'}",
            f"embedding={'used' if embedding_used else 'skipped'}",
        ]
        if hierarchy.small_category:
            summary_parts[0] += f"/{hierarchy.small_category}"
        if top_example:
            summary_parts.append(
                f"similar={top_example['file_name']} (category={top_example['category']}, score={top_example['similarity']:.3f})"
            )
        feedback_value = feedback_scores.get(predicted_category, 0.0)
        if feedback_value > 0:
            summary_parts.append(f"feedback={feedback_value:.3f}")
        duplicate_value = duplicate_scores.get(predicted_category, 0.0)
        if duplicate_value > 0:
            summary_parts.append(f"duplicate={duplicate_value:.3f}")
        explanation = ClassificationExplanation(
            summary=" | ".join(summary_parts),
            matched_rules=matched[:10],
            source_scores={
                "embedding": embedding_breakdown["scores"].get(predicted_category, 0.0),
                "feedback": feedback_value,
                "duplicate": duplicate_value,
                "metadata": metadata_scores.get(predicted_category, 0.0),
                "filename": filename_scores.get(predicted_category, 0.0),
            },
            evidence_snippets=[text[:240], text[len(text) // 2 : (len(text) // 2) + 240], text[-240:]],
            metadata_signals={
                "large_category": hierarchy.large_category,
                "middle_category": hierarchy.middle_category,
            },
            classifier_contributions={
                "rules": 0.45,
                "embedding": 0.2,
                "metadata": 0.1,
                "filename": 0.1,
                "feedback": 0.1,
                "duplicate": 0.05,
            },
        )
        return explanation
