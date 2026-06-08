"""Hybrid hierarchical classifier with explainable score breakdowns."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from src.confidence import ConfidencePolicy
from src.document_patterns import build_evidence_groups
from src.document_features import DocumentFeatureBundle, DocumentFeatureExtractor
from src.feedback import build_rule_suggestions
from src.lexical_features import (
    build_category_profiles_from_rows,
    compute_lexical_scores,
    flatten_lexical_scores,
)
from src.models import ClassificationExplanation, HierarchyPrediction
from src.performance import build_file_latency_analysis, normalize_stage_timings
from src.rule_classifier import RuleBasedClassifier, build_rule_input_text
from src.storage import ClassificationRepository
from src.taxonomy import Taxonomy, UNCATEGORIZED
from src.type_classifier import TypeClassifier, TypePrediction
from src.vectorizer import SentenceTransformerEmbedder


CLASSIFIER_VERSION = "2.0"

BASE_SCORE_WEIGHTS = {
    "rule": 0.25,
    "embedding": 0.25,
    "lexical": 0.25,
    "feedback": 0.10,
    "layout": 0.07,
    "filename": 0.05,
    "metadata": 0.02,
    "duplicate": 0.01,
}


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
    predicted_type: str = ""
    type_confidence: float = 0.0
    review_reasons: list[str] = field(default_factory=list)
    suggested_tags: list[dict[str, Any]] = field(default_factory=list)
    rule_evidence: dict[str, Any] = field(default_factory=dict)
    ml_evidence: dict[str, Any] = field(default_factory=dict)
    semantic_evidence: list[dict[str, Any]] = field(default_factory=list)
    layout_evidence: list[dict[str, Any]] = field(default_factory=list)
    structure_evidence: list[dict[str, Any]] = field(default_factory=list)
    ocr_evidence: list[dict[str, Any]] = field(default_factory=list)
    cluster_candidate_id: int | None = None
    lexical_score: float = 0.0
    layout_score: float = 0.0
    lexical_evidence: dict[str, Any] = field(default_factory=dict)
    score_breakdown: dict[str, Any] = field(default_factory=dict)


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
        feature_extractor: DocumentFeatureExtractor | None = None,
        type_classifier: TypeClassifier | None = None,
        confidence_policy: ConfidencePolicy | None = None,
        ml_enabled: bool = False,
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
        self.feature_extractor = feature_extractor or DocumentFeatureExtractor()
        self.type_classifier = type_classifier or TypeClassifier()
        self.confidence_policy = confidence_policy or ConfidencePolicy(threshold=review_threshold)
        self.ml_enabled = ml_enabled

    def classify_file(
        self,
        file_id: int,
        file_hash: str,
        text: str,
        duplicate_of_file_id: int | None,
        file_name: str | None = None,
        document_features: DocumentFeatureBundle | dict[str, Any] | None = None,
    ) -> ClassificationResult:
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
            file_id=file_id,
            document_features=document_features,
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
        file_id: int | None = None,
        document_features: DocumentFeatureBundle | dict[str, Any] | None = None,
        initial_stage_timings: dict[str, float] | None = None,
    ) -> ClassificationResult:
        classify_start = time.perf_counter()
        if categories is None:
            categories = self.repository.list_categories()
        if not categories:
            raise ValueError("No categories are available.")

        normalized_text = self.rule_classifier.normalize_text(text)
        feature_start = time.perf_counter()
        feature_bundle = self._coerce_or_extract_features(
            document_features=document_features,
            file_name=file_name or "",
            text=normalized_text,
            file_hash=file_hash,
        )
        if file_id is not None:
            self._persist_features(file_id=file_id, file_hash=file_hash, feature_bundle=feature_bundle)
        feature_time = time.perf_counter() - feature_start
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
                        feature_bundle.compressed_text,
                        repository=self.repository,
                        file_hash=file_hash,
                        text_kind="compressed_query",
                        embedding_version="2.1-compressed",
                    )
                    embedding_meta = self.embedder.get_last_encode_meta()
                else:
                    query_embedding = self.embedder.encode(feature_bundle.compressed_text)
                    embedding_meta = {"cache_hit": None, "elapsed": 0.0, "model_name": "custom"}
            embedding_time = time.perf_counter() - embedding_start
            embedding_breakdown = self.embedder.score_against_examples(
                query_embedding=query_embedding,
                examples=confirmed_examples,
                categories=categories,
            )
            if file_id is not None and query_embedding:
                self.repository.upsert_document_vector(
                    file_id=file_id,
                    vector_type="compressed_embedding",
                    vector_key=str(embedding_meta.get("cache_key", "")),
                    vector_json="",
                    model_version=str(embedding_meta.get("model_name", "embedding")),
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
        active_profiles = self.repository.list_category_profiles(include_inactive=False)
        category_profiles = build_category_profiles_from_rows(active_profiles, categories)
        lexical_results = compute_lexical_scores(feature_bundle.compressed_text or normalized_text, category_profiles)
        lexical_scores = flatten_lexical_scores(lexical_results, categories)
        layout_scores = self._build_layout_scores(categories=categories, profiles=category_profiles, feature_bundle=feature_bundle)
        adjustment_time = time.perf_counter() - adjustment_start
        combine_start = time.perf_counter()
        score_weights = self._build_score_weights(
            file_name=file_name,
            feature_bundle=feature_bundle,
            top_rule_score=top_rule_score,
            top_rule_match_count=top_rule_match_count,
            feedback_scores=feedback_scores,
        )
        candidate_scores = self._build_candidate_scores(
            categories=categories,
            rule_scores=rule_scores,
            embedding_scores=embedding_scores,
            lexical_scores=lexical_scores,
            feedback_scores=feedback_scores,
            duplicate_scores=duplicate_scores,
            metadata_scores=metadata_scores,
            filename_scores=filename_scores,
            layout_scores=layout_scores,
            weights=score_weights,
        )
        combine_time = time.perf_counter() - combine_start

        weak_unverified_match = top_rule_score < self.low_rule_confidence_threshold and not should_use_embedding
        if (
            (top_rule_match_count == 0 or weak_unverified_match)
            and max(embedding_scores.values(), default=0.0) <= 0
            and max(lexical_scores.values(), default=0.0) <= 0
        ):
            predicted_category = UNCATEGORIZED
            candidate_scores[predicted_category] = 0.0
        else:
            predicted_category = self._pick_top_category(candidate_scores, categories)

        confidence = max(candidate_scores.get(predicted_category, 0.0), rule_scores.get(predicted_category, 0.0))
        type_prediction = TypePrediction(
            predicted_type="",
            confidence=0.0,
            available=False,
            evidence={
                "status": "disabled",
                "reason": "ml_disabled_by_config",
                "replacement": "unsupervised_batch_pipeline",
            },
        )
        top_embedding_category = self._pick_top_category(embedding_scores, categories)
        embedding_available = max(embedding_scores.values(), default=0.0) > 0
        confidence_decision = self.confidence_policy.evaluate(
            confidence=type_prediction.confidence if type_prediction.available else confidence,
            candidate_scores=type_prediction.candidate_scores if type_prediction.available else candidate_scores,
            rule_prediction=top_rule_category,
            ml_prediction=type_prediction.predicted_type,
            ml_available=type_prediction.available,
            embedding_prediction=top_embedding_category,
            embedding_available=embedding_available,
        )
        legacy_review_required = self._needs_review(candidate_scores, confidence, top_rule_category, embedding_scores)
        review_required = legacy_review_required or confidence_decision.review_required
        review_reasons = list(confidence_decision.review_reasons)
        if legacy_review_required and "legacy_ambiguity" not in review_reasons:
            review_reasons.append("legacy_ambiguity")
        layout_review_reason = self._layout_conflict_reason(
            predicted_type=type_prediction.predicted_type,
            layout_features=feature_bundle.layout_features,
        )
        if type_prediction.available and layout_review_reason and layout_review_reason not in review_reasons:
            review_reasons.append(layout_review_reason)
            review_required = True
        hierarchy = self._resolve_hierarchy(predicted_category, confidence)
        suggested_tags = self._suggest_tags(feature_bundle, predicted_type=type_prediction.predicted_type)
        evidence_groups = build_evidence_groups(
            predicted_type=type_prediction.predicted_type or predicted_category,
            text=feature_bundle.compressed_text,
            structural_features=feature_bundle.structural_features,
            layout_features=feature_bundle.layout_features,
            text_stats=feature_bundle.text_stats,
        )
        explanation_start = time.perf_counter()
        explanation_obj = self._build_explanation(
            predicted_category=predicted_category,
            hierarchy=hierarchy,
            rule_breakdown=rule_breakdown,
            embedding_breakdown=embedding_breakdown,
            lexical_results=lexical_results,
            lexical_scores=lexical_scores,
            feedback_scores=feedback_scores,
            duplicate_scores=duplicate_scores,
            metadata_scores=metadata_scores,
            filename_scores=filename_scores,
            layout_scores=layout_scores,
            score_weights=score_weights,
            embedding_used=should_use_embedding,
            text=normalized_text,
        )
        explanation_time = time.perf_counter() - explanation_start

        predicted_middle_category = hierarchy.middle_category or predicted_category
        stage_timings = normalize_stage_timings(initial_stage_timings)
        stage_timings.update(
            {
                "confirmed_example_lookup": example_lookup_time,
                "feature_extraction": feature_time,
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
            "score_weights": score_weights,
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
            final_score=candidate_scores.get(predicted_middle_category, confidence),
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
                "lexical": lexical_scores,
                "feedback": feedback_scores,
                "duplicate": duplicate_scores,
                "metadata": metadata_scores,
                "filename": filename_scores,
                "layout": layout_scores,
            },
            evidence_snippets=explanation_obj.evidence_snippets,
            metadata_signals=explanation_obj.metadata_signals,
            classifier_contributions=explanation_obj.classifier_contributions,
            explanation=explanation_obj.to_dict(),
            processing_profile=processing_profile,
            predicted_type=type_prediction.predicted_type,
            type_confidence=type_prediction.confidence,
            review_reasons=review_reasons,
            suggested_tags=suggested_tags,
            rule_evidence={
                "prediction": top_rule_category,
                "score": top_rule_score,
                "matches": rule_breakdown["matches"].get(top_rule_category, []),
                "scores": rule_scores,
            },
            ml_evidence=type_prediction.evidence | {"candidate_scores": type_prediction.candidate_scores},
            semantic_evidence=evidence_groups.get("semantic", []),
            layout_evidence=evidence_groups.get("layout", []),
            structure_evidence=evidence_groups.get("structure", []),
            ocr_evidence=evidence_groups.get("ocr", []),
            lexical_score=lexical_scores.get(predicted_middle_category, 0.0),
            layout_score=layout_scores.get(predicted_middle_category, 0.0),
            lexical_evidence=lexical_results.get(predicted_middle_category, {}),
            score_breakdown={
                "weights": score_weights,
                "scores": {
                    "rule": rule_scores.get(predicted_middle_category, 0.0),
                    "embedding": embedding_scores.get(predicted_middle_category, 0.0),
                    "lexical": lexical_scores.get(predicted_middle_category, 0.0),
                    "feedback": feedback_scores.get(predicted_middle_category, 0.0),
                    "layout": layout_scores.get(predicted_middle_category, 0.0),
                    "filename": filename_scores.get(predicted_middle_category, 0.0),
                    "metadata": metadata_scores.get(predicted_middle_category, 0.0),
                    "duplicate": duplicate_scores.get(predicted_middle_category, 0.0),
                },
            },
        )

    def persist_classification(self, file_id: int, result: ClassificationResult) -> int:
        classification_id = self.repository.insert_classification(
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
                    "lexical_evidence": result.lexical_evidence,
                    "score_breakdown": result.score_breakdown,
                    "ocr_used": result.ocr_used,
                    "llm_used": result.llm_used,
                },
                ensure_ascii=False,
            ),
            performance_json=json.dumps(result.processing_profile, ensure_ascii=False),
            classifier_version=CLASSIFIER_VERSION,
            config_version=CLASSIFIER_VERSION,
            predicted_type=result.predicted_type,
            type_confidence=result.type_confidence,
            review_reasons_json=json.dumps(result.review_reasons, ensure_ascii=False),
            suggested_tags_json=json.dumps(result.suggested_tags, ensure_ascii=False),
            cluster_candidate_id=result.cluster_candidate_id,
            ml_evidence_json=json.dumps(result.ml_evidence, ensure_ascii=False),
            rule_evidence_json=json.dumps(result.rule_evidence, ensure_ascii=False),
            lexical_score=result.lexical_score,
            layout_score=result.layout_score,
        )
        for item in result.suggested_tags:
            tag = str(item.get("tag", "")).strip()
            if not tag or tag.startswith("type:"):
                continue
            self.repository.upsert_document_tag(
                file_id=file_id,
                tag=tag,
                confidence=float(item.get("confidence", 0.0)),
                source=str(item.get("source", "classifier")),
            )
        return classification_id

    def suggest_rules(self, min_occurrences: int = 2) -> list[dict[str, Any]]:
        return build_rule_suggestions(self.repository, min_occurrences=min_occurrences)

    def _coerce_or_extract_features(
        self,
        *,
        document_features: DocumentFeatureBundle | dict[str, Any] | None,
        file_name: str,
        text: str,
        file_hash: str = "",
    ) -> DocumentFeatureBundle:
        if isinstance(document_features, DocumentFeatureBundle):
            return document_features
        if isinstance(document_features, dict) and "compressed_text" in document_features:
                return DocumentFeatureBundle(
                    feature_version=str(document_features.get("feature_version") or self.feature_extractor.version),
                    filename_features=dict(document_features.get("filename_features") or {}),
                    metadata_features=dict(document_features.get("metadata_features") or {}),
                    structural_features=dict(document_features.get("structural_features") or {}),
                    layout_features=dict(document_features.get("layout_features") or {}),
                    text_stats=dict(document_features.get("text_stats") or {}),
                    compressed_text=str(document_features.get("compressed_text") or text),
                compressed_text_hash=str(document_features.get("compressed_text_hash") or ""),
            )
        if file_hash:
            cached = self.repository.get_document_features_by_hash(file_hash, self.feature_extractor.version)
            if cached is not None:
                return DocumentFeatureBundle(
                    feature_version=str(cached["extractor_version"]),
                    filename_features=json.loads(str(cached["filename_features_json"])),
                    metadata_features=json.loads(str(cached["metadata_features_json"])),
                    structural_features=json.loads(str(cached["structural_features_json"])),
                    layout_features=json.loads(str(cached["layout_features_json"])),
                    text_stats=json.loads(str(cached["text_stats_json"])),
                    compressed_text=str(cached["compressed_text"]),
                    compressed_text_hash=str(cached["compressed_text_hash"]),
                )
        return self.feature_extractor.extract(
            file_name=file_name,
            file_ext=f".{file_name.rsplit('.', 1)[-1].lower()}" if "." in file_name else "",
            text=text,
        )

    def _persist_features(self, *, file_id: int, file_hash: str, feature_bundle: DocumentFeatureBundle) -> None:
        try:
            self.repository.upsert_document_features(
                file_id=file_id,
                file_hash=file_hash,
                extractor_version=feature_bundle.feature_version,
                filename_features=feature_bundle.filename_features,
                metadata_features=feature_bundle.metadata_features,
            structural_features=feature_bundle.structural_features,
            layout_features=feature_bundle.layout_features,
            text_stats=feature_bundle.text_stats,
                compressed_text=feature_bundle.compressed_text,
                compressed_text_hash=feature_bundle.compressed_text_hash,
            )
        except Exception:
            # Some unit paths classify without first inserting a files row.
            return

    def _suggest_tags(self, feature_bundle: DocumentFeatureBundle, *, predicted_type: str) -> list[dict[str, Any]]:
        tags: list[dict[str, Any]] = []
        text = " ".join(
            [
                str(feature_bundle.filename_features.get("normalized_stem", "")),
                feature_bundle.compressed_text,
            ]
        ).lower()
        tag_rules = {
            "AI": ("ai", "인공지능", "machine learning", "deep learning", "transformer"),
            "의료": ("medical", "mri", "의료", "병원", "진단"),
            "컴퓨터비전": ("vision", "image", "segmentation", "detection", "컴퓨터비전"),
            "수업자료": ("수업", "강의", "lecture", "캡스톤", "과제"),
        }
        for tag, keywords in tag_rules.items():
            if any(keyword in text for keyword in keywords):
                tags.append({"tag": tag, "confidence": 0.75, "source": "feature_rule"})
        if predicted_type:
            tags.append({"tag": f"type:{predicted_type}", "confidence": 0.5, "source": "type_hint"})
        layout_scores = {
            "영수증형": feature_bundle.layout_features.get("receipt_pattern_score", 0.0),
            "증명서형": feature_bundle.layout_features.get("certificate_pattern_score", 0.0),
            "발표자료형": feature_bundle.layout_features.get("slide_like_layout_score", 0.0),
            "논문형": feature_bundle.layout_features.get("dense_text_score", 0.0),
        }
        for tag, score in layout_scores.items():
            try:
                value = float(score)
            except (TypeError, ValueError):
                continue
            if value >= 0.65:
                tags.append({"tag": tag, "confidence": round(value, 4), "source": "layout"})
        return tags[:8]

    def _layout_conflict_reason(self, *, predicted_type: str, layout_features: dict[str, Any]) -> str:
        if not predicted_type:
            return ""
        layout_scores = {
            "영수증": float(layout_features.get("receipt_pattern_score", 0.0) or 0.0),
            "증명서": float(layout_features.get("certificate_pattern_score", 0.0) or 0.0),
            "발표자료": float(layout_features.get("slide_like_layout_score", 0.0) or 0.0),
            "논문": float(layout_features.get("dense_text_score", 0.0) or 0.0),
        }
        top_layout_type, top_score = max(layout_scores.items(), key=lambda item: item[1])
        if top_score < 0.75:
            return ""
        if top_layout_type not in predicted_type:
            return f"layout_{top_layout_type}_conflict"
        return ""

    def _build_candidate_scores(
        self,
        categories: list[str],
        rule_scores: dict[str, float],
        embedding_scores: dict[str, float],
        lexical_scores: dict[str, float],
        feedback_scores: dict[str, float],
        duplicate_scores: dict[str, float],
        metadata_scores: dict[str, float],
        filename_scores: dict[str, float],
        layout_scores: dict[str, float],
        weights: dict[str, float],
    ) -> dict[str, float]:
        candidate_scores: dict[str, float] = {}
        for category in categories:
            candidate_scores[category] = round(
                (rule_scores.get(category, 0.0) * weights.get("rule", 0.0))
                + (embedding_scores.get(category, 0.0) * weights.get("embedding", 0.0))
                + (lexical_scores.get(category, 0.0) * weights.get("lexical", 0.0))
                + (feedback_scores.get(category, 0.0) * weights.get("feedback", 0.0))
                + (layout_scores.get(category, 0.0) * weights.get("layout", 0.0))
                + (filename_scores.get(category, 0.0) * weights.get("filename", 0.0))
                + (metadata_scores.get(category, 0.0) * weights.get("metadata", 0.0))
                + (duplicate_scores.get(category, 0.0) * weights.get("duplicate", 0.0)),
                4,
            )
        return candidate_scores

    def _build_score_weights(
        self,
        *,
        file_name: str | None,
        feature_bundle: DocumentFeatureBundle,
        top_rule_score: float,
        top_rule_match_count: int,
        feedback_scores: dict[str, float],
        renormalize: bool = True,
    ) -> dict[str, float]:
        weights = dict(BASE_SCORE_WEIGHTS)
        quality = self._text_quality_factor(feature_bundle.text_stats)
        if quality < 1.0:
            weights["lexical"] *= quality
            weights["embedding"] *= max(0.35, quality)
        if self._is_generic_filename(file_name):
            weights["filename"] = 0.0
        if top_rule_score >= self.rule_skip_embedding_threshold and top_rule_match_count >= self.min_rule_matches_for_skip:
            weights["rule"] *= 1.15
        if max(feedback_scores.values(), default=0.0) < 0.34:
            weights["feedback"] *= 0.6
        weights["duplicate"] = min(weights.get("duplicate", 0.0), 0.01)
        if renormalize:
            total = sum(weights.values())
            if total > 0:
                weights = {key: round(value / total, 4) for key, value in weights.items()}
        return weights

    def _text_quality_factor(self, text_stats: dict[str, Any]) -> float:
        char_count = float(text_stats.get("char_count", text_stats.get("ocr_text_length", 0)) or 0)
        low_quality_scan_score = float(text_stats.get("low_quality_scan_score", 0.0) or 0.0)
        unreadable_ratio = float(text_stats.get("unreadable_ratio", 0.0) or 0.0)
        length_factor = min(1.0, max(0.2, char_count / 600.0))
        noise_factor = max(0.2, 1.0 - max(low_quality_scan_score, unreadable_ratio * 2.0))
        return round(max(0.15, min(1.0, length_factor * noise_factor)), 4)

    def _build_layout_scores(
        self,
        *,
        categories: list[str],
        profiles: dict[str, dict[str, Any]],
        feature_bundle: DocumentFeatureBundle,
    ) -> dict[str, float]:
        combined_features = {
            **feature_bundle.layout_features,
            **feature_bundle.structural_features,
            **feature_bundle.text_stats,
        }
        scores = {category: 0.0 for category in categories}
        for category in categories:
            profile = profiles.get(category, {})
            signals = profile.get("profile_signals") if isinstance(profile, dict) else {}
            core_features = signals.get("core_features", []) if isinstance(signals, dict) else []
            values = [
                float(combined_features.get(feature, 0.0) or 0.0)
                for feature in core_features
                if feature in combined_features and isinstance(combined_features.get(feature), (int, float))
            ]
            if values:
                scores[category] = round(min(1.0, sum(values) / len(values)), 4)
        return scores

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
        if not file_name or self.taxonomy is None or self._is_generic_filename(file_name):
            return scores
        lowered = file_name.lower()
        for entry in self.taxonomy.entries:
            if entry.middle_category in scores and any(alias.lower() in lowered for alias in entry.aliases + [entry.flat_label]):
                scores[entry.middle_category] = 0.8
        return scores

    def _is_generic_filename(self, file_name: str | None) -> bool:
        if not file_name:
            return True
        stem = file_name.rsplit(".", 1)[0].lower().strip()
        return bool(
            re.fullmatch(r"(scan|image|img|document|doc|file|page)[_\-\s]?\d*", stem)
            or re.fullmatch(r"kakaotalk[_\-\s].*", stem)
            or stem in {"untitled", "new document", "새 문서"}
        )

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
        lexical_results: dict[str, dict[str, Any]],
        lexical_scores: dict[str, float],
        feedback_scores: dict[str, float],
        duplicate_scores: dict[str, float],
        metadata_scores: dict[str, float],
        filename_scores: dict[str, float],
        layout_scores: dict[str, float],
        score_weights: dict[str, float],
        embedding_used: bool,
        text: str,
    ) -> ClassificationExplanation:
        matched = rule_breakdown["matches"].get(predicted_category, [])
        top_example = embedding_breakdown["top_examples"].get(predicted_category)
        lexical_result = lexical_results.get(predicted_category, {})
        summary_parts = [
            f"recommend={hierarchy.large_category}/{hierarchy.middle_category}",
            f"rules={', '.join(matched[:5]) if matched else 'none'}",
            f"embedding={'used' if embedding_used else 'skipped'}",
            f"lexical={lexical_scores.get(predicted_category, 0.0):.3f}",
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
                "lexical": lexical_scores.get(predicted_category, 0.0),
                "tfidf": float(lexical_result.get("tfidf_score", 0.0) or 0.0),
                "ngram": float(lexical_result.get("ngram_score", 0.0) or 0.0),
                "bow": float(lexical_result.get("bow_score", 0.0) or 0.0),
                "feedback": feedback_value,
                "duplicate": duplicate_value,
                "layout": layout_scores.get(predicted_category, 0.0),
                "metadata": metadata_scores.get(predicted_category, 0.0),
                "filename": filename_scores.get(predicted_category, 0.0),
            },
            evidence_snippets=[text[:240], text[len(text) // 2 : (len(text) // 2) + 240], text[-240:]],
            metadata_signals={
                "large_category": hierarchy.large_category,
                "middle_category": hierarchy.middle_category,
                "top_lexical_terms": ", ".join(str(item) for item in lexical_result.get("top_terms", [])[:8]),
                "top_ngram_matches": ", ".join(str(item) for item in lexical_result.get("top_ngram_matches", [])[:8]),
            },
            classifier_contributions={
                "rules": score_weights.get("rule", 0.0),
                "embedding": score_weights.get("embedding", 0.0),
                "lexical": score_weights.get("lexical", 0.0),
                "feedback": score_weights.get("feedback", 0.0),
                "layout": score_weights.get("layout", 0.0),
                "filename": score_weights.get("filename", 0.0),
                "metadata": score_weights.get("metadata", 0.0),
                "duplicate": score_weights.get("duplicate", 0.0),
            },
        )
        return explanation
