"""CPU-first in-memory document type classifier."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.hash_utils import compute_raw_text_hash

TYPE_CLASSIFIER_VERSION = "2.1"


@dataclass
class TypePrediction:
    predicted_type: str
    confidence: float
    candidate_scores: dict[str, float] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    available: bool = False


class TypeClassifier:
    """Train a lightweight sklearn classifier from confirmed feedback at runtime."""

    def __init__(
        self,
        *,
        version: str = TYPE_CLASSIFIER_VERSION,
        min_examples: int = 4,
        filename_weight: float = 2.0,
    ) -> None:
        self.version = version
        self.min_examples = min_examples
        self.filename_weight = filename_weight

    def predict(
        self,
        *,
        training_rows: list[dict[str, Any]],
        file_name: str,
        body_text: str,
        structural_features: dict[str, Any],
        fallback_type: str,
    ) -> TypePrediction:
        prepared = self._prepare_training_rows(training_rows)
        labels = sorted({row["label"] for row in prepared})
        if len(prepared) < self.min_examples or len(labels) < 2:
            return TypePrediction(
                predicted_type=fallback_type,
                confidence=0.0,
                available=False,
                evidence={
                    "status": "unavailable",
                    "reason": "not_enough_training_data",
                    "training_count": len(prepared),
                    "label_count": len(labels),
                },
            )

        try:
            import numpy as np
            from scipy.sparse import csr_matrix, hstack
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import LabelEncoder
        except Exception as error:
            return TypePrediction(
                predicted_type=fallback_type,
                confidence=0.0,
                available=False,
                evidence={"status": "unavailable", "reason": f"sklearn_unavailable: {error}"},
            )

        train_filenames = [row["file_name"] for row in prepared]
        train_bodies = [row["body_text"] for row in prepared]
        train_structural = [row["structural_features"] for row in prepared]
        target = [row["label"] for row in prepared]
        real_count = sum(1 for row in prepared if row.get("source") != "category_profile")
        sample_weights = self._sample_weights(prepared, real_count=real_count)

        try:
            filename_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1)
            word_vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, max_features=4000)
            char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=3000)
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(target)
            x_filename = filename_vectorizer.fit_transform(train_filenames) * self.filename_weight
            x_word = word_vectorizer.fit_transform(train_bodies)
            x_char = char_vectorizer.fit_transform(train_bodies)
            x_structural = csr_matrix([self._structural_vector(item) for item in train_structural])
            x_train = hstack([x_filename, x_word, x_char, x_structural], format="csr")

            classifier = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )
            classifier.fit(x_train, y, sample_weight=sample_weights)

            x_query = hstack(
                [
                    filename_vectorizer.transform([file_name]) * self.filename_weight,
                    word_vectorizer.transform([body_text]),
                    char_vectorizer.transform([body_text]),
                    csr_matrix([self._structural_vector(structural_features)]),
                ],
                format="csr",
            )
            probabilities = classifier.predict_proba(x_query)[0]
            classes = label_encoder.inverse_transform(np.arange(len(probabilities)))
            scores = {str(label): round(float(score), 4) for label, score in zip(classes, probabilities)}
            predicted = max(scores, key=scores.get)
            confidence = scores[predicted]
            return TypePrediction(
                predicted_type=predicted,
                confidence=confidence,
                candidate_scores=scores,
                available=True,
                evidence={
                    "status": "available",
                    "model": "LogisticRegression",
                    "version": self.version,
                    "training_count": len(prepared),
                    "real_training_count": real_count,
                    "synthetic_training_count": len(prepared) - real_count,
                    "training_signature": self._training_signature(prepared),
                    "labels": labels,
                    "filename_weight": self.filename_weight,
                },
            )
        except Exception as error:
            return TypePrediction(
                predicted_type=fallback_type,
                confidence=0.0,
                available=False,
                evidence={"status": "unavailable", "reason": f"training_failed: {error}"},
            )

    def _prepare_training_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for row in rows:
            label = str(row.get("label") or row.get("final_category") or row.get("category") or "").strip()
            if not label:
                continue
            file_name = str(row.get("file_name") or "")
            body_text = str(row.get("body_text") or row.get("extracted_text") or row.get("source_text") or "")
            structural_raw = row.get("structural_features") or row.get("structural_features_json") or {}
            layout_raw = row.get("layout_features") or row.get("layout_features_json") or {}
            if isinstance(structural_raw, str):
                try:
                    structural = json.loads(structural_raw)
                except json.JSONDecodeError:
                    structural = {}
            elif isinstance(structural_raw, dict):
                structural = structural_raw
            else:
                structural = {}
            if isinstance(layout_raw, str):
                try:
                    layout = json.loads(layout_raw)
                except json.JSONDecodeError:
                    layout = {}
            elif isinstance(layout_raw, dict):
                layout = layout_raw
            else:
                layout = {}
            structural.update({f"layout_{key}": value for key, value in layout.items()})
            key = (label, file_name, body_text[:120])
            if key in seen:
                continue
            seen.add(key)
            prepared.append(
                {
                    "label": label,
                    "file_name": file_name,
                    "body_text": body_text,
                    "structural_features": structural,
                    "source": str(row.get("source") or "real"),
                    "sample_weight": float(row.get("sample_weight", 1.0) or 1.0),
                    "source_id": row.get("source_id"),
                }
            )
        return prepared

    def _sample_weights(self, rows: list[dict[str, Any]], *, real_count: int) -> list[float]:
        weights: list[float] = []
        synthetic_scale = 0.35 if real_count >= self.min_examples else 1.0
        for row in rows:
            weight = float(row.get("sample_weight", 1.0) or 1.0)
            if row.get("source") == "category_profile":
                weight *= synthetic_scale
            weights.append(weight)
        return weights

    def _training_signature(self, rows: list[dict[str, Any]]) -> str:
        payload = [
            {
                "label": row.get("label"),
                "source": row.get("source"),
                "source_id": row.get("source_id"),
                "weight": row.get("sample_weight"),
                "text_hash": compute_raw_text_hash(str(row.get("body_text", ""))),
            }
            for row in rows
        ]
        return compute_raw_text_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    def _structural_vector(self, features: dict[str, Any]) -> list[float]:
        keys = [
            "page_count",
            "slide_count",
            "sheet_count",
            "table_count",
            "image_count",
            "bullet_ratio",
            "citation_count",
            "has_abstract",
            "has_references",
            "has_doi",
            "contract_terms_count",
            "receipt_terms_count",
            "average_sentence_length",
            "token_count",
            "layout_text_density",
            "layout_whitespace_ratio",
            "layout_receipt_pattern_score",
            "layout_certificate_pattern_score",
            "layout_slide_like_layout_score",
            "layout_dense_text_score",
            "layout_two_column_score",
            "layout_image_area_ratio",
            "layout_numeric_line_density",
            "layout_bullet_density",
            "layout_header_block_score",
            "layout_footer_pattern_score",
            "layout_signature_area_score",
            "layout_chart_presence_score",
            "layout_section_divider_score",
            "layout_numeric_column_score",
            "layout_approval_block_score",
            "layout_repeated_line_pattern_score",
            "clause_pattern_score",
            "legal_term_density",
            "research_structure_score",
            "report_structure_score",
            "contact_pattern_score",
            "heading_density",
            "ocr_text_length",
            "unreadable_ratio",
            "symbol_noise_ratio",
            "low_quality_scan_score",
        ]
        vector: list[float] = []
        for key in keys:
            value = features.get(key, 0)
            if isinstance(value, bool):
                vector.append(1.0 if value else 0.0)
            else:
                try:
                    vector.append(float(value))
                except (TypeError, ValueError):
                    vector.append(0.0)
        return vector
