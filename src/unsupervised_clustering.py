"""Batch unsupervised clustering hooks for unknown documents.

This module intentionally stays out of the real-time classifier path. It keeps
the offline discovery flow explicit: build/reuse embeddings first, then cluster
those vectors with HDBSCAN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class ClusterInput:
    item_id: int
    file_hash: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusterAssignment:
    item_id: int
    cluster_id: int
    score: float = 0.0


@dataclass
class ClusterRunResult:
    algorithm: str
    assignments: list[ClusterAssignment]
    representatives: dict[int, list[int]]
    metrics: dict[str, Any] = field(default_factory=dict)


class UnsupervisedClusterer(Protocol):
    algorithm: str

    def fit_predict(self, items: list[ClusterInput]) -> ClusterRunResult:
        """Cluster unknown items and return assignments. -1 means noise."""


class SklearnTextClusterer:
    """Offline clusterer: embeddings first, HDBSCAN second.

    A DBSCAN fallback is kept for environments where the optional `hdbscan`
    wheel is not installed yet, but it still runs after embedding generation.
    """

    def __init__(
        self,
        *,
        algorithm: str = "hdbscan",
        eps: float = 0.72,
        min_cluster_size: int | None = None,
        min_samples: int = 3,
        max_features: int = 3000,
        embedder: Any | None = None,
        repository: Any | None = None,
    ) -> None:
        self.algorithm = algorithm
        self.eps = eps
        self.min_cluster_size = min_cluster_size or min_samples
        self.min_samples = min_samples
        self.max_features = max_features
        self.embedder = embedder
        self.repository = repository

    def fit_predict(self, items: list[ClusterInput]) -> ClusterRunResult:
        if len(items) < self.min_cluster_size:
            return ClusterRunResult(
                algorithm=self.algorithm,
                assignments=[ClusterAssignment(item.item_id, -1, 0.0) for item in items],
                representatives={},
                metrics={"status": "not_enough_items", "item_count": len(items)},
            )

        embeddings, embedding_metrics = self._build_embeddings(items)
        if embeddings is None:
            return ClusterRunResult(
                algorithm=self.algorithm,
                assignments=[ClusterAssignment(item.item_id, -1, 0.0) for item in items],
                representatives={},
                metrics=embedding_metrics,
            )

        labels, strengths, cluster_metrics = self._cluster_embeddings(embeddings)
        if labels is None:
            return ClusterRunResult(
                algorithm=self.algorithm,
                assignments=[ClusterAssignment(item.item_id, -1, 0.0) for item in items],
                representatives={},
                metrics={**embedding_metrics, **cluster_metrics},
            )

        assignments = [
            ClusterAssignment(item.item_id, int(label), float(strength))
            for item, label, strength in zip(items, labels, strengths)
        ]
        representatives = extract_cluster_representatives(assignments, limit=5)
        clustered_count = sum(1 for assignment in assignments if assignment.cluster_id >= 0)
        return ClusterRunResult(
            algorithm=str(cluster_metrics.get("algorithm", self.algorithm)),
            assignments=assignments,
            representatives=representatives,
            metrics={
                **embedding_metrics,
                **cluster_metrics,
                "status": "ok",
                "item_count": len(items),
                "clustered_count": clustered_count,
                "noise_count": len(items) - clustered_count,
                "cluster_count": len(representatives),
                "min_cluster_size": self.min_cluster_size,
                "min_samples": self.min_samples,
            },
        )

    def _build_embeddings(self, items: list[ClusterInput]) -> tuple[np.ndarray | None, dict[str, Any]]:
        precomputed = [item.metadata.get("embedding") for item in items]
        if all(isinstance(value, list) and value for value in precomputed):
            try:
                matrix = np.asarray(precomputed, dtype=np.float32)
            except (TypeError, ValueError) as error:
                return None, {"status": "embedding_failed", "error": str(error)}
            return matrix, {"embedding_stage": "precomputed", "embedding_count": int(matrix.shape[0])}

        texts = [item.text for item in items]
        file_hashes = [item.file_hash for item in items]
        if self.embedder is not None:
            try:
                embeddings = self.embedder.encode_many(
                    texts,
                    repository=self.repository,
                    file_hashes=file_hashes,
                    text_kind="unknown_pool",
                    embedding_version="2.1-unknown-pool",
                )
                matrix = np.asarray(embeddings, dtype=np.float32)
            except Exception as error:
                return None, {"status": "embedding_failed", "error": str(error)}
            return matrix, {"embedding_stage": "sentence_transformer", "embedding_count": int(matrix.shape[0])}

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except Exception as error:
            return None, {"status": "embedding_unavailable", "error": str(error)}
        try:
            vectorizer = TfidfVectorizer(
                token_pattern=r"(?u)[가-힣A-Za-z0-9_]{1,}",
                ngram_range=(1, 2),
                analyzer="word",
                max_features=self.max_features,
            )
            matrix = vectorizer.fit_transform(texts).toarray().astype(np.float32)
        except Exception as error:
            return None, {"status": "embedding_failed", "error": str(error)}
        return matrix, {"embedding_stage": "tfidf_fallback", "embedding_count": int(matrix.shape[0])}

    def _cluster_embeddings(self, matrix: np.ndarray) -> tuple[list[int] | None, list[float], dict[str, Any]]:
        try:
            import hdbscan

            model = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric="euclidean",
            )
            labels = [int(label) for label in model.fit_predict(matrix)]
            probabilities = getattr(model, "probabilities_", np.ones(len(labels)))
            strengths = [float(value) for value in probabilities]
            return labels, strengths, {"algorithm": "hdbscan", "cluster_stage": "hdbscan"}
        except Exception as hdbscan_error:
            try:
                from sklearn.cluster import DBSCAN

                model = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="cosine")
                labels = [int(label) for label in model.fit_predict(matrix)]
            except Exception as error:
                return None, [], {
                    "status": "fit_failed",
                    "hdbscan_error": str(hdbscan_error),
                    "error": str(error),
                }

            strengths = [1.0 if label >= 0 else 0.0 for label in labels]
            return labels, strengths, {
                "algorithm": "dbscan_embedding_fallback",
                "cluster_stage": "dbscan_after_embedding",
                "hdbscan_error": str(hdbscan_error),
                "eps": self.eps,
            }


def extract_cluster_representatives(
    assignments: list[ClusterAssignment],
    *,
    limit: int = 5,
) -> dict[int, list[int]]:
    representatives: dict[int, list[int]] = {}
    for assignment in assignments:
        if assignment.cluster_id < 0:
            continue
        representatives.setdefault(assignment.cluster_id, [])
        if len(representatives[assignment.cluster_id]) < limit:
            representatives[assignment.cluster_id].append(assignment.item_id)
    return representatives


def build_category_name_proposal_payload(
    *,
    cluster_id: int,
    representatives: list[ClusterInput],
) -> dict[str, Any]:
    """Build an AI-ready payload without auto-confirming a category name."""
    return {
        "cluster_id": cluster_id,
        "instruction": "Suggest a category name, description, and lexical signals. Do not auto-confirm.",
        "representatives": [
            {
                "item_id": item.item_id,
                "file_hash": item.file_hash,
                "text_sample": item.text[:800],
                "metadata": item.metadata,
            }
            for item in representatives
        ],
    }
