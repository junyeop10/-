"""Pending category candidate discovery from review/miscellaneous document groups."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import os
from typing import Any

import numpy as np

from src.text_cleaner import tokenize_text


@dataclass
class ClusterCandidate:
    suggested_name: str
    representative_file_ids: list[int]
    evidence: dict[str, Any] = field(default_factory=dict)


class ClusterCandidateFinder:
    """Find small, conservative pending category candidates.

    Preferred flow is embedding generation followed by HDBSCAN. The legacy
    token bucket fallback only exists to keep candidate review available when
    optional ML dependencies or embedding models are unavailable.
    """

    def __init__(
        self,
        *,
        min_cluster_size: int = 3,
        max_candidates: int = 5,
        embedder: Any | None = None,
        repository: Any | None = None,
    ) -> None:
        self.min_cluster_size = min_cluster_size
        self.max_candidates = max_candidates
        self.embedder = embedder
        self.repository = repository

    def find_candidates(self, rows: list[dict[str, Any]]) -> list[ClusterCandidate]:
        eligible = [
            row
            for row in rows
            if bool(row.get("review_required"))
            or str(row.get("predicted_type") or row.get("predicted_category") or "").lower() in {"기타", "misc", "miscellaneous", "uncategorized"}
        ]
        if len(eligible) < self.min_cluster_size:
            return []

        labels, source = self._embedding_hdbscan_labels(eligible)
        if labels is None:
            return self._fallback_candidates(eligible)

        candidates: list[ClusterCandidate] = []
        for label in sorted(set(int(item) for item in labels)):
            if label < 0:
                continue
            group = [row for row, row_label in zip(eligible, labels) if int(row_label) == label]
            if len(group) < self.min_cluster_size:
                continue
            candidate = self._build_candidate(group, source=f"{source}:{label}")
            if candidate is not None:
                candidates.append(candidate)
        if not candidates and len(eligible) >= self.min_cluster_size:
            fallback = self._build_candidate(eligible, source=f"{source}:fallback_all")
            if fallback is not None:
                candidates.append(fallback)
        return candidates[: self.max_candidates]

    def _embedding_hdbscan_labels(self, rows: list[dict[str, Any]]) -> tuple[list[int] | None, str]:
        texts = [str(row.get("compressed_text") or row.get("text") or row.get("file_name") or "") for row in rows]
        file_hashes = [str(row.get("file_hash") or row.get("file_id") or "") for row in rows]
        matrix: np.ndarray | None = None
        source = "embedding_hdbscan"
        try:
            if self.embedder is not None:
                embeddings = self.embedder.encode_many(
                    texts,
                    repository=self.repository,
                    file_hashes=file_hashes,
                    text_kind="cluster_candidate",
                    embedding_version="2.1-cluster-candidate",
                )
                matrix = np.asarray(embeddings, dtype=np.float32)
                source = "embedding_hdbscan"
            else:
                os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
                from sklearn.feature_extraction.text import TfidfVectorizer

                vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, max_features=1500)
                matrix = vectorizer.fit_transform(texts).toarray().astype(np.float32)
                source = "tfidf_embedding_hdbscan"

            import hdbscan

            model = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=1, metric="euclidean")
            return [int(label) for label in model.fit_predict(matrix)], source
        except Exception:
            if matrix is None:
                return None, "unavailable"
            try:
                from sklearn.cluster import DBSCAN

                model = DBSCAN(eps=0.72, min_samples=self.min_cluster_size, metric="cosine")
                return [int(label) for label in model.fit_predict(matrix)], f"{source}_dbscan_fallback"
            except Exception:
                return None, "unavailable"

    def _fallback_candidates(self, rows: list[dict[str, Any]]) -> list[ClusterCandidate]:
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            tokens = tokenize_text(str(row.get("file_name") or row.get("compressed_text") or ""))
            key = tokens[0] if tokens else "misc"
            buckets.setdefault(key, []).append(row)
        candidates = []
        for key, group in buckets.items():
            if len(group) >= self.min_cluster_size:
                candidate = self._build_candidate(group, source=f"token_bucket:{key}")
                if candidate is not None:
                    candidates.append(candidate)
        return candidates[: self.max_candidates]

    def _build_candidate(self, group: list[dict[str, Any]], *, source: str) -> ClusterCandidate | None:
        token_counter: Counter[str] = Counter()
        file_names: list[str] = []
        file_ids: list[int] = []
        for row in group:
            file_names.append(str(row.get("file_name") or ""))
            try:
                file_ids.append(int(row["file_id"]))
            except (KeyError, TypeError, ValueError):
                continue
            token_counter.update(tokenize_text(f"{row.get('file_name', '')} {row.get('compressed_text', '')}")[:40])
        if len(file_ids) < self.min_cluster_size:
            return None
        keywords = [token for token, _count in token_counter.most_common(8)]
        suggested_name = "_".join(keywords[:3]) if keywords else "new_document_group"
        return ClusterCandidate(
            suggested_name=suggested_name[:80],
            representative_file_ids=file_ids[:10],
            evidence={
                "source": source,
                "size": len(group),
                "keywords": keywords,
                "file_names": file_names[:10],
                "cohesion": "candidate",
            },
        )
