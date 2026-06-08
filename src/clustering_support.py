"""Embedding clustering and representative selection helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from src.feature_vector_builder import normalize_vector
from src.vectorizer import cosine_similarity


PARENT_CLUSTER_COSINE_THRESHOLD = 0.90


def cluster_embeddings(
    embeddings: list[list[float]],
    *,
    min_cluster_size: int = 2,
    min_samples: int | None = None,
    reducer: str | None = None,
    cluster_selection_method: str = "eom",
    normalize_embeddings: bool = True,
) -> dict[str, Any]:
    """Cluster embedding vectors with HDBSCAN; -1 labels remain noise."""
    if not embeddings:
        return {
            "cluster_ids": [],
            "algorithm": "hdbscan",
            "reducer": reducer,
            "cluster_selection_method": cluster_selection_method,
            "normalize_embeddings": normalize_embeddings,
            "status": "empty",
        }
    if len(embeddings) < max(2, min_cluster_size):
        return {
            "cluster_ids": [-1 for _ in embeddings],
            "probabilities": [0.0 for _ in embeddings],
            "algorithm": "hdbscan",
            "reducer": reducer,
            "cluster_selection_method": cluster_selection_method,
            "normalize_embeddings": normalize_embeddings,
            "status": "not_enough_documents",
        }
    matrix = np.asarray(embeddings, dtype=np.float32)
    if normalize_embeddings:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, 1e-12)
    if reducer in {"pca", "umap"}:
        matrix = _reduce_dimensions(matrix, reducer=reducer)
    try:
        import hdbscan

        model = hdbscan.HDBSCAN(
            min_cluster_size=max(2, min_cluster_size),
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method=cluster_selection_method,
        )
        labels = [int(label) for label in model.fit_predict(matrix)]
        probabilities = [float(value) for value in getattr(model, "probabilities_", np.ones(len(labels)))]
        return {
            "cluster_ids": labels,
            "probabilities": probabilities,
            "algorithm": "hdbscan",
            "reducer": reducer,
            "cluster_selection_method": cluster_selection_method,
            "normalize_embeddings": normalize_embeddings,
            "status": "ok",
        }
    except Exception as error:
        return {
            "cluster_ids": [-1 for _ in embeddings],
            "probabilities": [0.0 for _ in embeddings],
            "algorithm": "hdbscan",
            "reducer": reducer,
            "cluster_selection_method": cluster_selection_method,
            "normalize_embeddings": normalize_embeddings,
            "status": "failed",
            "error": str(error),
        }


def build_clustering_vector(
    text_embedding: list[float],
    pattern_vector: list[float],
    layout_vector: list[float] | None = None,
    *,
    layout_confidence: float = 0.0,
) -> list[float]:
    """Combine text, pattern, and optional layout signals into one clustering vector."""
    text_weight = 0.75 if layout_vector and layout_confidence > 0 else 0.85
    pattern_weight = 0.15
    layout_weight = 0.10 if layout_vector and layout_confidence > 0 else 0.0

    combined: list[float] = []
    combined.extend([value * text_weight for value in normalize_vector([float(value) for value in text_embedding])])
    combined.extend([value * pattern_weight for value in normalize_vector([float(value) for value in pattern_vector])])
    if layout_vector:
        confidence = max(0.0, min(float(layout_confidence), 1.0))
        combined.extend([value * layout_weight * confidence for value in normalize_vector([float(value) for value in layout_vector])])
    return normalize_vector(combined)


def select_representative_documents(
    cluster_docs: list[dict[str, Any]],
    embeddings: list[list[float]],
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Select documents closest to the cluster centroid."""
    if not cluster_docs or not embeddings:
        return []
    matrix = np.asarray(embeddings, dtype=np.float32)
    centroid = np.mean(matrix, axis=0)
    scored = []
    for document, embedding in zip(cluster_docs, embeddings):
        score = cosine_similarity([float(value) for value in centroid], [float(value) for value in embedding])
        scored.append((score, document))
    representatives = []
    for score, document in sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]:
        evidence = document.get("evidence", document)
        representatives.append(
            {
                "filename": evidence.get("filename", ""),
                "file_path": evidence.get("file_path", ""),
                "similarity_to_centroid": round(float(score), 4),
                "top_tokens": evidence.get("top_tokens", [])[:20],
                "sampled_text": evidence.get("sampled_text", ""),
                "structural_features": evidence.get("structural_features", {}),
                "text_stats": evidence.get("text_stats", {}),
            }
        )
    return representatives


def build_cluster_summaries(
    documents: list[dict[str, Any]],
    embeddings: list[list[float]],
    cluster_ids: list[int],
    *,
    representative_top_k: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build cluster summaries and separate noise documents."""
    grouped: dict[int, list[tuple[dict[str, Any], list[float]]]] = {}
    noise_documents: list[dict[str, Any]] = []
    for document, embedding, cluster_id in zip(documents, embeddings, cluster_ids):
        document["cluster_id"] = int(cluster_id)
        if int(cluster_id) == -1:
            noise_documents.append(document)
            continue
        grouped.setdefault(int(cluster_id), []).append((document, embedding))

    summaries = []
    for cluster_id, pairs in sorted(grouped.items()):
        cluster_docs = [document for document, _embedding in pairs]
        cluster_embeddings = [embedding for _document, embedding in pairs]
        summaries.append(
            {
                "cluster_id": cluster_id,
                "document_count": len(cluster_docs),
                "representative_documents": select_representative_documents(
                    cluster_docs,
                    cluster_embeddings,
                    top_k=representative_top_k,
                ),
                "common_signals": build_common_signals(cluster_docs),
            }
        )
    return summaries, noise_documents


def build_parent_cluster_groups(
    documents: list[dict[str, Any]],
    embeddings: list[list[float]],
    cluster_ids: list[int],
    *,
    similarity_threshold: float = PARENT_CLUSTER_COSINE_THRESHOLD,
    representative_top_k: int = 5,
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Group fine HDBSCAN clusters into conservative parent type candidates."""
    grouped: dict[int, list[tuple[dict[str, Any], list[float]]]] = {}
    for document, embedding, cluster_id in zip(documents, embeddings, cluster_ids):
        fine_cluster_id = int(cluster_id)
        document["parent_cluster_id"] = -1
        document.setdefault("evidence", document)["parent_cluster_id"] = -1
        if fine_cluster_id != -1:
            grouped.setdefault(fine_cluster_id, []).append((document, embedding))
    if not grouped:
        return [], {}

    fine_cluster_ids = sorted(grouped)
    centroid_matrix = np.asarray(
        [
            normalize_vector(
                np.mean(np.asarray([embedding for _document, embedding in grouped[cluster_id]], dtype=np.float32), axis=0).tolist()
            )
            for cluster_id in fine_cluster_ids
        ],
        dtype=np.float32,
    )
    labels = _complete_link_parent_labels(
        centroid_matrix,
        similarity_threshold=similarity_threshold,
    )
    label_to_fine_ids: dict[int, list[int]] = {}
    for fine_cluster_id, label in zip(fine_cluster_ids, labels):
        label_to_fine_ids.setdefault(int(label), []).append(fine_cluster_id)
    ordered_groups = sorted(label_to_fine_ids.values(), key=lambda ids: (min(ids), ids))

    fine_to_parent: dict[int, int] = {}
    parent_summaries: list[dict[str, Any]] = []
    for parent_cluster_id, child_cluster_ids in enumerate(ordered_groups):
        pairs = [
            pair
            for child_cluster_id in child_cluster_ids
            for pair in grouped[child_cluster_id]
        ]
        parent_docs = [document for document, _embedding in pairs]
        parent_embeddings = [embedding for _document, embedding in pairs]
        for child_cluster_id in child_cluster_ids:
            fine_to_parent[child_cluster_id] = parent_cluster_id
        for document in parent_docs:
            document["parent_cluster_id"] = parent_cluster_id
            document.setdefault("evidence", document)["parent_cluster_id"] = parent_cluster_id
        parent_summaries.append(
            {
                "cluster_id": parent_cluster_id,
                "parent_cluster_id": parent_cluster_id,
                "fine_cluster_ids": child_cluster_ids,
                "fine_cluster_count": len(child_cluster_ids),
                "document_count": len(parent_docs),
                "representative_documents": select_representative_documents(
                    parent_docs,
                    parent_embeddings,
                    top_k=representative_top_k,
                ),
                "common_signals": build_common_signals(parent_docs),
                "grouping": {
                    "strategy": "complete_link_cosine",
                    "similarity_threshold": similarity_threshold,
                    "minimum_centroid_cosine_similarity": _minimum_group_similarity(
                        child_cluster_ids,
                        fine_cluster_ids,
                        centroid_matrix,
                    ),
                },
            }
        )
    return parent_summaries, fine_to_parent


def build_common_signals(cluster_docs: list[dict[str, Any]]) -> dict[str, Any]:
    token_counter: Counter[str] = Counter()
    filename_counter: Counter[str] = Counter()
    numeric_features: dict[str, list[float]] = {}
    for document in cluster_docs:
        evidence = document.get("evidence", document)
        for item in evidence.get("top_tokens", []):
            if isinstance(item, dict):
                token_counter[str(item.get("token", ""))] += int(item.get("count", 1) or 1)
        for token in evidence.get("filename_tokens", []):
            filename_counter[str(token)] += 1
        for feature_group in ("structural_features", "text_stats"):
            for key, value in (evidence.get(feature_group) or {}).items():
                if isinstance(value, (int, float, bool)):
                    numeric_features.setdefault(key, []).append(float(value))
    return {
        "common_top_tokens": [token for token, _count in token_counter.most_common(20) if token],
        "common_filename_tokens": [token for token, _count in filename_counter.most_common(20) if token],
        "average_structural_features": {
            key: round(sum(values) / max(len(values), 1), 4)
            for key, values in sorted(numeric_features.items())
        },
    }


def _complete_link_parent_labels(matrix: np.ndarray, *, similarity_threshold: float) -> list[int]:
    if len(matrix) < 2:
        return [0 for _ in matrix]
    from sklearn.cluster import AgglomerativeClustering

    kwargs = {
        "n_clusters": None,
        "linkage": "complete",
        "distance_threshold": max(0.0, min(2.0, 1.0 - float(similarity_threshold))),
    }
    try:
        model = AgglomerativeClustering(metric="cosine", **kwargs)
    except TypeError:
        model = AgglomerativeClustering(affinity="cosine", **kwargs)
    return [int(value) for value in model.fit_predict(matrix)]


def _minimum_group_similarity(
    group_ids: list[int],
    all_ids: list[int],
    centroid_matrix: np.ndarray,
) -> float:
    if len(group_ids) < 2:
        return 1.0
    positions = {cluster_id: index for index, cluster_id in enumerate(all_ids)}
    scores = [
        float(np.dot(centroid_matrix[positions[left]], centroid_matrix[positions[right]]))
        for index, left in enumerate(group_ids)
        for right in group_ids[index + 1 :]
    ]
    return round(min(scores), 4) if scores else 1.0


def _reduce_dimensions(matrix: np.ndarray, *, reducer: str) -> np.ndarray:
    if matrix.shape[0] < 3 or matrix.shape[1] <= 2:
        return matrix
    if reducer == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=min(10, matrix.shape[0] - 1, matrix.shape[1]), random_state=42).fit_transform(matrix)
    if reducer == "umap":
        try:
            import umap

            return umap.UMAP(n_components=min(10, matrix.shape[1]), random_state=42).fit_transform(matrix)
        except Exception:
            return matrix
    return matrix
