"""Embedding-to-clustering pipeline adapted from categories.zip."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.embedding_support import embed_texts
from src.feature_vector_builder import build_pattern_vector, normalize_vector
from src.type_embedding_builder import build_type_embedding_text


CATEGORIES_PIPELINE_VERSION = "type-focused-pattern-v2"
EMBEDDING_STRATEGY = "type-focused-text-plus-pattern20"
WEIGHT_FRONT = 0.5
WEIGHT_MIDDLE = 0.25
WEIGHT_REAR = 0.25
UMAP_N_COMPONENTS_MAX = 10
UMAP_N_NEIGHBORS = 10
UMAP_MIN_DIST = 0.0
UMAP_METRIC = "cosine"
UMAP_RANDOM_STATE = 42
MIN_CLUSTER_SIZE = 2
MIN_SAMPLES: int | None = 1
CLUSTER_SELECTION_METHOD = "leaf"
PATTERN_VECTOR_WEIGHT = 0.20
TEXT_VECTOR_WEIGHT = 0.80
CENTROID_MERGE_COSINE_THRESHOLD = 0.98
MINIMUM_CLUSTER_PROBABILITY = 0.0
DEFAULT_REDUCER = "pca"
PCA_N_COMPONENTS_MAX = 20


def split_three_sections(text: str) -> tuple[str, str, str]:
    """Split text into front, middle, and rear thirds like categories.zip."""
    value = str(text or "")
    length = len(value)
    first_end = length // 3
    second_end = 2 * length // 3
    return value[:first_end], value[first_end:second_end], value[second_end:]


def embed_documents_three_sections(
    documents: list[dict[str, Any]],
    *,
    embedder: Any,
    repository: Any | None = None,
    config: Any | None = None,
) -> list[list[float]]:
    """Embed document thirds in one batch and combine them with 0.5/0.25/0.25 weights."""
    if not documents:
        return []

    segments: list[str] = []
    file_hashes: list[str] = []
    for document in documents:
        evidence = document.get("evidence", document)
        front, middle, rear = split_three_sections(str(evidence.get("sampled_text", "")))
        document["embedding_segments"] = {
            "front_chars": len(front),
            "middle_chars": len(middle),
            "rear_chars": len(rear),
            "weights": [WEIGHT_FRONT, WEIGHT_MIDDLE, WEIGHT_REAR],
        }
        segments.extend([front, middle, rear])
        file_hash = str(document.get("file_hash", evidence.get("file_hash", "")))
        file_hashes.extend([file_hash, file_hash, file_hash])

    segment_vectors = embed_texts(
        segments,
        embedder=embedder,
        repository=repository,
        file_hashes=file_hashes,
        config=config,
        text_kind="categories_zip_segment",
        embedding_version=CATEGORIES_PIPELINE_VERSION,
    )
    for index, segment in enumerate(segments):
        if not segment.strip() and index < len(segment_vectors):
            segment_vectors[index] = [0.0 for _value in segment_vectors[index]]

    document_vectors: list[list[float]] = []
    for index, document in enumerate(documents):
        chunk = segment_vectors[index * 3 : index * 3 + 3]
        combined = _weighted_average(chunk)
        document["semantic_embedding"] = combined
        document["embedding_strategy"] = EMBEDDING_STRATEGY
        document_vectors.append(combined)
    return document_vectors


def embed_documents_type_focused(
    documents: list[dict[str, Any]],
    *,
    embedder: Any,
    repository: Any | None = None,
    config: Any | None = None,
) -> list[list[float]]:
    """Embed noise-suppressed type evidence and attach cheap pattern vectors."""
    if not documents:
        return []
    texts: list[str] = []
    file_hashes: list[str] = []
    for document in documents:
        evidence = document.get("evidence", document)
        type_text = build_type_embedding_text(evidence)
        document["type_embedding_text"] = type_text
        document["pattern_vector"] = build_pattern_vector(evidence)
        texts.append(type_text)
        file_hashes.append(str(document.get("file_hash", evidence.get("file_hash", ""))))
    embeddings = embed_texts(
        texts,
        embedder=embedder,
        repository=repository,
        file_hashes=file_hashes,
        config=config,
        text_kind="type_focused_cluster_text",
        embedding_version=CATEGORIES_PIPELINE_VERSION,
    )
    for document, embedding in zip(documents, embeddings):
        document["semantic_embedding"] = embedding
        document["embedding_strategy"] = EMBEDDING_STRATEGY
    return embeddings


def build_type_clustering_vectors(
    documents: list[dict[str, Any]],
    semantic_embeddings: list[list[float]],
) -> list[list[float]]:
    """Combine normalized text embeddings and pattern vectors without forcing layout extraction."""
    vectors: list[list[float]] = []
    for document, embedding in zip(documents, semantic_embeddings):
        pattern = [float(value) for value in document.get("pattern_vector", [])]
        combined = [
            *[value * TEXT_VECTOR_WEIGHT for value in normalize_vector([float(value) for value in embedding])],
            *[value * PATTERN_VECTOR_WEIGHT for value in normalize_vector(pattern)],
        ]
        vector = normalize_vector(combined)
        document["clustering_input_vector"] = vector
        vectors.append(vector)
    return vectors


def reduce_embeddings_umap(vectors: list[list[float]]) -> tuple[list[list[float]], dict[str, Any]]:
    """Reduce embeddings with the adaptive UMAP parameters from categories.zip."""
    if not vectors:
        return [], {"reducer": "umap", "status": "empty"}
    if len(vectors) < 5:
        return vectors, {
            "reducer": "skipped",
            "status": "not_enough_documents",
            "reason": "categories.zip clustering requires at least 5 documents before UMAP",
            "input_dim": len(vectors[0]) if vectors else 0,
            "output_dim": len(vectors[0]) if vectors else 0,
        }

    import umap

    matrix = np.asarray(vectors, dtype=np.float32)
    n_neighbors, n_components = _safe_umap_params(len(vectors))
    model = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
        low_memory=False,
    )
    reduced = model.fit_transform(matrix).astype(np.float32)
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    reduced = reduced / np.maximum(norms, 1e-12)
    return reduced.tolist(), {
        "reducer": "umap",
        "status": "ok",
        "input_dim": int(matrix.shape[1]),
        "output_dim": int(reduced.shape[1]),
        "n_neighbors": n_neighbors,
        "n_components": n_components,
        "min_dist": UMAP_MIN_DIST,
        "metric": UMAP_METRIC,
        "random_state": UMAP_RANDOM_STATE,
    }


def reduce_embeddings_pca(vectors: list[list[float]]) -> tuple[list[list[float]], dict[str, Any]]:
    """Reduce vectors quickly with deterministic PCA for interactive runs."""
    if not vectors:
        return [], {"reducer": "pca", "status": "empty"}
    if len(vectors) < 3:
        return vectors, {
            "reducer": "skipped",
            "status": "not_enough_documents",
            "input_dim": len(vectors[0]) if vectors else 0,
            "output_dim": len(vectors[0]) if vectors else 0,
        }
    from sklearn.decomposition import PCA

    matrix = np.asarray(vectors, dtype=np.float32)
    n_components = min(PCA_N_COMPONENTS_MAX, len(vectors) - 1, matrix.shape[1])
    reduced = PCA(n_components=n_components, random_state=42).fit_transform(matrix).astype(np.float32)
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    reduced = reduced / np.maximum(norms, 1e-12)
    return reduced.tolist(), {
        "reducer": "pca",
        "status": "ok",
        "input_dim": int(matrix.shape[1]),
        "output_dim": int(reduced.shape[1]),
        "n_components": n_components,
        "random_state": 42,
    }


def reduce_embeddings(
    vectors: list[list[float]],
    *,
    reducer: str | None = DEFAULT_REDUCER,
) -> tuple[list[list[float]], dict[str, Any]]:
    if reducer is None:
        reducer = "none"
    if reducer == "pca":
        return reduce_embeddings_pca(vectors)
    if reducer == "umap":
        return reduce_embeddings_umap(vectors)
    if reducer == "none":
        return vectors, {
            "reducer": "none",
            "status": "ok",
            "input_dim": len(vectors[0]) if vectors else 0,
            "output_dim": len(vectors[0]) if vectors else 0,
        }
    raise ValueError(f"Unsupported reducer: {reducer}")


def cluster_reduced_embeddings(
    reduced_vectors: list[list[float]],
    *,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    min_samples: int | None = MIN_SAMPLES,
    cluster_selection_method: str = CLUSTER_SELECTION_METHOD,
    reducer: str | None = DEFAULT_REDUCER,
) -> dict[str, Any]:
    """Cluster reduced vectors with the HDBSCAN settings from categories.zip."""
    if not reduced_vectors:
        return _empty_cluster_result("empty", min_cluster_size, min_samples, cluster_selection_method)
    if len(reduced_vectors) < 5:
        result = _empty_cluster_result("not_enough_documents", min_cluster_size, min_samples, cluster_selection_method)
        result["cluster_ids"] = [-1 for _ in reduced_vectors]
        result["probabilities"] = [0.0 for _ in reduced_vectors]
        return result

    import hdbscan

    matrix = np.asarray(reduced_vectors, dtype=np.float32)
    safe_min_cluster_size = min(max(2, int(min_cluster_size)), max(len(reduced_vectors) - 1, 2))
    model = hdbscan.HDBSCAN(
        min_cluster_size=safe_min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=cluster_selection_method,
        prediction_data=True,
    )
    labels = [int(label) for label in model.fit_predict(matrix)]
    probabilities = [float(value) for value in model.probabilities_]
    return {
        "cluster_ids": labels,
        "probabilities": probabilities,
        "algorithm": "hdbscan",
        "reducer": reducer,
        "metric": "euclidean",
        "min_cluster_size": safe_min_cluster_size,
        "min_samples": min_samples,
        "cluster_selection_method": cluster_selection_method,
        "prediction_data": True,
        "status": "ok",
    }


def merge_nearby_clusters(
    cluster_ids: list[int],
    clustering_vectors: list[list[float]],
    *,
    cosine_threshold: float = CENTROID_MERGE_COSINE_THRESHOLD,
) -> tuple[list[int], dict[str, Any]]:
    """Conservatively merge near-identical fine clusters using pre-UMAP vectors."""
    active_ids = sorted(set(int(cluster_id) for cluster_id in cluster_ids if int(cluster_id) != -1))
    if len(active_ids) < 2:
        return cluster_ids, {
            "enabled": True,
            "cosine_threshold": cosine_threshold,
            "cluster_count_before": len(active_ids),
            "cluster_count_after": len(active_ids),
            "merged_pairs": [],
        }
    matrix = np.asarray(clustering_vectors, dtype=np.float32)
    labels = np.asarray(cluster_ids, dtype=np.int32)
    centroids = {
        cluster_id: np.asarray(normalize_vector(np.mean(matrix[labels == cluster_id], axis=0).tolist()), dtype=np.float32)
        for cluster_id in active_ids
    }
    parent = {cluster_id: cluster_id for cluster_id in active_ids}

    def find(cluster_id: int) -> int:
        while parent[cluster_id] != cluster_id:
            parent[cluster_id] = parent[parent[cluster_id]]
            cluster_id = parent[cluster_id]
        return cluster_id

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    merged_pairs: list[dict[str, Any]] = []
    for index, left in enumerate(active_ids):
        for right in active_ids[index + 1 :]:
            similarity = float(np.dot(centroids[left], centroids[right]))
            if similarity < cosine_threshold:
                continue
            union(left, right)
            merged_pairs.append({"left": left, "right": right, "cosine_similarity": round(similarity, 6)})

    compact_ids: dict[int, int] = {}
    merged_ids: list[int] = []
    for cluster_id in cluster_ids:
        if int(cluster_id) == -1:
            merged_ids.append(-1)
            continue
        root = find(int(cluster_id))
        if root not in compact_ids:
            compact_ids[root] = len(compact_ids)
        merged_ids.append(compact_ids[root])
    return merged_ids, {
        "enabled": True,
        "cosine_threshold": cosine_threshold,
        "cluster_count_before": len(active_ids),
        "cluster_count_after": len(set(merged_ids) - {-1}),
        "merged_pairs": merged_pairs,
    }


def filter_low_probability_documents(
    cluster_ids: list[int],
    probabilities: list[float],
    *,
    minimum_probability: float = MINIMUM_CLUSTER_PROBABILITY,
) -> tuple[list[int], dict[str, Any]]:
    """Move ambiguous HDBSCAN assignments to the noise pool."""
    filtered = [
        int(cluster_id) if int(cluster_id) != -1 and float(probability) >= minimum_probability else -1
        for cluster_id, probability in zip(cluster_ids, probabilities)
    ]
    return filtered, {
        "enabled": True,
        "minimum_probability": minimum_probability,
        "moved_to_noise": sum(
            1
            for before, after in zip(cluster_ids, filtered)
            if int(before) != -1 and int(after) == -1
        ),
    }


def run_categories_clustering(
    documents: list[dict[str, Any]],
    *,
    embedder: Any,
    repository: Any | None = None,
    config: Any | None = None,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    min_samples: int | None = MIN_SAMPLES,
    cluster_selection_method: str = CLUSTER_SELECTION_METHOD,
    reducer: str | None = DEFAULT_REDUCER,
) -> dict[str, Any]:
    """Run the validated type-focused embedding -> optional reducer -> HDBSCAN flow."""
    semantic_embeddings = embed_documents_type_focused(
        documents,
        embedder=embedder,
        repository=repository,
        config=config,
    )
    clustering_input_vectors = build_type_clustering_vectors(documents, semantic_embeddings)
    reduced_vectors, reducer_result = reduce_embeddings(clustering_input_vectors, reducer=reducer)
    cluster_result = cluster_reduced_embeddings(
        reduced_vectors,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        reducer=reducer,
    )
    merged_cluster_ids, merge_result = merge_nearby_clusters(
        [int(cluster_id) for cluster_id in cluster_result.get("cluster_ids", [])],
        clustering_input_vectors,
    )
    cluster_result["cluster_ids_before_centroid_merge"] = cluster_result.get("cluster_ids", [])
    filtered_cluster_ids, probability_filter_result = filter_low_probability_documents(
        merged_cluster_ids,
        [float(value) for value in cluster_result.get("probabilities", [])],
    )
    cluster_result["cluster_ids_before_probability_filter"] = merged_cluster_ids
    cluster_result["cluster_ids"] = filtered_cluster_ids
    cluster_result["centroid_merge"] = merge_result
    cluster_result["probability_filter"] = probability_filter_result
    for document, reduced_vector in zip(documents, reduced_vectors):
        document["reduced_vector"] = reduced_vector
        document["clustering_vector"] = reduced_vector
        document["clustering_vector_version"] = CATEGORIES_PIPELINE_VERSION
    return {
        "semantic_embeddings": semantic_embeddings,
        "clustering_input_vectors": clustering_input_vectors,
        "reduced_vectors": reduced_vectors,
        "reducer_result": reducer_result,
        "cluster_result": cluster_result,
        "clustering_vector_version": CATEGORIES_PIPELINE_VERSION,
        "embedding_strategy": EMBEDDING_STRATEGY,
    }


def _weighted_average(vectors: list[list[float]]) -> list[float]:
    if len(vectors) != 3:
        raise ValueError("Expected exactly three segment embeddings.")
    matrix = np.asarray(vectors, dtype=np.float32)
    weights = np.asarray([WEIGHT_FRONT, WEIGHT_MIDDLE, WEIGHT_REAR], dtype=np.float32)
    combined = (matrix * weights[:, np.newaxis]).sum(axis=0)
    return normalize_vector([float(value) for value in combined.tolist()])


def _safe_umap_params(n_samples: int) -> tuple[int, int]:
    n_neighbors = min(UMAP_N_NEIGHBORS, n_samples - 1)
    n_components = min(UMAP_N_COMPONENTS_MAX, n_samples // 2)
    return n_neighbors, max(n_components, 2)


def _empty_cluster_result(
    status: str,
    min_cluster_size: int,
    min_samples: int | None,
    cluster_selection_method: str,
) -> dict[str, Any]:
    return {
        "cluster_ids": [],
        "probabilities": [],
        "algorithm": "hdbscan",
        "reducer": "umap",
        "metric": "euclidean",
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "cluster_selection_method": cluster_selection_method,
        "prediction_data": True,
        "status": status,
    }
