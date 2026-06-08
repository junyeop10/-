"""Compare unsupervised document clustering variants against folder labels.

Folder names are used only for offline evaluation. They are never included in
the embedding text or clustering vectors.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, homogeneity_score, normalized_mutual_info_score, v_measure_score
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.categories_cluster_pipeline import split_three_sections
from src.feature_vector_builder import build_optional_layout_vector, build_pattern_vector, get_layout_confidence
from src.hash_utils import compute_raw_text_hash
from src.type_embedding_builder import build_type_embedding_text
from src.vectorizer import SentenceTransformerEmbedder


RAW_WEIGHTS = np.asarray([0.50, 0.25, 0.25], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate clustering against input folder labels.")
    parser.add_argument("--evidence-json", required=True, help="Existing cluster_evidence.json")
    parser.add_argument("--input-root", default="input_files", help="Root containing labeled child folders")
    parser.add_argument("--output", default="outputs/cluster_quality_experiments.json")
    parser.add_argument("--embedding-cache", default="outputs/cluster_quality_embeddings.npz")
    parser.add_argument("--model-name", default="paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--skip-umap", action="store_true", help="Skip slower UMAP comparison")
    args = parser.parse_args()

    documents = load_labeled_documents(Path(args.evidence_json), Path(args.input_root))
    report = evaluate_documents(
        documents,
        model_name=args.model_name,
        include_umap=not args.skip_umap,
        embedding_cache=Path(args.embedding_cache),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "document_count": report["summary"]["document_count"],
                "experiment_count": report["summary"]["experiment_count"],
                "model_name": report["summary"]["model_name"],
                "best_result": {
                    "vector_variant": report["summary"]["best_result"]["vector_variant"],
                    "reducer": report["summary"]["best_result"]["reducer"],
                    "hdbscan": report["summary"]["best_result"]["hdbscan"],
                    "metrics": {
                        key: value
                        for key, value in report["summary"]["best_result"]["metrics"].items()
                        if key != "cluster_details"
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"saved: {output_path}")


def load_labeled_documents(evidence_json: Path, input_root: Path) -> list[dict[str, Any]]:
    root = input_root.resolve()
    loaded = json.loads(evidence_json.read_text(encoding="utf-8"))
    documents: list[dict[str, Any]] = []
    for item in loaded:
        evidence = item.get("evidence", item)
        path = Path(str(item.get("file_path") or evidence.get("file_path") or "")).resolve()
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if len(relative.parts) < 2:
            continue
        documents.append(
            {
                "filename": str(evidence.get("filename", path.name)),
                "file_path": str(path),
                "folder_label": relative.parts[0],
                "evidence": evidence,
            }
        )
    return documents


def evaluate_documents(
    documents: list[dict[str, Any]],
    *,
    model_name: str,
    include_umap: bool,
    embedding_cache: Path | None = None,
) -> dict[str, Any]:
    if not documents:
        raise ValueError("No labeled child-folder documents were found.")
    embedder = SentenceTransformerEmbedder(model_name=model_name, use_legacy_sqlite_cache=False)
    semantic_variants = build_semantic_variants(documents, embedder, cache_path=embedding_cache)
    vector_variants = build_vector_variants(documents, semantic_variants)
    reducers = ["none", "pca"] + (["umap"] if include_umap else [])
    results: list[dict[str, Any]] = []

    for vector_name, vectors in vector_variants.items():
        for reducer in reducers:
            reduced = reduce_vectors(vectors, reducer)
            for min_cluster_size in (2, 3, 4, 5):
                for min_samples in (1, 2, 3):
                    for method in ("eom", "leaf"):
                        labels, probabilities = hdbscan_labels(
                            reduced,
                            min_cluster_size=min_cluster_size,
                            min_samples=min_samples,
                            cluster_selection_method=method,
                        )
                        for minimum_probability in (0.0, 0.35, 0.50, 0.65):
                            filtered_labels = [
                                label if probability >= minimum_probability else -1
                                for label, probability in zip(labels, probabilities)
                            ]
                            metrics = score_clustering(documents, filtered_labels, probabilities)
                            results.append(
                                {
                                    "vector_variant": vector_name,
                                    "reducer": reducer,
                                    "hdbscan": {
                                        "min_cluster_size": min_cluster_size,
                                        "min_samples": min_samples,
                                        "cluster_selection_method": method,
                                        "minimum_probability": minimum_probability,
                                    },
                                    "metrics": metrics,
                                    "labels": filtered_labels,
                                }
                            )

    ranked = sorted(results, key=result_rank_key, reverse=True)
    best = ranked[0]
    return {
        "summary": {
            "document_count": len(documents),
            "folder_counts": dict(sorted(Counter(item["folder_label"] for item in documents).items())),
            "model_name": model_name,
            "folder_labels_used_only_for_evaluation": True,
            "experiment_count": len(results),
            "best_result": best,
            "top_results": ranked[:15],
        },
        "documents": [
            {
                "index": index,
                "filename": item["filename"],
                "file_path": item["file_path"],
                "folder_label": item["folder_label"],
            }
            for index, item in enumerate(documents)
        ],
        "experiments": ranked,
    }


def build_semantic_variants(
    documents: list[dict[str, Any]],
    embedder: SentenceTransformerEmbedder,
    *,
    cache_path: Path | None,
) -> dict[str, np.ndarray]:
    raw_segments: list[str] = []
    type_texts: list[str] = []
    for item in documents:
        evidence = item["evidence"]
        raw_segments.extend(split_three_sections(str(evidence.get("sampled_text", ""))))
        type_texts.append(build_type_embedding_text(evidence))

    cache_signature = compute_raw_text_hash(
        json.dumps(
            {
                "model_name": embedder.model_name,
                "raw_segments": raw_segments,
                "type_texts": type_texts,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    if cache_path is not None and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if str(cached["signature"].item()) == cache_signature:
            return {
                "raw_three_section": cached["raw_three_section"],
                "type_text": cached["type_text"],
                "hybrid_text": cached["hybrid_text"],
            }

    raw_vectors = np.asarray(
        embedder.encode_many(raw_segments, text_kind="quality_raw_segment", embedding_version="quality-v1"),
        dtype=np.float32,
    ).reshape(len(documents), 3, -1)
    raw = normalize((raw_vectors * RAW_WEIGHTS.reshape(1, 3, 1)).sum(axis=1))
    type_vectors = normalize(
        np.asarray(
            embedder.encode_many(type_texts, text_kind="quality_type_text", embedding_version="quality-v1"),
            dtype=np.float32,
        )
    )
    hybrid = normalize(0.45 * raw + 0.55 * type_vectors)
    variants = {"raw_three_section": raw, "type_text": type_vectors, "hybrid_text": hybrid}
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, signature=cache_signature, **variants)
    return variants


def build_vector_variants(
    documents: list[dict[str, Any]],
    semantic_variants: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    patterns = normalize(np.asarray([build_pattern_vector(item["evidence"]) for item in documents], dtype=np.float32))
    layouts = normalize(np.asarray([build_optional_layout_vector(item["evidence"]) for item in documents], dtype=np.float32))
    layout_confidence = np.asarray(
        [get_layout_confidence(item["evidence"]) for item in documents],
        dtype=np.float32,
    ).reshape(-1, 1)
    variants: dict[str, np.ndarray] = {}
    for name, semantic in semantic_variants.items():
        variants[name] = normalize(semantic)
        variants[f"{name}+pattern15"] = normalize(np.concatenate([semantic * 0.85, patterns * 0.15], axis=1))
        variants[f"{name}+pattern25"] = normalize(np.concatenate([semantic * 0.75, patterns * 0.25], axis=1))
        variants[f"{name}+pattern20+layout05"] = normalize(
            np.concatenate([semantic * 0.75, patterns * 0.20, layouts * layout_confidence * 0.05], axis=1)
        )
    return variants


def reduce_vectors(vectors: np.ndarray, reducer: str) -> np.ndarray:
    if reducer == "none":
        return normalize(vectors)
    if reducer == "pca":
        component_count = min(15, len(vectors) - 1, vectors.shape[1])
        return normalize(PCA(n_components=component_count, random_state=42).fit_transform(vectors))
    if reducer == "umap":
        import umap

        component_count = min(10, len(vectors) // 2, vectors.shape[1])
        reduced = umap.UMAP(
            n_components=max(2, component_count),
            n_neighbors=min(10, len(vectors) - 1),
            min_dist=0.0,
            metric="cosine",
            random_state=42,
            low_memory=False,
        ).fit_transform(vectors)
        return normalize(reduced)
    raise ValueError(f"Unsupported reducer: {reducer}")


def hdbscan_labels(
    vectors: np.ndarray,
    *,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_method: str,
) -> tuple[list[int], list[float]]:
    import hdbscan

    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=cluster_selection_method,
        prediction_data=True,
    )
    labels = [int(value) for value in model.fit_predict(vectors)]
    probabilities = [float(value) for value in model.probabilities_]
    return labels, probabilities


def score_clustering(
    documents: list[dict[str, Any]],
    labels: list[int],
    probabilities: list[float],
) -> dict[str, Any]:
    true_labels = [item["folder_label"] for item in documents]
    assigned_indices = [index for index, label in enumerate(labels) if label != -1]
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in assigned_indices:
        grouped[labels[index]].append(index)

    correct = 0
    cluster_details: list[dict[str, Any]] = []
    for cluster_id, indices in sorted(grouped.items()):
        counts = Counter(true_labels[index] for index in indices)
        majority_label, majority_count = counts.most_common(1)[0]
        correct += majority_count
        cluster_details.append(
            {
                "cluster_id": cluster_id,
                "size": len(indices),
                "majority_label": majority_label,
                "purity": round(majority_count / len(indices), 4),
                "folder_counts": dict(counts),
            }
        )

    assigned_count = len(assigned_indices)
    total = len(documents)
    purity = correct / assigned_count if assigned_count else 0.0
    coverage = assigned_count / total if total else 0.0
    overall_correct_rate = correct / total if total else 0.0
    assigned_truth = [true_labels[index] for index in assigned_indices]
    assigned_predicted = [labels[index] for index in assigned_indices]
    small_cluster_count = sum(1 for indices in grouped.values() if len(indices) <= 2)
    return {
        "classified_accuracy": round(purity, 4),
        "coverage": round(coverage, 4),
        "overall_correct_rate": round(overall_correct_rate, 4),
        "noise_ratio": round(1.0 - coverage, 4),
        "cluster_count": len(grouped),
        "small_cluster_count": small_cluster_count,
        "mean_probability": round(sum(probabilities) / max(len(probabilities), 1), 4),
        "homogeneity": round(homogeneity_score(assigned_truth, assigned_predicted), 4) if assigned_count else 0.0,
        "v_measure": round(v_measure_score(assigned_truth, assigned_predicted), 4) if assigned_count else 0.0,
        "nmi": round(normalized_mutual_info_score(assigned_truth, assigned_predicted), 4) if assigned_count else 0.0,
        "ari": round(adjusted_rand_score(assigned_truth, assigned_predicted), 4) if assigned_count else 0.0,
        "cluster_details": cluster_details,
    }


def result_rank_key(result: dict[str, Any]) -> tuple[float, ...]:
    metrics = result["metrics"]
    useful = metrics["coverage"] >= 0.65 and metrics["cluster_count"] >= 2
    return (
        1.0 if useful else 0.0,
        float(metrics["overall_correct_rate"]),
        float(metrics["classified_accuracy"]),
        float(metrics["homogeneity"]),
        -float(metrics["small_cluster_count"]),
    )


if __name__ == "__main__":
    main()
