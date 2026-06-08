"""Run the evidence -> embedding -> HDBSCAN pipeline without API labeling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api_category_labeler import build_cluster_labeling_payload
from src.categories_cluster_pipeline import (
    CATEGORIES_PIPELINE_VERSION,
    CLUSTER_SELECTION_METHOD,
    DEFAULT_REDUCER,
    MIN_CLUSTER_SIZE,
    MIN_SAMPLES,
    run_categories_clustering,
)
from src.cli import build_embedder, build_repository, load_categories
from src.cluster_projection import build_cluster_projection, render_cluster_projection_html
from src.clustering_support import build_cluster_summaries, build_parent_cluster_groups
from src.config import DEFAULT_CONFIG_PATH, load_app_config
from src.embedding_support import build_embedding_text
from src.evidence_pipeline import build_document_evidence
from src.file_reader import SUPPORTED_SUFFIXES, discover_supported_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect cluster-first document pipeline up to API payload creation.")
    parser.add_argument("input", help="Input folder or one supported file")
    parser.add_argument("--output", default="outputs", help="Output folder for JSON artifacts")
    parser.add_argument("--db", default="data/classifier.db", help="SQLite DB path")
    parser.add_argument("--categories", default="data/categories.json", help="Category seed JSON")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Application config JSON")
    parser.add_argument("--min-cluster-size", type=int, default=MIN_CLUSTER_SIZE, help="HDBSCAN min_cluster_size")
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES, help="HDBSCAN min_samples")
    parser.add_argument(
        "--cluster-selection-method",
        choices=["eom", "leaf"],
        default=None,
        help="HDBSCAN cluster selection method; leaf creates finer discovery clusters",
    )
    parser.add_argument("--no-normalize-embeddings", action="store_true", help="Disable L2 normalization before HDBSCAN")
    parser.add_argument("--representatives", type=int, default=5, help="Representative documents per cluster")
    parser.add_argument("--reducer", choices=["none", "pca", "umap"], default=DEFAULT_REDUCER, help="Dimension reducer before HDBSCAN")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR fallback during evidence extraction")
    args = parser.parse_args()

    result = run_pipeline(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        db_path=args.db,
        categories_path=args.categories,
        config_path=Path(args.config),
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        representative_top_k=args.representatives,
        reducer=args.reducer,
        cluster_selection_method=args.cluster_selection_method,
        normalize_embeddings=not args.no_normalize_embeddings,
        ocr_enabled=not args.no_ocr,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_pipeline(
    *,
    input_path: Path,
    output_dir: Path,
    db_path: str,
    categories_path: str,
    config_path: Path,
    min_cluster_size: int,
    min_samples: int | None,
    representative_top_k: int,
    reducer: str | None,
    cluster_selection_method: str | None = None,
    normalize_embeddings: bool = True,
    ocr_enabled: bool,
    embedder: Any | None = None,
) -> dict[str, Any]:
    del normalize_embeddings
    config = load_app_config(config_path)
    config.database_path = db_path
    config.taxonomy_path = categories_path
    repository = build_repository(db_path, config)
    repository.initialize_database()
    repository.seed_rules_from_categories(load_categories(Path(categories_path)))
    rules = _fetch_rules(repository)

    files = _resolve_input_files(input_path)
    evidence_documents = [
        build_document_evidence(
            file_path,
            rules=rules,
            min_text_chars=config.ocr.min_text_chars,
            ocr_enabled=ocr_enabled and config.ocr.enabled,
            ocr_max_pages=config.ocr.max_pages,
        )
        for file_path in files
    ]
    documents = [
        {
            "index": index,
            "file_path": evidence["file_path"],
            "filename": evidence["filename"],
            "file_hash": evidence["file_hash"],
            "evidence": evidence,
            "embedding_text": build_embedding_text(evidence),
        }
        for index, evidence in enumerate(evidence_documents)
    ]
    active_embedder = embedder or build_embedder(config)
    categories_result = run_categories_clustering(
        documents,
        embedder=active_embedder,
        repository=repository,
        config=config,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method
        or str(getattr(config.clustering, "cluster_selection_method", CLUSTER_SELECTION_METHOD)),
        reducer=reducer,
    )
    embeddings = categories_result["semantic_embeddings"]
    representative_vectors = categories_result["clustering_input_vectors"]
    clustering_vectors = categories_result["reduced_vectors"]
    cluster_result = categories_result["cluster_result"]
    cluster_ids = [int(cluster_id) for cluster_id in cluster_result.get("cluster_ids", [])]
    for document, cluster_id in zip(documents, cluster_ids):
        document["cluster_id"] = cluster_id
        document["evidence"]["cluster_id"] = cluster_id

    cluster_summaries, noise_documents = build_cluster_summaries(
        documents,
        representative_vectors,
        cluster_ids,
        representative_top_k=representative_top_k,
    )
    parent_cluster_summaries, _fine_to_parent = build_parent_cluster_groups(
        documents,
        representative_vectors,
        cluster_ids,
        representative_top_k=representative_top_k,
    )
    fine_cluster_payloads = [build_cluster_labeling_payload(summary) for summary in cluster_summaries]
    cluster_payloads = [build_cluster_labeling_payload(summary) for summary in parent_cluster_summaries]
    cluster_projection = build_cluster_projection(
        documents,
        clustering_vectors,
        cluster_ids,
        probabilities=[float(value) for value in cluster_result.get("probabilities", [])],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "cluster_evidence.json", documents)
    _write_json(output_dir / "cluster_payloads.json", cluster_payloads)
    _write_json(output_dir / "fine_cluster_payloads.json", fine_cluster_payloads)
    _write_json(output_dir / "parent_cluster_payloads.json", cluster_payloads)
    _write_json(output_dir / "noise_documents.json", noise_documents)
    _write_json(output_dir / "cluster_projection.json", cluster_projection)
    (output_dir / "cluster_projection.html").write_text(
        render_cluster_projection_html(cluster_projection),
        encoding="utf-8",
    )
    _write_json(
        output_dir / "cluster_run_summary.json",
        {
            "input": str(input_path),
            "file_count": len(files),
            "cluster_result": cluster_result,
            "cluster_count": len(cluster_summaries),
            "fine_cluster_count": len(cluster_summaries),
            "parent_cluster_count": len(parent_cluster_summaries),
            "noise_count": len(noise_documents),
            "projection_file": str(output_dir / "cluster_projection.html"),
            "clustering_vector_version": CATEGORIES_PIPELINE_VERSION,
            "embedding_strategy": categories_result["embedding_strategy"],
            "reducer_result": categories_result["reducer_result"],
            "api_call_performed": False,
        },
    )

    return {
        "file_count": len(files),
        "cluster_count": len(cluster_summaries),
        "parent_cluster_count": len(parent_cluster_summaries),
        "noise_count": len(noise_documents),
        "output_dir": str(output_dir),
        "api_call_performed": False,
        "files": {
            "cluster_evidence": str(output_dir / "cluster_evidence.json"),
            "cluster_payloads": str(output_dir / "cluster_payloads.json"),
            "noise_documents": str(output_dir / "noise_documents.json"),
            "cluster_projection": str(output_dir / "cluster_projection.html"),
            "cluster_run_summary": str(output_dir / "cluster_run_summary.json"),
        },
    }


def _resolve_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type: {input_path.suffix}")
        return [input_path]
    return discover_supported_files(input_path)


def _fetch_rules(repository: Any) -> list[dict[str, Any]]:
    return [
        {
            "category": str(rule["category"]),
            "rule_type": str(rule["rule_type"]),
            "pattern": str(rule["pattern"]),
            "weight": float(rule["weight"]),
            "rule_scope": str(rule["rule_scope"]) if "rule_scope" in rule.keys() else "content",
            "negative_weight": float(rule["negative_weight"]) if "negative_weight" in rule.keys() else 0.0,
        }
        for rule in repository.fetch_active_rules()
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
