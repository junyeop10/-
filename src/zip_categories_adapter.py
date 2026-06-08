"""Adapter for the vendored fixed-category ZIP classification pipeline."""

from __future__ import annotations

from collections import Counter, defaultdict
from importlib import import_module
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np

from src.classifier import ClassificationResult
from src.document_features import DocumentFeatureExtractor
from src.embedding_support import embed_texts
from src.hash_utils import compute_xxhash64
from src.text_cleaner import build_sampled_text, tokenize_text


ZIP_PIPELINE_VERSION = "categories-zip-xxhash-v2"
ZIP_SUPPLEMENTAL_CLUSTERING_ENABLED = False
ZIP_TEXT_KIND = "categories_zip_weighted_segments"
ZIP_SEED_TEXT_KIND = "categories_zip_description_seed"
ZIP_READER_EXTENDED = "extended"
ZIP_READER_ORIGINAL = "zip_original"
ZIP_UNCLASSIFIED = "_미분류_LLM위임"
VENDOR_DIR = Path(__file__).resolve().parents[1] / "vendor" / "categories_pipeline"
ZIP_CATEGORIES_PATH = VENDOR_DIR / "settings" / "categories.json"
PROFILE_MAX_KEYWORDS = 18
PROFILE_MAX_SNIPPETS = 3
PROFILE_SNIPPET_CHARS = 220
PROFILE_STOPWORDS = {
    "그리고",
    "그러나",
    "있는",
    "있다",
    "합니다",
    "대한",
    "위한",
    "관련",
    "통해",
    "경우",
    "또는",
    "자료",
    "문서",
    "파일",
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "pdf",
    "hwp",
    "hwpx",
    "docx",
    "xlsx",
}


def run_zip_categories_pipeline(
    documents: list[dict[str, Any]],
    *,
    embedder: Any,
    repository: Any | None = None,
    config: Any | None = None,
) -> dict[str, Any]:
    """Run ZIP rules, weighted embeddings, semantic classification, and optional clustering."""
    started = time.perf_counter()
    runtime_modules = ["file_reader", "rule_classifier", "semantic_core", "similarity"]
    if ZIP_SUPPLEMENTAL_CLUSTERING_ENABLED:
        runtime_modules.extend(["reducer", "clusterer"])
    vendor = _load_vendor_modules(*runtime_modules)
    categories = vendor["similarity"].load_categories(ZIP_CATEGORIES_PATH)
    _ensure_original_reader_evidence(documents, vendor["file_reader"])
    rule_results = [
        vendor["rule_classifier"].classify_by_filename(str(document["evidence"].get("filename", "")))
        for document in documents
    ]

    weighted_vectors = _embed_weighted_documents(
        documents,
        embedder=embedder,
        repository=repository,
        config=config,
    )
    confirmed_rows = _fetch_confirmed_feedback(repository)
    seed_texts, profile_enhancements = _build_enhanced_seed_texts(categories, confirmed_rows)
    seed_vectors = embed_texts(
        seed_texts,
        embedder=embedder,
        repository=repository,
        file_hashes=[""] * len(seed_texts),
        config=config,
        text_kind=ZIP_SEED_TEXT_KIND,
        embedding_version=ZIP_PIPELINE_VERSION,
    )
    semantic_store = vendor["semantic_core"].SemanticStore()
    for category, text, vector in zip(categories, seed_texts, seed_vectors):
        semantic_store.add(normalize_category_name(category.folder), np.asarray(vector, dtype=np.float32), text=text)
    category_names = {normalize_category_name(category.folder) for category in categories}
    feedback_examples_used = _load_confirmed_feedback(
        semantic_store,
        confirmed_rows=confirmed_rows,
        allowed_categories=category_names,
    )
    semantic_classifier = vendor["semantic_core"].SemanticClassifier(semantic_store)

    classified: list[dict[str, Any]] = []
    for document, vector, rule_result in zip(documents, weighted_vectors, rule_results):
        evidence = document["evidence"]
        if rule_result.is_confident:
            category = normalize_category_name(rule_result.category)
            confidence = float(rule_result.confidence)
            candidate_scores = {category: confidence}
            signals: dict[str, float] = {}
            semantic_top_k: list[tuple[str, float]] = []
            rule_confirmed = True
        else:
            semantic_result = semantic_classifier.classify(
                np.asarray(vector, dtype=np.float32),
                query_text=str(evidence.get("sampled_text", "")),
            )
            category = str(semantic_result.category or ZIP_UNCLASSIFIED)
            confidence = float(semantic_result.confidence)
            candidate_scores = {str(name): float(score) for name, score in semantic_result.top_k}
            signals = {str(name): float(score) for name, score in semantic_result.signals.items()}
            semantic_top_k = [(str(name), float(score)) for name, score in semantic_result.top_k]
            rule_confirmed = False
        classified.append(
            {
                **document,
                "category": category,
                "confidence": confidence,
                "review_required": category == ZIP_UNCLASSIFIED,
                "rule_confirmed": rule_confirmed,
                "matched_keywords": list(rule_result.matched_keywords),
                "rule_explanation": str(rule_result.explanation),
                "semantic_signals": signals,
                "semantic_top_k": semantic_top_k,
                "candidate_scores": candidate_scores,
                "embedding": vector,
            }
        )

    cluster_result = _cluster_all_documents(
        weighted_vectors,
        vendor,
        enabled=ZIP_SUPPLEMENTAL_CLUSTERING_ENABLED,
    )
    for item, cluster_id, probability, reduced_vector in zip(
        classified,
        cluster_result["cluster_ids"],
        cluster_result["probabilities"],
        cluster_result["reduced_vectors"],
    ):
        item["cluster_id"] = int(cluster_id)
        item["cluster_probability"] = float(probability)
        item["reduced_vector"] = [float(value) for value in reduced_vector]
        item["clustering_status"] = str(cluster_result["status"])
        item["evidence"]["cluster_id"] = int(cluster_id)

    return {
        "documents": classified,
        "categories": [normalize_category_name(category.folder) for category in categories],
        "cluster_result": cluster_result,
        "elapsed": round(time.perf_counter() - started, 4),
        "feedback_examples_used": feedback_examples_used,
        "profile_enhancements": profile_enhancements,
        "pipeline_version": ZIP_PIPELINE_VERSION,
    }


def build_zip_original_evidence(path: str | Path) -> dict[str, Any]:
    """Build a compact evidence object through the original ZIP reader only."""
    vendor = _load_vendor_modules("file_reader")
    file_path = Path(path)
    result = vendor["file_reader"].read_file(file_path)
    text = str(result.text or "")
    tokens = tokenize_text(f"{file_path.stem} {text}")
    features = DocumentFeatureExtractor().extract(
        file_name=file_path.name,
        file_ext=file_path.suffix,
        text=text,
        file_size=file_path.stat().st_size if file_path.exists() else 0,
        file_path=None,
    )
    return {
        "file_path": str(file_path),
        "filename": file_path.name,
        "file_hash": compute_xxhash64(file_path) if file_path.exists() else "",
        "extracted_text_length": len(text),
        "extraction_status": "text_ok" if text.strip() else "evidence_insufficient",
        "status_flags": ["zip_original_reader"],
        "extraction_error": str(result.error or ""),
        "filename_tokens": features.filename_features.get("tokens", []),
        "top_tokens": [
            {"token": token, "count": count}
            for token, count in Counter(tokens).most_common(30)
        ],
        "sampled_text": build_sampled_text(text, total_limit=4500, part_limit=1500),
        "compressed_preview": features.compressed_text[:1600],
        "text_stats": features.text_stats,
        "structural_features": features.structural_features,
        "layout_features": {},
        "old_rule_signals": {"status": "zip_original_reader", "keyword_signals": []},
        "ocr_cache_hit": False,
        "evidence_cache_hit": False,
        "reader_mode": ZIP_READER_ORIGINAL,
        "read_method": str(result.source_method),
        "api_reader": {
            "required": not bool(text.strip()),
            "status": "placeholder",
            "reason": "zip_original_reader_empty" if not text.strip() else "",
        },
    }


def classification_result_from_zip(item: dict[str, Any]) -> ClassificationResult:
    """Convert adapter output into the existing GUI/DB classification contract."""
    category = str(item["category"])
    confidence = float(item["confidence"])
    matched = [str(value) for value in item.get("matched_keywords", [])]
    signals = {str(key): float(value) for key, value in item.get("semantic_signals", {}).items()}
    rule_confirmed = bool(item.get("rule_confirmed"))
    review_required = bool(item.get("review_required"))
    reasoning = str(item.get("rule_explanation", "")) if rule_confirmed else (
        f"ZIP semantic fusion: {signals}" if signals else "ZIP semantic fusion: no confident category"
    )
    return ClassificationResult(
        predicted_category=category,
        confidence=confidence,
        final_score=confidence,
        rule_score=confidence if rule_confirmed else 0.0,
        embedding_score=confidence if not rule_confirmed else 0.0,
        llm_score=0.0,
        feedback_score=0.0,
        duplicate_score=0.0,
        similarity_score=float(signals.get("cos_max", 0.0)),
        embedding_used=not rule_confirmed,
        review_required=review_required,
        matched_rules=matched,
        candidate_scores=dict(item.get("candidate_scores", {})),
        reasoning=reasoning,
        query_embedding=[float(value) for value in item.get("embedding", [])],
        large_category="zip_categories",
        middle_category=category,
        middle_confidence=confidence,
        review_reasons=["zip_low_confidence"] if review_required else [],
        rule_evidence={
            "zip_rule_confirmed": rule_confirmed,
            "matched_keywords": matched,
            "explanation": item.get("rule_explanation", ""),
        },
        semantic_evidence=[
            {"signal": key, "score": value}
            for key, value in signals.items()
        ],
        score_breakdown={
            "pipeline": ZIP_PIPELINE_VERSION,
            "cluster_id": int(item.get("cluster_id", -1)),
            "cluster_probability": float(item.get("cluster_probability", 0.0)),
        },
    )


def _embed_weighted_documents(
    documents: list[dict[str, Any]],
    *,
    embedder: Any,
    repository: Any | None,
    config: Any | None,
) -> list[list[float]]:
    segments: list[str] = []
    file_hashes: list[str] = []
    for document in documents:
        evidence = document["evidence"]
        front, middle, rear = _split_three(str(evidence.get("sampled_text", "")))
        document["embedding_segments"] = {
            "front_chars": len(front),
            "middle_chars": len(middle),
            "rear_chars": len(rear),
            "weights": [0.5, 0.25, 0.25],
        }
        segments.extend([front, middle, rear])
        file_hash = str(document.get("file_hash", evidence.get("file_hash", "")))
        file_hashes.extend([file_hash, file_hash, file_hash])
    nonempty_indices = [index for index, segment in enumerate(segments) if segment.strip()]
    encoded = embed_texts(
        [segments[index] for index in nonempty_indices],
        embedder=embedder,
        repository=repository,
        file_hashes=[file_hashes[index] for index in nonempty_indices],
        config=config,
        text_kind=ZIP_TEXT_KIND,
        embedding_version=ZIP_PIPELINE_VERSION,
    )
    flat_vectors = [[0.0] * 384 for _segment in segments]
    for index, vector in zip(nonempty_indices, encoded):
        flat_vectors[index] = vector
    results = []
    for index in range(len(documents)):
        matrix = np.asarray(flat_vectors[index * 3 : index * 3 + 3], dtype=np.float32)
        combined = (matrix * np.asarray([0.5, 0.25, 0.25], dtype=np.float32)[:, None]).sum(axis=0)
        norm = float(np.linalg.norm(combined))
        results.append((combined / norm).astype(np.float32).tolist() if norm > 1e-8 else combined.tolist())
    return results


def _cluster_all_documents(
    vectors: list[list[float]],
    vendor: dict[str, Any],
    *,
    enabled: bool = ZIP_SUPPLEMENTAL_CLUSTERING_ENABLED,
) -> dict[str, Any]:
    if not enabled:
        return {
            "cluster_ids": [-1] * len(vectors),
            "probabilities": [0.0] * len(vectors),
            "reduced_vectors": vectors,
            "status": "disabled",
            "enabled": False,
            "reducer": "disabled",
            "reason": "supplemental UMAP and HDBSCAN are temporarily disconnected",
        }
    if len(vectors) < 5:
        return {
            "cluster_ids": [-1] * len(vectors),
            "probabilities": [0.0] * len(vectors),
            "reduced_vectors": vectors,
            "status": "not_enough_documents",
            "enabled": True,
            "reducer": "skipped",
        }
    try:
        # Cached float32 vectors and freshly encoded vectors can differ below
        # 1e-6. UMAP amplifies that noise enough to alter HDBSCAN labels.
        matrix = np.round(np.asarray(vectors, dtype=np.float32), decimals=6)
        reduced = vendor["reducer"].DimReducer().fit_transform(matrix)
        clustered = vendor["clusterer"].Clusterer().fit_predict(reduced.vectors)
        return {
            "cluster_ids": [int(value) for value in clustered.labels.tolist()],
            "probabilities": [float(value) for value in clustered.probabilities.tolist()],
            "reduced_vectors": reduced.vectors.astype(np.float32).tolist(),
            "status": "ok",
            "enabled": True,
            "reducer": "umap",
            "input_dim": int(reduced.n_input_dim),
            "output_dim": int(reduced.n_output_dim),
            "min_cluster_size": 3,
            "min_samples": None,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
        }
    except Exception as error:
        return {
            "cluster_ids": [-1] * len(vectors),
            "probabilities": [0.0] * len(vectors),
            "reduced_vectors": vectors,
            "status": "failed",
            "enabled": True,
            "reducer": "umap",
            "error": str(error),
        }


def _ensure_original_reader_evidence(documents: list[dict[str, Any]], file_reader: Any) -> None:
    del file_reader
    for document in documents:
        if "evidence" not in document:
            document["evidence"] = build_zip_original_evidence(document["file_path"])


def _split_three(text: str) -> tuple[str, str, str]:
    length = len(text)
    first = length // 3
    second = 2 * length // 3
    return text[:first], text[first:second], text[second:]


def _fetch_confirmed_feedback(repository: Any | None) -> list[dict[str, Any]]:
    if repository is None or not hasattr(repository, "fetch_confirmed_examples"):
        return []
    return [dict(row) for row in repository.fetch_confirmed_examples()]


def _build_enhanced_seed_texts(
    categories: list[Any],
    confirmed_rows: list[dict[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    grouped_texts: dict[str, list[str]] = defaultdict(list)
    for row in confirmed_rows:
        category = normalize_category_name(row.get("category", ""))
        source_text = str(row.get("source_text") or "").strip()
        if category and source_text:
            grouped_texts[category].append(source_text)

    seed_texts: list[str] = []
    enhanced_categories: dict[str, dict[str, Any]] = {}
    for category in categories:
        category_name = normalize_category_name(category.folder)
        base_text = f"{category_name}: {category.description}"
        examples = grouped_texts.get(category_name, [])
        if not examples:
            seed_texts.append(base_text)
            continue

        keywords = _extract_profile_keywords(examples)
        snippets = _extract_profile_snippets(examples)
        enhanced_parts = [
            base_text,
            f"confirmed_examples_count: {len(examples)}",
        ]
        if keywords:
            enhanced_parts.append(f"confirmed_keywords: {', '.join(keywords)}")
        if snippets:
            enhanced_parts.append("representative_evidence: " + " / ".join(snippets))
        seed_texts.append("\n".join(enhanced_parts))
        enhanced_categories[category_name] = {
            "confirmed_examples_count": len(examples),
            "keywords": keywords,
            "snippet_count": len(snippets),
        }

    return seed_texts, {
        "enabled": True,
        "confirmed_examples_seen": len(confirmed_rows),
        "enhanced_category_count": len(enhanced_categories),
        "categories": enhanced_categories,
    }


def _extract_profile_keywords(examples: list[str]) -> list[str]:
    counter: Counter[str] = Counter()
    for text in examples:
        seen_in_example: set[str] = set()
        for token in tokenize_text(text):
            normalized = token.strip().lower()
            if (
                len(normalized) < 2
                or normalized in PROFILE_STOPWORDS
                or normalized.isdigit()
            ):
                continue
            seen_in_example.add(normalized)
        counter.update(seen_in_example)
    return [token for token, _count in counter.most_common(PROFILE_MAX_KEYWORDS)]


def _extract_profile_snippets(examples: list[str]) -> list[str]:
    snippets: list[str] = []
    for text in examples:
        snippet = " ".join(text.split())
        if not snippet:
            continue
        snippets.append(snippet[:PROFILE_SNIPPET_CHARS])
        if len(snippets) >= PROFILE_MAX_SNIPPETS:
            break
    return snippets


def _load_confirmed_feedback(
    semantic_store: Any,
    *,
    confirmed_rows: list[dict[str, Any]],
    allowed_categories: set[str],
) -> int:
    if not confirmed_rows:
        return 0
    used = 0
    for row in confirmed_rows:
        category = normalize_category_name(row.get("category", ""))
        if category not in allowed_categories:
            continue
        try:
            vector = np.asarray(json.loads(str(row.get("embedding_json") or "[]")), dtype=np.float32)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if vector.shape != (384,):
            continue
        semantic_store.add(category, vector, text=str(row.get("source_text") or ""))
        used += 1
    return used


def normalize_category_name(value: Any) -> str:
    """Remove the display-order prefix used by the original ZIP categories."""
    return re.sub(r"^\s*\d+\.\s*", "", str(value or "")).strip()


def _load_vendor_modules(*only: str) -> dict[str, Any]:
    vendor_path = str(VENDOR_DIR)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)
    requested = set(only or ("clusterer", "file_reader", "reducer", "rule_classifier", "semantic_core", "similarity"))
    if "semantic_core" in requested:
        import_module("lexical")
    return {name: import_module(name) for name in requested}
