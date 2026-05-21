"""CLI commands for the file classifier MVP."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.adaptive import rebuild_adaptive_learning
from src.cluster_candidates import ClusterCandidateFinder
from src.classifier import (
    ClassificationResult,
    HybridClassifier,
    get_processing_method_label,
    get_processing_trace_text,
)
from src.config import DEFAULT_CONFIG_PATH, AppConfig, default_config, load_app_config
from src.document_features import DocumentFeatureExtractor
from src.embedding_repository import (
    create_embedding_repository,
    migrate_sqlite_embedding_cache_to_hdf5,
)
from src.fast_worker import process_file_fast
from src.file_reader import discover_supported_files, ensure_input_directory, extract_text_from_file
from src.hash_utils import compute_xxhash64
from src.llm_support import (
    DEFAULT_OLLAMA_MODEL,
    aggregate_category_scores,
    classify_with_ollama,
    classify_with_provider,
    should_use_llm,
)
from src.operations import commit_move_batch, preview_move_plan, restore_batch, restore_file, undo_last_move
from src.ocr_support import (
    DEFAULT_OCR_MIN_CHARS,
    OCR_MAX_PAGES,
    build_filename_hint_evidence,
    explain_ocr_decision,
    ocr_pdf_file,
)
from src.performance import build_file_latency_analysis
from src.recovery import create_safety_snapshot
from src.rule_classifier import RuleBasedClassifier, build_rule_input_text, score_text_with_rules
from src.storage import ClassificationRepository
from src.taxonomy import Taxonomy, load_taxonomy
from src.text_cleaner import normalize_text
from src.type_classifier import TypeClassifier
from src.vectorizer import SentenceTransformerEmbedder


FAST_TARGET_SECONDS = 10.0
MAX_FAST_WORKERS = min(4, os.cpu_count() or 1)
FAST_MIN_RULE_MATCHES_FOR_SKIP = 1
FAST_RULE_SKIP_EMBEDDING_THRESHOLD = 0.50
FAST_LOW_RULE_CONFIDENCE_THRESHOLD = 0.50
MAX_OCR_WORKERS = min(4, os.cpu_count() or 1)


def main() -> None:
    """Parse CLI arguments and run the selected command."""
    configure_console()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


def configure_console() -> None:
    """Use UTF-8 streams when the host supports it."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Build the command parser."""
    parser = argparse.ArgumentParser(description="Feedback-based document classifier")
    parser.set_defaults(func=lambda _: parser.print_help())

    common_parent = argparse.ArgumentParser(add_help=False)
    common_parent.add_argument("--db", default="data/classifier.db", help="SQLite DB path")
    common_parent.add_argument("--categories", default="data/categories.json", help="Category seed JSON")
    common_parent.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Application config JSON")

    subparsers = parser.add_subparsers(dest="command")

    init_db_parser = subparsers.add_parser("init-db", parents=[common_parent], help="Initialize DB")
    init_db_parser.set_defaults(func=handle_init_db)

    classify_parser = subparsers.add_parser("classify", parents=[common_parent], help="Classify files")
    classify_parser.add_argument("--input-dir", default="input_files", help="Input folder")
    classify_parser.add_argument(
        "--model-name",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="sentence-transformers model name",
    )
    classify_parser.add_argument("--review", action="store_true", help="Ask for user confirmation")
    classify_parser.add_argument("--fast", action="store_true", help="Parallel fast mode")
    classify_parser.add_argument("--workers", type=int, default=MAX_FAST_WORKERS, help="Fast mode worker count")
    classify_parser.add_argument("--ocr-workers", type=int, default=MAX_OCR_WORKERS, help="OCR worker count")
    classify_parser.add_argument(
        "--ocr-min-chars",
        type=int,
        default=DEFAULT_OCR_MIN_CHARS,
        help="Skip OCR when extracted text is at least this long",
    )
    classify_parser.add_argument("--use-llm", action="store_true", help="Use a local Ollama LLM for ambiguous files")
    classify_parser.add_argument("--llm-model", default=DEFAULT_OLLAMA_MODEL, help="Ollama model name")
    classify_parser.set_defaults(func=handle_classify)

    suggest_parser = subparsers.add_parser("suggest-rules", parents=[common_parent], help="Suggest rules")
    suggest_parser.add_argument("--min-occurrences", type=int, default=2, help="Minimum repeated token count")
    suggest_parser.add_argument("--save-candidates", action="store_true", help="Save candidate rules")
    suggest_parser.set_defaults(func=handle_suggest_rules)

    stats_parser = subparsers.add_parser("stats", parents=[common_parent], help="Show DB stats")
    stats_parser.set_defaults(func=handle_stats)

    add_profile_parser = subparsers.add_parser("add-category-profile", parents=[common_parent], help="Add a synthetic training category profile")
    add_profile_parser.add_argument("--type", required=True, dest="category_type", help="Document type/category name")
    add_profile_parser.add_argument("--text", required=True, help="Natural-language profile text")
    add_profile_parser.add_argument("--tags", default="", help="Comma-separated tags")
    add_profile_parser.add_argument("--weight", type=float, default=0.5, help="Synthetic sample weight")
    add_profile_parser.add_argument("--synthetic-count", type=int, default=5, help="Synthetic rows to generate")
    add_profile_parser.set_defaults(func=handle_add_category_profile)

    list_profile_parser = subparsers.add_parser("list-category-profiles", parents=[common_parent], help="List category profiles")
    list_profile_parser.add_argument("--include-inactive", action="store_true", help="Show inactive profiles too")
    list_profile_parser.set_defaults(func=handle_list_category_profiles)

    deactivate_profile_parser = subparsers.add_parser("deactivate-category-profile", parents=[common_parent], help="Deactivate a category profile")
    deactivate_profile_parser.add_argument("--profile-id", type=int, required=True, help="Profile id")
    deactivate_profile_parser.set_defaults(func=handle_deactivate_category_profile)

    debug_training_parser = subparsers.add_parser("debug-training-sources", parents=[common_parent], help="Inspect TypeClassifier training sources")
    debug_training_parser.set_defaults(func=handle_debug_training_sources)

    preview_move_parser = subparsers.add_parser("preview_move", parents=[common_parent], help="Stage a safe move preview")
    preview_move_parser.add_argument("--limit", type=int, default=None, help="Preview only the latest N files")
    preview_move_parser.set_defaults(func=handle_preview_move)

    commit_move_parser = subparsers.add_parser("commit_move", parents=[common_parent], help="Commit a staged move batch")
    commit_move_parser.add_argument("--batch-id", type=int, required=True, help="Move batch id from preview_move")
    commit_move_parser.set_defaults(func=handle_commit_move)

    undo_move_parser = subparsers.add_parser("undo_last_move", parents=[common_parent], help="Undo the most recent committed move batch")
    undo_move_parser.set_defaults(func=handle_undo_last_move)

    restore_batch_parser = subparsers.add_parser("restore_batch", parents=[common_parent], help="Restore a move batch")
    restore_batch_parser.add_argument("--batch-id", type=int, required=True, help="Batch id to restore")
    restore_batch_parser.set_defaults(func=handle_restore_batch)

    restore_file_parser = subparsers.add_parser("restore_file", parents=[common_parent], help="Restore a moved file by move item id")
    restore_file_parser.add_argument("--move-item-id", type=int, required=True, help="Move item id to restore")
    restore_file_parser.set_defaults(func=handle_restore_file)

    move_history_parser = subparsers.add_parser("list_move_history", parents=[common_parent], help="List move history")
    move_history_parser.add_argument("--category", default=None, help="Filter by middle category")
    move_history_parser.add_argument("--batch-id", type=int, default=None, help="Filter by batch id")
    move_history_parser.add_argument("--date-from", default=None, help="Filter from date/time")
    move_history_parser.add_argument("--date-to", default=None, help="Filter to date/time")
    move_history_parser.set_defaults(func=handle_list_move_history)

    list_feedback_parser = subparsers.add_parser("list_feedback_logs", parents=[common_parent], help="List feedback logs")
    list_feedback_parser.add_argument("--category", default=None, help="Filter by final middle category")
    list_feedback_parser.add_argument("--date-from", default=None, help="Filter from date/time")
    list_feedback_parser.add_argument("--date-to", default=None, help="Filter to date/time")
    list_feedback_parser.add_argument("--min-confidence", type=float, default=None, help="Filter by minimum classification confidence")
    list_feedback_parser.add_argument("--file-name", default=None, help="Filter by file name substring")
    list_feedback_parser.set_defaults(func=handle_list_feedback_logs)

    show_feedback_parser = subparsers.add_parser("show_feedback_log", parents=[common_parent], help="Show one feedback log")
    show_feedback_parser.add_argument("--feedback-log-id", type=int, required=True, help="Feedback log id")
    show_feedback_parser.set_defaults(func=handle_show_feedback_log)

    delete_feedback_parser = subparsers.add_parser("delete_feedback_log", parents=[common_parent], help="Delete one feedback log")
    delete_feedback_parser.add_argument("--feedback-log-id", type=int, required=True, help="Feedback log id")
    delete_feedback_parser.set_defaults(func=handle_delete_feedback_log)

    clear_feedback_parser = subparsers.add_parser("clear_feedback_logs", parents=[common_parent], help="Delete all feedback logs")
    clear_feedback_parser.set_defaults(func=handle_clear_feedback_logs)

    export_feedback_parser = subparsers.add_parser("export_feedback_logs", parents=[common_parent], help="Export feedback logs as JSON")
    export_feedback_parser.add_argument("--output-path", default="data/feedback_logs_export.json", help="Export path")
    export_feedback_parser.set_defaults(func=handle_export_feedback_logs)

    rebuild_feedback_parser = subparsers.add_parser("rebuild_feedback_learning", parents=[common_parent], help="Rebuild adaptive boosts from retained logs")
    rebuild_feedback_parser.add_argument("--min-occurrences", type=int, default=2, help="Minimum token support")
    rebuild_feedback_parser.set_defaults(func=handle_rebuild_feedback_learning)

    embedding_cache_stats_parser = subparsers.add_parser("embedding_cache_stats", parents=[common_parent], help="Show persistent embedding cache stats")
    embedding_cache_stats_parser.set_defaults(func=handle_embedding_cache_stats)

    clear_embedding_cache_parser = subparsers.add_parser("clear_embedding_cache", parents=[common_parent], help="Clear persistent embedding cache")
    clear_embedding_cache_parser.set_defaults(func=handle_clear_embedding_cache)

    rebuild_embedding_cache_parser = subparsers.add_parser("rebuild_embedding_cache", parents=[common_parent], help="Rebuild persistent embedding cache")
    rebuild_embedding_cache_parser.add_argument(
        "--model-name",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="sentence-transformers model name",
    )
    rebuild_embedding_cache_parser.add_argument("--clear-first", action="store_true", help="Clear existing embedding cache before rebuild")
    rebuild_embedding_cache_parser.set_defaults(func=handle_rebuild_embedding_cache)

    migrate_embedding_cache_parser = subparsers.add_parser("migrate_embedding_cache", parents=[common_parent], help="Migrate legacy SQLite embedding cache to HDF5")
    migrate_embedding_cache_parser.add_argument("--clear-target-first", action="store_true", help="Clear the HDF5 target before migration")
    migrate_embedding_cache_parser.set_defaults(func=handle_migrate_embedding_cache)

    migrate_confirmed_examples_parser = subparsers.add_parser(
        "migrate_confirmed_example_embeddings",
        parents=[common_parent],
        help="Backfill confirmed example embeddings into HDF5",
    )
    migrate_confirmed_examples_parser.add_argument(
        "--prune-legacy-json",
        action="store_true",
        help="Clear confirmed_examples.embedding_json after successful HDF5 migration",
    )
    migrate_confirmed_examples_parser.set_defaults(func=handle_migrate_confirmed_example_embeddings)

    snapshot_parser = subparsers.add_parser("create_snapshot", parents=[common_parent], help="Create a safety snapshot")
    snapshot_parser.add_argument("--reason", default="manual_snapshot", help="Snapshot reason")
    snapshot_parser.set_defaults(func=handle_create_snapshot)

    return parser


def handle_init_db(args: argparse.Namespace) -> None:
    """Initialize DB and seed rules."""
    config = load_runtime_config(args)
    repository = build_repository(args.db, config)
    repository.initialize_database()
    repository.seed_rules_from_categories(load_categories(Path(args.categories)))
    repository.seed_default_category_profiles()
    repository.save_config_version("app_config", config.version, config.to_dict())
    print(f"DB ready: {Path(args.db).resolve()}")


def handle_classify(args: argparse.Namespace) -> None:
    """Classify supported files from the input folder."""
    total_start = time.perf_counter()
    config = load_runtime_config(args)
    repository = build_repository(args.db, config)
    repository.initialize_database()
    repository.seed_rules_from_categories(load_categories(Path(args.categories)))
    repository.seed_default_category_profiles()
    taxonomy = load_runtime_taxonomy(args)

    input_dir = Path(args.input_dir)
    ensure_input_directory(input_dir)
    files = discover_supported_files(input_dir)
    if not files:
        print(f"No txt/pdf files found: {input_dir.resolve()}")
        return

    classifier = HybridClassifier(
        repository=repository,
        embedder=build_embedder(config, model_name=args.model_name),
        rule_classifier=RuleBasedClassifier(repository),
        taxonomy=taxonomy,
        rule_skip_embedding_threshold=FAST_RULE_SKIP_EMBEDDING_THRESHOLD if args.fast else 0.85,
        min_rule_matches_for_skip=FAST_MIN_RULE_MATCHES_FOR_SKIP if args.fast else 3,
        use_embedding_for_no_rule=not args.fast,
        low_rule_confidence_threshold=FAST_LOW_RULE_CONFIDENCE_THRESHOLD if args.fast else 0.20,
        feature_extractor=DocumentFeatureExtractor(version=config.features.extractor_version),
        type_classifier=TypeClassifier(
            version=config.ml.type_classifier_version,
            min_examples=config.ml.min_training_examples,
            filename_weight=config.ml.filename_weight,
        ),
    )

    if args.fast:
        summary = process_files_fast(
            repository=repository,
            classifier=classifier,
            files=files,
            review=args.review,
            max_workers=max(1, min(args.workers, MAX_FAST_WORKERS)),
            ocr_workers=max(1, min(args.ocr_workers, MAX_OCR_WORKERS)),
            ocr_min_chars=max(0, args.ocr_min_chars),
            total_start=total_start,
            use_llm=args.use_llm,
            llm_model=args.llm_model,
            config=config,
        )
    else:
        summary = process_files_sequential(
            repository=repository,
            classifier=classifier,
            files=files,
            review=args.review,
            ocr_workers=max(1, min(args.ocr_workers, MAX_OCR_WORKERS)),
            ocr_min_chars=max(0, args.ocr_min_chars),
            total_start=total_start,
            use_llm=args.use_llm,
            llm_model=args.llm_model,
            config=config,
        )

    print_performance_summary(summary)


def process_files_fast(
    repository: ClassificationRepository,
    classifier: HybridClassifier,
    files: list[Path],
    review: bool,
    max_workers: int,
    ocr_workers: int,
    ocr_min_chars: int,
    total_start: float,
    use_llm: bool,
    llm_model: str,
    config: AppConfig,
) -> dict[str, Any]:
    """Run extraction and rule scoring in worker processes."""
    rules = serialize_active_rules(repository)
    print(
        f"Classify start: {len(files)} files, fast=True, "
        f"workers={max_workers}, ocr_workers={ocr_workers}, ocr_min_chars={ocr_min_chars}"
    )

    summary = create_summary(total_files=len(files))
    llm_runtime: dict[str, bool] = {"available": True}
    worker_results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_file_fast, str(file_path), rules): file_path
            for file_path in files
        }
        for future in as_completed(futures):
            try:
                worker_result = future.result()
            except Exception as error:
                summary["failed"] += 1
                print(f"[failed] {futures[future].name}: worker failed - {error}")
                continue

            if not worker_result["ok"]:
                summary["failed"] += 1
                print_failed_worker_result(worker_result)
                continue

            worker_results.append(worker_result)

    apply_parallel_ocr_fallback(worker_results, rules, ocr_workers=ocr_workers, ocr_min_chars=ocr_min_chars)
    precompute_fast_embeddings(repository, classifier, worker_results)

    for worker_result in worker_results:
        result, classification_id = finalize_worker_result(
            repository=repository,
            classifier=classifier,
            worker_result=worker_result,
            use_llm=use_llm,
            llm_model=llm_model,
            llm_runtime=llm_runtime,
            config=config,
        )
        summary["success"] += 1
        summary["embedding_used"] += int(result.embedding_used)
        summary["embedding_skipped"] += int(not result.embedding_used)
        if isinstance(result.processing_profile, dict):
            analysis = result.processing_profile.get("analysis", {})
            if isinstance(analysis, dict):
                summary["slowest_files"].append(
                    {
                        "file_name": str(worker_result["file_name"]),
                        "total_time": float(analysis.get("total_time", 0.0)),
                        "dominant_stage": str(analysis.get("dominant_stage", "unknown")),
                    }
                )
        print_classification_result(
            worker_result["file_name"],
            result,
            worker_result["timings"],
            ocr_status=str(worker_result.get("ocr_status", "not_checked")),
            ocr_reason=str(worker_result.get("ocr_reason", "")),
        )

        if review:
            review_and_save_feedback(
                repository=repository,
                file_id=int(worker_result["file_id"]),
                classification_id=classification_id,
                predicted_category=result.predicted_category,
                normalized_text=str(worker_result["evidence_text"]),
                result=result,
                classifier=classifier,
            )

    summary["elapsed"] = time.perf_counter() - total_start
    return summary


def precompute_fast_embeddings(
    repository: ClassificationRepository,
    classifier: HybridClassifier,
    worker_results: list[dict[str, Any]],
) -> None:
    """Batch embedding calls for ambiguous fast-mode files."""
    confirmed_examples = repository.fetch_confirmed_examples()
    if not confirmed_examples:
        return

    categories = repository.list_categories()
    embedding_targets: list[dict[str, Any]] = []
    for worker_result in worker_results:
        rule_breakdown = worker_result["rule_breakdown"]
        rule_scores = classifier._normalize_scores(rule_breakdown["scores"], categories)
        top_rule_category = classifier._pick_top_category(rule_scores, categories)
        top_rule_score = rule_scores.get(top_rule_category, 0.0)
        top_rule_match_count = len(rule_breakdown["matches"].get(top_rule_category, []))
        strong_rule_match = (
            top_rule_score >= classifier.rule_skip_embedding_threshold
            and top_rule_match_count >= classifier.min_rule_matches_for_skip
        )
        can_use_embedding = classifier.use_embedding_for_no_rule
        if not strong_rule_match and can_use_embedding:
            embedding_targets.append(worker_result)

    if not embedding_targets:
        return

    try:
        texts = [
            str(worker_result.get("document_features", {}).get("compressed_text") or worker_result["evidence_text"])
            for worker_result in embedding_targets
        ]
        file_hashes = [str(worker_result["xxhash64"]) for worker_result in embedding_targets]
        if isinstance(classifier.embedder, SentenceTransformerEmbedder):
            embeddings = classifier.embedder.encode_many(
                texts,
                repository=repository,
                file_hashes=file_hashes,
                text_kind="compressed_query",
                embedding_version="2.1-compressed",
            )
            embedding_meta_rows = classifier.embedder.get_last_batch_encode_meta()
        else:
            embeddings = classifier.embedder.encode_many(texts)
            embedding_meta_rows = [{} for _ in embeddings]
    except Exception as error:
        print(f"Embedding batch skipped: {error}")
        return

    for worker_result, embedding, embedding_meta in zip(embedding_targets, embeddings, embedding_meta_rows):
        worker_result["precomputed_embedding"] = embedding
        worker_result["precomputed_embedding_meta"] = embedding_meta


def process_files_sequential(
    repository: ClassificationRepository,
    classifier: HybridClassifier,
    files: list[Path],
    review: bool,
    ocr_workers: int,
    ocr_min_chars: int,
    total_start: float,
    use_llm: bool,
    llm_model: str,
    config: AppConfig,
) -> dict[str, Any]:
    """Run classification sequentially with batched OCR fallback for empty PDFs."""
    print(
        f"Classify start: {len(files)} files, fast=False, "
        f"ocr_workers={ocr_workers}, ocr_min_chars={ocr_min_chars}"
    )
    summary = create_summary(total_files=len(files))
    llm_runtime: dict[str, bool] = {"available": True}
    pending_ocr_records: list[dict[str, Any]] = []

    for file_path in files:
        prepared = prepare_sequential_record(file_path, ocr_min_chars=ocr_min_chars)
        if not prepared["ok"]:
            summary["failed"] += 1
            print(f"[failed] {file_path.name}: {prepared['error']}")
            continue

        if prepared["pending_ocr"]:
            pending_ocr_records.append(prepared)
            continue

        result = classify_prepared_record(
            repository=repository,
            classifier=classifier,
            prepared=prepared,
            review=review,
            use_llm=use_llm,
            llm_model=llm_model,
            llm_runtime=llm_runtime,
            config=config,
        )
        summary["success"] += 1
        summary["embedding_used"] += int(result.embedding_used)
        summary["embedding_skipped"] += int(not result.embedding_used)
        if isinstance(result.processing_profile, dict):
            analysis = result.processing_profile.get("analysis", {})
            if isinstance(analysis, dict):
                summary["slowest_files"].append(
                    {
                        "file_name": str(prepared["file_name"]),
                        "total_time": float(analysis.get("total_time", 0.0)),
                        "dominant_stage": str(analysis.get("dominant_stage", "unknown")),
                    }
                )

    apply_parallel_ocr_fallback(
        pending_ocr_records,
        serialize_active_rules(repository),
        ocr_workers=ocr_workers,
        ocr_min_chars=ocr_min_chars,
    )
    for prepared in pending_ocr_records:
        if not str(prepared["evidence_text"]).strip():
            summary["failed"] += 1
            detail = prepared["ocr_error"] or "no text after OCR"
            print(f"[failed] {prepared['file_name']}: OCR fallback failed - {detail}")
            continue

        result = classify_prepared_record(
            repository=repository,
            classifier=classifier,
            prepared=prepared,
            review=review,
            use_llm=use_llm,
            llm_model=llm_model,
            llm_runtime=llm_runtime,
            config=config,
        )
        summary["success"] += 1
        summary["embedding_used"] += int(result.embedding_used)
        summary["embedding_skipped"] += int(not result.embedding_used)
        if isinstance(result.processing_profile, dict):
            analysis = result.processing_profile.get("analysis", {})
            if isinstance(analysis, dict):
                summary["slowest_files"].append(
                    {
                        "file_name": str(prepared["file_name"]),
                        "total_time": float(analysis.get("total_time", 0.0)),
                        "dominant_stage": str(analysis.get("dominant_stage", "unknown")),
                    }
                )

    summary["elapsed"] = time.perf_counter() - total_start
    return summary


def finalize_worker_result(
    repository: ClassificationRepository,
    classifier: HybridClassifier,
    worker_result: dict[str, Any],
    use_llm: bool,
    llm_model: str,
    llm_runtime: dict[str, bool],
    config: AppConfig,
) -> tuple[ClassificationResult, int]:
    """Save worker output in DB and finish scoring in the main process."""
    timings = worker_result.setdefault("timings", {})
    finalize_start = time.perf_counter()
    duplicate_start = time.perf_counter()
    duplicate_of_file_id = repository.find_duplicate_file_id(
        str(worker_result["xxhash64"]),
        str(Path(worker_result["file_path"]).resolve()),
    )
    timings["duplicate_lookup"] = time.perf_counter() - duplicate_start
    upsert_start = time.perf_counter()
    file_id = repository.upsert_file(
        file_path=str(Path(worker_result["file_path"]).resolve()),
        file_name=str(worker_result["file_name"]),
        file_ext=str(worker_result["file_ext"]),
        file_size=int(worker_result["file_size"]),
        xxhash64=str(worker_result["xxhash64"]),
        duplicate_of_file_id=duplicate_of_file_id,
        extracted_text=str(worker_result["evidence_text"]),
    )
    timings["db_upsert"] = time.perf_counter() - upsert_start
    worker_result["file_id"] = file_id
    classify_start = time.perf_counter()
    result = classifier.classify_with_rule_breakdown(
        file_hash=str(worker_result["xxhash64"]),
        text=str(worker_result["evidence_text"]),
        duplicate_of_file_id=duplicate_of_file_id,
        rule_breakdown=worker_result["rule_breakdown"],
        precomputed_query_embedding=worker_result.get("precomputed_embedding"),
        precomputed_embedding_meta=worker_result.get("precomputed_embedding_meta"),
        file_name=str(worker_result["file_name"]),
        file_id=file_id,
        document_features=worker_result.get("document_features"),
    )
    timings["classification"] = time.perf_counter() - classify_start
    result = maybe_refine_with_llm(
        result=result,
        evidence_text=str(worker_result["evidence_text"]),
        use_llm=use_llm,
        llm_model=llm_model,
        llm_runtime=llm_runtime,
        config=config,
    )
    result = apply_ocr_reasoning(
        result=result,
        ocr_used=bool(worker_result.get("ocr_used")),
        ocr_pages=int(worker_result.get("ocr_pages", 0)),
    )
    persist_start = time.perf_counter()
    timings["total"] = timings.get("worker_time", 0.0) + (time.perf_counter() - finalize_start)
    result = attach_result_performance_profile(
        result,
        stage_timings={key.replace("_time", ""): float(value) for key, value in timings.items()},
        text_length=len(str(worker_result["evidence_text"])),
        file_size=int(worker_result["file_size"]),
        ocr_status=str(worker_result.get("ocr_status", "not_checked")),
        ocr_pages=int(worker_result.get("ocr_pages", 0)),
        duplicate_detected=duplicate_of_file_id is not None,
    )
    result = attach_cluster_candidate_if_needed(repository, result, file_id=file_id, config=config)
    classification_id = classifier.persist_classification(file_id=file_id, result=result)
    timings["db_persist"] = time.perf_counter() - persist_start
    timings["total"] = timings.get("worker_time", 0.0) + timings["duplicate_lookup"] + timings["db_upsert"] + timings["classification"] + timings["db_persist"]
    result = attach_result_performance_profile(
        result,
        stage_timings={key.replace("_time", ""): float(value) for key, value in timings.items()},
        text_length=len(str(worker_result["evidence_text"])),
        file_size=int(worker_result["file_size"]),
        ocr_status=str(worker_result.get("ocr_status", "not_checked")),
        ocr_pages=int(worker_result.get("ocr_pages", 0)),
        duplicate_detected=duplicate_of_file_id is not None,
    )
    worker_result["performance"] = result.processing_profile
    return result, classification_id


def prepare_sequential_record(file_path: Path, ocr_min_chars: int) -> dict[str, Any]:
    """Read one file and prepare metadata before classification."""
    file_start = time.perf_counter()
    try:
        extract_start = time.perf_counter()
        evidence_text = extract_text_from_file(file_path, fast=False)
        read_extract_time = time.perf_counter() - extract_start
    except Exception as error:
        return {
            "ok": False,
            "file_path": str(file_path),
            "file_name": file_path.name,
            "error": f"read failed - {error}",
        }

    preprocess_start = time.perf_counter()
    normalized_text = normalize_text(evidence_text)
    preprocess_time = time.perf_counter() - preprocess_start

    try:
        file_hash = compute_xxhash64(file_path)
    except Exception as error:
        return {
            "ok": False,
            "file_path": str(file_path),
            "file_name": file_path.name,
            "error": f"hash failed - {error}",
        }

    record = {
        "ok": True,
        "file_path": str(file_path),
        "file_name": file_path.name,
        "file_ext": file_path.suffix.lower(),
        "file_size": file_path.stat().st_size,
        "xxhash64": file_hash,
        "evidence_text": normalized_text,
        "document_features": DocumentFeatureExtractor().extract(
            file_name=file_path.name,
            file_ext=file_path.suffix.lower(),
            text=normalized_text,
            file_size=file_path.stat().st_size,
            file_path=file_path,
        ).to_storage_dict(),
        "rule_breakdown": {"scores": {}, "matches": {}},
        "ocr_used": False,
        "ocr_pages": 0,
        "ocr_error": "",
        "ocr_status": "not_checked",
        "ocr_reason": "",
        "precomputed_embedding": None,
        "timings": {
            "read_extract_time": read_extract_time,
            "preprocess_time": preprocess_time,
            "rule_time": 0.0,
            "ocr_time": 0.0,
            "worker_time": time.perf_counter() - file_start,
        },
        "error": "",
    }
    apply_ocr_plan(record, ocr_min_chars=ocr_min_chars)
    return record


def classify_prepared_record(
    repository: ClassificationRepository,
    classifier: HybridClassifier,
    prepared: dict[str, Any],
    review: bool,
    use_llm: bool,
    llm_model: str,
    llm_runtime: dict[str, bool],
    config: AppConfig,
) -> ClassificationResult:
    """Classify a prepared record and save the result."""
    timings = prepared.setdefault("timings", {})
    duplicate_start = time.perf_counter()
    duplicate_of_file_id = repository.find_duplicate_file_id(
        str(prepared["xxhash64"]),
        str(Path(prepared["file_path"]).resolve()),
    )
    timings["duplicate_lookup"] = time.perf_counter() - duplicate_start
    upsert_start = time.perf_counter()
    file_id = repository.upsert_file(
        file_path=str(Path(prepared["file_path"]).resolve()),
        file_name=str(prepared["file_name"]),
        file_ext=str(prepared["file_ext"]),
        file_size=int(prepared["file_size"]),
        xxhash64=str(prepared["xxhash64"]),
        duplicate_of_file_id=duplicate_of_file_id,
        extracted_text=str(prepared["evidence_text"]),
    )
    timings["db_upsert"] = time.perf_counter() - upsert_start

    classify_start = time.perf_counter()
    result = classifier.classify_file(
        file_id=file_id,
        file_hash=str(prepared["xxhash64"]),
        text=str(prepared["evidence_text"]),
        duplicate_of_file_id=duplicate_of_file_id,
        file_name=str(prepared["file_name"]),
    )
    timings["classification"] = time.perf_counter() - classify_start
    result = maybe_refine_with_llm(
        result=result,
        evidence_text=str(prepared["evidence_text"]),
        use_llm=use_llm,
        llm_model=llm_model,
        llm_runtime=llm_runtime,
        config=config,
    )
    result = apply_ocr_reasoning(
        result=result,
        ocr_used=bool(prepared.get("ocr_used")),
        ocr_pages=int(prepared.get("ocr_pages", 0)),
    )

    persist_start = time.perf_counter()
    result = attach_result_performance_profile(
        result,
        stage_timings={key.replace("_time", ""): float(value) for key, value in timings.items()},
        text_length=len(str(prepared["evidence_text"])),
        file_size=int(prepared["file_size"]),
        ocr_status=str(prepared.get("ocr_status", "not_checked")),
        ocr_pages=int(prepared.get("ocr_pages", 0)),
        duplicate_detected=duplicate_of_file_id is not None,
    )
    result = attach_cluster_candidate_if_needed(repository, result, file_id=file_id, config=config)
    classification_id = classifier.persist_classification(file_id=file_id, result=result)
    timings["db_persist"] = time.perf_counter() - persist_start
    timings["total"] = sum(float(value) for value in timings.values())
    result = attach_result_performance_profile(
        result,
        stage_timings={key.replace("_time", ""): float(value) for key, value in timings.items()},
        text_length=len(str(prepared["evidence_text"])),
        file_size=int(prepared["file_size"]),
        ocr_status=str(prepared.get("ocr_status", "not_checked")),
        ocr_pages=int(prepared.get("ocr_pages", 0)),
        duplicate_detected=duplicate_of_file_id is not None,
    )
    prepared["performance"] = result.processing_profile
    print_classification_result(
        str(prepared["file_name"]),
        result,
        prepared["timings"],
        ocr_status=str(prepared.get("ocr_status", "not_checked")),
        ocr_reason=str(prepared.get("ocr_reason", "")),
    )

    if review:
        review_and_save_feedback(
            repository=repository,
            file_id=file_id,
            classification_id=classification_id,
            predicted_category=result.predicted_category,
            normalized_text=str(prepared["evidence_text"]),
            result=result,
            classifier=classifier,
        )
    return result


def apply_ocr_plan(record: dict[str, Any], ocr_min_chars: int) -> None:
    """Decide whether OCR is needed and keep a loggable reason."""
    decision = explain_ocr_decision(
        file_path=str(record["file_path"]),
        extracted_text=str(record.get("evidence_text", "")),
        classification_hint=record.get("classification_hint"),
        min_text_length=ocr_min_chars,
    )
    record["classification_hint"] = decision["classification_hint"]
    record["pending_ocr"] = bool(decision["run_ocr"])
    record["ocr_reason"] = str(decision["reason"])

    if decision["run_ocr"]:
        record["ocr_status"] = "queued"
        print(f"[ocr-run] {record['file_name']}: {record['ocr_reason']}")
        return

    record["ocr_status"] = "skipped"
    if decision["classification_hint"]:
        hint_evidence = decision["hint_evidence"] or build_filename_hint_evidence(
            str(record["file_path"]),
            str(decision["classification_hint"]),
        )
        current_text = str(record.get("evidence_text", "")).strip()
        record["evidence_text"] = f"{hint_evidence} {current_text}".strip()
    print(f"[ocr-skip] {record['file_name']}: {record['ocr_reason']}")


def apply_parallel_ocr_fallback(
    records: list[dict[str, Any]],
    rules: list[dict[str, Any]],
    ocr_workers: int,
    ocr_min_chars: int,
) -> None:
    """OCR only the PDF files that still need it after prechecks."""
    for record in records:
        if "pending_ocr" not in record:
            apply_ocr_plan(record, ocr_min_chars=ocr_min_chars)

    targets = [record for record in records if record.get("pending_ocr")]
    if not targets:
        return

    max_workers = max(1, min(len(targets), ocr_workers))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(ocr_pdf_file, str(record["file_path"]), OCR_MAX_PAGES): record
            for record in targets
        }
        for future in as_completed(futures):
            record = futures[future]
            try:
                ocr_result = future.result()
            except Exception as error:
                record["ocr_error"] = str(error)
                record["ocr_status"] = "failed"
                print(f"[ocr-failed] {record['file_name']}: {record['ocr_error']}")
                continue
            merge_ocr_result(record, ocr_result, rules)


def merge_ocr_result(
    record: dict[str, Any],
    ocr_result: dict[str, Any],
    rules: list[dict[str, Any]],
) -> None:
    """Update a prepared record with OCR text and refreshed rule scores."""
    record["timings"]["ocr_time"] = float(ocr_result.get("elapsed", 0.0))

    if not ocr_result.get("ok"):
        record["ocr_error"] = str(ocr_result.get("error", "OCR failed"))
        record["ocr_status"] = "failed"
        print(f"[ocr-failed] {record['file_name']}: {record['ocr_error']}")
        return

    normalized_text = normalize_text(str(ocr_result.get("text", "")))
    if not normalized_text:
        record["ocr_error"] = "OCR returned no text"
        record["ocr_pages"] = int(ocr_result.get("pages_scanned", 0))
        record["ocr_status"] = "empty"
        print(f"[ocr-empty] {record['file_name']}: pages={record['ocr_pages']}")
        return

    rule_start = time.perf_counter()
    rule_input_text = build_rule_input_text(normalized_text, str(record["file_name"]))
    rule_breakdown = score_text_with_rules(rule_input_text, rules)

    record["evidence_text"] = normalized_text
    record["document_features"] = DocumentFeatureExtractor().extract(
        file_name=str(record["file_name"]),
        file_ext=str(record.get("file_ext", "")),
        text=normalized_text,
        file_size=int(record.get("file_size", 0)),
        file_path=str(record["file_path"]),
    ).to_storage_dict()
    record["rule_breakdown"] = rule_breakdown
    record["ocr_used"] = True
    record["ocr_pages"] = int(ocr_result.get("pages_scanned", 0))
    record["ocr_error"] = ""
    record["pending_ocr"] = False
    record["ocr_status"] = "used"
    record["timings"]["rule_time"] = time.perf_counter() - rule_start
    record["timings"]["worker_time"] += float(ocr_result.get("elapsed", 0.0))
    print(f"[ocr-used] {record['file_name']}: pages={record['ocr_pages']}, chars={len(normalized_text)}")


def apply_ocr_reasoning(
    result: ClassificationResult,
    ocr_used: bool,
    ocr_pages: int,
) -> ClassificationResult:
    """Append OCR usage to the visible reason string."""
    if not ocr_used:
        return result

    return replace(
        result,
        reasoning=f"{result.reasoning} | ocr=used(pages={ocr_pages})",
        ocr_used=True,
        explanation={**result.explanation, "ocr_used": True, "ocr_pages": ocr_pages},
    )


def attach_result_performance_profile(
    result: ClassificationResult,
    *,
    stage_timings: dict[str, float],
    text_length: int,
    file_size: int,
    ocr_status: str,
    ocr_pages: int,
    duplicate_detected: bool,
) -> ClassificationResult:
    """Attach a unified performance profile to a result object."""
    classifier_profile = result.processing_profile if isinstance(result.processing_profile, dict) else {}
    classifier_stage_timings = classifier_profile.get("stage_timings", {}) if isinstance(classifier_profile, dict) else {}
    merged_timings = dict(stage_timings)
    for key, value in classifier_stage_timings.items():
        if key == "total":
            continue
        merged_timings.setdefault(f"classifier_{key}", float(value))

    analysis = build_file_latency_analysis(
        merged_timings,
        text_length=text_length,
        file_size=file_size,
        ocr_used=ocr_status == "used",
        ocr_status=ocr_status,
        ocr_pages=ocr_pages,
        embedding_used=result.embedding_used,
        embedding_cache_hit=classifier_profile.get("embedding_meta", {}).get("cache_hit")
        if isinstance(classifier_profile.get("embedding_meta", {}), dict)
        else None,
        strong_rule_match=bool(classifier_profile.get("strong_rule_match")) if isinstance(classifier_profile, dict) else False,
        review_required=result.review_required,
        matched_rules_count=len(result.matched_rules),
        llm_used=result.llm_used,
        duplicate_detected=duplicate_detected,
    )
    profile = {
        "stage_timings": merged_timings,
        "analysis": analysis,
        "classifier_profile": classifier_profile,
    }
    return replace(result, processing_profile=profile)


def attach_cluster_candidate_if_needed(
    repository: ClassificationRepository,
    result: ClassificationResult,
    *,
    file_id: int,
    config: AppConfig,
) -> ClassificationResult:
    """Create a conservative pending category candidate for review/misc groups."""
    if not config.clustering.enabled or not result.review_required:
        return result
    finder = ClusterCandidateFinder(
        min_cluster_size=config.clustering.min_cluster_size,
        max_candidates=config.clustering.max_candidates,
    )
    rows = repository.fetch_cluster_candidate_rows()
    rows.append(
        {
            "file_id": file_id,
            "file_name": "",
            "text": "",
            "predicted_category": result.predicted_category,
            "predicted_type": result.predicted_type,
            "review_required": result.review_required,
            "compressed_text": " ".join(
                str(snippet) for snippet in result.evidence_snippets if isinstance(snippet, str)
            ),
        }
    )
    candidates = finder.find_candidates(rows)
    for candidate in candidates:
        if file_id not in candidate.representative_file_ids:
            continue
        candidate_id = repository.insert_category_candidate(
            source=str(candidate.evidence.get("source", "cluster")),
            suggested_name=candidate.suggested_name,
            representative_file_ids=candidate.representative_file_ids,
            evidence=candidate.evidence,
            status="pending",
        )
        return replace(
            result,
            cluster_candidate_id=candidate_id,
            review_reasons=[*result.review_reasons, "pending_category_candidate"],
        )
    return result


def serialize_active_rules(repository: ClassificationRepository) -> list[dict[str, Any]]:
    """Return active rules as pickle-safe dictionaries."""
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


def create_summary(total_files: int) -> dict[str, Any]:
    """Create a performance summary accumulator."""
    return {
        "total_files": total_files,
        "success": 0,
        "failed": 0,
        "elapsed": 0.0,
        "embedding_used": 0,
        "embedding_skipped": 0,
        "slowest_files": [],
    }


def print_failed_worker_result(worker_result: dict[str, Any]) -> None:
    """Print a failed worker result without stopping the run."""
    timings = worker_result.get("timings", {})
    print("")
    print("=" * 48)
    print(f"file: {worker_result.get('file_name', 'unknown')}")
    print("category: review_required")
    print("confidence: 0.000")
    print("matched_rules: none")
    print("similarity: skipped")
    print("review_required: yes")
    print(f"error: {worker_result.get('error', '')}")
    print(f"time: {float(timings.get('worker_time', 0.0)):.2f}s")


def print_classification_result(
    file_name: str,
    result: ClassificationResult,
    timings: dict[str, float],
    ocr_status: str = "not_checked",
    ocr_reason: str = "",
) -> None:
    """Print a concise classification result."""
    matched_rules = ", ".join(result.matched_rules) if result.matched_rules else "none"
    review_text = "yes" if result.review_required else "no"
    similarity_text = f"{result.similarity_score:.3f}" if result.embedding_used else "skipped"
    analysis = {}
    if isinstance(result.processing_profile, dict):
        analysis = result.processing_profile.get("analysis", {})

    print("")
    print("=" * 48)
    print(f"file: {file_name}")
    print(f"category: {result.large_category}/{result.middle_category or result.predicted_category}")
    print(f"predicted_type: {result.predicted_type or result.predicted_category}")
    print(f"type_confidence: {result.type_confidence:.3f}")
    print(f"confidence: {result.confidence:.3f}")
    print(f"matched_rules: {matched_rules}")
    print(f"similarity: {similarity_text}")
    print(f"review_required: {review_text}")
    if result.review_reasons:
        print("review_reasons: " + " | ".join(result.review_reasons))
    if result.suggested_tags:
        tag_text = ", ".join(f"{item.get('tag')}:{float(item.get('confidence', 0.0)):.2f}" for item in result.suggested_tags)
        print(f"suggested_tags: {tag_text}")
    if result.cluster_candidate_id is not None:
        print(f"cluster_candidate_id: {result.cluster_candidate_id}")
    print(f"processing: {get_processing_trace_text(result)}")
    print(f"primary_method: {get_processing_method_label(result)}")
    ocr_text = ocr_status
    if ocr_reason:
        ocr_text = f"{ocr_status} ({ocr_reason})"
    print(f"ocr: {ocr_text}")
    print(
        "scores: "
        f"rule={result.rule_score:.3f}, "
        f"embedding={result.embedding_score:.3f}, "
        f"feedback={result.feedback_score:.3f}, "
        f"llm={result.llm_score:.3f}, "
        f"final={result.final_score:.3f}"
    )
    print(
        "hierarchy_confidence: "
        f"large={result.large_confidence:.3f}, "
        f"middle={result.middle_confidence:.3f}, "
        f"small={result.small_confidence:.3f}"
    )
    print(
        "timing: "
        f"read={float(timings.get('read_extract_time', 0.0)):.2f}s, "
        f"preprocess={float(timings.get('preprocess_time', 0.0)):.3f}s, "
        f"rule={float(timings.get('rule_time', 0.0)):.3f}s, "
        f"ocr={float(timings.get('ocr_time', 0.0)):.2f}s, "
        f"total={float(timings.get('worker_time', 0.0)):.2f}s"
    )
    if isinstance(analysis, dict) and analysis:
        print(
            "latency: "
            f"dominant={analysis.get('dominant_stage', 'unknown')}, "
            f"total={float(analysis.get('total_time', 0.0)):.2f}s"
        )
        reasons = analysis.get("reasons", [])
        if isinstance(reasons, list) and reasons:
            print("latency_reasons: " + " | ".join(str(reason) for reason in reasons))
    print(f"reason: {result.reasoning}")


def print_performance_summary(summary: dict[str, Any]) -> None:
    """Print whole-run performance metrics."""
    total_files = int(summary["total_files"])
    elapsed = float(summary["elapsed"])
    average = elapsed / total_files if total_files else 0.0
    target = "yes" if elapsed <= FAST_TARGET_SECONDS else "no"

    print("")
    print("Performance summary")
    print(f"- total_files: {total_files}")
    print(f"- success: {summary['success']}")
    print(f"- failed: {summary['failed']}")
    print(f"- total_time: {elapsed:.2f}s")
    print(f"- avg_time_per_file: {average:.2f}s")
    print(f"- target_10s_met: {target}")
    print(f"- embedding_used: {summary['embedding_used']}")
    print(f"- embedding_skipped: {summary['embedding_skipped']}")
    slowest_files = sorted(summary.get("slowest_files", []), key=lambda row: row["total_time"], reverse=True)[:5]
    if slowest_files:
        print("- slowest_files:")
        for row in slowest_files:
            print(
                f"  * {row['file_name']} | {float(row['total_time']):.2f}s | "
                f"dominant_stage={row['dominant_stage']}"
            )


def handle_suggest_rules(args: argparse.Namespace) -> None:
    """Suggest new rules from correction logs."""
    repository = build_repository(args.db, load_runtime_config(args))
    repository.initialize_database()

    classifier = HybridClassifier(
        repository=repository,
        embedder=build_embedder(load_runtime_config(args)),
        rule_classifier=RuleBasedClassifier(repository),
    )

    suggestions = classifier.suggest_rules(min_occurrences=args.min_occurrences)
    if not suggestions:
        print("No rule suggestions.")
        return

    print(f"Rule suggestions: {len(suggestions)}")
    for suggestion in suggestions:
        print(
            f"- {suggestion['category']} | {suggestion['pattern']} | "
            f"weight={suggestion['weight']} | support={suggestion['support_count']} | "
            f"from={suggestion['predicted_category']}"
        )

        if args.save_candidates:
            repository.insert_rule_candidate(
                category=suggestion["category"],
                rule_type=suggestion["rule_type"],
                pattern=suggestion["pattern"],
                weight=suggestion["weight"],
                source=suggestion["source"],
                evidence={
                    "predicted_category": suggestion["predicted_category"],
                    "support_count": suggestion["support_count"],
                    "evidence": suggestion["evidence"],
                },
            )

    if args.save_candidates:
        print("Saved candidate rules.")


def handle_stats(args: argparse.Namespace) -> None:
    """Print DB stats."""
    repository = build_repository(args.db, load_runtime_config(args))
    repository.initialize_database()
    stats = repository.get_stats()

    print("DB stats")
    print(f"- schema_version: {stats['schema_version']}")
    print(f"- files: {stats['files_count']}")
    print(f"- classifications: {stats['classifications_count']}")
    print(f"- feedback_logs: {stats['feedback_logs_count']}")
    print(f"- confirmed_examples: {stats['confirmed_examples_count']}")
    print(f"- rules: {stats['rules_count']}")
    print(f"- move_batches: {stats['move_batches_count']}")
    print(f"- snapshots: {stats['snapshots_count']}")
    print(f"- adaptive_rules: {stats['adaptive_rules_count']}")
    print(f"- embedding_cache: {stats['embedding_cache_count']}")
    print(f"- document_features: {stats.get('document_features_count', 0)}")
    print(f"- category_candidates: {stats.get('category_candidates_count', 0)}")
    print(f"- document_tags: {stats.get('document_tags_count', 0)}")
    print(f"- category_profiles: {stats.get('category_profiles_count', 0)}")
    print("")
    print("Recent feedback")

    if not stats["recent_feedback"]:
        print("- none")
        return

    for row in stats["recent_feedback"]:
        print(
            f"- {row['created_at']} | {row['file_name']} | "
            f"{row['predicted_category']} -> {row['final_category']} | {row['feedback_action']}"
        )


def handle_add_category_profile(args: argparse.Namespace) -> None:
    repository = build_repository(args.db, load_runtime_config(args))
    repository.initialize_database()
    tags = [tag.strip() for tag in str(args.tags).split(",") if tag.strip()]
    profile_id = repository.add_category_profile(
        category_type=args.category_type.strip(),
        profile_text=args.text.strip(),
        tags=tags,
        weight=args.weight,
        synthetic_count=args.synthetic_count,
    )
    print(f"category_profile_added: id={profile_id}, type={args.category_type.strip()}")


def handle_list_category_profiles(args: argparse.Namespace) -> None:
    repository = build_repository(args.db, load_runtime_config(args))
    repository.initialize_database()
    rows = repository.list_category_profiles(include_inactive=args.include_inactive)
    if not rows:
        print("No category profiles.")
        return
    print(f"training_signature: {repository.get_category_profile_training_signature()}")
    for row in rows:
        print(
            f"- id={row['id']} type={row['type']} status={row['status']} "
            f"weight={row['weight']} synthetic_count={row['synthetic_count']}"
        )


def handle_deactivate_category_profile(args: argparse.Namespace) -> None:
    repository = build_repository(args.db, load_runtime_config(args))
    repository.initialize_database()
    updated = repository.deactivate_category_profile(args.profile_id)
    print(f"category_profile_deactivated: {updated}")


def handle_debug_training_sources(args: argparse.Namespace) -> None:
    config = load_runtime_config(args)
    repository = build_repository(args.db, config)
    repository.initialize_database()
    rows = repository.fetch_type_training_examples()
    active_profiles = repository.list_category_profiles(include_inactive=False)
    stats = repository.get_stats()
    reviewed_feedback_count = repository.count_reviewed_feedback_logs()
    synthetic_rows = [row for row in rows if row.get("source") == "category_profile"]
    type_counts: dict[str, int] = {}
    source_weights: dict[str, list[float]] = {}
    for row in rows:
        label = str(row.get("label") or "unknown")
        source = str(row.get("source") or "real")
        type_counts[label] = type_counts.get(label, 0) + 1
        source_weights.setdefault(source, []).append(float(row.get("sample_weight", 1.0) or 1.0))

    labels = {str(row.get("label") or "") for row in rows if str(row.get("label") or "").strip()}
    learnable = True
    reasons: list[str] = []
    if len(rows) < config.ml.min_training_examples:
        learnable = False
        reasons.append(f"training rows below minimum ({len(rows)} < {config.ml.min_training_examples})")
    if len(labels) < 2:
        learnable = False
        reasons.append(f"need at least 2 labels, found {len(labels)}")
    try:
        import scipy  # noqa: F401
        import sklearn  # noqa: F401
    except Exception as error:
        learnable = False
        reasons.append(f"sklearn/scipy unavailable: {error}")

    print("Training source diagnostics")
    print(f"- confirmed_examples: {stats['confirmed_examples_count']}")
    print(f"- reviewed_feedback_logs: {reviewed_feedback_count}")
    print(f"- active_category_profiles: {len(active_profiles)}")
    print(f"- synthetic_rows: {len(synthetic_rows)}")
    print("- type_row_counts:")
    if type_counts:
        for label, count in sorted(type_counts.items()):
            print(f"  * {label}: {count}")
    else:
        print("  * none: 0")
    print("- source_sample_weight_avg:")
    if source_weights:
        for source, weights in sorted(source_weights.items()):
            average = sum(weights) / max(len(weights), 1)
            print(f"  * {source}: {average:.3f}")
    else:
        print("  * none: 0.000")
    print(f"- type_classifier_learnable: {'yes' if learnable else 'no'}")
    if not learnable:
        print("- learnability_reasons:")
        for reason in reasons:
            print(f"  * {reason}")
    print(f"- training_signature: {repository.get_category_profile_training_signature()}")
    print("- active_profiles:")
    if not active_profiles:
        print("  * none")
    for profile in active_profiles:
        print(
            f"  * id={profile['id']} type={profile['type']} "
            f"weight={profile['weight']} synthetic_count={profile['synthetic_count']}"
        )


def handle_preview_move(args: argparse.Namespace) -> None:
    config = load_runtime_config(args)
    repository = build_repository(args.db, config)
    repository.initialize_database()
    plan = preview_move_plan(repository=repository, config=config, limit=args.limit)
    print(f"batch_id: {plan['batch_id']}")
    print(f"manifest: {plan['manifest_path']}")
    print(f"items: {len(plan['items'])}")
    for item in plan["items"][:20]:
        print(
            f"- {item['source_path']} -> {item['destination_path']} "
            f"({item['large_category']}/{item['middle_category']}, confidence={item['confidence']:.3f})"
        )


def handle_commit_move(args: argparse.Namespace) -> None:
    repository = build_repository(args.db, load_runtime_config(args))
    repository.initialize_database()
    config = load_runtime_config(args)
    create_safety_snapshot(repository, config, reason=f"pre_commit_move_batch_{args.batch_id}")
    result = commit_move_batch(repository=repository, batch_id=args.batch_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def handle_undo_last_move(args: argparse.Namespace) -> None:
    repository = build_repository(args.db, load_runtime_config(args))
    repository.initialize_database()
    result = undo_last_move(repository)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def handle_restore_batch(args: argparse.Namespace) -> None:
    repository = build_repository(args.db, load_runtime_config(args))
    repository.initialize_database()
    result = restore_batch(repository=repository, batch_id=args.batch_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def handle_restore_file(args: argparse.Namespace) -> None:
    repository = build_repository(args.db, load_runtime_config(args))
    repository.initialize_database()
    result = restore_file(repository=repository, move_item_id=args.move_item_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def handle_list_move_history(args: argparse.Namespace) -> None:
    repository = build_repository(args.db, load_runtime_config(args))
    repository.initialize_database()
    rows = repository.list_move_history(
        category=args.category,
        date_from=args.date_from,
        date_to=args.date_to,
        batch_id=args.batch_id,
    )
    if not rows:
        print("No move history.")
        return
    for row in rows:
        print(
            f"- item_id={row['id']} batch={row['batch_id']} status={row['status']} "
            f"{row['source_path']} -> {row['actual_destination_path'] or row['destination_path']}"
        )


def handle_list_feedback_logs(args: argparse.Namespace) -> None:
    repository = build_repository(args.db, load_runtime_config(args))
    repository.initialize_database()
    rows = repository.list_feedback_logs(
        category=args.category,
        date_from=args.date_from,
        date_to=args.date_to,
        min_confidence=args.min_confidence,
        file_name_query=args.file_name,
    )
    if not rows:
        print("No feedback logs.")
        return
    for row in rows:
        print(
            f"- id={row['id']} file={row['file_name']} "
            f"{row['predicted_middle_category'] or row['predicted_category']} -> "
            f"{row['final_middle_category'] or row['final_category']} "
            f"confidence={row['final_score']:.3f} action={row['feedback_action']}"
        )


def handle_show_feedback_log(args: argparse.Namespace) -> None:
    repository = build_repository(args.db, load_runtime_config(args))
    repository.initialize_database()
    row = repository.get_feedback_log(args.feedback_log_id)
    if row is None:
        print("Feedback log not found.")
        return
    print(json.dumps(dict(row), ensure_ascii=False, indent=2))


def handle_delete_feedback_log(args: argparse.Namespace) -> None:
    repository = build_repository(args.db, load_runtime_config(args))
    repository.initialize_database()
    deleted = repository.delete_feedback_log(args.feedback_log_id)
    print(f"deleted: {deleted}")


def handle_clear_feedback_logs(args: argparse.Namespace) -> None:
    repository = build_repository(args.db, load_runtime_config(args))
    repository.initialize_database()
    deleted = repository.clear_feedback_logs()
    print(f"deleted_all: {deleted}")


def handle_export_feedback_logs(args: argparse.Namespace) -> None:
    repository = build_repository(args.db, load_runtime_config(args))
    repository.initialize_database()
    rows = repository.export_feedback_logs()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"exported: {output_path.resolve()}")


def handle_rebuild_feedback_learning(args: argparse.Namespace) -> None:
    repository = build_repository(args.db, config)
    repository.initialize_database()
    summary = rebuild_adaptive_learning(repository=repository, min_occurrences=args.min_occurrences)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def handle_embedding_cache_stats(args: argparse.Namespace) -> None:
    config = load_runtime_config(args)
    repository = build_repository(args.db, config)
    repository.initialize_database()
    legacy_stats = repository.get_embedding_cache_stats()
    embedding_repository = create_embedding_repository(config)
    print("Embedding cache stats")
    if embedding_repository is not None:
        hdf5_stats = embedding_repository.get_stats()
        print(f"- backend: hdf5")
        print(f"- path: {hdf5_stats['path']}")
        print(f"- entries: {hdf5_stats['entries']}")
        print(f"- vector_dim: {hdf5_stats['vector_dim']}")
        print(f"- legacy_sqlite_entries: {legacy_stats['entries']}")
        confirmed_stats = repository.get_confirmed_example_embedding_stats()
        print(
            "- confirmed_examples: "
            f"total={confirmed_stats['total']}, "
            f"with_key={confirmed_stats['with_embedding_key']}, "
            f"legacy_json={confirmed_stats['with_legacy_embedding_json']}, "
            f"hdf5_available={confirmed_stats['hdf5_available']}, "
            f"missing_hdf5={confirmed_stats['missing_hdf5']}"
        )
    else:
        print(f"- backend: sqlite")
        print(f"- entries: {legacy_stats['entries']}")
        print(f"- total_hits: {legacy_stats['total_hits']}")
        print("By model")
        if not legacy_stats["models"]:
            print("- none")
        else:
            for row in legacy_stats["models"]:
                print(f"- {row['model_name']}: entries={row['entries']}, hits={row['hits']}")
        print("By text kind")
        if not legacy_stats["text_kinds"]:
            print("- none")
        else:
            for row in legacy_stats["text_kinds"]:
                print(f"- {row['text_kind']}: entries={row['entries']}")


def handle_clear_embedding_cache(args: argparse.Namespace) -> None:
    config = load_runtime_config(args)
    repository = build_repository(args.db, config)
    repository.initialize_database()
    embedding_repository = create_embedding_repository(config)
    if embedding_repository is not None:
        deleted = embedding_repository.clear()
        legacy_deleted = repository.clear_embedding_cache() if config.embedding.use_legacy_sqlite_cache else 0
    else:
        deleted = repository.clear_embedding_cache()
        legacy_deleted = 0
    repository.record_operation(
        operation_type="clear_embedding_cache",
        status="completed",
        details={"deleted_entries": deleted, "legacy_deleted_entries": legacy_deleted},
    )
    print(f"deleted_embedding_cache_entries: {deleted}")
    if embedding_repository is not None:
        print(f"deleted_legacy_sqlite_entries: {legacy_deleted}")


def handle_rebuild_embedding_cache(args: argparse.Namespace) -> None:
    config = load_runtime_config(args)
    repository = build_repository(args.db, config)
    repository.initialize_database()
    embedding_repository = create_embedding_repository(config)
    if args.clear_first:
        if embedding_repository is not None:
            embedding_repository.clear()
        if config.embedding.use_legacy_sqlite_cache:
            repository.clear_embedding_cache()

    embedder = build_embedder(
        config,
        model_name=args.model_name,
        use_legacy_sqlite_cache=False,
        dual_write_legacy_sqlite=False,
    )
    sources = repository.fetch_embedding_rebuild_sources()
    built = 0
    skipped = 0
    start = time.perf_counter()
    for row in sources:
        text_value = str(row["text_value"])
        if not text_value.strip():
            skipped += 1
            continue
        embedding = embedder.encode(
            text_value,
            repository=repository,
            file_hash=str(row["file_hash"]),
            text_kind=str(row["text_kind"]),
        )
        if embedding:
            built += 1
        else:
            skipped += 1
    elapsed = time.perf_counter() - start
    repository.record_operation(
        operation_type="rebuild_embedding_cache",
        status="completed",
        details={"built": built, "skipped": skipped, "elapsed": round(elapsed, 3)},
    )
    print(f"rebuilt_embedding_cache_entries: {built}")
    print(f"skipped_sources: {skipped}")
    print(f"elapsed: {elapsed:.2f}s")


def handle_migrate_embedding_cache(args: argparse.Namespace) -> None:
    config = load_runtime_config(args)
    repository = build_repository(args.db, config)
    repository.initialize_database()
    embedding_repository = create_embedding_repository(config)
    if embedding_repository is None:
        print("HDF5 embedding repository is disabled in config.")
        return
    summary = migrate_sqlite_embedding_cache_to_hdf5(
        repository,
        embedding_repository,
        clear_target_first=args.clear_target_first,
    )
    repository.record_operation(
        operation_type="migrate_embedding_cache",
        status="completed",
        details=summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def handle_migrate_confirmed_example_embeddings(args: argparse.Namespace) -> None:
    config = load_runtime_config(args)
    repository = build_repository(args.db, config)
    repository.initialize_database()
    summary = repository.migrate_confirmed_examples_to_hdf5(prune_legacy_json=args.prune_legacy_json)
    repository.record_operation(
        operation_type="migrate_confirmed_example_embeddings",
        status="completed",
        details=summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def handle_create_snapshot(args: argparse.Namespace) -> None:
    repository = build_repository(args.db)
    repository.initialize_database()
    config = load_runtime_config(args)
    manifest = create_safety_snapshot(repository=repository, config=config, reason=args.reason)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def build_repository(db_path: str, config: AppConfig | None = None) -> ClassificationRepository:
    """Create a repository instance."""
    repository = ClassificationRepository(db_path)
    if config is not None:
        repository.attach_embedding_repository(create_embedding_repository(config))
    return repository


def build_embedder(
    config: AppConfig,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    *,
    use_legacy_sqlite_cache: bool | None = None,
    dual_write_legacy_sqlite: bool | None = None,
) -> SentenceTransformerEmbedder:
    embedding_repository = create_embedding_repository(config)
    return SentenceTransformerEmbedder(
        model_name=model_name,
        embedding_repository=embedding_repository,
        use_legacy_sqlite_cache=(
            config.embedding.use_legacy_sqlite_cache
            if use_legacy_sqlite_cache is None
            else use_legacy_sqlite_cache
        ),
        migrate_legacy_cache_on_hit=config.embedding.migrate_legacy_cache_on_hit,
        dual_write_legacy_sqlite=(
            config.embedding.dual_write_legacy_sqlite
            if dual_write_legacy_sqlite is None
            else dual_write_legacy_sqlite
        ),
    )


def load_categories(path: Path) -> dict[str, list[str]]:
    """Load seed categories."""
    if not path.exists():
        raise FileNotFoundError(f"Category file not found: {path}")
    taxonomy = load_taxonomy(path)
    return taxonomy.list_flat_keywords()


def load_runtime_config(args: argparse.Namespace) -> AppConfig:
    config = load_app_config(Path(args.config))
    if getattr(args, "db", None):
        config.database_path = str(args.db)
    if getattr(args, "categories", None):
        config.taxonomy_path = str(args.categories)
    return config


def load_runtime_taxonomy(args: argparse.Namespace) -> Taxonomy:
    return load_taxonomy(Path(args.categories))


def maybe_refine_with_llm(
    result: ClassificationResult,
    evidence_text: str,
    use_llm: bool,
    llm_model: str,
    llm_runtime: dict[str, bool],
    config: AppConfig | None = None,
) -> ClassificationResult:
    """Use the local LLM only when the current result is ambiguous."""
    config = config or default_config()
    if not use_llm:
        return replace(result, explanation={**result.explanation, "llm_status": "disabled"})
    if not llm_runtime.get("available", True):
        return replace(result, explanation={**result.explanation, "llm_status": "unavailable"})
    if not should_use_llm(result.confidence):
        return replace(
            result,
            explanation={**result.explanation, "llm_status": f"skipped_confidence({result.confidence:.3f})"},
        )

    if not use_llm or not llm_runtime.get("available", True) or not should_use_llm(result.confidence):
        return result

    try:
        if config.llm.provider == "ollama":
            llm_decision = classify_with_ollama(
                evidence_text=evidence_text,
                category_scores=aggregate_category_scores(result.candidate_scores),
                matched_keywords=result.matched_rules,
                model=llm_model,
                timeout_seconds=config.llm.timeout_seconds,
            )
        else:
            llm_decision = classify_with_provider(
                provider_name=config.llm.provider,
                evidence_text=evidence_text,
                category_scores=aggregate_category_scores(result.candidate_scores),
                matched_keywords=result.matched_rules,
                model=llm_model,
                timeout_seconds=config.llm.timeout_seconds,
                base_url=config.llm.base_url,
                api_key_env=config.llm.api_key_env,
            )
    except RuntimeError as error:
        llm_runtime["available"] = False
        print(f"LLM skipped: {error}")
        return replace(result, explanation={**result.explanation, "llm_status": f"error:{error}"})

    llm_confidence = llm_decision.confidence
    llm_reason = f"{result.reasoning} | llm={llm_decision.reason}"
    candidate_scores = dict(result.candidate_scores)
    mapped_category = llm_decision.recommended_category.split("/", 1)[-1]
    candidate_scores[mapped_category] = llm_confidence

    return replace(
        result,
        predicted_category=mapped_category,
        middle_category=mapped_category,
        large_category=llm_decision.recommended_category.split("/", 1)[0],
        confidence=llm_confidence,
        final_score=llm_confidence,
        llm_score=llm_confidence,
        review_required=llm_confidence < 0.8,
        candidate_scores=candidate_scores,
        reasoning=llm_reason,
        llm_used=True,
        explanation={
            **result.explanation,
            "llm_status": "applied",
            "llm_reason": llm_decision.reason,
            "llm_evidence": llm_decision.evidence or [],
        },
    )


def review_and_save_feedback(
    repository: ClassificationRepository,
    file_id: int,
    classification_id: int,
    predicted_category: str,
    normalized_text: str,
    result: ClassificationResult,
    classifier: HybridClassifier,
) -> None:
    """Prompt the user to confirm or correct the selected category."""
    ranked_categories = sorted(result.candidate_scores.items(), key=lambda item: (-item[1], item[0]))

    print("Input: Enter=confirm, number=select, text=new category")
    for index, (category, score) in enumerate(ranked_categories[:5], start=1):
        print(f"  {index}. {category} ({score:.3f})")

    user_input = input("Select: ").strip()
    if not user_input:
        final_category = predicted_category
        feedback_action = "confirmed"
    elif user_input.isdigit() and 1 <= int(user_input) <= min(5, len(ranked_categories)):
        final_category = ranked_categories[int(user_input) - 1][0]
        feedback_action = "confirmed" if final_category == predicted_category else "corrected"
    else:
        final_category = user_input
        feedback_action = "confirmed" if final_category == predicted_category else "corrected"

    note = input("Note(optional): ").strip() or None
    feedback_log_id = repository.save_feedback(
        file_id=file_id,
        classification_id=classification_id,
        predicted_category=predicted_category,
        final_category=final_category,
        feedback_action=feedback_action,
        user_note=note,
        predicted_hierarchy={
            "large_category": result.large_category,
            "middle_category": result.middle_category or predicted_category,
            "small_category": result.small_category,
        },
        final_hierarchy={
            "large_category": result.large_category,
            "middle_category": final_category,
            "small_category": result.small_category,
        },
        evidence_text=normalized_text,
        metadata={"matched_rules": result.matched_rules, "reasoning": result.reasoning},
        source_scores={
            "rule": result.rule_score,
            "embedding": result.embedding_score,
            "feedback": result.feedback_score,
            "duplicate": result.duplicate_score,
            "llm": result.llm_score,
        },
        ocr_used=result.ocr_used,
        llm_used=result.llm_used,
    )

    embedding = result.query_embedding
    if not embedding and classifier.embedder is not None:
        if isinstance(classifier.embedder, SentenceTransformerEmbedder):
            embedding = classifier.embedder.encode(
                normalized_text,
                repository=repository,
                file_hash=str(file_id),
                text_kind="evidence",
            )
        else:
            embedding = classifier.embedder.encode(normalized_text)

    if embedding:
        repository.save_confirmed_example(
            file_id=file_id,
            category=final_category,
            source_text=normalized_text,
            embedding=embedding,
            source_feedback_log_id=feedback_log_id,
        )
    print(f"saved: {predicted_category} -> {final_category} ({feedback_action})")
