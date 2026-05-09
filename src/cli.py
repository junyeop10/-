"""CLI commands for the file classifier MVP."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.classifier import ClassificationResult, HybridClassifier
from src.fast_worker import process_file_fast
from src.file_reader import discover_supported_files, ensure_input_directory, extract_text_from_file
from src.hash_utils import compute_xxhash64
from src.rule_classifier import RuleBasedClassifier
from src.storage import ClassificationRepository
from src.text_cleaner import normalize_text
from src.vectorizer import SentenceTransformerEmbedder


FAST_TARGET_SECONDS = 10.0
MAX_FAST_WORKERS = min(4, os.cpu_count() or 1)
FAST_MIN_RULE_MATCHES_FOR_SKIP = 1
FAST_RULE_SKIP_EMBEDDING_THRESHOLD = 0.50
FAST_LOW_RULE_CONFIDENCE_THRESHOLD = 0.50


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
    classify_parser.set_defaults(func=handle_classify)

    suggest_parser = subparsers.add_parser("suggest-rules", parents=[common_parent], help="Suggest rules")
    suggest_parser.add_argument("--min-occurrences", type=int, default=2, help="Minimum repeated token count")
    suggest_parser.add_argument("--save-candidates", action="store_true", help="Save candidate rules")
    suggest_parser.set_defaults(func=handle_suggest_rules)

    stats_parser = subparsers.add_parser("stats", parents=[common_parent], help="Show DB stats")
    stats_parser.set_defaults(func=handle_stats)

    return parser


def handle_init_db(args: argparse.Namespace) -> None:
    """Initialize DB and seed rules."""
    repository = build_repository(args.db)
    repository.initialize_database()
    repository.seed_rules_from_categories(load_categories(Path(args.categories)))
    print(f"DB ready: {Path(args.db).resolve()}")


def handle_classify(args: argparse.Namespace) -> None:
    """Classify supported files from the input folder."""
    total_start = time.perf_counter()
    repository = build_repository(args.db)
    repository.initialize_database()
    repository.seed_rules_from_categories(load_categories(Path(args.categories)))

    input_dir = Path(args.input_dir)
    ensure_input_directory(input_dir)
    files = discover_supported_files(input_dir)
    if not files:
        print(f"No txt/pdf files found: {input_dir.resolve()}")
        return

    classifier = HybridClassifier(
        repository=repository,
        embedder=SentenceTransformerEmbedder(model_name=args.model_name),
        rule_classifier=RuleBasedClassifier(repository),
        rule_skip_embedding_threshold=FAST_RULE_SKIP_EMBEDDING_THRESHOLD if args.fast else 0.85,
        min_rule_matches_for_skip=FAST_MIN_RULE_MATCHES_FOR_SKIP if args.fast else 3,
        use_embedding_for_no_rule=not args.fast,
        low_rule_confidence_threshold=FAST_LOW_RULE_CONFIDENCE_THRESHOLD if args.fast else 0.20,
    )

    if args.fast:
        summary = process_files_fast(
            repository=repository,
            classifier=classifier,
            files=files,
            review=args.review,
            max_workers=max(1, min(args.workers, MAX_FAST_WORKERS)),
            total_start=total_start,
        )
    else:
        summary = process_files_sequential(
            repository=repository,
            classifier=classifier,
            files=files,
            review=args.review,
            total_start=total_start,
        )

    print_performance_summary(summary)


def process_files_fast(
    repository: ClassificationRepository,
    classifier: HybridClassifier,
    files: list[Path],
    review: bool,
    max_workers: int,
    total_start: float,
) -> dict[str, Any]:
    """Run extraction and rule scoring in worker processes."""
    rules = serialize_active_rules(repository)
    print(f"Classify start: {len(files)} files, fast=True, workers={max_workers}")

    summary = create_summary(total_files=len(files))
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

    precompute_fast_embeddings(repository, classifier, worker_results)

    for worker_result in worker_results:
        result, classification_id = finalize_worker_result(
            repository=repository,
            classifier=classifier,
            worker_result=worker_result,
        )
        summary["success"] += 1
        summary["embedding_used"] += int(result.embedding_used)
        summary["embedding_skipped"] += int(not result.embedding_used)
        print_classification_result(worker_result["file_name"], result, worker_result["timings"])

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
        texts = [str(worker_result["evidence_text"]) for worker_result in embedding_targets]
        embeddings = classifier.embedder.encode_many(texts)
    except Exception as error:
        print(f"Embedding batch skipped: {error}")
        return

    for worker_result, embedding in zip(embedding_targets, embeddings):
        worker_result["precomputed_embedding"] = embedding


def process_files_sequential(
    repository: ClassificationRepository,
    classifier: HybridClassifier,
    files: list[Path],
    review: bool,
    total_start: float,
) -> dict[str, Any]:
    """Run the original sequential classification flow."""
    print(f"Classify start: {len(files)} files, fast=False")
    summary = create_summary(total_files=len(files))
    for file_path in files:
        result = process_single_file(
            repository=repository,
            classifier=classifier,
            file_path=file_path,
            review=review,
        )
        if result is None:
            summary["failed"] += 1
            continue
        summary["success"] += 1
        summary["embedding_used"] += int(result.embedding_used)
        summary["embedding_skipped"] += int(not result.embedding_used)

    summary["elapsed"] = time.perf_counter() - total_start
    return summary


def finalize_worker_result(
    repository: ClassificationRepository,
    classifier: HybridClassifier,
    worker_result: dict[str, Any],
) -> tuple[ClassificationResult, int]:
    """Save worker output in DB and finish scoring in the main process."""
    duplicate_of_file_id = repository.find_duplicate_file_id(
        str(worker_result["xxhash64"]),
        str(Path(worker_result["file_path"]).resolve()),
    )
    file_id = repository.upsert_file(
        file_path=str(Path(worker_result["file_path"]).resolve()),
        file_name=str(worker_result["file_name"]),
        file_ext=str(worker_result["file_ext"]),
        file_size=int(worker_result["file_size"]),
        xxhash64=str(worker_result["xxhash64"]),
        duplicate_of_file_id=duplicate_of_file_id,
        extracted_text=str(worker_result["evidence_text"]),
    )
    worker_result["file_id"] = file_id
    result = classifier.classify_with_rule_breakdown(
        file_hash=str(worker_result["xxhash64"]),
        text=str(worker_result["evidence_text"]),
        duplicate_of_file_id=duplicate_of_file_id,
        rule_breakdown=worker_result["rule_breakdown"],
        precomputed_query_embedding=worker_result.get("precomputed_embedding"),
    )
    classification_id = classifier.persist_classification(file_id=file_id, result=result)
    return result, classification_id


def process_single_file(
    repository: ClassificationRepository,
    classifier: HybridClassifier,
    file_path: Path,
    review: bool,
) -> ClassificationResult | None:
    """Read, classify, print, and optionally review one file."""
    file_start = time.perf_counter()
    try:
        extract_start = time.perf_counter()
        evidence_text = extract_text_from_file(file_path, fast=False)
        read_extract_time = time.perf_counter() - extract_start
    except Exception as error:
        print(f"[failed] {file_path.name}: read failed - {error}")
        return None

    preprocess_start = time.perf_counter()
    normalized_text = normalize_text(evidence_text)
    preprocess_time = time.perf_counter() - preprocess_start
    if not normalized_text:
        print(f"[failed] {file_path.name}: no text")
        return None

    try:
        file_hash = compute_xxhash64(file_path)
    except Exception as error:
        print(f"[failed] {file_path.name}: hash failed - {error}")
        return None

    duplicate_of_file_id = repository.find_duplicate_file_id(file_hash, str(file_path.resolve()))
    file_id = repository.upsert_file(
        file_path=str(file_path.resolve()),
        file_name=file_path.name,
        file_ext=file_path.suffix.lower(),
        file_size=file_path.stat().st_size,
        xxhash64=file_hash,
        duplicate_of_file_id=duplicate_of_file_id,
        extracted_text=normalized_text,
    )

    try:
        result = classifier.classify_file(
            file_id=file_id,
            file_hash=file_hash,
            text=normalized_text,
            duplicate_of_file_id=duplicate_of_file_id,
        )
    except RuntimeError as error:
        print(f"[failed] {file_path.name}: {error}")
        return None

    classification_id = classifier.persist_classification(file_id=file_id, result=result)
    timings = {
        "read_extract_time": read_extract_time,
        "preprocess_time": preprocess_time,
        "rule_time": 0.0,
        "worker_time": time.perf_counter() - file_start,
    }
    print_classification_result(file_path.name, result, timings)

    if review:
        review_and_save_feedback(
            repository=repository,
            file_id=file_id,
            classification_id=classification_id,
            predicted_category=result.predicted_category,
            normalized_text=normalized_text,
            result=result,
            classifier=classifier,
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
) -> None:
    """Print a concise classification result."""
    matched_rules = ", ".join(result.matched_rules) if result.matched_rules else "none"
    review_text = "yes" if result.review_required else "no"
    similarity_text = f"{result.similarity_score:.3f}" if result.embedding_used else "skipped"

    print("")
    print("=" * 48)
    print(f"file: {file_name}")
    print(f"category: {result.predicted_category}")
    print(f"confidence: {result.confidence:.3f}")
    print(f"matched_rules: {matched_rules}")
    print(f"similarity: {similarity_text}")
    print(f"review_required: {review_text}")
    print(
        "scores: "
        f"rule={result.rule_score:.3f}, "
        f"embedding={result.embedding_score:.3f}, "
        f"feedback={result.feedback_score:.3f}, "
        f"final={result.final_score:.3f}"
    )
    print(
        "timing: "
        f"read={float(timings.get('read_extract_time', 0.0)):.2f}s, "
        f"preprocess={float(timings.get('preprocess_time', 0.0)):.3f}s, "
        f"rule={float(timings.get('rule_time', 0.0)):.3f}s, "
        f"total={float(timings.get('worker_time', 0.0)):.2f}s"
    )
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


def handle_suggest_rules(args: argparse.Namespace) -> None:
    """Suggest new rules from correction logs."""
    repository = build_repository(args.db)
    repository.initialize_database()

    classifier = HybridClassifier(
        repository=repository,
        embedder=SentenceTransformerEmbedder(),
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
    repository = build_repository(args.db)
    repository.initialize_database()
    stats = repository.get_stats()

    print("DB stats")
    print(f"- files: {stats['files_count']}")
    print(f"- classifications: {stats['classifications_count']}")
    print(f"- feedback_logs: {stats['feedback_logs_count']}")
    print(f"- confirmed_examples: {stats['confirmed_examples_count']}")
    print(f"- rules: {stats['rules_count']}")
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


def build_repository(db_path: str) -> ClassificationRepository:
    """Create a repository instance."""
    return ClassificationRepository(db_path)


def load_categories(path: Path) -> dict[str, list[str]]:
    """Load seed categories."""
    if not path.exists():
        raise FileNotFoundError(f"Category file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


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
    )

    embedding = result.query_embedding
    if not embedding:
        embedding = classifier.embedder.encode(normalized_text)

    repository.save_confirmed_example(
        file_id=file_id,
        category=final_category,
        source_text=normalized_text,
        embedding=embedding,
        source_feedback_log_id=feedback_log_id,
    )
    print(f"saved: {predicted_category} -> {final_category} ({feedback_action})")
