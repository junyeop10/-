"""Benchmark the GUI classification backend with isolated cold and warm caches."""

from __future__ import annotations

import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli import build_cluster_evidence_parallel, build_embedder, summarize_evidence_timings
from src.cluster_projection import build_cluster_projection
from src.config import load_app_config
from src.embedding_repository import create_embedding_repository
from src.file_reader import discover_supported_files
from src.storage import ClassificationRepository
from src.zip_categories_adapter import (
    ZIP_PIPELINE_VERSION,
    classification_result_from_zip,
    normalize_category_name,
    run_zip_categories_pipeline,
)


INPUT_ROOT = Path(r"C:\Users\jyok3\OneDrive\바탕 화면\데이터")
SOURCE_DB = ROOT / "data" / "classifier.db"
OUTPUT_ROOT = ROOT / "outputs" / "presentation_benchmark"
FOLDER_CATEGORY_MAP = {
    "5-1발표자료": "발표자료",
    "발표자료": "발표자료",
    "계약": "견적_계약_정산",
    "공고_지침_양식": "공고_지침_양식",
    "사업계획서": "사업계획서 수행계획서",
    "인증서": "기업 인증서",
    "조사_참고자료": "조사_참고자료",
    "중간_최종 결과물 및 보고서": "중간_최종 결과물 및 보고서",
}


def main() -> None:
    started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / started_at
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    config = deepcopy(load_app_config(ROOT / "data" / "app_config.json"))
    config.database_path = str(cache_dir / "benchmark.db")
    config.embedding.path = str(cache_dir / "embeddings.h5")
    config.embedding.use_legacy_sqlite_cache = False
    config.embedding.migrate_legacy_cache_on_hit = False
    config.features.evidence_cache_dir = str(cache_dir / "evidence")
    config.features.evidence_cache_enabled = True
    config.ocr.enabled = False

    if SOURCE_DB.exists():
        shutil.copy2(SOURCE_DB, config.database_path)
    repository = ClassificationRepository(config.database_path)
    repository.attach_embedding_repository(create_embedding_repository(config))
    repository.initialize_database()
    embedder = build_embedder(config, use_legacy_sqlite_cache=False, dual_write_legacy_sqlite=False)

    warmup_started = time.perf_counter()
    embedder.encode("초기화", text_kind="benchmark_warmup", embedding_version=ZIP_PIPELINE_VERSION)
    warmup_elapsed = time.perf_counter() - warmup_started

    files = discover_supported_files(INPUT_ROOT)
    runs = [
        run_once("first_run_cold_cache", files, repository, embedder, config),
        run_once("second_run_warm_cache", files, repository, embedder, config),
    ]
    report = {
        "input_root": str(INPUT_ROOT),
        "supported_file_count": len(files),
        "ignored_file_count": sum(1 for path in INPUT_ROOT.rglob("*") if path.is_file()) - len(files),
        "pipeline_version": ZIP_PIPELINE_VERSION,
        "embedding_model": embedder.model_name,
        "ocr_enabled": config.ocr.enabled,
        "supplemental_clustering_enabled": False,
        "model_warmup_seconds": round(warmup_elapsed, 3),
        "runs": runs,
        "cache_reduction": build_cache_reduction(runs),
    }
    (output_dir / "benchmark_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "benchmark_summary.md").write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "saved": str(output_dir),
                "supported_file_count": report["supported_file_count"],
                "model_warmup_seconds": report["model_warmup_seconds"],
                "runs": [
                    {
                        "name": run["name"],
                        "total_seconds": run["total_seconds"],
                        "accuracy_percent": run["accuracy"]["accuracy_percent"],
                        "mismatch_count": run["accuracy"]["mismatch_count"],
                    }
                    for run in runs
                ],
                "cache_reduction": report["cache_reduction"],
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    print(f"saved: {output_dir}")


def run_once(
    run_name: str,
    files: list[Path],
    repository: ClassificationRepository,
    embedder: Any,
    config: Any,
) -> dict[str, Any]:
    run_started = time.perf_counter()
    evidence_started = time.perf_counter()
    evidence_documents = build_cluster_evidence_parallel(
        files=files,
        rules=[],
        min_text_chars=config.ocr.min_text_chars,
        ocr_enabled=False,
        ocr_max_pages=config.ocr.max_pages,
        ocr_cache_path=config.database_path,
        ocr_cache_enabled=config.ocr.cache_enabled,
        evidence_cache_dir=config.features.evidence_cache_dir,
        evidence_cache_enabled=True,
        evidence_workers=4,
    )
    evidence_elapsed = time.perf_counter() - evidence_started
    documents = [
        {
            "index": index,
            "file_path": evidence["file_path"],
            "filename": evidence["filename"],
            "file_hash": evidence["file_hash"],
            "evidence": evidence,
        }
        for index, evidence in enumerate(evidence_documents)
    ]

    embedding_started = time.perf_counter()
    result = run_zip_categories_pipeline(
        documents,
        embedder=embedder,
        repository=repository,
        config=config,
    )
    classification_elapsed = time.perf_counter() - embedding_started
    classified = list(result["documents"])

    projection_started = time.perf_counter()
    cluster_result = result["cluster_result"]
    build_cluster_projection(
        classified,
        list(cluster_result.get("reduced_vectors", [])),
        [int(value) for value in cluster_result.get("cluster_ids", [])],
        probabilities=[float(value) for value in cluster_result.get("probabilities", [])],
    )
    projection_elapsed = time.perf_counter() - projection_started

    persistence_started = time.perf_counter()
    for item in classified:
        persist_document(repository, item)
    persistence_elapsed = time.perf_counter() - persistence_started

    accuracy = evaluate_accuracy(classified)
    evidence_summary = summarize_evidence_timings(evidence_documents)
    return {
        "name": run_name,
        "total_seconds": round(time.perf_counter() - run_started, 3),
        "evidence_seconds": round(evidence_elapsed, 3),
        "embedding_and_semantic_seconds": round(classification_elapsed, 3),
        "projection_seconds": round(projection_elapsed, 3),
        "db_persistence_seconds": round(persistence_elapsed, 3),
        "evidence_cache_hits": int(evidence_summary["evidence_cache_hits"]),
        "embedding_cache_entries": len(embedder.embedding_repository.list_ids()),
        "accuracy": accuracy,
    }


def persist_document(repository: ClassificationRepository, item: dict[str, Any]) -> None:
    result = classification_result_from_zip(item)
    path = Path(str(item["file_path"]))
    file_hash = str(item.get("file_hash", "")).strip()
    duplicate_file_id = repository.find_duplicate_file_id(file_hash, str(path))
    text = str((item.get("evidence") or {}).get("sampled_text", ""))
    file_id = repository.upsert_file(
        file_path=str(path),
        file_name=path.name,
        file_ext=path.suffix.lower(),
        file_size=path.stat().st_size,
        xxhash64=file_hash,
        duplicate_of_file_id=duplicate_file_id,
        extracted_text=text,
    )
    repository.insert_classification(
        file_id=file_id,
        predicted_category=result.predicted_category,
        rule_score=result.rule_score,
        embedding_score=result.embedding_score,
        llm_score=0.0,
        final_score=result.final_score,
        candidate_scores_json=json.dumps(result.candidate_scores, ensure_ascii=False),
        reasoning=result.reasoning,
        status="suggested",
        large_category=result.large_category,
        middle_category=result.middle_category,
        middle_confidence=result.middle_confidence,
        source_scores_json=json.dumps({"zip_semantic": result.candidate_scores}, ensure_ascii=False),
        evidence_json=json.dumps(
            {"zip_pipeline": ZIP_PIPELINE_VERSION, "clustering_status": item.get("clustering_status", "unknown")},
            ensure_ascii=False,
        ),
        classifier_version=ZIP_PIPELINE_VERSION,
        config_version=ZIP_PIPELINE_VERSION,
        review_reasons_json=json.dumps(result.review_reasons, ensure_ascii=False),
        rule_evidence_json=json.dumps(result.rule_evidence, ensure_ascii=False),
    )


def evaluate_accuracy(documents: list[dict[str, Any]]) -> dict[str, Any]:
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    mismatches: list[dict[str, Any]] = []
    correct = 0
    for document in documents:
        path = Path(str(document["file_path"]))
        source_folder = path.relative_to(INPUT_ROOT).parts[0]
        expected = FOLDER_CATEGORY_MAP[source_folder]
        predicted = normalize_category_name(document.get("category", ""))
        confusion[expected][predicted] += 1
        if predicted == expected:
            correct += 1
            continue
        evidence = document.get("evidence") or {}
        mismatches.append(
            {
                "filename": document.get("filename", ""),
                "source_folder": source_folder,
                "expected": expected,
                "predicted": predicted,
                "confidence": round(float(document.get("confidence", 0.0)), 4),
                "rule_confirmed": bool(document.get("rule_confirmed")),
                "matched_keywords": list(document.get("matched_keywords", [])),
                "candidate_scores": document.get("candidate_scores", {}),
                "semantic_signals": document.get("semantic_signals", {}),
                "filename_tokens": evidence.get("filename_tokens", []),
                "top_tokens": evidence.get("top_tokens", [])[:15],
                "sampled_text_preview": str(evidence.get("sampled_text", ""))[:900],
            }
        )
    total = len(documents)
    return {
        "correct": correct,
        "total": total,
        "accuracy_percent": round((correct / total * 100.0) if total else 0.0, 2),
        "rule_confirmed_count": sum(1 for document in documents if document.get("rule_confirmed")),
        "review_required_count": sum(1 for document in documents if document.get("review_required")),
        "confusion": {expected: dict(sorted(values.items())) for expected, values in sorted(confusion.items())},
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def build_cache_reduction(runs: list[dict[str, Any]]) -> dict[str, Any]:
    first, second = runs
    reduction = first["total_seconds"] - second["total_seconds"]
    return {
        "seconds_saved": round(reduction, 3),
        "percent_faster": round((reduction / first["total_seconds"] * 100.0) if first["total_seconds"] else 0.0, 2),
    }


def render_markdown(report: dict[str, Any]) -> str:
    first, second = report["runs"]
    lines = [
        "# GUI 분류 파이프라인 성능 측정",
        "",
        f"- 입력 폴더: `{report['input_root']}`",
        f"- 지원 파일: `{report['supported_file_count']}`개",
        f"- 제외 파일: `{report['ignored_file_count']}`개",
        f"- 모델 warm-up: `{report['model_warmup_seconds']}`초",
        "",
        "| 실행 | 전체 | evidence | 임베딩+의미분류 | PCA 표시 | DB 저장 | evidence cache hit | 정확도 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report["runs"]:
        lines.append(
            f"| {run['name']} | {run['total_seconds']}초 | {run['evidence_seconds']}초 | "
            f"{run['embedding_and_semantic_seconds']}초 | {run['projection_seconds']}초 | "
            f"{run['db_persistence_seconds']}초 | {run['evidence_cache_hits']} | "
            f"{run['accuracy']['accuracy_percent']}% |"
        )
    lines.extend(
        [
            "",
            f"- 재분류 절감 시간: `{report['cache_reduction']['seconds_saved']}`초",
            f"- 재분류 속도 개선: `{report['cache_reduction']['percent_faster']}`%",
            f"- 오분류 문서 수: `{second['accuracy']['mismatch_count']}`개",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
