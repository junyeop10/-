"""Performance profiling helpers for startup and per-file classification analysis."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping


def normalize_stage_timings(stage_timings: Mapping[str, Any] | None) -> dict[str, float]:
    """Return a sanitized stage timing dictionary."""
    normalized: dict[str, float] = {}
    if not stage_timings:
        return normalized
    for key, value in stage_timings.items():
        try:
            normalized[str(key)] = max(float(value), 0.0)
        except (TypeError, ValueError):
            continue
    return normalized


def build_file_latency_analysis(
    stage_timings: Mapping[str, Any] | None,
    *,
    text_length: int = 0,
    file_size: int = 0,
    ocr_used: bool = False,
    ocr_status: str = "not_checked",
    ocr_pages: int = 0,
    embedding_used: bool = False,
    embedding_cache_hit: bool | None = None,
    strong_rule_match: bool = False,
    review_required: bool = False,
    matched_rules_count: int = 0,
    llm_used: bool = False,
    duplicate_detected: bool = False,
) -> dict[str, Any]:
    """Explain why a file classification was slow or fast."""
    timings = normalize_stage_timings(stage_timings)
    total_time = timings.get("total", 0.0)
    if total_time <= 0:
        total_time = sum(value for key, value in timings.items() if key != "total")

    ranked = sorted(
        ((key, value) for key, value in timings.items() if key != "total" and value > 0),
        key=lambda item: item[1],
        reverse=True,
    )
    dominant_stage = ranked[0][0] if ranked else "unknown"
    dominant_time = ranked[0][1] if ranked else 0.0

    reasons: list[str] = []
    if ocr_used:
        reasons.append(f"OCR ran and scanned {ocr_pages} page(s).")
    elif ocr_status in {"queued", "used"}:
        reasons.append("OCR fallback path was involved.")

    if embedding_used:
        if embedding_cache_hit is True:
            reasons.append("Embedding reused the persistent cache.")
        elif embedding_cache_hit is False:
            reasons.append("Embedding inference ran for this file.")
        else:
            reasons.append("Embedding stage contributed to classification.")
    elif strong_rule_match:
        reasons.append("Strong rule match skipped embedding work.")
    else:
        reasons.append("Embedding stage was bypassed or unavailable.")

    if matched_rules_count >= 3:
        reasons.append("Multiple rule matches helped the classifier decide quickly.")
    elif matched_rules_count == 0:
        reasons.append("Few direct rule matches increased ambiguity.")

    if text_length >= 8000:
        reasons.append("Large extracted text increased parsing and scoring cost.")
    elif text_length <= 250 and ocr_used:
        reasons.append("Short extracted text required OCR assistance.")

    if file_size >= 8_000_000:
        reasons.append("Large file size increased extraction overhead.")
    elif file_size and file_size <= 80_000 and not ocr_used and not embedding_used:
        reasons.append("Small file with direct text extraction stayed lightweight.")

    if review_required:
        reasons.append("Low-confidence or conflicting signals required extra scoring checks.")
    if llm_used:
        reasons.append("LLM refinement added extra latency.")
    if duplicate_detected:
        reasons.append("Duplicate history lookup contributed a small DB lookup.")

    stage_reason_map = {
        "read_extract": "Text extraction dominated runtime.",
        "ocr": "OCR was the main bottleneck.",
        "classification": "Classifier scoring dominated runtime.",
        "embedding": "Embedding generation was the main bottleneck.",
        "db_upsert": "Database writes were the slowest step.",
        "db_persist": "Classification persistence was the slowest step.",
    }
    if dominant_stage in stage_reason_map:
        reasons.insert(0, stage_reason_map[dominant_stage])

    speed_band = "fast"
    if total_time >= 2.0:
        speed_band = "slow"
    elif total_time >= 0.8:
        speed_band = "moderate"

    if dominant_stage == "unknown":
        summary = "No timing stages were recorded."
    elif speed_band == "slow":
        summary = f"Slow file because {dominant_stage} took the largest share."
    elif speed_band == "moderate":
        summary = f"Moderate runtime with {dominant_stage} as the largest stage."
    else:
        summary = f"Fast file because expensive stages were mostly skipped or cached."

    return {
        "total_time": round(total_time, 6),
        "dominant_stage": dominant_stage,
        "dominant_time": round(dominant_time, 6),
        "speed_band": speed_band,
        "reasons": reasons,
        "summary": summary,
    }


def summarize_payload_profiles(
    payloads: list[dict[str, Any]],
    *,
    startup_profile: Mapping[str, Any] | None = None,
    run_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a run-level summary from GUI/CLI payloads."""
    file_rows: list[dict[str, Any]] = []
    stage_totals: dict[str, float] = defaultdict(float)

    for payload in payloads:
        performance = payload.get("performance")
        if not isinstance(performance, Mapping):
            continue
        stage_timings = normalize_stage_timings(performance.get("stage_timings"))
        analysis = performance.get("analysis")
        if not isinstance(analysis, Mapping):
            analysis = build_file_latency_analysis(stage_timings)
        for key, value in stage_timings.items():
            if key != "total":
                stage_totals[key] += value
        file_rows.append(
            {
                "file_name": str(payload.get("file_name", "")),
                "category": str(payload.get("result").predicted_category) if payload.get("result") else "",
                "total_time": float(analysis.get("total_time", stage_timings.get("total", 0.0))),
                "dominant_stage": str(analysis.get("dominant_stage", "unknown")),
                "summary": str(analysis.get("summary", "")),
                "reasons": list(analysis.get("reasons", [])),
            }
        )

    slowest_files = sorted(file_rows, key=lambda row: row["total_time"], reverse=True)
    total_file_time = sum(row["total_time"] for row in file_rows)
    file_count = len(file_rows)
    average_file_time = total_file_time / file_count if file_count else 0.0

    startup_total = 0.0
    startup_stages: dict[str, float] = {}
    if startup_profile:
        startup_stages = normalize_stage_timings(startup_profile.get("stages"))
        startup_total = float(startup_profile.get("startup_ready_total", 0.0))
        if startup_total <= 0:
            startup_total = sum(startup_stages.values())

    run_elapsed = 0.0
    if run_profile:
        run_elapsed = float(run_profile.get("elapsed", 0.0))
    if run_elapsed <= 0:
        run_elapsed = total_file_time

    return {
        "startup_total": round(startup_total, 6),
        "startup_stages": startup_stages,
        "run_elapsed": round(run_elapsed, 6),
        "classified_files": file_count,
        "average_file_time": round(average_file_time, 6),
        "stage_totals": dict(sorted(stage_totals.items(), key=lambda item: item[1], reverse=True)),
        "slowest_files": slowest_files[:10],
    }
