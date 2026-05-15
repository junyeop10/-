"""Multiprocessing worker for fast classification preprocessing."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.file_reader import extract_text_from_file
from src.hash_utils import compute_xxhash64
from src.rule_classifier import build_rule_input_text, score_text_with_rules
from src.text_cleaner import normalize_text


def process_file_fast(file_path_text: str, rules: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract evidence text, hash, normalize, and rule-score one file."""
    worker_start = time.perf_counter()
    file_path = Path(file_path_text)
    timings: dict[str, float] = {}

    try:
        read_start = time.perf_counter()
        file_hash = compute_xxhash64(file_path)
        evidence_text = extract_text_from_file(file_path, fast=True)
        timings["read_extract_time"] = time.perf_counter() - read_start

        clean_start = time.perf_counter()
        normalized_text = normalize_text(evidence_text)
        timings["preprocess_time"] = time.perf_counter() - clean_start

        rule_start = time.perf_counter()
        rule_input_text = build_rule_input_text(normalized_text, file_path.name)
        rule_breakdown = score_text_with_rules(rule_input_text, rules)
        timings["rule_time"] = time.perf_counter() - rule_start

        timings["worker_time"] = time.perf_counter() - worker_start
        return {
            "ok": True,
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_ext": file_path.suffix.lower(),
            "file_size": file_path.stat().st_size,
            "xxhash64": file_hash,
            "evidence_text": normalized_text,
            "rule_breakdown": rule_breakdown,
            "ocr_used": False,
            "ocr_pages": 0,
            "ocr_error": "",
            "ocr_status": "not_checked",
            "ocr_reason": "",
            "classification_hint": None,
            "timings": timings,
            "error": "",
        }
    except Exception as error:
        timings["worker_time"] = time.perf_counter() - worker_start
        return {
            "ok": False,
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_ext": file_path.suffix.lower(),
            "file_size": file_path.stat().st_size if file_path.exists() else 0,
            "xxhash64": "",
            "evidence_text": "",
            "rule_breakdown": {"scores": {}, "matches": {}},
            "ocr_used": False,
            "ocr_pages": 0,
            "ocr_error": "",
            "ocr_status": "not_checked",
            "ocr_reason": "",
            "classification_hint": None,
            "timings": timings,
            "error": str(error),
        }
