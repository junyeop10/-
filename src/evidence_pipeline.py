"""Evidence-first document preparation for cluster labeling workflows."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time
from typing import Any, Callable

from src.document_features import DocumentFeatureExtractor
from src.evidence_cache import EVIDENCE_CACHE_VERSION, EvidenceCache
from src.file_reader import extract_text_from_file
from src.hash_utils import compute_raw_text_hash, compute_xxhash64
from src.ocr_cache import OcrCache
from src.ocr_support import DEFAULT_OCR_MIN_CHARS, OCR_MAX_IMAGE_EDGE, OCR_MAX_PAGES, OCR_RENDER_SCALE, ocr_pdf_file
from src.rule_classifier import build_rule_input_text, score_text_with_rules
from src.text_cleaner import build_sampled_text, normalize_text, tokenize_text


TEXT_OK = "text_ok"
OCR_USED = "ocr_used"
OCR_FAILED = "ocr_failed"
EVIDENCE_INSUFFICIENT = "evidence_insufficient"
API_READER_REQUIRED = "api_reader_required"


def load_cached_document_evidence(
    file_path: str | Path,
    *,
    rules: list[dict[str, Any]] | None = None,
    min_text_chars: int = DEFAULT_OCR_MIN_CHARS,
    min_tokens: int = 5,
    unreadable_threshold: float = 0.2,
    ocr_enabled: bool = True,
    ocr_max_pages: int = OCR_MAX_PAGES,
    evidence_cache_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    """Load cached evidence without running text extraction."""
    path = Path(file_path)
    if not evidence_cache_dir or not path.exists():
        return None
    total_start = time.perf_counter()
    hash_start = time.perf_counter()
    file_hash = compute_xxhash64(path)
    hash_elapsed = time.perf_counter() - hash_start
    lookup_start = time.perf_counter()
    cached = EvidenceCache(evidence_cache_dir).get(
        file_hash=file_hash,
        filename=path.name,
        settings_key=_evidence_cache_settings_key(
            rules=rules or [],
            min_text_chars=min_text_chars,
            min_tokens=min_tokens,
            unreadable_threshold=unreadable_threshold,
            ocr_enabled=ocr_enabled,
            ocr_max_pages=ocr_max_pages,
        ),
    )
    if cached is None:
        return None
    cached["file_path"] = str(path)
    cached["evidence_cache_hit"] = True
    cached["timings"] = {
        "file_hash": round(hash_elapsed, 4),
        "evidence_cache_lookup": round(time.perf_counter() - lookup_start, 4),
        "total": round(time.perf_counter() - total_start, 4),
    }
    return cached


def build_document_evidence(
    file_path: str | Path,
    *,
    rules: list[dict[str, Any]] | None = None,
    min_text_chars: int = DEFAULT_OCR_MIN_CHARS,
    min_tokens: int = 5,
    unreadable_threshold: float = 0.2,
    ocr_enabled: bool = True,
    ocr_max_pages: int = OCR_MAX_PAGES,
    ocr_cache_path: str | Path | None = None,
    ocr_cache_enabled: bool = True,
    evidence_cache_dir: str | Path | None = None,
    evidence_cache_enabled: bool = False,
    text_extractor: Callable[..., str] = extract_text_from_file,
    ocr_extractor: Callable[..., dict[str, Any]] = ocr_pdf_file,
) -> dict[str, Any]:
    """Build a document evidence object without making a final category decision."""
    path = Path(file_path)
    total_start = time.perf_counter()
    timings: dict[str, float] = {}
    raw_text = ""
    extraction_error = ""
    status_flags: list[str] = []
    ocr_cache_hit = False
    hash_start = time.perf_counter()
    file_hash = compute_xxhash64(path) if path.exists() else ""
    timings["file_hash"] = time.perf_counter() - hash_start
    evidence_cache_settings = _evidence_cache_settings_key(
        rules=rules or [],
        min_text_chars=min_text_chars,
        min_tokens=min_tokens,
        unreadable_threshold=unreadable_threshold,
        ocr_enabled=ocr_enabled,
        ocr_max_pages=ocr_max_pages,
    )
    if (
        evidence_cache_enabled
        and evidence_cache_dir
        and file_hash
        and text_extractor is extract_text_from_file
        and ocr_extractor is ocr_pdf_file
    ):
        cache_start = time.perf_counter()
        cached = EvidenceCache(evidence_cache_dir).get(
            file_hash=file_hash,
            filename=path.name,
            settings_key=evidence_cache_settings,
        )
        if cached is not None:
            cached["file_path"] = str(path)
            cached["evidence_cache_hit"] = True
            cached["timings"] = {
                "file_hash": round(timings["file_hash"], 4),
                "evidence_cache_lookup": round(time.perf_counter() - cache_start, 4),
                "total": round(time.perf_counter() - total_start, 4),
            }
            return cached

    extract_start = time.perf_counter()
    try:
        raw_text = text_extractor(path, fast=True)
    except Exception as error:
        extraction_error = str(error)
        raw_text = ""
    timings["extract_text"] = time.perf_counter() - extract_start

    normalized_text = normalize_text(raw_text)
    feature_start = time.perf_counter()
    initial_features = _extract_features(
        path,
        normalized_text,
        layout_enabled=_should_render_layout(
            path=path,
            text=normalized_text,
            min_text_chars=min_text_chars,
            min_tokens=min_tokens,
        ),
    )
    timings["feature_initial"] = time.perf_counter() - feature_start
    if _has_enough_evidence(
        normalized_text,
        initial_features.text_stats,
        min_text_chars=min_text_chars,
        min_tokens=min_tokens,
        unreadable_threshold=unreadable_threshold,
    ):
        status_flags.append(TEXT_OK)
        final_text = normalized_text
        final_features = initial_features
    else:
        final_text = normalized_text
        final_features = initial_features
        if ocr_enabled and path.suffix.lower() == ".pdf":
            ocr_result = _get_cached_ocr_result(
                file_hash=file_hash,
                cache_path=ocr_cache_path,
                max_pages=ocr_max_pages,
                cache_enabled=ocr_cache_enabled,
                timings=timings,
            )
            if ocr_result is None:
                ocr_start = time.perf_counter()
                ocr_result = ocr_extractor(path, max_pages=ocr_max_pages)
                timings["ocr"] = time.perf_counter() - ocr_start
                _save_cached_ocr_result(
                    file_hash=file_hash,
                    cache_path=ocr_cache_path,
                    max_pages=ocr_max_pages,
                    cache_enabled=ocr_cache_enabled,
                    ocr_result=ocr_result,
                )
            else:
                ocr_cache_hit = True
            if bool(ocr_result.get("ok")) and str(ocr_result.get("text") or "").strip():
                ocr_text = normalize_text(str(ocr_result.get("text") or ""))
                ocr_feature_start = time.perf_counter()
                ocr_features = _extract_features(path, ocr_text, layout_enabled=False)
                timings["feature_ocr"] = time.perf_counter() - ocr_feature_start
                if _has_enough_evidence(
                    ocr_text,
                    ocr_features.text_stats,
                    min_text_chars=min_text_chars,
                    min_tokens=min_tokens,
                    unreadable_threshold=unreadable_threshold,
                ):
                    status_flags.append(OCR_USED)
                    final_text = ocr_text
                    final_features = ocr_features
                else:
                    status_flags.extend([OCR_USED, EVIDENCE_INSUFFICIENT, API_READER_REQUIRED])
                    final_text = ocr_text
                    final_features = ocr_features
            else:
                status_flags.extend([OCR_FAILED, EVIDENCE_INSUFFICIENT, API_READER_REQUIRED])
        else:
            status_flags.extend([EVIDENCE_INSUFFICIENT, API_READER_REQUIRED])

    extraction_status = status_flags[0] if status_flags else EVIDENCE_INSUFFICIENT
    token_start = time.perf_counter()
    tokens = tokenize_text(f"{path.stem} {final_text}")
    top_tokens = Counter(tokens).most_common(30)
    timings["tokens"] = time.perf_counter() - token_start
    rule_start = time.perf_counter()
    old_rule_signals = _build_old_rule_signals(
        text=final_text,
        filename=path.name,
        rules=rules or [],
    )
    timings["rule_signals"] = time.perf_counter() - rule_start
    timings["total"] = time.perf_counter() - total_start

    result = {
        "file_path": str(path),
        "filename": path.name,
        "file_hash": file_hash,
        "extracted_text_length": len(final_text),
        "extraction_status": extraction_status,
        "status_flags": status_flags,
        "extraction_error": extraction_error,
        "filename_tokens": final_features.filename_features.get("tokens", []),
        "top_tokens": [{"token": token, "count": count} for token, count in top_tokens],
        "sampled_text": build_sampled_text(final_text, total_limit=1800, part_limit=600),
        "compressed_preview": final_features.compressed_text[:1600],
        "text_stats": final_features.text_stats,
        "structural_features": final_features.structural_features,
        "layout_features": final_features.layout_features,
        "old_rule_signals": old_rule_signals,
        "ocr_cache_hit": ocr_cache_hit,
        "evidence_cache_hit": False,
        "timings": {key: round(value, 4) for key, value in timings.items()},
        "api_reader": {
            "required": API_READER_REQUIRED in status_flags,
            "status": "placeholder",
            "reason": "insufficient_text_or_ocr" if API_READER_REQUIRED in status_flags else "",
        },
    }
    if evidence_cache_enabled and evidence_cache_dir and file_hash:
        EvidenceCache(evidence_cache_dir).save(result, settings_key=evidence_cache_settings)
    return result


def _evidence_cache_settings_key(
    *,
    rules: list[dict[str, Any]],
    min_text_chars: int,
    min_tokens: int,
    unreadable_threshold: float,
    ocr_enabled: bool,
    ocr_max_pages: int,
) -> str:
    rule_signature = compute_raw_text_hash(
        json.dumps(rules, ensure_ascii=False, sort_keys=True, default=str)
    )
    return "|".join(
        [
            EVIDENCE_CACHE_VERSION,
            f"min_chars={min_text_chars}",
            f"min_tokens={min_tokens}",
            f"unreadable={unreadable_threshold}",
            f"ocr={int(ocr_enabled)}",
            f"ocr_pages={ocr_max_pages}",
            f"rules={rule_signature}",
        ]
    )


def _extract_features(path: Path, text: str, *, layout_enabled: bool = True) -> Any:
    return DocumentFeatureExtractor().extract(
        file_name=path.name,
        file_ext=path.suffix,
        text=text,
        file_size=path.stat().st_size if path.exists() else 0,
        file_path=path if layout_enabled else None,
    )


def _ocr_cache_version(max_pages: int) -> str:
    return f"rapidocr_pages{max_pages}_scale{OCR_RENDER_SCALE}_edge{OCR_MAX_IMAGE_EDGE}_v1"


def _get_cached_ocr_result(
    *,
    file_hash: str,
    cache_path: str | Path | None,
    max_pages: int,
    cache_enabled: bool,
    timings: dict[str, float],
) -> dict[str, Any] | None:
    if not cache_enabled or not cache_path or not file_hash:
        return None
    lookup_start = time.perf_counter()
    try:
        cached = OcrCache(cache_path).get_cached_ocr(
            file_hash,
            "rapidocr",
            _ocr_cache_version(max_pages),
        )
    except Exception:
        cached = None
    timings["ocr_cache_lookup"] = time.perf_counter() - lookup_start
    if not cached:
        return None
    return {
        "ok": True,
        "file_path": "",
        "text": str(cached.get("cleaned_text") or cached.get("raw_text") or ""),
        "pages_scanned": int(cached.get("page_count", 0) or 0),
        "elapsed": 0.0,
        "error": "",
        "cache_hit": True,
    }


def _save_cached_ocr_result(
    *,
    file_hash: str,
    cache_path: str | Path | None,
    max_pages: int,
    cache_enabled: bool,
    ocr_result: dict[str, Any],
) -> None:
    if not cache_enabled or not cache_path or not file_hash or not ocr_result.get("ok"):
        return
    try:
        OcrCache(cache_path).save_ocr_cache(
            file_hash,
            "rapidocr",
            _ocr_cache_version(max_pages),
            {
                **ocr_result,
                "metadata": {
                    **dict(ocr_result.get("metadata") or {}),
                    "max_pages": max_pages,
                    "render_scale": OCR_RENDER_SCALE,
                    "max_image_edge": OCR_MAX_IMAGE_EDGE,
                },
            },
        )
    except Exception:
        return


def _should_render_layout(
    *,
    path: Path,
    text: str,
    min_text_chars: int,
    min_tokens: int,
) -> bool:
    """Render image layout only for partially readable PDFs that need extra evidence."""
    if path.suffix.lower() != ".pdf":
        return False
    normalized = normalize_text(text)
    if not normalized:
        return False
    token_count = len(tokenize_text(normalized))
    return len(normalized) < max(0, min_text_chars) or token_count < max(0, min_tokens)


def _has_enough_evidence(
    text: str,
    text_stats: dict[str, Any],
    *,
    min_text_chars: int,
    min_tokens: int,
    unreadable_threshold: float,
) -> bool:
    token_count = len(tokenize_text(text))
    unreadable_ratio = float(text_stats.get("unreadable_ratio", 0.0) or 0.0)
    return (
        len(normalize_text(text)) >= max(0, min_text_chars)
        and token_count >= max(0, min_tokens)
        and unreadable_ratio <= unreadable_threshold
    )


def _build_old_rule_signals(
    *,
    text: str,
    filename: str,
    rules: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rules:
        return {"status": "no_rules", "keyword_signals": []}

    rule_input = build_rule_input_text(text, filename)
    result = score_text_with_rules(rule_input, rules)
    keyword_signals = []
    seen_signals: set[str] = set()
    for _category, matches in result.get("matches", {}).items():
        for match in matches[:20]:
            signal = str(match).strip()
            if not signal or signal in seen_signals:
                continue
            seen_signals.add(signal)
            keyword_signals.append({"signal": signal})
    return {
        "status": "signals_only",
        "note": "Legacy rule matches are retained only as keyword evidence signals, not category labels.",
        "keyword_signals": keyword_signals[:80],
    }
