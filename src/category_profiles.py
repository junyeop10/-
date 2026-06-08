"""Category profile seeding and synthetic training row generation."""

from __future__ import annotations

import json
from typing import Any

from src.document_patterns import get_default_document_patterns
from src.hash_utils import compute_raw_text_hash
from src.text_cleaner import tokenize_text


DEFAULT_CATEGORY_PROFILES: list[dict[str, Any]] = get_default_document_patterns()


def build_synthetic_training_rows(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand one active category profile into weighted synthetic training rows."""
    category_type = str(profile.get("type", "")).strip()
    profile_text = str(profile.get("profile_text", "")).strip()
    if not category_type or not profile_text:
        return []
    synthetic_count = max(1, int(profile.get("synthetic_count", 5) or 5))
    weight = float(profile.get("weight", 0.5) or 0.5)
    tags = _coerce_json_list(profile.get("tags", profile.get("tags_json", [])))
    profile_signals = _coerce_json_object(profile.get("profile_signals", profile.get("profile_signals_json", {})))
    terms = _synthetic_terms(profile_text, profile_signals=profile_signals)
    pseudo_features = _synthetic_structural_features(profile_signals)

    rows: list[dict[str, Any]] = []
    for index in range(synthetic_count):
        sampled_terms = terms[index::synthetic_count] or terms[:8]
        focus = _synthetic_focus(profile_signals=profile_signals, index=index)
        synthetic_text = "\n".join(
            [
                "# TYPE",
                category_type,
                "# PROFILE",
                profile_text,
                "# FOCUS",
                focus["name"],
                "# SEMANTIC_SIGNALS",
                "\n".join(focus["semantic_signals"]),
                "# LAYOUT_SIGNALS",
                "\n".join(focus["layout_signals"]),
                "# STRUCTURAL_SIGNALS",
                "\n".join(focus["structural_signals"]),
                "# OCR_SIGNALS",
                "\n".join(focus["ocr_signals"]),
                "# NUMERIC_PATTERNS",
                "\n".join(focus["numeric_patterns"]),
                "# BUSINESS_EXAMPLES",
                "\n".join(focus["document_examples"]),
                "# SYNTHETIC_TERMS",
                "\n".join(sampled_terms[:12]),
                "# TAGS",
                ", ".join(str(tag) for tag in tags),
            ]
        )
        rows.append(
            {
                "label": category_type,
                "file_name": f"synthetic_{category_type}_{index + 1}.txt",
                "body_text": synthetic_text,
                "structural_features": pseudo_features["structural_features"],
                "layout_features": pseudo_features["layout_features"],
                "source": "category_profile",
                "source_id": profile.get("id"),
                "sample_weight": weight,
            }
        )
    return rows


def _synthetic_focus(profile_signals: dict[str, Any], *, index: int) -> dict[str, Any]:
    """Create varied synthetic rows instead of repeating the same profile block."""
    sections = {
        "semantic_signals": _signal_values(profile_signals, "semantic_signals"),
        "layout_signals": _signal_values(profile_signals, "layout_signals"),
        "structural_signals": _signal_values(profile_signals, "structural_signals"),
        "ocr_signals": _signal_values(profile_signals, "ocr_signals"),
        "numeric_patterns": _signal_values(profile_signals, "numeric_patterns"),
        "document_examples": _signal_values(profile_signals, "document_examples"),
    }
    profiles = [
        ("semantic", {"semantic_signals": 8, "ocr_signals": 4, "structural_signals": 2}),
        ("layout", {"layout_signals": 8, "structural_signals": 4, "document_examples": 2}),
        ("ocr", {"ocr_signals": 8, "semantic_signals": 4, "numeric_patterns": 2}),
        ("numeric", {"numeric_patterns": 8, "semantic_signals": 4, "layout_signals": 2}),
        ("structure", {"structural_signals": 8, "layout_signals": 4, "semantic_signals": 2}),
        ("business", {"document_examples": 8, "semantic_signals": 4, "structural_signals": 2}),
        (
            "mixed",
            {
                "semantic_signals": 4,
                "layout_signals": 4,
                "structural_signals": 4,
                "ocr_signals": 4,
                "numeric_patterns": 4,
                "document_examples": 3,
            },
        ),
    ]
    name, limits = profiles[index % len(profiles)]
    focused: dict[str, Any] = {"name": name}
    offset = index // len(profiles)
    for key, values in sections.items():
        limit = int(limits.get(key, 1))
        focused[key] = _rotating_sample(values, limit=limit, offset=offset)
    return focused


def _rotating_sample(values: list[str], *, limit: int, offset: int) -> list[str]:
    if not values or limit <= 0:
        return []
    rotated = values[offset % len(values) :] + values[: offset % len(values)]
    return rotated[: min(limit, len(rotated))]


def build_category_profile_signature(profiles: list[dict[str, Any]]) -> str:
    active_profiles = [profile for profile in profiles if str(profile.get("status", "active")) == "active"]
    payload = {
        "active_profile_count": len(active_profiles),
        "profiles": [
            {
                "id": profile.get("id"),
                "type": profile.get("type"),
                "updated_at": profile.get("updated_at"),
                "profile_text_hash": compute_raw_text_hash(str(profile.get("profile_text", ""))),
                "profile_signals_hash": compute_raw_text_hash(
                    json.dumps(
                        _coerce_json_object(profile.get("profile_signals", profile.get("profile_signals_json", {}))),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                ),
                "lexical_profile_hash": compute_raw_text_hash(
                    json.dumps(
                        _coerce_json_object(profile.get("lexical_profile", profile.get("lexical_profile_json", {}))),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                ),
                "synthetic_count": int(profile.get("synthetic_count", 0) or 0),
                "weight": float(profile.get("weight", 0.0) or 0.0),
            }
            for profile in active_profiles
        ],
    }
    return compute_raw_text_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _synthetic_terms(profile_text: str, *, profile_signals: dict[str, Any]) -> list[str]:
    tokens = tokenize_text(profile_text)
    for key in (
        "aliases",
        "semantic_signals",
        "layout_signals",
        "structural_signals",
        "ocr_signals",
        "numeric_patterns",
        "document_examples",
        "business_use_cases",
        "core_features",
    ):
        for item in _signal_values(profile_signals, key):
            tokens.extend(tokenize_text(item))
    seen: set[str] = set()
    terms: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms[:120]


def _synthetic_structural_features(profile_signals: dict[str, Any]) -> dict[str, dict[str, Any]]:
    core_features = {str(item) for item in _signal_values(profile_signals, "core_features")}
    layout_features = {
        "receipt_pattern_score": 0.75 if "receipt_pattern_score" in core_features else 0.0,
        "dense_text_score": 0.75 if "dense_text_score" in core_features else 0.0,
        "signature_area_score": 0.65 if "signature_area_score" in core_features else 0.0,
        "slide_like_layout_score": 0.75 if "slide_like_layout_score" in core_features else 0.0,
        "large_header_score": 0.65 if "large_header_score" in core_features else 0.0,
        "image_area_ratio": 0.55 if "image_area_ratio" in core_features else 0.0,
        "two_column_score": 0.65 if "two_column_score" in core_features else 0.0,
        "numeric_column_score": 0.7 if "numeric_column_score" in core_features else 0.0,
        "approval_block_score": 0.65 if "approval_block_score" in core_features else 0.0,
        "chart_presence_score": 0.6 if "chart_presence_score" in core_features else 0.0,
    }
    structural_features = {
        "citation_count": 4 if "citation_density" in core_features else 0,
        "contract_terms_count": 5 if "legal_term_density" in core_features else 0,
        "receipt_terms_count": 5 if "receipt_pattern_score" in core_features else 0,
        "table_count": 3 if any(item in core_features for item in {"table_structure_score", "table_density"}) else 0,
        "image_count": 3 if "image_area_ratio" in core_features else 0,
        "bullet_ratio": 0.45 if "bullet_density" in core_features else 0.0,
        "clause_pattern_score": 0.8 if "clause_pattern_score" in core_features else 0.0,
        "legal_term_density": 0.8 if "legal_term_density" in core_features else 0.0,
        "research_structure_score": 0.8 if "research_structure_score" in core_features else 0.0,
        "heading_density": 0.5 if "heading_density" in core_features else 0.0,
    }
    return {"structural_features": structural_features, "layout_features": layout_features}


def _signal_values(profile_signals: dict[str, Any], key: str) -> list[str]:
    values = profile_signals.get(key, [])
    if not isinstance(values, list):
        return []
    return [str(item).strip() for item in values if str(item).strip()]


def _coerce_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _coerce_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}
