"""Inspect the pipeline state before embedding generation.

Usage:
    .\.venv\Scripts\python.exe tools\inspect_pre_embedding.py "path\to\file.pdf"
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli import build_repository, load_categories
from src.config import DEFAULT_CONFIG_PATH, load_app_config
from src.document_features import DocumentFeatureExtractor
from src.file_reader import extract_text_from_file
from src.rule_classifier import build_rule_input_text, score_text_with_rules
from src.taxonomy import load_taxonomy
from src.text_cleaner import normalize_text, tokenize_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect extraction and keyword state before embeddings.")
    parser.add_argument("paths", nargs="+", help="File path(s) to inspect")
    parser.add_argument("--db", default="data/classifier.db", help="SQLite DB path")
    parser.add_argument("--categories", default="data/categories.json", help="Category seed JSON")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Application config JSON")
    parser.add_argument("--full", action="store_true", help="Print full normalized and compressed text")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only")
    parser.add_argument("--top", type=int, default=30, help="Number of top tokens to show")
    args = parser.parse_args()

    config = load_app_config(Path(args.config))
    config.database_path = str(args.db)
    config.taxonomy_path = str(args.categories)
    repository = build_repository(args.db, config)
    repository.initialize_database()
    repository.seed_rules_from_categories(load_categories(Path(args.categories)))
    taxonomy = load_taxonomy(Path(args.categories))
    rules = [
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

    reports = []
    for raw_path in args.paths:
        path = Path(raw_path).expanduser()
        reports.append(
            inspect_file(
                path=path,
                rules=rules,
                known_categories=taxonomy.list_flat_keywords(),
                top=args.top,
                include_full_text=args.full,
            )
        )

    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return

    for report in reports:
        print_report(report)


def inspect_file(
    *,
    path: Path,
    rules: list[dict[str, Any]],
    known_categories: dict[str, list[str]],
    top: int,
    include_full_text: bool,
) -> dict[str, Any]:
    raw_text = extract_text_from_file(path, fast=True)
    normalized_text = normalize_text(raw_text)
    features = DocumentFeatureExtractor().extract(
        file_name=path.name,
        file_ext=path.suffix,
        text=normalized_text,
        file_size=path.stat().st_size if path.exists() else 0,
        file_path=path,
    )
    rule_input = build_rule_input_text(normalized_text, path.name)
    rule_result = score_text_with_rules(rule_input, rules)
    token_counts = Counter(tokenize_text(features.compressed_text or normalized_text))
    ranked_scores = sorted(
        ((category, float(score)) for category, score in rule_result["scores"].items()),
        key=lambda item: (-item[1], item[0]),
    )
    matched = {
        category: values
        for category, values in rule_result["matches"].items()
        if values
    }
    category_keyword_hits = build_category_keyword_hits(
        text=rule_input,
        known_categories=known_categories,
    )

    report: dict[str, Any] = {
        "file": str(path),
        "exists": path.exists(),
        "raw_text_length": len(raw_text),
        "normalized_text_length": len(normalized_text),
        "compressed_text_length": len(features.compressed_text),
        "compressed_text_hash": features.compressed_text_hash,
        "filename_tokens": features.filename_features.get("tokens", []),
        "top_tokens": token_counts.most_common(top),
        "category_keyword_hits": category_keyword_hits,
        "rule_scores": ranked_scores[:10],
        "matched_rules": matched,
        "negative_matches": {
            category: values
            for category, values in rule_result["negative_matches"].items()
            if values
        },
        "text_stats": features.text_stats,
        "structural_features": features.structural_features,
        "layout_features": features.layout_features,
        "compressed_preview": features.compressed_text[:1200],
        "normalized_preview": normalized_text[:1200],
    }
    if include_full_text:
        report["compressed_text"] = features.compressed_text
        report["normalized_text"] = normalized_text
    return report


def build_category_keyword_hits(
    *,
    text: str,
    known_categories: dict[str, list[str]],
) -> dict[str, list[str]]:
    normalized = normalize_text(text)
    compact = normalized.replace(" ", "")
    hits: dict[str, list[str]] = {}
    for category, keywords in known_categories.items():
        found = []
        for keyword in keywords:
            normalized_keyword = normalize_text(str(keyword))
            if not normalized_keyword:
                continue
            compact_keyword = normalized_keyword.replace(" ", "")
            if normalized_keyword in normalized or compact_keyword in compact:
                found.append(str(keyword))
        if found:
            hits[category] = found[:20]
    return hits


def print_report(report: dict[str, Any]) -> None:
    print("=" * 80)
    print(f"file: {report['file']}")
    print(f"exists: {report['exists']}")
    print(f"raw_text_length: {report['raw_text_length']}")
    print(f"normalized_text_length: {report['normalized_text_length']}")
    print(f"compressed_text_length: {report['compressed_text_length']}")
    print(f"compressed_text_hash: {report['compressed_text_hash']}")
    print("")
    print("[filename_tokens]")
    print(", ".join(str(item) for item in report["filename_tokens"]) or "none")
    print("")
    print("[top_tokens]")
    for token, count in report["top_tokens"]:
        print(f"- {token}: {count}")
    print("")
    print("[category_keyword_hits]")
    if not report["category_keyword_hits"]:
        print("- none")
    for category, hits in report["category_keyword_hits"].items():
        print(f"- {category}: {', '.join(hits)}")
    print("")
    print("[rule_scores_top10]")
    for category, score in report["rule_scores"]:
        print(f"- {category}: {score:.3f}")
    print("")
    print("[matched_rules]")
    if not report["matched_rules"]:
        print("- none")
    for category, matches in report["matched_rules"].items():
        print(f"- {category}: {', '.join(matches[:20])}")
    print("")
    print("[text_stats]")
    print(json.dumps(report["text_stats"], ensure_ascii=False, indent=2))
    print("")
    print("[structural_features]")
    print(json.dumps(report["structural_features"], ensure_ascii=False, indent=2))
    print("")
    print("[compressed_preview]")
    print(report["compressed_preview"])
    print("")


if __name__ == "__main__":
    main()
