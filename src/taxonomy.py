"""Hierarchical taxonomy helpers with backward-compatible loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


UNCATEGORIZED = "검토필요"


@dataclass(frozen=True)
class TaxonomyEntry:
    large_category: str
    middle_category: str
    small_category: str | None
    flat_label: str
    keywords: list[str]
    aliases: list[str]
    folder_target: str


class Taxonomy:
    def __init__(self, entries: list[TaxonomyEntry], version: str = "2.0") -> None:
        self.entries = entries
        self.version = version
        self._by_flat = {entry.flat_label: entry for entry in entries}
        self._by_alias: dict[str, TaxonomyEntry] = {}
        for entry in entries:
            self._by_alias[entry.flat_label] = entry
            self._by_alias[entry.middle_category] = entry
            for alias in entry.aliases:
                self._by_alias[alias] = entry

    def list_middle_categories(self) -> list[str]:
        return sorted({entry.middle_category for entry in self.entries})

    def list_flat_keywords(self) -> dict[str, list[str]]:
        return {entry.flat_label: list(entry.keywords) for entry in self.entries}

    def resolve(self, category_name: str | None) -> TaxonomyEntry:
        if category_name and category_name in self._by_alias:
            return self._by_alias[category_name]
        return TaxonomyEntry(
            large_category="miscellaneous",
            middle_category=UNCATEGORIZED,
            small_category=None,
            flat_label=UNCATEGORIZED,
            keywords=[],
            aliases=[],
            folder_target="miscellaneous/review",
        )


def load_taxonomy(path: str | Path) -> Taxonomy:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "categories" in raw:
        entries = [
            TaxonomyEntry(
                large_category=str(item["large_category"]),
                middle_category=str(item["middle_category"]),
                small_category=str(item["small_category"]) if item.get("small_category") else None,
                flat_label=str(item.get("flat_label") or item["middle_category"]),
                keywords=[str(keyword) for keyword in item.get("keywords", [])],
                aliases=[str(alias) for alias in item.get("aliases", [])],
                folder_target=str(item.get("folder_target") or item["middle_category"]),
            )
            for item in raw["categories"]
        ]
        return Taxonomy(entries=entries, version=str(raw.get("version", "2.0")))

    if isinstance(raw, dict):
        entries = []
        for flat_label, keywords in raw.items():
            entries.append(
                TaxonomyEntry(
                    large_category=_infer_large_category(flat_label),
                    middle_category=str(flat_label),
                    small_category=None,
                    flat_label=str(flat_label),
                    keywords=[str(keyword) for keyword in keywords],
                    aliases=[str(flat_label)],
                    folder_target=f"{_infer_large_category(flat_label)}/{flat_label}",
                )
            )
        return Taxonomy(entries=entries, version="1.0-compat")

    raise ValueError("Unsupported taxonomy format.")


def _infer_large_category(label: str) -> str:
    mapping = {
        "계약서": "legal",
        "청구서": "finance",
        "영수증": "finance",
        "보고서": "reporting",
        "발표자료": "reporting",
        "공지": "operations",
        "사업계획서": "planning",
        "과업지시서": "operations",
        "사업자등록증": "corporate",
        "법인등기부등본": "corporate",
        "재무제표증명": "finance",
        "중소기업확인서": "certifications",
        "지방세완납증명서": "certifications",
        "벤처기업확인서": "certifications",
        "데이터": "data",
    }
    return mapping.get(label, "miscellaneous")
