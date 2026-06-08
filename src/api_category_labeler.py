"""API labeling boundary for clustered document evidence."""

from __future__ import annotations

from typing import Any


def build_cluster_labeling_payload(cluster_summary: dict[str, Any]) -> dict[str, Any]:
    """Build the payload that a future category-labeling API will receive."""
    payload = {
        "cluster_id": cluster_summary.get("cluster_id"),
        "document_count": cluster_summary.get("document_count", 0),
        "representative_documents": cluster_summary.get("representative_documents", []),
        "common_signals": cluster_summary.get("common_signals", {}),
        "instruction": (
            "이 군집의 문서 카테고리를 판단하라. 기존 카테고리에 억지로 맞추지 말고 "
            "필요하면 새 카테고리 후보를 제안하라."
        ),
    }
    if "parent_cluster_id" in cluster_summary:
        payload.update(
            {
                "scope": "parent_type_candidate",
                "parent_cluster_id": cluster_summary.get("parent_cluster_id"),
                "fine_cluster_ids": cluster_summary.get("fine_cluster_ids", []),
                "fine_cluster_count": cluster_summary.get("fine_cluster_count", 0),
                "grouping": cluster_summary.get("grouping", {}),
            }
        )
    else:
        payload["scope"] = "fine_cluster"
    return payload


def call_category_labeling_api(payload: dict[str, Any]) -> dict[str, Any]:
    """Placeholder only; no real API call is made yet."""
    del payload
    return {
        "category": None,
        "confidence": 0.0,
        "reason": [],
        "new_category_needed": None,
        "status": "api_not_configured",
    }
