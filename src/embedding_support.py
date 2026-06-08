"""Embedding helpers for evidence-based cluster workflows."""

from __future__ import annotations

from typing import Any

from src.config import AppConfig
from src.vectorizer import SentenceTransformerEmbedder


def build_embedding_text(evidence: dict[str, Any]) -> str:
    """Create a compact, structured text payload for document embeddings."""
    structural = evidence.get("structural_features") or {}
    layout = evidence.get("layout_features") or {}
    text_stats = evidence.get("text_stats") or {}
    top_tokens = [
        str(item.get("token", ""))
        for item in evidence.get("top_tokens", [])
        if isinstance(item, dict) and str(item.get("token", "")).strip()
    ][:30]
    filename_tokens = [str(token) for token in evidence.get("filename_tokens", [])][:30]

    structure_keys = (
        "page_count",
        "slide_count",
        "sheet_count",
        "table_count",
        "image_count",
        "legal_term_density",
        "clause_pattern_score",
        "receipt_pattern_score",
        "approval_block_score",
        "research_structure_score",
        "report_structure_score",
    )
    layout_keys = (
        "signature_area_score",
        "chart_presence_score",
        "dense_text_score",
        "receipt_pattern_score",
        "certificate_pattern_score",
        "approval_block_score",
    )
    text_stat_keys = ("char_count", "token_count", "unique_token_count", "low_quality_scan_score")

    return "\n".join(
        line
        for line in (
            f"filename: {evidence.get('filename', '')}",
            f"filename_tokens: {', '.join(filename_tokens)}",
            f"top_tokens: {', '.join(top_tokens)}",
            f"sampled_text: {evidence.get('sampled_text', '')}",
            f"structure: {_format_feature_line(structural, structure_keys)}",
            f"layout: {_format_feature_line(layout, layout_keys)}",
            f"text_stats: {_format_feature_line(text_stats, text_stat_keys)}",
        )
        if line.strip()
    ).strip()


def embed_texts(
    texts: list[str],
    *,
    embedder: Any | None = None,
    repository: Any | None = None,
    file_hashes: list[str] | None = None,
    config: AppConfig | None = None,
    text_kind: str = "cluster_evidence",
    embedding_version: str | None = None,
) -> list[list[float]]:
    """Embed texts through the configured embedder while preserving cache hooks."""
    if not texts:
        return []
    active_embedder = embedder or SentenceTransformerEmbedder()
    if hasattr(active_embedder, "encode_many"):
        return active_embedder.encode_many(
            texts,
            repository=repository,
            file_hashes=file_hashes or [""] * len(texts),
            text_kind=text_kind,
            embedding_version=(
                embedding_version
                if embedding_version is not None
                else (config.embedding.model_version if config is not None else "cluster-evidence-v1")
            ),
        )
    return [active_embedder.encode(text) for text in texts]


def _format_feature_line(features: dict[str, Any], keys: tuple[str, ...]) -> str:
    return " ".join(f"{key}={features.get(key, 0)}" for key in keys if key in features)
