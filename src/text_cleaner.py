"""Text normalization, tokenization, and evidence sampling helpers."""

from __future__ import annotations

import re


TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z0-9]{2,}")


def normalize_text(text: str) -> str:
    """Normalize text for rules and embeddings."""
    if not isinstance(text, str):
        raise TypeError("normalize_text expects a string.")

    cleaned = text.replace("\x00", " ")
    cleaned = cleaned.lower().strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def tokenize_text(text: str) -> list[str]:
    """Extract simple Korean, English, and numeric tokens."""
    normalized = normalize_text(text)
    return TOKEN_PATTERN.findall(normalized)


def build_sampled_text(
    text: str,
    total_limit: int = 4500,
    part_limit: int = 1500,
) -> str:
    """Build evidence text from beginning, middle, and end excerpts."""
    if not isinstance(text, str):
        raise TypeError("build_sampled_text expects a string.")

    cleaned = text.strip()
    if len(cleaned) <= total_limit:
        return cleaned

    label_text = "[BEGIN_EXCERPT]\n\n[MIDDLE_EXCERPT]\n\n[END_EXCERPT]\n"
    available_text_limit = max(total_limit - len(label_text), 300)
    effective_part_limit = min(part_limit, available_text_limit // 3)
    begin = cleaned[:effective_part_limit]

    middle_start = max((len(cleaned) // 2) - (effective_part_limit // 2), 0)
    middle_end = middle_start + effective_part_limit
    middle = cleaned[middle_start:middle_end]

    end = cleaned[-effective_part_limit:]

    return (
        "[BEGIN_EXCERPT]\n"
        f"{begin}\n"
        "[MIDDLE_EXCERPT]\n"
        f"{middle}\n"
        "[END_EXCERPT]\n"
        f"{end}"
    )
