"""PDF text extraction helpers."""

from __future__ import annotations

from pathlib import Path

from src.text_cleaner import build_sampled_text


def extract_pdf_text(
    path: str | Path,
    fast: bool = True,
    total_limit: int = 4500,
    part_limit: int = 1500,
    pages_per_section: int = 2,
) -> str:
    """Extract evidence text from first, middle, and last PDF pages."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    try:
        import fitz
    except ImportError as error:
        raise ImportError("PyMuPDF is required. Run: pip install -r requirements.txt") from error

    page_texts: list[str] = []
    with fitz.open(file_path) as document:
        page_count = len(document)
        if page_count == 0:
            return ""

        page_indexes = _select_page_indexes(
            page_count=page_count,
            fast=fast,
            pages_per_section=pages_per_section,
        )
        for page_index in page_indexes:
            page = document.load_page(page_index)
            page_texts.append(page.get_text("text"))

    combined_text = "\n".join(text.strip() for text in page_texts if text.strip())
    return build_sampled_text(combined_text, total_limit=total_limit, part_limit=part_limit)


def _select_page_indexes(page_count: int, fast: bool, pages_per_section: int) -> list[int]:
    """Select PDF pages for evidence extraction."""
    if not fast:
        return list(range(page_count))

    selected: set[int] = set()
    windows = [
        0,
        max((page_count // 2) - (pages_per_section // 2), 0),
        max(page_count - pages_per_section, 0),
    ]

    for start in windows:
        for offset in range(pages_per_section):
            page_index = start + offset
            if 0 <= page_index < page_count:
                selected.add(page_index)

    return sorted(selected)
