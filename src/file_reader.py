"""File discovery and text extraction helpers."""

from __future__ import annotations

from pathlib import Path

from src.office_reader import extract_docx_text, extract_pptx_text, extract_xlsx_text
from src.pdf_reader import extract_pdf_text
from src.text_cleaner import build_sampled_text


SUPPORTED_SUFFIXES = {".txt", ".pdf", ".docx", ".xlsx", ".pptx"}


def ensure_input_directory(input_dir: str | Path) -> Path:
    """Create the input directory when it does not exist."""
    directory = Path(input_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def discover_supported_files(input_dir: str | Path) -> list[Path]:
    """Find supported input files under an input directory."""
    directory = ensure_input_directory(input_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {directory}")

    files = [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return sorted(files)


def read_txt_file(path: str | Path) -> str:
    """Read a txt file with common encodings."""
    file_path = Path(path)
    for encoding in ("utf-8", "cp949"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        "text",
        b"",
        0,
        1,
        f"Could not read with utf-8 or cp949: {file_path}",
    )


def extract_text_from_file(
    path: str | Path,
    fast: bool = True,
    total_limit: int = 4500,
    part_limit: int = 1500,
) -> str:
    """Extract evidence text from a supported file."""
    file_path = Path(path)

    if file_path.suffix.lower() == ".txt":
        text = read_txt_file(file_path)
        return build_sampled_text(text, total_limit=total_limit, part_limit=part_limit)
    if file_path.suffix.lower() == ".pdf":
        return extract_pdf_text(
            file_path,
            fast=fast,
            total_limit=total_limit,
            part_limit=part_limit,
        )
    if file_path.suffix.lower() == ".docx":
        return extract_docx_text(
            file_path,
            total_limit=total_limit,
            part_limit=part_limit,
        )
    if file_path.suffix.lower() == ".xlsx":
        return extract_xlsx_text(
            file_path,
            fast=fast,
            total_limit=total_limit,
            part_limit=part_limit,
        )
    if file_path.suffix.lower() == ".pptx":
        return extract_pptx_text(
            file_path,
            total_limit=total_limit,
            part_limit=part_limit,
        )

    raise ValueError(f"Unsupported file type: {file_path.suffix}")
