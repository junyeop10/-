"""xxhash 기반 파일 해시 계산 모듈입니다."""

from __future__ import annotations

from pathlib import Path

from src.text_cleaner import normalize_text


def _load_xxhash():
    try:
        import xxhash
    except ImportError as error:
        raise ImportError("xxhash is required. Install it with `pip install xxhash`.") from error
    return xxhash


def compute_raw_text_hash(text: str) -> str:
    """Compute a stable xxHash64 digest without text normalization."""
    return _load_xxhash().xxh64((text or "").encode("utf-8")).hexdigest()


def compute_xxhash64(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """파일 내용을 스트리밍으로 읽어 xxhash64 해시를 계산합니다."""
    try:
        import xxhash
    except ImportError as error:
        raise ImportError(
            "xxhash가 설치되어 있지 않습니다. `pip install xxhash`로 설치하세요."
        ) from error

    file_path = Path(path)
    hasher = xxhash.xxh64()

    with file_path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)

    return hasher.hexdigest()


def compute_file_hash(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Backward-compatible alias for the xxHash64 file-content digest."""
    return compute_xxhash64(path, chunk_size=chunk_size)


def compute_text_hash(text: str) -> str:
    """Compute a stable xxHash64 digest from cleaned text."""
    cleaned = normalize_text(text or "")
    return compute_raw_text_hash(cleaned)
