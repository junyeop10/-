"""xxhash 기반 파일 해시 계산 모듈입니다."""

from __future__ import annotations

from pathlib import Path


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
