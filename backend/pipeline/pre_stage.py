"""
pre_stage.py — 업로드 직후 사전 검사·캐시 조회

[역할] 지원 확장자·파일 크기를 검사하고, 동일 파일(xxhash) 재업로드 시 캐시 히트를 반환합니다.
[입력] file_bytes, filename, modified_at
[출력] {"status": "continue"|"cached"|"review_queue", "xxhash", ...}
[다음] continue → stage0_extract / cached → main.py 에서 즉시 완료
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from db import cache

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".hwp",
    ".hwpx",
    ".pptx",
    ".xlsx",
    ".jpg",
    ".jpeg",
    ".png",
}

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))


def run(file_bytes: bytes, filename: str, modified_at: float) -> dict:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return {"status": "review_queue", "reason": "지원하지 않는 파일 형식"}

    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        return {"status": "review_queue", "reason": "파일 크기 초과"}

    file_hash = cache.compute_hash(file_bytes)
    cached = cache.get_cached(file_hash)
    if cached is not None:
        return {"status": "cached", "cached_result": cached, "xxhash": file_hash}

    return {"status": "ok", "xxhash": file_hash}
