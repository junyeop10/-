"""LLM 스모크 테스트용 — 디스크 파일 → EvidencePackage (main 파이프라인과 동일 추출·증거 단계)."""

from __future__ import annotations

from pathlib import Path

from db import cache
from models.schemas import EvidencePackage
from pipeline import pre_stage, stage0_extract, stage2_ocr, stage4_embedding
from pipeline.pre_stage import SUPPORTED_EXTENSIONS


def iter_test_paths(paths: list[str]) -> list[Path]:
    """
    파일 경로 또는 폴더 경로 목록을 테스트 대상 파일 Path 리스트로 펼칩니다.

    폴더인 경우 지원 확장자만 1단계 glob (하위 폴더는 제외).
    """
    resolved: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"경로 없음: {path}")
        if path.is_dir():
            for ext in sorted(SUPPORTED_EXTENSIONS):
                resolved.extend(sorted(path.glob(f"*{ext}")))
            continue
        if path.is_file():
            resolved.append(path)
            continue
        raise ValueError(f"파일/폴더가 아님: {path}")

    if not resolved:
        raise ValueError("테스트할 파일이 없습니다 (폴더가 비었거나 지원 확장자 없음)")
    return resolved


def build_evidence_from_path(file_path: Path) -> EvidencePackage:
    """
    단일 파일을 읽어 pre → 추출 → OCR → EvidencePackage 를 만듭니다.

    main.run_pipeline 과 동일한 추출·증거 단계이며 LLM 분류는 호출하지 않습니다.
    """
    file_bytes = file_path.read_bytes()
    filename = file_path.name
    ext = file_path.suffix.lower()
    modified_at = file_path.stat().st_mtime
    size_kb = len(file_bytes) / 1024

    pre_result = pre_stage.run(file_bytes, filename, modified_at)
    if pre_result["status"] == "review_queue":
        raise ValueError(f"{filename}: {pre_result.get('reason', 'pre_stage 거부')}")

    if pre_result["status"] == "cached":
        xxhash = pre_result.get("xxhash") or cache.compute_hash(file_bytes)
    else:
        xxhash = pre_result["xxhash"]

    extract_result = stage0_extract.run(file_bytes, filename, ext)
    extract_result = stage2_ocr.run(file_bytes, filename, ext, extract_result)

    return stage4_embedding.run(
        file_bytes, filename, ext, size_kb, modified_at, xxhash, extract_result
    )
