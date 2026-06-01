"""
로컬 LLM(Ollama/Qwen) 분류 스모크 테스트.

사용 예:
  python test_qwen.py C:\\samples\\보고서.pdf
  python test_qwen.py file1.docx file2.pdf
  python test_qwen.py C:\\samples\\          # 폴더 내 지원 확장자 전부
  python test_qwen.py report.pdf --dry-run   # 추출·증거만 확인
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from models.schemas import EvidencePackage
from pipeline.stage5_llm_local import classify_with_qwen
from test_llm_helpers import build_evidence_from_path, iter_test_paths


def _print_json(obj: object) -> None:
    sys.stdout.buffer.write(
        (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )


def _text_chars(pkg: EvidencePackage) -> int:
    return len(pkg.text_front or "") + len(pkg.text_middle or "") + len(pkg.text_rear or "")


async def _run_file(path: Path, dry_run: bool, force_llm: bool) -> dict:
    pkg = build_evidence_from_path(path)
    text_len = _text_chars(pkg)
    row: dict = {
        "file": str(path),
        "filename": pkg.filename,
        "extract_status": pkg.extract_status,
        "extract_method": pkg.extract_method,
        "text_chars": text_len,
        "keyword_hits": pkg.keyword_hits,
        "version_hint": pkg.version_hint,
        "text_front_preview": (pkg.text_front or "")[:200],
    }

    if pkg.extract_status != "success":
        row["warning"] = (
            "본문 추출 실패 또는 미구현 형식 — LLM은 파일명·메타만 보고 추측합니다. "
            "신뢰도가 높아도 실제 분류로 쓰면 안 됩니다."
        )

    if dry_run:
        row["dry_run"] = True
        return row

    if pkg.extract_status != "success" and not force_llm:
        row["llm_skipped"] = True
        return row

    row["llm"] = await classify_with_qwen(pkg)
    return row


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="로컬 LLM(Ollama) 파일 경로/폴더 분류 테스트"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="파일 경로 또는 폴더 (폴더면 지원 확장자 파일 전부)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="LLM 호출 없이 추출·EvidencePackage 메타만 출력",
    )
    parser.add_argument(
        "--force-llm",
        action="store_true",
        help="추출 실패(hwp 등)여도 LLM 호출 (기본은 스킵)",
    )
    args = parser.parse_args()

    files = iter_test_paths(args.paths)
    results: list[dict] = []
    errors: list[dict] = []

    for path in files:
        try:
            results.append(await _run_file(path, args.dry_run, args.force_llm))
        except Exception as exc:
            errors.append({"file": str(path), "error": str(exc)})

    _print_json({"ok": results, "errors": errors})
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
