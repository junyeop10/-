"""
Claude API 분류 스모크 테스트 (실제 파일 경로/폴더).

사용 예:
  python test_claude.py C:\\samples\\보고서.pdf
  python test_claude.py ./docs/*.pdf
  python test_claude.py C:\\samples\\ --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from models.schemas import EvidencePackage
from pipeline.stage5_llm_claude import classify_with_claude
from pipeline.stage5_llm_common import build_user_prompt
from test_llm_helpers import build_evidence_from_path, iter_test_paths

load_dotenv(Path(__file__).resolve().parent / ".env")


def _print_json(obj: object) -> None:
    sys.stdout.buffer.write(
        (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )


def _text_chars(pkg: EvidencePackage) -> int:
    return len(pkg.text_front or "") + len(pkg.text_middle or "") + len(pkg.text_rear or "")


async def _run_file(
    path: Path, dry_run: bool, skip_api: bool, force_llm: bool
) -> dict:
    pkg = build_evidence_from_path(path)
    row: dict = {
        "file": str(path),
        "filename": pkg.filename,
        "extract_status": pkg.extract_status,
        "extract_method": pkg.extract_method,
        "text_chars": _text_chars(pkg),
        "keyword_hits": pkg.keyword_hits,
        "version_hint": pkg.version_hint,
        "text_front_preview": (pkg.text_front or "")[:200],
    }
    if pkg.extract_status != "success":
        row["warning"] = (
            "본문 추출 실패 — LLM은 파일명만으로 추측할 수 있습니다."
        )

    if dry_run or skip_api:
        row["dry_run"] = True
        row["prompt_preview"] = build_user_prompt(pkg)[:500]
        if skip_api and not dry_run:
            row["skipped"] = "ANTHROPIC_API_KEY 없음"
        return row

    if pkg.extract_status != "success" and not force_llm:
        row["llm_skipped"] = True
        return row

    row["llm"] = await classify_with_claude(pkg)
    return row


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Claude API 파일 경로/폴더 분류 테스트"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="파일 경로 또는 폴더",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="API 호출 없이 프롬프트 미리보기만",
    )
    parser.add_argument(
        "--force-llm",
        action="store_true",
        help="추출 실패여도 API 호출 (기본은 스킵)",
    )
    args = parser.parse_args()

    skip_api = not os.getenv("ANTHROPIC_API_KEY")
    files = iter_test_paths(args.paths)
    results: list[dict] = []
    errors: list[dict] = []

    for path in files:
        try:
            results.append(
                await _run_file(path, args.dry_run, skip_api, args.force_llm)
            )
        except Exception as exc:
            errors.append({"file": str(path), "error": str(exc)})

    _print_json({"ok": results, "errors": errors})
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
