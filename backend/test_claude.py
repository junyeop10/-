"""
test_claude.py — Claude 1차 분류 CLI 테스트 (서버 없이)

[역할] 로컬 파일·폴더를 넣어 추출 + Claude 1차 분류만 빠르게 확인합니다.
       RAG 2차·검토큐는 stage5_classify (서버) 에서만 동작합니다.
[사용]
  python test_claude.py 파일.pdf              실제 API 호출
  python test_claude.py 파일.pdf --dry-run  프롬프트만 (비용 0)
  python test_claude.py 폴더/               지원 확장자 일괄
[의존] test_llm_helpers.py (파일 → EvidencePackage 변환)
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
from pipeline.stage5_claude import classify_with_claude
from pipeline.stage5_common import build_user_prompt
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
