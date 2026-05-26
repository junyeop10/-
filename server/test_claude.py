"""stage5_llm_claude 모듈 동작 확인 (더미 EvidencePackage)."""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from models.schemas import EvidencePackage
from pipeline.stage5_llm_claude import (
    _build_user_prompt,
    classify_with_claude,
)

load_dotenv(Path(__file__).resolve().parent / ".env")


def _dummy_package() -> EvidencePackage:
    """테스트용 더미 EvidencePackage."""
    return EvidencePackage(
        xxhash="test_hash_001",
        filename="2024_사업보고서_최종.pdf",
        ext=".pdf",
        size_kb=512.0,
        modified_at=0.0,
        text_front="1. 개요\n본 보고서는 2024년 사업 성과를 분석한 결과입니다.",
        text_middle="2. 분석 결과\n매출 증가율 15%, 주요 지표는 첨부 표 참조.",
        text_rear="3. 결론\n향후 전략은 데이터 기반 의사결정 강화입니다.",
        trigger_chunks=[],
        keyword_hits=["보고서", "분석", "결과"],
        pattern_flags={
            "is_announcement": False,
            "is_contract": False,
            "is_draft": False,
        },
        version_hint="최종",
        embedding=[],
        extract_method="pymupdf",
        extract_status="success",
    )


async def main() -> None:
    pkg = _dummy_package()

    print("=== 유저 프롬프트 미리보기 (앞 500자) ===")
    prompt = _build_user_prompt(pkg)
    print(prompt[:500])
    print("...\n")

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY 없음 — API 호출 생략")
        print("(프롬프트 생성까지 확인 완료)")
        return

    print("=== classify_with_claude 호출 ===")
    result = await classify_with_claude(pkg)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
