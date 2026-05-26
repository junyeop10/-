"""Ollama Qwen 분류 스모크 테스트 (Ollama 실행 필요)."""

import asyncio

from models.schemas import EvidencePackage
from pipeline.stage5_llm_local import classify_with_qwen


async def main() -> None:
    pkg = EvidencePackage(
        xxhash="test",
        filename="2024_최종_보고서_v3.docx",
        ext=".docx",
        size_kb=120.5,
        modified_at=0.0,
        text_front="본 보고서는 2024년 4분기 실적을 확정하였습니다.",
        text_middle="",
        text_rear="",
        trigger_chunks=[],
        keyword_hits=["최종", "보고서"],
        pattern_flags={},
        version_hint="최종",
        embedding=[],
        extract_method="docx",
        extract_status="ok",
    )
    result = await classify_with_qwen(pkg)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
