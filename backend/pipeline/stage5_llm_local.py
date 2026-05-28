"""Stage 5 LLM — 로컬 Qwen (Ollama) 비동기 래퍼 (stage3_llm_ollama 위임)."""

import asyncio

from models.schemas import EvidencePackage
from pipeline.stage3_llm_ollama import classify_with_ollama


async def classify_with_qwen(pkg: EvidencePackage) -> dict:
    """
    EvidencePackage를 Ollama로 분류합니다 (비동기 파이프라인 호환).

    실제 구현은 stage3_llm_ollama.classify_with_ollama (requests, /api/generate) 입니다.
    """
    return await asyncio.to_thread(classify_with_ollama, pkg)
