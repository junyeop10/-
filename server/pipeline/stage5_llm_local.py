"""Stage 5 LLM — 로컬 Qwen (Ollama) 분류 모듈."""

import asyncio
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

from models.schemas import EvidencePackage
from pipeline.stage5_llm_common import (
    SYSTEM_PROMPT,
    build_user_prompt,
    failure_result,
    parse_response_text,
)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_TIMEOUT_SEC = float(os.getenv("OLLAMA_TIMEOUT_SEC", "120"))
MAX_CONCURRENT_LOCAL_LLM = int(os.getenv("MAX_CONCURRENT_LOCAL_LLM", "2"))

_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    """동시 로컬 LLM 호출 수 제한 (CPU 부하 완화)."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENT_LOCAL_LLM)
    return _semaphore


async def _call_ollama_once(user_prompt: str) -> dict | None:
    """Ollama /api/chat 1회 호출."""
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
    }
    async with _get_semaphore():
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT_SEC) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            text = data.get("message", {}).get("content", "")
            return parse_response_text(text)


async def classify_with_qwen(pkg: EvidencePackage) -> dict:
    """
    EvidencePackage를 Ollama 로컬 Qwen으로 분류합니다.

    Ollama 미실행·연결 실패 시 failure_result 반환. 성공 시 JSON dict 반환.
    1회 재시도 후 파싱 실패 시 failure_result.
    """
    user_prompt = build_user_prompt(pkg)

    for attempt in range(2):
        try:
            parsed = await _call_ollama_once(user_prompt)
            if parsed is not None:
                return parsed
        except (httpx.HTTPError, OSError, TimeoutError):
            if attempt == 1:
                return failure_result("Ollama 미연결")
            continue
        except Exception:
            if attempt == 1:
                return failure_result("API 오류")
            continue

    return failure_result("JSON 파싱 실패")
