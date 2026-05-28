"""
Stage 3 LLM — Ollama / Claude 런타임 자동 선택 라우터.

프로세스 시작 시 Ollama 서버 가용 여부를 1회 확인하고,
이후 classify_with_llm 호출마다 동일한 백엔드를 사용합니다.
"""

import logging
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

from models.schemas import EvidencePackage
from pipeline.stage3_llm_claude import classify_with_claude
from pipeline.stage3_llm_ollama import classify_with_ollama

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_PROBE_TIMEOUT_SEC = float(os.getenv("OLLAMA_PROBE_TIMEOUT_SEC", "3"))

_ollama_available: bool | None = None
_last_backend: str = ""


def probe_ollama_server() -> bool:
    """
    Ollama 서버 응답 여부를 확인합니다 (GET 루트).

    연결 실패·타임아웃 시 False 를 반환합니다.
    """
    try:
        response = requests.get(
            f"{OLLAMA_BASE_URL}/",
            timeout=OLLAMA_PROBE_TIMEOUT_SEC,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


def _ensure_probe() -> bool:
    """최초 1회 Ollama 프로브 후 결과를 캐시합니다."""
    global _ollama_available
    if _ollama_available is None:
        _ollama_available = probe_ollama_server()
        if _ollama_available:
            logger.info("LLM 백엔드: Ollama (로컬, %s)", OLLAMA_BASE_URL)
        else:
            logger.info(
                "LLM 백엔드: Claude API (Ollama 미응답, %s)", OLLAMA_BASE_URL
            )
    return _ollama_available


def reset_router_cache() -> None:
    """테스트용: Ollama 가용성 캐시를 초기화합니다."""
    global _ollama_available, _last_backend
    _ollama_available = None
    _last_backend = ""


def get_last_llm_backend() -> str:
    """마지막 classify_with_llm 호출에 사용된 백엔드 (ollama | claude)."""
    return _last_backend


def is_ollama_available() -> bool:
    """캐시된 Ollama 가용 여부 (없으면 프로브 수행)."""
    return _ensure_probe()


def classify_with_llm(pkg: EvidencePackage) -> dict:
    """
    런타임 전략에 따라 Ollama 또는 Claude로 분류합니다.

    Ollama 응답 가능 → classify_with_ollama, 그렇지 않으면 classify_with_claude.
    사용 백엔드는 로그 및 get_last_llm_backend() 로 확인할 수 있습니다.
    """
    global _last_backend

    if _ensure_probe():
        _last_backend = "ollama"
        logger.debug("classify_with_llm → Ollama")
        return classify_with_ollama(pkg)

    _last_backend = "claude"
    logger.debug("classify_with_llm → Claude API")
    return classify_with_claude(pkg)


# 모듈 import 시점에 Ollama 프로브 (프로그램 시작 시 1회)
_ensure_probe()
