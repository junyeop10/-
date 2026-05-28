"""
Stage 3 LLM — Ollama 로컬 백엔드 (qwen2.5:3b).

환경 안내 (MVP, 코드 미포함):
- Windows: Ollama Desktop 설치 후 `ollama pull qwen2.5:3b`
- WSL2 + GPU: NVIDIA 드라이버·CUDA 후 Ollama Linux 설치 권장
- CPU만: 양자화 모델(qwen2.5:3b) 사용, 응답 60초 내 타임아웃 처리
"""

import logging
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

from models.schemas import EvidencePackage
from pipeline.stage5_llm_common import (
    build_generate_prompt,
    failure_result,
    parse_response_text,
)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_TIMEOUT_SEC = int(os.getenv("OLLAMA_TIMEOUT_SEC", "60"))


def classify_with_ollama(pkg: EvidencePackage) -> dict:
    """
    Ollama REST API(/api/generate)로 EvidencePackage를 분류합니다.

    stream=False로 전체 응답을 수신하며, JSON 파싱 실패·타임아웃 시 분류불가를 반환합니다.
    """
    prompt = build_generate_prompt(pkg)
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }

    for attempt in range(2):
        try:
            response = requests.post(
                OLLAMA_GENERATE_URL,
                json=payload,
                timeout=OLLAMA_TIMEOUT_SEC,
            )
            response.raise_for_status()
            data = response.json()
            text = str(data.get("response", ""))
            parsed = parse_response_text(text)
            if parsed is not None:
                return parsed
        except requests.Timeout:
            logger.warning("Ollama 타임아웃 (%ss)", OLLAMA_TIMEOUT_SEC)
            return failure_result("타임아웃 초과")
        except requests.HTTPError as exc:
            err_body = ""
            if exc.response is not None:
                try:
                    err_body = str(exc.response.json().get("error", ""))
                except (ValueError, AttributeError):
                    err_body = exc.response.text[:200]
            logger.warning(
                "Ollama HTTP 오류 (시도 %s): %s %s",
                attempt + 1,
                exc.response.status_code if exc.response else "",
                err_body,
            )
            if "unable to allocate" in err_body.lower() or "memory" in err_body.lower():
                return failure_result("메모리 부족 — qwen2.5:0.5b 등 더 작은 모델 권장")
            if attempt == 1:
                return failure_result("Ollama 서버 오류")
            continue
        except requests.RequestException as exc:
            logger.warning("Ollama 요청 실패 (시도 %s): %s", attempt + 1, exc)
            if attempt == 1:
                return failure_result("Ollama 미연결")
            continue
        except (ValueError, KeyError):
            if attempt == 1:
                return failure_result("JSON 파싱 실패")
            continue

    return failure_result("JSON 파싱 실패")
