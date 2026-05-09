"""선택적으로 LLM 보정을 붙일 수 있도록 분리한 모듈입니다."""

from __future__ import annotations


def get_optional_llm_score(enabled: bool) -> float:
    """현재 기본 구현에서는 LLM 보정을 사용하지 않으므로 0점을 반환합니다."""
    if enabled:
        return 0.0
    return 0.0
