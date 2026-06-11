"""
stage4_version.py — 버전·중복 그룹 정리 (비활성)

[상태] 플로우차트 최종에 버전 정리 단계 없음. 항상 빈 목록을 반환합니다.
[이유] API 응답의 version_groups 필드 호환을 위해 모듈은 유지합니다.
"""

from models.schemas import ClassifyResult


def register_embedding(xxhash: str, embedding: list[float]) -> None:
    """비활성 — 호출해도 아무 동작 없음."""
    _ = (xxhash, embedding)


def clear_embeddings() -> None:
    """비활성 — 호출해도 아무 동작 없음."""
    return None


def run(results: list[ClassifyResult]) -> list[dict]:
    """버전 정리 비활성. results 를 받지만 사용하지 않고 [] 를 반환합니다."""
    _ = results
    return []
