"""
stage6_cluster.py — Stage 6: 군집화 (비활성)

[상태] 팀 회의 결정으로 HDBSCAN 군집 미사용. 항상 빈 목록을 반환합니다.
[이유] API 응답의 clusters 필드 호환을 위해 모듈은 유지합니다.
[담당] 천승원 (feature/stage6-cluster) — 대체 방식 확정 시 이 파일 수정
"""

from typing import Any


def run(job_items: list[dict[str, Any]]) -> list[dict]:
    """군집화 비활성. job_items 를 받지만 사용하지 않고 [] 를 반환합니다."""
    _ = job_items
    return []
