"""
stage2_cluster.py — (레거시) HDBSCAN 스텁

[역할] 예전 설계의 Stage 2 군집 자리. 현재는 입력을 그대로 반환합니다.
[참고] stage6_cluster.py 는 HDBSCAN 미사용으로 비활성(항상 []).
"""

from models.schemas import ClassifyResult


def run(results: list[ClassifyResult]) -> list[ClassifyResult]:
    """Stage 2 (HDBSCAN) is not implemented; pass through to stage 3."""
    return results
