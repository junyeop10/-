"""
stage2_cluster.py — (레거시) HDBSCAN 스텁

[역할] 예전 설계의 Stage 2 군집 자리. 현재는 입력을 그대로 반환합니다.
[참고] 실제 군집은 stage6_cluster.py (job 단위). 회의 결정으로 HDBSCAN 제거 예정.
"""

from models.schemas import ClassifyResult


def run(results: list[ClassifyResult]) -> list[ClassifyResult]:
    """Stage 2 (HDBSCAN) is not implemented; pass through to stage 3."""
    return results
