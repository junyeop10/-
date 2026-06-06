"""
stage6_cluster.py — Stage 6: HDBSCAN 군집화 (job 단위)

[역할] 한 번에 업로드된 여러 파일의 임베딩을 묶어 유사 문서 클러스터를 만듭니다.
[입력] [{"xxhash", "embedding", "filename"}, ...]
[출력] [{"cluster_id", "filenames", "xxhashes", "size"}, ...] (최대 3개)
[상태] 회의 결정으로 HDBSCAN 미사용 예정 — 제거·대체 전까지 유지
[담당] 천승원 (feature/stage6-cluster)
"""

from typing import Any

import numpy as np

MAX_CLUSTERS = 3
MIN_CLUSTER_SIZE = 2


def run(job_items: list[dict[str, Any]]) -> list[dict]:
    """
    job 내 파일 임베딩을 HDBSCAN으로 군집합니다.

    job_items: [{"xxhash", "embedding", "filename"}, ...]
    반환: [{"cluster_id", "filenames", "xxhashes"}, ...] (최대 3개 클러스터만 라벨)
    """
    try:
        valid = [
            item
            for item in job_items
            if item.get("embedding") and len(item["embedding"]) > 0
        ]
        if len(valid) < MIN_CLUSTER_SIZE:
            return []

        matrix = np.array([item["embedding"] for item in valid], dtype=float)

        import hdbscan

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=MIN_CLUSTER_SIZE,
            min_samples=1,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(matrix)

        groups: dict[int, list[dict]] = {}
        for item, label in zip(valid, labels):
            lid = int(label)
            if lid < 0:
                continue
            groups.setdefault(lid, []).append(item)

        if not groups:
            return []

        sorted_ids = sorted(
            groups.keys(),
            key=lambda k: len(groups[k]),
            reverse=True,
        )[:MAX_CLUSTERS]

        result = []
        for cluster_id, cid in enumerate(sorted_ids):
            members = groups[cid]
            result.append(
                {
                    "cluster_id": cluster_id,
                    "filenames": [m["filename"] for m in members],
                    "xxhashes": [m["xxhash"] for m in members],
                    "size": len(members),
                }
            )
        return result
    except Exception:
        return []
