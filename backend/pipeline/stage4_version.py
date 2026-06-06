"""
stage4_version.py — 버전·중복 문서 그룹 정리

[역할] job 내 파일들의 임베딩·파일명 유사도로 버전/중복 그룹을 만듭니다.
       업로드 배치가 끝난 뒤 main.py 가 한 번 호출합니다.
[입력] results (ClassifyResult 목록), job 단위 임베딩 맵
[출력] version_groups — 대표 파일 + 버전 목록
"""

import os
from pathlib import Path

import Levenshtein
import numpy as np

from models.schemas import ClassifyResult

_embeddings_by_xxhash: dict[str, list[float]] = {}


def register_embedding(xxhash: str, embedding: list[float]) -> None:
    if embedding:
        _embeddings_by_xxhash[xxhash] = embedding


def clear_embeddings() -> None:
    _embeddings_by_xxhash.clear()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def _version_rank(hint: str) -> int:
    order = {"최종": 3, "rev": 2, "draft": 1}
    return order.get(hint, 0)


def _modified_at(result: ClassifyResult) -> float:
    if result.file_path and os.path.exists(result.file_path):
        return os.path.getmtime(result.file_path)
    return 0.0


def _sort_versions(versions: list[ClassifyResult]) -> list[ClassifyResult]:
    return sorted(
        versions,
        key=lambda r: (_version_rank(r.version_hint), _modified_at(r)),
        reverse=True,
    )


def run(results: list[ClassifyResult]) -> list[dict]:
    if not results:
        return []

    groups: list[dict] = []
    used: set[str] = set()

    by_hash: dict[str, list[ClassifyResult]] = {}
    for r in results:
        by_hash.setdefault(r.xxhash, []).append(r)

    for xxhash, dupes in by_hash.items():
        if len(dupes) > 1:
            representative = dupes[0]
            groups.append(
                {
                    "representative": representative,
                    "versions": dupes,
                    "is_duplicate": True,
                }
            )
            for d in dupes:
                used.add(id(d))
        elif dupes:
            used.add(id(dupes[0]))

    remaining = [r for r in results if id(r) not in used]

    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        i = 0
        while i < len(remaining):
            candidate = remaining[i]
            dist = Levenshtein.distance(
                Path(seed.filename).stem.lower(),
                Path(candidate.filename).stem.lower(),
            )
            if dist <= 5:
                emb_a = _embeddings_by_xxhash.get(seed.xxhash, [])
                emb_b = _embeddings_by_xxhash.get(candidate.xxhash, [])
                sim = _cosine_similarity(emb_a, emb_b)
                if not emb_a or not emb_b or sim >= 0.85:
                    cluster.append(candidate)
                    remaining.pop(i)
                    continue
            i += 1

        sorted_cluster = _sort_versions(cluster)
        groups.append(
            {
                "representative": sorted_cluster[0],
                "versions": sorted_cluster,
                "is_duplicate": False,
            }
        )

    return groups
