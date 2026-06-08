"""
embed_cache.py
--------------
[우선순위 3] 임베딩 디스크 캐시 — 같은 본문을 다시 임베딩하지 않는다.

문서 본문(front/middle/rear)의 해시를 키로 *최종 3구간 가중평균 벡터*를
pickle dict에 저장한다. 재실행/재분류 시 변하지 않은 파일은 모델 호출을
건너뛰어 즉시 반환 → 반복 작업이 크게 빨라진다.

키 = sha1(model_name | weights | front | middle | rear)
  → 모델이나 가중치가 바뀌면 자동으로 캐시 미스(안전).

첫 실행에는 효과 없음(전부 미스). 두 번째부터 효과.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingCache:
    def __init__(self, path: Path | str | None, model_name: str,
                 weights: tuple[float, float, float], enabled: bool = True) -> None:
        self.enabled = enabled and path is not None
        self.path = Path(path) if path is not None else None
        self.model_name = model_name
        self.weights = weights
        self._cache: dict[str, np.ndarray] = {}
        self.hits = 0
        self.misses = 0
        self._dirty = False
        if self.enabled and self.path is not None and self.path.exists():
            try:
                with self.path.open("rb") as f:
                    self._cache = pickle.load(f)
                logger.info("임베딩 캐시 로드: %d건 (%s)", len(self._cache), self.path)
            except Exception as e:  # noqa: BLE001
                logger.warning("임베딩 캐시 로드 실패 → 새로 시작: %s", e)
                self._cache = {}

    def key(self, front: str, middle: str, rear: str) -> str:
        h = hashlib.sha1()
        tag = f"{self.model_name}|{self.weights}|"
        h.update(tag.encode("utf-8"))
        h.update(front.encode("utf-8", "replace"))
        h.update(b"\x00")
        h.update(middle.encode("utf-8", "replace"))
        h.update(b"\x00")
        h.update(rear.encode("utf-8", "replace"))
        return h.hexdigest()

    def get(self, key: str) -> np.ndarray | None:
        if not self.enabled:
            return None
        v = self._cache.get(key)
        if v is None:
            self.misses += 1
            return None
        self.hits += 1
        return v

    def put(self, key: str, vector: np.ndarray) -> None:
        if not self.enabled:
            return
        self._cache[key] = vector.astype(np.float32)
        self._dirty = True

    def save(self) -> None:
        if not self.enabled or self.path is None or not self._dirty:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("wb") as f:
                pickle.dump(self._cache, f)
            logger.info("임베딩 캐시 저장: %d건 → %s", len(self._cache), self.path)
        except Exception as e:  # noqa: BLE001
            logger.warning("임베딩 캐시 저장 실패: %s", e)

    def stats(self) -> str:
        total = self.hits + self.misses
        rate = (self.hits / total * 100) if total else 0.0
        return f"캐시 적중 {self.hits}/{total} ({rate:.0f}%)"
