"""
reducer.py
----------
UMAP 기반 차원 축소 모듈.

역할
----
embedder.py가 생성한 고차원 벡터(384차원)를 HDBSCAN이 안정적으로
밀도를 추정할 수 있는 저차원 공간으로 압축한다.

왜 차원 축소가 필요한가
-----------------------
HDBSCAN은 '밀도(density)' 기반 알고리즘이다.
고차원 공간에서는 차원의 저주(curse of dimensionality)로 인해
모든 점 간 거리가 비슷해지고 밀도 차이가 사라진다.
→ UMAP으로 저차원으로 압축해야 HDBSCAN이 유의미한 군집을 찾는다.

UMAP 주요 파라미터
------------------
- n_components  : 목표 차원 (기본 15). 샘플 수에 따라 자동 조정됨.
                  안전 상한: min(50, N//2, n_neighbors-1)
- n_neighbors   : 지역 이웃 수. 클수록 전역 구조 보존, 작을수록 지역 구조.
                  샘플 수 N에 따라 min(15, N-1)로 자동 조정.
- min_dist      : 0.0 → 군집이 촘촘하게 뭉침 (HDBSCAN 입력에 최적).
- metric        : 'cosine' → 임베딩 벡터 간 방향 유사도 기준.

사용 예시
---------
    reducer = DimReducer()

    # 학습 + 변환 (처음 파일 배치)
    result = reducer.fit_transform(vectors)     # (N, 384) → (N, k)

    # 신규 파일 벡터 변환 (모델 재학습 없이)
    result = reducer.transform(new_vector)      # (384,) → (k,)

    # 모델 저장 / 로드
    reducer.save("umap_model.pkl")
    reducer2 = DimReducer.load("umap_model.pkl")
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import umap

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 기본 하이퍼파라미터
# ──────────────────────────────────────────────

N_COMPONENTS_MAX = 15    # 목표 차원 (의미 보존 + HDBSCAN 안정성의 sweet spot)
N_NEIGHBORS      = 15    # 지역 구조 탐색 반경 (샘플 수 적으면 자동 축소)
MIN_DIST         = 0.0   # 0.0 → 군집 내부를 촘촘하게 → HDBSCAN에 유리
METRIC           = "cosine"
RANDOM_STATE     = 42    # 재현성 보장


# ──────────────────────────────────────────────
# 결과 컨테이너
# ──────────────────────────────────────────────

@dataclass
class ReduceResult:
    """fit_transform / transform 한 번 호출의 결과."""
    vectors: np.ndarray   # shape (N, n_components) 또는 (n_components,)
    n_input_dim: int      # 입력 차원 (384)
    n_output_dim: int     # 출력 차원 (실제 사용된 n_components)
    n_samples: int        # 처리된 벡터 수

    @property
    def shape(self) -> tuple[int, ...]:
        return self.vectors.shape


# ──────────────────────────────────────────────
# DimReducer
# ──────────────────────────────────────────────

class DimReducer:
    """
    UMAP 래퍼(wrapper). 학습(fit)과 변환(transform)을 분리해 관리한다.

    파이프라인 흐름
    ---------------
    Stage 1 → embedder 벡터 수집
    Stage 2 → DimReducer.fit_transform(all_vectors)   ← 이 모듈
           → Clusterer.fit_predict(reduced_vectors)   ← clusterer.py
    신규 파일 → DimReducer.transform(new_vector) → Clusterer.predict(...)
    """

    def __init__(
        self,
        n_components_max: int = N_COMPONENTS_MAX,
        n_neighbors: int = N_NEIGHBORS,
        min_dist: float = MIN_DIST,
        metric: str = METRIC,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.n_components_max = n_components_max
        self.n_neighbors      = n_neighbors
        self.min_dist         = min_dist
        self.metric           = metric
        self.random_state     = random_state
        self._is_fitted       = False
        self._model: Optional[umap.UMAP] = None
        self._actual_n_components: int = 0

    # ── 내부 헬퍼 ──────────────────────────────

    def _safe_params(self, n_samples: int) -> tuple[int, int]:
        """
        샘플 수에 따라 안전한 (n_neighbors, n_components)를 계산한다.

        UMAP spectral 초기화 제약
        -------------------------
        n_components < n_neighbors 이어야 spectral 초기화가 안정적이다.
        또한 n_neighbors <= n_samples - 1 이어야 한다.

        따라서:
            n_nbrs  = min(self.n_neighbors, n_samples - 1)
            n_comp  = min(n_components_max, n_samples // 2, n_nbrs - 1)
            n_comp  = max(n_comp, 2)          # 최소 2차원 보장
        """
        n_nbrs = min(self.n_neighbors, n_samples - 1)
        n_comp = min(self.n_components_max, n_samples // 2, n_nbrs - 1)
        n_comp = max(n_comp, 2)

        if n_nbrs != self.n_neighbors:
            logger.warning("n_neighbors %d → %d (샘플 수: %d)", self.n_neighbors, n_nbrs, n_samples)
        if n_comp != self.n_components_max:
            logger.warning("n_components %d → %d (샘플 수: %d)", self.n_components_max, n_comp, n_samples)

        return n_nbrs, n_comp

    def _build_model(self, n_samples: int) -> umap.UMAP:
        n_nbrs, n_comp = self._safe_params(n_samples)
        self._actual_n_components = n_comp
        return umap.UMAP(
            n_components=n_comp,
            n_neighbors=n_nbrs,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
            low_memory=False,
        )

    # ── 공개 API ──────────────────────────────

    def fit_transform(self, vectors: np.ndarray) -> ReduceResult:
        """
        벡터 행렬을 학습하면서 동시에 변환한다.
        처음 파일 배치를 처리할 때 호출한다.

        Parameters
        ----------
        vectors : np.ndarray, shape (N, 384)
            L2 정규화된 임베딩 벡터 행렬. N >= 2 이어야 한다.

        Returns
        -------
        ReduceResult
            reduced.vectors shape: (N, n_components)
        """
        n, d = vectors.shape
        if n < 2:
            raise ValueError(f"UMAP은 최소 2개 샘플이 필요합니다. (현재: {n}개)")

        logger.info("UMAP fit_transform 시작: (%d, %d)", n, d)

        self._model = self._build_model(n)
        reduced = self._model.fit_transform(vectors).astype(np.float32)
        self._is_fitted = True

        actual_dim = reduced.shape[1]
        logger.info("UMAP 완료: (%d, %d) → (%d, %d)", n, d, n, actual_dim)

        return ReduceResult(
            vectors=reduced,
            n_input_dim=d,
            n_output_dim=actual_dim,
            n_samples=n,
        )

    def transform(self, vectors: np.ndarray) -> ReduceResult:
        """
        이미 학습된 UMAP 모델로 신규 벡터를 변환한다.
        모델을 재학습하지 않으므로 빠르다.

        Parameters
        ----------
        vectors : np.ndarray
            shape (384,) 또는 (N, 384).
            단일 벡터(1D)이면 자동으로 2D로 처리 후 원래 shape으로 반환.

        Returns
        -------
        ReduceResult
            shape (n_components,) 또는 (N, n_components)
        """
        if not self._is_fitted:
            raise RuntimeError("fit_transform을 먼저 호출해야 합니다.")

        squeeze = vectors.ndim == 1
        if squeeze:
            vectors = vectors[np.newaxis, :]   # (1, 384)

        reduced = self._model.transform(vectors).astype(np.float32)
        if squeeze:
            reduced = reduced[0]   # (n_components,)

        n = 1 if squeeze else vectors.shape[0]
        d = vectors.shape[-1]
        logger.debug("UMAP transform: %d 벡터 처리 완료", n)

        return ReduceResult(
            vectors=reduced,
            n_input_dim=d,
            n_output_dim=self._actual_n_components,
            n_samples=n,
        )

    # ── 직렬화 ────────────────────────────────

    def save(self, path: str | Path) -> None:
        """학습된 UMAP 모델을 pickle로 저장한다."""
        if not self._is_fitted:
            raise RuntimeError("저장할 모델이 없습니다. fit_transform을 먼저 호출하세요.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "actual_n_components": self._actual_n_components,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("UMAP 모델 저장 완료: %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "DimReducer":
        """저장된 UMAP 모델을 로드해 DimReducer 인스턴스를 반환한다."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
        with open(path, "rb") as f:
            payload = pickle.load(f)
        model: umap.UMAP = payload["model"]

        instance = cls(
            n_components_max=payload["actual_n_components"],
            n_neighbors=model.n_neighbors,
            min_dist=model.min_dist,
            metric=model.metric,
        )
        instance._model = model
        instance._is_fitted = True
        instance._actual_n_components = payload["actual_n_components"]
        logger.info("UMAP 모델 로드 완료: %s", path)
        return instance

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def actual_n_components(self) -> int:
        """실제로 사용된 출력 차원 (샘플 수에 따라 조정된 값)."""
        return self._actual_n_components

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"DimReducer(n_components_max={self.n_components_max}, "
            f"n_neighbors={self.n_neighbors}, "
            f"metric='{self.metric}', {status})"
        )
