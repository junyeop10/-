"""
clusterer.py
------------
HDBSCAN 기반 프로젝트 군집화 모듈.

역할
----
reducer.py가 생성한 저차원 벡터(N, k)를 입력받아
각 파일이 어느 프로젝트(군집)에 속하는지 자동으로 결정한다.

왜 HDBSCAN인가
--------------
- KMeans와 달리 군집 수를 사전에 지정할 필요가 없다.
  → 파일을 올리기 전에 프로젝트가 몇 개인지 모르는 상황에 적합.
- 밀도가 낮은 파일(noise)을 -1로 표시해 강제 배정하지 않는다.
  → 어느 프로젝트에도 속하지 않는 파일을 검토 큐로 보낼 수 있다.
- 비구형(non-spherical) 군집도 탐지한다.

HDBSCAN 주요 파라미터
---------------------
- min_cluster_size   : 군집으로 인정할 최소 파일 수.
                       작게 할수록 군집이 많아지고, 크게 할수록 줄어든다.
                       기본값 3 (소규모 파일 배치에서 유연하게 작동).
- min_samples        : None → min_cluster_size와 동일하게 설정됨.
                       올릴수록 noise 포인트가 늘고 군집이 줄어든다.
- metric             : 'euclidean'. UMAP 출력은 더 이상 cosine 공간이
                       아니므로 euclidean을 사용한다.
- cluster_selection_method : 'eom' (Excess of Mass, 기본).
                              작은 군집까지 잘 찾는다.
- prediction_data    : True → approximate_predict로 신규 파일 처리 가능.

사용 예시
---------
    clusterer = Clusterer()

    # 학습 + 예측 (처음 파일 배치)
    result = clusterer.fit_predict(reduced_vectors)

    # 군집별 파일 인덱스 확인
    for cid, idxs in result.cluster_to_indices.items():
        print(f"군집 {cid}: 파일 {idxs}")

    # noise 파일 (검토 큐로 보낼 대상)
    print("noise 파일 인덱스:", result.noise_indices)

    # 신규 파일 1건 예측 (모델 재학습 없이)
    pred = clusterer.predict(new_reduced_vector)

    # 모델 저장 / 로드
    clusterer.save("hdbscan_model.pkl")
    clusterer2 = Clusterer.load("hdbscan_model.pkl")
"""

from __future__ import annotations

import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import hdbscan
import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 기본 하이퍼파라미터
# ──────────────────────────────────────────────

MIN_CLUSTER_SIZE         = 3          # 군집 최소 파일 수
MIN_SAMPLES              = None       # None → min_cluster_size와 동일
METRIC                   = "euclidean"
CLUSTER_SELECTION_METHOD = "eom"      # Excess of Mass
NOISE_LABEL              = -1         # HDBSCAN noise 포인트 레이블


# ──────────────────────────────────────────────
# 결과 컨테이너
# ──────────────────────────────────────────────

@dataclass
class ClusterResult:
    """fit_predict 한 번 호출의 결과."""
    labels: np.ndarray                              # shape (N,), -1 = noise
    probabilities: np.ndarray                       # shape (N,), 0.0~1.0
    n_clusters: int                                 # noise 제외 군집 수
    n_noise: int                                    # noise 포인트 수
    cluster_to_indices: dict[int, list[int]]        # {군집 ID: [파일 인덱스, ...]}
    noise_indices: list[int]                        # noise 파일 인덱스 목록

    def summary(self) -> str:
        lines = [
            f"군집 수: {self.n_clusters}  |  noise: {self.n_noise}개",
        ]
        for cid, idxs in sorted(self.cluster_to_indices.items()):
            probs = self.probabilities[idxs]
            lines.append(
                f"  군집 {cid:2d}: {len(idxs):3d}개 파일  "
                f"(확신도 평균: {probs.mean():.3f}, 최소: {probs.min():.3f})"
            )
        if self.noise_indices:
            lines.append(f"  noise  : {self.noise_indices}")
        return "\n".join(lines)


@dataclass
class PredictResult:
    """predict(신규 파일) 한 번 호출의 결과."""
    label: int        # 군집 ID, -1이면 어느 군집에도 속하지 않음 (noise)
    probability: float

    @property
    def is_noise(self) -> bool:
        return self.label == NOISE_LABEL


# ──────────────────────────────────────────────
# Clusterer
# ──────────────────────────────────────────────

class Clusterer:
    """
    HDBSCAN 래퍼(wrapper).

    입력은 반드시 DimReducer.fit_transform/transform의 출력이어야 한다.
    (고차원 임베딩 벡터를 직접 넣으면 차원의 저주로 군집이 잘 나오지 않는다.)
    """

    def __init__(
        self,
        min_cluster_size: int = MIN_CLUSTER_SIZE,
        min_samples: Optional[int] = MIN_SAMPLES,
        metric: str = METRIC,
        cluster_selection_method: str = CLUSTER_SELECTION_METHOD,
    ) -> None:
        self.min_cluster_size         = min_cluster_size
        self.min_samples              = min_samples
        self.metric                   = metric
        self.cluster_selection_method = cluster_selection_method
        self._is_fitted               = False
        self._model: Optional[hdbscan.HDBSCAN] = None

    # ── 내부 헬퍼 ──────────────────────────────

    def _safe_min_cluster_size(self, n_samples: int) -> int:
        """
        min_cluster_size는 n_samples 이하여야 한다.
        파일 수가 매우 적을 때 자동으로 줄여 오류를 방지한다.
        """
        safe = min(self.min_cluster_size, max(n_samples - 1, 2))
        if safe != self.min_cluster_size:
            logger.warning(
                "min_cluster_size %d → %d (샘플 수: %d)",
                self.min_cluster_size, safe, n_samples,
            )
        return safe

    def _build_model(self, n_samples: int) -> hdbscan.HDBSCAN:
        mcs = self._safe_min_cluster_size(n_samples)
        return hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method,
            prediction_data=True,   # approximate_predict 활성화
        )

    @staticmethod
    def _build_cluster_map(
        labels: np.ndarray,
    ) -> tuple[dict[int, list[int]], list[int]]:
        """labels 배열로부터 군집별 인덱스 맵과 noise 목록을 생성한다."""
        cluster_map: dict[int, list[int]] = defaultdict(list)
        noise_indices: list[int] = []
        for idx, lbl in enumerate(labels.tolist()):
            if lbl == NOISE_LABEL:
                noise_indices.append(idx)
            else:
                cluster_map[lbl].append(idx)
        return dict(cluster_map), noise_indices

    # ── 공개 API ──────────────────────────────

    def fit_predict(self, vectors: np.ndarray) -> ClusterResult:
        """
        저차원 벡터 행렬을 학습하면서 군집 레이블을 반환한다.
        처음 파일 배치를 처리할 때 호출한다.

        Parameters
        ----------
        vectors : np.ndarray, shape (N, k)
            DimReducer.fit_transform 출력. L2 정규화 불필요 (euclidean 사용).

        Returns
        -------
        ClusterResult
            labels shape: (N,). -1은 noise (어느 군집에도 미배정).
        """
        n = vectors.shape[0]
        if n < 2:
            raise ValueError(f"HDBSCAN은 최소 2개 샘플이 필요합니다. (현재: {n}개)")

        logger.info("HDBSCAN fit_predict 시작: %d개 파일", n)

        self._model = self._build_model(n)
        labels: np.ndarray = self._model.fit_predict(vectors)
        probs: np.ndarray  = self._model.probabilities_.astype(np.float32)
        self._is_fitted = True

        cluster_map, noise_idxs = self._build_cluster_map(labels)
        n_clusters = len(cluster_map)
        n_noise    = len(noise_idxs)

        logger.info(
            "HDBSCAN 완료: 군집 %d개, noise %d개 (전체 %d개)",
            n_clusters, n_noise, n,
        )

        return ClusterResult(
            labels=labels,
            probabilities=probs,
            n_clusters=n_clusters,
            n_noise=n_noise,
            cluster_to_indices=cluster_map,
            noise_indices=noise_idxs,
        )

    def predict(self, vector: np.ndarray) -> PredictResult:
        """
        이미 학습된 모델로 신규 벡터 1건의 군집을 예측한다.
        approximate_predict를 사용하므로 모델을 재학습하지 않는다.

        Parameters
        ----------
        vector : np.ndarray, shape (k,)
            DimReducer.transform 출력 (1건).

        Returns
        -------
        PredictResult
            label=-1이면 noise(검토 큐로 이동 권장).
        """
        if not self._is_fitted:
            raise RuntimeError("fit_predict를 먼저 호출해야 합니다.")

        vec_2d = vector[np.newaxis, :]   # (1, k)
        pred_labels, pred_probs = hdbscan.approximate_predict(self._model, vec_2d)

        label = int(pred_labels[0])
        prob  = float(pred_probs[0])

        logger.debug("HDBSCAN predict: label=%d, prob=%.3f", label, prob)
        return PredictResult(label=label, probability=prob)

    # ── 직렬화 ────────────────────────────────

    def save(self, path: str | Path) -> None:
        """학습된 HDBSCAN 모델을 pickle로 저장한다."""
        if not self._is_fitted:
            raise RuntimeError("저장할 모델이 없습니다. fit_predict를 먼저 호출하세요.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
        logger.info("HDBSCAN 모델 저장 완료: %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "Clusterer":
        """저장된 HDBSCAN 모델을 로드해 Clusterer 인스턴스를 반환한다."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
        with open(path, "rb") as f:
            model: hdbscan.HDBSCAN = pickle.load(f)

        instance = cls(
            min_cluster_size=model.min_cluster_size,
            min_samples=model.min_samples,
            metric=model.metric,
            cluster_selection_method=model.cluster_selection_method,
        )
        instance._model = model
        instance._is_fitted = True
        logger.info("HDBSCAN 모델 로드 완료: %s", path)
        return instance

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"Clusterer(min_cluster_size={self.min_cluster_size}, "
            f"metric='{self.metric}', {status})"
        )
