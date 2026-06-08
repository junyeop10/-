"""
semantic_core.py
----------------
의미기반 분류의 *코어*. 설계 의도: "분류의 핵심은 의미 기반".

단일 cosine-max(similarity.py)만으로는 약했다. 그래서 서로 다른 의미 신호
4개를 late-fusion(후기 융합)으로 합쳐 robust하게 만든다. 넷 다 데이터에서
나오는 신호 — 무거운 모델/하드코딩 룰이 아니다.

  S1. cos_max  : 카테고리 내 가장 가까운 이력 1건과의 cosine (최근접 이웃)
  S2. centroid : 카테고리 평균 벡터와의 cosine (카테고리 무게중심 → 자석 완화)
  S3. knn_vote : 전체 이력 중 top-K 이웃의 cosine 가중 다수결 (지역 밀도)
  S4. bm25     : 시드/피드백 본문에 대한 BM25 어휘 매칭 (정확 단어)

융합
----
    final[C] = 0.40·cos_max + 0.30·centroid + 0.15·knn_vote + 0.15·bm25

네 신호 모두 [0,1]로 맞춰 가중 평균. best=argmax(final), confidence=final[best]
를 threshold와 비교. cos_max는 이력이 많을수록 과대평가되기 쉬워 centroid와
분산하고, knn/bm25가 "여러 신호가 같은 카테고리를 가리키는가"를 보강한다.

속도
----
무거운 모델 0개. 이력 corpus가 수십~수백 개라 cos_max를 위한 행렬곱은
이미 하고 있고, centroid(카테고리수 내적)·knn(argsort)·bm25(토큰 산수)는
모두 그 위에서 무시할 비용. (수백만 규모가 되면 ANN 인덱스로 대체)

SimilarityClassifier 상위 호환: SemanticResult는 ClassifyResult와 같은 속성
(category/confidence/threshold/top_k/is_confident)을 노출해 evidence 빌더를
그대로 쓴다.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from lexical import BM25Index, normalize_scores

logger = logging.getLogger(__name__)

EMBED_DIM = 384

# 융합 가중치 (합 = 1.0)
# data1 ablation 결과 — BM25가 MVP(정확 단어), cos_max가 backbone,
# centroid/kNN은 이질적 카테고리에서 자석이 되기 쉬워 소량만.
# (cos_max 0.45 / centroid 0.15 / kNN 0.10 / BM25 0.30)
W_COS_MAX = 0.45
W_CENTROID = 0.15
W_KNN = 0.10
W_BM25 = 0.30

DEFAULT_THRESHOLD = 0.50   # 융합 점수 기준 (cosine-max 단독 0.6과 다른 스케일)
DEFAULT_KNN_K = 7


@dataclass
class SemanticResult:
    category: Optional[str]
    confidence: float                     # final[best] — threshold와 비교
    threshold: float
    top_k: list[tuple[str, float]]        # (카테고리, final 점수) 내림차순
    signals: dict[str, float] = field(default_factory=dict)  # best 카테고리 신호 분해

    @property
    def is_confident(self) -> bool:
        return self.category is not None and self.confidence >= self.threshold

    def breakdown(self) -> str:
        s = self.signals
        return (f"cos_max={s.get('cos_max', 0):.3f} cent={s.get('centroid', 0):.3f} "
                f"knn={s.get('knn', 0):.3f} bm25={s.get('bm25', 0):.3f}")


class SemanticStore:
    """확정/시드 이력을 (category, vector, text)로 저장.

    similarity.EmbeddingStore의 확장 — 벡터 외에 *본문 텍스트*도 보관해 BM25
    어휘 신호를 함께 계산. centroid/BM25는 lazy 캐시, 온라인 add는 증분.
    """

    def __init__(self) -> None:
        self._labels: list[str] = []
        self._vecs: list[np.ndarray] = []
        self._texts: list[str] = []
        self._matrix: Optional[np.ndarray] = None
        self._centroids: Optional[dict[str, np.ndarray]] = None
        self._bm25: Optional[BM25Index] = None

    def add(self, category: str, vector: np.ndarray, text: str = "") -> None:
        if vector.shape != (EMBED_DIM,):
            raise ValueError(f"벡터 차원 불일치: expected ({EMBED_DIM},), got {vector.shape}")
        self._labels.append(category)
        self._vecs.append(vector.astype(np.float32))
        self._texts.append(text or "")
        self._matrix = None
        self._centroids = None
        if self._bm25 is not None:       # 이미 빌드돼 있으면 증분 (전체 재빌드 회피)
            self._bm25.add(text or "")

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            self._matrix = (np.stack(self._vecs, axis=0) if self._vecs
                            else np.empty((0, EMBED_DIM), dtype=np.float32))
        return self._matrix

    @property
    def labels(self) -> list[str]:
        return self._labels

    @property
    def categories(self) -> list[str]:
        return list(dict.fromkeys(self._labels))

    def is_empty(self) -> bool:
        return not self._labels

    def __len__(self) -> int:
        return len(self._labels)

    def centroids(self) -> dict[str, np.ndarray]:
        if self._centroids is None:
            acc: dict[str, list[np.ndarray]] = defaultdict(list)
            for lbl, v in zip(self._labels, self._vecs):
                acc[lbl].append(v)
            cents: dict[str, np.ndarray] = {}
            for lbl, vs in acc.items():
                m = np.mean(np.stack(vs, axis=0), axis=0)
                n = np.linalg.norm(m)
                cents[lbl] = (m / n).astype(np.float32) if n > 1e-8 else m.astype(np.float32)
            self._centroids = cents
        return self._centroids

    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            idx = BM25Index()
            idx.add_batch(self._texts)
            self._bm25 = idx
        return self._bm25


class SemanticClassifier:
    """4신호(cos_max·centroid·knn·bm25) 가중 융합 분류기."""

    def __init__(
        self,
        store: SemanticStore,
        threshold: float = DEFAULT_THRESHOLD,
        knn_k: int = DEFAULT_KNN_K,
        weights: tuple[float, float, float, float] = (W_COS_MAX, W_CENTROID, W_KNN, W_BM25),
    ) -> None:
        self.store = store
        self.threshold = threshold
        self.knn_k = knn_k
        self.weights = weights

    def classify(self, query_vector: np.ndarray, query_text: str = "",
                 top_k: int = 3) -> SemanticResult:
        if query_vector.shape != (EMBED_DIM,):
            raise ValueError(f"벡터 차원 불일치: expected ({EMBED_DIM},), got {query_vector.shape}")
        if self.store.is_empty():
            return SemanticResult(None, 0.0, self.threshold, [], {})

        labels = self.store.labels
        matrix = self.store.matrix
        cats = self.store.categories
        sims = matrix @ query_vector            # (N,) — cos_max를 위해 이미 필요한 계산

        # S1. cos_max
        cos_max: dict[str, float] = defaultdict(lambda: -1.0)
        for lbl, s in zip(labels, sims.tolist()):
            if s > cos_max[lbl]:
                cos_max[lbl] = s

        # S2. centroid cosine
        cents = self.store.centroids()
        centroid: dict[str, float] = {c: float(query_vector @ cents[c]) for c in cats}

        # S3. knn 가중 다수결 (이미 구한 sims 재활용 — 추가 비용 거의 0)
        K = min(self.knn_k, len(labels))
        top_idx = np.argsort(sims)[::-1][:K]
        clipped = {int(i): max(0.0, float(sims[i])) for i in top_idx}
        denom = sum(clipped.values())
        knn: dict[str, float] = defaultdict(float)
        if denom > 1e-9:
            for i, w in clipped.items():
                knn[labels[i]] += w / denom

        # S4. bm25 (본문 어휘)
        bm25: dict[str, float] = defaultdict(float)
        if query_text.strip():
            norm = normalize_scores(self.store.bm25().scores(query_text))
            for lbl, s in zip(labels, norm):
                if s > bm25[lbl]:
                    bm25[lbl] = s

        # 가중 융합
        w_cm, w_ce, w_kn, w_bm = self.weights
        final: dict[str, float] = {}
        for c in cats:
            final[c] = (w_cm * max(0.0, cos_max.get(c, 0.0))
                        + w_ce * max(0.0, centroid.get(c, 0.0))
                        + w_kn * knn.get(c, 0.0)
                        + w_bm * bm25.get(c, 0.0))

        ranked = sorted(final.items(), key=lambda x: x[1], reverse=True)
        best_cat, best_score = ranked[0]
        signals = {
            "cos_max": float(cos_max.get(best_cat, 0.0)),
            "centroid": float(centroid.get(best_cat, 0.0)),
            "knn": float(knn.get(best_cat, 0.0)),
            "bm25": float(bm25.get(best_cat, 0.0)),
        }
        category = best_cat if best_score >= self.threshold else None
        if category is not None:
            logger.info("의미 융합 확정: '%s' final=%.3f (%s)", category, best_score,
                        " ".join(f"{k}={v:.2f}" for k, v in signals.items()))
        return SemanticResult(category, best_score, self.threshold, ranked[:top_k], signals)

    def top_n_similar(self, query: np.ndarray, n: int = 5) -> list[tuple[int, str, float]]:
        """근거 표시용 — SimilarityClassifier와 동일 인터페이스."""
        if self.store.is_empty():
            return []
        scores = self.store.matrix @ query
        top = np.argsort(scores)[::-1][:n]
        return [(int(i), self.store.labels[i], float(scores[i])) for i in top]
