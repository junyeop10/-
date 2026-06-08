"""
similarity.py
-------------
cosine similarity 기반 카테고리 분류 모듈 (Stage 3, 2단계).

역할
----
사용자 확정 이력(피드백 DB)의 임베딩 벡터와
신규 파일의 임베딩 벡터를 비교해 가장 가까운 카테고리를 결정한다.

Stage 3 전체 흐름 중 이 모듈의 위치
--------------------------------------
1단계 — 룰 기반   : conf >= 0.80 → 확정  (rule_classifier.py)
2단계 — cosine sim : conf >= 0.75 → 확정  ← 이 모듈 (similarity.py)
3단계 — LLM        : conf >= 0.60 → 확정  (llm_classifier.py)
미달 → 검토 큐

왜 KNN이 아닌 cosine similarity인가
-------------------------------------
KNN은 별도 학습(fit) 단계가 필요하고, 피드백이 누적될 때마다
재학습해야 한다. cosine similarity는 정규화된 벡터끼리의 내적이므로
이력 벡터를 그냥 행렬로 쌓아두고 dot product만 계산하면 된다.
→ 피드백 1건이 추가될 때마다 행에 append하면 즉시 반영된다.

카테고리 목록은 categories.json에서 동적 로드 (폴더 수는 가변).

핵심 계산
---------
    # 모든 이력 벡터와 한 번에 비교 (행렬 내적)
    scores = corpus_matrix @ query_vector   # (N,)
    # 같은 카테고리 안에서 가장 높은 점수를 카테고리 점수로 사용
    category_score = max(scores[카테고리 == C] for C in categories)

사용 예시
---------
    store = EmbeddingStore()

    # 피드백(확정 이력) 추가
    store.add("공고_양식",    vec_1)
    store.add("공고_양식",    vec_2)
    store.add("사업계획서",   vec_3)

    # 신규 파일 분류
    sim = SimilarityClassifier(store, threshold=0.75)
    result = sim.classify(query_vector)

    if result.is_confident:
        print(f"분류 확정: {result.category} (conf={result.confidence:.3f})")
    else:
        print("신뢰도 미달 → LLM 단계로 위임")
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.75   # 이 값 이상이면 분류 확정, 미달이면 LLM으로 위임
EMBED_DIM            = 384    # embedder.py와 일치해야 한다

DEFAULT_CATEGORIES_PATH = Path(__file__).with_name("categories.json")


# ──────────────────────────────────────────────
# 카테고리 정의 (JSON에서 동적 로드 — 폴더 수 가변)
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class Category:
    folder: str
    # cosine similarity cold-start 시드용 자연어 설명.
    # 운영 단계에서는 사용자 확정 이력이 쌓이면 이 설명 기반 시드보다
    # 실제 파일 벡터가 우선시된다.
    description: str = ""


def load_categories(path: Path | str | None = None) -> list[Category]:
    """JSON 파일에서 카테고리 목록 로드 (기본: categories.json)."""
    json_path = Path(path) if path is not None else DEFAULT_CATEGORIES_PATH
    with json_path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"카테고리 JSON은 비어있지 않은 배열이어야 함: {json_path}")
    return [
        Category(folder=str(item["folder"]), description=str(item.get("description", "")))
        for item in raw
    ]


def category_folders(categories: list[Category]) -> list[str]:
    """카테고리 폴더명 리스트."""
    return [c.folder for c in categories]


# ──────────────────────────────────────────────
# 결과 컨테이너
# ──────────────────────────────────────────────

@dataclass
class ClassifyResult:
    """SimilarityClassifier.classify 한 번 호출의 결과."""
    category: Optional[str]    # 확정 카테고리. 신뢰도 미달이면 None.
    confidence: float          # 0.0~1.0 (정규화된 벡터 간 cosine similarity)
    threshold: float           # 이 분류에 적용된 임계값
    top_k: list[tuple[str, float]]   # [(카테고리, 최고 점수), ...] 상위 3개

    @property
    def is_confident(self) -> bool:
        """임계값 이상이면 True → 분류 확정. False → LLM 위임."""
        return self.confidence >= self.threshold

    def summary(self) -> str:
        status = "확정" if self.is_confident else "LLM 위임"
        lines = [f"[{status}] category={self.category!r}  conf={self.confidence:.3f}"]
        for cat, score in self.top_k:
            marker = "*" if cat == self.category else " "
            lines.append(f"  {marker} {cat}: {score:.4f}")
        return "\n".join(lines)


# ──────────────────────────────────────────────
# EmbeddingStore: 확정 이력 벡터를 관리하는 인메모리 저장소
# ──────────────────────────────────────────────

class EmbeddingStore:
    """
    카테고리별 확정 이력 임베딩을 메모리에 저장한다.

    구조
    ----
    _labels  : list[str]         — 각 벡터의 카테고리
    _matrix  : np.ndarray        — shape (N, 384), N = 누적 이력 수

    add() 한 번에 벡터 1개씩 append하는 방식이므로
    Stage 6 피드백이 들어올 때마다 즉시 반영된다.
    (별도 재학습(re-fit) 불필요)
    """

    def __init__(self) -> None:
        self._labels: list[str]          = []
        self._vecs: list[np.ndarray]     = []   # 최종 행렬 구성 전 버퍼
        self._matrix: Optional[np.ndarray] = None   # cache: (N, 384)

    def add(self, category: str, vector: np.ndarray) -> None:
        """
        확정 이력 1건을 추가한다.

        Parameters
        ----------
        category : str
            사용자가 확정한 카테고리명.
        vector : np.ndarray, shape (384,)
            L2 정규화된 임베딩 벡터 (embedder.py 출력).
        """
        if vector.shape != (EMBED_DIM,):
            raise ValueError(f"벡터 차원 불일치: expected ({EMBED_DIM},), got {vector.shape}")

        self._labels.append(category)
        self._vecs.append(vector.astype(np.float32))
        self._matrix = None   # 캐시 무효화
        logger.debug("이력 추가: category='%s', 누적 %d건", category, len(self._labels))

    def add_batch(self, categories: list[str], vectors: np.ndarray) -> None:
        """
        확정 이력 N건을 한 번에 추가한다.

        Parameters
        ----------
        categories : list[str], length N
        vectors    : np.ndarray, shape (N, 384)
        """
        if len(categories) != vectors.shape[0]:
            raise ValueError("categories와 vectors의 길이가 다릅니다.")
        for cat, vec in zip(categories, vectors):
            self.add(cat, vec)

    @property
    def matrix(self) -> np.ndarray:
        """전체 이력 벡터 행렬. shape (N, 384)."""
        if self._matrix is None:
            if not self._vecs:
                return np.empty((0, EMBED_DIM), dtype=np.float32)
            self._matrix = np.stack(self._vecs, axis=0)   # (N, 384)
        return self._matrix

    @property
    def labels(self) -> list[str]:
        return self._labels

    @property
    def categories(self) -> list[str]:
        """중복 없는 카테고리 목록."""
        return list(dict.fromkeys(self._labels))

    def __len__(self) -> int:
        return len(self._labels)

    def is_empty(self) -> bool:
        return len(self._labels) == 0


# ──────────────────────────────────────────────
# SimilarityClassifier
# ──────────────────────────────────────────────

class SimilarityClassifier:
    """
    EmbeddingStore의 이력 벡터와 cosine similarity를 비교해
    신규 파일의 카테고리를 결정한다.

    분류 로직
    ---------
    1. query_vector와 모든 이력 벡터의 cosine similarity 계산
       scores = corpus_matrix @ query_vector    # (N,)
    2. 카테고리별로 scores의 최댓값을 카테고리 점수로 사용
       (가장 비슷한 이력 1건이 기준 — 다수결이 아니라 최근접 이웃)
    3. 가장 높은 카테고리 점수 >= threshold → 분류 확정
    4. 미달 → category=None 반환 → 호출자가 LLM 단계로 위임
    """

    def __init__(
        self,
        store: EmbeddingStore,
        threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self.store     = store
        self.threshold = threshold

    def classify(self, query: np.ndarray, top_k: int = 3) -> ClassifyResult:
        """
        신규 파일 임베딩 벡터의 카테고리를 cosine similarity로 결정한다.

        Parameters
        ----------
        query : np.ndarray, shape (384,)
            L2 정규화된 임베딩 벡터.
        top_k : int
            결과에 포함할 상위 카테고리 수 (기본 3).

        Returns
        -------
        ClassifyResult
            is_confident=True이면 category가 확정 카테고리.
            False이면 category=None → LLM 단계 위임.
        """
        if query.shape != (EMBED_DIM,):
            raise ValueError(f"벡터 차원 불일치: expected ({EMBED_DIM},), got {query.shape}")

        if self.store.is_empty():
            logger.warning("이력 DB가 비어 있습니다. 분류 불가 → LLM 위임.")
            return ClassifyResult(
                category=None,
                confidence=0.0,
                threshold=self.threshold,
                top_k=[],
            )

        # 1. 모든 이력 벡터와 cosine similarity 계산
        #    정규화 벡터이므로 dot product = cosine similarity
        corpus = self.store.matrix          # (N, 384)
        scores = corpus @ query             # (N,) = cosine similarity 배열

        # 2. 카테고리별 최대 점수 집계
        category_best: dict[str, float] = defaultdict(float)
        for lbl, score in zip(self.store.labels, scores.tolist()):
            if score > category_best[lbl]:
                category_best[lbl] = score

        # 3. 점수 내림차순 정렬
        ranked = sorted(category_best.items(), key=lambda x: x[1], reverse=True)

        best_cat, best_score = ranked[0]
        top_k_list = ranked[:top_k]

        # 4. 임계값 비교
        if best_score >= self.threshold:
            category = best_cat
            logger.info(
                "cosine sim 분류 확정: '%s'  conf=%.3f  (임계값: %.2f)",
                category, best_score, self.threshold,
            )
        else:
            category = None
            logger.info(
                "cosine sim 신뢰도 미달: best='%s' conf=%.3f < %.2f → LLM 위임",
                best_cat, best_score, self.threshold,
            )

        return ClassifyResult(
            category=category,
            confidence=best_score,
            threshold=self.threshold,
            top_k=top_k_list,
        )

    # ── 유틸리티 ──────────────────────────────

    def top_n_similar(
        self,
        query: np.ndarray,
        n: int = 5,
    ) -> list[tuple[int, str, float]]:
        """
        개별 이력 벡터 기준 상위 N개 유사 항목을 반환한다.
        카테고리 집계 없이 벡터 1건씩의 점수를 반환한다.
        (디버깅 / 분류 근거 표시용)

        Returns
        -------
        list of (이력 인덱스, 카테고리, cosine similarity)
        """
        if self.store.is_empty():
            return []

        scores = self.store.matrix @ query   # (N,)
        top_indices = np.argsort(scores)[::-1][:n]
        return [
            (int(i), self.store.labels[i], float(scores[i]))
            for i in top_indices
        ]
