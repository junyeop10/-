"""
embedder.py
-----------
EvidencePackage 텍스트를 sentence-transformers로 임베딩하는 베이스 모듈.

모델  : paraphrase-multilingual-MiniLM-L12-v2  (384차원, 다국어/한국어 친화)
정규화: L2 normalize → 이후 cosine similarity = 내적(dot product)으로 계산 가능
장치  : CUDA 있으면 GPU, 없으면 CPU 자동 선택

사용 예시:
    embedder = Embedder()

    # 단일 텍스트
    vec = embedder.embed("사업계획서 제출 안내문입니다.")

    # 배치 (리스트)
    vecs = embedder.embed_batch(["문서 A 본문", "문서 B 본문", "문서 C 본문"])

    # EvidencePackage 텍스트 3구간 통합 임베딩
    combined = embedder.embed_evidence(front, middle, rear)

    # cosine similarity (정규화 벡터이므로 내적 = cosine)
    score = embedder.cosine_similarity(vec_a, vec_b)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────

# 한국어 포함 50여개 언어 지원 다국어 모델 (XLM-R 기반).
# 원본 코드의 all-MiniLM-L6-v2는 영어 전용이라 한국어 문서에서 의미 분리가 떨어짐.
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM  = 384                    # 모델 고정 출력 차원
MAX_SEQ_LEN = 128                   # 토큰 상한 (모델 기본값; 초과분 자동 truncation)

# 3구간 가중치 (front에 높은 가중치: 문서 앞부분에 핵심 정보 집중)
WEIGHT_FRONT  = 0.5
WEIGHT_MIDDLE = 0.25
WEIGHT_REAR   = 0.25


# ──────────────────────────────────────────────
# 데이터 클래스
# ──────────────────────────────────────────────

@dataclass
class EmbedResult:
    """embed 호출 한 건의 결과를 담는 컨테이너."""
    vector: np.ndarray          # shape (384,), dtype float32, L2 정규화 완료
    text_preview: str           # 원본 텍스트 앞 80자 (디버깅용)
    model_name: str = MODEL_NAME
    dim: int = EMBED_DIM

    def __post_init__(self) -> None:
        assert self.vector.shape == (EMBED_DIM,), (
            f"벡터 차원 불일치: expected ({EMBED_DIM},), got {self.vector.shape}"
        )

    @property
    def norm(self) -> float:
        """L2 노름. 정규화가 올바르면 ≈ 1.0."""
        return float(np.linalg.norm(self.vector))


# ──────────────────────────────────────────────
# Embedder 클래스
# ──────────────────────────────────────────────

class Embedder:
    """
    sentence-transformers 래퍼(wrapper).

    - 모델은 최초 호출 시 1회만 로드 (싱글턴 패턴 불필요, 인스턴스 재사용)
    - 모든 출력 벡터는 L2 정규화(normalize_embeddings=True) 적용
    - 빈 텍스트 입력 시 zero 벡터 반환 (파이프라인 중단 방지)
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,   # None → 자동 선택 (GPU > CPU)
        batch_size: int = 32,           # embed_batch의 내부 배치 크기
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size

        logger.info("모델 로드 중: %s", model_name)
        self._model = SentenceTransformer(model_name, device=device)
        self._device = str(self._model.device)
        logger.info("모델 로드 완료 | 장치(device): %s", self._device)

    # ── 내부 헬퍼 ──────────────────────────────

    def _encode(self, texts: list[str]) -> np.ndarray:
        """
        texts 리스트를 모델에 넣어 정규화된 벡터 행렬 반환.

        반환 shape: (len(texts), EMBED_DIM)
        """
        # 빈 문자열은 모델이 처리하지만, 의미 없는 벡터가 나올 수 있으므로
        # 빈 항목의 인덱스를 미리 파악해 zero 벡터로 대체
        empty_mask = [t.strip() == "" for t in texts]

        # 실제 인코딩
        vectors: np.ndarray = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,  # L2 정규화 → cosine sim = dot product
            show_progress_bar=False,
            convert_to_numpy=True,
        )   # shape: (N, 384)

        # 빈 텍스트 → zero 벡터로 강제 치환
        for idx, is_empty in enumerate(empty_mask):
            if is_empty:
                vectors[idx] = np.zeros(EMBED_DIM, dtype=np.float32)
                logger.warning("빈 텍스트 감지 (index=%d) → zero 벡터 할당", idx)

        return vectors.astype(np.float32)

    # ── 공개 API ──────────────────────────────

    def embed(self, text: str) -> EmbedResult:
        """
        텍스트 1건 임베딩.

        Parameters
        ----------
        text : str
            임베딩할 텍스트. 빈 문자열이면 zero 벡터 반환.

        Returns
        -------
        EmbedResult
            정규화된 384차원 벡터 + 메타.
        """
        vector = self._encode([text])[0]   # (384,)
        return EmbedResult(
            vector=vector,
            text_preview=text[:80],
        )

    def embed_batch(self, texts: list[str]) -> list[EmbedResult]:
        """
        텍스트 N건 배치 임베딩.

        Parameters
        ----------
        texts : list[str]
            임베딩할 텍스트 리스트.

        Returns
        -------
        list[EmbedResult]
            입력 순서와 동일한 순서의 결과 리스트.
        """
        if not texts:
            return []

        vectors = self._encode(texts)   # (N, 384)
        return [
            EmbedResult(vector=vectors[i], text_preview=t[:80])
            for i, t in enumerate(texts)
        ]

    def embed_evidence(
        self,
        text_front: str,
        text_middle: str,
        text_rear: str,
        weights: tuple[float, float, float] = (
            WEIGHT_FRONT, WEIGHT_MIDDLE, WEIGHT_REAR
        ),
    ) -> EmbedResult:
        """
        EvidencePackage의 3구간 텍스트를 가중 평균(weighted average)으로
        단일 임베딩 벡터로 통합.

        동작 방식
        ---------
        1. front / middle / rear를 각각 독립적으로 임베딩 (이미 정규화됨)
        2. 가중 합산(weighted sum) → 재정규화(re-normalize)
           (정규화된 벡터의 선형 결합은 방향 보존, 크기는 재정규화 필요)

        Parameters
        ----------
        text_front  : 전반 텍스트 (~1,500자)
        text_middle : 중반 텍스트 (~1,500자)
        text_rear   : 후반 텍스트 (~1,500자)
        weights     : (w_front, w_middle, w_rear) — 합이 1.0이 되도록 권장

        Returns
        -------
        EmbedResult
            가중 평균 후 재정규화된 384차원 벡터.
        """
        texts = [text_front, text_middle, text_rear]
        vectors = self._encode(texts)   # (3, 384)

        w = np.array(weights, dtype=np.float32)   # (3,)
        w /= w.sum()                              # 합이 1.0이 되도록 정규화

        combined = (vectors * w[:, np.newaxis]).sum(axis=0)   # (384,)

        # 재정규화: zero 벡터(모든 구간이 빈 텍스트) 처리
        norm = np.linalg.norm(combined)
        if norm > 1e-8:
            combined /= norm
        else:
            logger.warning("embed_evidence: 모든 구간이 빈 텍스트 → zero 벡터 반환")

        return EmbedResult(
            vector=combined.astype(np.float32),
            text_preview=text_front[:80],
        )

    # ── 유사도 유틸 ────────────────────────────

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        두 정규화 벡터의 cosine similarity.

        L2 정규화가 완료된 벡터끼리의 cosine similarity = 내적(dot product).
        별도의 나눗셈 없이 np.dot만으로 계산 가능.

        Returns
        -------
        float
            -1.0 ~ 1.0 범위. 1.0에 가까울수록 의미적으로 유사.
        """
        return float(np.dot(vec_a, vec_b))

    @staticmethod
    def cosine_matrix(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
        """
        query 1개 vs corpus N개의 cosine similarity를 한 번에 계산.

        Parameters
        ----------
        query  : shape (384,)   — 정규화 완료 벡터
        corpus : shape (N, 384) — 정규화 완료 벡터 행렬

        Returns
        -------
        np.ndarray shape (N,)
            각 corpus 항목과의 cosine similarity.
        """
        return corpus @ query   # (N, 384) @ (384,) = (N,)

    @property
    def device(self) -> str:
        return self._device

    def __repr__(self) -> str:
        return (
            f"Embedder(model='{self.model_name}', "
            f"dim={EMBED_DIM}, device='{self._device}')"
        )
