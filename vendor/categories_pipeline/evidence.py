"""
evidence.py
-----------
문서별 / 단계별 분류 증거 패키지 (EvidencePackage).

목적
----
파이프라인 매 단계의 중간 산출물(임베딩 벡터, 클러스터 라벨,
top-k 카테고리 점수 등)을 문서 1건당 하나의 객체에 모아
다음 네 가지 용도로 활용한다:

  1. 신뢰도 계산         (overall_confidence)
     - similarity 점수를 기본으로, cluster 확신도로 보정
  2. 분류 근거 설명      (explain)
     - 사람이 읽을 수 있는 사유 텍스트 (사용자 확인 / 감사)
  3. 이후 학습에 사용    (to_training_record)
     - 사용자 확정 라벨 + 벡터 → EmbeddingStore.add()용 페이로드
  4. LLM 위임 페이로드   (to_llm_payload)
     - 분류 실패 시 LLM 단계로 넘길 컨텍스트 (원문 + top-k + 사유)

각 단계 evidence는 builder 함수로 만들어 EvidencePackage에 합친다.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from clusterer import ClusterResult, NOISE_LABEL
from embedder import EmbedResult
from reducer import ReduceResult
from similarity import ClassifyResult


# ──────────────────────────────────────────────
# 단계별 evidence (각 단계 산출물 요약)
# ──────────────────────────────────────────────

@dataclass
class EmbeddingEvidence:
    """STEP 1 — 임베딩 단계 증거."""
    model_name: str
    dim: int
    norm: float                  # 정규화 확인 (≈1.0이어야 정상)
    text_preview: str            # 앞 80자
    front_chars: int             # 3구간 입력 길이
    middle_chars: int
    rear_chars: int
    weights: tuple[float, float, float]
    vector: np.ndarray           # (384,) — 다음 단계로 전달 + 학습용 보존

    def to_dict(self, include_vector: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model_name": self.model_name,
            "dim": self.dim,
            "norm": float(self.norm),
            "text_preview": self.text_preview,
            "front_chars": self.front_chars,
            "middle_chars": self.middle_chars,
            "rear_chars": self.rear_chars,
            "weights": list(self.weights),
        }
        if include_vector:
            d["vector"] = self.vector.tolist()
        return d


@dataclass
class ReduceEvidence:
    """STEP 2 — 차원 축소 단계 증거."""
    input_dim: int
    output_dim: int
    metric: str = "cosine"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClusterEvidence:
    """STEP 3 — HDBSCAN 단계 증거."""
    label: int                   # 군집 ID, -1이면 noise
    probability: float           # HDBSCAN 소속 확신도 (0~1)
    cluster_size: int            # 같은 군집에 속한 문서 수 (이 문서 포함)
    is_noise: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SimilarityEvidence:
    """STEP 4 — cosine similarity 단계 증거."""
    decision_category: Optional[str]   # 확정 카테고리, 미달이면 None
    confidence: float                  # 최고 카테고리 점수 (0~1)
    threshold: float                   # 적용된 임계값
    is_confident: bool                 # 확정 여부
    top_k_categories: list[tuple[str, float]]            # 상위 카테고리 점수
    nearest_neighbors: list[tuple[int, str, float]]      # (이력 idx, 카테고리, sim)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_category": self.decision_category,
            "confidence": float(self.confidence),
            "threshold": float(self.threshold),
            "is_confident": self.is_confident,
            "top_k_categories": [(c, float(s)) for c, s in self.top_k_categories],
            "nearest_neighbors": [(i, c, float(s)) for i, c, s in self.nearest_neighbors],
        }


# ──────────────────────────────────────────────
# 통합 EvidencePackage (문서 1건당 1개)
# ──────────────────────────────────────────────

@dataclass
class EvidencePackage:
    """
    문서 1건에 대한 모든 파이프라인 증거를 모은 패키지.

    필드
    ----
    doc_id           : 문서 식별자
    raw_text         : 원본 본문 (LLM 위임 시 필요)
    timestamp        : 생성 시각 (UTC ISO 8601)
    embedding        : STEP 1 증거
    reduce           : STEP 2 증거
    cluster          : STEP 3 증거
    similarity       : STEP 4 증거
    """
    doc_id: str
    raw_text: str
    embedding: EmbeddingEvidence
    reduce: ReduceEvidence
    cluster: ClusterEvidence
    similarity: SimilarityEvidence
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ── 1. 신뢰도 계산 ────────────────────────

    @property
    def overall_confidence(self) -> float:
        """
        통합 신뢰도 점수.

        similarity.confidence를 기본 신호로 사용하되, HDBSCAN noise(-1)인
        경우 *주변 군집 지지가 없음*을 의미하므로 30% 페널티를 준다.
        그 외에는 cluster.probability로 ±15% 보정.

        반환 범위: 0.0 ~ 1.0
        """
        base = float(self.similarity.confidence)
        if self.cluster.is_noise:
            return max(0.0, base * 0.7)
        # 보정 계수 0.85 + 0.15 * cluster_prob → [0.85, 1.0]
        boost = 0.85 + 0.15 * float(self.cluster.probability)
        return min(1.0, base * boost)

    @property
    def final_category(self) -> Optional[str]:
        return self.similarity.decision_category

    @property
    def needs_llm(self) -> bool:
        """신뢰도 임계값 미달 → LLM 위임 필요."""
        return not self.similarity.is_confident

    # ── 2. 분류 근거 설명 ─────────────────────

    def explain(self) -> str:
        """사람이 읽을 수 있는 분류 근거 텍스트."""
        lines = [
            f"문서: {self.doc_id}",
            f"  생성 시각        : {self.timestamp}",
            f"  본문 미리보기    : {self.embedding.text_preview!r}",
            "",
            "[STEP 1] 임베딩",
            f"  모델           : {self.embedding.model_name} ({self.embedding.dim}d)",
            f"  벡터 norm      : {self.embedding.norm:.4f}",
            f"  3구간 길이     : front={self.embedding.front_chars} / "
            f"middle={self.embedding.middle_chars} / rear={self.embedding.rear_chars}",
            "",
            "[STEP 2] 차원 축소",
            f"  {self.reduce.input_dim}d → {self.reduce.output_dim}d ({self.reduce.metric})",
            "",
            "[STEP 3] HDBSCAN",
            (
                f"  noise (어떤 군집에도 미배정)"
                if self.cluster.is_noise
                else f"  군집 {self.cluster.label} 소속 ({self.cluster.cluster_size}개 문서),"
                     f" 확신도 {self.cluster.probability:.3f}"
            ),
            "",
            "[STEP 4] cosine similarity",
            f"  임계값         : {self.similarity.threshold:.2f}",
            f"  최고 점수      : {self.similarity.confidence:.4f}"
            f"  ({'>=' if self.similarity.is_confident else '<'} 임계값)",
            "  상위 카테고리:",
        ]
        for cat, score in self.similarity.top_k_categories:
            marker = "*" if cat == self.similarity.decision_category else " "
            lines.append(f"    {marker} {cat}: {score:.4f}")
        if self.similarity.nearest_neighbors:
            lines.append("  가장 가까운 이력 벡터:")
            for idx, cat, score in self.similarity.nearest_neighbors:
                lines.append(f"      이력[{idx}] {cat}: {score:.4f}")

        lines += [
            "",
            f"[종합] overall_confidence = {self.overall_confidence:.4f}",
            f"       최종 결정        = "
            + (f"'{self.final_category}' 확정" if not self.needs_llm
               else "LLM 단계로 위임"),
        ]
        return "\n".join(lines)

    # ── 3. 이후 학습에 사용 ───────────────────

    def to_training_record(
        self,
        confirmed_category: str,
        source: str = "auto",
    ) -> dict[str, Any]:
        """
        사용자 확정 라벨이 들어왔을 때 EmbeddingStore.add()에 사용할 학습 레코드.

        Parameters
        ----------
        confirmed_category : str
            최종 확정 카테고리 (자동 확정 또는 사용자가 정정한 라벨).
        source : "auto" | "user" | "llm"
            라벨의 출처. 통계/감사용.
        """
        return {
            "doc_id": self.doc_id,
            "category": confirmed_category,
            "vector": self.embedding.vector,   # np.ndarray (384,) — store.add에 그대로 전달
            "source": source,
            "predicted_category": self.final_category,
            "was_auto_confirmed": not self.needs_llm,
            "confidence": self.overall_confidence,
            "timestamp": self.timestamp,
        }

    # ── 4. LLM 위임 페이로드 ──────────────────

    def to_llm_payload(self, max_text_chars: int = 4500) -> dict[str, Any]:
        """
        신뢰도 미달 시 LLM 단계로 넘길 컨텍스트.

        포함 항목:
          - 원본 본문 (잘려서)
          - cosine이 본 후보 카테고리 top-k와 점수
          - 가장 가까운 이력 문서 인덱스 (분류 사례 제시)
          - 자동 분류 실패 사유

        Parameters
        ----------
        max_text_chars : int
            LLM 토큰 절약을 위한 원문 truncation 길이.
        """
        return {
            "doc_id": self.doc_id,
            "text": self.raw_text[:max_text_chars],
            "candidates": [
                {"category": c, "cosine_score": float(s)}
                for c, s in self.similarity.top_k_categories
            ],
            "nearest_history": [
                {"index": i, "category": c, "cosine_score": float(s)}
                for i, c, s in self.similarity.nearest_neighbors
            ],
            "cluster_label": self.cluster.label,
            "cluster_probability": float(self.cluster.probability),
            "failure_reason": (
                f"최고 cosine 점수 {self.similarity.confidence:.3f} < "
                f"임계값 {self.similarity.threshold:.2f}"
            ),
            "overall_confidence": self.overall_confidence,
        }

    # ── 직렬화 ────────────────────────────────

    def to_dict(self, include_vector: bool = False) -> dict[str, Any]:
        """JSON 직렬화 가능한 dict (vector는 옵션)."""
        return {
            "doc_id": self.doc_id,
            "timestamp": self.timestamp,
            "raw_text_chars": len(self.raw_text),
            "embedding": self.embedding.to_dict(include_vector=include_vector),
            "reduce": self.reduce.to_dict(),
            "cluster": self.cluster.to_dict(),
            "similarity": self.similarity.to_dict(),
            "overall_confidence": self.overall_confidence,
            "final_category": self.final_category,
            "needs_llm": self.needs_llm,
        }

    def to_json(self, include_vector: bool = False, **dumps_kwargs: Any) -> str:
        return json.dumps(self.to_dict(include_vector=include_vector),
                          ensure_ascii=False, **dumps_kwargs)


# ──────────────────────────────────────────────
# 빌더 함수 (main.py 또는 호출자가 사용)
# ──────────────────────────────────────────────

def build_embedding_evidence(
    embed_result: EmbedResult,
    front: str,
    middle: str,
    rear: str,
    weights: tuple[float, float, float] = (0.5, 0.25, 0.25),
) -> EmbeddingEvidence:
    return EmbeddingEvidence(
        model_name=embed_result.model_name,
        dim=embed_result.dim,
        norm=embed_result.norm,
        text_preview=embed_result.text_preview,
        front_chars=len(front),
        middle_chars=len(middle),
        rear_chars=len(rear),
        weights=weights,
        vector=embed_result.vector,
    )


def build_reduce_evidence(
    reduce_result: ReduceResult,
    metric: str = "cosine",
) -> ReduceEvidence:
    return ReduceEvidence(
        input_dim=reduce_result.n_input_dim,
        output_dim=reduce_result.n_output_dim,
        metric=metric,
    )


def build_cluster_evidence(
    doc_index: int,
    cluster_result: ClusterResult,
) -> ClusterEvidence:
    label = int(cluster_result.labels[doc_index])
    is_noise = label == NOISE_LABEL
    cluster_size = (
        0 if is_noise
        else len(cluster_result.cluster_to_indices.get(label, []))
    )
    return ClusterEvidence(
        label=label,
        probability=float(cluster_result.probabilities[doc_index]),
        cluster_size=cluster_size,
        is_noise=is_noise,
    )


def build_similarity_evidence(
    classify_result: ClassifyResult,
    nearest_neighbors: list[tuple[int, str, float]],
) -> SimilarityEvidence:
    return SimilarityEvidence(
        decision_category=classify_result.category,
        confidence=float(classify_result.confidence),
        threshold=float(classify_result.threshold),
        is_confident=classify_result.is_confident,
        top_k_categories=[(c, float(s)) for c, s in classify_result.top_k],
        nearest_neighbors=nearest_neighbors,
    )


def build_evidence_package(
    doc_id: str,
    raw_text: str,
    embedding: EmbeddingEvidence,
    reduce: ReduceEvidence,
    cluster: ClusterEvidence,
    similarity: SimilarityEvidence,
) -> EvidencePackage:
    return EvidencePackage(
        doc_id=doc_id,
        raw_text=raw_text,
        embedding=embedding,
        reduce=reduce,
        cluster=cluster,
        similarity=similarity,
    )
