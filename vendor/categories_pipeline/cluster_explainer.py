"""
cluster_explainer.py
--------------------
HDBSCAN 군집이 *왜 그렇게 묶였는지* 설명 데이터를 생성한다.

각 군집별로:
  - 대표 문서(centroid에 가장 가까운 N개)
  - 군집 공통 키워드 (TF-IDF 상위 N개) — 한국어 명사/단어 기준 단순화
  - 군집의 카테고리 매핑 결과(centroid → cosine top-1)
  - 군집 평균 신뢰도

노이즈(-1) 처리:
  - 가장 가까운 군집과의 거리
  - 가장 가까운 이력 벡터의 카테고리
  - 미분류 사유 (낮은 cosine / 모호한 분류 / cluster noise)
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np


# 한국어 + 영어 단어 토큰화. 짧은 stopword 처리.
_TOKEN_RE = re.compile(r"[가-힣]{2,}|[A-Za-z]{3,}")
_STOPWORDS = {
    "그리고", "있는", "있다", "하는", "위해", "통해", "관련", "대한",
    "위한", "이는", "그것", "이를", "이상", "이하", "기준", "경우",
    "또는", "the", "and", "for", "with", "this", "that", "from",
    "are", "was", "you", "have", "has",
}


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text) if t.lower() not in _STOPWORDS]


def _tf_idf_top(texts: list[str], top_n: int = 5) -> list[tuple[str, float]]:
    """간이 TF-IDF: 단어 빈도 × log(전체 / 그 단어가 등장한 문서 수)."""
    if not texts:
        return []
    tokenized = [_tokenize(t) for t in texts]
    df = Counter()
    for toks in tokenized:
        df.update(set(toks))
    N = len(texts)
    tf = Counter()
    for toks in tokenized:
        tf.update(toks)
    scored: list[tuple[str, float]] = []
    for w, f in tf.items():
        d = df[w]
        if d == 0:
            continue
        # 군집 내부 단어이므로 IDF가 의미 없을 수 있어, 빈도 위주 + 너무 흔한 것 페널티
        score = f * (1.0 + np.log((N + 1) / (d + 1)))
        scored.append((w, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


@dataclass
class ClusterExplanation:
    cluster_label: int               # -1이면 noise
    is_noise: bool
    size: int                        # 군집 소속 문서 수 (noise면 0)
    centroid: np.ndarray | None      # 평균 임베딩 벡터 (noise는 None)
    representative_doc_ids: list[str]    # centroid에 가까운 순 3개
    top_keywords: list[tuple[str, float]] # 공통 키워드 top-N
    mapped_category: str | None      # centroid → cosine top-1
    mapped_confidence: float         # cosine top-1 점수
    avg_member_confidence: float     # 멤버 문서 평균 신뢰도

    def summary(self) -> str:
        if self.is_noise:
            return (
                f"[noise] {self.size}개 문서 — HDBSCAN이 어떤 군집에도 묶을 만큼 "
                f"밀집된 이웃이 없어서 미배정"
            )
        kw = ", ".join(w for w, _ in self.top_keywords) if self.top_keywords else "(키워드 없음)"
        rep = ", ".join(self.representative_doc_ids[:3])
        return (
            f"[군집 {self.cluster_label}] {self.size}개 문서  → "
            f"카테고리 매핑: '{self.mapped_category}' (cosine={self.mapped_confidence:.3f})\n"
            f"  공통 키워드: {kw}\n"
            f"  대표 문서   : {rep}\n"
            f"  평균 신뢰도 : {self.avg_member_confidence:.3f}"
        )


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a (N,D), b (M,D) → (N,M) cosine similarity (둘 다 L2-normalize 됐다고 가정)."""
    return a @ b.T


def explain_clusters(
    *,
    doc_ids: list[str],
    doc_texts: list[str],
    doc_embeddings: np.ndarray,         # (N, 384) — 정규화됨
    cluster_labels: np.ndarray,         # (N,) — HDBSCAN 라벨
    category_folders: list[str],        # 카테고리 폴더명 리스트
    category_embeddings: np.ndarray,    # (M, 384) — 정규화됨
    member_confidences: list[float],    # (N,) overall_confidence 리스트
) -> list[ClusterExplanation]:
    """군집별 해설을 생성한다. 노이즈 군집(-1)도 1개 항목으로 포함된다."""

    explanations: list[ClusterExplanation] = []
    unique_labels = sorted(set(int(l) for l in cluster_labels))

    for label in unique_labels:
        idxs = [i for i, l in enumerate(cluster_labels) if int(l) == label]
        if label == -1:
            explanations.append(ClusterExplanation(
                cluster_label=-1,
                is_noise=True,
                size=len(idxs),
                centroid=None,
                representative_doc_ids=[doc_ids[i] for i in idxs[:3]],
                top_keywords=_tf_idf_top([doc_texts[i] for i in idxs], top_n=5),
                mapped_category=None,
                mapped_confidence=0.0,
                avg_member_confidence=(
                    float(np.mean([member_confidences[i] for i in idxs])) if idxs else 0.0
                ),
            ))
            continue

        # 1) centroid
        embs = doc_embeddings[idxs]
        centroid = embs.mean(axis=0)
        # 재정규화
        norm = np.linalg.norm(centroid)
        if norm > 1e-8:
            centroid = centroid / norm

        # 2) centroid와의 거리로 대표 문서 정렬
        sims_to_centroid = embs @ centroid    # (n_members,)
        order = np.argsort(sims_to_centroid)[::-1]
        rep_ids = [doc_ids[idxs[k]] for k in order[:3]]

        # 3) 공통 키워드
        keywords = _tf_idf_top([doc_texts[i] for i in idxs], top_n=5)

        # 4) centroid → 카테고리 매핑
        cat_sims = _cosine(centroid[np.newaxis, :], category_embeddings)[0]  # (M,)
        best = int(np.argmax(cat_sims))

        # 5) 평균 신뢰도
        avg_conf = float(np.mean([member_confidences[i] for i in idxs]))

        explanations.append(ClusterExplanation(
            cluster_label=label,
            is_noise=False,
            size=len(idxs),
            centroid=centroid,
            representative_doc_ids=rep_ids,
            top_keywords=keywords,
            mapped_category=category_folders[best],
            mapped_confidence=float(cat_sims[best]),
            avg_member_confidence=avg_conf,
        ))

    # 군집 라벨 순서대로 (noise를 마지막에)
    explanations.sort(key=lambda e: (e.is_noise, e.cluster_label))
    return explanations


# ──────────────────────────────────────────────
# 미분류 / 노이즈 문서 개별 사유 분석
# ──────────────────────────────────────────────

@dataclass
class UnclassifiedReason:
    doc_id: str
    cluster_label: int                  # 보통 -1 (noise) 또는 임계값 미달
    overall_confidence: float
    threshold: float
    top_candidates: list[tuple[str, float]]   # 후보 top-3
    reason: str                          # 사람이 읽을 한국어 사유

    def explain(self) -> str:
        cands = ", ".join(f"{c}({s:.3f})" for c, s in self.top_candidates)
        return (
            f"{self.doc_id}\n"
            f"  사유  : {self.reason}\n"
            f"  후보  : {cands}\n"
            f"  신뢰도: {self.overall_confidence:.3f} (임계값 {self.threshold:.2f})"
        )


def explain_unclassified(
    *,
    packages: list,                # list[EvidencePackage]
    threshold: float,
) -> list[UnclassifiedReason]:
    """needs_llm=True인 문서마다 사유를 생성."""
    reasons: list[UnclassifiedReason] = []
    for pkg in packages:
        if not pkg.needs_llm:
            continue

        top_k = pkg.similarity.top_k_categories[:3]
        conf = pkg.overall_confidence

        # 사유 결정
        if pkg.cluster.is_noise and conf < threshold:
            why = (
                "HDBSCAN 군집에서 noise(-1)로 분류됨 + cosine 신뢰도도 임계값 미달. "
                "이웃 이력이 부족하거나 의미가 모호함."
            )
        elif pkg.cluster.is_noise:
            why = "HDBSCAN noise(-1) — 어떤 군집에도 밀집되어 묶이지 않음."
        else:
            if len(top_k) >= 2 and abs(top_k[0][1] - top_k[1][1]) < 0.05:
                why = (
                    f"상위 후보 두 카테고리의 cosine 차이가 매우 작음 "
                    f"({top_k[0][1]:.3f} vs {top_k[1][1]:.3f}) — 모호함."
                )
            else:
                why = (
                    f"최고 cosine 점수({pkg.similarity.confidence:.3f})가 "
                    f"임계값({threshold:.2f}) 미만."
                )

        reasons.append(UnclassifiedReason(
            doc_id=pkg.doc_id,
            cluster_label=pkg.cluster.label,
            overall_confidence=conf,
            threshold=threshold,
            top_candidates=top_k,
            reason=why,
        ))
    return reasons
