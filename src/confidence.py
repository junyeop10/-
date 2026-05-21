"""Review routing policy for hybrid document intelligence results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConfidenceDecision:
    review_required: bool
    review_reasons: list[str] = field(default_factory=list)


class ConfidencePolicy:
    """Centralize confidence and conflict checks."""

    def __init__(self, *, threshold: float = 0.65, margin: float = 0.08, low_cluster_similarity: float = 0.35) -> None:
        self.threshold = threshold
        self.margin = margin
        self.low_cluster_similarity = low_cluster_similarity

    def evaluate(
        self,
        *,
        confidence: float,
        candidate_scores: dict[str, float],
        rule_prediction: str,
        ml_prediction: str,
        ml_available: bool,
        embedding_prediction: str,
        embedding_available: bool,
        cluster_similarity: float | None = None,
        in_new_cluster: bool = False,
    ) -> ConfidenceDecision:
        reasons: list[str] = []
        if confidence < self.threshold:
            reasons.append("low_confidence")
        ranked = sorted(candidate_scores.values(), reverse=True)
        if len(ranked) > 1 and ranked[0] - ranked[1] < self.margin:
            reasons.append("small_margin")
        if ml_available and rule_prediction and ml_prediction and rule_prediction != ml_prediction:
            reasons.append("rule_ml_conflict")
        if ml_available and embedding_available and embedding_prediction and embedding_prediction != ml_prediction:
            reasons.append("embedding_ml_conflict")
        if in_new_cluster and cluster_similarity is not None and cluster_similarity < self.low_cluster_similarity:
            reasons.append("low_similarity_new_cluster")
        return ConfidenceDecision(review_required=bool(reasons), review_reasons=reasons)
