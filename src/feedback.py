"""사용자 수정 패턴에서 새 규칙 후보를 찾습니다."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from src.storage import ClassificationRepository
from src.text_cleaner import tokenize_text


def build_rule_suggestions(
    repository: ClassificationRepository,
    min_occurrences: int = 2,
) -> list[dict[str, Any]]:
    """반복 수정 사례에서 공통 토큰을 추출해 규칙 후보를 만듭니다."""
    feedback_rows = repository.fetch_correction_examples()
    if not feedback_rows:
        return []

    active_rule_tokens = repository.get_active_rule_tokens_by_category()
    grouped_tokens: dict[tuple[str, str], list[str]] = defaultdict(list)
    evidence_map: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in feedback_rows:
        pair_key = (row["predicted_category"], row["final_category"])
        tokens = tokenize_text(row["extracted_text"])
        grouped_tokens[pair_key].extend(tokens)
        evidence_map[pair_key].append({"file_id": row["file_id"], "file_name": row["file_name"]})

    candidates: list[dict[str, Any]] = []

    for (predicted_category, final_category), tokens in grouped_tokens.items():
        token_counter = Counter(tokens)
        existing_tokens = active_rule_tokens.get(final_category, set())

        for token, count in token_counter.items():
            if count < min_occurrences or token in existing_tokens:
                continue

            candidates.append(
                {
                    "category": final_category,
                    "rule_type": "keyword",
                    "pattern": token,
                    "weight": round(1.0 + (count * 0.2), 2),
                    "source": "feedback_pattern",
                    "predicted_category": predicted_category,
                    "support_count": count,
                    "evidence": evidence_map[(predicted_category, final_category)][:5],
                }
            )

    candidates.sort(key=lambda item: (-item["support_count"], item["category"], item["pattern"]))
    return candidates
