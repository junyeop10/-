"""Safe rebuildable feedback learning helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from src.storage import ClassificationRepository
from src.text_cleaner import tokenize_text


def rebuild_adaptive_learning(repository: ClassificationRepository, min_occurrences: int = 2) -> dict[str, Any]:
    feedback_rows = repository.fetch_feedback_learning_rows()
    grouped_tokens: dict[str, list[str]] = defaultdict(list)
    confusion_pairs: Counter[tuple[str, str]] = Counter()

    for row in feedback_rows:
        final_category = str(row["final_middle_category"])
        grouped_tokens[final_category].extend(tokenize_text(str(row["evidence_text"])))
        confusion_pairs[(str(row["predicted_middle_category"]), final_category)] += 1

    repository.clear_adaptive_rule_boosts()
    inserted = 0
    for category, tokens in grouped_tokens.items():
        token_counter = Counter(tokens)
        for token, count in token_counter.items():
            if count < min_occurrences:
                continue
            repository.insert_adaptive_rule_boost(
                category=category,
                token=token,
                boost=round(min(0.25, 0.03 * count), 4),
                source="feedback_rebuild",
                support_count=count,
            )
            inserted += 1

    return {
        "adaptive_rules_inserted": inserted,
        "confusion_pairs": [
            {"predicted": predicted, "corrected": corrected, "count": count}
            for (predicted, corrected), count in confusion_pairs.most_common()
        ],
    }
