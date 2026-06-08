"""Lexical feature scoring for category profile similarity."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


LEXICAL_WEIGHTS = {
    "tfidf": 0.60,
    "ngram": 0.30,
    "bow": 0.10,
}
TOKEN_PATTERN = r"(?u)[가-힣A-Za-z0-9_]{1,}"


@dataclass
class LexicalVectorizers:
    bow: CountVectorizer
    tfidf: TfidfVectorizer
    ngram: TfidfVectorizer


def normalize_text(text: str | None) -> str:
    """Normalize text while preserving Korean, English, numbers, and basic punctuation."""
    if not text:
        return ""
    cleaned = str(text).replace("\x00", " ").lower()
    cleaned = re.sub(r"[^가-힣a-z0-9\s.,;:!?()\[\]{}<>/%+\-_'\"|]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def build_lexical_vectorizers(training_texts: list[str]) -> LexicalVectorizers | None:
    """Build BoW, TF-IDF, and n-gram vectorizers from non-empty training text."""
    normalized = [normalize_text(text) for text in training_texts if normalize_text(text)]
    if len(normalized) < 2:
        return None
    try:
        bow = CountVectorizer(token_pattern=TOKEN_PATTERN, ngram_range=(1, 1), lowercase=False)
        tfidf = TfidfVectorizer(token_pattern=TOKEN_PATTERN, ngram_range=(1, 2), lowercase=False, sublinear_tf=True)
        ngram = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=False, min_df=1)
        bow.fit(normalized)
        tfidf.fit(normalized)
        ngram.fit(normalized)
    except ValueError:
        return None
    return LexicalVectorizers(bow=bow, tfidf=tfidf, ngram=ngram)


def compute_lexical_scores(
    text: str,
    category_profiles: dict[str, Any],
    vectorizers: LexicalVectorizers | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute lexical similarity scores between input text and category profiles."""
    query = normalize_text(text)
    profile_texts = {
        category: normalize_text(_profile_to_text(profile))
        for category, profile in category_profiles.items()
    }
    if not query or not profile_texts or not any(profile_texts.values()):
        return _zero_scores(category_profiles)

    active_profiles = [value for value in profile_texts.values() if value]
    vectorizers = vectorizers or build_lexical_vectorizers([query, *active_profiles])
    if vectorizers is None:
        return _zero_scores(category_profiles)

    results: dict[str, dict[str, Any]] = {}
    for category, profile_text in profile_texts.items():
        if not profile_text:
            results[category] = _zero_score()
            continue
        tfidf_score = _similarity(vectorizers.tfidf, query, profile_text)
        ngram_score = _similarity(vectorizers.ngram, query, profile_text)
        bow_score = _similarity(vectorizers.bow, query, profile_text)
        lexical_score = (
            tfidf_score * LEXICAL_WEIGHTS["tfidf"]
            + ngram_score * LEXICAL_WEIGHTS["ngram"]
            + bow_score * LEXICAL_WEIGHTS["bow"]
        )
        results[category] = {
            "tfidf_score": round(tfidf_score, 4),
            "ngram_score": round(ngram_score, 4),
            "bow_score": round(bow_score, 4),
            "lexical_score": round(float(lexical_score), 4),
            "top_terms": _top_shared_terms(query, profile_text, limit=12),
            "top_ngram_matches": _top_shared_char_ngrams(query, profile_text, limit=12),
        }
    return results


def flatten_lexical_scores(lexical_results: dict[str, dict[str, Any]], categories: list[str]) -> dict[str, float]:
    """Return category -> lexical_score for score fusion."""
    return {
        category: float(lexical_results.get(category, {}).get("lexical_score", 0.0) or 0.0)
        for category in categories
    }


def build_category_profiles_from_rows(rows: list[dict[str, Any]], categories: list[str]) -> dict[str, dict[str, Any]]:
    """Convert DB category profile rows into lexical scoring profiles."""
    profiles: dict[str, dict[str, Any]] = {category: {"profile_text": category} for category in categories}
    for row in rows:
        category = str(row.get("type", "")).strip()
        if not category or category not in profiles:
            continue
        lexical_profile = _loads_object(row.get("lexical_profile_json", "{}"))
        profile_signals = _loads_object(row.get("profile_signals_json", "{}"))
        profiles[category] = {
            "profile_text": str(row.get("profile_text", "")),
            "lexical_signals": lexical_profile.get("lexical_signals", []),
            "profile_signals": profile_signals,
            "tags": _loads_list(row.get("tags_json", "[]")),
        }
    return profiles


def _similarity(vectorizer: CountVectorizer | TfidfVectorizer, left: str, right: str) -> float:
    try:
        matrix = vectorizer.transform([left, right])
        value = cosine_similarity(matrix[0], matrix[1])[0][0]
        if np.isnan(value):
            return 0.0
        return float(max(0.0, min(1.0, value)))
    except Exception:
        return 0.0


def _profile_to_text(profile: Any) -> str:
    if isinstance(profile, str):
        return profile
    if isinstance(profile, list):
        return " ".join(str(item) for item in profile)
    if not isinstance(profile, dict):
        return ""
    parts: list[str] = []
    for key in ("profile_text", "description", "lexical_signals", "tags"):
        value = profile.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value:
            parts.append(str(value))
    signals = profile.get("profile_signals")
    if isinstance(signals, dict):
        for value in signals.values():
            if isinstance(value, list):
                parts.extend(str(item) for item in value)
            elif value:
                parts.append(str(value))
    return " ".join(parts)


def _top_shared_terms(left: str, right: str, *, limit: int) -> list[str]:
    left_terms = re.findall(TOKEN_PATTERN, left)
    right_counts = {term: right.count(term) for term in set(re.findall(TOKEN_PATTERN, right))}
    scored = [(term, right_counts.get(term, 0), left_terms.count(term)) for term in set(left_terms)]
    scored = [item for item in scored if item[1] > 0]
    scored.sort(key=lambda item: (-item[1], -item[2], item[0]))
    return [term for term, _right_count, _left_count in scored[:limit]]


def _top_shared_char_ngrams(left: str, right: str, *, limit: int) -> list[str]:
    left_ngrams = _char_ngrams(left)
    right_ngrams = _char_ngrams(right)
    shared = left_ngrams & right_ngrams
    return sorted(shared, key=lambda value: (-len(value), value))[:limit]


def _char_ngrams(text: str) -> set[str]:
    compact = re.sub(r"\s+", " ", text)
    ngrams: set[str] = set()
    for size in (3, 4, 5):
        for index in range(max(0, len(compact) - size + 1)):
            item = compact[index : index + size].strip()
            if len(item) == size:
                ngrams.add(item)
    return ngrams


def _zero_scores(category_profiles: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {category: _zero_score() for category in category_profiles}


def _zero_score() -> dict[str, Any]:
    return {
        "tfidf_score": 0.0,
        "ngram_score": 0.0,
        "bow_score": 0.0,
        "lexical_score": 0.0,
        "top_terms": [],
        "top_ngram_matches": [],
    }


def _loads_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _loads_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []
