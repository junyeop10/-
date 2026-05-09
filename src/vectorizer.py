"""sentence-transformers 임베딩과 코사인 유사도 계산을 담당합니다."""

from __future__ import annotations

import json
import math
from typing import Any


class SentenceTransformerEmbedder:
    """문장 임베딩 생성과 confirmed_examples 비교를 담당합니다."""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> None:
        """사용할 모델 이름을 저장합니다."""
        self.model_name = model_name
        self._model = None

    def encode(self, text: str) -> list[float]:
        """입력 텍스트를 임베딩 벡터로 변환합니다."""
        model = self._load_model()
        vector = model.encode(text, normalize_embeddings=True)
        return [float(value) for value in vector.tolist()]

    def encode_many(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts in one model call for fast mode."""
        if not texts:
            return []
        model = self._load_model()
        vectors = model.encode(texts, normalize_embeddings=True)
        return [[float(value) for value in vector.tolist()] for vector in vectors]

    def score_against_examples(
        self,
        query_embedding: list[float],
        examples: list[Any],
        categories: list[str],
    ) -> dict[str, Any]:
        """확정 예시와 비교해 카테고리별 임베딩 점수를 계산합니다."""
        scores = {category: 0.0 for category in categories}
        top_examples: dict[str, dict[str, Any]] = {}
        grouped_similarities: dict[str, list[float]] = {category: [] for category in categories}

        for example in examples:
            example_embedding = json.loads(example["embedding_json"])
            if not example_embedding or len(example_embedding) != len(query_embedding):
                continue
            similarity = cosine_similarity(query_embedding, example_embedding)
            category = str(example["category"])
            grouped_similarities.setdefault(category, []).append(similarity)

            current_top = top_examples.get(category)
            if current_top is None or similarity > current_top["similarity"]:
                top_examples[category] = {
                    "category": category,
                    "similarity": similarity,
                    "file_name": example["file_name"],
                }

        for category, similarities in grouped_similarities.items():
            if similarities:
                top_values = sorted(similarities, reverse=True)[:3]
                scores[category] = round(sum(top_values) / len(top_values), 4)

        return {"scores": scores, "top_examples": top_examples}

    def _load_model(self) -> Any:
        """필요한 순간에만 sentence-transformers 모델을 불러옵니다."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError(
                    "sentence-transformers를 찾지 못했습니다. "
                    "`pip install -r requirements.txt`를 다시 실행하세요."
                ) from error

            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as error:
                raise RuntimeError(
                    f"임베딩 모델 로딩 실패: {self.model_name}. "
                    "처음 실행이라면 모델 다운로드가 필요할 수 있습니다."
                ) from error

        return self._model


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """두 벡터의 코사인 유사도를 계산합니다."""
    if len(left) != len(right):
        raise ValueError("벡터 길이가 다릅니다.")

    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (left_norm * right_norm)
