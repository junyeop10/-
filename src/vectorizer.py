"""sentence-transformers embeddings, cosine similarity, and persistent cache support."""

from __future__ import annotations

import json
import logging
import math
import time
from typing import Any

import numpy as np

from src.embedding_repository import EmbeddingRepository
from src.hash_utils import compute_raw_text_hash
from src.storage import ClassificationRepository
from src.text_cleaner import normalize_text


EMBEDDING_CACHE_VERSION = "2-xxhash"
logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder:
    """Generate embeddings and compare them with confirmed examples."""

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        *,
        embedding_repository: EmbeddingRepository | None = None,
        use_legacy_sqlite_cache: bool = True,
        migrate_legacy_cache_on_hit: bool = True,
        dual_write_legacy_sqlite: bool = False,
    ) -> None:
        self.model_name = model_name
        self.embedding_repository = embedding_repository
        self.use_legacy_sqlite_cache = use_legacy_sqlite_cache
        self.migrate_legacy_cache_on_hit = migrate_legacy_cache_on_hit
        self.dual_write_legacy_sqlite = dual_write_legacy_sqlite
        self._model = None
        self._last_encode_meta: dict[str, Any] = {}
        self._last_batch_encode_meta: list[dict[str, Any]] = []
        self._last_model_load_time = 0.0

    def encode(
        self,
        text: str,
        repository: ClassificationRepository | None = None,
        file_hash: str = "",
        text_kind: str = "query",
        embedding_version: str = EMBEDDING_CACHE_VERSION,
    ) -> list[float]:
        cache_payload = self._build_cache_payload(
            text=text,
            file_hash=file_hash,
            text_kind=text_kind,
            embedding_version=embedding_version,
        )
        cached_embedding = self._load_cached_embedding(repository, cache_payload, text_kind=text_kind)
        if cached_embedding is not None:
            return cached_embedding

        start = time.perf_counter()
        model = self._load_model()
        vector = model.encode(text, normalize_embeddings=True)
        embedding = [float(value) for value in vector.tolist()]
        elapsed = time.perf_counter() - start
        self._last_encode_meta = {
            "cache_hit": False,
            "cache_key": cache_payload["cache_key"] if cache_payload is not None else "",
            "elapsed": elapsed,
            "model_name": self.model_name,
            "text_kind": text_kind,
            "model_load_time": self._last_model_load_time,
        }
        print(f"[embedding-cache-miss] kind={text_kind} model={self.model_name} time={elapsed:.3f}s")

        self._store_cached_embedding(repository, cache_payload, embedding, text_kind=text_kind)
        return embedding

    def encode_many(
        self,
        texts: list[str],
        repository: ClassificationRepository | None = None,
        file_hashes: list[str] | None = None,
        text_kind: str = "query",
        embedding_version: str = EMBEDDING_CACHE_VERSION,
    ) -> list[list[float]]:
        """Encode multiple texts, reusing persistent cache for hits."""
        if not texts:
            return []
        file_hashes = file_hashes or [""] * len(texts)
        if len(file_hashes) != len(texts):
            raise ValueError("file_hashes length must match texts length.")

        results: list[list[float] | None] = [None] * len(texts)
        misses: list[str] = []
        miss_indices: list[int] = []
        miss_payloads: list[dict[str, str] | None] = []
        batch_meta: list[dict[str, Any] | None] = [None] * len(texts)

        for index, text in enumerate(texts):
            payload = self._build_cache_payload(
                text=text,
                file_hash=file_hashes[index],
                text_kind=text_kind,
                embedding_version=embedding_version,
            )
            cached = self._load_cached_embedding(repository, payload, text_kind=text_kind)
            if cached is not None:
                batch_meta[index] = dict(self._last_encode_meta)
                results[index] = cached
                continue
            misses.append(text)
            miss_indices.append(index)
            miss_payloads.append(payload)

        if misses:
            start = time.perf_counter()
            model = self._load_model()
            vectors = model.encode(misses, normalize_embeddings=True)
            elapsed = time.perf_counter() - start
            print(
                f"[embedding-cache-miss] kind={text_kind} model={self.model_name} "
                f"batch={len(misses)} time={elapsed:.3f}s"
            )
            for result_index, vector, payload in zip(miss_indices, vectors, miss_payloads):
                embedding = [float(value) for value in vector.tolist()]
                results[result_index] = embedding
                per_item_elapsed = elapsed / max(len(misses), 1)
                batch_meta[result_index] = {
                    "cache_hit": False,
                    "cache_key": payload["cache_key"] if payload is not None else "",
                    "elapsed": per_item_elapsed,
                    "model_name": self.model_name,
                    "text_kind": text_kind,
                    "model_load_time": self._last_model_load_time,
                }
                self._store_cached_embedding(repository, payload, embedding, text_kind=text_kind)

        self._last_batch_encode_meta = [
            meta if meta is not None else {"cache_hit": False, "elapsed": 0.0, "model_name": self.model_name, "text_kind": text_kind}
            for meta in batch_meta
        ]
        return [embedding if embedding is not None else [] for embedding in results]

    def score_against_examples(
        self,
        query_embedding: list[float],
        examples: list[Any],
        categories: list[str],
    ) -> dict[str, Any]:
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
        if self._model is None:
            load_start = time.perf_counter()
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError(
                    "sentence-transformers not found. Run `pip install -r requirements.txt`."
                ) from error

            try:
                self._model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                try:
                    self._model = SentenceTransformer(self.model_name)
                except Exception as error:
                    raise RuntimeError(
                        f"Embedding model load failed: {self.model_name}. "
                        "The first run may need to download model weights."
                    ) from error
            self._last_model_load_time = time.perf_counter() - load_start

        return self._model

    def _build_cache_payload(
        self,
        text: str,
        file_hash: str,
        text_kind: str,
        embedding_version: str,
    ) -> dict[str, str] | None:
        normalized_text = normalize_text(text)
        if not self._is_cacheable_text(normalized_text):
            return None
        text_signature = compute_raw_text_hash(normalized_text)
        safe_file_hash = file_hash or ""
        cache_key = "|".join(
            [
                safe_file_hash,
                self.model_name,
                text_signature,
                embedding_version,
                text_kind,
            ]
        )
        if self.embedding_repository is not None and text_kind == "compressed_query":
            storage_id = f"compressed_{compute_raw_text_hash(cache_key)}"
        elif self.embedding_repository is not None:
            storage_id = self.embedding_repository.build_storage_id(safe_file_hash, cache_key)
        else:
            storage_id = safe_file_hash or compute_raw_text_hash(cache_key)
        return {
            "cache_key": cache_key,
            "file_hash": safe_file_hash,
            "text_signature": text_signature,
            "storage_id": storage_id,
            "embedding_version": embedding_version,
            "text_kind": text_kind,
        }

    def _is_cacheable_text(self, text: str) -> bool:
        return bool(text.strip())

    def get_last_encode_meta(self) -> dict[str, Any]:
        """Return metadata for the most recent single-text embedding call."""
        return dict(self._last_encode_meta)

    def get_last_batch_encode_meta(self) -> list[dict[str, Any]]:
        """Return metadata for the most recent batch embedding call."""
        return [dict(item) for item in self._last_batch_encode_meta]

    def _load_cached_embedding(
        self,
        repository: ClassificationRepository | None,
        cache_payload: dict[str, str] | None,
        *,
        text_kind: str,
    ) -> list[float] | None:
        if cache_payload is None:
            return None

        if self.embedding_repository is not None:
            cached_vector = self.embedding_repository.get_embedding(cache_payload["storage_id"])
            cached_metadata = self.embedding_repository.get_metadata(cache_payload["storage_id"])
            if cached_vector is not None and self._metadata_matches_cache_payload(cached_metadata, cache_payload):
                self._last_encode_meta = {
                    "cache_hit": True,
                    "cache_backend": "hdf5",
                    "cache_key": cache_payload["cache_key"],
                    "elapsed": 0.0,
                    "model_name": self.model_name,
                    "text_kind": text_kind,
                }
                logger.info("embedding-cache-hit backend=hdf5 kind=%s model=%s", text_kind, self.model_name)
                return [float(value) for value in cached_vector.tolist()]

        if repository is not None and self.use_legacy_sqlite_cache:
            cached = repository.get_cached_embedding(cache_payload["cache_key"])
            if cached is not None:
                embedding = [float(value) for value in json.loads(cached["embedding_json"])]
                self._last_encode_meta = {
                    "cache_hit": True,
                    "cache_backend": "sqlite",
                    "cache_key": cache_payload["cache_key"],
                    "elapsed": 0.0,
                    "model_name": self.model_name,
                    "text_kind": text_kind,
                }
                print(
                    f"[embedding-cache-hit] backend=sqlite kind={text_kind} model={self.model_name} "
                    f"hits={cached['hit_count']}"
                )
                if self.embedding_repository is not None and self.migrate_legacy_cache_on_hit:
                    self._save_to_hdf5_cache(cache_payload, embedding)
                return embedding
        return None

    def _store_cached_embedding(
        self,
        repository: ClassificationRepository | None,
        cache_payload: dict[str, str] | None,
        embedding: list[float],
        *,
        text_kind: str,
    ) -> None:
        if cache_payload is None:
            return

        if self.embedding_repository is not None:
            self._save_to_hdf5_cache(cache_payload, embedding)

        if repository is not None and (self.embedding_repository is None or self.dual_write_legacy_sqlite):
            repository.cache_embedding(
                cache_key=cache_payload["cache_key"],
                file_hash=cache_payload["file_hash"],
                model_name=self.model_name,
                text_signature=cache_payload["text_signature"],
                embedding_version=cache_payload["embedding_version"],
                text_kind=text_kind,
                embedding=embedding,
            )

    def _save_to_hdf5_cache(self, cache_payload: dict[str, str], embedding: list[float]) -> None:
        if self.embedding_repository is None:
            return
        metadata = {
            "cache_key": cache_payload["cache_key"],
            "file_hash": cache_payload["file_hash"],
            "model_name": self.model_name,
            "text_signature": cache_payload["text_signature"],
            "embedding_version": cache_payload["embedding_version"],
            "text_kind": cache_payload["text_kind"],
        }
        self.embedding_repository.save_embedding(
            cache_payload["storage_id"],
            np.asarray(embedding, dtype=np.float32),
            metadata,
            overwrite=True,
        )

    def _metadata_matches_cache_payload(
        self,
        metadata: dict[str, Any] | None,
        cache_payload: dict[str, str],
    ) -> bool:
        if not metadata:
            return False
        return (
            str(metadata.get("model_name", "")) == self.model_name
            and str(metadata.get("text_signature", "")) == cache_payload["text_signature"]
            and str(metadata.get("embedding_version", "")) == cache_payload["embedding_version"]
            and str(metadata.get("text_kind", "")) == cache_payload["text_kind"]
        )


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(left) != len(right):
        raise ValueError("Vector lengths do not match.")

    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (left_norm * right_norm)
