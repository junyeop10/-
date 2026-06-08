"""HDF5-backed embedding repository with lazy open and migration helpers."""

from __future__ import annotations

import json
import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
from filelock import FileLock

from src.config import AppConfig
from src.hash_utils import compute_raw_text_hash

if TYPE_CHECKING:
    from src.storage import ClassificationRepository


logger = logging.getLogger(__name__)

HDF5_VECTOR_GROUP = "vectors"
HDF5_METADATA_GROUP = "metadata"


class EmbeddingRepository:
    """Persist embeddings in HDF5 and metadata as JSON strings."""

    def __init__(
        self,
        path: str | Path,
        *,
        lock_timeout_seconds: float = 15.0,
        enforce_dimension: bool = True,
    ) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(f"{self.path.suffix or '.h5'}.lock")
        self.lock_timeout_seconds = lock_timeout_seconds
        self.enforce_dimension = enforce_dimension
        self._local_lock = threading.RLock()
        self._file_lock = FileLock(str(self.lock_path), timeout=lock_timeout_seconds)
        self._ensure_container()

    def __enter__(self) -> EmbeddingRepository:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback

    def save_embedding(
        self,
        file_id: str,
        vector: np.ndarray,
        metadata: dict[str, Any] | None = None,
        *,
        overwrite: bool = True,
    ) -> None:
        safe_id = self._validate_file_id(file_id)
        normalized = self._normalize_vector(vector)
        metadata_payload = dict(metadata or {})
        metadata_payload.setdefault("file_id", safe_id)
        metadata_payload.setdefault("vector_dim", int(normalized.shape[0]))

        with self._locked_file("a") as h5:
            import h5py

            vectors_group = h5.require_group(HDF5_VECTOR_GROUP)
            metadata_group = h5.require_group(HDF5_METADATA_GROUP)
            self._validate_dimension(h5, normalized.shape[0])

            if safe_id in vectors_group:
                if not overwrite:
                    raise ValueError(f"Embedding already exists for id={safe_id}")
                del vectors_group[safe_id]
            if safe_id in metadata_group:
                del metadata_group[safe_id]

            vectors_group.create_dataset(safe_id, data=normalized, dtype="float32")
            metadata_group.create_dataset(
                safe_id,
                data=json.dumps(metadata_payload, ensure_ascii=False),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            h5.flush()
        logger.info("embedding-saved file_id=%s dim=%s", safe_id, normalized.shape[0])

    def get_embedding(self, file_id: str) -> np.ndarray | None:
        safe_id = self._validate_file_id(file_id)
        try:
            with self._locked_file("r") as h5:
                vectors_group = h5.get(HDF5_VECTOR_GROUP)
                if vectors_group is None or safe_id not in vectors_group:
                    return None
                vector = np.asarray(vectors_group[safe_id][()], dtype=np.float32)
                if vector.ndim != 1:
                    logger.warning("embedding-corrupt-non1d file_id=%s", safe_id)
                    return None
                expected_dim = int(h5.attrs.get("vector_dim", vector.shape[0]))
                if self.enforce_dimension and vector.shape[0] != expected_dim:
                    logger.warning(
                        "embedding-corrupt-dimension file_id=%s expected=%s actual=%s",
                        safe_id,
                        expected_dim,
                        vector.shape[0],
                    )
                    return None
                return vector
        except Exception as error:
            logger.warning("embedding-read-failed file_id=%s error=%s", safe_id, error)
            return None

    def has_embedding(self, file_id: str) -> bool:
        return self.get_embedding(file_id) is not None

    def delete_embedding(self, file_id: str) -> bool:
        safe_id = self._validate_file_id(file_id)
        deleted = False
        with self._locked_file("a") as h5:
            vectors_group = h5.require_group(HDF5_VECTOR_GROUP)
            metadata_group = h5.require_group(HDF5_METADATA_GROUP)
            if safe_id in vectors_group:
                del vectors_group[safe_id]
                deleted = True
            if safe_id in metadata_group:
                del metadata_group[safe_id]
                deleted = True or deleted
            if deleted:
                h5.flush()
        if deleted:
            logger.info("embedding-deleted file_id=%s", safe_id)
        return deleted

    def list_ids(self) -> list[str]:
        with self._locked_file("r") as h5:
            vectors_group = h5.get(HDF5_VECTOR_GROUP)
            if vectors_group is None:
                return []
            return sorted(str(key) for key in vectors_group.keys())

    def get_metadata(self, file_id: str) -> dict[str, Any] | None:
        safe_id = self._validate_file_id(file_id)
        try:
            with self._locked_file("r") as h5:
                metadata_group = h5.get(HDF5_METADATA_GROUP)
                if metadata_group is None or safe_id not in metadata_group:
                    return None
                raw = metadata_group[safe_id][()]
                if isinstance(raw, bytes):
                    raw_text = raw.decode("utf-8")
                else:
                    raw_text = raw.tobytes().decode("utf-8") if hasattr(raw, "tobytes") else str(raw)
                return json.loads(raw_text)
        except json.JSONDecodeError as error:
            logger.warning("embedding-metadata-corrupt file_id=%s error=%s", safe_id, error)
            return None
        except Exception as error:
            logger.warning("embedding-metadata-read-failed file_id=%s error=%s", safe_id, error)
            return None

    def get_stats(self) -> dict[str, Any]:
        ids = self.list_ids()
        vector_dim = 0
        if ids:
            first = self.get_embedding(ids[0])
            if first is not None:
                vector_dim = int(first.shape[0])
        return {
            "path": str(self.path),
            "entries": len(ids),
            "vector_dim": vector_dim,
        }

    def clear(self) -> int:
        ids = self.list_ids()
        if not ids:
            return 0
        with self._locked_file("a") as h5:
            if HDF5_VECTOR_GROUP in h5:
                del h5[HDF5_VECTOR_GROUP]
            if HDF5_METADATA_GROUP in h5:
                del h5[HDF5_METADATA_GROUP]
            h5.require_group(HDF5_VECTOR_GROUP)
            h5.require_group(HDF5_METADATA_GROUP)
            h5.flush()
        logger.info("embedding-repository-cleared count=%s", len(ids))
        return len(ids)

    def build_storage_id(self, file_hash: str, cache_key: str) -> str:
        cache_digest = compute_raw_text_hash(cache_key)
        if file_hash.strip():
            return f"{file_hash.strip()}_{cache_digest}"
        return f"adhoc_{cache_digest}"

    @contextmanager
    def _locked_file(self, mode: str) -> Iterator[Any]:
        with self._local_lock, self._file_lock:
            import h5py

            self.path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(self.path, mode) as h5:
                if mode in {"a", "w", "r+"}:
                    h5.require_group(HDF5_VECTOR_GROUP)
                    h5.require_group(HDF5_METADATA_GROUP)
                yield h5

    def _ensure_container(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self._locked_file("a") as h5:
                h5.attrs.setdefault("format_version", "1")
                h5.attrs.setdefault("vector_dim", 0)
                h5.flush()

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        normalized = np.asarray(vector, dtype=np.float32).reshape(-1)
        if normalized.ndim != 1 or normalized.size == 0:
            raise ValueError("Embedding vector must be a non-empty 1D array.")
        return normalized

    def _validate_dimension(self, h5: Any, vector_dim: int) -> None:
        if not self.enforce_dimension:
            return
        stored_dim = int(h5.attrs.get("vector_dim", 0))
        if stored_dim == 0:
            h5.attrs["vector_dim"] = int(vector_dim)
            return
        if stored_dim != int(vector_dim):
            raise ValueError(f"Embedding dimension mismatch: expected {stored_dim}, got {vector_dim}")

    def _validate_file_id(self, file_id: str) -> str:
        safe_id = str(file_id).strip()
        if not safe_id:
            raise ValueError("file_id must not be empty.")
        if "/" in safe_id:
            raise ValueError("file_id must not contain '/'.")
        return safe_id


def create_embedding_repository(config: AppConfig | None) -> EmbeddingRepository | None:
    if config is None or not config.embedding.enabled:
        return None
    if config.embedding.backend.lower() != "hdf5":
        return None
    return EmbeddingRepository(
        config.embedding.path,
        lock_timeout_seconds=config.embedding.lock_timeout_seconds,
    )


def migrate_sqlite_embedding_cache_to_hdf5(
    repository: "ClassificationRepository",
    embedding_repository: EmbeddingRepository,
    *,
    clear_target_first: bool = False,
) -> dict[str, int]:
    if clear_target_first:
        embedding_repository.clear()

    migrated = 0
    skipped = 0
    failed = 0
    for row in repository.fetch_legacy_embedding_cache_entries():
        try:
            cache_key = str(row["cache_key"])
            file_hash = str(row["file_hash"] or "")
            file_id = embedding_repository.build_storage_id(file_hash, cache_key)
            vector = np.asarray(json.loads(row["embedding_json"]), dtype=np.float32)
            metadata = {
                "cache_key": cache_key,
                "file_hash": file_hash,
                "model_name": str(row["model_name"]),
                "text_signature": str(row["text_signature"]),
                "embedding_version": str(row["embedding_version"]),
                "text_kind": str(row["text_kind"]),
                "legacy_hit_count": int(row["hit_count"]),
                "migrated_from": "sqlite_embedding_cache",
            }
            embedding_repository.save_embedding(file_id, vector, metadata, overwrite=True)
            migrated += 1
        except Exception as error:
            logger.warning("embedding-migration-failed cache_key=%s error=%s", row["cache_key"], error)
            failed += 1
    return {"migrated": migrated, "skipped": skipped, "failed": failed}
