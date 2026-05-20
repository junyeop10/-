# HDF5 Embedding Storage Design

## Goal

Move persistent embedding vector storage out of SQLite and into HDF5 while keeping SQLite for relational metadata and operational state.

## Target split

- SQLite
  - file path
  - classification result
  - feedback log
  - OCR usage
  - confidence
  - move/recovery metadata
  - legacy embedding cache during transition
  - confirmed example lookup metadata and fallback JSON during transition

- HDF5
  - embedding vectors
  - vector-side metadata for cache validation
  - confirmed example vectors

## Current implementation direction

The new HDF5 repository lives in:

- [src/embedding_repository.py](C:/Users/jyok3/OneDrive/바탕 화면/파일분류/src/embedding_repository.py)

It provides:

- `save_embedding(...)`
- `get_embedding(...)`
- `has_embedding(...)`
- `delete_embedding(...)`
- `list_ids()`
- `get_metadata(...)`
- `clear()`
- `get_stats()`

## Storage model

HDF5 layout:

```text
embeddings.h5
├── /vectors/
│   ├── <file_id>
│   ├── <file_id>
│   └── ...
└── /metadata/
    ├── <file_id>
    ├── <file_id>
    └── ...
```

- vectors are stored as `float32`
- metadata is stored as JSON string payloads
- overwrite is supported
- missing IDs return `None`

## file_id strategy

Primary rule:

- if `xxhash64` exists, use it as `file_id`

Fallback rule:

- if no stable file hash is available, use a deterministic derived id from the cache key

This keeps duplicate detection and embedding reuse aligned around the same content identity.

## Cache validation strategy

Even when `xxhash64` is reused as the main id, the metadata still records:

- `cache_key`
- `file_hash`
- `model_name`
- `text_signature`
- `embedding_version`
- `text_kind`

That means a stored vector is only reused if:

- same model
- same normalized evidence text signature
- same embedding version
- same text kind

So the HDF5 store behaves like a persistent cache, not just a raw vector dump.

## SQLite integration changes

Minimal-change integration was chosen.

Current status:

- the classifier and vectorizer can now use HDF5 as the primary persistent cache
- legacy SQLite cache is still readable during migration
- optional migration-on-read is supported
- legacy rebuild/stat/clear flows are still present
- confirmed examples can also store vectors in HDF5 while SQLite keeps metadata and lookup keys

Files touched for integration:

- [src/vectorizer.py](C:/Users/jyok3/OneDrive/바탕 화면/파일분류/src/vectorizer.py)
- [src/config.py](C:/Users/jyok3/OneDrive/바탕 화면/파일분류/src/config.py)
- [src/cli.py](C:/Users/jyok3/OneDrive/바탕 화면/파일분류/src/cli.py)
- [src/gui.py](C:/Users/jyok3/OneDrive/바탕 화면/파일분류/src/gui.py)
- [src/storage.py](C:/Users/jyok3/OneDrive/바탕 화면/파일분류/src/storage.py)
- [src/embedding_repository.py](C:/Users/jyok3/OneDrive/바탕 화면/파일분류/src/embedding_repository.py)

## Confirmed example storage refactor

Previous state:

- `confirmed_examples.embedding_json` stored the full vector directly in SQLite

Current transition state:

- new confirmed examples save the vector to HDF5
- SQLite stores `embedding_key` for lookup
- `embedding_json` is left empty for newly written confirmed examples
- old rows with `embedding_json` still work through fallback

Read path:

1. try HDF5 using `embedding_key`
2. if missing, read legacy `embedding_json`
3. if fallback succeeds, optionally backfill to HDF5

Write path:

1. insert confirmed example metadata row
2. create `confirmed_example_<id>` lookup key
3. save vector to HDF5
4. update SQLite row with `embedding_key`

Migration command:

- `app.py migrate_confirmed_example_embeddings`

Optional cleanup:

- `app.py migrate_confirmed_example_embeddings --prune-legacy-json`

## Migration strategy

Recommended phased migration:

1. keep SQLite cache table as read-compatible legacy store
2. write new embeddings to HDF5
3. reuse legacy SQLite entries when needed
4. migrate legacy entries into HDF5 with a dedicated command
5. verify HDF5 hit rate in normal operation
6. eventually disable SQLite legacy cache reads

CLI support:

- `app.py migrate_embedding_cache`
- `app.py migrate_confirmed_example_embeddings`
- `app.py embedding_cache_stats`
- `app.py clear_embedding_cache`
- `app.py rebuild_embedding_cache --clear-first`

## Why this is low-risk

- existing duplicate detection is preserved
- existing SQLite schema does not need destructive changes
- HDF5 repository is injected, not hard-wired across unrelated modules
- cache migration can happen gradually

## Future expansion path

This structure is meant to support later ANN search.

Natural next steps:

1. keep HDF5 as the source of truth for vectors
2. add a separate indexing layer
3. build FAISS index from `list_ids() + get_embedding(...)`
4. store FAISS index metadata separately from both SQLite and HDF5

Recommended future split:

- SQLite: relational metadata and audit history
- HDF5: vector store
- FAISS: retrieval index

## Locking and concurrency

Current realistic strategy:

- lazy open
- file lock around HDF5 access
- conservative single-writer style
- safe enough for current desktop/CLI workflow

Why this is acceptable now:

- current scale is small
- the workload is mostly local and sequential or lightly parallel
- correctness is more important than maximizing read throughput

Possible future improvement:

- SWMR-oriented read strategy
- background index builder
- batched write queue for heavy parallel ingestion

## Directory structure suggestion

```text
data/
  classifier.db
  embeddings.h5
  snapshots/
  manifests/
src/
  embedding_repository.py
  vectorizer.py
  storage.py
  classifier.py
tests/
  test_embedding_repository.py
  test_embedding_cache.py
docs/
  embedding-storage.md
```

## Example usage

```python
import numpy as np

from src.embedding_repository import EmbeddingRepository

repo = EmbeddingRepository("data/embeddings.h5")

repo.save_embedding(
    file_id="abc123",
    vector=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
    metadata={"category": "계약서", "confirmed": True},
)

vector = repo.get_embedding("abc123")
metadata = repo.get_metadata("abc123")
exists = repo.has_embedding("abc123")
```
