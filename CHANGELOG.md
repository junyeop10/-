# CHANGELOG

This file tracks implementation progress in chronological order.

## 2026-05-20

### Version 2 document intelligence direction

- Documented the version 2 direction as a feature-first document intelligence system.
- Clarified that the project should not depend on full-document AI reading for every run.
- Added the planned split between rule-based signals, ML type classification, active learning, clustering, and optional LLM assistance.
- Captured CPU-first implementation guidance for OCR, embeddings, TF-IDF, clustering, and caching.
- Added future storage concepts for document features, vectors, model runs, category candidates, and multi-tag classification.

## 2026-05-18

### Visible processing-route explanation

- Added a shared processing-route formatter for classification results.
- Results now explicitly show whether the final outcome was handled primarily by rules, embeddings, or LLM refinement.
- OCR is now shown as a visible supporting step rather than being hidden only inside the reason string.
- Updated both CLI and GUI result displays to expose the processing route consistently.

### Documentation alignment for presentation and handoff

- Reorganized the project documentation so each file has a clear role.
- Added a dedicated presentation summary document with architecture, pipeline, model, cache, OCR, and duplicate-detection details.
- Clarified that `xxhash64` duplicate detection is active in both CLI and GUI classification flows.
- Added explicit documentation for the persistent embedding cache design and rebuild commands.

### Duplicate-version grouping in move previews

- Added duplicate-aware move planning based on active `xxhash64` content hashes.
- When multiple files share the same content hash, the move planner now creates a representative subfolder under the category path.
- The representative folder name is based on the earliest file name stem in that duplicate group.
- This lets different filename variants of the same content appear together inside the categorized result tree.

### HDF5 embedding repository groundwork

- Added a new HDF5-backed embedding repository module for persistent vector storage.
- Kept SQLite as the relational metadata store while moving vector persistence toward HDF5.
- Wired the sentence-transformers cache flow so HDF5 can act as the primary persistent embedding cache.
- Added legacy SQLite-to-HDF5 migration support and new cache-management documentation.

### Confirmed example embedding separation

- Refactored `confirmed_examples` so new example vectors are stored in HDF5 instead of SQLite `embedding_json`.
- Added `embedding_key` lookup storage in SQLite for confirmed examples.
- Kept backward-compatible fallback for existing `embedding_json` rows.
- Added backfill/migration support so legacy confirmed example vectors can be moved into HDF5 gradually.
- Extended embedding cache diagnostics to show confirmed-example HDF5 coverage.

## 2026-05-15

### Enterprise MVP upgrade foundation

- Added hierarchical taxonomy support and versioned runtime config.
- Expanded SQLite with additive migration support for recovery snapshots, move journals, OCR cache, adaptive boosts, and richer classification records.
- Added preview-first file movement commands: `preview_move`, `commit_move`, `undo_last_move`, `restore_batch`, `restore_file`, and `list_move_history`.
- Added transparent feedback log management commands and rebuildable adaptive learning.
- Added recovery snapshot creation plus operator and developer documentation.

### OCR fallback and optimization

- Added OCR fallback for scanned PDFs with `RapidOCR`.
- Limited OCR to selected cases only and capped scanning to 5 pages.
- Added OCR caching and OCR decision logging.
- Optimized OCR triggers with filename hints and minimum extracted-text length.

### Local LLM routing

- Added ambiguous-only local LLM fallback through Ollama `qwen2.5:3b`.
- Limited LLM usage to low-confidence cases instead of every file.
- Preserved the existing rule/embedding result if LLM fails.

### Input format expansion

- Added `docx`, `xlsx`, and `pptx` support.
- Added English keyword coverage alongside existing Korean-oriented rules.

### GUI improvements

- Added OCR-used indicators in the GUI result list.
- Switched startup behavior so the main screen can appear first while heavier embedding work loads in the background.

## 2026-05-13

### Repository cleanup and GitHub-ready baseline

- Removed unused external LLM call remnants.
- Cleaned the repository structure for safe GitHub publishing.
- Added and refined repository documentation, ignore rules, and testing support.

### Rule-based classification recovery

- Restored Korean rule/category data.
- Improved keyword and context-rule coverage.
- Added stronger support for certificate and corporate-document categories.
