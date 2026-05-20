# Architecture Overview

## Summary

This project now uses a safety-first enterprise MVP architecture built around four pillars:

- `classification v2`: hierarchical, explainable hybrid scoring
- `operations v1`: preview, commit, undo, restore, and move journaling
- `learning v1`: transparent feedback logs and rebuildable adaptive boosts
- `recovery v1`: schema migrations, snapshots, config versioning, and operation journals

## Runtime Layers

- `src/config.py`
  Loads versioned application settings for scoring, OCR, LLM routing, and filesystem policies.
- `src/taxonomy.py`
  Loads hierarchical category definitions and keeps backward compatibility with legacy flat labels.
- `src/classifier.py`
  Produces hierarchical classification results with separate source scores, explanation payloads, and per-file processing profiles.
- `src/storage.py`
  Owns SQLite schema initialization, additive migrations, feedback history, OCR cache, embedding cache, move journals, and snapshots.
- `src/operations.py`
  Implements preview-first file movement, commit, undo, restore, and conflict-safe path handling.
- `src/adaptive.py`
  Rebuilds safe feedback-derived boosts from retained logs only.
- `src/recovery.py`
  Creates safety snapshots for database and config state before risky operations.
- `src/performance.py`
  Normalizes startup/per-file timings and builds human-readable latency explanations for GUI and CLI.
- `src/gui.py`
  Provides interactive classification review, category-tree browsing, drag-and-drop recategorization, feedback/admin panels, and performance inspection.
- `src/vectorizer.py`
  Handles sentence-transformer loading, persistent embedding cache lookup/storage, and cache-hit metadata for observability.

## Data Model Additions

- `classifications`
  Now stores `large_category`, `middle_category`, `small_category`, per-level confidences, explanation JSON, source score JSON, evidence JSON, performance JSON, and version metadata.
- `feedback_logs`
  Now stores predicted/final hierarchy, evidence snapshot, metadata, source scores, and OCR/LLM participation flags.
- `confirmed_examples`
  Continues to store reusable example embeddings and is now cleaned up automatically when linked feedback logs are deleted.
- `move_batches`, `move_items`
  Record staged and committed move operations with reversible restore metadata.
- `snapshots`, `operation_journal`
  Preserve recovery state and operator-auditable history.
- `adaptive_rule_boosts`
  Stores rebuildable feedback-derived boosts without mutating hand-authored rules.
- `ocr_cache`, `embedding_cache`, `llm_audit`, `config_versions`, `schema_migrations`
  Support observability, caching, and rollback.

## GUI Interaction Layer

- Classification in the GUI is blocked until embedding readiness reaches `ready`.
- Results are rendered as category folders with child file nodes instead of a flat list.
- Users can drag a file node onto another category node to recategorize it interactively.
- The GUI preserves open category folders during tree refreshes and highlights drag source/target rows for visibility.
- Feedback log management, move history, move preview, embedding cache management, and performance analysis are available from the GUI side panel.

## Performance Observability

- Startup timing is captured from GUI boot through configuration load, DB init, rule load, UI build, and embedding warmup.
- File-level timing captures extraction, normalization, OCR decisions, OCR runtime, duplicate lookup, DB writes, classifier time, and persistence.
- Latency explanations identify likely causes such as OCR execution, embedding inference, cache reuse, strong-rule embedding bypass, large text payloads, and review-path ambiguity.
- CLI and GUI reuse the same performance-analysis helpers so operators see a consistent explanation model.

## Safety Defaults

- File organization is `preview first`.
- No file movement occurs without `commit_move`.
- Restore operations never overwrite existing files silently.
- Adaptive learning is rebuildable and removable.
- Schema evolution is additive and legacy flows remain available.
