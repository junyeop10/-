# Developer Progress Report

## What Changed

- Added versioned app config and hierarchical taxonomy loading.
- Upgraded classification results to carry hierarchy, per-source scores, and explanation payloads.
- Added additive SQLite migration support for snapshots, move journals, adaptive learning, OCR cache, and LLM audit records.
- Added preview/commit/undo/restore move services and CLI commands.
- Added feedback log management and adaptive learning rebuild commands.
- Added recovery snapshot support and operator/developer documentation.
- Added persistent embedding cache reuse metadata and cache observability improvements.
- Expanded the GUI with category-tree browsing, drag-and-drop recategorization, move/history/admin panels, and embedding readiness gating.
- Added startup and per-file classification performance profiling with visible bottleneck reasons.
- Fixed feedback log deletion so linked confirmed-example rows are removed safely.
- Preserved expanded category state during GUI recategorization refreshes and added drag source/target highlighting.

## Why It Changed

- To move from a flat-label prototype toward a reversible, enterprise-style document pipeline.
- To keep learning transparent and rebuildable.
- To make file operations recoverable and auditable.
- To support future provider swaps and taxonomy growth without destructive rewrites.
- To make GUI-based review and correction practical without dropping users back to CLI for common workflows.
- To make latency visible enough to tune OCR, extraction, and embedding behavior with evidence instead of guesswork.

## Files Modified

- `src/config.py`
- `src/taxonomy.py`
- `src/models.py`
- `src/adaptive.py`
- `src/recovery.py`
- `src/operations.py`
- `src/performance.py`
- `src/storage.py`
- `src/classifier.py`
- `src/llm_support.py`
- `src/rule_classifier.py`
- `src/cli.py`
- `src/gui.py`
- `src/vectorizer.py`
- `app_gui.py`
- `data/categories.json`
- `data/app_config.json`
- `tests/test_embedding_cache.py`
- `tests/test_llm_support.py`
- `tests/test_enterprise_ops.py`
- `tests/test_gui_embedding_state.py`
- `tests/test_gui_grouping.py`
- `tests/test_performance_analysis.py`
- docs added under `docs/`

## Tests Performed

- `.\.venv\Scripts\python.exe -m compileall app.py src tests`
- `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`
- `.\.venv\Scripts\python.exe -m compileall app_gui.py src tests`
- GUI interaction changes were verified with targeted tests for grouping, embedding gating, enterprise operations, embedding cache metadata, and performance analysis.

## Remaining Risks

- GUI now exposes the main review/admin workflows, but some advanced recovery operations are still richer in CLI form.
- OCR provider abstraction is config-ready, but only RapidOCR is fully exercised.
- Hosted LLM provider integrations depend on credentials and endpoint availability.
- Performance guidance is diagnostic rather than auto-optimizing; operators still need to choose the next tuning step.

## Rollback Method

1. Use `create_snapshot` or restore files from `data/snapshots/`.
2. Use `undo_last_move` or `restore_batch` for file movement recovery.
3. Restore `data/app_config.json` and `data/categories.json` from a snapshot if config rollback is needed.
4. Rebuild adaptive learning after feedback deletions or restores.
5. Clear or rebuild embedding cache if model/version changes need a clean baseline.
