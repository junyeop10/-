# Operator Safety Notes

## Default Safety Rules

- No file movement without explicit `commit_move`.
- No silent overwrite on move or restore.
- Snapshots are recommended before bulk operations and are created automatically before `commit_move`.
- Recovery history is stored in SQLite journals and snapshot manifests.

## Recommended Workflow

1. Initialize the DB and config.
2. Run classification.
3. Inspect results and feedback logs.
4. Run `preview_move`.
5. Review the generated manifest in `data/manifests/`.
6. Run `commit_move` only when satisfied.
7. If needed, use `undo_last_move` or `restore_batch`.

## Known MVP Boundaries

- GUI still focuses on classification and review; the full move/recovery management surface is CLI-first.
- OCR backend selection is config-ready, but RapidOCR remains the main implementation path in this MVP stage.
- Non-Ollama LLM providers are pluggable, but hosted provider use still depends on available credentials.
