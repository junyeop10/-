# Enterprise Document Classification MVP

> **My contribution to this team project:** Claude API pipeline integration (Stage 5) — see [docs/my-contribution-claude-api.md](docs/my-contribution-claude-api.md).

Safety-first document classification and organization pipeline for office files and scanned PDFs.

This project started as a flat-label classifier and is now moving into version 2 as a CPU-first document intelligence system. The target direction is feature extraction, ML type classification, clustering, active learning, and user-feedback-driven improvement without requiring an AI model to read every full document on every run.

The current implementation includes:

- hierarchical classification with `large / middle / small` category support
- hybrid scoring from rules, embeddings, metadata, filename hints, feedback, duplicates, OCR, and optional LLM fallback
- transparent feedback logging and rebuildable adaptive boosts
- preview-first file movement with batch manifests, undo, and restore
- recovery snapshots, operation journals, and additive SQLite migrations
- persistent embedding cache for reusable document/query/evidence embeddings
- GUI category tree interaction, drag-and-drop recategorization, and performance analysis views

## Documentation

- [My Contribution — Claude API Pipeline Integration](docs/my-contribution-claude-api.md)
- [Presentation Summary](docs/presentation-summary.md)
- [Version History](docs/version-history.md)
- [Embedding Storage Design](docs/embedding-storage.md)
- [Architecture](docs/architecture.md)
- [Recovery Guide](docs/recovery-guide.md)
- [Feedback Learning Guide](docs/feedback-learning.md)
- [Operator Safety](docs/operator-safety.md)
- [Developer Progress Report](docs/developer-progress-report.md)
- [Version 2 Document Intelligence Architecture](docs/v2-document-intelligence-architecture.md)

## Supported Inputs

- `txt`
- `pdf`
- `docx`
- `xlsx`
- `pptx`

## Core Safety Defaults

- actual file movement is never the default
- use `preview_move` first
- `commit_move` is required for real relocation
- restore paths are conflict-safe
- feedback-derived learning is rebuildable and removable
- recovery snapshots are available through the CLI
- duplicate detection is active through `xxhash64`
- embedding reuse is supported through a persistent SQLite cache

## Project Layout

```text
backend/          # FastAPI 파일 분류 API (팀 백엔드)
  main.py
  pipeline/
  docs/
src/              # CLI·GUI 문서 분류 (레거시/별도 실행)
app.py
app_gui.py
data/
rules/            # Git 워크플로 (팀 공통)
docs/             # 아키텍처·운영 문서
tests/
```

팀 백엔드 실행·컨벤션: [`backend/README.md`](backend/README.md), [`backend/docs/CONVENTIONS.md`](backend/docs/CONVENTIONS.md)

## Install

Python 3.11+ is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Initialize

```powershell
.\.venv\Scripts\python.exe app.py init-db
```

## Classify Files

Fast mode:

```powershell
.\.venv\Scripts\python.exe app.py classify --fast
```

Fast mode with review:

```powershell
.\.venv\Scripts\python.exe app.py classify --fast --review
```

Sequential mode:

```powershell
.\.venv\Scripts\python.exe app.py classify
```

Ambiguous-only LLM routing:

```powershell
.\.venv\Scripts\python.exe app.py classify --fast --use-llm
```

## Inspect the System

Stats:

```powershell
.\.venv\Scripts\python.exe app.py stats
```

Suggest feedback-derived rules:

```powershell
.\.venv\Scripts\python.exe app.py suggest-rules
```

## Feedback Log Management

List logs:

```powershell
.\.venv\Scripts\python.exe app.py list_feedback_logs
```

Show one log:

```powershell
.\.venv\Scripts\python.exe app.py show_feedback_log --feedback-log-id 1
```

Delete one log:

```powershell
.\.venv\Scripts\python.exe app.py delete_feedback_log --feedback-log-id 1
```

Clear all logs:

```powershell
.\.venv\Scripts\python.exe app.py clear_feedback_logs
```

Export logs:

```powershell
.\.venv\Scripts\python.exe app.py export_feedback_logs --output-path data/feedback_logs_export.json
```

Rebuild adaptive learning:

```powershell
.\.venv\Scripts\python.exe app.py rebuild_feedback_learning
```

## Embedding Cache

Show cache stats:

```powershell
.\.venv\Scripts\python.exe app.py embedding_cache_stats
```

Clear cache:

```powershell
.\.venv\Scripts\python.exe app.py clear_embedding_cache
```

Rebuild cache:

```powershell
.\.venv\Scripts\python.exe app.py rebuild_embedding_cache --clear-first
```

Migrate legacy SQLite cache to HDF5:

```powershell
.\.venv\Scripts\python.exe app.py migrate_embedding_cache
```

Backfill confirmed example embeddings to HDF5:

```powershell
.\.venv\Scripts\python.exe app.py migrate_confirmed_example_embeddings
```

## Safe File Movement

Preview a move plan:

```powershell
.\.venv\Scripts\python.exe app.py preview_move
```

Commit a staged batch:

```powershell
.\.venv\Scripts\python.exe app.py commit_move --batch-id 1
```

Undo the latest move:

```powershell
.\.venv\Scripts\python.exe app.py undo_last_move
```

Restore a batch:

```powershell
.\.venv\Scripts\python.exe app.py restore_batch --batch-id 1
```

Restore a single move item:

```powershell
.\.venv\Scripts\python.exe app.py restore_file --move-item-id 1
```

List move history:

```powershell
.\.venv\Scripts\python.exe app.py list_move_history
```

## Recovery

Create a snapshot:

```powershell
.\.venv\Scripts\python.exe app.py create_snapshot --reason "before_bulk_change"
```

Additional recovery details are documented in:

- [Architecture](docs/architecture.md)
- [Recovery Guide](docs/recovery-guide.md)
- [Feedback Learning Guide](docs/feedback-learning.md)
- [Operator Safety](docs/operator-safety.md)
- [Developer Progress Report](docs/developer-progress-report.md)
- [Presentation Summary](docs/presentation-summary.md)

## GUI

Launch the desktop UI:

```powershell
.\run_gui.bat
```

The GUI supports:

- embedding-readiness gating before classification starts
- category-folder style result browsing
- file drag-and-drop recategorization inside the result tree
- move preview, move history, feedback log management, and embedding cache management windows
- startup and classification performance analysis views

Current interaction notes:

- dragging a classified file onto another category updates the classification result and records feedback
- expanded category folders stay open across in-GUI recategorization refreshes
- feedback log deletion also removes linked confirmed-example rows so deletion works safely
- actual filesystem relocation remains safety-first and still uses the preview/commit workflow

## Testing

```powershell
.\.venv\Scripts\python.exe -m compileall app.py src tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Known Limitations

- RapidOCR is the primary OCR backend in this MVP stage.
- Non-Ollama LLM providers are wired as pluggable adapters but may require credentials or endpoint setup.
- GUI support for move/recovery and feedback administration is not yet as complete as the CLI surface.
