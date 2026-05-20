# Presentation Summary

## 1. Project goal

This project is a safety-first document classification MVP for office files and scanned PDFs.

The main goal is:

- read mixed business documents automatically
- classify them with explainable evidence
- keep user corrections as reusable learning signals
- move files only through preview-first and reversible workflows

## 2. End-to-end pipeline

The current pipeline is:

1. discover supported files
2. extract evidence text from each file
3. compute `xxhash64` for duplicate detection
4. run rule-based scoring first
5. add embedding similarity when needed
6. add OCR only when text extraction is insufficient
7. optionally call a local LLM only for ambiguous cases
8. persist explanation, evidence, and performance data
9. let the user review, confirm, correct, or move files safely

## 3. Operator experience in the GUI

The desktop UI is no longer just a simple run button.

It now supports:

- background embedding warmup before classification becomes available
- category-folder style browsing after classification
- expanding a category to inspect the files inside it
- dragging one classified file onto another category folder to recategorize it
- move preview/history panels
- feedback log management
- embedding cache management
- performance analysis windows

Important safety behavior:

- classification buttons stay disabled until embedding readiness is confirmed
- actual filesystem movement is still preview-first and commit-based
- drag-and-drop in the result tree changes the classification result and records feedback, not an immediate destructive disk move

## 4. How each file type is read

- `txt`
  - Read directly from disk.
  - Encoding fallback: `utf-8` first, then `cp949`.
  - The text is sampled into a shorter evidence block for classification.

- `pdf`
  - Read with `PyMuPDF`.
  - In fast mode, only selected pages from the front, middle, and end are sampled.
  - If the extracted text is empty or too short, OCR fallback can run.

- `docx`
  - Read with `python-docx`.
  - Paragraph text and table cell text are both collected.

- `xlsx`
  - Read with `openpyxl`.
  - Sheet names and top rows are collected.
  - The design intentionally reads structural hints rather than the whole workbook.

- `pptx`
  - Read with `python-pptx`.
  - Slide text is collected from visible shapes.

## 5. Rule-based classification

Rule-based classification is the first and most important stage.

It uses:

- seeded category keywords from `data/categories.json`
- context rules in `src/rule_classifier.py`
- filename hints for strong document names
- metadata signals such as file extension and obvious tokens

This stage is explainable because the system keeps:

- matched keywords
- matched context rules
- source-by-source score contributions

## 6. Embedding stage

The embedding model is:

- `sentence-transformers`
- model name: `paraphrase-multilingual-MiniLM-L12-v2`

The embedding stage is used when rule signals are not already strong enough.

How it works:

- the system encodes the current document evidence text
- it compares that vector against previously confirmed examples
- it aggregates top similarities by category
- the similarity score becomes one of the inputs to the final decision

## 7. Persistent embedding cache

The embedding cache was added to reduce repeated model inference cost.

How it works:

- a cache key is built from:
  - file hash
  - model name
  - normalized text signature
  - embedding version
  - text kind such as `query` or `evidence`
- the embedding vector is stored in SQLite `embedding_cache`
- when the same text/model combination appears again, the cached vector is reused
- cache hits also increase a `hit_count`, so reuse can be measured later

Why this matters:

- repeated classification runs become faster
- repeated feedback/example reuse becomes cheaper
- the system avoids unnecessary embedding recomputation for identical content
- cache-hit metadata can now be surfaced in timing analysis so operators know whether latency came from fresh inference or reuse

Supporting commands:

- `app.py embedding_cache_stats`
- `app.py clear_embedding_cache`
- `app.py rebuild_embedding_cache --clear-first`

## 8. OCR fallback design

OCR is not the default path.

The system only runs OCR when:

- the file is a PDF
- text extraction is empty or too short
- filename hints are not already strong enough

OCR details:

- backend: `rapidocr_onnxruntime`
- rendered through `PyMuPDF`
- limited to up to 5 pages
- process-local lazy singleton reuse for OCR engine initialization
- optional OCR caching through SQLite

This means the system avoids paying OCR cost for every PDF.

## 9. Local LLM support

The local LLM is:

- provider: `Ollama`
- model: `qwen2.5:3b`

It is not used for every file.

It is only called when the current result is ambiguous enough to justify extra cost.

This keeps:

- speed under control
- privacy local
- rule-based interpretability as the default behavior

## 10. Duplicate detection with xxhash64

Yes, `xxhash64` duplicate detection is active right now.

How it works:

- every file is hashed with `compute_xxhash64(...)`
- the hash is stored in the `files.xxhash64` column
- before inserting a new file result, the repository checks whether another file already has the same hash
- if it does, `duplicate_of_file_id` is recorded
- that duplicate history contributes a small duplicate score during classification

Why this matters:

- identical files do not behave like completely unrelated new files
- corrected or confirmed duplicate history can help later decisions
- duplicate lookup is lightweight compared with rereading the whole content

Where it is active:

- CLI classification flow
- fast worker path
- GUI classification flow

Duplicate-aware organization behavior:

- when safe move planning runs, files with the same `xxhash64` can be grouped together
- the system creates a subfolder inside the category folder using the representative file name stem
- this makes same-content files with different names appear together as one version group

## 11. Feedback learning

The system stores user confirmation/correction logs.

Those logs are used to:

- keep an auditable correction history
- save confirmed examples
- rebuild adaptive token boosts later

Important design choice:

- adaptive learning is rebuildable
- it does not silently mutate the original seed rules

This makes the system safer for enterprise-style operation.

Additional feedback-log behavior:

- deleting a feedback log also removes linked confirmed-example rows
- rebuilding adaptive learning after deletions keeps future classification effects explainable

## 12. Safe operations and recovery

This project does not move files by default.

Instead it uses:

- `preview_move`
- `commit_move`
- `undo_last_move`
- `restore_batch`
- `restore_file`

Recovery support includes:

- move journals
- operation logs
- config version snapshots
- database snapshots

This makes the system more suitable for real business folders where mistaken file movement is costly.

## 13. Performance observability

The system now measures both startup latency and file-level classification latency.

Captured startup stages include:

- configuration load
- taxonomy load
- database initialization
- rule loading
- UI build
- embedding warmup/readiness

Captured file-level stages include:

- text extraction
- normalization
- OCR decision and OCR runtime
- duplicate lookup
- classification time
- embedding involvement
- database persistence

The system also explains why a file was relatively slow or fast, for example:

- OCR ran and scanned multiple pages
- embedding inference ran instead of hitting cache
- strong rules allowed embedding to be skipped
- large extracted text increased parsing/scoring cost
- ambiguity required extra verification

This is useful for deciding where to optimize next.

## 14. Suggested presentation phrasing

You can explain the project like this:

"This system starts with explainable rule-based classification, then selectively adds embeddings, OCR, and a local LLM only when needed. We added persistent embedding caching, duplicate detection through xxhash64, GUI-based category review with drag-and-drop correction, and preview-first file operations so the system stays fast, transparent, and safe enough for real business document workflows."
