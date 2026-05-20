# Feedback Learning Guide

## What is Stored

Each feedback log stores:

- original prediction
- corrected category
- predicted/final hierarchy
- evidence snapshot
- source score summary
- metadata and note
- classifier/config versions
- OCR and LLM participation flags

## Commands

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

Delete all logs:

```powershell
.\.venv\Scripts\python.exe app.py clear_feedback_logs
```

Export logs:

```powershell
.\.venv\Scripts\python.exe app.py export_feedback_logs --output-path data/feedback_export.json
```

Rebuild adaptive boosts:

```powershell
.\.venv\Scripts\python.exe app.py rebuild_feedback_learning
```

## Safety Model

- Adaptive boosts are derived from retained logs only.
- Deleting logs removes them from future rebuilds.
- Hand-authored rules are not rewritten silently.
- Current learning remains incremental and explainable rather than autonomous online training.
