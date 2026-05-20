# Recovery and Rollback Guide

## Safety Snapshot

Create a manual snapshot at any time:

```powershell
.\.venv\Scripts\python.exe app.py create_snapshot --reason "before_experiment"
```

Snapshots copy the SQLite database and active config files into `data/snapshots/` and record the action in the database journal.

## Move Recovery

Preview a move plan:

```powershell
.\.venv\Scripts\python.exe app.py preview_move
```

Commit a staged batch:

```powershell
.\.venv\Scripts\python.exe app.py commit_move --batch-id 1
```

Undo the latest committed move:

```powershell
.\.venv\Scripts\python.exe app.py undo_last_move
```

Restore a specific batch:

```powershell
.\.venv\Scripts\python.exe app.py restore_batch --batch-id 1
```

Restore an individual move item:

```powershell
.\.venv\Scripts\python.exe app.py restore_file --move-item-id 3
```

## Feedback Learning Recovery

Export feedback logs:

```powershell
.\.venv\Scripts\python.exe app.py export_feedback_logs
```

Delete one feedback log:

```powershell
.\.venv\Scripts\python.exe app.py delete_feedback_log --feedback-log-id 10
```

Rebuild adaptive learning after changes:

```powershell
.\.venv\Scripts\python.exe app.py rebuild_feedback_learning
```

## Config Rollback

- `data/app_config.json` is versioned in the DB via `config_versions`.
- Snapshots preserve the on-disk config files.
- To roll back, restore the desired config file from `data/snapshots/` and rerun the affected command.
