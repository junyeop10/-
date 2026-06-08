"""Safe file move preview, commit, undo, and restore helpers."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import AppConfig
from src.storage import ClassificationRepository


UNCATEGORIZED_MOVE_LABEL = "review_required"


@dataclass
class MovePlanItem:
    file_id: int
    classification_id: int
    source_path: str
    destination_path: str
    large_category: str
    middle_category: str
    small_category: str | None
    confidence: float
    duplicate_group_folder: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def preview_move_plan(
    repository: ClassificationRepository,
    config: AppConfig,
    limit: int | None = None,
) -> dict[str, Any]:
    rows = repository.fetch_latest_classifications(limit=limit)
    return _create_move_plan(repository=repository, config=config, rows=rows)


def preview_move_plan_for_classifications(
    repository: ClassificationRepository,
    config: AppConfig,
    classification_ids: list[int],
) -> dict[str, Any]:
    rows = repository.fetch_classifications_by_ids(classification_ids)
    return _create_move_plan(repository=repository, config=config, rows=rows)


def preview_direct_folder_move_plan(
    repository: ClassificationRepository,
    config: AppConfig,
    payloads: list[dict[str, Any]],
    destination_dir: Path,
    transfer_mode: str = "move",
    cleanup_empty_source_dirs: bool = False,
) -> dict[str, Any]:
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    transfer_mode = "copy" if transfer_mode == "copy" else "move"
    items: list[MovePlanItem] = []
    for payload in payloads:
        source_path = Path(str(payload.get("file_path", "")))
        if not source_path.exists():
            continue
        category = sanitize_folder_name(str(payload.get("category") or "미분류"))
        category_dir = destination_dir / category
        item = MovePlanItem(
            file_id=int(payload["file_id"]),
            classification_id=int(payload["classification_id"]),
            source_path=str(source_path),
            destination_path=str(build_conflict_safe_path(category_dir / source_path.name)),
            large_category="manual",
            middle_category=category,
            small_category=None,
            confidence=float(payload.get("confidence", 0.0)),
            duplicate_group_folder=None,
        )
        items.append(item)

    batch_id = repository.create_move_batch(
        operation_mode="direct_folder_preview",
        status="staged",
        manifest_json=json.dumps(
            {
                "destination_dir": str(destination_dir),
                "transfer_mode": transfer_mode,
                "cleanup_empty_source_dirs": bool(cleanup_empty_source_dirs),
                "items": [item.to_dict() for item in items],
            },
            ensure_ascii=False,
        ),
    )
    for item in items:
        repository.add_move_item(batch_id=batch_id, item=item.to_dict(), status="staged")

    manifest_dir = Path(config.filesystem.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"direct_move_preview_{batch_id}.json"
    manifest_path.write_text(
        json.dumps(
            {
                "batch_id": batch_id,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "destination_dir": str(destination_dir),
                "transfer_mode": transfer_mode,
                "cleanup_empty_source_dirs": bool(cleanup_empty_source_dirs),
                "items": [item.to_dict() for item in items],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    repository.record_operation(
        operation_type="direct_folder_preview",
        status="staged",
        batch_id=batch_id,
        details={
            "manifest_path": str(manifest_path),
            "destination_dir": str(destination_dir),
            "transfer_mode": transfer_mode,
            "cleanup_empty_source_dirs": bool(cleanup_empty_source_dirs),
            "count": len(items),
        },
    )
    return {"batch_id": batch_id, "manifest_path": str(manifest_path), "items": [item.to_dict() for item in items]}


def _create_move_plan(
    repository: ClassificationRepository,
    config: AppConfig,
    rows: list[Any],
) -> dict[str, Any]:
    managed_root = Path(config.filesystem.managed_root)
    managed_root.mkdir(parents=True, exist_ok=True)
    items: list[MovePlanItem] = []
    duplicate_folder_cache: dict[str, str | None] = {}

    for row in rows:
        source_path = Path(str(row["file_path"]))
        if not source_path.exists():
            continue

        large_category = str(row["large_category"] or "miscellaneous")
        middle_category = str(row["middle_category"] or UNCATEGORIZED_MOVE_LABEL)
        relative_parts = [large_category, middle_category]
        if row["small_category"]:
            relative_parts.append(str(row["small_category"]))

        file_hash = str(row["xxhash64"] or "")
        duplicate_group_folder = duplicate_folder_cache.get(file_hash)
        if file_hash and file_hash not in duplicate_folder_cache:
            duplicate_group_folder = repository.get_duplicate_group_folder_name(file_hash)
            duplicate_folder_cache[file_hash] = duplicate_group_folder
        if duplicate_group_folder:
            relative_parts.append(duplicate_group_folder)

        destination_dir = managed_root.joinpath(*relative_parts)
        destination_path = build_conflict_safe_path(destination_dir / source_path.name)
        items.append(
            MovePlanItem(
                file_id=int(row["file_id"]),
                classification_id=int(row["classification_id"]),
                source_path=str(source_path),
                destination_path=str(destination_path),
                large_category=large_category,
                middle_category=middle_category,
                small_category=str(row["small_category"]) if row["small_category"] else None,
                confidence=float(row["final_score"]),
                duplicate_group_folder=duplicate_group_folder,
            )
        )

    batch_id = repository.create_move_batch(
        operation_mode="preview",
        status="staged",
        manifest_json=json.dumps([item.to_dict() for item in items], ensure_ascii=False),
    )
    for item in items:
        repository.add_move_item(batch_id=batch_id, item=item.to_dict(), status="staged")

    manifest_dir = Path(config.filesystem.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"move_preview_{batch_id}.json"
    manifest_path.write_text(
        json.dumps(
            {
                "batch_id": batch_id,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "items": [item.to_dict() for item in items],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    repository.record_operation(
        operation_type="preview_move",
        status="staged",
        batch_id=batch_id,
        details={"manifest_path": str(manifest_path), "count": len(items)},
    )
    return {"batch_id": batch_id, "manifest_path": str(manifest_path), "items": [item.to_dict() for item in items]}


def commit_move_batch(repository: ClassificationRepository, batch_id: int) -> dict[str, Any]:
    rows = repository.fetch_move_items(batch_id=batch_id, status="staged")
    batch = repository.fetch_move_batch(batch_id) if hasattr(repository, "fetch_move_batch") else None
    manifest = _load_move_batch_manifest(batch)
    transfer_mode = str(manifest.get("transfer_mode") or "move")
    cleanup_empty_source_dirs = bool(manifest.get("cleanup_empty_source_dirs"))
    source_cleanup_dirs: list[Path] = []
    moved = 0
    failed = 0
    for row in rows:
        source = Path(str(row["source_path"]))
        destination = Path(str(row["destination_path"]))
        try:
            if not source.exists():
                raise FileNotFoundError(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            final_destination = build_conflict_safe_path(destination)
            if transfer_mode == "copy":
                shutil.copy2(str(source), str(final_destination))
            else:
                shutil.move(str(source), str(final_destination))
                if cleanup_empty_source_dirs:
                    source_cleanup_dirs.append(source.parent)
            repository.update_move_item_status(
                move_item_id=int(row["id"]),
                status="committed",
                actual_destination_path=str(final_destination),
            )
            moved += 1
        except Exception as error:
            repository.update_move_item_status(
                move_item_id=int(row["id"]),
                status="failed",
                error_message=str(error),
            )
            failed += 1
    if cleanup_empty_source_dirs and source_cleanup_dirs:
        cleanup_empty_move_dirs(source_cleanup_dirs, _source_cleanup_stop_dirs(source_cleanup_dirs))
    repository.update_move_batch_status(batch_id=batch_id, status="committed" if failed == 0 else "partial_failed")
    repository.record_operation(
        operation_type="commit_move",
        status="completed" if failed == 0 else "partial_failed",
        batch_id=batch_id,
        details={"moved": moved, "failed": failed, "transfer_mode": transfer_mode},
    )
    return {"batch_id": batch_id, "moved": moved, "failed": failed}


def undo_last_move(repository: ClassificationRepository) -> dict[str, Any]:
    batch = repository.fetch_last_committed_move_batch()
    if batch is None:
        return {"batch_id": None, "restored": 0, "failed": 0}
    return restore_batch(repository, int(batch["id"]), restore_to_original=True, status_label="undone")


def restore_batch(
    repository: ClassificationRepository,
    batch_id: int,
    restore_to_original: bool = True,
    status_label: str = "restored",
) -> dict[str, Any]:
    rows = repository.fetch_move_items(batch_id=batch_id)
    batch = repository.fetch_move_batch(batch_id) if hasattr(repository, "fetch_move_batch") else None
    manifest = _load_move_batch_manifest(batch)
    transfer_mode = str(manifest.get("transfer_mode") or "move")
    cleanup_stop_dirs = _move_batch_cleanup_stop_dirs(batch)
    cleanup_start_dirs: list[Path] = []
    restored = 0
    failed = 0
    for row in rows:
        current_path = Path(str(row["actual_destination_path"] or row["destination_path"]))
        target_path = Path(str(row["source_path"] if restore_to_original else row["destination_path"]))
        target_path = build_conflict_safe_path(target_path)
        try:
            if not current_path.exists():
                raise FileNotFoundError(current_path)
            if transfer_mode == "copy" and restore_to_original:
                current_path.unlink()
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(current_path), str(target_path))
            if cleanup_stop_dirs:
                cleanup_start_dirs.append(current_path.parent)
            repository.update_move_item_restore(
                move_item_id=int(row["id"]),
                status=status_label,
                restored_path=str(target_path if transfer_mode != "copy" else row["source_path"]),
            )
            restored += 1
        except Exception as error:
            repository.update_move_item_status(
                move_item_id=int(row["id"]),
                status="restore_failed",
                error_message=str(error),
            )
            failed += 1
    if cleanup_stop_dirs:
        cleanup_empty_move_dirs(cleanup_start_dirs, cleanup_stop_dirs)
    repository.record_operation(
        operation_type="restore_batch",
        status="completed" if failed == 0 else "partial_failed",
        batch_id=batch_id,
        details={"restored": restored, "failed": failed, "mode": status_label},
    )
    return {"batch_id": batch_id, "restored": restored, "failed": failed}


def restore_file(repository: ClassificationRepository, move_item_id: int) -> dict[str, Any]:
    row = repository.fetch_move_item(move_item_id)
    if row is None:
        return {"move_item_id": move_item_id, "restored": False, "error": "not_found"}
    result = restore_batch(repository, batch_id=int(row["batch_id"]), restore_to_original=True, status_label="restored")
    return {"move_item_id": move_item_id, **result}


def build_conflict_safe_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    counter = 1
    while True:
        candidate = path.with_name(f"{stem}__conflict_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def sanitize_folder_name(value: str) -> str:
    cleaned = "".join("_" if char in '<>:"/\\|?*' else char for char in value.strip())
    cleaned = cleaned.strip().rstrip(".")
    return cleaned or "미분류"


def _move_batch_cleanup_stop_dirs(batch: Any | None) -> set[Path]:
    manifest = _load_move_batch_manifest(batch)
    destination_dir = str(manifest.get("destination_dir") or "").strip()
    if not destination_dir:
        return set()
    return {Path(destination_dir).resolve().parent}


def _load_move_batch_manifest(batch: Any | None) -> dict[str, Any]:
    if batch is None:
        return {}
    try:
        manifest = json.loads(str(batch["manifest_json"] or "{}"))
    except Exception:
        return {}
    return manifest if isinstance(manifest, dict) else {}


def _source_cleanup_stop_dirs(start_dirs: list[Path]) -> set[Path]:
    resolved = [path.resolve() for path in start_dirs]
    if not resolved:
        return set()
    try:
        common = Path(os.path.commonpath([str(path) for path in resolved]))
    except Exception:
        common = resolved[0].parent
    # Keep the parent of the common source folder; empty source folders below it may be removed.
    return {common.resolve().parent}


def cleanup_empty_move_dirs(start_dirs: list[Path], stop_dirs: set[Path]) -> None:
    resolved_stops = {path.resolve() for path in stop_dirs}
    for start_dir in sorted({path.resolve() for path in start_dirs}, key=lambda path: len(path.parts), reverse=True):
        current = start_dir
        while current not in resolved_stops and current.parent != current:
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent
