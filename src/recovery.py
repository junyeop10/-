"""Recovery and snapshot helpers."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import AppConfig
from src.storage import ClassificationRepository


def create_safety_snapshot(
    repository: ClassificationRepository,
    config: AppConfig,
    reason: str,
) -> dict[str, Any]:
    snapshot_dir = Path(config.filesystem.snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_id = f"snapshot_{timestamp}"
    db_source = repository.db_path
    db_target = snapshot_dir / f"{snapshot_id}_classifier.db"
    if db_source.exists():
        shutil.copy2(db_source, db_target)

    config_source = Path(config.taxonomy_path)
    copied_config_paths: list[str] = []
    for source in (Path("data/app_config.json"), config_source):
        if source.exists():
            target = snapshot_dir / f"{snapshot_id}_{source.name}"
            shutil.copy2(source, target)
            copied_config_paths.append(str(target))

    manifest = {
        "snapshot_id": snapshot_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "reason": reason,
        "database_snapshot": str(db_target) if db_target.exists() else "",
        "copied_files": copied_config_paths,
    }
    manifest_path = snapshot_dir / f"{snapshot_id}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    repository.record_snapshot(snapshot_id=snapshot_id, snapshot_type="safety", manifest=manifest, reason=reason)
    repository.record_operation(
        operation_type="snapshot",
        status="created",
        details={"snapshot_id": snapshot_id, "reason": reason, "manifest_path": str(manifest_path)},
    )
    return manifest
