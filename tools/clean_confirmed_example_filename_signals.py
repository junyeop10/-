from __future__ import annotations

import argparse
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path


FILENAME_SIGNAL_PREFIXES = (
    "filename:",
    "file_name:",
    "file path:",
    "file_path:",
    "path:",
    "filename_tokens:",
    "file_tokens:",
)


def strip_filename_signal_lines(text: str) -> str:
    kept_lines: list[str] = []
    for line in text.splitlines():
        normalized = line.strip().lower()
        if any(normalized.startswith(prefix) for prefix in FILENAME_SIGNAL_PREFIXES):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines).strip()


def clean_confirmed_examples(db_path: Path, *, dry_run: bool = False) -> dict[str, object]:
    db_path = db_path.resolve()
    backup_path = None
    updates: list[tuple[str, int]] = []

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT id, source_text FROM confirmed_examples ORDER BY id"
        ).fetchall()
        for row in rows:
            original = str(row["source_text"] or "")
            cleaned = strip_filename_signal_lines(original)
            if cleaned != original:
                updates.append((cleaned, int(row["id"])))

        if updates and not dry_run:
            backup_dir = db_path.parent / "snapshots"
            backup_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"{db_path.stem}_before_confirmed_examples_clean_{timestamp}{db_path.suffix}"
            shutil.copy2(db_path, backup_path)
            connection.executemany(
                """
                UPDATE confirmed_examples
                SET source_text = ?, embedding_json = '', embedding_key = ''
                WHERE id = ?
                """,
                updates,
            )

    return {
        "db_path": str(db_path),
        "total_rows": len(rows),
        "updated_rows": len(updates),
        "updated_ids_preview": [row_id for _, row_id in updates[:20]],
        "backup_path": str(backup_path) if backup_path else "",
        "dry_run": dry_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove filename-derived signal lines from confirmed_examples.source_text."
    )
    parser.add_argument("--db", default="data/classifier.db", help="SQLite DB path")
    parser.add_argument("--dry-run", action="store_true", help="Only show affected rows")
    args = parser.parse_args()

    summary = clean_confirmed_examples(Path(args.db), dry_run=args.dry_run)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
