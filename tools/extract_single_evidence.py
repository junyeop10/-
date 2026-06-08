"""Extract evidence JSON for one document file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evidence_pipeline import build_document_evidence


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract evidence JSON for a single document.")
    parser.add_argument("file_path", help="Document path to inspect")
    parser.add_argument("--output", default="outputs/single_evidence.json", help="Output JSON path")
    parser.add_argument("--min-text-chars", type=int, default=80, help="Minimum text length before OCR fallback")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR fallback")
    args = parser.parse_args()

    file_path = Path(args.file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Input path is not a file: {file_path}")

    evidence = build_document_evidence(
        file_path,
        min_text_chars=max(0, args.min_text_chars),
        ocr_enabled=not args.no_ocr,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved: {output_path}")
    print(f"filename: {evidence.get('filename', '')}")
    print(f"extraction_status: {evidence.get('extraction_status', '')}")
    print(f"extracted_text_length: {evidence.get('extracted_text_length', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
