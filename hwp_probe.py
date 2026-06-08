"""Probe HWP/HWPX text extraction, structure hints, and keyword candidates.

Usage:
    .\.venv\Scripts\python.exe hwp_probe.py "C:\path\sample.hwpx"
    .\.venv\Scripts\python.exe hwp_probe.py "C:\path\sample.hwp" --json

This is a diagnostic script, not a perfect renderer. It helps check whether a
Hangul document can be read well enough for classification/clustering.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import zipfile
import zlib
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET


TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z0-9]{2,}")
STOPWORDS = {
    "그리고",
    "그러나",
    "또는",
    "대한",
    "관련",
    "문서",
    "파일",
    "내용",
    "입니다",
    "합니다",
    "있는",
    "없는",
    "수신",
    "발신",
    "제목",
    "작성",
    "확인",
    "the",
    "and",
    "for",
    "with",
    "from",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract rough text, structure, and keywords from .hwp/.hwpx files.")
    parser.add_argument("path", help="Path to a .hwp or .hwpx file")
    parser.add_argument("--limit", type=int, default=4500, help="Maximum text sample length")
    parser.add_argument("--top", type=int, default=30, help="Number of keywords to show")
    parser.add_argument("--json", action="store_true", help="Print full result as JSON")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 2

    if path.suffix.lower() == ".hwpx":
        result = probe_hwpx(path, text_limit=args.limit, keyword_top=args.top)
    elif path.suffix.lower() == ".hwp":
        result = probe_hwp(path, text_limit=args.limit, keyword_top=args.top)
    else:
        print("Only .hwp and .hwpx files are supported.", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_report(result)
    return 0


def probe_hwpx(path: Path, *, text_limit: int, keyword_top: int) -> dict:
    texts: list[str] = []
    section_files: list[str] = []
    structure = {
        "format": "hwpx",
        "paragraph_count": 0,
        "run_count": 0,
        "table_count": 0,
        "row_count": 0,
        "cell_count": 0,
        "section_file_count": 0,
        "xml_file_count": 0,
    }

    with zipfile.ZipFile(path) as archive:
        xml_names = [name for name in archive.namelist() if name.lower().endswith(".xml")]
        section_files = [name for name in xml_names if "section" in name.lower()]
        structure["xml_file_count"] = len(xml_names)
        structure["section_file_count"] = len(section_files)

        for name in section_files or xml_names:
            try:
                root = ET.fromstring(archive.read(name))
            except ET.ParseError:
                continue
            for elem in root.iter():
                local = strip_namespace(elem.tag)
                if local == "p":
                    structure["paragraph_count"] += 1
                elif local == "run":
                    structure["run_count"] += 1
                elif local == "tbl":
                    structure["table_count"] += 1
                elif local == "tr":
                    structure["row_count"] += 1
                elif local == "tc":
                    structure["cell_count"] += 1
                elif local == "t" and elem.text:
                    texts.append(elem.text)

    full_text = normalize_extracted_text("\n".join(texts))
    return build_result(path, full_text, structure, section_files, text_limit=text_limit, keyword_top=keyword_top)


def probe_hwp(path: Path, *, text_limit: int, keyword_top: int) -> dict:
    structure = {
        "format": "hwp",
        "paragraph_count": 0,
        "body_stream_count": 0,
        "compressed": None,
        "olefile_available": False,
        "extraction_mode": "utf16_scan_fallback",
    }
    texts: list[str] = []
    stream_names: list[str] = []

    try:
        import olefile  # type: ignore

        structure["olefile_available"] = True
        with olefile.OleFileIO(path) as ole:
            stream_names = ["/".join(item) for item in ole.listdir(streams=True)]
            structure["compressed"] = hwp_is_compressed(ole)
            body_streams = sorted(name for name in stream_names if name.startswith("BodyText/Section"))
            structure["body_stream_count"] = len(body_streams)
            structure["extraction_mode"] = "hwp_body_streams"
            for name in body_streams:
                raw = ole.openstream(name.split("/")).read()
                if structure["compressed"]:
                    raw = zlib.decompress(raw, -15)
                texts.extend(extract_hwp_record_text(raw))
    except ImportError:
        texts.append(extract_utf16_strings(path.read_bytes()))
        structure["warning"] = "Install olefile for better .hwp stream extraction: pip install olefile"
    except Exception as error:
        texts.append(extract_utf16_strings(path.read_bytes()))
        structure["warning"] = f"HWP stream extraction failed; used byte scan fallback: {error}"

    full_text = normalize_extracted_text("\n".join(texts))
    structure["paragraph_count"] = count_paragraphs(full_text)
    return build_result(path, full_text, structure, stream_names, text_limit=text_limit, keyword_top=keyword_top)


def hwp_is_compressed(ole) -> bool:
    try:
        header = ole.openstream("FileHeader").read()
        flags = struct.unpack_from("<I", header, 36)[0]
        return bool(flags & 0x01)
    except Exception:
        return True


def extract_hwp_record_text(data: bytes) -> list[str]:
    """Extract text from HWP records, preferring paragraph text records."""
    texts: list[str] = []
    offset = 0
    while offset + 4 <= len(data):
        header = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        tag_id = header & 0x3FF
        size = (header >> 20) & 0xFFF
        if size == 0xFFF:
            if offset + 4 > len(data):
                break
            size = struct.unpack_from("<I", data, offset)[0]
            offset += 4
        payload = data[offset : offset + size]
        offset += size
        if tag_id == 67:
            text = decode_hwp_para_text(payload)
            if text.strip():
                texts.append(text)
    if texts:
        return texts
    fallback = extract_utf16_strings(data)
    return [fallback] if fallback else []


def decode_hwp_para_text(payload: bytes) -> str:
    text = payload.decode("utf-16le", errors="ignore")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_utf16_strings(data: bytes) -> str:
    decoded = data.decode("utf-16le", errors="ignore")
    chunks = re.findall(r"[가-힣A-Za-z0-9 .,;:()\\[\\]{}<>/%+-]{4,}", decoded)
    return "\n".join(chunk.strip() for chunk in chunks if chunk.strip())


def build_result(
    path: Path,
    full_text: str,
    structure: dict,
    source_parts: list[str],
    *,
    text_limit: int,
    keyword_top: int,
) -> dict:
    keywords = extract_keywords(full_text, top_n=keyword_top)
    return {
        "file": str(path),
        "extension": path.suffix.lower(),
        "structure": {
            **structure,
            "source_part_count": len(source_parts),
            "text_length": len(full_text),
            "line_count": len([line for line in full_text.splitlines() if line.strip()]),
        },
        "keywords": keywords,
        "patterns": detect_patterns(full_text),
        "text_sample": sample_text(full_text, limit=text_limit),
        "source_parts_preview": source_parts[:20],
    }


def extract_keywords(text: str, *, top_n: int) -> list[dict[str, int]]:
    tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
    tokens = [token for token in tokens if token not in STOPWORDS and not token.isdigit()]
    counter = Counter(tokens)
    return [{"keyword": token, "count": count} for token, count in counter.most_common(top_n)]


def detect_patterns(text: str) -> dict[str, int]:
    return {
        "dates": len(re.findall(r"\b\d{4}[./-]\d{1,2}[./-]\d{1,2}\b|\b\d{1,2}[./-]\d{1,2}\b", text)),
        "money_like": len(re.findall(r"\b\d{1,3}(?:,\d{3})+\s*(?:원|KRW)?\b|\b\d+\s*원\b", text)),
        "business_numbers": len(re.findall(r"\b\d{3}-\d{2}-\d{5}\b", text)),
        "emails": len(re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)),
        "phone_like": len(re.findall(r"\b0\d{1,2}-\d{3,4}-\d{4}\b", text)),
        "numeric_tokens": len(re.findall(r"\b\d+(?:[.,]\d+)*\b", text)),
    }


def sample_text(text: str, *, limit: int) -> str:
    if len(text) <= limit:
        return text
    part = max(limit // 3, 300)
    middle_start = max(len(text) // 2 - part // 2, 0)
    return (
        "[BEGIN]\n"
        + text[:part].strip()
        + "\n\n[MIDDLE]\n"
        + text[middle_start : middle_start + part].strip()
        + "\n\n[END]\n"
        + text[-part:].strip()
    )


def normalize_extracted_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def count_paragraphs(text: str) -> int:
    return len([line for line in text.splitlines() if line.strip()])


def strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def print_report(result: dict) -> None:
    print(f"# File: {result['file']}")
    print("\n## Structure")
    for key, value in result["structure"].items():
        print(f"- {key}: {value}")

    print("\n## Pattern Counts")
    for key, value in result["patterns"].items():
        print(f"- {key}: {value}")

    print("\n## Keywords")
    for item in result["keywords"]:
        print(f"- {item['keyword']}: {item['count']}")

    print("\n## Text Sample")
    print(result["text_sample"] or "(no text extracted)")


if __name__ == "__main__":
    raise SystemExit(main())
