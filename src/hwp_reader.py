"""Hangul HWP/HWPX text extraction helpers."""

from __future__ import annotations

import re
import struct
import zipfile
import zlib
from pathlib import Path
from xml.etree import ElementTree as ET

from src.text_cleaner import build_sampled_text


def extract_hwpx_text(
    path: str | Path,
    total_limit: int = 4500,
    part_limit: int = 1500,
) -> str:
    """Extract visible text from a HWPX document's XML sections."""
    chunks: list[str] = []
    file_path = Path(path)

    with zipfile.ZipFile(file_path) as archive:
        xml_names = [name for name in archive.namelist() if name.lower().endswith(".xml")]
        section_names = sorted(name for name in xml_names if "section" in name.lower())
        for name in section_names or xml_names:
            try:
                root = ET.fromstring(archive.read(name))
            except ET.ParseError:
                continue
            for elem in root.iter():
                if _local_name(elem.tag) == "t" and elem.text:
                    text = elem.text.strip()
                    if text:
                        chunks.append(text)

    return build_sampled_text(_normalize_extracted_text("\n".join(chunks)), total_limit=total_limit, part_limit=part_limit)


def extract_hwp_text(
    path: str | Path,
    total_limit: int = 4500,
    part_limit: int = 1500,
) -> str:
    """Extract rough text from a binary HWP document.

    HWP is an OLE compound document. This reads BodyText/Section streams when
    olefile is available, then falls back to scanning UTF-16LE strings.
    """
    file_path = Path(path)
    try:
        import olefile  # type: ignore
    except ImportError:
        return build_sampled_text(
            _extract_utf16_strings(file_path.read_bytes()),
            total_limit=total_limit,
            part_limit=part_limit,
        )

    chunks: list[str] = []
    try:
        with olefile.OleFileIO(file_path) as ole:
            compressed = _hwp_is_compressed(ole)
            body_streams = sorted(
                "/".join(item)
                for item in ole.listdir(streams=True)
                if "/".join(item).startswith("BodyText/Section")
            )
            for stream_name in body_streams:
                raw = ole.openstream(stream_name.split("/")).read()
                if compressed:
                    raw = zlib.decompress(raw, -15)
                chunks.extend(_extract_hwp_record_text(raw))
    except Exception:
        chunks = [_extract_utf16_strings(file_path.read_bytes())]

    return build_sampled_text(_normalize_extracted_text("\n".join(chunks)), total_limit=total_limit, part_limit=part_limit)


def _hwp_is_compressed(ole: object) -> bool:
    try:
        header = ole.openstream("FileHeader").read()
        flags = struct.unpack_from("<I", header, 36)[0]
        return bool(flags & 0x01)
    except Exception:
        return True


def _extract_hwp_record_text(data: bytes) -> list[str]:
    chunks: list[str] = []
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
            text = _decode_hwp_para_text(payload)
            if text:
                chunks.append(text)
    if chunks:
        return chunks
    fallback = _extract_utf16_strings(data)
    return [fallback] if fallback else []


def _decode_hwp_para_text(payload: bytes) -> str:
    text = payload.decode("utf-16le", errors="ignore")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_utf16_strings(data: bytes) -> str:
    decoded = data.decode("utf-16le", errors="ignore")
    chunks = re.findall(r"[가-힣A-Za-z0-9 .,;:()\[\]{}<>/%+-]{4,}", decoded)
    return "\n".join(chunk.strip() for chunk in chunks if chunk.strip())


def _normalize_extracted_text(text: str) -> str:
    cleaned = text.replace("\x00", " ")
    cleaned = re.sub(r"[ \t\r\f\v]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag
