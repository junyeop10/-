"""
file_reader.py
--------------
실제 문서 파일(PDF/HWP/HWPX/DOCX/XLSX/TXT 등)에서 본문 텍스트를 추출한다.
추출 실패 시에는 **파일명**을 텍스트로 사용 — 한국어 파일명은
"중소기업_사업계획서.hwp", "벤처기업인증서.pdf"처럼 의미 신호가 강함.

지원 포맷
---------
- .pdf       : PyMuPDF (fitz)
- .hwpx      : ZIP + XML (표준 라이브러리만)
- .hwp       : 바이너리 포맷 — 직접 파싱 안 함 → 파일명 fallback
- .xlsx      : openpyxl (셀 텍스트 추출)
- .docx      : python-docx (있으면) → 없으면 파일명 fallback
- .txt/.md   : utf-8 / cp949 자동 시도
- 그 외      : 파일명 fallback

반환 구조
---------
ReadResult{ text, source_method, char_count, error }
"""

from __future__ import annotations

import logging
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

# 본문 너무 길면 임베딩 효율 ↓ — 4500자에서 자름 (설계 문서 기준)
MAX_CHARS = 4500


@dataclass
class ReadResult:
    text: str
    source_method: str   # "pdf" | "hwpx" | "xlsx" | "docx" | "txt" | "filename"
    char_count: int
    error: Optional[str] = None


# ──────────────────────────────────────────────
# 포맷별 reader
# ──────────────────────────────────────────────

def _read_pdf(path: Path) -> str:
    import fitz  # PyMuPDF
    text_parts: list[str] = []
    with fitz.open(str(path)) as doc:
        for page in doc:
            text_parts.append(page.get_text())
            if sum(len(t) for t in text_parts) > MAX_CHARS * 2:
                break
    return "\n".join(text_parts)


def _read_hwpx(path: Path) -> str:
    """HWPX는 ZIP 안에 section*.xml — 그 안의 <hp:t> 노드 텍스트를 모은다."""
    ns = {"hp": "http://www.hancom.co.kr/hwpml/2011/paragraph"}
    text_parts: list[str] = []
    with zipfile.ZipFile(path) as zf:
        section_names = sorted(
            n for n in zf.namelist() if re.match(r"Contents/section\d+\.xml", n)
        )
        for name in section_names:
            with zf.open(name) as f:
                tree = ET.parse(f)
            for t in tree.findall(".//hp:t", ns):
                if t.text:
                    text_parts.append(t.text)
            if sum(len(t) for t in text_parts) > MAX_CHARS * 2:
                break
    return "\n".join(text_parts)


def _read_xlsx(path: Path) -> str:
    import openpyxl
    wb = openpyxl.load_workbook(str(path), data_only=True, read_only=True)
    text_parts: list[str] = []
    for ws in wb.worksheets:
        for row in ws.iter_rows(values_only=True):
            for cell in row:
                if cell is None:
                    continue
                text_parts.append(str(cell))
            if sum(len(t) for t in text_parts) > MAX_CHARS * 2:
                break
        if sum(len(t) for t in text_parts) > MAX_CHARS * 2:
            break
    wb.close()
    return " ".join(text_parts)


def _read_docx(path: Path) -> str:
    try:
        import docx  # python-docx
    except ImportError:
        raise RuntimeError("python-docx 미설치")
    d = docx.Document(str(path))
    return "\n".join(p.text for p in d.paragraphs if p.text)


def _read_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="replace")


def _filename_as_text(path: Path) -> str:
    """파일명을 본문 대신 사용. 확장자 제거, 구분자 공백 치환."""
    stem = path.stem
    # 언더스코어/하이픈 → 공백, 다중 공백 정리
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem


# ──────────────────────────────────────────────
# 통합 진입점
# ──────────────────────────────────────────────

def read_file(path: Path) -> ReadResult:
    """파일에서 텍스트를 추출한다. 실패 시 파일명을 fallback으로 사용."""
    ext = path.suffix.lower()

    raw: str = ""
    method = "filename"
    error: Optional[str] = None

    try:
        if ext == ".pdf":
            raw = _read_pdf(path)
            method = "pdf"
        elif ext == ".hwpx":
            raw = _read_hwpx(path)
            method = "hwpx"
        elif ext == ".xlsx":
            raw = _read_xlsx(path)
            method = "xlsx"
        elif ext == ".docx":
            raw = _read_docx(path)
            method = "docx"
        elif ext in (".txt", ".md"):
            raw = _read_text(path)
            method = "txt"
        else:
            # .hwp (바이너리), 그 외 모두 파일명 fallback
            raw = ""
            method = "filename"
    except Exception as e:  # noqa: BLE001 — 어떤 예외든 fallback
        error = f"{type(e).__name__}: {e}"
        logger.warning("read 실패: %s (%s) → 파일명 fallback", path.name, error)
        raw = ""

    # ── [실험 토글] 순수 의미기반 모드 — 파일명 신호 완전 배제 ──
    # 환경변수 FILE_READER_BODY_ONLY=1 이면 파일명 prefix/fallback을 쓰지 않고
    # *추출된 본문만* 사용한다. 본문이 없으면(.hwp/이미지 등) 빈 텍스트 →
    # 임베딩이 zero 벡터가 되어 분류 불가(LLM 위임)로 이어진다.
    # 기본값(미설정)은 기존 동작과 100% 동일.
    if os.environ.get("FILE_READER_BODY_ONLY") == "1":
        body = raw.strip()
        if not body:
            method = "empty(no-body)"
        if len(body) > MAX_CHARS:
            body = body[:MAX_CHARS]
        return ReadResult(text=body, source_method=method,
                          char_count=len(body), error=error)

    # 본문이 너무 짧으면(스캔 PDF 등) 파일명을 앞에 붙여 보강
    fname_hint = _filename_as_text(path)
    if len(raw.strip()) < 50:
        raw = fname_hint + "\n" + raw
        if method != "filename":
            method = method + "+filename"
    else:
        # 파일명을 항상 강한 prefix로 넣어 의미 신호 강화
        raw = fname_hint + "\n" + raw

    # 길이 상한
    if len(raw) > MAX_CHARS:
        raw = raw[:MAX_CHARS]

    return ReadResult(
        text=raw,
        source_method=method,
        char_count=len(raw),
        error=error,
    )


def split_three(text: str) -> tuple[str, str, str]:
    """본문을 front/middle/rear로 대략 1:1:1 분할."""
    n = len(text)
    a = n // 3
    b = 2 * n // 3
    return text[:a], text[a:b], text[b:]
