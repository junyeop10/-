# 문서 넣기 / 텍스트 추출 코드 발췌

## 1. 문서 넣기: 드래그 앤 드롭, 폴더 선택, 폴더 삽입

### 지원 확장자

```python
# src/file_reader.py
SUPPORTED_SUFFIXES = {".txt", ".pdf", ".docx", ".xlsx", ".pptx", ".hwp", ".hwpx"}
```

### 폴더 안의 지원 문서 찾기

```python
# src/file_reader.py
def discover_supported_files(input_dir: str | Path) -> list[Path]:
    """Find supported input files under an input directory."""
    directory = ensure_input_directory(input_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {directory}")

    files = [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return sorted(files)
```

### GUI 드래그 앤 드롭 영역

```python
# src/gui.py
drop_text = "여기에 txt/pdf/docx/xlsx/pptx 파일 또는 폴더를 드래그하세요"
if not DRAG_AND_DROP_AVAILABLE:
    drop_text = "드래그 앤 드롭은 tkinterdnd2 설치 후 사용할 수 있습니다"
self.drop_label = ttk.Label(self, text=drop_text, anchor="center", padding=12, relief="ridge")
self.drop_label.pack(fill="x", padx=10, pady=(0, 10))
if DRAG_AND_DROP_AVAILABLE:
    self.drop_label.drop_target_register(DND_FILES)
    self.drop_label.dnd_bind("<<Drop>>", self.on_drop_files)
```

### 드롭된 파일/폴더를 분류 대상 파일 목록으로 변환

```python
# src/gui.py
def collect_supported_drop_files(paths: list[Path]) -> list[Path]:
    """Expand dropped files/folders into supported document files."""
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(discover_supported_files(path))
        elif path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(path)
    return sorted({file_path.resolve() for file_path in files})
```

### 드래그 앤 드롭 이벤트 처리

```python
# src/gui.py
def on_drop_files(self, event: object) -> None:
    if not self._ensure_classification_available():
        return
    raw_data = getattr(event, "data", "")
    dropped_paths = [Path(value) for value in self.tk.splitlist(raw_data)]
    files = collect_supported_drop_files(dropped_paths)

    self._clear_results()
    self.start_classify_files(files)
```

### 폴더 선택 창

```python
# src/gui.py
def choose_folder(self) -> None:
    selected = filedialog.askdirectory(initialdir=self.input_dir.get() or ".")
    if selected:
        self.input_dir.set(selected)
```

### 선택한 입력 폴더의 파일 분류 시작

```python
# src/gui.py
def start_classify(self) -> None:
    if not self._ensure_classification_available():
        return
    self._clear_results()
    self.status_text.set("파일 목록 준비 중")
    input_dir = ensure_input_directory(self.input_dir.get())
    files = discover_supported_files(input_dir)
    self.start_classify_files(files)
```

## 2. 텍스트 추출: 한국어 중심, 영어도 처리 가능

### 텍스트 정규화 및 토큰화

```python
# src/text_cleaner.py
TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z0-9]{2,}")


def normalize_text(text: str) -> str:
    """Normalize text for rules and embeddings."""
    if not isinstance(text, str):
        raise TypeError("normalize_text expects a string.")

    cleaned = text.replace("\x00", " ")
    cleaned = cleaned.lower().strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def tokenize_text(text: str) -> list[str]:
    """Extract simple Korean, English, and numeric tokens."""
    normalized = normalize_text(text)
    return TOKEN_PATTERN.findall(normalized)
```

### 1500 / 1500 / 1500 샘플링

```python
# src/text_cleaner.py
def build_sampled_text(
    text: str,
    total_limit: int = 4500,
    part_limit: int = 1500,
) -> str:
    """Build evidence text from beginning, middle, and end excerpts."""
    if not isinstance(text, str):
        raise TypeError("build_sampled_text expects a string.")

    cleaned = text.strip()
    if len(cleaned) <= total_limit:
        return cleaned

    label_text = "[BEGIN_EXCERPT]\n\n[MIDDLE_EXCERPT]\n\n[END_EXCERPT]\n"
    available_text_limit = max(total_limit - len(label_text), 300)
    effective_part_limit = min(part_limit, available_text_limit // 3)
    begin = cleaned[:effective_part_limit]

    middle_start = max((len(cleaned) // 2) - (effective_part_limit // 2), 0)
    middle_end = middle_start + effective_part_limit
    middle = cleaned[middle_start:middle_end]

    end = cleaned[-effective_part_limit:]

    return (
        "[BEGIN_EXCERPT]\n"
        f"{begin}\n"
        "[MIDDLE_EXCERPT]\n"
        f"{middle}\n"
        "[END_EXCERPT]\n"
        f"{end}"
    )
```

### 확장자별 텍스트 추출 분기

```python
# src/file_reader.py
def extract_text_from_file(
    path: str | Path,
    fast: bool = True,
    total_limit: int = 4500,
    part_limit: int = 1500,
) -> str:
    """Extract evidence text from a supported file."""
    file_path = Path(path)

    if file_path.suffix.lower() == ".txt":
        text = read_txt_file(file_path)
        return build_sampled_text(text, total_limit=total_limit, part_limit=part_limit)
    if file_path.suffix.lower() == ".pdf":
        return extract_pdf_text(
            file_path,
            fast=fast,
            total_limit=total_limit,
            part_limit=part_limit,
        )
    if file_path.suffix.lower() == ".docx":
        return extract_docx_text(
            file_path,
            total_limit=total_limit,
            part_limit=part_limit,
        )
    if file_path.suffix.lower() == ".xlsx":
        return extract_xlsx_text(
            file_path,
            fast=fast,
            total_limit=total_limit,
            part_limit=part_limit,
        )
    if file_path.suffix.lower() == ".pptx":
        return extract_pptx_text(
            file_path,
            total_limit=total_limit,
            part_limit=part_limit,
        )
    if file_path.suffix.lower() == ".hwp":
        return extract_hwp_text(
            file_path,
            total_limit=total_limit,
            part_limit=part_limit,
        )
    if file_path.suffix.lower() == ".hwpx":
        return extract_hwpx_text(
            file_path,
            total_limit=total_limit,
            part_limit=part_limit,
        )

    raise ValueError(f"Unsupported file type: {file_path.suffix}")
```

### TXT: UTF-8 / CP949 읽기

```python
# src/file_reader.py
def read_txt_file(path: str | Path) -> str:
    """Read a txt file with common encodings."""
    file_path = Path(path)
    for encoding in ("utf-8", "cp949"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        "text",
        b"",
        0,
        1,
        f"Could not read with utf-8 or cp949: {file_path}",
    )
```

### PDF: PyMuPDF 기반 페이지 텍스트 추출

```python
# src/pdf_reader.py
def extract_pdf_text(
    path: str | Path,
    fast: bool = True,
    total_limit: int = 4500,
    part_limit: int = 1500,
    pages_per_section: int = 2,
) -> str:
    """Extract evidence text from first, middle, and last PDF pages."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required to read PDF files.") from exc

    with fitz.open(file_path) as document:
        page_count = len(document)
        if page_count == 0:
            return ""

        if fast:
            page_indexes = _sample_page_indexes(page_count, pages_per_section=pages_per_section)
        else:
            page_indexes = list(range(page_count))

        chunks = []
        for page_index in page_indexes:
            page = document.load_page(page_index)
            text = page.get_text("text").strip()
            if text:
                chunks.append(f"[page {page_index + 1}]\n{text}")

    return build_sampled_text(
        "\n\n".join(chunks),
        total_limit=total_limit,
        part_limit=part_limit,
    )
```

### DOCX: python-docx 기반 문단/표 텍스트 추출

```python
# src/office_reader.py
def extract_docx_text(
    path: str | Path,
    total_limit: int = 4500,
    part_limit: int = 1500,
) -> str:
    """Extract paragraph and table text from a DOCX file."""
    from docx import Document

    document = Document(str(path))
    chunks: list[str] = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            chunks.append(text)

    for table in document.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                chunks.append(row_text)

    return build_sampled_text("\n".join(chunks), total_limit=total_limit, part_limit=part_limit)
```

### XLSX: openpyxl 기반 시트/셀 텍스트 추출

```python
# src/office_reader.py
def extract_xlsx_text(
    path: str | Path,
    fast: bool = True,
    total_limit: int = 4500,
    part_limit: int = 1500,
) -> str:
    """Extract sheet names and top rows from an XLSX file."""
    from openpyxl import load_workbook

    workbook = load_workbook(filename=str(path), read_only=True, data_only=True)
    chunks: list[str] = []

    max_rows = 20 if fast else 100
    for sheet in workbook.worksheets:
        chunks.append(f"sheet: {sheet.title}")
        for row in sheet.iter_rows(max_row=max_rows, values_only=True):
            row_text = " | ".join(str(value).strip() for value in row if value is not None and str(value).strip())
            if row_text:
                chunks.append(row_text)

    return build_sampled_text("\n".join(chunks), total_limit=total_limit, part_limit=part_limit)
```

### PPTX: python-pptx 기반 슬라이드 텍스트 추출

```python
# src/office_reader.py
def extract_pptx_text(
    path: str | Path,
    total_limit: int = 4500,
    part_limit: int = 1500,
) -> str:
    """Extract visible text from PPTX slides."""
    from pptx import Presentation

    presentation = Presentation(str(path))
    chunks: list[str] = []

    for slide_index, slide in enumerate(presentation.slides, start=1):
        chunks.append(f"slide: {slide_index}")
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                chunks.append(shape.text.strip())

    return build_sampled_text("\n".join(chunks), total_limit=total_limit, part_limit=part_limit)
```

### HWPX: zip/XML 기반 텍스트 추출

```python
# src/hwp_reader.py
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
            with archive.open(name) as xml_file:
                tree = ET.parse(xml_file)
            root = tree.getroot()
            for element in root.iter():
                if element.text and element.text.strip():
                    chunks.append(element.text.strip())

    return build_sampled_text("\n".join(chunks), total_limit=total_limit, part_limit=part_limit)
```

### HWP: OLE/UTF-16LE 기반 텍스트 추출

```python
# src/hwp_reader.py
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
        import olefile
    except ImportError:
        return _extract_utf16le_strings(file_path, total_limit=total_limit, part_limit=part_limit)

    if not olefile.isOleFile(str(file_path)):
        return _extract_utf16le_strings(file_path, total_limit=total_limit, part_limit=part_limit)

    chunks: list[str] = []
    with olefile.OleFileIO(str(file_path)) as ole:
        streams = [
            item
            for item in ole.listdir()
            if len(item) >= 2 and item[0] == "BodyText" and item[-1].startswith("Section")
        ]
        for stream in sorted(streams, key=lambda value: value[-1]):
            data = ole.openstream(stream).read()
            text = _decode_hwp_section(data)
            if text.strip():
                chunks.append(text)

    if chunks:
        return build_sampled_text("\n".join(chunks), total_limit=total_limit, part_limit=part_limit)
    return _extract_utf16le_strings(file_path, total_limit=total_limit, part_limit=part_limit)
```
