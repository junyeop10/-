from __future__ import annotations

import unittest
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

from docx import Document
from openpyxl import Workbook
from pptx import Presentation

from src.file_reader import SUPPORTED_SUFFIXES, discover_supported_files, extract_text_from_file


class FileReaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.base_dir = Path("tests_runtime") / f"reader_case_{uuid4().hex}"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.base_dir.exists():
            rmtree(self.base_dir, ignore_errors=True)

    def test_supported_suffixes_include_office_formats(self) -> None:
        self.assertIn(".docx", SUPPORTED_SUFFIXES)
        self.assertIn(".xlsx", SUPPORTED_SUFFIXES)
        self.assertIn(".pptx", SUPPORTED_SUFFIXES)

    def test_discover_supported_files_finds_office_documents(self) -> None:
        (self.base_dir / "sample.txt").write_text("report summary", encoding="utf-8")
        self._create_docx(self.base_dir / "sample.docx")
        self._create_xlsx(self.base_dir / "sample.xlsx")
        self._create_pptx(self.base_dir / "sample.pptx")

        files = discover_supported_files(self.base_dir)
        suffixes = sorted(path.suffix.lower() for path in files)
        self.assertEqual(suffixes, [".docx", ".pptx", ".txt", ".xlsx"])

    def test_extract_text_from_docx(self) -> None:
        path = self.base_dir / "contract.docx"
        self._create_docx(path)

        text = extract_text_from_file(path, fast=True)
        self.assertIn("contract agreement", text.lower())
        self.assertIn("payment term", text.lower())

    def test_extract_text_from_xlsx(self) -> None:
        path = self.base_dir / "invoice.xlsx"
        self._create_xlsx(path)

        text = extract_text_from_file(path, fast=True)
        self.assertIn("sheet: invoice", text.lower())
        self.assertIn("amount due", text.lower())

    def test_extract_text_from_pptx(self) -> None:
        path = self.base_dir / "presentation.pptx"
        self._create_pptx(path)

        text = extract_text_from_file(path, fast=True)
        self.assertIn("slide: 1", text.lower())
        self.assertIn("business plan presentation", text.lower())

    def _create_docx(self, path: Path) -> None:
        document = Document()
        document.add_paragraph("Contract agreement for project delivery")
        document.add_paragraph("Payment term and report summary")
        document.save(path)

    def _create_xlsx(self, path: Path) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Invoice"
        sheet.append(["Invoice", "Amount Due", "Status"])
        sheet.append(["INV-001", 1000, "Open"])
        workbook.save(path)

    def _create_pptx(self, path: Path) -> None:
        presentation = Presentation()
        slide = presentation.slides.add_slide(presentation.slide_layouts[1])
        slide.shapes.title.text = "Business Plan Presentation"
        slide.placeholders[1].text = "Target market and execution plan"
        presentation.save(path)


if __name__ == "__main__":
    unittest.main()
