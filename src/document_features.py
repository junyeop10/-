"""Document feature extraction and compressed-text helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.layout_features import DocumentLayoutFeatureExtractor
from src.hash_utils import compute_raw_text_hash
from src.text_cleaner import build_sampled_text, normalize_text, tokenize_text


FEATURE_EXTRACTOR_VERSION = "2.3"


@dataclass
class DocumentFeatureBundle:
    feature_version: str
    filename_features: dict[str, Any]
    metadata_features: dict[str, Any]
    structural_features: dict[str, Any]
    layout_features: dict[str, Any]
    text_stats: dict[str, Any]
    compressed_text: str
    compressed_text_hash: str

    def to_storage_dict(self) -> dict[str, Any]:
        return {
            "feature_version": self.feature_version,
            "filename_features": self.filename_features,
            "metadata_features": self.metadata_features,
            "structural_features": self.structural_features,
            "layout_features": self.layout_features,
            "text_stats": self.text_stats,
            "compressed_text": self.compressed_text,
            "compressed_text_hash": self.compressed_text_hash,
        }


class DocumentFeatureExtractor:
    """Extract CPU-friendly reusable features from filename and sampled text."""

    def __init__(self, version: str = FEATURE_EXTRACTOR_VERSION) -> None:
        self.version = version
        self.layout_extractor = DocumentLayoutFeatureExtractor()

    def extract(
        self,
        *,
        file_name: str,
        file_ext: str = "",
        text: str,
        file_size: int = 0,
        file_path: str | Path | None = None,
    ) -> DocumentFeatureBundle:
        normalized_text = normalize_text(text or "")
        normalized_name = normalize_text(Path(file_name).stem)
        extension = (file_ext or Path(file_name).suffix).lower()
        lines = [line.strip() for line in re.split(r"[\r\n]+", text or "") if line.strip()]
        title = self._pick_title(lines, normalized_name)
        headings = self._pick_headings(lines)

        filename_features = self._filename_features(file_name=file_name, normalized_name=normalized_name)
        metadata_features = {
            "file_ext": extension,
            "file_size": int(file_size or 0),
            "extension_type": extension.lstrip("."),
        }
        structural_features = self._structural_features(
            text=normalized_text,
            raw_lines=lines,
            file_ext=extension,
        )
        layout_features = self._layout_features(file_path=file_path, text=normalized_text)
        text_stats = self._text_stats(normalized_text)
        structural_features.update(self._pattern_scores(normalized_text, raw_lines=lines))
        text_stats.update(self._ocr_quality_features(text or ""))
        compressed_text = self._build_compressed_text(
            file_name=file_name,
            title=title,
            headings=headings,
            text=normalized_text,
            structural_features=structural_features,
            layout_features=layout_features,
        )
        compressed_text_hash = compute_raw_text_hash(compressed_text)
        filename_features["title_candidate"] = title
        filename_features["headings"] = headings

        return DocumentFeatureBundle(
            feature_version=self.version,
            filename_features=filename_features,
            metadata_features=metadata_features,
            structural_features=structural_features,
            layout_features=layout_features,
            text_stats=text_stats,
            compressed_text=compressed_text,
            compressed_text_hash=compressed_text_hash,
        )

    def _filename_features(self, *, file_name: str, normalized_name: str) -> dict[str, Any]:
        tokens = tokenize_text(normalized_name)
        return {
            "file_name": file_name,
            "stem": Path(file_name).stem,
            "normalized_stem": normalized_name,
            "tokens": tokens,
            "token_count": len(tokens),
            "has_korean_contract_hint": any(token in normalized_name for token in ("계약", "근로", "용역")),
            "has_presentation_hint": any(token in normalized_name for token in ("발표", "ppt", "presentation", "캡스톤")),
            "has_paper_hint": any(token in normalized_name for token in ("paper", "review", "논문", "transformer")),
            "has_receipt_hint": any(token in normalized_name for token in ("영수증", "receipt", "invoice", "청구")),
        }

    def _structural_features(self, *, text: str, raw_lines: list[str], file_ext: str) -> dict[str, Any]:
        bullet_lines = [line for line in raw_lines if re.match(r"^\s*([-*•]|\d+[.)])\s+", line)]
        sentence_parts = [part for part in re.split(r"[.!?。！？]\s*", text) if part.strip()]
        citation_count = len(re.findall(r"(\[[0-9]{1,3}\]|\bet al\.|\bdoi\b|arxiv)", text, flags=re.IGNORECASE))
        contract_terms = ("제1조", "갑", "을", "계약", "계약기간", "근로", "용역")
        receipt_terms = ("승인번호", "사업자번호", "공급가액", "합계금액", "영수증", "invoice", "receipt")
        table_count = text.count(" | ")
        image_count = len(re.findall(r"\b(image|figure|그림|사진)\b", text, flags=re.IGNORECASE))
        return {
            "page_count": self._safe_count_marker(text, "page:"),
            "slide_count": self._safe_count_marker(text, "slide:") if file_ext == ".pptx" else 0,
            "sheet_count": self._safe_count_marker(text, "sheet:") if file_ext == ".xlsx" else 0,
            "table_count": table_count,
            "image_count": image_count,
            "bullet_ratio": round(len(bullet_lines) / max(len(raw_lines), 1), 4),
            "citation_count": citation_count,
            "has_abstract": bool(re.search(r"\babstract\b|초록|요약", text, flags=re.IGNORECASE)),
            "has_references": bool(re.search(r"\breferences\b|참고문헌|bibliography", text, flags=re.IGNORECASE)),
            "has_doi": bool(re.search(r"\bdoi\b|10\.\d{4,9}/", text, flags=re.IGNORECASE)),
            "contract_terms_count": sum(1 for term in contract_terms if term.lower() in text),
            "receipt_terms_count": sum(1 for term in receipt_terms if term.lower() in text),
            "sentence_count": len(sentence_parts),
            "line_count": len(raw_lines),
        }

    def _text_stats(self, text: str) -> dict[str, Any]:
        tokens = tokenize_text(text)
        sentences = [part.strip() for part in re.split(r"[.!?。！？]\s*", text) if part.strip()]
        avg_sentence_length = sum(len(sentence) for sentence in sentences) / max(len(sentences), 1)
        return {
            "char_count": len(text),
            "token_count": len(tokens),
            "unique_token_count": len(set(tokens)),
            "average_sentence_length": round(avg_sentence_length, 2),
        }

    def _pattern_scores(self, text: str, *, raw_lines: list[str]) -> dict[str, Any]:
        lowered = text.lower()
        clause_matches = re.findall(r"(제\s*\d+\s*조|\barticle\s+\d+)", text, flags=re.IGNORECASE)
        legal_terms = ("계약", "갑", "을", "손해배상", "비밀유지", "준거법", "해지", "contract", "confidentiality")
        research_terms = ("abstract", "method", "results", "references", "doi", "citation")
        report_terms = ("요약", "현황", "분석", "결론", "성과", "summary", "analysis", "conclusion")
        resume_terms = ("경력", "학력", "기술", "experience", "education", "skills", "email", "phone")
        headings = self._pick_headings(raw_lines)
        token_count = max(len(tokenize_text(text)), 1)
        return {
            "clause_pattern_score": round(min(1.0, len(clause_matches) / 6.0), 4),
            "legal_term_density": round(sum(lowered.count(term.lower()) for term in legal_terms) / token_count, 4),
            "research_structure_score": round(sum(1 for term in research_terms if term in lowered) / len(research_terms), 4),
            "report_structure_score": round(sum(1 for term in report_terms if term.lower() in lowered) / len(report_terms), 4),
            "contact_pattern_score": round(
                min(
                    1.0,
                    (1.0 if re.search(r"[\w.+-]+@[\w.-]+\.\w+", text) else 0.0) * 0.45
                    + (1.0 if re.search(r"(\+?\d[\d\s-]{7,}\d)", text) else 0.0) * 0.35
                    + (sum(1 for term in resume_terms if term in lowered) / len(resume_terms)) * 0.2,
                ),
                4,
            ),
            "heading_density": round(len(headings) / max(len(raw_lines), 1), 4),
        }

    def _ocr_quality_features(self, raw_text: str) -> dict[str, Any]:
        chars = [char for char in raw_text if not char.isspace()]
        unreadable = sum(1 for char in chars if char in {"�", "□", "▯"})
        symbols = sum(1 for char in chars if not char.isalnum() and not ("\uac00" <= char <= "\ud7a3"))
        unreadable_ratio = unreadable / max(len(chars), 1)
        symbol_noise_ratio = symbols / max(len(chars), 1)
        low_quality_scan_score = min(1.0, unreadable_ratio * 2.5 + max(0.0, symbol_noise_ratio - 0.25) * 1.5)
        return {
            "ocr_text_length": len(raw_text or ""),
            "ocr_confidence_mean": 0.0,
            "unreadable_ratio": round(unreadable_ratio, 4),
            "symbol_noise_ratio": round(symbol_noise_ratio, 4),
            "low_quality_scan_score": round(low_quality_scan_score, 4),
        }

    def _build_compressed_text(
        self,
        *,
        file_name: str,
        title: str,
        headings: list[str],
        text: str,
        structural_features: dict[str, Any],
        layout_features: dict[str, Any],
    ) -> str:
        structural_summary = " ".join(
            f"{key}={value}"
            for key, value in structural_features.items()
            if key
            in {
                "slide_count",
                "sheet_count",
                "table_count",
                "image_count",
                "citation_count",
                "has_abstract",
                "has_references",
                "has_doi",
                "contract_terms_count",
                "receipt_terms_count",
                "clause_pattern_score",
                "legal_term_density",
                "research_structure_score",
                "report_structure_score",
                "contact_pattern_score",
                "heading_density",
            }
        )
        layout_summary = " ".join(
            f"{key}={value}"
            for key, value in layout_features.items()
            if key
            in {
                "portrait_or_landscape",
                "text_density",
                "whitespace_ratio",
                "receipt_pattern_score",
                "certificate_pattern_score",
                "slide_like_layout_score",
                "dense_text_score",
                "two_column_score",
                "image_area_ratio",
                "header_block_score",
                "footer_pattern_score",
                "signature_area_score",
                "chart_presence_score",
                "section_divider_score",
                "numeric_column_score",
                "approval_block_score",
                "repeated_line_pattern_score",
            }
        )
        parts = [
            f"filename: {file_name}",
            f"title: {title}",
            "headings: " + " | ".join(headings[:8]),
            "sampled_text: " + build_sampled_text(text, total_limit=2600, part_limit=850),
            "structure: " + structural_summary,
            "layout: " + layout_summary,
        ]
        return normalize_text("\n".join(part for part in parts if part.strip()))

    def _layout_features(self, *, file_path: str | Path | None, text: str) -> dict[str, Any]:
        if file_path is None:
            return self.layout_extractor.extract_from_images([], ocr_text=text).features
        try:
            return self.layout_extractor.extract_from_file(file_path, ocr_text=text).features
        except Exception:
            return self.layout_extractor.extract_from_images([], ocr_text=text).features

    def _pick_title(self, lines: list[str], normalized_name: str) -> str:
        for line in lines[:12]:
            compact = line.strip()
            if 3 <= len(compact) <= 120:
                return compact
        return normalized_name

    def _pick_headings(self, lines: list[str]) -> list[str]:
        headings: list[str] = []
        for line in lines[:80]:
            compact = line.strip()
            if not compact or len(compact) > 90:
                continue
            if re.match(r"^(\d+[\.)]|#+|제\s*\d+\s*장|제\s*\d+\s*조)", compact):
                headings.append(compact)
            elif compact.isupper() and len(compact) >= 4:
                headings.append(compact)
        return headings[:12]

    def _safe_count_marker(self, text: str, marker: str) -> int:
        count = len(re.findall(re.escape(marker), text, flags=re.IGNORECASE))
        return count
