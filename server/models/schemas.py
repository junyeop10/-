from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Category(str, Enum):
    FINAL = "최종본"
    PRESENTATION = "발표자료"
    REPORT = "보고서"
    DATA = "데이터"
    REFERENCE = "참고자료"
    DRAFT = "작업중"
    UNCLASSIFIED = "분류불가"


@dataclass
class EvidencePackage:
    xxhash: str
    filename: str
    ext: str
    size_kb: float
    modified_at: float
    text_front: str
    text_middle: str
    text_rear: str
    trigger_chunks: list[str]
    keyword_hits: list[str]
    pattern_flags: dict
    version_hint: str
    embedding: list[float]
    extract_method: str
    extract_status: str  # 'success' / 'ocr_fallback' / 'failed'


@dataclass
class FeedbackLog:
    xxhash: str
    embedding: list[float]
    system_category: Category
    user_category: Category
    corrected: bool
    correction_stage: str
    timestamp: float


@dataclass
class FinalizedDocument:
    file_path: str
    xxhash: str
    sha256: str
    category: Category
    finalized_at: float


@dataclass
class ClassifyResult:
    filename: str
    file_path: str
    xxhash: str
    category: Category
    confidence: float
    reason: str
    keywords: list[str]
    classify_method: str  # 'rule' / 'embedding' / 'llm_local' / 'llm_api' / 'review_queue'
    version_hint: str = ""
    review_reason: str = ""
