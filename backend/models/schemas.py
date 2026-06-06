from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Category(str, Enum):
    NOTICE_FORM = "공고_지침_양식"
    BUSINESS_PLAN = "사업계획서 수행계획서"
    RESEARCH_REFERENCE = "조사_참고자료"
    DELIVERABLE_REPORT = "중간_최종 결과물 및 보고서"
    PRESENTATION = "발표자료"
    ESTIMATE_CONTRACT = "견적_계약_정산"
    CERTIFICATE = "기업 인증서"
    OTHER = "기타"
    UNCLASSIFIED = "분류불가"  # 시스템 전용: API 오류·검토 큐


# LLM·룰 분류에 사용하는 8개 (분류불가 제외)
CLASSIFICATION_CATEGORIES: tuple[str, ...] = tuple(
    c.value for c in Category if c != Category.UNCLASSIFIED
)


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
    classify_method: str  # rule / embedding / claude_api / claude_rag / review_queue
    version_hint: str = ""
    review_reason: str = ""
    is_new_category: bool = False
    suggested_category: Optional[dict] = None  # {"name": str, "description": str}
