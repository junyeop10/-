"""Tkinter GUI for the classifier with embedding gating and operations UX."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass, replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from src.adaptive import rebuild_adaptive_learning
from src.classifier import (
    ClassificationResult,
    HybridClassifier,
    get_primary_processing_method,
    get_processing_method_label,
    get_processing_trace_text,
)
from src.cli import build_embedder, load_categories
from src.embedding_repository import create_embedding_repository
from src.config import AppConfig, load_app_config
from src.cluster_candidates import ClusterCandidateFinder
from src.document_features import DocumentFeatureExtractor
from src.file_reader import discover_supported_files, ensure_input_directory, extract_text_from_file
from src.hash_utils import compute_xxhash64
from src.operations import commit_move_batch, preview_move_plan, restore_batch, restore_file, undo_last_move
from src.ocr_support import DEFAULT_OCR_MIN_CHARS, explain_ocr_decision, ocr_pdf_file
from src.performance import build_file_latency_analysis, summarize_payload_profiles
from src.recovery import create_safety_snapshot
from src.rule_classifier import RuleBasedClassifier
from src.storage import ClassificationRepository
from src.taxonomy import Taxonomy, load_taxonomy
from src.text_cleaner import normalize_text
from src.type_classifier import TypeClassifier
from src.vectorizer import SentenceTransformerEmbedder


DB_PATH = "data/classifier.db"
CATEGORIES_PATH = "data/categories.json"
EMBEDDING_STATE_NOT_STARTED = "not_started"
EMBEDDING_STATE_LOADING = "loading"
EMBEDDING_STATE_READY = "ready"
EMBEDDING_STATE_FAILED = "failed"
EMBEDDING_STATE_DISABLED = "disabled"
EMBEDDING_LOAD_WARNING_SECONDS = 20.0
GUI_IMPORT_STARTED_AT = time.perf_counter()

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    BaseWindow = TkinterDnD.Tk
    DRAG_AND_DROP_AVAILABLE = True
except ImportError:
    DND_FILES = ""
    BaseWindow = tk.Tk
    DRAG_AND_DROP_AVAILABLE = False


def classification_allowed_for_embedding_state(state: str) -> bool:
    return state == EMBEDDING_STATE_READY


def embedding_state_status_text(state: str, error_message: str = "") -> str:
    if state == EMBEDDING_STATE_NOT_STARTED:
        return "Embedding model not started"
    if state == EMBEDDING_STATE_LOADING:
        return "Embedding model loading..."
    if state == EMBEDDING_STATE_READY:
        return "Embedding model ready"
    if state == EMBEDDING_STATE_FAILED:
        suffix = f": {error_message}" if error_message else ""
        return f"Embedding model failed{suffix}"
    if state == EMBEDDING_STATE_DISABLED:
        return "Embedding model disabled"
    return "Embedding state unknown"


def classification_block_reason_for_state(state: str) -> str:
    if state in {EMBEDDING_STATE_NOT_STARTED, EMBEDDING_STATE_LOADING}:
        return "Embedding model is still loading. Classification will be available soon."
    if state == EMBEDDING_STATE_FAILED:
        return "Embedding model failed to load. Classification is disabled in GUI."
    if state == EMBEDDING_STATE_DISABLED:
        return "Embedding model is disabled. Classification is unavailable in GUI."
    return ""


def group_payloads_by_category(
    payloads: list[dict[str, object]],
    query: str = "",
    category_filter: str = "all",
) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    lowered_query = query.strip().lower()
    normalized_filter = category_filter.strip().lower()
    for payload in payloads:
        result = payload.get("result")
        if not isinstance(result, ClassificationResult):
            continue
        category = result.predicted_category
        if normalized_filter not in {"", "all", "전체"} and category.lower() != normalized_filter:
            continue
        file_name = str(payload.get("file_name", ""))
        if lowered_query and lowered_query not in file_name.lower():
            continue
        grouped.setdefault(category, []).append(payload)
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def summarize_processing_methods(payloads: list[dict[str, object]]) -> dict[str, int]:
    counts = {"rule": 0, "embedding": 0, "llm": 0}
    for payload in payloads:
        result = payload.get("result")
        if not isinstance(result, ClassificationResult):
            continue
        counts[get_primary_processing_method(result)] += 1
    return counts


def build_user_rationale_summary(result: ClassificationResult, payload: dict[str, object] | None = None) -> str:
    """Build a Korean, user-facing explanation without raw debug JSON."""
    del payload
    predicted_type = result.predicted_type or result.predicted_category
    confidence_label = _confidence_label(result.confidence)
    lines = [
        f"이 문서는 '{predicted_type}' 유형으로 판단했습니다.",
        f"분류 신뢰도는 {result.confidence:.3f}로 {confidence_label} 수준입니다.",
    ]

    if result.review_required:
        reasons = _translate_review_reasons(result.review_reasons)
        if reasons:
            lines.append("다만 " + ", ".join(reasons) + " 때문에 검토가 필요합니다.")
        else:
            lines.append("다만 여러 판단 신호가 충분히 강하지 않아 검토가 필요합니다.")
    else:
        lines.append("현재 기준으로는 자동 분류해도 되는 문서로 보입니다.")

    evidence_lines: list[str] = []
    if result.matched_rules:
        evidence_lines.append(f"규칙 근거: '{result.matched_rules[0]}' 신호가 감지되었습니다.")
    if result.embedding_used and result.similarity_score > 0:
        evidence_lines.append(f"유사 문서 근거: 기존 확인 문서와의 유사도가 {result.similarity_score:.3f}입니다.")
    if result.type_confidence > 0:
        evidence_lines.append(f"ML 유형 판단: '{predicted_type}' 쪽 점수가 가장 높았습니다.")
    layout_tags = [str(item.get("tag")) for item in result.suggested_tags if str(item.get("source", "")) == "layout"]
    if layout_tags:
        evidence_lines.append("레이아웃 근거: " + ", ".join(layout_tags[:3]) + " 구조와 비슷합니다.")
    if result.cluster_candidate_id is not None:
        evidence_lines.append("새 카테고리 후보 그룹에 포함될 가능성이 있어 후보로 저장했습니다.")

    if evidence_lines:
        lines.append("")
        lines.append("판단 근거")
        lines.extend(f"- {line}" for line in evidence_lines)

    tag_names = [str(item.get("tag")) for item in result.suggested_tags if not str(item.get("tag", "")).startswith("type:")]
    if tag_names:
        lines.append("")
        lines.append("추천 태그: " + ", ".join(tag_names[:6]))

    lines.append("")
    lines.append("자세한 점수와 내부 근거는 '더보기'에서 확인할 수 있습니다.")
    return "\n".join(lines)


def build_debug_detail(result: ClassificationResult, payload: dict[str, object], performance: dict[str, Any]) -> str:
    """Build the verbose internal detail shown behind the more/less toggle."""
    matched_rules = ", ".join(result.matched_rules) if result.matched_rules else "없음"
    similarity_text = f"{result.similarity_score:.3f}" if result.embedding_used else "skipped"
    analysis = performance.get("analysis", {}) if isinstance(performance, dict) else {}
    if not isinstance(analysis, dict):
        analysis = {}
    stage_timings = performance.get("stage_timings", {}) if isinstance(performance, dict) else {}
    if not isinstance(stage_timings, dict):
        stage_timings = {}
    detail = (
        "\n\n--- 상세 정보 ---\n"
        f"파일: {payload.get('file_name', '')}\n"
        f"경로: {payload.get('file_path', '')}\n"
        f"추천 카테고리: {result.predicted_category}\n"
        f"계층: {result.large_category}/{result.middle_category}\n"
        f"predicted_type: {result.predicted_type or result.predicted_category}\n"
        f"type_confidence: {result.type_confidence:.3f}\n"
        f"confidence: {result.confidence:.3f}\n"
        f"review_required: {'yes' if result.review_required else 'no'}\n"
        f"review_reasons: {', '.join(result.review_reasons) if result.review_reasons else 'none'}\n"
        f"suggested_tags: {json.dumps(result.suggested_tags, ensure_ascii=False)}\n"
        f"cluster_candidate_id: {result.cluster_candidate_id if result.cluster_candidate_id is not None else 'none'}\n"
        f"processing: {get_processing_trace_text(result)}\n"
        f"similarity: {similarity_text}\n"
        f"점수: rule={result.rule_score:.3f}, embedding={result.embedding_score:.3f}, "
        f"feedback={result.feedback_score:.3f}, final={result.final_score:.3f}\n"
        f"매칭 규칙: {matched_rules}\n"
        f"후보 점수: {json.dumps(result.candidate_scores, ensure_ascii=False)}\n"
        f"ml_evidence: {json.dumps(result.ml_evidence, ensure_ascii=False)}\n"
        f"rule_evidence: {json.dumps(result.rule_evidence, ensure_ascii=False)}\n"
        f"근거 원문: {result.reasoning}\n"
    )
    stage_lines = "\n".join(
        f"  - {key}: {float(value):.3f}s"
        for key, value in sorted(stage_timings.items(), key=lambda item: float(item[1]), reverse=True)
        if key != "total"
    )
    reason_lines = "\n".join(f"  - {reason}" for reason in analysis.get("reasons", []))
    detail += (
        f"\nperformance_total: {float(analysis.get('total_time', stage_timings.get('total', 0.0))):.3f}s\n"
        f"dominant_stage: {analysis.get('dominant_stage', 'unknown')}\n"
        f"latency_summary: {analysis.get('summary', '')}\n"
        f"stage_breakdown:\n{stage_lines or '  - no stages recorded'}\n"
        f"latency_reasons:\n{reason_lines or '  - no latency reasons recorded'}\n"
    )
    return detail


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.8:
        return "높은"
    if confidence >= 0.6:
        return "보통"
    return "낮은"


def _translate_review_reasons(reasons: list[str]) -> list[str]:
    translations = {
        "low_confidence": "신뢰도가 낮음",
        "small_margin": "1순위와 2순위 차이가 작음",
        "rule_ml_conflict": "규칙 판단과 ML 판단이 다름",
        "embedding_ml_conflict": "유사 문서 판단과 ML 판단이 다름",
        "low_similarity_new_cluster": "기존 카테고리와 유사도가 낮음",
        "legacy_ambiguity": "기존 점수 기준에서도 애매함",
        "pending_category_candidate": "새 카테고리 후보 가능성",
    }
    translated: list[str] = []
    for reason in reasons:
        if reason.startswith("layout_") and reason.endswith("_conflict"):
            translated.append("문서 레이아웃과 텍스트 판단이 다름")
        else:
            translated.append(translations.get(reason, reason))
    return translated


def upsert_payload_by_file_path(
    payloads: list[dict[str, object]],
    payload: dict[str, object],
) -> list[dict[str, object]]:
    """Replace an existing payload for the same file path instead of duplicating it."""
    target_path = str(payload.get("file_path", "")).strip()
    if not target_path:
        return [*payloads, payload]
    updated: list[dict[str, object]] = []
    replaced = False
    for existing in payloads:
        existing_path = str(existing.get("file_path", "")).strip()
        if existing_path == target_path:
            updated.append(payload)
            replaced = True
        else:
            updated.append(existing)
    if not replaced:
        updated.append(payload)
    return updated


def can_drag_tree_meta(meta: dict[str, object] | None) -> bool:
    return bool(meta and meta.get("kind") == "file")


def drop_target_category_from_meta(meta: dict[str, object] | None) -> str:
    if not meta:
        return ""
    if meta.get("kind") == "category":
        return str(meta.get("category", ""))
    if meta.get("kind") == "file":
        payload = meta.get("payload")
        if isinstance(payload, dict):
            result = payload.get("result")
            if isinstance(result, ClassificationResult):
                return result.predicted_category
    return ""


def read_boot_started_at() -> float:
    """Best-effort app boot start time for startup profiling."""
    raw_value = os.environ.get("FILE_CLASSIFIER_BOOT_START", "")
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return GUI_IMPORT_STARTED_AT


@dataclass
class AppResources:
    """Prepared objects reused across the GUI session."""

    repository: ClassificationRepository
    embedder: SentenceTransformerEmbedder
    rule_classifier: RuleBasedClassifier
    config: AppConfig
    taxonomy: Taxonomy


class ClassifierGui(BaseWindow):
    """Desktop UI for the document classifier."""

    def __init__(self) -> None:
        self.app_boot_started_at = read_boot_started_at()
        window_init_start = time.perf_counter()
        super().__init__()
        self.title("파일 분류 MVP")
        self.geometry("1320x780")
        self.minsize(1100, 660)
        self.protocol("WM_DELETE_WINDOW", self._handle_close)

        self.startup_profile: dict[str, Any] = {"stages": {"window_init": time.perf_counter() - window_init_start}}
        config_start = time.perf_counter()
        config = load_app_config()
        self.startup_profile["stages"]["config_load"] = time.perf_counter() - config_start
        taxonomy_start = time.perf_counter()
        taxonomy = load_taxonomy(Path(CATEGORIES_PATH))
        self.startup_profile["stages"]["taxonomy_load"] = time.perf_counter() - taxonomy_start
        repository_start = time.perf_counter()
        repository = ClassificationRepository(DB_PATH)
        repository.attach_embedding_repository(create_embedding_repository(config))
        repository.initialize_database()
        self.startup_profile["stages"]["db_init"] = time.perf_counter() - repository_start
        rules_start = time.perf_counter()
        repository.seed_rules_from_categories(load_categories(Path(CATEGORIES_PATH)))
        self.startup_profile["stages"]["rules_load"] = time.perf_counter() - rules_start
        embedder_start = time.perf_counter()
        embedder = build_embedder(config)
        self.startup_profile["stages"]["embedder_init"] = time.perf_counter() - embedder_start
        self.resources = AppResources(
            repository=repository,
            embedder=embedder,
            rule_classifier=RuleBasedClassifier(repository),
            config=config,
            taxonomy=taxonomy,
        )

        self.embedding_state = EMBEDDING_STATE_NOT_STARTED
        self.embedding_error_message = ""
        self.embedding_ready = False
        self.embedding_load_started_at = 0.0
        self.last_preview_batch_id: int | None = None
        self.last_preview_manifest_path = ""
        self.last_preview_items: list[dict[str, Any]] = []
        self.last_run_summary: dict[str, Any] = {}
        self.current_run_profile: dict[str, Any] = {}

        self.input_dir = tk.StringVar(value=str(Path("input_files").resolve()))
        self.final_category = tk.StringVar()
        self.search_query = tk.StringVar()
        self.category_filter = tk.StringVar(value="전체")
        self.status_text = tk.StringVar(value="준비 완료")
        self.progress_text = tk.StringVar(value="진행률 0/0")
        self.processing_summary_text = tk.StringVar(value="룰 0 | 임베딩 0 | LLM 0")
        self.progress_value = tk.DoubleVar(value=0.0)
        self.embedding_status_text = tk.StringVar(value=embedding_state_status_text(self.embedding_state))
        self.operation_status_text = tk.StringVar(value="이동 미리보기 없음")

        self.result_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.embedding_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.all_payloads: list[dict[str, object]] = []
        self.tree_meta: dict[str, dict[str, object]] = {}
        self.category_combo: ttk.Combobox | None = None
        self.total_files = 0
        self.processed_files = 0
        self.drag_source_item_id: str | None = None
        self.drag_source_payload: dict[str, object] | None = None
        self.drag_target_item_id: str | None = None
        self.force_open_categories: set[str] = set()
        self.detail_text: tk.Text | None = None
        self.detail_more_button: ttk.Button | None = None
        self.current_detail_summary = ""
        self.current_detail_debug = ""
        self.detail_more_expanded = False
        self.stats_label: ttk.Label | None = None
        self.drop_label: ttk.Label | None = None
        self.tree: ttk.Treeview | None = None
        self.progress_bar: ttk.Progressbar | None = None
        self.classify_button: ttk.Button | None = None
        self.review_save_button: ttk.Button | None = None
        self.preview_move_button: ttk.Button | None = None
        self.commit_move_button: ttk.Button | None = None

        self.search_query.trace_add("write", lambda *_args: self.apply_filename_filter())
        self.category_filter.trace_add("write", lambda *_args: self.apply_category_filter())

        ui_build_start = time.perf_counter()
        self._build_main_ui()
        self.startup_profile["stages"]["ui_build"] = time.perf_counter() - ui_build_start
        stats_start = time.perf_counter()
        self.refresh_stats()
        self.startup_profile["stages"]["stats_refresh"] = time.perf_counter() - stats_start
        self.startup_profile["main_window_ready_total"] = time.perf_counter() - self.app_boot_started_at
        self._set_embedding_state(EMBEDDING_STATE_NOT_STARTED)
        self._start_embedding_warmup()

    def _build_main_ui(self) -> None:
        top_bar = ttk.Frame(self, padding=10)
        top_bar.pack(fill="x")

        ttk.Label(top_bar, text="입력 폴더").pack(side="left")
        ttk.Entry(top_bar, textvariable=self.input_dir).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(top_bar, text="폴더 선택", command=self.choose_folder).pack(side="left")
        ttk.Button(top_bar, text="DB 초기화", command=self.init_db).pack(side="left", padx=(8, 0))
        self.classify_button = ttk.Button(top_bar, text="분류 실행", command=self.start_classify)
        self.classify_button.pack(side="left", padx=(8, 0))
        ttk.Button(top_bar, text="통계", command=self.refresh_stats).pack(side="left", padx=(8, 0))
        ttk.Button(top_bar, text="이동 미리보기", command=self.preview_moves).pack(side="left", padx=(16, 0))
        self.commit_move_button = ttk.Button(top_bar, text="미리보기 커밋", command=self.commit_last_preview)
        self.commit_move_button.pack(side="left", padx=(8, 0))
        ttk.Button(top_bar, text="마지막 이동 되돌리기", command=self.undo_last_move_ui).pack(side="left", padx=(8, 0))

        drop_text = "여기에 txt/pdf/docx/xlsx/pptx 파일 또는 폴더를 드래그하세요"
        if not DRAG_AND_DROP_AVAILABLE:
            drop_text = "드래그 앤 드롭은 tkinterdnd2 설치 후 사용할 수 있습니다"
        self.drop_label = ttk.Label(self, text=drop_text, anchor="center", padding=12, relief="ridge")
        self.drop_label.pack(fill="x", padx=10, pady=(0, 10))
        if DRAG_AND_DROP_AVAILABLE:
            self.drop_label.drop_target_register(DND_FILES)
            self.drop_label.dnd_bind("<<Drop>>", self.on_drop_files)

        main_pane = ttk.PanedWindow(self, orient="horizontal")
        main_pane.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        left_frame = ttk.Frame(main_pane)
        right_frame = ttk.Frame(main_pane, padding=(10, 0, 0, 0))
        main_pane.add(left_frame, weight=4)
        main_pane.add(right_frame, weight=3)

        search_frame = ttk.Frame(left_frame)
        search_frame.pack(fill="x", pady=(0, 8))
        ttk.Label(search_frame, text="파일명 검색").pack(side="left")
        ttk.Entry(search_frame, textvariable=self.search_query).pack(side="left", fill="x", expand=True, padx=(8, 0))
        ttk.Label(search_frame, text="카테고리").pack(side="left", padx=(8, 0))
        self.category_combo = ttk.Combobox(
            search_frame,
            textvariable=self.category_filter,
            values=["전체"],
            state="readonly",
            width=16,
        )
        self.category_combo.pack(side="left", padx=(8, 0))
        ttk.Button(search_frame, text="초기화", command=self.clear_filename_filter).pack(side="left", padx=(8, 0))

        table_frame = ttk.Frame(left_frame)
        table_frame.pack(fill="both", expand=True)
        columns = ("kind", "score", "meta")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="tree headings", height=24)
        self.tree.heading("#0", text="카테고리 / 파일")
        self.tree.heading("kind", text="종류")
        self.tree.heading("score", text="점수/개수")
        self.tree.heading("meta", text="메타")
        self.tree.column("#0", width=340, anchor="w")
        self.tree.column("kind", width=90, anchor="center")
        self.tree.column("score", width=100, anchor="center")
        self.tree.column("meta", width=180, anchor="w")
        self.tree.tag_configure("ocr_used", foreground="#6c6c6c")
        self.tree.tag_configure("drag_source", background="#fff4cc")
        self.tree.tag_configure("drag_target", background="#d9f2e6")
        self.tree.bind("<<TreeviewSelect>>", self.on_select_result)
        self.tree.bind("<ButtonPress-1>", self.on_tree_drag_start)
        self.tree.bind("<B1-Motion>", self.on_tree_drag_motion)
        self.tree.bind("<ButtonRelease-1>", self.on_tree_drag_release)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

        progress_frame = ttk.Frame(left_frame)
        progress_frame.pack(fill="x", pady=(8, 0))
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_value, maximum=100, mode="determinate")
        self.progress_bar.pack(side="left", fill="x", expand=True)
        ttk.Label(progress_frame, textvariable=self.progress_text, width=16, anchor="e").pack(side="left", padx=(8, 0))
        ttk.Label(progress_frame, textvariable=self.processing_summary_text, width=28, anchor="e").pack(side="left", padx=(8, 0))

        detail_header = ttk.Frame(right_frame)
        detail_header.pack(fill="x")
        ttk.Label(detail_header, text="상세 정보").pack(side="left", anchor="w")
        self.detail_more_button = ttk.Button(detail_header, text="더보기", command=self.toggle_detail_more)
        self.detail_more_button.pack(side="right")
        self.detail_text = tk.Text(right_frame, height=16, wrap="word")
        self.detail_text.pack(fill="both", expand=True, pady=(4, 10))

        review_frame = ttk.LabelFrame(right_frame, text="분류 검토", padding=10)
        review_frame.pack(fill="x")
        ttk.Label(review_frame, text="최종 카테고리").pack(anchor="w")
        ttk.Entry(review_frame, textvariable=self.final_category).pack(fill="x", pady=(4, 8))
        self.review_save_button = ttk.Button(review_frame, text="선택 결과 확정/수정 저장", command=self.save_selected_feedback)
        self.review_save_button.pack(fill="x")

        operations_frame = ttk.LabelFrame(right_frame, text="작업 패널", padding=10)
        operations_frame.pack(fill="x", pady=(10, 0))
        ttk.Label(operations_frame, textvariable=self.operation_status_text, justify="left").pack(anchor="w", pady=(0, 8))
        ttk.Button(operations_frame, text="이동 미리보기 보기", command=self.preview_moves).pack(fill="x")
        ttk.Button(operations_frame, text="이동 이력 보기", command=self.show_move_history_window).pack(fill="x", pady=(6, 0))
        ttk.Button(operations_frame, text="피드백 로그 관리", command=self.show_feedback_logs_window).pack(fill="x", pady=(6, 0))
        ttk.Button(operations_frame, text="임베딩 캐시 관리", command=self.show_embedding_cache_window).pack(fill="x", pady=(6, 0))
        ttk.Button(operations_frame, text="성능 분석 보기", command=self.show_performance_window).pack(fill="x", pady=(6, 0))

        status_frame = ttk.LabelFrame(right_frame, text="시스템 상태", padding=10)
        status_frame.pack(fill="x", pady=(10, 0))
        ttk.Label(status_frame, text="DB ready / Rules ready / OCR cache ready", justify="left").pack(anchor="w")
        ttk.Label(status_frame, textvariable=self.embedding_status_text, justify="left").pack(anchor="w", pady=(4, 0))

        stats_frame = ttk.LabelFrame(right_frame, text="DB 통계", padding=10)
        stats_frame.pack(fill="x", pady=(10, 0))
        self.stats_label = ttk.Label(stats_frame, text="", justify="left")
        self.stats_label.pack(anchor="w")

        ttk.Label(self, textvariable=self.status_text, padding=(10, 0, 10, 8)).pack(fill="x")

    def _start_embedding_warmup(self) -> None:
        self.embedding_load_started_at = time.perf_counter()
        self._set_embedding_state(EMBEDDING_STATE_LOADING)
        self.status_text.set("메인 화면 준비 완료 | 임베딩 모델 백그라운드 로딩 중")
        worker = threading.Thread(target=self._embedding_warmup_worker, daemon=True)
        worker.start()
        self.after(150, self._poll_embedding_queue)

    def _embedding_warmup_worker(self) -> None:
        try:
            self.resources.embedder.encode("초기화")
            self.embedding_queue.put(("ready", None))
        except Exception as error:
            self.embedding_queue.put(("error", str(error)))

    def _poll_embedding_queue(self) -> None:
        while not self.embedding_queue.empty():
            event, payload = self.embedding_queue.get()
            if event == "ready":
                self.startup_profile["stages"]["embedding_check"] = time.perf_counter() - self.embedding_load_started_at
                self.startup_profile["startup_ready_total"] = time.perf_counter() - self.app_boot_started_at
                self._set_embedding_state(EMBEDDING_STATE_READY)
                self.status_text.set("준비 완료 | 임베딩 모델 로드됨")
                return
            if event == "error":
                self.startup_profile["stages"]["embedding_check"] = time.perf_counter() - self.embedding_load_started_at
                self.startup_profile["startup_ready_total"] = time.perf_counter() - self.app_boot_started_at
                self.startup_profile["embedding_error"] = str(payload)
                self._set_embedding_state(EMBEDDING_STATE_FAILED, str(payload))
                self.status_text.set(f"준비 완료 | 임베딩 로드 실패: {payload}")
                return

        if self.embedding_state == EMBEDDING_STATE_LOADING:
            elapsed = time.perf_counter() - self.embedding_load_started_at
            if elapsed >= EMBEDDING_LOAD_WARNING_SECONDS:
                self.embedding_status_text.set(
                    f"{embedding_state_status_text(self.embedding_state)} (taking longer than expected)"
                )
            self.after(150, self._poll_embedding_queue)

    def _set_embedding_state(self, state: str, error_message: str = "") -> None:
        self.embedding_state = state
        self.embedding_error_message = error_message
        self.embedding_ready = state == EMBEDDING_STATE_READY
        self.embedding_status_text.set(embedding_state_status_text(state, error_message))
        self._update_classification_controls()

    def _update_classification_controls(self) -> None:
        is_allowed = classification_allowed_for_embedding_state(self.embedding_state)
        button_state = "normal" if is_allowed else "disabled"
        if self.classify_button is not None:
            self.classify_button.configure(state=button_state)
        if self.drop_label is not None:
            block_reason = classification_block_reason_for_state(self.embedding_state)
            base_text = (
                "여기에 txt/pdf/docx/xlsx/pptx 파일 또는 폴더를 드래그하세요"
                if DRAG_AND_DROP_AVAILABLE
                else "드래그 앤 드롭은 tkinterdnd2 설치 후 사용할 수 있습니다"
            )
            self.drop_label.configure(text=f"{base_text}\n{block_reason}" if block_reason else base_text)
        if self.review_save_button is not None:
            self.review_save_button.configure(state="normal")
        if self.commit_move_button is not None:
            self.commit_move_button.configure(state="normal" if self.last_preview_batch_id is not None else "disabled")

    def _ensure_classification_available(self) -> bool:
        if classification_allowed_for_embedding_state(self.embedding_state):
            return True
        reason = classification_block_reason_for_state(self.embedding_state)
        self.status_text.set(reason)
        messagebox.showinfo("안내", reason)
        return False

    def choose_folder(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.input_dir.get() or ".")
        if selected:
            self.input_dir.set(selected)

    def init_db(self) -> None:
        self.resources.repository.initialize_database()
        self.resources.repository.seed_rules_from_categories(load_categories(Path(CATEGORIES_PATH)))
        self.resources.repository.seed_default_category_profiles()
        self.refresh_stats()
        self.status_text.set("DB 초기화 완료")

    def refresh_stats(self) -> None:
        if self.stats_label is None:
            return
        stats = self.resources.repository.get_stats()
        self.stats_label.configure(
            text=(
                f"files: {stats['files_count']}\n"
                f"classifications: {stats['classifications_count']}\n"
                f"feedback_logs: {stats['feedback_logs_count']}\n"
                f"confirmed_examples: {stats['confirmed_examples_count']}\n"
                f"rules: {stats['rules_count']}\n"
                f"move_batches: {stats.get('move_batches_count', 0)}\n"
                f"embedding_cache: {stats.get('embedding_cache_count', 0)}\n"
                f"document_features: {stats.get('document_features_count', 0)}\n"
                f"category_candidates: {stats.get('category_candidates_count', 0)}\n"
                f"document_tags: {stats.get('document_tags_count', 0)}\n"
                f"category_profiles: {stats.get('category_profiles_count', 0)}"
            )
        )

    def start_classify(self) -> None:
        if not self._ensure_classification_available():
            return
        self._clear_results()
        self.status_text.set("파일 목록 준비 중")
        input_dir = ensure_input_directory(self.input_dir.get())
        files = discover_supported_files(input_dir)
        self.start_classify_files(files)

    def start_classify_files(self, files: list[Path]) -> None:
        if not self._ensure_classification_available():
            return
        files = sorted({file_path.resolve() for file_path in files})
        if not files:
            messagebox.showinfo("안내", "분류할 지원 파일이 없습니다.")
            return

        self.status_text.set("분류 엔진 준비 중")
        self.current_run_profile = {"started_at": time.perf_counter(), "file_count": len(files)}
        self.last_run_summary = {}
        self._reset_progress(len(files))
        worker = threading.Thread(target=self._classify_worker, args=(files,), daemon=True)
        worker.start()
        self.after(100, self._drain_queue)

    def on_drop_files(self, event: object) -> None:
        if not self._ensure_classification_available():
            return
        raw_data = getattr(event, "data", "")
        dropped_paths = [Path(value) for value in self.tk.splitlist(raw_data)]
        files: list[Path] = []
        for path in dropped_paths:
            if path.is_dir():
                files.extend(discover_supported_files(path))
            elif path.is_file() and path.suffix.lower() in {".txt", ".pdf", ".docx", ".xlsx", ".pptx"}:
                files.append(path)

        self._clear_results()
        self.start_classify_files(sorted(set(files)))

    def _classify_worker(self, files: list[Path]) -> None:
        self.result_queue.put(("status", "분류 엔진 준비 중"))
        classifier = HybridClassifier(
            repository=self.resources.repository,
            embedder=self.resources.embedder,
            rule_classifier=self.resources.rule_classifier,
            taxonomy=self.resources.taxonomy,
            feature_extractor=DocumentFeatureExtractor(version=self.resources.config.features.extractor_version),
            type_classifier=TypeClassifier(
                version=self.resources.config.ml.type_classifier_version,
                min_examples=self.resources.config.ml.min_training_examples,
                filename_weight=self.resources.config.ml.filename_weight,
            ),
        )
        self.result_queue.put(("status", f"분류 시작: {len(files)}개"))

        for index, file_path in enumerate(files, start=1):
            try:
                self.result_queue.put(("start_file", {"index": index, "file_name": file_path.name}))
                payload = self._classify_one(classifier, file_path)
                self.result_queue.put(("result", payload))
            except Exception as error:
                self.result_queue.put(("error", f"{file_path.name}: {error}"))

        self.result_queue.put(("done", None))

    def _classify_one(self, classifier: HybridClassifier, file_path: Path) -> dict[str, object]:
        file_start = time.perf_counter()
        stage_timings: dict[str, float] = {}
        extract_start = time.perf_counter()
        raw_text = extract_text_from_file(file_path, fast=True)
        stage_timings["read_extract"] = time.perf_counter() - extract_start
        normalize_start = time.perf_counter()
        normalized_text = normalize_text(raw_text)
        stage_timings["normalize"] = time.perf_counter() - normalize_start
        ocr_used = False
        ocr_pages = 0
        file_size = file_path.stat().st_size

        ocr_decision_start = time.perf_counter()
        ocr_decision = explain_ocr_decision(
            file_path=file_path,
            extracted_text=normalized_text,
            min_text_length=DEFAULT_OCR_MIN_CHARS,
        )
        stage_timings["ocr_decision"] = time.perf_counter() - ocr_decision_start
        if ocr_decision["run_ocr"]:
            ocr_start = time.perf_counter()
            ocr_result = ocr_pdf_file(file_path)
            stage_timings["ocr"] = time.perf_counter() - ocr_start
            ocr_text = normalize_text(str(ocr_result.get("text", "")))
            if ocr_result.get("ok") and ocr_text:
                normalized_text = ocr_text
                ocr_used = True
                ocr_pages = int(ocr_result.get("pages_scanned", 0))
        elif ocr_decision["classification_hint"]:
            hint_evidence = str(ocr_decision.get("hint_evidence", "")).strip()
            if hint_evidence:
                normalized_text = f"{hint_evidence} {normalized_text}".strip()

        hash_start = time.perf_counter()
        file_hash = compute_xxhash64(file_path)
        stage_timings["hash"] = time.perf_counter() - hash_start
        duplicate_start = time.perf_counter()
        duplicate_of_file_id = self.resources.repository.find_duplicate_file_id(file_hash, str(file_path.resolve()))
        stage_timings["duplicate_lookup"] = time.perf_counter() - duplicate_start

        upsert_start = time.perf_counter()
        file_id = self.resources.repository.upsert_file(
            file_path=str(file_path.resolve()),
            file_name=file_path.name,
            file_ext=file_path.suffix.lower(),
            file_size=file_size,
            xxhash64=file_hash,
            duplicate_of_file_id=duplicate_of_file_id,
            extracted_text=normalized_text,
        )
        stage_timings["db_upsert"] = time.perf_counter() - upsert_start
        classify_start = time.perf_counter()
        document_features = DocumentFeatureExtractor(version=self.resources.config.features.extractor_version).extract(
            file_name=file_path.name,
            file_ext=file_path.suffix.lower(),
            text=normalized_text,
            file_size=file_size,
            file_path=file_path,
        ).to_storage_dict()
        result = classifier.classify_file(
            file_id=file_id,
            file_hash=file_hash,
            text=normalized_text,
            duplicate_of_file_id=duplicate_of_file_id,
            file_name=file_path.name,
            document_features=document_features,
        )
        stage_timings["classification"] = time.perf_counter() - classify_start
        if ocr_used:
            result = replace(
                result,
                reasoning=f"{result.reasoning} | ocr=used(pages={ocr_pages})",
                ocr_used=True,
                explanation={**result.explanation, "ocr_used": True, "ocr_pages": ocr_pages},
            )
        classifier_profile = result.processing_profile if isinstance(result.processing_profile, dict) else {}
        classifier_stage_timings = classifier_profile.get("stage_timings", {}) if isinstance(classifier_profile, dict) else {}
        for key, value in classifier_stage_timings.items():
            if key == "total":
                continue
            stage_timings[f"classifier_{key}"] = float(value)
        persist_start = time.perf_counter()
        performance_profile = {
            "stage_timings": stage_timings,
            "analysis": {},
            "classifier_profile": classifier_profile,
        }
        embedding_meta = classifier_profile.get("embedding_meta", {}) if isinstance(classifier_profile, dict) else {}
        performance_profile["analysis"] = build_file_latency_analysis(
            {**stage_timings, "total": time.perf_counter() - file_start},
            text_length=len(normalized_text),
            file_size=file_size,
            ocr_used=ocr_used,
            ocr_status="used" if ocr_used else str(ocr_decision.get("reason", "")),
            ocr_pages=ocr_pages,
            embedding_used=result.embedding_used,
            embedding_cache_hit=embedding_meta.get("cache_hit") if isinstance(embedding_meta, dict) else None,
            strong_rule_match=bool(classifier_profile.get("strong_rule_match")) if isinstance(classifier_profile, dict) else False,
            review_required=result.review_required,
            matched_rules_count=len(result.matched_rules),
            llm_used=result.llm_used,
            duplicate_detected=duplicate_of_file_id is not None,
        )
        result = replace(result, processing_profile=performance_profile)
        result = self._attach_cluster_candidate_if_needed(result, file_id=file_id)
        classification_id = classifier.persist_classification(file_id=file_id, result=result)
        stage_timings["db_persist"] = time.perf_counter() - persist_start
        stage_timings["total"] = time.perf_counter() - file_start
        performance_profile["stage_timings"] = stage_timings
        performance_profile["analysis"] = build_file_latency_analysis(
            stage_timings,
            text_length=len(normalized_text),
            file_size=file_size,
            ocr_used=ocr_used,
            ocr_status="used" if ocr_used else "skipped",
            ocr_pages=ocr_pages,
            embedding_used=result.embedding_used,
            embedding_cache_hit=embedding_meta.get("cache_hit") if isinstance(embedding_meta, dict) else None,
            strong_rule_match=bool(classifier_profile.get("strong_rule_match")) if isinstance(classifier_profile, dict) else False,
            review_required=result.review_required,
            matched_rules_count=len(result.matched_rules),
            llm_used=result.llm_used,
            duplicate_detected=duplicate_of_file_id is not None,
        )
        result = replace(result, processing_profile=performance_profile)

        return {
            "file_id": file_id,
            "classification_id": classification_id,
            "file_name": file_path.name,
            "file_path": str(file_path.resolve()),
            "text": normalized_text,
            "ocr_used": ocr_used,
            "result": result,
            "performance": performance_profile,
        }

    def _attach_cluster_candidate_if_needed(self, result: ClassificationResult, *, file_id: int) -> ClassificationResult:
        if not self.resources.config.clustering.enabled or not result.review_required:
            return result
        finder = ClusterCandidateFinder(
            min_cluster_size=self.resources.config.clustering.min_cluster_size,
            max_candidates=self.resources.config.clustering.max_candidates,
        )
        rows = self.resources.repository.fetch_cluster_candidate_rows()
        rows.append(
            {
                "file_id": file_id,
                "file_name": "",
                "text": "",
                "predicted_category": result.predicted_category,
                "predicted_type": result.predicted_type,
                "review_required": result.review_required,
                "compressed_text": " ".join(result.evidence_snippets),
            }
        )
        for candidate in finder.find_candidates(rows):
            if file_id not in candidate.representative_file_ids:
                continue
            candidate_id = self.resources.repository.insert_category_candidate(
                source=str(candidate.evidence.get("source", "cluster")),
                suggested_name=candidate.suggested_name,
                representative_file_ids=candidate.representative_file_ids,
                evidence=candidate.evidence,
                status="pending",
            )
            return replace(
                result,
                cluster_candidate_id=candidate_id,
                review_reasons=[*result.review_reasons, "pending_category_candidate"],
            )
        return result

    def _drain_queue(self) -> None:
        while not self.result_queue.empty():
            event, payload = self.result_queue.get()
            if event == "status":
                self.status_text.set(str(payload))
            elif event == "start_file":
                self._show_current_file(payload)  # type: ignore[arg-type]
            elif event == "result":
                self._insert_result(payload)  # type: ignore[arg-type]
                self._advance_progress()
            elif event == "error":
                self._append_detail(f"[처리 실패] {payload}\n")
                self._advance_progress()
            elif event == "done":
                suffix = " | 임베딩 모델 로드됨" if self.embedding_ready else f" | {embedding_state_status_text(self.embedding_state)}"
                self.status_text.set(f"분류 완료{suffix}")
                self.current_run_profile["elapsed"] = time.perf_counter() - float(
                    self.current_run_profile.get("started_at", time.perf_counter())
                )
                self.last_run_summary = summarize_payload_profiles(
                    self.all_payloads,
                    startup_profile=self.startup_profile,
                    run_profile=self.current_run_profile,
                )
                self.status_text.set(
                    f"{self.status_text.get()} | run={self.last_run_summary.get('run_elapsed', 0.0):.2f}s "
                    f"| avg={self.last_run_summary.get('average_file_time', 0.0):.2f}s"
                )
                self._set_progress(self.total_files, self.total_files)
                self.refresh_stats()
                return
        self.after(100, self._drain_queue)

    def _insert_result(self, payload: dict[str, object]) -> None:
        self.all_payloads = upsert_payload_by_file_path(self.all_payloads, payload)
        self._update_processing_summary()
        self._refresh_category_options()
        self._refresh_result_tree()

    def apply_filename_filter(self) -> None:
        self._refresh_result_tree()

    def apply_category_filter(self) -> None:
        self._refresh_result_tree()

    def clear_filename_filter(self) -> None:
        self.search_query.set("")
        self.category_filter.set("전체")

    def _capture_open_categories(self) -> set[str]:
        if self.tree is None:
            return set()
        open_categories: set[str] = set()
        for item_id, meta in self.tree_meta.items():
            if meta.get("kind") == "category" and bool(self.tree.item(item_id, "open")):
                open_categories.add(str(meta.get("category", "")))
        return open_categories

    def _apply_drag_tree_tags(self) -> None:
        if self.tree is None:
            return
        for item_id, meta in self.tree_meta.items():
            tags: list[str] = []
            if bool(meta.get("ocr_used")):
                tags.append("ocr_used")
            if item_id == self.drag_source_item_id:
                tags.append("drag_source")
            if item_id == self.drag_target_item_id:
                tags.append("drag_target")
            self.tree.item(item_id, tags=tuple(tags))

    def _refresh_result_tree(self) -> None:
        if self.tree is None:
            return
        selected_meta = self._selected_tree_meta()
        open_categories = self._capture_open_categories()
        open_categories.update(self.force_open_categories)
        for item_id in self.tree.get_children():
            self.tree.delete(item_id)
        self.tree_meta.clear()

        grouped = group_payloads_by_category(
            self.all_payloads,
            query=self.search_query.get(),
            category_filter=self.category_filter.get(),
        )

        for category, payloads in grouped.items():
            avg_confidence = sum(
                payload["result"].confidence
                for payload in payloads
                if isinstance(payload.get("result"), ClassificationResult)
            ) / max(len(payloads), 1)
            parent_id = self.tree.insert(
                "",
                "end",
                text=category,
                values=("category", f"{len(payloads)} files", f"avg {avg_confidence:.3f}"),
                open=category in open_categories,
            )
            self.tree_meta[parent_id] = {"kind": "category", "category": category, "payloads": payloads}
            for payload in sorted(payloads, key=lambda item: str(item.get("file_name", "")).lower()):
                result = payload["result"]
                assert isinstance(result, ClassificationResult)
                child_id = self.tree.insert(
                    parent_id,
                    "end",
                    text=str(payload["file_name"]),
                    values=(
                        "file",
                        f"{result.final_score:.3f}",
                        get_processing_method_label(result),
                    ),
                    tags=("ocr_used",) if bool(payload.get("ocr_used")) else (),
                )
                self.tree_meta[child_id] = {"kind": "file", "payload": payload, "ocr_used": bool(payload.get("ocr_used"))}

        visible_categories = len(grouped)
        visible_files = sum(len(payloads) for payloads in grouped.values())
        if self.search_query.get().strip() or self.category_filter.get() != "전체":
            self.status_text.set(f"필터 결과: 카테고리 {visible_categories}개 / 파일 {visible_files}개")

        if selected_meta is not None:
            self._restore_tree_selection(selected_meta)
        elif self.tree.get_children():
            first_id = self.tree.get_children()[0]
            self.tree.selection_set(first_id)
            self.tree.focus(first_id)
            self.on_select_result()
        self.force_open_categories.clear()
        self._apply_drag_tree_tags()

    def _selected_tree_meta(self) -> dict[str, object] | None:
        if self.tree is None:
            return None
        selection = self.tree.selection()
        if not selection:
            return None
        return self.tree_meta.get(selection[0])

    def on_tree_drag_start(self, event: tk.Event[tk.Misc]) -> None:
        if self.tree is None:
            return
        item_id = self.tree.identify_row(event.y)
        meta = self.tree_meta.get(item_id)
        if not can_drag_tree_meta(meta):
            self.drag_source_item_id = None
            self.drag_source_payload = None
            self.drag_target_item_id = None
            self._apply_drag_tree_tags()
            return
        self.drag_source_item_id = item_id
        payload = meta.get("payload") if meta else None
        self.drag_source_payload = payload if isinstance(payload, dict) else None
        self.drag_target_item_id = None
        self._apply_drag_tree_tags()

    def on_tree_drag_motion(self, event: tk.Event[tk.Misc]) -> None:
        if self.tree is None or self.drag_source_payload is None:
            return
        item_id = self.tree.identify_row(event.y)
        target_meta = self.tree_meta.get(item_id)
        self.drag_target_item_id = item_id if target_meta and target_meta.get("kind") == "category" else None
        self._apply_drag_tree_tags()
        target_category = drop_target_category_from_meta(target_meta)
        if target_category:
            self.status_text.set(
                f"드래그 이동: {self.drag_source_payload.get('file_name', '')} -> {target_category}"
            )

    def on_tree_drag_release(self, event: tk.Event[tk.Misc]) -> None:
        if self.tree is None or self.drag_source_payload is None:
            self.drag_source_item_id = None
            self.drag_source_payload = None
            self.drag_target_item_id = None
            self._apply_drag_tree_tags()
            return
        item_id = self.tree.identify_row(event.y)
        target_meta = self.tree_meta.get(item_id)
        target_category = drop_target_category_from_meta(target_meta)
        source_result = self.drag_source_payload.get("result")
        if not isinstance(source_result, ClassificationResult):
            self.drag_source_item_id = None
            self.drag_source_payload = None
            self.drag_target_item_id = None
            self._apply_drag_tree_tags()
            return
        if not target_category or target_category == source_result.predicted_category:
            self.drag_source_item_id = None
            self.drag_source_payload = None
            self.drag_target_item_id = None
            self._apply_drag_tree_tags()
            return
        self.move_selected_file_to_category(self.drag_source_payload, target_category)
        self.drag_source_item_id = None
        self.drag_source_payload = None
        self.drag_target_item_id = None
        self._apply_drag_tree_tags()

    def move_selected_file_to_category(self, payload: dict[str, object], target_category: str) -> None:
        result = payload.get("result")
        if not isinstance(result, ClassificationResult):
            return
        self.force_open_categories.update({result.predicted_category, target_category})
        taxonomy_entry = self.resources.taxonomy.resolve(target_category)
        action = "confirmed" if target_category == result.predicted_category else "corrected"
        feedback_id = self.resources.repository.save_feedback(
            file_id=int(payload["file_id"]),
            classification_id=int(payload["classification_id"]),
            predicted_category=result.predicted_category,
            final_category=target_category,
            feedback_action=action,
            user_note="gui_drag_move",
            predicted_hierarchy={
                "large_category": result.large_category,
                "middle_category": result.middle_category or result.predicted_category,
                "small_category": result.small_category,
            },
            final_hierarchy={
                "large_category": taxonomy_entry.large_category,
                "middle_category": taxonomy_entry.middle_category,
                "small_category": taxonomy_entry.small_category,
            },
            evidence_text=str(payload.get("text", "")),
            metadata={"source": "gui_drag_drop", "file_name": payload.get("file_name", "")},
            source_scores={
                "rule": result.rule_score,
                "embedding": result.embedding_score,
                "feedback": result.feedback_score,
                "duplicate": result.duplicate_score,
                "llm": result.llm_score,
            },
            ocr_used=result.ocr_used,
            llm_used=result.llm_used,
        )
        embedding = result.query_embedding
        if not embedding and self.embedding_ready:
            embedding = self.resources.embedder.encode(str(payload["text"]))
        if embedding:
            self.resources.repository.save_confirmed_example(
                file_id=int(payload["file_id"]),
                category=target_category,
                source_text=str(payload["text"]),
                embedding=embedding,
                source_feedback_log_id=feedback_id,
            )

        updated_scores = dict(result.candidate_scores)
        updated_scores[target_category] = max(updated_scores.get(target_category, 0.0), result.confidence)
        updated_result = replace(
            result,
            predicted_category=taxonomy_entry.middle_category,
            middle_category=taxonomy_entry.middle_category,
            large_category=taxonomy_entry.large_category,
            small_category=taxonomy_entry.small_category,
            matched_rules=[*result.matched_rules],
            candidate_scores=updated_scores,
            reasoning=f"{result.reasoning} | gui_drag_move={result.predicted_category}->{taxonomy_entry.middle_category}",
        )
        payload["result"] = updated_result
        self.refresh_stats()
        self._refresh_category_options()
        self._refresh_result_tree()
        self.final_category.set(taxonomy_entry.middle_category)
        self.status_text.set(
            f"카테고리 이동 완료: {payload.get('file_name', '')} | {result.predicted_category} -> {taxonomy_entry.middle_category}"
        )

    def _restore_tree_selection(self, selected_meta: dict[str, object]) -> None:
        if self.tree is None:
            return
        target_kind = str(selected_meta.get("kind", ""))
        if target_kind == "category":
            target_category = str(selected_meta.get("category", ""))
            for item_id, meta in self.tree_meta.items():
                if meta.get("kind") == "category" and meta.get("category") == target_category:
                    self.tree.selection_set(item_id)
                    self.tree.focus(item_id)
                    return
        if target_kind == "file":
            target_payload = selected_meta.get("payload")
            if isinstance(target_payload, dict):
                target_file_name = str(target_payload.get("file_name", ""))
                for item_id, meta in self.tree_meta.items():
                    payload = meta.get("payload")
                    if meta.get("kind") == "file" and isinstance(payload, dict) and str(payload.get("file_name", "")) == target_file_name:
                        parent_id = self.tree.parent(item_id)
                        if parent_id:
                            self.tree.item(parent_id, open=True)
                        self.tree.selection_set(item_id)
                        self.tree.focus(item_id)
                        return

    def _reset_progress(self, total_files: int) -> None:
        self.total_files = total_files
        self.processed_files = 0
        self._set_progress(0, total_files)
        self._update_processing_summary()

    def _advance_progress(self) -> None:
        self.processed_files = min(self.processed_files + 1, self.total_files)
        self._set_progress(self.processed_files, self.total_files)

    def _set_progress(self, processed_files: int, total_files: int) -> None:
        percent = (processed_files / total_files * 100) if total_files else 0.0
        self.progress_value.set(percent)
        self.progress_text.set(f"진행률 {processed_files}/{total_files}")

    def _update_processing_summary(self) -> None:
        counts = summarize_processing_methods(self.all_payloads)
        self.processing_summary_text.set(
            f"룰 {counts['rule']} | 임베딩 {counts['embedding']} | LLM {counts['llm']}"
        )

    def _show_current_file(self, payload: dict[str, object]) -> None:
        index = int(payload["index"])
        file_name = str(payload["file_name"])
        suffix = " | 임베딩 모델 로드됨" if self.embedding_ready else f" | {embedding_state_status_text(self.embedding_state)}"
        self.status_text.set(f"처리 중 ({index}/{self.total_files}): {file_name}{suffix}")
        self.progress_text.set(f"진행률 {self.processed_files}/{self.total_files}")

    def _refresh_category_options(self) -> None:
        if self.category_combo is None:
            return
        categories = sorted(
            {
                payload["result"].predicted_category
                for payload in self.all_payloads
                if isinstance(payload.get("result"), ClassificationResult)
            }
        )
        values = ["전체", *categories]
        self.category_combo.configure(values=values)
        if self.category_filter.get() not in values:
            self.category_filter.set("전체")

    def on_select_result(self, _event: object | None = None) -> None:
        meta = self._selected_tree_meta()
        if meta is None:
            return
        if meta.get("kind") == "category":
            category = str(meta.get("category", ""))
            payloads = meta.get("payloads", [])
            if not isinstance(payloads, list):
                return
            file_names = [str(payload.get("file_name", "")) for payload in payloads[:10] if isinstance(payload, dict)]
            avg_confidence = 0.0
            if payloads:
                avg_confidence = sum(
                    payload["result"].confidence
                    for payload in payloads
                    if isinstance(payload.get("result"), ClassificationResult)
                ) / len(payloads)
            performance_summary = summarize_payload_profiles(payloads)
            slowest = performance_summary.get("slowest_files", [])
            stage_totals = performance_summary.get("stage_totals", {})
            top_stage = next(iter(stage_totals), "n/a")
            detail = (
                f"카테고리: {category}\n"
                f"파일 수: {len(payloads)}\n"
                f"평균 confidence: {avg_confidence:.3f}\n"
                f"샘플 파일:\n- " + "\n- ".join(file_names)
            )
            detail += (
                f"\n\navg_time: {performance_summary.get('average_file_time', 0.0):.3f}s\n"
                f"dominant_stage: {top_stage}\n"
                + (
                    f"slowest_in_category: {slowest[0]['file_name']} ({slowest[0]['total_time']:.2f}s)\n"
                    if slowest
                    else ""
                )
            )
            self.final_category.set(category)
            self._set_detail(detail)
            return

        payload = meta.get("payload")
        if not isinstance(payload, dict):
            return
        result = payload.get("result")
        if not isinstance(result, ClassificationResult):
            return

        self.final_category.set(result.predicted_category)
        performance = payload.get("performance", {})
        if not isinstance(performance, dict):
            performance = {}
        self.current_detail_summary = build_user_rationale_summary(result, payload)
        self.current_detail_debug = build_debug_detail(result, payload, performance)
        self.detail_more_expanded = False
        self._refresh_detail_text()
        return
        matched_rules = ", ".join(result.matched_rules) if result.matched_rules else "없음"
        similarity_text = f"{result.similarity_score:.3f}" if result.embedding_used else "skipped"
        performance = payload.get("performance", {})
        if not isinstance(performance, dict):
            performance = {}
        analysis = performance.get("analysis", {})
        if not isinstance(analysis, dict):
            analysis = {}
        stage_timings = performance.get("stage_timings", {})
        if not isinstance(stage_timings, dict):
            stage_timings = {}
        detail = (
            f"파일: {payload['file_name']}\n"
            f"경로: {payload.get('file_path', '')}\n"
            f"추천: {result.predicted_category}\n"
            f"계층: {result.large_category}/{result.middle_category}\n"
            f"confidence: {result.confidence:.3f}\n"
            f"review_required: {'yes' if result.review_required else 'no'}\n"
            f"processing: {get_processing_trace_text(result)}\n"
            f"similarity: {similarity_text}\n"
            f"점수: rule={result.rule_score:.3f}, embedding={result.embedding_score:.3f}, "
            f"feedback={result.feedback_score:.3f}, final={result.final_score:.3f}\n"
            f"매칭 규칙: {matched_rules}\n"
            f"후보 점수: {json.dumps(result.candidate_scores, ensure_ascii=False)}\n"
            f"근거: {result.reasoning}\n"
        )
        detail += (
            f"predicted_type: {result.predicted_type or result.predicted_category}\n"
            f"type_confidence: {result.type_confidence:.3f}\n"
            f"review_reasons: {', '.join(result.review_reasons) if result.review_reasons else 'none'}\n"
            f"suggested_tags: {json.dumps(result.suggested_tags, ensure_ascii=False)}\n"
            f"cluster_candidate_id: {result.cluster_candidate_id if result.cluster_candidate_id is not None else 'none'}\n"
            f"ml_evidence: {json.dumps(result.ml_evidence, ensure_ascii=False)}\n"
            f"rule_evidence: {json.dumps(result.rule_evidence, ensure_ascii=False)}\n"
        )
        stage_lines = "\n".join(
            f"  - {key}: {float(value):.3f}s"
            for key, value in sorted(stage_timings.items(), key=lambda item: float(item[1]), reverse=True)
            if key != "total"
        )
        reason_lines = "\n".join(f"  - {reason}" for reason in analysis.get("reasons", []))
        detail += (
            f"\nperformance_total: {float(analysis.get('total_time', stage_timings.get('total', 0.0))):.3f}s\n"
            f"dominant_stage: {analysis.get('dominant_stage', 'unknown')}\n"
            f"latency_summary: {analysis.get('summary', '')}\n"
            f"stage_breakdown:\n{stage_lines or '  - no stages recorded'}\n"
            f"latency_reasons:\n{reason_lines or '  - no latency reasons recorded'}\n"
        )
        self._set_detail(detail)

    def save_selected_feedback(self) -> None:
        meta = self._selected_tree_meta()
        if meta is None or meta.get("kind") != "file":
            messagebox.showinfo("안내", "파일 항목을 먼저 선택하세요.")
            return
        payload = meta.get("payload")
        if not isinstance(payload, dict):
            return
        result = payload.get("result")
        if not isinstance(result, ClassificationResult):
            return

        final_category = self.final_category.get().strip()
        if not final_category:
            messagebox.showinfo("안내", "최종 카테고리를 입력하세요.")
            return

        action = "confirmed" if final_category == result.predicted_category else "corrected"
        feedback_id = self.resources.repository.save_feedback(
            file_id=int(payload["file_id"]),
            classification_id=int(payload["classification_id"]),
            predicted_category=result.predicted_category,
            final_category=final_category,
            feedback_action=action,
            user_note="gui",
        )
        embedding = result.query_embedding
        if not embedding and self.embedding_ready:
            embedding = self.resources.embedder.encode(str(payload["text"]))
        if embedding:
            self.resources.repository.save_confirmed_example(
                file_id=int(payload["file_id"]),
                category=final_category,
                source_text=str(payload["text"]),
                embedding=embedding,
                source_feedback_log_id=feedback_id,
            )
        self.refresh_stats()
        self.status_text.set(f"저장 완료: {result.predicted_category} -> {final_category}")

    def preview_moves(self) -> None:
        plan = preview_move_plan(
            repository=self.resources.repository,
            config=self.resources.config,
        )
        self.last_preview_batch_id = int(plan["batch_id"])
        self.last_preview_manifest_path = str(plan["manifest_path"])
        self.last_preview_items = list(plan["items"])
        self.operation_status_text.set(
            f"미리보기 batch={self.last_preview_batch_id} | items={len(self.last_preview_items)}\n{self.last_preview_manifest_path}"
        )
        self._update_classification_controls()
        self._show_preview_window(plan)

    def commit_last_preview(self) -> None:
        if self.last_preview_batch_id is None:
            messagebox.showinfo("안내", "먼저 이동 미리보기를 생성하세요.")
            return
        create_safety_snapshot(
            repository=self.resources.repository,
            config=self.resources.config,
            reason=f"gui_pre_commit_move_batch_{self.last_preview_batch_id}",
        )
        result = commit_move_batch(self.resources.repository, batch_id=self.last_preview_batch_id)
        self.status_text.set(
            f"이동 커밋 완료: batch={result['batch_id']}, moved={result['moved']}, failed={result['failed']}"
        )
        messagebox.showinfo(
            "이동 커밋",
            f"batch={result['batch_id']}\n이동={result['moved']}\n실패={result['failed']}",
        )
        self.refresh_stats()

    def undo_last_move_ui(self) -> None:
        result = undo_last_move(self.resources.repository)
        self.status_text.set(
            f"마지막 이동 되돌리기: batch={result['batch_id']}, restored={result['restored']}, failed={result['failed']}"
        )
        messagebox.showinfo(
            "되돌리기",
            f"batch={result['batch_id']}\n복원={result['restored']}\n실패={result['failed']}",
        )
        self.refresh_stats()

    def show_move_history_window(self) -> None:
        rows = self.resources.repository.list_move_history()
        window = tk.Toplevel(self)
        window.title("이동 이력")
        window.geometry("980x460")
        tree = ttk.Treeview(
            window,
            columns=("batch", "category", "status", "source", "destination"),
            show="headings",
        )
        for key, title, width in (
            ("batch", "batch", 80),
            ("category", "category", 120),
            ("status", "status", 100),
            ("source", "source", 280),
            ("destination", "destination", 280),
        ):
            tree.heading(key, text=title)
            tree.column(key, width=width, anchor="w")
        tree.pack(fill="both", expand=True, padx=10, pady=10)

        row_map: dict[str, dict[str, Any]] = {}
        for row in rows:
            item_id = tree.insert(
                "",
                "end",
                values=(
                    row["batch_id"],
                    row["middle_category"],
                    row["status"],
                    row["source_path"],
                    row["actual_destination_path"] or row["destination_path"],
                ),
            )
            row_map[item_id] = dict(row)

        actions = ttk.Frame(window, padding=10)
        actions.pack(fill="x")

        def restore_selected_batch() -> None:
            selection = tree.selection()
            if not selection:
                messagebox.showinfo("안내", "이력 항목을 선택하세요.")
                return
            selected = row_map[selection[0]]
            result = restore_batch(self.resources.repository, batch_id=int(selected["batch_id"]))
            messagebox.showinfo(
                "배치 복원",
                f"batch={result['batch_id']}\n복원={result['restored']}\n실패={result['failed']}",
            )
            window.destroy()
            self.refresh_stats()
            detail.delete("1.0", "end")

        def restore_selected_file() -> None:
            selection = tree.selection()
            if not selection:
                messagebox.showinfo("안내", "이력 항목을 선택하세요.")
                return
            selected = row_map[selection[0]]
            result = restore_file(self.resources.repository, move_item_id=int(selected["id"]))
            messagebox.showinfo(
                "파일 복원",
                f"item={result['move_item_id']}\nbatch={result['batch_id']}\n복원={result['restored']}",
            )
            window.destroy()
            self.refresh_stats()

        ttk.Button(actions, text="선택 배치 복원", command=restore_selected_batch).pack(side="left")
        ttk.Button(actions, text="선택 파일 복원", command=restore_selected_file).pack(side="left", padx=(8, 0))

    def show_feedback_logs_window(self) -> None:
        window = tk.Toplevel(self)
        window.title("피드백 로그 관리")
        window.geometry("1040x520")

        tree = ttk.Treeview(
            window,
            columns=("id", "file", "from", "to", "confidence", "action"),
            show="headings",
        )
        for key, title, width in (
            ("id", "id", 60),
            ("file", "file", 220),
            ("from", "from", 120),
            ("to", "to", 120),
            ("confidence", "confidence", 90),
            ("action", "action", 90),
        ):
            tree.heading(key, text=title)
            tree.column(key, width=width, anchor="w")
        tree.pack(fill="both", expand=True, padx=10, pady=10)

        detail = tk.Text(window, height=8, wrap="word")
        detail.pack(fill="x", padx=10, pady=(0, 10))

        row_map: dict[str, dict[str, Any]] = {}

        def refresh() -> None:
            for item_id in tree.get_children():
                tree.delete(item_id)
            row_map.clear()
            for row in self.resources.repository.list_feedback_logs():
                item_id = tree.insert(
                    "",
                    "end",
                    values=(
                        row["id"],
                        row["file_name"],
                        row["predicted_middle_category"] or row["predicted_category"],
                        row["final_middle_category"] or row["final_category"],
                        f"{row['final_score']:.3f}",
                        row["feedback_action"],
                    ),
                )
                row_map[item_id] = dict(row)

        def on_select(_event: object | None = None) -> None:
            selection = tree.selection()
            if not selection:
                return
            selected = row_map[selection[0]]
            detail.delete("1.0", "end")
            detail.insert("1.0", json.dumps(selected, ensure_ascii=False, indent=2))

        def delete_selected() -> None:
            selection = tree.selection()
            if not selection:
                messagebox.showinfo("안내", "로그를 선택하세요.")
                return
            selected = row_map[selection[0]]
            deleted = 0
            try:
                deleted = self.resources.repository.delete_feedback_log(int(selected["id"]))
                rebuild_adaptive_learning(self.resources.repository, min_occurrences=2)
            except Exception as error:
                messagebox.showerror("피드백 로그", f"삭제 실패: {error}")
                return
            if deleted == 0:
                messagebox.showinfo("피드백 로그", "삭제할 로그를 찾지 못했습니다.")
                return
            refresh()
            self.refresh_stats()
            detail.delete("1.0", "end")
            messagebox.showinfo("피드백 로그", "선택한 로그를 삭제했습니다.")

        def clear_all() -> None:
            try:
                deleted = self.resources.repository.clear_feedback_logs()
            except Exception as error:
                messagebox.showerror("피드백 로그", f"전체 삭제 실패: {error}")
                return
            messagebox.showinfo("피드백 로그", f"삭제된 로그 수: {deleted}")
            refresh()
            self.refresh_stats()

            detail.delete("1.0", "end")

        def export_logs() -> None:
            path = filedialog.asksaveasfilename(
                title="피드백 로그 저장",
                defaultextension=".json",
                filetypes=[("JSON", "*.json")],
            )
            if not path:
                return
            rows = self.resources.repository.export_feedback_logs()
            Path(path).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            messagebox.showinfo("피드백 로그", f"저장 완료: {path}")

        def rebuild_learning() -> None:
            summary = rebuild_adaptive_learning(self.resources.repository, min_occurrences=2)
            messagebox.showinfo("학습 재구축", json.dumps(summary, ensure_ascii=False, indent=2))
            self.refresh_stats()

        tree.bind("<<TreeviewSelect>>", on_select)
        refresh()

        actions = ttk.Frame(window, padding=10)
        actions.pack(fill="x")
        ttk.Button(actions, text="새로고침", command=refresh).pack(side="left")
        ttk.Button(actions, text="선택 삭제", command=delete_selected).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="전체 삭제", command=clear_all).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="내보내기", command=export_logs).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="학습 재구축", command=rebuild_learning).pack(side="left", padx=(8, 0))

    def show_embedding_cache_window(self) -> None:
        window = tk.Toplevel(self)
        window.title("임베딩 캐시 관리")
        window.geometry("700x420")

        text = tk.Text(window, wrap="word")
        text.pack(fill="both", expand=True, padx=10, pady=10)

        def refresh() -> None:
            stats = self.resources.repository.get_embedding_cache_stats()
            text.delete("1.0", "end")
            text.insert("1.0", json.dumps(stats, ensure_ascii=False, indent=2))

        def clear_cache() -> None:
            deleted = self.resources.repository.clear_embedding_cache()
            messagebox.showinfo("임베딩 캐시", f"삭제된 엔트리 수: {deleted}")
            refresh()
            self.refresh_stats()

        def rebuild_cache() -> None:
            embedder = build_embedder(self.resources.config)
            built = 0
            for row in self.resources.repository.fetch_embedding_rebuild_sources():
                text_value = str(row["text_value"])
                if not text_value.strip():
                    continue
                embedding = embedder.encode(
                    text_value,
                    repository=self.resources.repository,
                    file_hash=str(row["file_hash"]),
                    text_kind=str(row["text_kind"]),
                )
                if embedding:
                    built += 1
            messagebox.showinfo("임베딩 캐시", f"재구축 완료: {built}개")
            refresh()
            self.refresh_stats()

        actions = ttk.Frame(window, padding=10)
        actions.pack(fill="x")
        ttk.Button(actions, text="새로고침", command=refresh).pack(side="left")
        ttk.Button(actions, text="비우기", command=clear_cache).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="재구축", command=rebuild_cache).pack(side="left", padx=(8, 0))
        refresh()

    def show_performance_window(self) -> None:
        window = tk.Toplevel(self)
        window.title("Performance Analysis")
        window.geometry("860x560")

        text = tk.Text(window, wrap="word")
        text.pack(fill="both", expand=True, padx=10, pady=10)

        summary = summarize_payload_profiles(
            self.all_payloads,
            startup_profile=self.startup_profile,
            run_profile=self.current_run_profile,
        )
        self.last_run_summary = summary
        startup_lines = "\n".join(
            f"- {key}: {float(value):.3f}s"
            for key, value in summary.get("startup_stages", {}).items()
        ) or "- no startup stages recorded"
        stage_total_lines = "\n".join(
            f"- {key}: {float(value):.3f}s"
            for key, value in summary.get("stage_totals", {}).items()
        ) or "- no stage totals recorded"
        slow_lines = "\n".join(
            f"- {row['file_name']} | {row['total_time']:.3f}s | {row['dominant_stage']} | {row['summary']}"
            for row in summary.get("slowest_files", [])
        ) or "- no classified files yet"
        text.insert(
            "1.0",
            (
                "Startup timings\n"
                f"- app_ready_total: {summary.get('startup_total', 0.0):.3f}s\n"
                f"{startup_lines}\n\n"
                "Run timings\n"
                f"- run_elapsed: {summary.get('run_elapsed', 0.0):.3f}s\n"
                f"- classified_files: {summary.get('classified_files', 0)}\n"
                f"- average_file_time: {summary.get('average_file_time', 0.0):.3f}s\n\n"
                "Stage totals\n"
                f"{stage_total_lines}\n\n"
                "Slowest files\n"
                f"{slow_lines}\n"
            ),
        )

    def _show_preview_window(self, plan: dict[str, Any]) -> None:
        window = tk.Toplevel(self)
        window.title(f"이동 미리보기 - batch {plan['batch_id']}")
        window.geometry("980x520")
        tree = ttk.Treeview(
            window,
            columns=("count", "destination", "confidence"),
            show="tree headings",
        )
        tree.heading("#0", text="카테고리 / 파일")
        tree.heading("count", text="개수/파일")
        tree.heading("destination", text="대상 경로")
        tree.heading("confidence", text="confidence")
        tree.column("#0", width=240, anchor="w")
        tree.column("count", width=120, anchor="center")
        tree.column("destination", width=420, anchor="w")
        tree.column("confidence", width=100, anchor="center")
        tree.pack(fill="both", expand=True, padx=10, pady=10)

        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in plan["items"]:
            grouped.setdefault(str(item["middle_category"]), []).append(item)
        for category, items in sorted(grouped.items(), key=lambda item: item[0]):
            parent_id = tree.insert("", "end", text=category, values=(f"{len(items)} files", "", ""))
            for item in items:
                tree.insert(
                    parent_id,
                    "end",
                    text=Path(str(item["source_path"])).name,
                    values=(
                        "file",
                        item["destination_path"],
                        f"{float(item['confidence']):.3f}",
                    ),
                )

        footer = ttk.Frame(window, padding=10)
        footer.pack(fill="x")
        ttk.Label(
            footer,
            text=f"batch={plan['batch_id']} | items={len(plan['items'])}\nmanifest={plan['manifest_path']}",
            justify="left",
        ).pack(side="left")
        ttk.Button(footer, text="이 미리보기 커밋", command=self.commit_last_preview).pack(side="right")

    def _clear_results(self) -> None:
        if self.tree is not None:
            for item_id in self.tree.get_children():
                self.tree.delete(item_id)
        self.all_payloads.clear()
        self.tree_meta.clear()
        self._set_progress(0, 0)
        self._update_processing_summary()
        self._refresh_category_options()
        self.current_detail_summary = ""
        self.current_detail_debug = ""
        self.detail_more_expanded = False
        self._set_detail("")
        if self.detail_more_button is not None:
            self.detail_more_button.configure(text="더보기")

    def _set_detail(self, text: str) -> None:
        if self.detail_text is None:
            return
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("1.0", text)
        self.detail_text.configure(state="disabled")

    def toggle_detail_more(self) -> None:
        self.detail_more_expanded = not self.detail_more_expanded
        self._refresh_detail_text()

    def _refresh_detail_text(self) -> None:
        if self.detail_more_expanded:
            self._set_detail(f"{self.current_detail_summary}{self.current_detail_debug}")
            if self.detail_more_button is not None:
                self.detail_more_button.configure(text="접기")
            return
        self._set_detail(self.current_detail_summary)
        if self.detail_more_button is not None:
            self.detail_more_button.configure(text="더보기")

    def _append_detail(self, text: str) -> None:
        if self.detail_text is None:
            return
        self.detail_text.configure(state="normal")
        self.detail_text.insert("end", text)
        self.detail_text.configure(state="disabled")

    def _handle_close(self) -> None:
        self.destroy()


def main() -> None:
    app = ClassifierGui()
    app.mainloop()
