"""Tkinter GUI for the classifier with embedding gating and operations UX."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import tkinter as tk
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any

from src.adaptive import rebuild_adaptive_learning
from src.classifier import (
    ClassificationResult,
    HybridClassifier,
    get_primary_processing_method,
    get_processing_method_label,
    get_processing_trace_text,
)
from src.cli import _build_cluster_evidence_worker, build_embedder, load_categories
from src.cluster_projection import build_cluster_projection, render_cluster_projection_html
from src.embedding_repository import create_embedding_repository
from src.evidence_pipeline import load_cached_document_evidence
from src.config import DEFAULT_CONFIG_PATH, AppConfig, load_app_config, save_app_config
from src.cluster_candidates import ClusterCandidateFinder
from src.document_features import DocumentFeatureExtractor
from src.file_reader import SUPPORTED_SUFFIXES, discover_supported_files, ensure_input_directory, extract_text_from_file
from src.hash_utils import compute_xxhash64
from src.operations import (
    commit_move_batch,
    preview_move_plan,
    preview_direct_folder_move_plan,
    preview_move_plan_for_classifications,
    restore_batch,
    restore_file,
    undo_last_move,
)
from src.ocr_support import DEFAULT_OCR_MIN_CHARS, explain_ocr_decision, ocr_pdf_file
from src.performance import build_file_latency_analysis, summarize_payload_profiles
from src.recovery import create_safety_snapshot
from src.rule_classifier import RuleBasedClassifier
from src.server_client import RemoteServerError, build_remote_client
from src.storage import ClassificationRepository
from src.taxonomy import Taxonomy, load_taxonomy
from src.text_cleaner import normalize_text
from src.vectorizer import SentenceTransformerEmbedder
from src.zip_categories_adapter import (
    ZIP_PIPELINE_VERSION,
    ZIP_READER_EXTENDED,
    classification_result_from_zip,
    run_zip_categories_pipeline,
)


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
        if payload.get("pipeline") == "cluster":
            cluster_id = int(payload.get("cluster_id", -1))
            category = str(payload.get("category") or ("Noise / API review" if cluster_id == -1 else f"Cluster {cluster_id}"))
            file_name = str(payload.get("file_name", ""))
            if not is_all_category_filter(normalized_filter) and category.lower() != normalized_filter:
                continue
            if lowered_query and lowered_query not in file_name.lower():
                continue
            grouped.setdefault(category, []).append(payload)
            continue
        result = payload.get("result")
        if not isinstance(result, ClassificationResult):
            continue
        category = result.predicted_category
        if not is_all_category_filter(normalized_filter) and category.lower() != normalized_filter:
            continue
        file_name = str(payload.get("file_name", ""))
        if lowered_query and lowered_query not in file_name.lower():
            continue
        grouped.setdefault(category, []).append(payload)
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def is_all_category_filter(value: str) -> bool:
    return value.strip().lower() in {"", "all", "전체", "?꾩껜"}


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
    final_category = result.predicted_category
    ml_type = result.predicted_type or ""
    confidence_label = _confidence_label(result.confidence)
    lines = [
        f"최종 분류는 '{final_category}'입니다.",
        f"최종 분류 신뢰도는 {result.confidence:.3f}로 {confidence_label} 수준입니다.",
    ]
    if ml_type:
        lines.append(f"ML 유형 후보는 '{ml_type}'이며, 유형 신뢰도는 {result.type_confidence:.3f}입니다.")
        if ml_type != final_category:
            lines.append("최종 분류와 ML 유형 후보가 달라서 검토가 필요할 수 있습니다.")

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
        evidence_lines.append(f"ML 유형 후보: '{ml_type or final_category}' 쪽 점수가 가장 높았습니다.")
    if result.semantic_evidence:
        evidence_lines.append(f"의미 근거: {len(result.semantic_evidence)}개의 문서 표현이 유형 신호와 맞습니다.")
    if result.layout_evidence:
        evidence_lines.append(f"레이아웃 근거: {len(result.layout_evidence)}개의 구조 특징이 감지됐습니다.")
    if result.structure_evidence:
        evidence_lines.append(f"구조 근거: {len(result.structure_evidence)}개의 문서 구조 특징이 감지됐습니다.")
    if result.ocr_evidence:
        evidence_lines.append(f"OCR 근거: {len(result.ocr_evidence)}개의 OCR 품질/문자 신호가 확인됐습니다.")
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
        f"최종 분류: {result.predicted_category}\n"
        f"계층: {result.large_category}/{result.middle_category}\n"
        f"ML 유형 후보: {result.predicted_type or 'none'}\n"
        f"ML 유형 신뢰도: {result.type_confidence:.3f}\n"
        f"최종 분류 신뢰도: {result.confidence:.3f}\n"
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
        f"semantic_evidence: {json.dumps(result.semantic_evidence, ensure_ascii=False)}\n"
        f"layout_evidence: {json.dumps(result.layout_evidence, ensure_ascii=False)}\n"
        f"structure_evidence: {json.dumps(result.structure_evidence, ensure_ascii=False)}\n"
        f"ocr_evidence: {json.dumps(result.ocr_evidence, ensure_ascii=False)}\n"
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


def collect_supported_drop_files(paths: list[Path]) -> list[Path]:
    """Expand dropped files/folders into supported document files."""
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(discover_supported_files(path))
        elif path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(path)
    return sorted({file_path.resolve() for file_path in files})


def build_content_only_confirmed_text(payload: dict[str, object]) -> str:
    """Return content evidence for learning, excluding filename-derived signals."""
    evidence = payload.get("evidence")
    text = ""
    if isinstance(evidence, dict):
        text = str(evidence.get("sampled_text") or "").strip()
    if not text:
        text = str(payload.get("text", "")).strip()
    return _strip_filename_signal_lines(text)


def _strip_filename_signal_lines(text: str) -> str:
    metadata_prefixes = (
        "filename:",
        "file_name:",
        "file path:",
        "file_path:",
        "path:",
        "filename_tokens:",
        "file_tokens:",
    )
    kept_lines = []
    for line in text.splitlines():
        normalized = line.strip().lower()
        if any(normalized.startswith(prefix) for prefix in metadata_prefixes):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines).strip()


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
        self.remote_job_id = ""
        self.remote_job_result: dict[str, Any] = {}
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
        self.reader_mode = tk.StringVar(value=ZIP_READER_EXTENDED)

        self.result_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.embedding_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.selected_classification_files: list[Path] = []
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

        ttk.Button(top_bar, text="분류 파일", command=self.show_classification_files_window).pack(side="left")
        ttk.Button(top_bar, text="분류 확정", command=self.show_classification_confirmation_window).pack(side="left", padx=(8, 0))
        ttk.Button(top_bar, text="설정", command=self.show_settings_window).pack(side="left", padx=(8, 0))

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
        self.tree = ttk.Treeview(table_frame, columns=columns, show="tree headings", height=24, selectmode="extended")
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

        operations_frame = ttk.LabelFrame(right_frame, text="작업 패널", padding=10)
        operations_frame.pack(fill="x", pady=(10, 0))
        ttk.Label(operations_frame, textvariable=self.operation_status_text, justify="left").pack(anchor="w", pady=(0, 8))

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

    def _set_selected_classification_files(self, files: list[Path]) -> None:
        self.selected_classification_files = sorted({file_path.resolve() for file_path in files})
        if self.selected_classification_files:
            common_parent = self.selected_classification_files[0].parent
            self.input_dir.set(str(common_parent))
        self.status_text.set(f"분류 파일 준비: {len(self.selected_classification_files)}개")

    def _add_selected_classification_paths(self, paths: list[Path]) -> list[Path]:
        files = collect_supported_drop_files(paths)
        self._set_selected_classification_files([*self.selected_classification_files, *files])
        return files

    def show_classification_files_window(self) -> None:
        window = tk.Toplevel(self)
        window.title("분류 파일")
        window.geometry("900x560")

        selected_files = list(self.selected_classification_files)
        path_value = tk.StringVar(value=self.input_dir.get())
        count_value = tk.StringVar(value=f"선택된 파일: {len(selected_files)}개")

        top = ttk.Frame(window, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="경로").pack(side="left")
        ttk.Entry(top, textvariable=path_value).pack(side="left", fill="x", expand=True, padx=(8, 0))
        ttk.Button(top, text="폴더 찾아보기", command=lambda: browse_folder()).pack(side="left", padx=(8, 0))

        list_frame = ttk.Frame(window, padding=(10, 0, 10, 0))
        list_frame.pack(fill="both", expand=True)
        tree = ttk.Treeview(list_frame, columns=("path",), show="tree headings", selectmode="extended")
        tree.heading("#0", text="파일명")
        tree.heading("path", text="경로")
        tree.column("#0", width=260, anchor="w")
        tree.column("path", width=560, anchor="w")
        tree.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

        drop_text = "여기에 파일 또는 폴더를 드래그하세요" if DRAG_AND_DROP_AVAILABLE else "드래그 앤 드롭은 tkinterdnd2 설치 후 사용할 수 있습니다"
        drop_label = ttk.Label(window, text=drop_text, anchor="center", padding=10, relief="ridge")
        drop_label.pack(fill="x", padx=10, pady=(8, 0))
        ttk.Label(window, textvariable=count_value, padding=(10, 6, 10, 0)).pack(fill="x")
        tree_item_paths: dict[str, set[Path]] = {}

        def refresh_list() -> None:
            for item_id in tree.get_children():
                tree.delete(item_id)
            tree_item_paths.clear()
            grouped: dict[Path, list[Path]] = {}
            for file_path in selected_files:
                grouped.setdefault(file_path.parent, []).append(file_path)
            for folder_path in sorted(grouped):
                files_in_folder = sorted(grouped[folder_path])
                folder_id = tree.insert(
                    "",
                    "end",
                    text=folder_path.name or str(folder_path),
                    values=(str(folder_path),),
                    open=True,
                )
                tree_item_paths[folder_id] = {file_path.resolve() for file_path in files_in_folder}
                for file_path in files_in_folder:
                    file_id = tree.insert(folder_id, "end", text=file_path.name, values=(str(file_path),))
                    tree_item_paths[file_id] = {file_path.resolve()}
            count_value.set(f"선택된 파일: {len(selected_files)}개")

        def add_paths(paths: list[Path]) -> None:
            nonlocal selected_files
            files = collect_supported_drop_files(paths)
            selected_files = sorted({*selected_files, *(file.resolve() for file in files)})
            if selected_files:
                path_value.set(str(selected_files[0].parent))
            refresh_list()

        def load_path() -> None:
            raw_path = path_value.get().strip().strip('"')
            if not raw_path:
                messagebox.showinfo("안내", "불러올 파일 또는 폴더 경로를 입력하세요.")
                return
            path = Path(raw_path)
            if not path.exists():
                messagebox.showerror("경로 오류", f"경로를 찾을 수 없습니다.\n{path}")
                return
            add_paths([path])

        def browse_files() -> None:
            paths = filedialog.askopenfilenames(
                title="분류할 파일 선택",
                filetypes=[("지원 문서", "*.txt *.pdf *.docx *.xlsx *.pptx *.hwp *.hwpx"), ("모든 파일", "*.*")],
            )
            if paths:
                add_paths([Path(path) for path in paths])

        def browse_folder() -> None:
            selected = filedialog.askdirectory(title="분류할 폴더 선택", initialdir=path_value.get() or ".")
            if selected:
                path_value.set(selected)
                add_paths([Path(selected)])

        def remove_selected() -> None:
            nonlocal selected_files
            selected_paths: set[Path] = set()
            for item_id in tree.selection():
                selected_paths.update(tree_item_paths.get(item_id, set()))
            selected_files = [file_path for file_path in selected_files if file_path.resolve() not in selected_paths]
            refresh_list()

        def clear_all() -> None:
            nonlocal selected_files
            selected_files = []
            refresh_list()

        def apply_selection() -> None:
            self._set_selected_classification_files(selected_files)
            window.destroy()

        def apply_and_classify() -> None:
            self._set_selected_classification_files(selected_files)
            window.destroy()
            self.start_classify()

        def apply_and_server_classify() -> None:
            self._set_selected_classification_files(selected_files)
            window.destroy()
            self.start_server_classify()

        def on_drop_to_window(event: object) -> None:
            raw_data = getattr(event, "data", "")
            dropped_paths = [Path(value) for value in self.tk.splitlist(raw_data)]
            add_paths(dropped_paths)

        if DRAG_AND_DROP_AVAILABLE:
            drop_label.drop_target_register(DND_FILES)
            drop_label.dnd_bind("<<Drop>>", on_drop_to_window)

        button_row = ttk.Frame(window, padding=10)
        button_row.pack(fill="x")
        ttk.Button(button_row, text="경로로 불러오기", command=load_path).pack(side="left")
        ttk.Button(button_row, text="파일 추가", command=browse_files).pack(side="left", padx=(8, 0))
        ttk.Button(button_row, text="폴더 추가", command=browse_folder).pack(side="left", padx=(8, 0))
        ttk.Button(button_row, text="선택 제거", command=remove_selected).pack(side="left", padx=(8, 0))
        ttk.Button(button_row, text="전체 비우기", command=clear_all).pack(side="left", padx=(8, 0))
        ttk.Button(button_row, text="확인", command=apply_selection).pack(side="right")
        ttk.Button(button_row, text="분류 실행", command=apply_and_classify).pack(side="right", padx=(0, 8))
        ttk.Button(button_row, text="서버로 분류", command=apply_and_server_classify).pack(side="right", padx=(0, 8))

        refresh_list()

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

    def show_stats_window(self) -> None:
        stats = self.resources.repository.get_stats()
        self.refresh_stats()
        window = tk.Toplevel(self)
        window.title("통계")
        window.geometry("520x460")
        text = tk.Text(window, wrap="word")
        text.pack(fill="both", expand=True, padx=10, pady=10)
        text.insert("1.0", json.dumps(stats, ensure_ascii=False, indent=2))
        text.configure(state="disabled")

    def start_classify(self) -> None:
        if not self._ensure_classification_available():
            return
        if not self.selected_classification_files:
            self.show_classification_files_window()
            self.status_text.set("분류 파일을 먼저 선택하세요.")
            return
        self._clear_results()
        self.status_text.set("파일 목록 준비 중")
        self.start_classify_files(list(self.selected_classification_files))

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

    def start_server_classify(self) -> None:
        if not self.selected_classification_files:
            self.show_classification_files_window()
            self.status_text.set("서버로 보낼 파일을 먼저 선택하세요.")
            return
        files = sorted({file_path.resolve() for file_path in self.selected_classification_files})
        if not files:
            messagebox.showinfo("안내", "서버로 보낼 파일이 없습니다.")
            return
        self._clear_results()
        self.current_run_profile = {"started_at": time.perf_counter(), "file_count": len(files), "pipeline": "remote"}
        self.last_run_summary = {}
        self.remote_job_id = ""
        self.remote_job_result = {}
        self._reset_progress(len(files))
        worker = threading.Thread(target=self._server_classify_worker, args=(files,), daemon=True)
        worker.start()
        self.after(100, self._drain_queue)

    def _server_classify_worker(self, files: list[Path]) -> None:
        client = build_remote_client(self.resources.config)
        local_by_name = {file_path.name: file_path for file_path in files}
        try:
            self.result_queue.put(("status", f"서버 업로드 시작: {len(files)}개 | {client.base_url.rstrip('/')}"))
            upload_result = client.upload_files(files)
            job_id = str(upload_result.get("job_id", ""))
            self.remote_job_id = job_id
            self.result_queue.put(("remote_job", upload_result))
            self.result_queue.put(("status", f"서버 job_id={job_id} | 결과 대기 중"))
            result = client.wait_for_result(
                job_id,
                timeout_seconds=None,
                on_poll=lambda data: self.result_queue.put(
                    (
                        "status",
                        f"서버 상태: {data.get('status', 'unknown')} | results={len(data.get('results', []))} | review={len(data.get('review_queue', []))}",
                    )
                ),
            )
            self.remote_job_result = result
            for item in result.get("results", []):
                if isinstance(item, dict):
                    self.result_queue.put(("result", self._remote_result_to_payload(item, local_by_name)))
            for item in result.get("review_queue", []):
                if isinstance(item, dict):
                    self.result_queue.put(("error", f"{item.get('filename', 'unknown')}: {item.get('reason', 'review_queue')}"))
            self.last_run_summary = {
                "pipeline": "remote_server",
                "job_id": job_id,
                "server_status": result.get("status", "unknown"),
                "file_count": len(files),
                "result_count": len(result.get("results", [])),
                "review_queue_count": len(result.get("review_queue", [])),
                "cluster_count": len(result.get("clusters", [])),
                "api_call_performed": True,
                "run_elapsed": round(time.perf_counter() - float(self.current_run_profile.get("started_at", time.perf_counter())), 3),
            }
            self.result_queue.put(("done", None))
        except RemoteServerError as error:
            self.result_queue.put(("error", f"서버 연결 실패: {error}"))
            self.result_queue.put(("done", None))

    def _remote_result_to_payload(self, item: dict[str, Any], local_by_name: dict[str, Path]) -> dict[str, object]:
        filename = str(item.get("filename") or "unknown")
        local_path = local_by_name.get(filename)
        category = str(item.get("category") or "서버 미분류")
        confidence = float(item.get("confidence") or 0.0)
        method = str(item.get("classify_method") or "remote")
        result = ClassificationResult(
            predicted_category=category,
            confidence=confidence,
            final_score=confidence,
            rule_score=confidence if method == "rule" else 0.0,
            embedding_score=confidence if method != "rule" else 0.0,
            llm_score=confidence if "llm" in method or "api" in method else 0.0,
            feedback_score=0.0,
            duplicate_score=0.0,
            similarity_score=confidence,
            embedding_used=method != "rule",
            review_required=confidence < 0.65,
            matched_rules=[str(value) for value in item.get("keywords", []) if value],
            candidate_scores={category: confidence},
            reasoning=str(item.get("reason") or "remote_server_result"),
            query_embedding=[],
            large_category=category,
            middle_category=category,
            middle_confidence=confidence,
            source_scores={"remote": {category: confidence}},
            explanation={"remote": item, "job_id": self.remote_job_id},
            llm_used="llm" in method or "api" in method,
            predicted_type=category,
            type_confidence=confidence,
            review_reasons=[] if confidence >= 0.65 else ["remote_low_confidence"],
        )
        return {
            "pipeline": "remote",
            "file_name": filename,
            "file_path": str(local_path or item.get("file_path") or filename),
            "file_hash": str(item.get("xxhash") or ""),
            "category": category,
            "confidence": confidence,
            "server_job_id": self.remote_job_id,
            "server_result": item,
            "text": str(item.get("reason") or ""),
            "ocr_used": False,
            "db_persisted": False,
            "confirmation_saved": False,
            "result": result,
        }

    def on_drop_files(self, event: object) -> None:
        if not self._ensure_classification_available():
            return
        raw_data = getattr(event, "data", "")
        dropped_paths = [Path(value) for value in self.tk.splitlist(raw_data)]
        files = collect_supported_drop_files(dropped_paths)

        self._clear_results()
        self.start_classify_files(files)

    def _cluster_pipeline_worker_v2(self, files: list[Path]) -> None:
        pipeline_start = time.perf_counter()
        self.result_queue.put(("status", f"ZIP 분류 파이프라인 준비 중 | reader={self.reader_mode.get()}"))
        evidence_start = time.perf_counter()
        evidence_documents = self._build_gui_evidence_documents(files)
        evidence_elapsed = time.perf_counter() - evidence_start
        self.result_queue.put(("status", f"evidence 완료: {len(evidence_documents)}개, {evidence_elapsed:.1f}s"))
        documents = [
            {
                "index": index,
                "file_path": evidence["file_path"],
                "file_name": evidence["filename"],
                "filename": evidence["filename"],
                "file_hash": evidence["file_hash"],
                "evidence": evidence,
            }
            for index, evidence in enumerate(evidence_documents, start=1)
        ]
        classify_start = time.perf_counter()
        self.result_queue.put(("status", f"ZIP 룰 + 3구간 임베딩 + 의미 분류 시작: {len(documents)}개"))
        zip_result = run_zip_categories_pipeline(
            documents,
            embedder=self.resources.embedder,
            repository=self.resources.repository,
            config=self.resources.config,
        )
        classify_elapsed = time.perf_counter() - classify_start
        classified_documents = list(zip_result["documents"])
        cluster_result = dict(zip_result["cluster_result"])
        cluster_ids = [int(cluster_id) for cluster_id in cluster_result.get("cluster_ids", [])]
        clustering_vectors = list(cluster_result.get("reduced_vectors", []))
        cluster_projection = build_cluster_projection(
            classified_documents,
            clustering_vectors,
            cluster_ids,
            probabilities=[float(value) for value in cluster_result.get("probabilities", [])],
        )
        for item, point in zip(classified_documents, cluster_projection.get("points", [])):
            item["projection"] = {"x": point.get("x"), "y": point.get("y")}
        for item in classified_documents:
            item["pipeline"] = "cluster"
            item["ocr_used"] = item["evidence"].get("extraction_status") == "ocr_used"
            item["result"] = classification_result_from_zip(item)
            item["text"] = str((item.get("evidence") or {}).get("sampled_text", ""))
            item["db_persisted"] = False
            item["confirmation_saved"] = False
            self.result_queue.put(("result", item))

        projection_dir = Path("outputs")
        projection_dir.mkdir(parents=True, exist_ok=True)
        projection_json_path = projection_dir / "cluster_projection.json"
        projection_html_path = projection_dir / "cluster_projection.html"
        projection_json_path.write_text(json.dumps(cluster_projection, ensure_ascii=False, indent=2), encoding="utf-8")
        projection_html_path.write_text(render_cluster_projection_html(cluster_projection), encoding="utf-8")
        active_clusters = {cluster_id for cluster_id in cluster_ids if cluster_id != -1}
        clustering_enabled = bool(cluster_result.get("enabled", True))
        noise_count = sum(1 for cluster_id in cluster_ids if cluster_id == -1) if clustering_enabled else 0
        self.last_run_summary = {
            "pipeline": ZIP_PIPELINE_VERSION,
            "reader_mode": self.reader_mode.get(),
            "file_count": len(classified_documents),
            "cluster_count": len(active_clusters),
            "noise_count": noise_count,
            "cluster_result": cluster_result,
            "projection_file": str(projection_html_path),
            "api_call_performed": False,
            "feedback_examples_used": int(zip_result.get("feedback_examples_used", 0)),
            "evidence_elapsed": round(evidence_elapsed, 3),
            "classification_elapsed": round(classify_elapsed, 3),
            "run_elapsed": round(time.perf_counter() - pipeline_start, 3),
        }
        cluster_status = str(cluster_result.get("status", "unknown"))
        self.result_queue.put(("status", f"ZIP 분류 완료: categories={len(zip_result['categories'])}, supplemental_clustering={cluster_status}, clusters={len(active_clusters)}, noise={noise_count}, projection={projection_html_path}"))
        self.result_queue.put(("done", None))

    def _build_gui_evidence_documents(self, files: list[Path]) -> list[dict[str, Any]]:
        if not files:
            return []
        max_workers = 1 if len(files) <= 8 else min(4, len(files))
        results: list[dict[str, Any] | None] = [None] * len(files)
        worker_args = (
            [],
            self.resources.config.ocr.min_text_chars,
            self.resources.config.ocr.enabled,
            self.resources.config.ocr.max_pages,
            DB_PATH,
            self.resources.config.ocr.cache_enabled,
            self.resources.config.features.evidence_cache_dir,
            self.resources.config.features.evidence_cache_enabled,
        )
        pending_indices = []
        for index, file_path in enumerate(files):
            cached = load_cached_document_evidence(
                file_path,
                rules=[],
                min_text_chars=self.resources.config.ocr.min_text_chars,
                ocr_enabled=self.resources.config.ocr.enabled,
                ocr_max_pages=self.resources.config.ocr.max_pages,
                evidence_cache_dir=self.resources.config.features.evidence_cache_dir,
            )
            if cached is None:
                pending_indices.append(index)
            else:
                results[index] = cached
        if not pending_indices:
            self.result_queue.put(("status", f"evidence cache 사용: {len(files)}개"))
            return [result for result in results if result is not None]
        pending_files = [files[index] for index in pending_indices]
        self.result_queue.put(("status", f"evidence cache: hit={len(files) - len(pending_files)}, miss={len(pending_files)}"))
        max_workers = 1 if len(pending_files) <= 8 else min(4, len(pending_files))
        if max_workers == 1:
            for completed, index in enumerate(pending_indices, start=1):
                file_path = files[index]
                results[index] = _build_cluster_evidence_worker(str(file_path), *worker_args)
                status = str(results[index].get("extraction_status", "unknown")) if results[index] else "failed"
                self.result_queue.put(("status", f"evidence {completed}/{len(pending_files)}: {file_path.name} ({status})"))
            return [result for result in results if result is not None]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_build_cluster_evidence_worker, str(files[index]), *worker_args): index
                for index in pending_indices
            }
            completed = 0
            for future in as_completed(futures):
                index = futures[future]
                file_path = files[index]
                completed += 1
                try:
                    results[index] = future.result()
                    status = str(results[index].get("extraction_status", "unknown")) if results[index] else "failed"
                    self.result_queue.put(("status", f"evidence {completed}/{len(pending_files)}: {file_path.name} ({status})"))
                except Exception as error:
                    self.result_queue.put(("error", f"{file_path.name}: evidence failed - {error}"))
                    results[index] = None
        return [result for result in results if result is not None]

    def _cluster_pipeline_worker(self, files: list[Path]) -> None:
        self._cluster_pipeline_worker_v2(files)

    def _persist_zip_payload(self, payload: dict[str, Any]) -> bool:
        """Store ZIP results through the existing DB contract so feedback keeps working."""
        if payload.get("file_id") is not None and payload.get("classification_id") is not None:
            payload["db_persisted"] = True
            return True
        result = payload.get("result")
        if not isinstance(result, ClassificationResult):
            return False
        file_path = Path(str(payload["file_path"]))
        xxhash64 = str(payload.get("file_hash", "")).strip()
        if not xxhash64:
            xxhash64 = compute_xxhash64(file_path)
        duplicate_file_id = self.resources.repository.find_duplicate_file_id(xxhash64, str(file_path))
        text = str((payload.get("evidence") or {}).get("sampled_text", ""))
        file_id = self.resources.repository.upsert_file(
            file_path=str(file_path),
            file_name=file_path.name,
            file_ext=file_path.suffix.lower(),
            file_size=file_path.stat().st_size if file_path.exists() else 0,
            xxhash64=xxhash64,
            duplicate_of_file_id=duplicate_file_id,
            extracted_text=text,
        )
        classification_id = self.resources.repository.insert_classification(
            file_id=file_id,
            predicted_category=result.predicted_category,
            rule_score=result.rule_score,
            embedding_score=result.embedding_score,
            llm_score=0.0,
            final_score=result.final_score,
            candidate_scores_json=json.dumps(result.candidate_scores, ensure_ascii=False),
            reasoning=result.reasoning,
            status="suggested",
            large_category=result.large_category,
            middle_category=result.middle_category,
            middle_confidence=result.middle_confidence,
            source_scores_json=json.dumps({"zip_semantic": result.candidate_scores}, ensure_ascii=False),
            evidence_json=json.dumps(
                {
                    "zip_pipeline": ZIP_PIPELINE_VERSION,
                    "reader_mode": self.reader_mode.get(),
                    "clustering_status": payload.get("clustering_status", "unknown"),
                    "cluster_id": payload.get("cluster_id", -1),
                    "cluster_probability": payload.get("cluster_probability", 0.0),
                    "projection": payload.get("projection", {}),
                    "ocr_used": payload.get("ocr_used", False),
                },
                ensure_ascii=False,
            ),
            classifier_version=ZIP_PIPELINE_VERSION,
            config_version=ZIP_PIPELINE_VERSION,
            review_reasons_json=json.dumps(result.review_reasons, ensure_ascii=False),
            rule_evidence_json=json.dumps(result.rule_evidence, ensure_ascii=False),
        )
        payload["file_id"] = file_id
        payload["classification_id"] = classification_id
        payload["text"] = text
        payload["db_persisted"] = True
        return True

    def _classify_worker(self, files: list[Path]) -> None:
        self._cluster_pipeline_worker(files)
        return
        self.result_queue.put(("status", "분류 엔진 준비 중"))
        classifier = HybridClassifier(
            repository=self.resources.repository,
            embedder=self.resources.embedder,
            rule_classifier=self.resources.rule_classifier,
            taxonomy=self.resources.taxonomy,
            feature_extractor=DocumentFeatureExtractor(version=self.resources.config.features.extractor_version),
            ml_enabled=False,
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
            embedder=self.resources.embedder,
            repository=self.resources.repository,
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
            elif event == "remote_job":
                if isinstance(payload, dict):
                    self.remote_job_id = str(payload.get("job_id", ""))
                    self.operation_status_text.set(
                        f"서버 job_id: {self.remote_job_id}\n업로드 파일: {payload.get('file_count', 0)}"
                    )
            elif event == "start_file":
                self._show_current_file(payload)  # type: ignore[arg-type]
            elif event == "result":
                self._insert_result(payload)  # type: ignore[arg-type]
                self._advance_progress()
            elif event == "error":
                self._append_detail(f"[처리 실패] {payload}\n")
                self._advance_progress()
            elif event == "done":
                if self.remote_job_id:
                    self.status_text.set(
                        f"서버 분류 완료 | job_id={self.remote_job_id} | files={len(self.all_payloads)}"
                    )
                    self._set_progress(self.total_files, self.total_files)
                    return
                if any(item.get("pipeline") == "cluster" for item in self.all_payloads):
                    clusters = {
                        int(item.get("cluster_id", -1))
                        for item in self.all_payloads
                        if int(item.get("cluster_id", -1)) != -1
                    }
                    noise = sum(1 for item in self.all_payloads if int(item.get("cluster_id", -1)) == -1)
                    categories = {str(item.get("category", "")) for item in self.all_payloads}
                    self.status_text.set(f"ZIP 분류 완료 | categories={len(categories)} | clusters={len(clusters)} | noise={noise}")
                    self._set_progress(self.total_files, self.total_files)
                    self.refresh_stats()
                    return
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
            is_cluster_group = any(payload.get("pipeline") == "cluster" for payload in payloads if isinstance(payload, dict))
            avg_confidence = 0.0 if is_cluster_group else sum(
                payload["result"].confidence
                for payload in payloads
                if isinstance(payload.get("result"), ClassificationResult)
            ) / max(len(payloads), 1)
            parent_id = self.tree.insert(
                "",
                "end",
                text=category,
                values=("category", f"{len(payloads)} files", f"avg {avg_confidence:.3f}" if not is_cluster_group else "ZIP fixed category"),
                open=category in open_categories,
            )
            self.tree_meta[parent_id] = {"kind": "category", "category": category, "payloads": payloads}
            for payload in sorted(payloads, key=lambda item: str(item.get("file_name", "")).lower()):
                if payload.get("pipeline") == "cluster":
                    evidence = payload.get("evidence", {})
                    status = str(evidence.get("extraction_status", "")) if isinstance(evidence, dict) else ""
                    cluster_meta = (
                        "cluster=off"
                        if payload.get("clustering_status") == "disabled"
                        else f"cluster={payload.get('cluster_id', -1)} p={float(payload.get('cluster_probability', 0.0)):.2f}"
                    )
                    child_id = self.tree.insert(
                        parent_id,
                        "end",
                        text=str(payload["file_name"]),
                        values=(
                            "file",
                            f"{float(payload.get('confidence', 0.0)):.3f}",
                            f"{cluster_meta} | {status}",
                        ),
                        tags=("ocr_used",) if bool(payload.get("ocr_used")) else (),
                    )
                    self.tree_meta[child_id] = {"kind": "file", "payload": payload, "ocr_used": bool(payload.get("ocr_used"))}
                    continue
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
        if not self._persist_zip_payload(payload):
            messagebox.showerror("저장 실패", "분류 결과를 DB에 저장하지 못했습니다.")
            return
        confirmed_content_text = build_content_only_confirmed_text(payload)
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
            evidence_text=confirmed_content_text,
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
            confirmation_batch_id=self._new_confirmation_batch_id("drag"),
            confirmation_batch_name=self._new_confirmation_batch_name("드래그 수정"),
        )
        embedding = result.query_embedding
        if not embedding and self.embedding_ready:
            embedding = self.resources.embedder.encode(confirmed_content_text)
        if embedding:
            self.resources.repository.save_confirmed_example(
                file_id=int(payload["file_id"]),
                category=target_category,
                source_text=confirmed_content_text,
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
        if any(payload.get("pipeline") == "cluster" for payload in self.all_payloads):
            if any(payload.get("clustering_status") == "disabled" for payload in self.all_payloads):
                categories = {str(payload.get("category", "")) for payload in self.all_payloads}
                self.processing_summary_text.set(f"categories {len(categories)} | supplemental clustering off")
                return
            clusters = {
                int(payload.get("cluster_id", -1))
                for payload in self.all_payloads
                if int(payload.get("cluster_id", -1)) != -1
            }
            categories = {str(payload.get("category", "")) for payload in self.all_payloads}
            noise = sum(1 for payload in self.all_payloads if int(payload.get("cluster_id", -1)) == -1)
            self.processing_summary_text.set(f"categories {len(categories)} | clusters {len(clusters)} | noise {noise}")
            return
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
                (
                    str(payload.get("category") or "Noise / API review")
                )
                if payload.get("pipeline") == "cluster"
                else payload["result"].predicted_category
                for payload in self.all_payloads
                if payload.get("pipeline") == "cluster" or isinstance(payload.get("result"), ClassificationResult)
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
            if any(isinstance(payload, dict) and payload.get("pipeline") == "cluster" for payload in payloads):
                file_names = [str(payload.get("file_name", "")) for payload in payloads[:12] if isinstance(payload, dict)]
                statuses: dict[str, int] = {}
                top_tokens: dict[str, int] = {}
                for payload in payloads:
                    if not isinstance(payload, dict):
                        continue
                    evidence = payload.get("evidence", {})
                    if not isinstance(evidence, dict):
                        continue
                    status = str(evidence.get("extraction_status", "unknown"))
                    statuses[status] = statuses.get(status, 0) + 1
                    for item in evidence.get("top_tokens", [])[:10]:
                        if isinstance(item, dict):
                            token = str(item.get("token", "")).strip()
                            if token:
                                top_tokens[token] = top_tokens.get(token, 0) + 1
                token_line = ", ".join(
                    token for token, _count in sorted(top_tokens.items(), key=lambda item: item[1], reverse=True)[:15]
                )
                status_line = ", ".join(f"{name}={count}" for name, count in sorted(statuses.items()))
                detail = (
                    f"ZIP 고정 카테고리: {category}\n"
                    f"문서 수: {len(payloads)}\n"
                    f"외부 API 상태: not called\n"
                    f"추출 상태: {status_line or 'none'}\n"
                    f"공통 토큰 후보: {token_line or 'none'}\n\n"
                    f"샘플 파일:\n- " + "\n- ".join(file_names)
                )
                self.final_category.set("")
                self._set_detail(detail)
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
        if payload.get("pipeline") == "cluster":
            evidence = payload.get("evidence", {})
            if not isinstance(evidence, dict):
                evidence = {}
            self.final_category.set(str(payload.get("category", "")))
            top_tokens = evidence.get("top_tokens", [])
            token_text = ", ".join(
                str(item.get("token", ""))
                for item in top_tokens[:15]
                if isinstance(item, dict)
            )
            detail = (
                f"파일: {payload.get('file_name', '')}\n"
                f"경로: {payload.get('file_path', '')}\n"
                f"ZIP 고정 카테고리: {payload.get('category', '')}\n"
                f"confidence: {float(payload.get('confidence', 0.0)):.3f}\n"
                f"rule_confirmed: {bool(payload.get('rule_confirmed'))}\n"
                f"matched_keywords: {', '.join(str(value) for value in payload.get('matched_keywords', [])) or 'none'}\n"
                f"semantic_signals: {json.dumps(payload.get('semantic_signals', {}), ensure_ascii=False)}\n"
                f"clustering_status: {payload.get('clustering_status', 'unknown')}\n"
                f"cluster_id: {payload.get('cluster_id', -1)}\n"
                f"cluster_probability: {float(payload.get('cluster_probability', 0.0)):.3f}\n"
                f"UMAP 좌표: {json.dumps(payload.get('projection', {}), ensure_ascii=False)}\n"
                f"reader_mode: {self.reader_mode.get()}\n"
                f"extraction_status: {evidence.get('extraction_status', '')}\n"
                f"ocr_used: {bool(payload.get('ocr_used'))}\n"
                f"api_reader_required: {bool((evidence.get('api_reader') or {}).get('required')) if isinstance(evidence.get('api_reader'), dict) else False}\n"
                f"top_tokens: {token_text}\n\n"
                f"sampled_text:\n{evidence.get('sampled_text', '')}\n\n"
                f"structural_features:\n{json.dumps(evidence.get('structural_features', {}), ensure_ascii=False, indent=2)}\n\n"
                f"layout_features:\n{json.dumps(evidence.get('layout_features', {}), ensure_ascii=False, indent=2)}\n\n"
                f"timings:\n{json.dumps(evidence.get('timings', {}), ensure_ascii=False, indent=2)}"
            )
            self._set_detail(detail)
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
            f"최종 분류: {result.predicted_category}\n"
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
            f"ML 유형 후보: {result.predicted_type or 'none'}\n"
            f"ML 유형 신뢰도: {result.type_confidence:.3f}\n"
            f"review_reasons: {', '.join(result.review_reasons) if result.review_reasons else 'none'}\n"
            f"suggested_tags: {json.dumps(result.suggested_tags, ensure_ascii=False)}\n"
            f"cluster_candidate_id: {result.cluster_candidate_id if result.cluster_candidate_id is not None else 'none'}\n"
            f"ml_evidence: {json.dumps(result.ml_evidence, ensure_ascii=False)}\n"
            f"rule_evidence: {json.dumps(result.rule_evidence, ensure_ascii=False)}\n"
            f"semantic_evidence: {json.dumps(result.semantic_evidence, ensure_ascii=False)}\n"
            f"layout_evidence: {json.dumps(result.layout_evidence, ensure_ascii=False)}\n"
            f"structure_evidence: {json.dumps(result.structure_evidence, ensure_ascii=False)}\n"
            f"ocr_evidence: {json.dumps(result.ocr_evidence, ensure_ascii=False)}\n"
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

    def _selected_file_payloads(self) -> list[dict[str, object]]:
        if self.tree is None:
            return []
        payloads: list[dict[str, object]] = []
        for item_id in self.tree.selection():
            meta = self.tree_meta.get(item_id)
            if not isinstance(meta, dict):
                continue
            if meta.get("kind") == "category":
                for payload in meta.get("payloads", []):
                    if isinstance(payload, dict):
                        payloads.append(payload)
                continue
            if meta.get("kind") != "file":
                continue
            payload = meta.get("payload")
            if isinstance(payload, dict):
                payloads.append(payload)
        unique: dict[str, dict[str, object]] = {}
        for payload in payloads:
            unique[str(payload.get("file_path", id(payload)))] = payload
        return list(unique.values())

    def _save_confirmed_payload(
        self,
        payload: dict[str, object],
        *,
        final_category: str,
        user_note: str,
        confirmation_batch_id: str = "",
        confirmation_batch_name: str = "",
    ) -> bool:
        result = payload.get("result")
        if not isinstance(result, ClassificationResult) or not final_category:
            return False
        if bool(payload.get("confirmation_saved")) and payload.get("confirmed_category") == final_category:
            return True
        if not self._persist_zip_payload(payload):
            return False

        action = "confirmed" if final_category == result.predicted_category else "corrected"
        confirmed_content_text = build_content_only_confirmed_text(payload)
        feedback_id = self.resources.repository.save_feedback(
            file_id=int(payload["file_id"]),
            classification_id=int(payload["classification_id"]),
            predicted_category=result.predicted_category,
            final_category=final_category,
            feedback_action=action,
            user_note=user_note,
            evidence_text=confirmed_content_text,
            source_scores={
                "rule": result.rule_score,
                "embedding": result.embedding_score,
                "feedback": result.feedback_score,
                "duplicate": result.duplicate_score,
                "llm": result.llm_score,
            },
            ocr_used=result.ocr_used,
            llm_used=result.llm_used,
            confirmation_batch_id=confirmation_batch_id,
            confirmation_batch_name=confirmation_batch_name,
        )
        embedding = result.query_embedding
        if not embedding and self.embedding_ready:
            embedding = self.resources.embedder.encode(confirmed_content_text)
        if embedding:
            self.resources.repository.save_confirmed_example(
                file_id=int(payload["file_id"]),
                category=final_category,
                source_text=confirmed_content_text,
                embedding=embedding,
                source_feedback_log_id=feedback_id,
            )
        payload["confirmation_saved"] = True
        payload["confirmed_category"] = final_category
        return True

    def _new_confirmation_batch_id(self, prefix: str) -> str:
        return f"{prefix}-{time.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"

    def _new_confirmation_batch_name(self, label: str) -> str:
        return f"{label} {time.strftime('%Y-%m-%d %H:%M:%S')}"

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

        batch_id = self._new_confirmation_batch_id("single")
        batch_name = self._new_confirmation_batch_name("선택 확정")
        if not self._save_confirmed_payload(
            payload,
            final_category=final_category,
            user_note="gui_single_confirm",
            confirmation_batch_id=batch_id,
            confirmation_batch_name=batch_name,
        ):
            messagebox.showerror("저장 실패", "선택한 파일을 확정 저장하지 못했습니다.")
            return
        self.refresh_stats()
        self.status_text.set(f"저장 완료: {result.predicted_category} -> {final_category}")

    def confirm_selected_results(self) -> None:
        payloads = self._selected_file_payloads()
        if not payloads:
            messagebox.showinfo("안내", "확정할 파일 항목을 하나 이상 선택하세요.")
            return
        saved = 0
        failed = 0
        batch_id = self._new_confirmation_batch_id("selected")
        batch_name = self._new_confirmation_batch_name("선택 확정")
        for payload in payloads:
            result = payload.get("result")
            if not isinstance(result, ClassificationResult):
                failed += 1
                continue
            if self._save_confirmed_payload(
                payload,
                final_category=result.predicted_category,
                user_note="gui_selected_bulk_confirm",
                confirmation_batch_id=batch_id,
                confirmation_batch_name=batch_name,
            ):
                saved += 1
            else:
                failed += 1
        self.refresh_stats()
        self.status_text.set(f"선택 확정 저장 완료: saved={saved}, failed={failed}")
        messagebox.showinfo("분류 확정", f"선택 확정 저장 완료\n저장: {saved}\n실패: {failed}")

    def confirm_selected_remote_result(self) -> None:
        if not self.remote_job_id:
            messagebox.showinfo("안내", "서버 job_id가 없습니다. 먼저 서버로 분류를 실행하세요.")
            return
        payloads = [
            payload
            for payload in self._selected_file_payloads()
            if isinstance(payload, dict) and payload.get("pipeline") == "remote"
        ]
        if not payloads:
            messagebox.showinfo("안내", "서버에 확정 전송할 서버 분류 결과를 선택하세요.")
            return
        corrections = []
        for payload in payloads:
            result = payload.get("result")
            category = self.final_category.get().strip()
            if not category and isinstance(result, ClassificationResult):
                category = result.predicted_category
            corrections.append(
                {
                    "filename": str(payload.get("file_name") or ""),
                    "user_category": category,
                    "folder_description": "",
                }
            )
        try:
            response = build_remote_client(self.resources.config).confirm_job(self.remote_job_id, corrections)
        except RemoteServerError as error:
            messagebox.showerror("서버 확정 실패", str(error))
            return
        saved = int(response.get("saved", 0))
        self.operation_status_text.set(f"서버 확정 전송 완료: saved={saved}\njob_id={self.remote_job_id}")
        messagebox.showinfo("서버 확정 전송", json.dumps(response, ensure_ascii=False, indent=2))

    def confirm_all_results(self) -> None:
        payloads = [
            payload
            for payload in self.all_payloads
            if isinstance(payload, dict) and isinstance(payload.get("result"), ClassificationResult)
        ]
        if not payloads:
            messagebox.showinfo("안내", "확정할 분류 결과가 없습니다.")
            return
        if not messagebox.askyesno("전체 분류 확정", f"현재 결과 {len(payloads)}개를 모두 확정 저장할까요?"):
            return
        saved = 0
        failed = 0
        batch_id = self._new_confirmation_batch_id("all")
        batch_name = self._new_confirmation_batch_name("전체 확정")
        for payload in payloads:
            result = payload.get("result")
            if not isinstance(result, ClassificationResult):
                failed += 1
                continue
            if self._save_confirmed_payload(
                payload,
                final_category=result.predicted_category,
                user_note="gui_all_confirm",
                confirmation_batch_id=batch_id,
                confirmation_batch_name=batch_name,
            ):
                saved += 1
            else:
                failed += 1
        self.refresh_stats()
        self.status_text.set(f"전체 확정 저장 완료: saved={saved}, failed={failed}")
        messagebox.showinfo("분류 확정", f"전체 확정 저장 완료\n저장: {saved}\n실패: {failed}")

    def _confirm_and_move_payloads(
        self,
        payloads: list[dict[str, object]],
        *,
        user_note: str,
        title: str,
    ) -> None:
        valid_payloads = [
            payload
            for payload in payloads
            if isinstance(payload, dict) and isinstance(payload.get("result"), ClassificationResult)
        ]
        if not valid_payloads:
            messagebox.showinfo("안내", "확정 후 이동할 분류 결과가 없습니다.")
            return
        if not messagebox.askyesno(
            title,
            f"{len(valid_payloads)}개 파일을 현재 분류 결과로 확정 저장한 뒤 실제 파일 위치를 이동할까요?\n"
            "이동 후에는 '마지막 이동 되돌리기'로 복구할 수 있습니다.",
        ):
            return

        saved = 0
        failed = 0
        classification_ids: list[int] = []
        confirmation_batch_id = self._new_confirmation_batch_id("move")
        confirmation_batch_name = self._new_confirmation_batch_name("확정 후 이동")
        for payload in valid_payloads:
            result = payload.get("result")
            if not isinstance(result, ClassificationResult):
                failed += 1
                continue
            if self._save_confirmed_payload(
                payload,
                final_category=result.predicted_category,
                user_note=user_note,
                confirmation_batch_id=confirmation_batch_id,
                confirmation_batch_name=confirmation_batch_name,
            ):
                saved += 1
                classification_id = payload.get("classification_id")
                if classification_id is not None:
                    classification_ids.append(int(classification_id))
            else:
                failed += 1

        if not classification_ids:
            self.refresh_stats()
            messagebox.showerror("이동 실패", f"확정 저장된 분류 ID가 없습니다.\n저장: {saved}\n실패: {failed}")
            return

        plan = preview_move_plan_for_classifications(
            repository=self.resources.repository,
            config=self.resources.config,
            classification_ids=classification_ids,
        )
        self.last_preview_batch_id = int(plan["batch_id"])
        self.last_preview_manifest_path = str(plan["manifest_path"])
        self.last_preview_items = list(plan["items"])
        if not self.last_preview_items:
            self._update_classification_controls()
            self.refresh_stats()
            messagebox.showinfo(
                "이동 대상 없음",
                f"확정은 저장됐지만 이동 가능한 원본 파일이 없습니다.\n저장: {saved}\n실패: {failed}",
            )
            return

        create_safety_snapshot(
            repository=self.resources.repository,
            config=self.resources.config,
            reason=f"gui_confirm_and_move_batch_{self.last_preview_batch_id}",
        )
        result = commit_move_batch(self.resources.repository, batch_id=self.last_preview_batch_id)
        self.operation_status_text.set(
            f"확정 후 이동 batch={result['batch_id']} | moved={result['moved']} | failed={result['failed']}\n"
            f"{self.last_preview_manifest_path}"
        )
        self._update_classification_controls()
        self.refresh_stats()
        self.status_text.set(
            f"확정 후 이동 완료: saved={saved}, confirm_failed={failed}, "
            f"moved={result['moved']}, move_failed={result['failed']}"
        )
        messagebox.showinfo(
            "확정 후 이동 완료",
            f"확정 저장: {saved}\n확정 실패: {failed}\n이동: {result['moved']}\n이동 실패: {result['failed']}",
        )

    def confirm_selected_and_move(self) -> None:
        payloads = self._selected_file_payloads()
        if not payloads:
            messagebox.showinfo("안내", "확정 후 이동할 파일 항목을 하나 이상 선택하세요.")
            return
        self._confirm_and_move_payloads(
            payloads,
            user_note="gui_selected_confirm_and_move",
            title="선택 확정 후 바로 이동",
        )

    def confirm_all_and_move(self) -> None:
        payloads = [
            payload
            for payload in self.all_payloads
            if isinstance(payload, dict) and isinstance(payload.get("result"), ClassificationResult)
        ]
        self._confirm_and_move_payloads(
            payloads,
            user_note="gui_all_confirm_and_move",
            title="전체 확정 후 바로 이동",
        )

    def _confirmed_current_payloads(self) -> list[dict[str, object]]:
        payloads: list[dict[str, object]] = []
        for payload in self.all_payloads:
            if not isinstance(payload, dict):
                continue
            if not bool(payload.get("confirmation_saved")):
                continue
            if payload.get("file_id") is None or payload.get("classification_id") is None:
                continue
            payloads.append(payload)
        return payloads

    def _ask_transfer_mode(self) -> str | None:
        dialog = tk.Toplevel(self)
        dialog.title("파일 이동 방식")
        dialog.geometry("360x170")
        dialog.transient(self)
        dialog.grab_set()
        result = {"mode": None}
        frame = ttk.Frame(dialog, padding=16)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="확정 파일을 어떻게 처리할까요?", font=("", 10, "bold")).pack(anchor="w", pady=(0, 12))

        def choose(mode: str | None) -> None:
            result["mode"] = mode
            dialog.destroy()

        ttk.Button(frame, text="이동", command=lambda: choose("move")).pack(fill="x")
        ttk.Button(frame, text="복제해서 이동", command=lambda: choose("copy")).pack(fill="x", pady=(8, 0))
        ttk.Button(frame, text="취소", command=lambda: choose(None)).pack(fill="x", pady=(12, 0))
        self.wait_window(dialog)
        return result["mode"]

    def _resolve_duplicate_move_payloads(self, move_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for payload in move_payloads:
            file_hash = str(payload.get("file_hash") or "").strip()
            if not file_hash:
                path = Path(str(payload.get("file_path", "")))
                if path.exists():
                    file_hash = compute_xxhash64(path)
                    payload["file_hash"] = file_hash
            if file_hash:
                groups.setdefault(file_hash, []).append(payload)
        duplicate_groups = {file_hash: items for file_hash, items in groups.items() if len(items) > 1}
        if not duplicate_groups:
            return move_payloads

        dialog = tk.Toplevel(self)
        dialog.title("중복파일 처리")
        dialog.geometry("900x420")
        dialog.transient(self)
        dialog.grab_set()
        ttk.Label(
            dialog,
            text="xxhash 기준으로 중복 파일이 발견됐습니다. 삭제하지 않으면 그대로 이동/복제됩니다.",
            padding=10,
        ).pack(fill="x")
        tree = ttk.Treeview(dialog, columns=("hash", "representative", "path"), show="headings", selectmode="extended")
        for key, title, width in (
            ("hash", "xxhash", 130),
            ("representative", "보관 기준 파일", 250),
            ("path", "삭제 후보 파일", 470),
        ):
            tree.heading(key, text=title)
            tree.column(key, width=width, anchor="w")
        tree.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        delete_candidates: dict[str, dict[str, Any]] = {}
        for file_hash, items in duplicate_groups.items():
            representative = items[0]
            representative_path = str(representative.get("file_path", ""))
            for payload in items[1:]:
                path = str(payload.get("file_path", ""))
                item_id = tree.insert("", "end", values=(file_hash, representative_path, path))
                delete_candidates[item_id] = payload

        result = {"payloads": move_payloads}

        def delete_payloads(targets: list[dict[str, Any]]) -> None:
            target_paths = {str(payload.get("file_path", "")) for payload in targets}
            deleted = 0
            failed = 0
            for payload in targets:
                path = Path(str(payload.get("file_path", "")))
                try:
                    if path.exists():
                        path.unlink()
                    deleted += 1
                except Exception:
                    failed += 1
            result["payloads"] = [
                payload for payload in move_payloads if str(payload.get("file_path", "")) not in target_paths
            ]
            messagebox.showinfo("중복파일 삭제", f"삭제: {deleted}\n실패: {failed}")
            dialog.destroy()

        def delete_all() -> None:
            targets = list(delete_candidates.values())
            if not messagebox.askyesno("중복파일 전체 삭제", f"중복 삭제 후보 {len(targets)}개를 실제로 삭제할까요?"):
                return
            delete_payloads(targets)

        def delete_selected() -> None:
            targets = [delete_candidates[item_id] for item_id in tree.selection() if item_id in delete_candidates]
            if not targets:
                messagebox.showinfo("안내", "삭제할 중복 파일을 선택하세요.")
                return
            if not messagebox.askyesno("선택 중복파일 삭제", f"선택한 중복 파일 {len(targets)}개를 실제로 삭제할까요?"):
                return
            delete_payloads(targets)

        def skip() -> None:
            result["payloads"] = move_payloads
            dialog.destroy()

        actions = ttk.Frame(dialog, padding=(10, 0, 10, 10))
        actions.pack(fill="x")
        ttk.Button(actions, text="중복파일 전체 삭제", command=delete_all).pack(side="left")
        ttk.Button(actions, text="선택 삭제", command=delete_selected).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="건너뛰기", command=skip).pack(side="right")
        self.wait_window(dialog)
        return list(result["payloads"])

    def move_confirmed_files_to_custom_folder(self) -> None:
        payloads = self._confirmed_current_payloads()
        if not payloads:
            messagebox.showinfo("안내", "먼저 파일을 확정 저장하세요. 확정된 현재 결과만 직접 이동할 수 있습니다.")
            return

        transfer_mode = self._ask_transfer_mode()
        if transfer_mode is None:
            return

        parent_dir = filedialog.askdirectory(title="확정 파일을 넣을 상위 폴더 선택")
        if not parent_dir:
            return
        folder_name = simpledialog.askstring(
            "이동 폴더 이름",
            "확정 파일들을 넣을 새 폴더 이름을 입력하세요.",
            initialvalue=time.strftime("confirmed_%Y%m%d_%H%M%S"),
            parent=self,
        )
        if not folder_name:
            return
        folder_name = folder_name.strip().replace("/", "_").replace("\\", "_")
        if not folder_name:
            messagebox.showinfo("안내", "폴더 이름을 입력하세요.")
            return

        destination_dir = Path(parent_dir) / folder_name
        if not messagebox.askyesno(
            "확정 파일 직접 이동",
            f"확정된 파일 {len(payloads)}개를 아래 폴더로 이동할까요?\n\n{destination_dir}\n\n"
            "원래 위치는 이동 이력에 저장되며, '마지막 이동 되돌리기'로 복구할 수 있습니다.",
        ):
            return

        move_payloads: list[dict[str, Any]] = []
        for payload in payloads:
            result = payload.get("result")
            confidence = result.confidence if isinstance(result, ClassificationResult) else 0.0
            category = str(payload.get("confirmed_category") or "")
            if not category and isinstance(result, ClassificationResult):
                category = result.predicted_category
            move_payloads.append(
                {
                    "file_id": int(payload["file_id"]),
                    "classification_id": int(payload["classification_id"]),
                    "file_path": str(payload.get("file_path", "")),
                    "category": category,
                    "file_hash": str(payload.get("file_hash", "")),
                    "confidence": float(confidence),
                }
            )
        move_payloads = self._resolve_duplicate_move_payloads(move_payloads)
        if not move_payloads:
            messagebox.showinfo("안내", "중복파일 처리 후 이동/복제할 파일이 없습니다.")
            return
        cleanup_empty_source_dirs = False
        if transfer_mode == "move":
            cleanup_empty_source_dirs = messagebox.askyesno(
                "원본 빈 폴더 삭제",
                "파일 이동 후 원래 위치의 폴더가 비어 있으면 삭제할까요?\n"
                "되돌리기를 실행하면 필요한 원본 폴더는 다시 생성됩니다.",
            )

        plan = preview_direct_folder_move_plan(
            repository=self.resources.repository,
            config=self.resources.config,
            payloads=move_payloads,
            destination_dir=destination_dir,
            transfer_mode=transfer_mode,
            cleanup_empty_source_dirs=cleanup_empty_source_dirs,
        )
        self.last_preview_batch_id = int(plan["batch_id"])
        self.last_preview_manifest_path = str(plan["manifest_path"])
        self.last_preview_items = list(plan["items"])
        if not self.last_preview_items:
            self._update_classification_controls()
            messagebox.showinfo("이동 대상 없음", "이동 가능한 원본 파일이 없습니다.")
            return

        create_safety_snapshot(
            repository=self.resources.repository,
            config=self.resources.config,
            reason=f"gui_direct_confirmed_move_batch_{self.last_preview_batch_id}",
        )
        result = commit_move_batch(self.resources.repository, batch_id=self.last_preview_batch_id)
        moved_paths_by_source = {}
        for row in self.resources.repository.fetch_move_items(batch_id=self.last_preview_batch_id):
            actual_destination_path = str(row["actual_destination_path"] or row["destination_path"])
            moved_paths_by_source[str(row["source_path"])] = actual_destination_path
        for payload in payloads:
            destination_path = moved_paths_by_source.get(str(payload.get("file_path", "")))
            if destination_path and transfer_mode == "move":
                payload["original_file_path"] = payload.get("file_path", "")
                payload["file_path"] = destination_path

        self.operation_status_text.set(
            f"확정 파일 {'복제' if transfer_mode == 'copy' else '이동'} batch={result['batch_id']} | moved={result['moved']} | failed={result['failed']}\n"
            f"{destination_dir}"
        )
        self._update_classification_controls()
        self.refresh_stats()
        self.status_text.set(f"확정 파일 {'복제' if transfer_mode == 'copy' else '이동'} 완료: moved={result['moved']}, failed={result['failed']}")
        messagebox.showinfo(
            "확정 파일 처리 완료",
            f"대상 폴더: {destination_dir}\n이동: {result['moved']}\n실패: {result['failed']}",
        )

    def preview_moves(self) -> None:
        payloads = self._confirmed_current_payloads()
        if not payloads:
            messagebox.showinfo("안내", "먼저 현재 결과에서 파일을 확정하세요. 이동 미리보기는 확정된 현재 결과만 사용합니다.")
            return
        classification_ids = [
            int(payload["classification_id"])
            for payload in payloads
            if payload.get("classification_id") is not None
        ]
        if not classification_ids:
            messagebox.showinfo("안내", "미리보기를 만들 확정 분류 ID가 없습니다.")
            return
        plan = preview_move_plan_for_classifications(
            repository=self.resources.repository,
            config=self.resources.config,
            classification_ids=classification_ids,
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

    def show_classification_confirmation_window(self) -> None:
        window = tk.Toplevel(self)
        window.title("분류 확정")
        window.geometry("360x260")
        frame = ttk.Frame(window, padding=16)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="분류 확정", font=("", 12, "bold")).pack(anchor="w", pady=(0, 12))
        ttk.Button(frame, text="선택 확정", command=self.confirm_selected_results).pack(fill="x")
        ttk.Button(frame, text="전체 확정", command=self.confirm_all_results).pack(fill="x", pady=(8, 0))
        ttk.Button(frame, text="서버 확정 전송", command=self.confirm_selected_remote_result).pack(fill="x", pady=(8, 0))
        ttk.Button(frame, text="확정 파일 이동", command=self.move_confirmed_files_to_custom_folder).pack(fill="x", pady=(8, 0))
        ttk.Button(frame, text="확정 수정", command=self.show_confirmation_management_window).pack(fill="x", pady=(8, 0))
        ttk.Button(frame, text="닫기", command=window.destroy).pack(fill="x", pady=(12, 0))

    def show_settings_window(self) -> None:
        window = tk.Toplevel(self)
        window.title("설정")
        window.geometry("360x410")
        frame = ttk.Frame(window, padding=16)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="설정", font=("", 12, "bold")).pack(anchor="w", pady=(0, 12))
        ttk.Button(frame, text="통계", command=self.show_stats_window).pack(fill="x")
        ttk.Button(frame, text="DB 초기화", command=self.init_db).pack(fill="x", pady=(8, 0))
        ttk.Button(frame, text="이동 이력 보기", command=self.show_move_history_window).pack(fill="x", pady=(8, 0))
        ttk.Button(frame, text="마지막 이동 되돌리기", command=self.undo_last_move_ui).pack(fill="x", pady=(8, 0))
        ttk.Button(frame, text="피드백 로그 관리", command=self.show_feedback_logs_window).pack(fill="x", pady=(8, 0))
        ttk.Button(frame, text="임베딩 캐시 관리", command=self.show_embedding_cache_window).pack(fill="x", pady=(8, 0))
        ttk.Button(frame, text="성능 분석 보기", command=self.show_performance_window).pack(fill="x", pady=(8, 0))
        ttk.Button(frame, text="서버 연결 설정", command=self.show_remote_server_settings_window).pack(fill="x", pady=(8, 0))
        ttk.Button(frame, text="닫기", command=window.destroy).pack(fill="x", pady=(12, 0))

    def show_remote_server_settings_window(self) -> None:
        window = tk.Toplevel(self)
        window.title("서버 연결 설정")
        window.geometry("520x260")
        frame = ttk.Frame(window, padding=16)
        frame.pack(fill="both", expand=True)
        remote = self.resources.config.remote
        enabled_value = tk.BooleanVar(value=bool(remote.enabled))
        websocket_value = tk.BooleanVar(value=bool(remote.websocket_enabled))
        base_url_value = tk.StringVar(value=str(remote.base_url))
        timeout_value = tk.StringVar(value=str(remote.timeout_seconds))
        poll_value = tk.StringVar(value=str(remote.poll_interval_seconds))

        ttk.Checkbutton(frame, text="서버 연결 사용", variable=enabled_value).pack(anchor="w")
        ttk.Label(frame, text="서버 주소").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=base_url_value).pack(fill="x")
        ttk.Label(frame, text="요청 timeout seconds").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=timeout_value).pack(fill="x")
        ttk.Label(frame, text="결과 조회 interval seconds").pack(anchor="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=poll_value).pack(fill="x")
        ttk.Checkbutton(frame, text="WebSocket 진행 상태 사용", variable=websocket_value).pack(anchor="w", pady=(10, 0))

        def save_remote_settings() -> None:
            try:
                remote.enabled = bool(enabled_value.get())
                remote.base_url = base_url_value.get().strip() or "http://localhost:8000"
                remote.timeout_seconds = int(float(timeout_value.get()))
                remote.poll_interval_seconds = float(poll_value.get())
                remote.websocket_enabled = bool(websocket_value.get())
                save_app_config(self.resources.config, DEFAULT_CONFIG_PATH)
            except ValueError:
                messagebox.showerror("설정 오류", "timeout과 interval은 숫자로 입력하세요.")
                return
            self.status_text.set(f"서버 설정 저장: {remote.base_url}")
            window.destroy()

        button_row = ttk.Frame(frame)
        button_row.pack(fill="x", pady=(14, 0))
        ttk.Button(button_row, text="저장", command=save_remote_settings).pack(side="right")
        ttk.Button(button_row, text="닫기", command=window.destroy).pack(side="right", padx=(0, 8))

    def show_confirmation_management_window(self) -> None:
        window = tk.Toplevel(self)
        window.title("확정 수정")
        window.geometry("1120x620")

        notebook = ttk.Notebook(window)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        file_tab = ttk.Frame(notebook)
        batch_tab = ttk.Frame(notebook)
        notebook.add(batch_tab, text="실행별 확정")
        notebook.add(file_tab, text="파일별 확정")

        file_tree = ttk.Treeview(
            file_tab,
            columns=("id", "batch", "file", "from", "to", "action", "created"),
            show="headings",
            selectmode="extended",
        )
        for key, title, width in (
            ("id", "log id", 70),
            ("batch", "묶음", 170),
            ("file", "파일", 260),
            ("from", "기존", 120),
            ("to", "확정", 120),
            ("action", "동작", 90),
            ("created", "시간", 150),
        ):
            file_tree.heading(key, text=title)
            file_tree.column(key, width=width, anchor="w")
        file_tree.pack(fill="both", expand=True, padx=8, pady=8)

        batch_search = tk.StringVar()
        batch_search_frame = ttk.Frame(batch_tab)
        batch_search_frame.pack(fill="x", padx=8, pady=(8, 0))
        ttk.Label(batch_search_frame, text="실행 검색").pack(side="left")
        ttk.Entry(batch_search_frame, textvariable=batch_search).pack(side="left", fill="x", expand=True, padx=(8, 0))

        batch_tree = ttk.Treeview(
            batch_tab,
            columns=("name", "count", "categories", "created", "updated", "batch"),
            show="headings",
        )
        for key, title, width in (
            ("name", "실행 이름", 220),
            ("count", "파일 수", 80),
            ("categories", "확정 카테고리", 260),
            ("created", "시작", 150),
            ("updated", "마지막", 150),
            ("batch", "id", 180),
        ):
            batch_tree.heading(key, text=title)
            batch_tree.column(key, width=width, anchor="w")
        batch_tree.pack(fill="both", expand=True, padx=8, pady=(8, 4))

        batch_file_tree = ttk.Treeview(
            batch_tab,
            columns=("file", "from", "to", "action", "created"),
            show="headings",
            height=8,
        )
        for key, title, width in (
            ("file", "실행 안의 파일", 320),
            ("from", "기존", 120),
            ("to", "확정", 120),
            ("action", "동작", 90),
            ("created", "시간", 150),
        ):
            batch_file_tree.heading(key, text=title)
            batch_file_tree.column(key, width=width, anchor="w")
        batch_file_tree.pack(fill="x", padx=8, pady=(0, 8))

        detail = tk.Text(window, height=8, wrap="word")
        detail.pack(fill="x", padx=10, pady=(0, 10))

        file_rows: dict[str, dict[str, Any]] = {}
        batch_rows: dict[str, dict[str, Any]] = {}
        batch_file_rows: dict[str, dict[str, Any]] = {}

        def refresh() -> None:
            for tree in (file_tree, batch_tree, batch_file_tree):
                for item_id in tree.get_children():
                    tree.delete(item_id)
            file_rows.clear()
            batch_rows.clear()
            batch_file_rows.clear()
            detail.delete("1.0", "end")

            all_file_rows = []
            for row in self.resources.repository.list_feedback_logs():
                row_dict = dict(row)
                batch_id = str(row_dict.get("confirmation_batch_id") or f"legacy-{row_dict['id']}")
                row_dict["display_confirmation_batch_id"] = batch_id
                all_file_rows.append(row_dict)
                item_id = file_tree.insert(
                    "",
                    "end",
                    values=(
                        row_dict["id"],
                        batch_id,
                        row_dict["file_name"],
                        row_dict["predicted_middle_category"] or row_dict["predicted_category"],
                        row_dict["final_middle_category"] or row_dict["final_category"],
                        row_dict["feedback_action"],
                        row_dict["created_at"],
                    ),
                )
                file_rows[item_id] = row_dict

            query = batch_search.get().strip().lower()
            for index, row in enumerate(self.resources.repository.list_confirmation_batches(), start=1):
                row_dict = dict(row)
                display_name = str(row_dict.get("confirmation_batch_name") or "").strip() or f"실행 {index}"
                row_dict["display_name"] = display_name
                searchable = " ".join(
                    [
                        display_name,
                        str(row_dict.get("confirmation_batch_id") or ""),
                        str(row_dict.get("categories") or ""),
                        str(row_dict.get("file_names") or ""),
                    ]
                ).lower()
                if query and query not in searchable:
                    continue
                item_id = batch_tree.insert(
                    "",
                    "end",
                    values=(
                        display_name,
                        row_dict["file_count"],
                        row_dict.get("categories") or "",
                        row_dict.get("created_at") or "",
                        row_dict.get("updated_at") or "",
                        row_dict["confirmation_batch_id"],
                    ),
                )
                batch_rows[item_id] = row_dict

        def show_file_detail(_event: object | None = None) -> None:
            selection = file_tree.selection()
            if not selection:
                return
            selected = [file_rows[item_id] for item_id in selection if item_id in file_rows]
            detail.delete("1.0", "end")
            detail.insert("1.0", json.dumps(selected, ensure_ascii=False, indent=2))

        def show_batch_detail(_event: object | None = None) -> None:
            selection = batch_tree.selection()
            if not selection:
                return
            selected = batch_rows.get(selection[0])
            for item_id in batch_file_tree.get_children():
                batch_file_tree.delete(item_id)
            batch_file_rows.clear()
            if selected:
                selected_batch_id = str(selected["confirmation_batch_id"])
                for row in self.resources.repository.list_feedback_logs():
                    row_dict = dict(row)
                    row_batch_id = str(row_dict.get("confirmation_batch_id") or f"legacy-{row_dict['id']}")
                    if row_batch_id != selected_batch_id:
                        continue
                    item_id = batch_file_tree.insert(
                        "",
                        "end",
                        values=(
                            row_dict["file_name"],
                            row_dict["predicted_middle_category"] or row_dict["predicted_category"],
                            row_dict["final_middle_category"] or row_dict["final_category"],
                            row_dict["feedback_action"],
                            row_dict["created_at"],
                        ),
                    )
                    batch_file_rows[item_id] = row_dict
            detail.delete("1.0", "end")
            detail.insert("1.0", json.dumps(selected or {}, ensure_ascii=False, indent=2))

        def delete_selected_files() -> None:
            selection = file_tree.selection()
            if not selection:
                messagebox.showinfo("안내", "취소할 확정 파일을 선택하세요.")
                return
            targets = [file_rows[item_id] for item_id in selection if item_id in file_rows]
            if not messagebox.askyesno("파일별 확정 취소", f"선택한 확정 {len(targets)}개를 취소할까요?"):
                return
            deleted = 0
            for row in targets:
                deleted += self.resources.repository.delete_feedback_log(int(row["id"]))
            rebuild_adaptive_learning(self.resources.repository, min_occurrences=2)
            refresh()
            self.refresh_stats()
            messagebox.showinfo("파일별 확정 취소", f"취소 완료: {deleted}개")

        def delete_selected_batch() -> None:
            selection = batch_tree.selection()
            if not selection:
                messagebox.showinfo("안내", "취소할 확정 묶음을 선택하세요.")
                return
            selected = batch_rows[selection[0]]
            batch_id = str(selected["confirmation_batch_id"])
            count = int(selected["file_count"])
            if not messagebox.askyesno("묶음별 확정 취소", f"'{batch_id}' 묶음의 확정 {count}개를 모두 취소할까요?"):
                return
            deleted = self.resources.repository.delete_confirmation_batch(batch_id)
            rebuild_adaptive_learning(self.resources.repository, min_occurrences=2)
            refresh()
            self.refresh_stats()
            messagebox.showinfo("묶음별 확정 취소", f"취소 완료: {deleted}개")

        def rename_selected_batch() -> None:
            selection = batch_tree.selection()
            if not selection:
                messagebox.showinfo("안내", "이름을 변경할 실행을 선택하세요.")
                return
            selected = batch_rows[selection[0]]
            batch_id = str(selected["confirmation_batch_id"])
            current_name = str(selected.get("display_name") or "")
            new_name = simpledialog.askstring(
                "실행 이름 변경",
                "새 실행 이름을 입력하세요.",
                initialvalue=current_name,
                parent=window,
            )
            if new_name is None:
                return
            updated = self.resources.repository.update_confirmation_batch_name(batch_id, new_name.strip())
            if updated <= 0:
                messagebox.showinfo("실행 이름 변경", "변경할 실행을 찾지 못했습니다.")
                return
            refresh()
            messagebox.showinfo("실행 이름 변경", "실행 이름을 변경했습니다.")

        file_tree.bind("<<TreeviewSelect>>", show_file_detail)
        batch_tree.bind("<<TreeviewSelect>>", show_batch_detail)
        batch_search.trace_add("write", lambda *_args: refresh())

        actions = ttk.Frame(window, padding=(10, 0, 10, 10))
        actions.pack(fill="x")
        ttk.Button(actions, text="새로고침", command=refresh).pack(side="left")
        ttk.Button(actions, text="선택 파일 확정 취소", command=delete_selected_files).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="선택 묶음 확정 취소", command=delete_selected_batch).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="실행 이름 변경", command=rename_selected_batch).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="닫기", command=window.destroy).pack(side="right")

        refresh()

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

        if self.last_run_summary.get("pipeline") == "cluster" or any(
            payload.get("pipeline") == "cluster" for payload in self.all_payloads
        ):
            stage_totals: dict[str, float] = {}
            slowest_rows: list[dict[str, Any]] = []
            status_counts: dict[str, int] = {}
            cache_hits = 0
            for payload in self.all_payloads:
                if payload.get("pipeline") != "cluster":
                    continue
                evidence = payload.get("evidence", {})
                if not isinstance(evidence, dict):
                    continue
                timings = evidence.get("timings", {})
                if not isinstance(timings, dict):
                    timings = {}
                total = 0.0
                for key, value in timings.items():
                    try:
                        elapsed = float(value)
                    except (TypeError, ValueError):
                        continue
                    if key != "total":
                        stage_totals[str(key)] = stage_totals.get(str(key), 0.0) + elapsed
                    total += elapsed
                status = str(evidence.get("extraction_status", "unknown"))
                status_counts[status] = status_counts.get(status, 0) + 1
                if bool(evidence.get("ocr_cache_hit")):
                    cache_hits += 1
                slowest_rows.append(
                    {
                        "file_name": str(payload.get("file_name", "")),
                        "total_time": total,
                        "status": status,
                    }
                )

            stage_total_lines = "\n".join(
                f"- {key}: {value:.3f}s"
                for key, value in sorted(stage_totals.items(), key=lambda item: item[1], reverse=True)
            ) or "- no evidence timings recorded"
            slow_lines = "\n".join(
                f"- {row['file_name']} | {row['total_time']:.3f}s | {row['status']}"
                for row in sorted(slowest_rows, key=lambda item: item["total_time"], reverse=True)[:10]
            ) or "- no clustered files yet"
            status_lines = "\n".join(
                f"- {status}: {count}" for status, count in sorted(status_counts.items())
            ) or "- no extraction statuses recorded"
            summary = self.last_run_summary
            text.insert(
                "1.0",
                (
                    "Cluster pipeline timings\n"
                    f"- run_elapsed: {float(summary.get('run_elapsed', 0.0)):.3f}s\n"
                    f"- evidence_elapsed: {float(summary.get('evidence_elapsed', 0.0)):.3f}s\n"
                    f"- embedding_elapsed: {float(summary.get('embedding_elapsed', 0.0)):.3f}s\n"
                    f"- clustering_elapsed: {float(summary.get('clustering_elapsed', 0.0)):.3f}s\n"
                    f"- file_count: {summary.get('file_count', len(self.all_payloads))}\n"
                    f"- cluster_count: {summary.get('cluster_count', 0)}\n"
                    f"- noise_count: {summary.get('noise_count', 0)}\n"
                    f"- projection_file: {summary.get('projection_file', '')}\n"
                    f"- api_call_performed: {summary.get('api_call_performed', False)}\n"
                    f"- ocr_cache_hits: {cache_hits}\n\n"
                    "Extraction statuses\n"
                    f"{status_lines}\n\n"
                    "Evidence stage totals\n"
                    f"{stage_total_lines}\n\n"
                    "Slowest evidence files\n"
                    f"{slow_lines}\n"
                ),
            )
            return

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
