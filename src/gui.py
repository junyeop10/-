"""Tkinter GUI for the classifier."""

from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from src.classifier import ClassificationResult, HybridClassifier
from src.cli import load_categories
from src.file_reader import discover_supported_files, ensure_input_directory, extract_text_from_file
from src.hash_utils import compute_xxhash64
from src.ocr_support import DEFAULT_OCR_MIN_CHARS, explain_ocr_decision, ocr_pdf_file
from src.rule_classifier import RuleBasedClassifier
from src.storage import ClassificationRepository
from src.text_cleaner import normalize_text
from src.vectorizer import SentenceTransformerEmbedder


DB_PATH = "data/classifier.db"
CATEGORIES_PATH = "data/categories.json"

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    BaseWindow = TkinterDnD.Tk
    DRAG_AND_DROP_AVAILABLE = True
except ImportError:
    DND_FILES = ""
    BaseWindow = tk.Tk
    DRAG_AND_DROP_AVAILABLE = False


@dataclass
class AppResources:
    """Prepared objects reused across the GUI session."""

    repository: ClassificationRepository
    embedder: SentenceTransformerEmbedder
    rule_classifier: RuleBasedClassifier


class ClassifierGui(BaseWindow):
    """Desktop UI for the document classifier."""

    def __init__(self) -> None:
        super().__init__()
        self.title("파일 분류 MVP")
        self.geometry("1080x680")
        self.minsize(900, 560)
        self.protocol("WM_DELETE_WINDOW", self._handle_close)

        repository = ClassificationRepository(DB_PATH)
        repository.initialize_database()
        repository.seed_rules_from_categories(load_categories(Path(CATEGORIES_PATH)))
        embedder = SentenceTransformerEmbedder()
        self.resources = AppResources(
            repository=repository,
            embedder=embedder,
            rule_classifier=RuleBasedClassifier(repository),
        )
        self.embedding_ready = False

        self.input_dir = tk.StringVar(value=str(Path("input_files").resolve()))
        self.final_category = tk.StringVar()
        self.search_query = tk.StringVar()
        self.category_filter = tk.StringVar(value="전체")
        self.status_text = tk.StringVar(value="준비 완료")
        self.progress_text = tk.StringVar(value="진행률 0/0")
        self.progress_value = tk.DoubleVar(value=0.0)

        self.result_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.embedding_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.all_payloads: list[dict[str, object]] = []
        self.item_payloads: dict[str, dict[str, object]] = {}
        self.category_combo: ttk.Combobox | None = None
        self.total_files = 0
        self.processed_files = 0
        self.detail_text: tk.Text | None = None
        self.stats_label: ttk.Label | None = None
        self.drop_label: ttk.Label | None = None
        self.tree: ttk.Treeview | None = None
        self.progress_bar: ttk.Progressbar | None = None

        self.search_query.trace_add("write", lambda *_args: self.apply_filename_filter())
        self.category_filter.trace_add("write", lambda *_args: self.apply_category_filter())

        self._build_main_ui()
        self.refresh_stats()
        self._start_embedding_warmup()

    def _build_main_ui(self) -> None:
        """Build the main window layout."""
        top_bar = ttk.Frame(self, padding=10)
        top_bar.pack(fill="x")

        ttk.Label(top_bar, text="입력 폴더").pack(side="left")
        ttk.Entry(top_bar, textvariable=self.input_dir).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(top_bar, text="폴더 선택", command=self.choose_folder).pack(side="left")
        ttk.Button(top_bar, text="DB 초기화", command=self.init_db).pack(side="left", padx=(8, 0))
        ttk.Button(top_bar, text="분류 실행", command=self.start_classify).pack(side="left", padx=(8, 0))
        ttk.Button(top_bar, text="통계", command=self.refresh_stats).pack(side="left", padx=(8, 0))

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
        main_pane.add(left_frame, weight=3)
        main_pane.add(right_frame, weight=2)

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

        columns = ("file", "category", "final_score", "rule_score", "embedding_score", "feedback_score")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=18)
        self.tree.heading("file", text="파일")
        self.tree.heading("category", text="추천")
        self.tree.heading("final_score", text="최종")
        self.tree.heading("rule_score", text="룰")
        self.tree.heading("embedding_score", text="임베딩")
        self.tree.heading("feedback_score", text="피드백")
        self.tree.column("file", width=320)
        self.tree.column("category", width=120, anchor="center")
        self.tree.column("final_score", width=70, anchor="center")
        self.tree.column("rule_score", width=70, anchor="center")
        self.tree.column("embedding_score", width=70, anchor="center")
        self.tree.column("feedback_score", width=70, anchor="center")
        self.tree.tag_configure("ocr_used", foreground="#7a7a7a")
        self.tree.bind("<<TreeviewSelect>>", self.on_select_result)
        self.tree.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

        progress_frame = ttk.Frame(left_frame)
        progress_frame.pack(fill="x", pady=(8, 0))
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_value,
            maximum=100,
            mode="determinate",
        )
        self.progress_bar.pack(side="left", fill="x", expand=True)
        ttk.Label(progress_frame, textvariable=self.progress_text, width=16, anchor="e").pack(
            side="left",
            padx=(8, 0),
        )

        ttk.Label(right_frame, text="추천 근거").pack(anchor="w")
        self.detail_text = tk.Text(right_frame, height=16, wrap="word")
        self.detail_text.pack(fill="both", expand=True, pady=(4, 10))

        review_frame = ttk.LabelFrame(right_frame, text="검토 저장", padding=10)
        review_frame.pack(fill="x")
        ttk.Label(review_frame, text="최종 카테고리").pack(anchor="w")
        ttk.Entry(review_frame, textvariable=self.final_category).pack(fill="x", pady=(4, 8))
        ttk.Button(review_frame, text="선택 결과 확정/수정 저장", command=self.save_selected_feedback).pack(
            fill="x"
        )

        stats_frame = ttk.LabelFrame(right_frame, text="DB 통계", padding=10)
        stats_frame.pack(fill="x", pady=(10, 0))
        self.stats_label = ttk.Label(stats_frame, text="", justify="left")
        self.stats_label.pack(anchor="w")

        ttk.Label(self, textvariable=self.status_text, padding=(10, 0, 10, 8)).pack(fill="x")

    def _start_embedding_warmup(self) -> None:
        """Load the embedding model in the background after the main UI is visible."""
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
                self.embedding_ready = True
                self.status_text.set("준비 완료 | 임베딩 모델 로드됨")
                return
            if event == "error":
                self.status_text.set(f"준비 완료 | 임베딩 로드 실패: {payload}")
                return
        self.after(150, self._poll_embedding_queue)

    def choose_folder(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.input_dir.get() or ".")
        if selected:
            self.input_dir.set(selected)

    def init_db(self) -> None:
        self.resources.repository.initialize_database()
        self.resources.repository.seed_rules_from_categories(load_categories(Path(CATEGORIES_PATH)))
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
                f"rules: {stats['rules_count']}"
            )
        )

    def start_classify(self) -> None:
        self._clear_results()
        self.status_text.set("파일 목록 준비 중")
        input_dir = ensure_input_directory(self.input_dir.get())
        files = discover_supported_files(input_dir)
        self.start_classify_files(files)

    def start_classify_files(self, files: list[Path]) -> None:
        if not files:
            messagebox.showinfo("안내", "분류할 txt/pdf 파일이 없습니다.")
            return

        self.status_text.set("분류 엔진 준비 중")
        self._reset_progress(len(files))
        worker = threading.Thread(target=self._classify_worker, args=(files,), daemon=True)
        worker.start()
        self.after(100, self._drain_queue)

    def on_drop_files(self, event: object) -> None:
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
        raw_text = extract_text_from_file(file_path, fast=True)
        normalized_text = normalize_text(raw_text)
        ocr_used = False
        ocr_pages = 0

        ocr_decision = explain_ocr_decision(
            file_path=file_path,
            extracted_text=normalized_text,
            min_text_length=DEFAULT_OCR_MIN_CHARS,
        )
        if ocr_decision["run_ocr"]:
            ocr_result = ocr_pdf_file(file_path)
            ocr_text = normalize_text(str(ocr_result.get("text", "")))
            if ocr_result.get("ok") and ocr_text:
                normalized_text = ocr_text
                ocr_used = True
                ocr_pages = int(ocr_result.get("pages_scanned", 0))
        elif ocr_decision["classification_hint"]:
            hint_evidence = str(ocr_decision.get("hint_evidence", "")).strip()
            if hint_evidence:
                normalized_text = f"{hint_evidence} {normalized_text}".strip()

        file_hash = compute_xxhash64(file_path)
        duplicate_of_file_id = self.resources.repository.find_duplicate_file_id(file_hash, str(file_path.resolve()))

        file_id = self.resources.repository.upsert_file(
            file_path=str(file_path.resolve()),
            file_name=file_path.name,
            file_ext=file_path.suffix.lower(),
            file_size=file_path.stat().st_size,
            xxhash64=file_hash,
            duplicate_of_file_id=duplicate_of_file_id,
            extracted_text=normalized_text,
        )
        result = classifier.classify_file(
            file_id=file_id,
            file_hash=file_hash,
            text=normalized_text,
            duplicate_of_file_id=duplicate_of_file_id,
            file_name=file_path.name,
        )
        if ocr_used:
            result = ClassificationResult(
                predicted_category=result.predicted_category,
                confidence=result.confidence,
                final_score=result.final_score,
                rule_score=result.rule_score,
                embedding_score=result.embedding_score,
                llm_score=result.llm_score,
                feedback_score=result.feedback_score,
                duplicate_score=result.duplicate_score,
                similarity_score=result.similarity_score,
                embedding_used=result.embedding_used,
                review_required=result.review_required,
                matched_rules=result.matched_rules,
                candidate_scores=result.candidate_scores,
                reasoning=f"{result.reasoning} | ocr=used(pages={ocr_pages})",
                query_embedding=result.query_embedding,
            )
        classification_id = classifier.persist_classification(file_id=file_id, result=result)

        return {
            "file_id": file_id,
            "classification_id": classification_id,
            "file_name": file_path.name,
            "text": normalized_text,
            "ocr_used": ocr_used,
            "result": result,
        }

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
                suffix = " | 임베딩 모델 로드됨" if self.embedding_ready else " | 임베딩 백그라운드 로딩 중"
                self.status_text.set(f"분류 완료{suffix}")
                self._set_progress(self.total_files, self.total_files)
                self.refresh_stats()
                return
        self.after(100, self._drain_queue)

    def _insert_result(self, payload: dict[str, object]) -> None:
        self.all_payloads.append(payload)
        self._refresh_category_options()
        self._refresh_result_table()

    def apply_filename_filter(self) -> None:
        self._refresh_result_table()

    def apply_category_filter(self) -> None:
        self._refresh_result_table()

    def clear_filename_filter(self) -> None:
        self.search_query.set("")
        self.category_filter.set("전체")

    def _refresh_result_table(self) -> None:
        if self.tree is None:
            return
        selected_file = self._selected_file_name()
        for item_id in self.tree.get_children():
            self.tree.delete(item_id)
        self.item_payloads.clear()

        query = self.search_query.get().strip().lower()
        category_filter = self.category_filter.get().strip()
        for payload in self.all_payloads:
            file_name = str(payload["file_name"])
            result = payload["result"]
            assert isinstance(result, ClassificationResult)
            if query and query not in file_name.lower():
                continue
            if category_filter and category_filter != "전체" and result.predicted_category != category_filter:
                continue
            item_id = self._insert_payload_row(payload)
            if selected_file and file_name == selected_file:
                self.tree.selection_set(item_id)
                self.tree.focus(item_id)

        visible_count = len(self.tree.get_children())
        total_count = len(self.all_payloads)
        if query or category_filter != "전체":
            self.status_text.set(f"필터 결과: {visible_count}/{total_count}개")

    def _selected_file_name(self) -> str | None:
        if self.tree is None:
            return None
        selection = self.tree.selection()
        if not selection:
            return None
        payload = self.item_payloads.get(selection[0])
        if not payload:
            return None
        return str(payload["file_name"])

    def _insert_payload_row(self, payload: dict[str, object]) -> str:
        if self.tree is None:
            raise RuntimeError("Tree view is not ready.")
        result = payload["result"]
        assert isinstance(result, ClassificationResult)
        item_id = self.tree.insert(
            "",
            "end",
            values=(
                payload["file_name"],
                result.predicted_category,
                f"{result.final_score:.3f}",
                f"{result.rule_score:.3f}",
                f"{result.embedding_score:.3f}",
                f"{result.feedback_score:.3f}",
            ),
            tags=("ocr_used",) if bool(payload.get("ocr_used")) else (),
        )
        self.item_payloads[item_id] = payload
        return item_id

    def _reset_progress(self, total_files: int) -> None:
        self.total_files = total_files
        self.processed_files = 0
        self._set_progress(0, total_files)

    def _advance_progress(self) -> None:
        self.processed_files = min(self.processed_files + 1, self.total_files)
        self._set_progress(self.processed_files, self.total_files)

    def _set_progress(self, processed_files: int, total_files: int) -> None:
        percent = (processed_files / total_files * 100) if total_files else 0.0
        self.progress_value.set(percent)
        self.progress_text.set(f"진행률 {processed_files}/{total_files}")

    def _show_current_file(self, payload: dict[str, object]) -> None:
        index = int(payload["index"])
        file_name = str(payload["file_name"])
        suffix = " | 임베딩 모델 로드됨" if self.embedding_ready else " | 임베딩 백그라운드 로딩 중"
        self.status_text.set(f"처리 중 ({index}/{self.total_files}): {file_name}{suffix}")
        self.progress_text.set(f"진행률 {self.processed_files}/{self.total_files}")

    def _refresh_category_options(self) -> None:
        if self.category_combo is None:
            return
        categories = sorted(
            {
                payload["result"].predicted_category
                for payload in self.all_payloads
                if isinstance(payload["result"], ClassificationResult)
            }
        )
        values = ["전체", *categories]
        self.category_combo.configure(values=values)
        if self.category_filter.get() not in values:
            self.category_filter.set("전체")

    def on_select_result(self, _event: object | None = None) -> None:
        if self.tree is None:
            return
        selection = self.tree.selection()
        if not selection:
            return

        payload = self.item_payloads.get(selection[0])
        if not payload:
            return

        result = payload["result"]
        assert isinstance(result, ClassificationResult)
        self.final_category.set(result.predicted_category)

        matched_rules = ", ".join(result.matched_rules) if result.matched_rules else "없음"
        similarity_text = f"{result.similarity_score:.3f}" if result.embedding_used else "skipped"
        detail = (
            f"파일: {payload['file_name']}\n"
            f"추천: {result.predicted_category}\n"
            f"confidence: {result.confidence:.3f}\n"
            f"review_required: {'yes' if result.review_required else 'no'}\n"
            f"similarity: {similarity_text}\n"
            f"점수: rule={result.rule_score:.3f}, embedding={result.embedding_score:.3f}, "
            f"feedback={result.feedback_score:.3f}, final={result.final_score:.3f}\n"
            f"매칭 규칙: {matched_rules}\n"
            f"후보 점수: {json.dumps(result.candidate_scores, ensure_ascii=False)}\n"
            f"근거: {result.reasoning}\n"
        )
        self._set_detail(detail)

    def save_selected_feedback(self) -> None:
        if self.tree is None:
            return
        selection = self.tree.selection()
        if not selection:
            messagebox.showinfo("안내", "먼저 결과를 선택하세요.")
            return

        payload = self.item_payloads.get(selection[0])
        if not payload:
            return

        result = payload["result"]
        assert isinstance(result, ClassificationResult)
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
        embedding = result.query_embedding or self.resources.embedder.encode(str(payload["text"]))

        self.resources.repository.save_confirmed_example(
            file_id=int(payload["file_id"]),
            category=final_category,
            source_text=str(payload["text"]),
            embedding=embedding,
            source_feedback_log_id=feedback_id,
        )
        self.refresh_stats()
        self.status_text.set(f"저장 완료: {result.predicted_category} -> {final_category}")

    def _clear_results(self) -> None:
        if self.tree is not None:
            for item_id in self.tree.get_children():
                self.tree.delete(item_id)
        self.all_payloads.clear()
        self.item_payloads.clear()
        self._set_progress(0, 0)
        self._refresh_category_options()
        self._set_detail("")

    def _set_detail(self, text: str) -> None:
        if self.detail_text is None:
            return
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("1.0", text)
        self.detail_text.configure(state="disabled")

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
