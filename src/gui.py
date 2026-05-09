"""Small Tkinter UI for the file classifier MVP."""

from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from src.classifier import ClassificationResult, HybridClassifier
from src.cli import load_categories
from src.file_reader import discover_supported_files, ensure_input_directory, extract_text_from_file
from src.hash_utils import compute_xxhash64
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


class ClassifierGui(BaseWindow):
    """A minimal desktop UI that keeps the existing CLI engine intact."""

    def __init__(self) -> None:
        """Create widgets and prepare the repository."""
        super().__init__()
        self.title("파일 분류 MVP")
        self.geometry("1080x680")
        self.minsize(900, 560)

        self.repository = ClassificationRepository(DB_PATH)
        self.repository.initialize_database()
        self.repository.seed_rules_from_categories(load_categories(Path(CATEGORIES_PATH)))

        self.input_dir = tk.StringVar(value=str(Path("input_files").resolve()))
        self.final_category = tk.StringVar()
        self.status_text = tk.StringVar(value="준비됨")
        self.result_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.item_payloads: dict[str, dict[str, object]] = {}

        self._build_ui()
        self.refresh_stats()

    def _build_ui(self) -> None:
        """Build the window layout."""
        top_bar = ttk.Frame(self, padding=10)
        top_bar.pack(fill="x")

        ttk.Label(top_bar, text="입력 폴더").pack(side="left")
        ttk.Entry(top_bar, textvariable=self.input_dir).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(top_bar, text="폴더 선택", command=self.choose_folder).pack(side="left")
        ttk.Button(top_bar, text="DB 초기화", command=self.init_db).pack(side="left", padx=(8, 0))
        ttk.Button(top_bar, text="분류 실행", command=self.start_classify).pack(side="left", padx=(8, 0))
        ttk.Button(top_bar, text="통계", command=self.refresh_stats).pack(side="left", padx=(8, 0))

        drop_text = "여기에 txt/pdf 파일 또는 폴더를 드래그하세요"
        if not DRAG_AND_DROP_AVAILABLE:
            drop_text = "드래그 기능: tkinterdnd2 설치 후 사용 가능"
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

        columns = ("file", "category", "final_score", "rule_score", "embedding_score", "feedback_score")
        self.tree = ttk.Treeview(left_frame, columns=columns, show="headings", height=18)
        self.tree.heading("file", text="파일")
        self.tree.heading("category", text="추천")
        self.tree.heading("final_score", text="최종")
        self.tree.heading("rule_score", text="규칙")
        self.tree.heading("embedding_score", text="유사도")
        self.tree.heading("feedback_score", text="피드백")
        self.tree.column("file", width=320)
        self.tree.column("category", width=110, anchor="center")
        self.tree.column("final_score", width=70, anchor="center")
        self.tree.column("rule_score", width=70, anchor="center")
        self.tree.column("embedding_score", width=70, anchor="center")
        self.tree.column("feedback_score", width=70, anchor="center")
        self.tree.bind("<<TreeviewSelect>>", self.on_select_result)
        self.tree.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

        ttk.Label(right_frame, text="추천 근거").pack(anchor="w")
        self.detail_text = tk.Text(right_frame, height=16, wrap="word")
        self.detail_text.pack(fill="both", expand=True, pady=(4, 10))

        review_frame = ttk.LabelFrame(right_frame, text="검토 저장", padding=10)
        review_frame.pack(fill="x")
        ttk.Label(review_frame, text="최종 카테고리").pack(anchor="w")
        ttk.Entry(review_frame, textvariable=self.final_category).pack(fill="x", pady=(4, 8))
        ttk.Button(review_frame, text="선택 결과 확정/수정 저장", command=self.save_selected_feedback).pack(fill="x")

        stats_frame = ttk.LabelFrame(right_frame, text="DB 통계", padding=10)
        stats_frame.pack(fill="x", pady=(10, 0))
        self.stats_label = ttk.Label(stats_frame, text="", justify="left")
        self.stats_label.pack(anchor="w")

        status_bar = ttk.Label(self, textvariable=self.status_text, padding=(10, 0, 10, 8))
        status_bar.pack(fill="x")

    def choose_folder(self) -> None:
        """Let the user pick an input folder."""
        selected = filedialog.askdirectory(initialdir=self.input_dir.get() or ".")
        if selected:
            self.input_dir.set(selected)

    def init_db(self) -> None:
        """Initialize DB and seed rules."""
        self.repository.initialize_database()
        self.repository.seed_rules_from_categories(load_categories(Path(CATEGORIES_PATH)))
        self.refresh_stats()
        self.status_text.set("DB 초기화 완료")

    def refresh_stats(self) -> None:
        """Refresh the DB count panel."""
        stats = self.repository.get_stats()
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
        """Run classification in a background thread."""
        self._clear_results()
        input_dir = ensure_input_directory(self.input_dir.get())
        files = discover_supported_files(input_dir)
        self.start_classify_files(files)

    def start_classify_files(self, files: list[Path]) -> None:
        """Run classification for a prepared file list."""
        if not files:
            messagebox.showinfo("안내", "분류할 txt/pdf 파일이 없습니다.")
            return

        self.status_text.set(f"분류 중: {len(files)}개")
        worker = threading.Thread(target=self._classify_worker, args=(files,), daemon=True)
        worker.start()
        self.after(100, self._drain_queue)

    def on_drop_files(self, event: object) -> None:
        """Classify files or folders dropped onto the drop area."""
        raw_data = getattr(event, "data", "")
        dropped_paths = [Path(value) for value in self.tk.splitlist(raw_data)]
        files: list[Path] = []

        for path in dropped_paths:
            if path.is_dir():
                files.extend(discover_supported_files(path))
            elif path.is_file() and path.suffix.lower() in {".txt", ".pdf"}:
                files.append(path)

        self._clear_results()
        self.start_classify_files(sorted(set(files)))

    def _classify_worker(self, files: list[Path]) -> None:
        """Classify files and send results to the UI queue."""
        classifier = HybridClassifier(
            repository=self.repository,
            embedder=SentenceTransformerEmbedder(),
            rule_classifier=RuleBasedClassifier(self.repository),
        )

        for file_path in files:
            try:
                payload = self._classify_one(classifier, file_path)
                self.result_queue.put(("result", payload))
            except Exception as error:
                self.result_queue.put(("error", f"{file_path.name}: {error}"))

        self.result_queue.put(("done", None))

    def _classify_one(self, classifier: HybridClassifier, file_path: Path) -> dict[str, object]:
        """Classify one file and return UI payload."""
        raw_text = extract_text_from_file(file_path, fast=True)
        normalized_text = normalize_text(raw_text)
        file_hash = compute_xxhash64(file_path)
        duplicate_of_file_id = self.repository.find_duplicate_file_id(file_hash, str(file_path.resolve()))

        file_id = self.repository.upsert_file(
            file_path=str(file_path.resolve()),
            file_name=file_path.name,
            file_ext=file_path.suffix.lower(),
            file_size=file_path.stat().st_size,
            xxhash64=file_hash,
            duplicate_of_file_id=duplicate_of_file_id,
            extracted_text=normalized_text,
        )
        result = classifier.classify_file(file_id, file_hash, normalized_text, duplicate_of_file_id)
        classification_id = classifier.persist_classification(file_id=file_id, result=result)

        return {
            "file_id": file_id,
            "classification_id": classification_id,
            "file_name": file_path.name,
            "text": normalized_text,
            "result": result,
        }

    def _drain_queue(self) -> None:
        """Move worker results into the Tkinter UI."""
        while not self.result_queue.empty():
            event, payload = self.result_queue.get()
            if event == "result":
                self._insert_result(payload)  # type: ignore[arg-type]
            elif event == "error":
                self._append_detail(f"[읽기 실패] {payload}\n")
            elif event == "done":
                self.status_text.set("분류 완료")
                self.refresh_stats()
                return

        self.after(100, self._drain_queue)

    def _insert_result(self, payload: dict[str, object]) -> None:
        """Insert one classification result into the table."""
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
        )
        self.item_payloads[item_id] = payload

    def on_select_result(self, _event: object | None = None) -> None:
        """Show details for the selected result."""
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
        """Save feedback for the selected row."""
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
        feedback_id = self.repository.save_feedback(
            file_id=int(payload["file_id"]),
            classification_id=int(payload["classification_id"]),
            predicted_category=result.predicted_category,
            final_category=final_category,
            feedback_action=action,
            user_note="gui",
        )
        embedding = result.query_embedding
        if not embedding:
            embedding = SentenceTransformerEmbedder().encode(str(payload["text"]))

        self.repository.save_confirmed_example(
            file_id=int(payload["file_id"]),
            category=final_category,
            source_text=str(payload["text"]),
            embedding=embedding,
            source_feedback_log_id=feedback_id,
        )
        self.refresh_stats()
        self.status_text.set(f"저장 완료: {result.predicted_category} -> {final_category}")

    def _clear_results(self) -> None:
        """Clear table and details."""
        for item_id in self.tree.get_children():
            self.tree.delete(item_id)
        self.item_payloads.clear()
        self._set_detail("")

    def _set_detail(self, text: str) -> None:
        """Replace the detail text content."""
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("1.0", text)
        self.detail_text.configure(state="disabled")

    def _append_detail(self, text: str) -> None:
        """Append text to the detail panel."""
        self.detail_text.configure(state="normal")
        self.detail_text.insert("end", text)
        self.detail_text.configure(state="disabled")


def main() -> None:
    """Start the GUI app."""
    app = ClassifierGui()
    app.mainloop()
