from __future__ import annotations

import unittest
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

from src.adaptive import rebuild_adaptive_learning
from src.config import AppConfig, FileSystemConfig
from src.operations import (
    commit_move_batch,
    preview_direct_folder_move_plan,
    preview_move_plan,
    preview_move_plan_for_classifications,
    restore_batch,
)
from src.storage import ClassificationRepository


class EnterpriseOperationsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.base_dir = Path("tests_runtime") / f"ops_case_{uuid4().hex}"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir = self.base_dir / "input"
        self.input_dir.mkdir()
        self.managed_root = self.base_dir / "organized"
        self.db_path = self.base_dir / "test.db"
        self.repository = ClassificationRepository(self.db_path)
        self.repository.initialize_database()
        self.config = AppConfig(
            filesystem=FileSystemConfig(
                managed_root=str(self.managed_root),
                preview_only_default=True,
                allow_overwrite=False,
                snapshot_dir=str(self.base_dir / "snapshots"),
                manifest_dir=str(self.base_dir / "manifests"),
            )
        )

    def tearDown(self) -> None:
        if self.base_dir.exists():
            rmtree(self.base_dir, ignore_errors=True)

    def test_preview_commit_and_restore_batch(self) -> None:
        sample_file = self.input_dir / "invoice.txt"
        sample_file.write_text("invoice amount due", encoding="utf-8")
        file_id = self.repository.upsert_file(
            file_path=str(sample_file.resolve()),
            file_name=sample_file.name,
            file_ext=".txt",
            file_size=sample_file.stat().st_size,
            xxhash64="hash-ops-1",
            duplicate_of_file_id=None,
            extracted_text="invoice amount due",
        )
        self.repository.insert_classification(
            file_id=file_id,
            predicted_category="invoice",
            rule_score=0.8,
            embedding_score=0.0,
            llm_score=0.0,
            final_score=0.8,
            candidate_scores_json='{"invoice":0.8}',
            reasoning="test",
            status="suggested",
            large_category="finance",
            middle_category="invoice",
            large_confidence=0.8,
            middle_confidence=0.8,
        )

        plan = preview_move_plan(self.repository, self.config)
        self.assertEqual(len(plan["items"]), 1)
        batch_id = int(plan["batch_id"])

        commit_result = commit_move_batch(self.repository, batch_id=batch_id)
        self.assertEqual(commit_result["moved"], 1)
        moved_path = Path(plan["items"][0]["destination_path"])
        self.assertTrue(moved_path.exists())
        self.assertFalse(sample_file.exists())

        restore_result = restore_batch(self.repository, batch_id=batch_id)
        self.assertEqual(restore_result["restored"], 1)
        self.assertTrue(sample_file.exists())

    def test_preview_can_target_confirmed_classification_ids(self) -> None:
        invoice_file = self.input_dir / "invoice.txt"
        report_file = self.input_dir / "report.txt"
        invoice_file.write_text("invoice amount due", encoding="utf-8")
        report_file.write_text("report summary", encoding="utf-8")
        invoice_file_id = self.repository.upsert_file(
            file_path=str(invoice_file.resolve()),
            file_name=invoice_file.name,
            file_ext=".txt",
            file_size=invoice_file.stat().st_size,
            xxhash64="hash-target-1",
            duplicate_of_file_id=None,
            extracted_text="invoice amount due",
        )
        report_file_id = self.repository.upsert_file(
            file_path=str(report_file.resolve()),
            file_name=report_file.name,
            file_ext=".txt",
            file_size=report_file.stat().st_size,
            xxhash64="hash-target-2",
            duplicate_of_file_id=None,
            extracted_text="report summary",
        )
        invoice_classification_id = self.repository.insert_classification(
            file_id=invoice_file_id,
            predicted_category="invoice",
            rule_score=0.8,
            embedding_score=0.0,
            llm_score=0.0,
            final_score=0.8,
            candidate_scores_json='{"invoice":0.8}',
            reasoning="test",
            status="suggested",
            large_category="finance",
            middle_category="invoice",
            large_confidence=0.8,
            middle_confidence=0.8,
        )
        self.repository.insert_classification(
            file_id=report_file_id,
            predicted_category="report",
            rule_score=0.7,
            embedding_score=0.0,
            llm_score=0.0,
            final_score=0.7,
            candidate_scores_json='{"report":0.7}',
            reasoning="test",
            status="suggested",
            large_category="reporting",
            middle_category="report",
            large_confidence=0.7,
            middle_confidence=0.7,
        )

        plan = preview_move_plan_for_classifications(
            self.repository,
            self.config,
            classification_ids=[invoice_classification_id],
        )

        self.assertEqual(len(plan["items"]), 1)
        self.assertEqual(Path(plan["items"][0]["source_path"]).name, "invoice.txt")

    def test_direct_folder_move_can_be_committed_and_restored(self) -> None:
        sample_file = self.input_dir / "confirmed.txt"
        sample_file.write_text("confirmed document", encoding="utf-8")
        file_id = self.repository.upsert_file(
            file_path=str(sample_file.resolve()),
            file_name=sample_file.name,
            file_ext=".txt",
            file_size=sample_file.stat().st_size,
            xxhash64="hash-direct-1",
            duplicate_of_file_id=None,
            extracted_text="confirmed document",
        )
        classification_id = self.repository.insert_classification(
            file_id=file_id,
            predicted_category="report",
            rule_score=0.8,
            embedding_score=0.0,
            llm_score=0.0,
            final_score=0.8,
            candidate_scores_json='{"report":0.8}',
            reasoning="test",
            status="reviewed",
            large_category="reporting",
            middle_category="report",
            large_confidence=0.8,
            middle_confidence=0.8,
        )
        destination_dir = self.base_dir / "manual-target" / "confirmed-set"

        plan = preview_direct_folder_move_plan(
            self.repository,
            self.config,
            payloads=[
                {
                    "file_id": file_id,
                    "classification_id": classification_id,
                    "file_path": str(sample_file.resolve()),
                    "category": "report",
                    "confidence": 0.8,
                }
            ],
            destination_dir=destination_dir,
        )
        self.assertEqual(len(plan["items"]), 1)
        self.assertEqual(Path(plan["items"][0]["destination_path"]).parent, destination_dir / "report")

        commit_result = commit_move_batch(self.repository, int(plan["batch_id"]))
        self.assertEqual(commit_result["moved"], 1)
        moved_path = destination_dir / "report" / "confirmed.txt"
        self.assertTrue(moved_path.exists())
        self.assertFalse(sample_file.exists())

        restore_result = restore_batch(self.repository, int(plan["batch_id"]))
        self.assertEqual(restore_result["restored"], 1)
        self.assertTrue(sample_file.exists())
        self.assertFalse((destination_dir / "report").exists())
        self.assertFalse(destination_dir.exists())

    def test_direct_folder_copy_restore_deletes_copy_and_keeps_original(self) -> None:
        sample_file = self.input_dir / "copy-source.txt"
        sample_file.write_text("copy me", encoding="utf-8")
        file_id = self.repository.upsert_file(
            file_path=str(sample_file.resolve()),
            file_name=sample_file.name,
            file_ext=".txt",
            file_size=sample_file.stat().st_size,
            xxhash64="hash-copy-1",
            duplicate_of_file_id=None,
            extracted_text="copy me",
        )
        classification_id = self.repository.insert_classification(
            file_id=file_id,
            predicted_category="report",
            rule_score=0.8,
            embedding_score=0.0,
            llm_score=0.0,
            final_score=0.8,
            candidate_scores_json='{"report":0.8}',
            reasoning="test",
            status="reviewed",
            large_category="reporting",
            middle_category="report",
            large_confidence=0.8,
            middle_confidence=0.8,
        )
        destination_dir = self.base_dir / "copy-target"
        plan = preview_direct_folder_move_plan(
            self.repository,
            self.config,
            payloads=[
                {
                    "file_id": file_id,
                    "classification_id": classification_id,
                    "file_path": str(sample_file.resolve()),
                    "category": "report",
                    "confidence": 0.8,
                }
            ],
            destination_dir=destination_dir,
            transfer_mode="copy",
        )

        commit_result = commit_move_batch(self.repository, int(plan["batch_id"]))
        copied_path = destination_dir / "report" / "copy-source.txt"
        self.assertEqual(commit_result["moved"], 1)
        self.assertTrue(sample_file.exists())
        self.assertTrue(copied_path.exists())

        restore_result = restore_batch(self.repository, int(plan["batch_id"]))
        self.assertEqual(restore_result["restored"], 1)
        self.assertTrue(sample_file.exists())
        self.assertFalse(copied_path.exists())
        self.assertFalse(destination_dir.exists())

    def test_move_can_cleanup_empty_source_folder_and_restore_recreates_it(self) -> None:
        source_dir = self.input_dir / "empty-after-move"
        source_dir.mkdir()
        sample_file = source_dir / "only.txt"
        sample_file.write_text("move and cleanup", encoding="utf-8")
        file_id = self.repository.upsert_file(
            file_path=str(sample_file.resolve()),
            file_name=sample_file.name,
            file_ext=".txt",
            file_size=sample_file.stat().st_size,
            xxhash64="hash-cleanup-1",
            duplicate_of_file_id=None,
            extracted_text="move and cleanup",
        )
        classification_id = self.repository.insert_classification(
            file_id=file_id,
            predicted_category="report",
            rule_score=0.8,
            embedding_score=0.0,
            llm_score=0.0,
            final_score=0.8,
            candidate_scores_json='{"report":0.8}',
            reasoning="test",
            status="reviewed",
            large_category="reporting",
            middle_category="report",
            large_confidence=0.8,
            middle_confidence=0.8,
        )
        destination_dir = self.base_dir / "cleanup-target"
        plan = preview_direct_folder_move_plan(
            self.repository,
            self.config,
            payloads=[
                {
                    "file_id": file_id,
                    "classification_id": classification_id,
                    "file_path": str(sample_file.resolve()),
                    "category": "report",
                    "confidence": 0.8,
                }
            ],
            destination_dir=destination_dir,
            transfer_mode="move",
            cleanup_empty_source_dirs=True,
        )

        commit_result = commit_move_batch(self.repository, int(plan["batch_id"]))
        self.assertEqual(commit_result["moved"], 1)
        self.assertFalse(source_dir.exists())

        restore_result = restore_batch(self.repository, int(plan["batch_id"]))
        self.assertEqual(restore_result["restored"], 1)
        self.assertTrue(sample_file.exists())
        self.assertTrue(source_dir.exists())

    def test_feedback_rebuild_creates_adaptive_boosts(self) -> None:
        file_path = self.input_dir / "report.txt"
        file_path.write_text("report analysis summary", encoding="utf-8")
        file_id = self.repository.upsert_file(
            file_path=str(file_path.resolve()),
            file_name=file_path.name,
            file_ext=".txt",
            file_size=file_path.stat().st_size,
            xxhash64="hash-ops-2",
            duplicate_of_file_id=None,
            extracted_text="report analysis summary",
        )
        classification_id = self.repository.insert_classification(
            file_id=file_id,
            predicted_category="contract",
            rule_score=0.4,
            embedding_score=0.0,
            llm_score=0.0,
            final_score=0.4,
            candidate_scores_json='{"contract":0.4}',
            reasoning="test",
            status="suggested",
            large_category="legal",
            middle_category="contract",
            large_confidence=0.4,
            middle_confidence=0.4,
        )
        self.repository.save_feedback(
            file_id=file_id,
            classification_id=classification_id,
            predicted_category="contract",
            final_category="report",
            feedback_action="corrected",
            user_note="wrong original category",
            predicted_hierarchy={"large_category": "legal", "middle_category": "contract"},
            final_hierarchy={"large_category": "reporting", "middle_category": "report"},
            evidence_text="report analysis summary",
            source_scores={"rule": 0.4},
        )

        summary = rebuild_adaptive_learning(self.repository, min_occurrences=1)
        self.assertGreater(summary["adaptive_rules_inserted"], 0)
        boosts = self.repository.fetch_adaptive_rule_boosts("report")
        self.assertTrue(any(row["token"] == "report" for row in boosts))

    def test_preview_groups_duplicate_files_under_representative_folder(self) -> None:
        first_file = self.input_dir / "proposal_v1.txt"
        second_file = self.input_dir / "proposal_final.txt"
        first_file.write_text("same content", encoding="utf-8")
        second_file.write_text("same content", encoding="utf-8")

        first_file_id = self.repository.upsert_file(
            file_path=str(first_file.resolve()),
            file_name=first_file.name,
            file_ext=".txt",
            file_size=first_file.stat().st_size,
            xxhash64="hash-dup-1",
            duplicate_of_file_id=None,
            extracted_text="same content",
        )
        second_file_id = self.repository.upsert_file(
            file_path=str(second_file.resolve()),
            file_name=second_file.name,
            file_ext=".txt",
            file_size=second_file.stat().st_size,
            xxhash64="hash-dup-1",
            duplicate_of_file_id=first_file_id,
            extracted_text="same content",
        )

        for file_id in (first_file_id, second_file_id):
            self.repository.insert_classification(
                file_id=file_id,
                predicted_category="report",
                rule_score=0.8,
                embedding_score=0.0,
                llm_score=0.0,
                final_score=0.8,
                candidate_scores_json='{"report":0.8}',
                reasoning="test",
                status="suggested",
                large_category="reporting",
                middle_category="report",
                large_confidence=0.8,
                middle_confidence=0.8,
            )

        plan = preview_move_plan(self.repository, self.config)
        self.assertEqual(len(plan["items"]), 2)
        destination_paths = [Path(item["destination_path"]) for item in plan["items"]]
        parent_names = {path.parent.name for path in destination_paths}
        self.assertEqual(parent_names, {"proposal_v1"})
        duplicate_group_values = {item["duplicate_group_folder"] for item in plan["items"]}
        self.assertEqual(duplicate_group_values, {"proposal_v1"})

    def test_delete_feedback_log_removes_confirmed_examples(self) -> None:
        file_path = self.input_dir / "feedback.txt"
        file_path.write_text("report analysis summary", encoding="utf-8")
        file_id = self.repository.upsert_file(
            file_path=str(file_path.resolve()),
            file_name=file_path.name,
            file_ext=".txt",
            file_size=file_path.stat().st_size,
            xxhash64="hash-ops-3",
            duplicate_of_file_id=None,
            extracted_text="report analysis summary",
        )
        classification_id = self.repository.insert_classification(
            file_id=file_id,
            predicted_category="report",
            rule_score=0.7,
            embedding_score=0.0,
            llm_score=0.0,
            final_score=0.7,
            candidate_scores_json='{"report":0.7}',
            reasoning="test",
            status="suggested",
            large_category="reporting",
            middle_category="report",
            large_confidence=0.7,
            middle_confidence=0.7,
        )
        feedback_id = self.repository.save_feedback(
            file_id=file_id,
            classification_id=classification_id,
            predicted_category="report",
            final_category="report",
            feedback_action="confirmed",
            user_note="keep",
        )
        self.repository.save_confirmed_example(
            file_id=file_id,
            category="report",
            source_text="report analysis summary",
            embedding=[0.1, 0.2],
            source_feedback_log_id=feedback_id,
        )

        deleted = self.repository.delete_feedback_log(feedback_id)
        self.assertEqual(deleted, 1)
        self.assertEqual(self.repository.fetch_confirmed_examples(), [])


if __name__ == "__main__":
    unittest.main()
