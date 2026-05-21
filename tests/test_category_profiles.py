from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

from src.category_profiles import build_synthetic_training_rows
from src.cli import build_parser
from src.storage import ClassificationRepository
from src.type_classifier import TypeClassifier


class CategoryProfileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.base_dir = Path("tests_runtime") / f"profiles_{uuid4().hex}"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.base_dir / "test.db"
        self.repository = ClassificationRepository(self.db_path)
        self.repository.initialize_database()

    def tearDown(self) -> None:
        rmtree(self.base_dir, ignore_errors=True)

    def test_migration_and_default_seed_profiles(self) -> None:
        inserted = self.repository.seed_default_category_profiles()
        self.assertGreaterEqual(inserted, 5)
        stats = self.repository.get_stats()
        self.assertGreaterEqual(stats["category_profiles_count"], 5)
        second_insert = self.repository.seed_default_category_profiles()
        self.assertEqual(second_insert, 0)

    def test_cli_add_list_and_deactivate_profile(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "add-category-profile",
                "--db",
                str(self.db_path),
                "--type",
                "계약서",
                "--text",
                "제1조 갑 을 계약기간 비밀유지 손해배상",
                "--tags",
                "법률,계약",
            ]
        )
        with redirect_stdout(io.StringIO()) as output:
            args.func(args)
        self.assertIn("category_profile_added", output.getvalue())

        rows = self.repository.list_category_profiles()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["type"], "계약서")

        updated = self.repository.deactivate_category_profile(int(rows[0]["id"]))
        self.assertEqual(updated, 1)
        self.assertEqual(self.repository.list_category_profiles(), [])

    def test_synthetic_rows_are_generated_from_profile(self) -> None:
        rows = build_synthetic_training_rows(
            {
                "id": 1,
                "type": "contract",
                "profile_text": "Contracts include parties, clauses, term, payment, confidentiality.",
                "tags": ["legal"],
                "weight": 0.5,
                "synthetic_count": 3,
            }
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["source"], "category_profile")
        self.assertEqual(rows[0]["label"], "contract")
        self.assertEqual(rows[0]["sample_weight"], 0.5)
        self.assertIn("# PROFILE", rows[0]["body_text"])

    def test_training_source_includes_active_profiles_and_excludes_inactive(self) -> None:
        active_id = self.repository.add_category_profile(
            category_type="contract",
            profile_text="contract party clause payment term",
            tags=["legal"],
        )
        inactive_id = self.repository.add_category_profile(
            category_type="receipt",
            profile_text="receipt total amount card approval",
            tags=["payment"],
        )
        self.repository.deactivate_category_profile(inactive_id)

        rows = self.repository.fetch_type_training_examples()
        profile_rows = [row for row in rows if row.get("source") == "category_profile"]
        self.assertTrue(profile_rows)
        self.assertTrue(all(row.get("source_id") == active_id for row in profile_rows))
        self.assertFalse(any(row["label"] == "receipt" for row in profile_rows))

    def test_training_signature_changes_when_profile_changes(self) -> None:
        before = self.repository.get_category_profile_training_signature()
        profile_id = self.repository.add_category_profile(
            category_type="contract",
            profile_text="contract party clause payment term",
            tags=["legal"],
        )
        after_add = self.repository.get_category_profile_training_signature()
        self.repository.deactivate_category_profile(profile_id)
        after_deactivate = self.repository.get_category_profile_training_signature()

        self.assertNotEqual(before, after_add)
        self.assertNotEqual(after_add, after_deactivate)

    def test_type_classifier_can_train_from_profiles_only(self) -> None:
        self.repository.add_category_profile(
            category_type="contract",
            profile_text="contract party clause payment term confidentiality damages",
            synthetic_count=5,
        )
        self.repository.add_category_profile(
            category_type="receipt",
            profile_text="receipt total amount card approval price payment tax",
            synthetic_count=5,
        )
        rows = self.repository.fetch_type_training_examples()
        prediction = TypeClassifier(min_examples=4).predict(
            training_rows=rows,
            file_name="contract_draft.txt",
            body_text="party clause payment term confidentiality",
            structural_features={},
            fallback_type="unknown",
        )

        self.assertTrue(prediction.available)
        self.assertEqual(prediction.predicted_type, "contract")
        self.assertGreater(prediction.confidence, 0.0)
        self.assertEqual(prediction.evidence["real_training_count"], 0)
        self.assertGreater(prediction.evidence["synthetic_training_count"], 0)

    def test_debug_training_sources_reports_profile_rows(self) -> None:
        self.repository.add_category_profile(
            category_type="contract",
            profile_text="contract party clause payment term",
            synthetic_count=4,
        )
        self.repository.add_category_profile(
            category_type="receipt",
            profile_text="receipt total amount approval price",
            synthetic_count=4,
        )
        parser = build_parser()
        args = parser.parse_args(["debug-training-sources", "--db", str(self.db_path)])

        with redirect_stdout(io.StringIO()) as output:
            args.func(args)

        text = output.getvalue()
        self.assertIn("Training source diagnostics", text)
        self.assertIn("- active_category_profiles: 2", text)
        self.assertIn("- synthetic_rows: 8", text)
        self.assertIn("contract: 4", text)
        self.assertIn("receipt: 4", text)
        self.assertIn("category_profile: 0.500", text)
        self.assertIn("- type_classifier_learnable: yes", text)
        self.assertIn("- training_signature:", text)


if __name__ == "__main__":
    unittest.main()
