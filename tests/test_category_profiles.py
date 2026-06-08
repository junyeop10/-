from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

from src.category_profiles import DEFAULT_CATEGORY_PROFILES, build_synthetic_training_rows
from src.cli import build_parser
from src.document_patterns import get_pattern_for_type
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
        self.assertGreaterEqual(inserted, 8)
        stats = self.repository.get_stats()
        self.assertGreaterEqual(stats["category_profiles_count"], 8)
        second_insert = self.repository.seed_default_category_profiles()
        self.assertEqual(second_insert, 0)
        rows = self.repository.list_category_profiles()
        self.assertTrue(all(row["profile_signals_json"] != "{}" for row in rows))

    def test_seed_backfill_does_not_overwrite_user_profile_text(self) -> None:
        profile_id = self.repository.add_category_profile(
            category_type="계약서",
            profile_text="사용자가 직접 작성한 계약서 설명",
            tags=["custom"],
        )

        inserted = self.repository.seed_default_category_profiles()
        rows = [row for row in self.repository.list_category_profiles(include_inactive=True) if int(row["id"]) == profile_id]

        self.assertEqual(inserted, len(DEFAULT_CATEGORY_PROFILES) - 1)
        self.assertEqual(rows[0]["profile_text"], "사용자가 직접 작성한 계약서 설명")
        self.assertEqual(rows[0]["profile_signals_json"], "{}")
        self.assertEqual(rows[0]["profile_origin"], "user")

    def test_backfill_signals_preserves_user_profile_text_and_origin(self) -> None:
        profile_id = self.repository.add_category_profile(
            category_type="계약서",
            profile_text="사용자가 직접 수정한 계약서 설명",
            tags=["custom"],
        )

        changed = self.repository.backfill_category_profile_signals()
        rows = [row for row in self.repository.list_category_profiles(include_inactive=True) if int(row["id"]) == profile_id]

        self.assertEqual([item["id"] for item in changed], [profile_id])
        self.assertEqual(rows[0]["profile_text"], "사용자가 직접 수정한 계약서 설명")
        self.assertEqual(rows[0]["profile_origin"], "user")
        self.assertNotEqual(rows[0]["profile_signals_json"], "{}")

    def test_backfill_signals_only_when_empty_and_does_not_overwrite_existing_signals(self) -> None:
        contract_pattern = get_pattern_for_type("계약서")
        self.assertIsNotNone(contract_pattern)
        empty_id = self.repository.add_category_profile(
            category_type="계약서",
            profile_text="empty signals",
        )
        existing_id = self.repository.add_category_profile(
            category_type="영수증",
            profile_text="already has signals",
            profile_signals={"semantic_signals": ["CUSTOM_ONLY"]},
        )

        changed = self.repository.backfill_category_profile_signals()
        rows = {
            int(row["id"]): row
            for row in self.repository.list_category_profiles(include_inactive=True)
            if int(row["id"]) in {empty_id, existing_id}
        }

        self.assertEqual([item["id"] for item in changed], [empty_id])
        self.assertIn("semantic_signals", rows[empty_id]["profile_signals_json"])
        self.assertIn("CUSTOM_ONLY", rows[existing_id]["profile_signals_json"])
        self.assertNotIn("승인번호", rows[existing_id]["profile_signals_json"])

    def test_cli_backfill_signals_dry_run_does_not_change_db(self) -> None:
        profile_id = self.repository.add_category_profile(
            category_type="계약서",
            profile_text="dry run profile",
        )
        parser = build_parser()
        args = parser.parse_args(["backfill-category-profile-signals", "--db", str(self.db_path), "--dry-run"])

        with redirect_stdout(io.StringIO()) as output:
            args.func(args)
        rows = [row for row in self.repository.list_category_profiles(include_inactive=True) if int(row["id"]) == profile_id]

        self.assertIn("category_profile_signals_would_backfill: 1", output.getvalue())
        self.assertEqual(rows[0]["profile_signals_json"], "{}")

    def test_cli_backfill_signals_updates_matching_empty_profiles(self) -> None:
        profile_id = self.repository.add_category_profile(
            category_type="계약서",
            profile_text="real run profile",
        )
        parser = build_parser()
        args = parser.parse_args(["backfill-category-profile-signals", "--db", str(self.db_path)])

        with redirect_stdout(io.StringIO()) as output:
            args.func(args)
        rows = [row for row in self.repository.list_category_profiles(include_inactive=True) if int(row["id"]) == profile_id]

        self.assertIn("category_profile_signals_backfilled: 1", output.getvalue())
        self.assertNotEqual(rows[0]["profile_signals_json"], "{}")

    def test_expand_training_data_updates_count_without_overwriting_text_or_origin(self) -> None:
        profile_id = self.repository.add_category_profile(
            category_type="계약서",
            profile_text="custom contract profile",
            synthetic_count=5,
        )

        changed = self.repository.expand_category_profile_training_data(synthetic_count=12)
        rows = [row for row in self.repository.list_category_profiles(include_inactive=True) if int(row["id"]) == profile_id]

        self.assertEqual([item["id"] for item in changed], [profile_id])
        self.assertEqual(rows[0]["profile_text"], "custom contract profile")
        self.assertEqual(rows[0]["profile_origin"], "user")
        self.assertEqual(int(rows[0]["synthetic_count"]), 12)

    def test_expand_training_data_dry_run_does_not_change_count(self) -> None:
        profile_id = self.repository.add_category_profile(
            category_type="계약서",
            profile_text="dry run expand",
            synthetic_count=5,
        )

        changed = self.repository.expand_category_profile_training_data(synthetic_count=12, dry_run=True)
        rows = [row for row in self.repository.list_category_profiles(include_inactive=True) if int(row["id"]) == profile_id]

        self.assertEqual([item["id"] for item in changed], [profile_id])
        self.assertEqual(int(rows[0]["synthetic_count"]), 5)

    def test_synthetic_rows_are_varied_by_focus(self) -> None:
        profile = get_pattern_for_type("계약서")
        assert profile is not None
        rows = build_synthetic_training_rows(
            {
                "id": 1,
                "type": "계약서",
                "profile_text": profile["profile_text"],
                "profile_signals": profile["profile_signals"],
                "synthetic_count": 8,
            }
        )
        focus_lines = {
            line
            for row in rows
            for line in row["body_text"].splitlines()
            if line in {"semantic", "layout", "ocr", "numeric", "structure", "business", "mixed"}
        }

        self.assertEqual(len(rows), 8)
        self.assertGreaterEqual(len(focus_lines), 5)

    def test_cli_expand_training_data_updates_matching_profiles(self) -> None:
        profile_id = self.repository.add_category_profile(
            category_type="계약서",
            profile_text="cli expand",
            synthetic_count=5,
        )
        parser = build_parser()
        args = parser.parse_args(["expand-category-profile-training", "--db", str(self.db_path), "--synthetic-count", "10"])

        with redirect_stdout(io.StringIO()) as output:
            args.func(args)
        rows = [row for row in self.repository.list_category_profiles(include_inactive=True) if int(row["id"]) == profile_id]

        self.assertIn("category_profile_training_expanded: 1", output.getvalue())
        self.assertEqual(int(rows[0]["synthetic_count"]), 10)

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
        self.assertIn("# SEMANTIC_SIGNALS", rows[0]["body_text"])

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
