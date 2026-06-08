"""피드백 루프가 실제로 반영되는지 검증하는 테스트입니다."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

from src.classifier import HybridClassifier
from src.rule_classifier import RuleBasedClassifier
from src.storage import ClassificationRepository


class FakeEmbedder:
    """테스트용 고정 임베딩기입니다."""

    def encode(self, text: str) -> list[float]:
        """텍스트 키워드에 따라 단순 벡터를 반환합니다."""
        contract_count = sum(keyword in text for keyword in ("contract", "party", "payment", "term"))
        report_count = sum(keyword in text for keyword in ("report", "analysis", "result", "summary"))
        return [float(contract_count), float(report_count), 1.0]

    def score_against_examples(self, query_embedding, examples, categories):
        """저장된 예시와 코사인 유사도를 비교합니다."""
        from src.vectorizer import cosine_similarity

        scores = {category: 0.0 for category in categories}
        top_examples = {}

        for example in examples:
            embedding = json.loads(example["embedding_json"])
            similarity = cosine_similarity(query_embedding, embedding)
            category = example["category"]
            scores[category] = max(scores.get(category, 0.0), similarity)
            if category not in top_examples or similarity > top_examples[category]["similarity"]:
                top_examples[category] = {
                    "category": category,
                    "similarity": similarity,
                    "file_name": example["file_name"],
                }

        return {"scores": scores, "top_examples": top_examples}


class FeedbackLoopTest(unittest.TestCase):
    """사용자 수정이 다음 분류 결과에 반영되는지 확인합니다."""

    def setUp(self) -> None:
        """임시 DB와 기본 규칙을 준비합니다."""
        self.base_dir = Path("tests_runtime") / f"case_{uuid4().hex}"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.base_dir / "test.db"

        self.repository = ClassificationRepository(self.db_path)
        self.repository.initialize_database()
        self.repository.seed_rules_from_categories(
            {
                "contract": ["contract", "party", "payment", "term"],
                "report": ["report", "analysis", "result", "summary"],
            }
        )
        self.classifier = HybridClassifier(
            repository=self.repository,
            embedder=FakeEmbedder(),
            rule_classifier=RuleBasedClassifier(self.repository),
        )

    def tearDown(self) -> None:
        """임시 자원을 정리합니다."""
        if self.base_dir.exists():
            rmtree(self.base_dir, ignore_errors=True)

    def test_feedback_and_confirmed_examples_are_saved(self) -> None:
        """수정 시 feedback_logs와 confirmed_examples가 저장되는지 확인합니다."""
        file_id = self.repository.upsert_file(
            file_path="first.txt",
            file_name="first.txt",
            file_ext=".txt",
            file_size=10,
            xxhash64="hash-1",
            duplicate_of_file_id=None,
            extracted_text="contract draft and report analysis result",
        )
        result = self.classifier.classify_file(file_id, "hash-1", "contract draft and report analysis result", None)
        classification_id = self.classifier.persist_classification(file_id, result)

        feedback_log_id = self.repository.save_feedback(
            file_id=file_id,
            classification_id=classification_id,
            predicted_category=result.predicted_category,
            final_category="report",
            feedback_action="corrected",
            user_note="report is correct",
        )
        self.repository.save_confirmed_example(
            file_id=file_id,
            category="report",
            source_text="contract draft and report analysis result",
            embedding=result.query_embedding or self.classifier.embedder.encode("contract draft and report analysis result"),
            source_feedback_log_id=feedback_log_id,
        )

        stats = self.repository.get_stats()
        self.assertEqual(stats["feedback_logs_count"], 1)
        self.assertEqual(stats["confirmed_examples_count"], 1)

    def test_confirmed_example_is_not_duplicated_for_same_file_and_category(self) -> None:
        text = "report analysis result"
        file_id = self.repository.upsert_file(
            file_path="dedupe.txt",
            file_name="dedupe.txt",
            file_ext=".txt",
            file_size=10,
            xxhash64="hash-dedupe",
            duplicate_of_file_id=None,
            extracted_text=text,
        )
        result = self.classifier.classify_file(file_id, "hash-dedupe", text, None)
        classification_id = self.classifier.persist_classification(file_id, result)
        first_feedback_id = self.repository.save_feedback(
            file_id=file_id,
            classification_id=classification_id,
            predicted_category=result.predicted_category,
            final_category="report",
            feedback_action="confirmed",
            user_note="first confirm",
        )
        second_feedback_id = self.repository.save_feedback(
            file_id=file_id,
            classification_id=classification_id,
            predicted_category=result.predicted_category,
            final_category="report",
            feedback_action="confirmed",
            user_note="second confirm",
        )

        first_example_id = self.repository.save_confirmed_example(
            file_id=file_id,
            category="report",
            source_text=text,
            embedding=result.query_embedding or self.classifier.embedder.encode(text),
            source_feedback_log_id=first_feedback_id,
        )
        second_example_id = self.repository.save_confirmed_example(
            file_id=file_id,
            category="report",
            source_text=text,
            embedding=result.query_embedding or self.classifier.embedder.encode(text),
            source_feedback_log_id=second_feedback_id,
        )

        stats = self.repository.get_stats()
        self.assertEqual(first_example_id, second_example_id)
        self.assertEqual(stats["feedback_logs_count"], 2)
        self.assertEqual(stats["confirmed_examples_count"], 1)

    def test_feedback_loop_affects_next_classification(self) -> None:
        """수정된 예시가 다음 분류에서 embedding_score와 feedback_score에 반영되는지 확인합니다."""
        first_text = "contract party report"
        first_file_id = self.repository.upsert_file(
            file_path="first.txt",
            file_name="first.txt",
            file_ext=".txt",
            file_size=10,
            xxhash64="hash-1",
            duplicate_of_file_id=None,
            extracted_text=first_text,
        )
        first_result = self.classifier.classify_file(first_file_id, "hash-1", first_text, None)
        first_classification_id = self.classifier.persist_classification(first_file_id, first_result)
        first_feedback_id = self.repository.save_feedback(
            file_id=first_file_id,
            classification_id=first_classification_id,
            predicted_category=first_result.predicted_category,
            final_category="report",
            feedback_action="corrected",
            user_note="report is correct",
        )
        self.repository.save_confirmed_example(
            file_id=first_file_id,
            category="report",
            source_text=first_text,
            embedding=first_result.query_embedding or self.classifier.embedder.encode(first_text),
            source_feedback_log_id=first_feedback_id,
        )

        second_text = "contract report"
        second_file_id = self.repository.upsert_file(
            file_path="second.txt",
            file_name="second.txt",
            file_ext=".txt",
            file_size=10,
            xxhash64="hash-2",
            duplicate_of_file_id=None,
            extracted_text=second_text,
        )
        second_result = self.classifier.classify_file(second_file_id, "hash-2", second_text, None)

        self.assertGreater(second_result.embedding_score, 0.0)
        self.assertGreater(second_result.feedback_score, 0.0)
        self.assertEqual(second_result.predicted_category, "report")

        log_buffer = io.StringIO()
        with redirect_stdout(log_buffer):
            print(
                "rule_score={:.3f}, embedding_score={:.3f}, feedback_score={:.3f}, final_score={:.3f}".format(
                    second_result.rule_score,
                    second_result.embedding_score,
                    second_result.feedback_score,
                    second_result.final_score,
                )
            )
        log_output = log_buffer.getvalue()

        self.assertIn("rule_score=", log_output)
        self.assertIn("embedding_score=", log_output)
        self.assertIn("feedback_score=", log_output)
        self.assertIn("final_score=", log_output)

    def test_confirmation_batch_can_be_listed_and_deleted(self) -> None:
        batch_id = "selected-test-batch"
        for index in range(2):
            text = f"report analysis summary {index}"
            file_id = self.repository.upsert_file(
                file_path=f"batch-{index}.txt",
                file_name=f"batch-{index}.txt",
                file_ext=".txt",
                file_size=10,
                xxhash64=f"hash-batch-{index}",
                duplicate_of_file_id=None,
                extracted_text=text,
            )
            result = self.classifier.classify_file(file_id, f"hash-batch-{index}", text, None)
            classification_id = self.classifier.persist_classification(file_id, result)
            feedback_id = self.repository.save_feedback(
                file_id=file_id,
                classification_id=classification_id,
                predicted_category=result.predicted_category,
                final_category="report",
                feedback_action="confirmed",
                user_note="batch confirm",
                confirmation_batch_id=batch_id,
            )
            self.repository.save_confirmed_example(
                file_id=file_id,
                category="report",
                source_text=text,
                embedding=result.query_embedding or self.classifier.embedder.encode(text),
                source_feedback_log_id=feedback_id,
            )

        batches = [dict(row) for row in self.repository.list_confirmation_batches()]
        matching = [row for row in batches if row["confirmation_batch_id"] == batch_id]

        self.assertEqual(len(matching), 1)
        self.assertEqual(int(matching[0]["file_count"]), 2)
        deleted = self.repository.delete_confirmation_batch(batch_id)
        self.assertEqual(deleted, 2)
        stats = self.repository.get_stats()
        self.assertEqual(stats["feedback_logs_count"], 0)
        self.assertEqual(stats["confirmed_examples_count"], 0)

    def test_confirmation_batch_name_can_be_updated(self) -> None:
        batch_id = "rename-test-batch"
        for index in range(2):
            text = f"report rename summary {index}"
            file_id = self.repository.upsert_file(
                file_path=f"rename-{index}.txt",
                file_name=f"rename-{index}.txt",
                file_ext=".txt",
                file_size=10,
                xxhash64=f"hash-rename-{index}",
                duplicate_of_file_id=None,
                extracted_text=text,
            )
            result = self.classifier.classify_file(file_id, f"hash-rename-{index}", text, None)
            classification_id = self.classifier.persist_classification(file_id, result)
            self.repository.save_feedback(
                file_id=file_id,
                classification_id=classification_id,
                predicted_category=result.predicted_category,
                final_category="report",
                feedback_action="confirmed",
                user_note="rename confirm",
                confirmation_batch_id=batch_id,
                confirmation_batch_name="old name",
            )

        updated = self.repository.update_confirmation_batch_name(batch_id, "발표용 검증 실행")
        batches = [dict(row) for row in self.repository.list_confirmation_batches()]
        matching = [row for row in batches if row["confirmation_batch_id"] == batch_id]

        self.assertEqual(updated, 2)
        self.assertEqual(matching[0]["confirmation_batch_name"], "발표용 검증 실행")


if __name__ == "__main__":
    unittest.main()
