from __future__ import annotations

import unittest
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

from src.classifier import HybridClassifier
from src.rule_classifier import RuleBasedClassifier, build_rule_input_text, score_text_with_rules
from src.storage import ClassificationRepository


class RuleClassifierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.base_dir = Path("tests_runtime") / f"rule_case_{uuid4().hex}"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.base_dir / "test.db"
        self.repository = ClassificationRepository(self.db_path)
        self.repository.initialize_database()
        self.repository.seed_rules_from_categories(
            {
                "계약서": ["계약서", "계약", "갑", "을", "계약기간"],
                "공고": ["공고", "모집 공고", "제출서류"],
                "청구서": ["청구서", "세금계산서", "공급가액", "합계금액"],
                "사업자등록증": ["사업자등록증", "사업자등록번호", "개업연월일", "대표자"],
                "법인등기부등본": ["법인등기부등본", "등기부등본", "상호", "본점", "회사성립연월일"],
                "벤처기업인증서": ["벤처기업인증서", "벤처기업", "인증번호"],
                "지방세완납증명서": ["지방세완납증명서", "완납증명서", "지방세"],
                "중소기업확인서": ["중소기업확인서", "중소기업", "확인번호"],
                "재무제표증명": ["표준재무제표증명", "재무상태표", "손익계산서"],
            }
        )

    def tearDown(self) -> None:
        if self.base_dir.exists():
            rmtree(self.base_dir, ignore_errors=True)

    def test_context_rule_matches_contract_document(self) -> None:
        rules = [
            {
                "category": str(rule["category"]),
                "rule_type": str(rule["rule_type"]),
                "pattern": str(rule["pattern"]),
                "weight": float(rule["weight"]),
            }
            for rule in self.repository.fetch_active_rules()
        ]
        result = score_text_with_rules("갑 과 을은 다음 계약기간 동안 본 계약을 이행한다.", rules)
        self.assertGreaterEqual(result["scores"]["계약서"], 4.0)
        self.assertTrue(any("계약기간" in match or "문맥:" in match for match in result["matches"]["계약서"]))

    def test_context_rule_matches_invoice_category(self) -> None:
        rules = [
            {
                "category": str(rule["category"]),
                "rule_type": str(rule["rule_type"]),
                "pattern": str(rule["pattern"]),
                "weight": float(rule["weight"]),
            }
            for rule in self.repository.fetch_active_rules()
        ]
        result = score_text_with_rules("전자세금계산서 공급가액 및 합계금액을 확인 바랍니다.", rules)
        self.assertGreaterEqual(result["scores"]["청구서"], 4.0)

    def test_filename_hint_can_classify_business_registration_without_text(self) -> None:
        classifier = HybridClassifier(
            repository=self.repository,
            embedder=None,  # type: ignore[arg-type]
            rule_classifier=RuleBasedClassifier(self.repository),
            use_embedding_for_no_rule=False,
        )
        result = classifier.classify_file(
            file_id=1,
            file_hash="hash-1",
            text="",
            duplicate_of_file_id=None,
            file_name="사업자등록증_(주)에네이.pdf",
        )
        self.assertEqual(result.predicted_category, "사업자등록증")
        self.assertEqual(result.ml_evidence.get("status"), "disabled")
        self.assertEqual(result.predicted_type, "")
        self.assertEqual(result.type_confidence, 0.0)

    def test_ml_disabled_does_not_call_type_classifier(self) -> None:
        class FailingTypeClassifier:
            version = "test"

            def predict(self, **_: object) -> object:
                raise AssertionError("TypeClassifier should not be called when ML is disabled")

        classifier = HybridClassifier(
            repository=self.repository,
            embedder=None,  # type: ignore[arg-type]
            rule_classifier=RuleBasedClassifier(self.repository),
            use_embedding_for_no_rule=False,
            type_classifier=FailingTypeClassifier(),  # type: ignore[arg-type]
            ml_enabled=False,
        )

        result = classifier.classify_file(
            file_id=1,
            file_hash="hash-ml-off",
            text="계약서 갑 을 계약기간",
            duplicate_of_file_id=None,
            file_name="계약서.pdf",
        )

        self.assertEqual(result.ml_evidence.get("reason"), "ml_disabled_by_config")

    def test_filename_hint_can_classify_registry_without_text(self) -> None:
        rules = [
            {
                "category": str(rule["category"]),
                "rule_type": str(rule["rule_type"]),
                "pattern": str(rule["pattern"]),
                "weight": float(rule["weight"]),
            }
            for rule in self.repository.fetch_active_rules()
        ]
        result = score_text_with_rules(build_rule_input_text("", "14. 법인등기부등본_(주)커넥트스토리.pdf"), rules)
        self.assertGreater(result["scores"]["법인등기부등본"], 0.0)

    def test_filename_hint_can_classify_certificate_documents_without_text(self) -> None:
        rules = [
            {
                "category": str(rule["category"]),
                "rule_type": str(rule["rule_type"]),
                "pattern": str(rule["pattern"]),
                "weight": float(rule["weight"]),
            }
            for rule in self.repository.fetch_active_rules()
        ]
        venture_result = score_text_with_rules(
            build_rule_input_text("", "8. 벤처기업인증서_(주)커넥트스토리.pdf"),
            rules,
        )
        tax_result = score_text_with_rules(
            build_rule_input_text("", "지방세완납증명서(6.10).pdf"),
            rules,
        )

        self.assertGreater(venture_result["scores"]["벤처기업인증서"], 0.0)
        self.assertGreater(tax_result["scores"]["지방세완납증명서"], 0.0)


if __name__ == "__main__":
    unittest.main()
