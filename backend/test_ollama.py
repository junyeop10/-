"""Ollama LLM 모듈 테스트 (Ollama 미설치 환경에서 mock 사용)."""

import unittest
from unittest.mock import MagicMock, patch

from models.schemas import EvidencePackage
from pipeline import stage3_llm_router
from pipeline.stage3_llm_ollama import classify_with_ollama
from pipeline.stage3_llm_router import classify_with_llm, get_last_llm_backend


def _sample_pkg() -> EvidencePackage:
    return EvidencePackage(
        xxhash="test-hash",
        filename="2024_최종_보고서.docx",
        ext=".docx",
        size_kb=100.0,
        modified_at=0.0,
        text_front="확정된 최종 보고서입니다.",
        text_middle="",
        text_rear="",
        trigger_chunks=[],
        keyword_hits=["최종"],
        pattern_flags={},
        version_hint="최종",
        embedding=[],
        extract_method="docx",
        extract_status="ok",
    )


class TestClassifyWithOllama(unittest.TestCase):
    @patch("pipeline.stage3_llm_ollama.requests.post")
    def test_success_parses_json(self, mock_post: MagicMock) -> None:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "response": (
                '{"category": "보고서", "confidence": 0.9, '
                '"reason": "보고서 키워드", "keywords": ["보고서"]}'
            )
        }
        mock_post.return_value.raise_for_status = MagicMock()

        result = classify_with_ollama(_sample_pkg())
        self.assertEqual(result["category"], "보고서")
        self.assertGreaterEqual(result["confidence"], 0.9)

    @patch("pipeline.stage3_llm_ollama.requests.post")
    def test_timeout_returns_unclassified(self, mock_post: MagicMock) -> None:
        import requests

        mock_post.side_effect = requests.Timeout()
        result = classify_with_ollama(_sample_pkg())
        self.assertEqual(result["category"], "분류불가")
        self.assertEqual(result["reason"], "타임아웃 초과")


class TestLlmRouter(unittest.TestCase):
    def setUp(self) -> None:
        stage3_llm_router.reset_router_cache()

    def tearDown(self) -> None:
        stage3_llm_router.reset_router_cache()

    @patch("pipeline.stage3_llm_router.classify_with_ollama")
    @patch("pipeline.stage3_llm_router.probe_ollama_server", return_value=True)
    def test_router_uses_ollama_when_available(
        self, _probe: MagicMock, mock_ollama: MagicMock
    ) -> None:
        mock_ollama.return_value = {
            "category": "보고서",
            "confidence": 0.85,
            "reason": "mock",
            "keywords": [],
        }
        stage3_llm_router.reset_router_cache()

        result = classify_with_llm(_sample_pkg())
        self.assertEqual(result["category"], "보고서")
        self.assertEqual(get_last_llm_backend(), "ollama")
        mock_ollama.assert_called_once()

    @patch("pipeline.stage3_llm_router.classify_with_claude")
    @patch("pipeline.stage3_llm_router.probe_ollama_server", return_value=False)
    def test_router_uses_claude_when_ollama_down(
        self, _probe: MagicMock, mock_claude: MagicMock
    ) -> None:
        mock_claude.return_value = {
            "category": "최종본",
            "confidence": 0.9,
            "reason": "mock claude",
            "keywords": [],
        }
        stage3_llm_router.reset_router_cache()

        result = classify_with_llm(_sample_pkg())
        self.assertEqual(result["category"], "최종본")
        self.assertEqual(get_last_llm_backend(), "claude")
        mock_claude.assert_called_once()


if __name__ == "__main__":
    unittest.main()
