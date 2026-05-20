from __future__ import annotations

import unittest

from src.gui import (
    EMBEDDING_STATE_FAILED,
    EMBEDDING_STATE_LOADING,
    EMBEDDING_STATE_NOT_STARTED,
    EMBEDDING_STATE_READY,
    classification_allowed_for_embedding_state,
    classification_block_reason_for_state,
    embedding_state_status_text,
)


class GuiEmbeddingStateTest(unittest.TestCase):
    def test_classification_blocked_while_embedding_loading(self) -> None:
        self.assertFalse(classification_allowed_for_embedding_state(EMBEDDING_STATE_NOT_STARTED))
        self.assertFalse(classification_allowed_for_embedding_state(EMBEDDING_STATE_LOADING))
        self.assertIn("still loading", classification_block_reason_for_state(EMBEDDING_STATE_LOADING))

    def test_classification_enabled_when_embedding_ready(self) -> None:
        self.assertTrue(classification_allowed_for_embedding_state(EMBEDDING_STATE_READY))
        self.assertEqual(classification_block_reason_for_state(EMBEDDING_STATE_READY), "")

    def test_failure_state_shows_visible_message(self) -> None:
        message = embedding_state_status_text(EMBEDDING_STATE_FAILED, "mock failure")
        self.assertIn("failed", message)
        self.assertIn("mock failure", message)
        self.assertFalse(classification_allowed_for_embedding_state(EMBEDDING_STATE_FAILED))


if __name__ == "__main__":
    unittest.main()
