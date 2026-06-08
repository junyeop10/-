from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import xxhash

from src.hash_utils import compute_file_hash, compute_raw_text_hash, compute_text_hash, compute_xxhash64


class HashUtilsTest(unittest.TestCase):
    def test_file_hash_alias_uses_xxhash64(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.bin"
            payload = b"document-cache-payload"
            path.write_bytes(payload)

            expected = xxhash.xxh64(payload).hexdigest()
            self.assertEqual(compute_xxhash64(path), expected)
            self.assertEqual(compute_file_hash(path), expected)
            self.assertEqual(len(expected), 16)

    def test_text_hash_uses_normalized_xxhash64(self) -> None:
        expected = compute_raw_text_hash("hello world")
        self.assertEqual(compute_text_hash("  hello   world  "), expected)
        self.assertEqual(len(expected), 16)


if __name__ == "__main__":
    unittest.main()
