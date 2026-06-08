from __future__ import annotations

import unittest
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.config import default_config
from src.server_client import RemoteServerClient, build_remote_client


class RemoteServerClientTest(unittest.TestCase):
    def test_build_remote_client_uses_config_defaults(self) -> None:
        config = default_config()
        config.remote.base_url = "http://example.test:9000"
        config.remote.timeout_seconds = 12
        config.remote.poll_interval_seconds = 0.5
        config.remote.websocket_enabled = False

        client = build_remote_client(config)

        self.assertEqual(client.base_url, "http://example.test:9000/")
        self.assertEqual(client.websocket_base_url, "ws://example.test:9000/")
        self.assertEqual(client.timeout_seconds, 12)
        self.assertEqual(client.poll_interval_seconds, 0.5)
        self.assertFalse(client.websocket_enabled)

    @patch("src.server_client.requests.post")
    def test_upload_files_posts_multipart_files_field(self, post: Mock) -> None:
        response = Mock()
        response.json.return_value = {"job_id": "job-1", "file_count": 1}
        response.raise_for_status.return_value = None
        post.return_value = response
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "한글.pdf"
            path.write_bytes(b"pdf")

            result = RemoteServerClient("http://localhost:8000").upload_files([path])

        self.assertEqual(result["job_id"], "job-1")
        self.assertEqual(post.call_args.kwargs["files"][0][0], "files")
        self.assertEqual(post.call_args.kwargs["files"][0][1][0], "한글.pdf")

    @patch("src.server_client.requests.get")
    def test_get_result_fetches_job_result(self, get: Mock) -> None:
        response = Mock()
        response.json.return_value = {"status": "completed", "results": []}
        response.raise_for_status.return_value = None
        get.return_value = response

        result = RemoteServerClient("http://localhost:8000").get_result("abc")

        self.assertEqual(result["status"], "completed")
        self.assertEqual(get.call_args.args[0], "http://localhost:8000/result/abc")

    @patch("src.server_client.requests.post")
    def test_confirm_job_posts_corrections(self, post: Mock) -> None:
        response = Mock()
        response.json.return_value = {"saved": 1}
        response.raise_for_status.return_value = None
        post.return_value = response
        correction = {"filename": "a.pdf", "user_category": "보고서", "folder_description": ""}

        result = RemoteServerClient("http://localhost:8000").confirm_job("abc", [correction])

        self.assertEqual(result["saved"], 1)
        self.assertEqual(post.call_args.args[0], "http://localhost:8000/confirm/abc")
        self.assertEqual(post.call_args.kwargs["json"], {"corrections": [correction]})

    def test_listen_progress_receives_websocket_messages(self) -> None:
        messages = [
            '{"stage":"text_extract","progress":"1/1","current_file":"a.pdf","status":"running"}',
            '{"stage":"folder_complete","progress":"1/1","current_file":"","status":"completed"}',
        ]
        received: list[dict[str, object]] = []

        class FakeWebSocket:
            def __init__(self) -> None:
                self.closed = False
                self.timeout = None

            def recv(self) -> str:
                return messages.pop(0)

            def settimeout(self, value: float) -> None:
                self.timeout = value

            def close(self) -> None:
                self.closed = True

        fake_ws = FakeWebSocket()
        fake_module = SimpleNamespace(create_connection=Mock(return_value=fake_ws))
        with patch.dict(sys.modules, {"websocket": fake_module}):
            RemoteServerClient("http://localhost:8000").listen_progress("abc", received.append, idle_timeout_seconds=5)

        fake_module.create_connection.assert_called_once_with("ws://localhost:8000/ws/abc", timeout=60)
        self.assertEqual(fake_ws.timeout, 5.0)
        self.assertEqual(received[-1]["stage"], "folder_complete")
        self.assertTrue(fake_ws.closed)

    def test_wait_for_result_polls_until_completed(self) -> None:
        client = RemoteServerClient("http://localhost:8000", poll_interval_seconds=0.1)
        with patch.object(
            client,
            "get_result",
            side_effect=[
                {"status": "processing", "results": []},
                {"status": "completed", "results": [{"filename": "a.pdf"}]},
            ],
        ) as get_result:
            with patch("src.server_client.time.sleep") as sleep:
                result = client.wait_for_result("abc", timeout_seconds=5)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(get_result.call_count, 2)
        sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()
