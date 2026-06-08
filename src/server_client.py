"""Thin client for the optional FastAPI classification backend."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin, urlparse, urlunparse

import requests


class RemoteServerError(RuntimeError):
    """Raised when the remote classification server cannot complete a request."""


@dataclass
class RemoteServerClient:
    base_url: str = "http://localhost:8000"
    timeout_seconds: int = 60
    poll_interval_seconds: float = 1.0
    websocket_enabled: bool = True

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/") + "/"

    @property
    def websocket_base_url(self) -> str:
        parsed = urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        return urlunparse((scheme, parsed.netloc, parsed.path, "", "", "")).rstrip("/") + "/"

    def upload_files(self, file_paths: list[Path]) -> dict[str, Any]:
        paths = [Path(path) for path in file_paths]
        if not paths:
            raise RemoteServerError("No files were provided for upload.")
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise RemoteServerError(f"Upload path is not a file: {missing[0]}")

        handles = []
        try:
            files = []
            for path in paths:
                handle = path.open("rb")
                handles.append(handle)
                files.append(("files", (path.name, handle, "application/octet-stream")))
            response = requests.post(
                urljoin(self.base_url, "upload"),
                files=files,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            return _json_response(response)
        except requests.RequestException as error:
            raise RemoteServerError(f"Remote upload failed: {error}") from error
        finally:
            for handle in handles:
                handle.close()

    def get_result(self, job_id: str) -> dict[str, Any]:
        if not job_id.strip():
            raise RemoteServerError("job_id is required.")
        try:
            response = requests.get(
                urljoin(self.base_url, f"result/{job_id.strip()}"),
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            return _json_response(response)
        except requests.RequestException as error:
            raise RemoteServerError(f"Remote result lookup failed: {error}") from error

    def confirm_job(self, job_id: str, corrections: list[dict[str, Any]]) -> dict[str, Any]:
        if not job_id.strip():
            raise RemoteServerError("job_id is required.")
        try:
            response = requests.post(
                urljoin(self.base_url, f"confirm/{job_id.strip()}"),
                json={"corrections": corrections},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            return _json_response(response)
        except requests.RequestException as error:
            raise RemoteServerError(f"Remote confirmation failed: {error}") from error

    def wait_for_result(
        self,
        job_id: str,
        *,
        on_poll: Callable[[dict[str, Any]], None] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
        while True:
            result = self.get_result(job_id)
            if on_poll:
                on_poll(result)
            if result.get("status") in {"completed", "not_found"}:
                return result
            if deadline is not None and time.monotonic() >= deadline:
                raise RemoteServerError(f"Timed out waiting for job {job_id}.")
            time.sleep(max(0.1, float(self.poll_interval_seconds)))

    def listen_progress(
        self,
        job_id: str,
        on_message: Callable[[dict[str, Any]], None],
        *,
        stop_event: threading.Event | None = None,
        idle_timeout_seconds: float | None = None,
    ) -> None:
        if not self.websocket_enabled:
            raise RemoteServerError("Remote WebSocket progress is disabled in config.")
        try:
            import websocket
        except ImportError as error:
            raise RemoteServerError(
                "WebSocket progress requires the 'websocket-client' package."
            ) from error

        ws_url = urljoin(self.websocket_base_url, f"ws/{job_id.strip()}")
        try:
            ws = websocket.create_connection(ws_url, timeout=self.timeout_seconds)
            if idle_timeout_seconds is not None:
                ws.settimeout(float(idle_timeout_seconds))
        except Exception as error:  # websocket-client uses several exception types.
            raise RemoteServerError(f"Remote WebSocket connection failed: {error}") from error

        try:
            while stop_event is None or not stop_event.is_set():
                try:
                    raw_message = ws.recv()
                except Exception as error:
                    raise RemoteServerError(f"Remote WebSocket receive failed: {error}") from error
                try:
                    message = json.loads(raw_message)
                except json.JSONDecodeError:
                    message = {"raw": raw_message}
                on_message(message)
                if message.get("stage") == "folder_complete" and message.get("status") == "completed":
                    return
        finally:
            ws.close()


def build_remote_client(config: Any) -> RemoteServerClient:
    remote = getattr(config, "remote", None)
    return RemoteServerClient(
        base_url=str(getattr(remote, "base_url", "http://localhost:8000")),
        timeout_seconds=int(getattr(remote, "timeout_seconds", 60)),
        poll_interval_seconds=float(getattr(remote, "poll_interval_seconds", 1.0)),
        websocket_enabled=bool(getattr(remote, "websocket_enabled", True)),
    )


def _json_response(response: requests.Response) -> dict[str, Any]:
    try:
        value = response.json()
    except ValueError as error:
        raise RemoteServerError(f"Remote server returned non-JSON response: {response.text[:200]}") from error
    if not isinstance(value, dict):
        raise RemoteServerError("Remote server returned an unexpected JSON shape.")
    return value
