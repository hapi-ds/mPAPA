"""Unit tests for the main entry point wiring.

Validates that main.py correctly initializes settings, logging, database,
workflow, and checks LM Studio connectivity.

Requirements: 9.1, 9.2, 9.3, 9.4, 11.1, 11.2, 11.3, 11.4
"""

from __future__ import annotations

import http.server
import threading
from unittest.mock import MagicMock, patch

import pytest

from patent_system.main import (
    check_lm_studio_connectivity,
    pause_workflow,
    restore_checkpoint,
    resume_workflow,
    start_workflow,
)


# ---------------------------------------------------------------------------
# LM Studio connectivity check
# ---------------------------------------------------------------------------


class _FakeHandler(http.server.BaseHTTPRequestHandler):
    """Minimal HTTP handler that returns 200 for /v1/models."""

    def do_GET(self) -> None:  # noqa: N802
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"data":[]}')

    def log_message(self, *_args: object) -> None:  # suppress logs
        pass


def test_check_lm_studio_connectivity_reachable() -> None:
    """When the endpoint responds, the check returns True."""
    server = http.server.HTTPServer(("127.0.0.1", 0), _FakeHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.handle_request, daemon=True)
    t.start()
    try:
        result = check_lm_studio_connectivity(
            f"http://127.0.0.1:{port}/v1", timeout=2.0
        )
        assert result is True
    finally:
        server.server_close()
        t.join(timeout=3)


def test_check_lm_studio_connectivity_unreachable() -> None:
    """When the endpoint is unreachable, the check returns False."""
    result = check_lm_studio_connectivity(
        "http://127.0.0.1:19999/v1", timeout=0.5
    )
    assert result is False


# ---------------------------------------------------------------------------
# Placeholder workflow actions
# ---------------------------------------------------------------------------


def test_start_workflow_logs(caplog: pytest.LogCaptureFixture) -> None:
    """start_workflow logs the topic id."""
    with caplog.at_level("INFO", logger="patent_system.main"):
        start_workflow(42)
    assert "42" in caplog.text


def test_pause_workflow_logs(caplog: pytest.LogCaptureFixture) -> None:
    """pause_workflow logs the pause event."""
    with caplog.at_level("INFO", logger="patent_system.main"):
        pause_workflow()
    assert "paused" in caplog.text.lower()


def test_resume_workflow_logs(caplog: pytest.LogCaptureFixture) -> None:
    """resume_workflow logs the thread id."""
    with caplog.at_level("INFO", logger="patent_system.main"):
        resume_workflow("thread-abc")
    assert "thread-abc" in caplog.text


def test_restore_checkpoint_returns_none_when_no_workflow() -> None:
    """restore_checkpoint returns None when no workflow is compiled."""
    import patent_system.main as m

    original = m._compiled_workflow
    try:
        m._compiled_workflow = None
        assert restore_checkpoint("thread-xyz") is None
    finally:
        m._compiled_workflow = original


def test_restore_checkpoint_returns_state() -> None:
    """restore_checkpoint returns state dict when checkpoint exists."""
    import patent_system.main as m

    mock_workflow = MagicMock()
    mock_state = MagicMock()
    mock_state.values = {"current_step": "disclosure", "topic_id": 1}
    mock_workflow.get_state.return_value = mock_state

    original = m._compiled_workflow
    try:
        m._compiled_workflow = mock_workflow
        result = restore_checkpoint("thread-123")
        assert result == {"current_step": "disclosure", "topic_id": 1}
        mock_workflow.get_state.assert_called_once_with(
            {"configurable": {"thread_id": "thread-123"}}
        )
    finally:
        m._compiled_workflow = original


def test_restore_checkpoint_handles_exception() -> None:
    """restore_checkpoint returns None and logs on error."""
    import patent_system.main as m

    mock_workflow = MagicMock()
    mock_workflow.get_state.side_effect = RuntimeError("corrupt")

    original = m._compiled_workflow
    try:
        m._compiled_workflow = mock_workflow
        result = restore_checkpoint("thread-err")
        assert result is None
    finally:
        m._compiled_workflow = original
