"""Integration tests for checkpoint save and restore.

Tests the restore_checkpoint function from main.py with mocked
compiled workflows.

Requirements: 9.2, 9.3
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import patent_system.main as main_module
from patent_system.main import restore_checkpoint


@pytest.fixture(autouse=True)
def _reset_compiled_workflow():
    """Ensure _compiled_workflow is reset after each test."""
    original = main_module._compiled_workflow
    yield
    main_module._compiled_workflow = original


class TestRestoreCheckpoint:
    """Test checkpoint save and restore via restore_checkpoint."""

    def test_returns_none_when_no_workflow_compiled(self):
        main_module._compiled_workflow = None
        assert restore_checkpoint("thread-1") is None

    def test_returns_state_dict_when_checkpoint_exists(self):
        mock_workflow = MagicMock()
        mock_state = MagicMock()
        mock_state.values = {
            "topic_id": 42,
            "current_step": "prior_art_search",
            "claims_text": "Claim 1",
            "iteration_count": 1,
        }
        mock_workflow.get_state.return_value = mock_state
        main_module._compiled_workflow = mock_workflow

        result = restore_checkpoint("thread-42")

        assert result is not None
        assert result["topic_id"] == 42
        assert result["current_step"] == "prior_art_search"
        assert result["claims_text"] == "Claim 1"
        mock_workflow.get_state.assert_called_once_with(
            {"configurable": {"thread_id": "thread-42"}}
        )

    def test_returns_none_when_state_is_none(self):
        mock_workflow = MagicMock()
        mock_workflow.get_state.return_value = None
        main_module._compiled_workflow = mock_workflow

        assert restore_checkpoint("thread-empty") is None

    def test_returns_none_when_state_values_empty(self):
        mock_workflow = MagicMock()
        mock_state = MagicMock()
        mock_state.values = {}
        mock_workflow.get_state.return_value = mock_state
        main_module._compiled_workflow = mock_workflow

        assert restore_checkpoint("thread-no-values") is None

    def test_returns_none_on_exception(self):
        mock_workflow = MagicMock()
        mock_workflow.get_state.side_effect = RuntimeError("corrupt checkpoint")
        main_module._compiled_workflow = mock_workflow

        assert restore_checkpoint("thread-err") is None

    def test_passes_correct_thread_id_config(self):
        mock_workflow = MagicMock()
        mock_state = MagicMock()
        mock_state.values = {"current_step": "disclosure"}
        mock_workflow.get_state.return_value = mock_state
        main_module._compiled_workflow = mock_workflow

        restore_checkpoint("my-thread-id")

        mock_workflow.get_state.assert_called_once_with(
            {"configurable": {"thread_id": "my-thread-id"}}
        )

    def test_restored_state_is_plain_dict(self):
        """Ensure the returned value is a plain dict, not a proxy."""
        mock_workflow = MagicMock()
        mock_state = MagicMock()
        mock_state.values = {"topic_id": 7, "current_step": "novelty_analysis"}
        mock_workflow.get_state.return_value = mock_state
        main_module._compiled_workflow = mock_workflow

        result = restore_checkpoint("thread-dict")
        assert isinstance(result, dict)
