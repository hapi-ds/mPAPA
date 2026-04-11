"""Unit tests for structured logging setup."""

import json
import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from patent_system.logging_config import (
    StructuredFormatter,
    log_agent_invocation,
    log_db_error,
    log_external_request,
    log_llm_call,
    setup_logging,
)


@pytest.fixture
def json_logger(tmp_path: Path) -> tuple[logging.Logger, Path]:
    """Create a logger with StructuredFormatter writing to a temp file."""
    log_file = tmp_path / "test.log"
    logger = logging.getLogger(f"test_logging_{id(log_file)}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    handler = logging.FileHandler(str(log_file), encoding="utf-8")
    handler.setFormatter(StructuredFormatter())
    logger.addHandler(handler)
    return logger, log_file


def _read_last_entry(log_file: Path) -> dict:
    """Read and parse the last JSON log line from a file."""
    lines = log_file.read_text().strip().splitlines()
    return json.loads(lines[-1])


class TestStructuredFormatter:
    """Verify the JSON formatter produces valid structured entries."""

    def test_basic_entry_has_required_fields(
        self, json_logger: tuple[logging.Logger, Path]
    ) -> None:
        logger, log_file = json_logger
        logger.info("hello world")
        entry = _read_last_entry(log_file)
        assert "timestamp" in entry
        assert entry["level"] == "INFO"
        assert "module" in entry
        assert entry["message"] == "hello world"

    def test_timestamp_is_valid_iso(
        self, json_logger: tuple[logging.Logger, Path]
    ) -> None:
        logger, log_file = json_logger
        logger.warning("check ts")
        entry = _read_last_entry(log_file)
        # Should parse without error
        datetime.fromisoformat(entry["timestamp"])

    def test_extra_fields_merged(
        self, json_logger: tuple[logging.Logger, Path]
    ) -> None:
        logger, log_file = json_logger
        logger.info("with extras", extra={"extra_fields": {"foo": "bar"}})
        entry = _read_last_entry(log_file)
        assert entry["foo"] == "bar"

    def test_each_line_is_valid_json(
        self, json_logger: tuple[logging.Logger, Path]
    ) -> None:
        logger, log_file = json_logger
        logger.info("line one")
        logger.error("line two")
        for line in log_file.read_text().strip().splitlines():
            json.loads(line)  # Should not raise


class TestSetupLogging:
    """Verify setup_logging configures the root logger correctly."""

    def test_creates_log_directory(self, tmp_path: Path) -> None:
        log_file = tmp_path / "subdir" / "app.log"
        settings = MagicMock(log_file_path=log_file, log_level="INFO")
        setup_logging(settings)
        assert log_file.parent.exists()
        # Cleanup: remove handler we just added to root
        logging.getLogger().handlers = [
            h
            for h in logging.getLogger().handlers
            if not (isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file))
        ]

    def test_respects_log_level(self, tmp_path: Path) -> None:
        log_file = tmp_path / "level.log"
        settings = MagicMock(log_file_path=log_file, log_level="WARNING")
        setup_logging(settings)
        root = logging.getLogger()
        assert root.level == logging.WARNING
        # Cleanup
        root.handlers = [
            h
            for h in root.handlers
            if not (isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file))
        ]


class TestLogAgentInvocation:
    """Verify agent invocation helper produces correct fields."""

    def test_fields_present(
        self, json_logger: tuple[logging.Logger, Path]
    ) -> None:
        logger, log_file = json_logger
        log_agent_invocation(logger, "DisclosureAgent", "input_s", "output_s", 123.4)
        entry = _read_last_entry(log_file)
        assert entry["event_type"] == "agent_invocation"
        assert entry["name"] == "DisclosureAgent"
        assert entry["input"] == "input_s"
        assert entry["output"] == "output_s"
        assert entry["duration"] == 123.4


class TestLogExternalRequest:
    """Verify external request helper produces correct fields."""

    def test_fields_present(
        self, json_logger: tuple[logging.Logger, Path]
    ) -> None:
        logger, log_file = json_logger
        log_external_request(logger, "DEPATISnet", "query_x", "200", 456.7)
        entry = _read_last_entry(log_file)
        assert entry["event_type"] == "external_request"
        assert entry["source"] == "DEPATISnet"
        assert entry["query"] == "query_x"
        assert entry["status"] == "200"
        assert entry["response_time"] == 456.7


class TestLogDbError:
    """Verify DB error helper produces correct fields."""

    def test_fields_present(
        self, json_logger: tuple[logging.Logger, Path]
    ) -> None:
        logger, log_file = json_logger
        log_db_error(logger, "INSERT", "patents", "UNIQUE constraint failed")
        entry = _read_last_entry(log_file)
        assert entry["event_type"] == "db_error"
        assert entry["operation"] == "INSERT"
        assert entry["table"] == "patents"
        assert entry["error"] == "UNIQUE constraint failed"


class TestLogLlmCall:
    """Verify LLM call helper produces correct fields."""

    def test_basic_fields(
        self, json_logger: tuple[logging.Logger, Path]
    ) -> None:
        logger, log_file = json_logger
        log_llm_call(logger, "llama3", 100, 200, 1500.0)
        entry = _read_last_entry(log_file)
        assert entry["event_type"] == "llm_call"
        assert entry["model"] == "llama3"
        assert entry["prompt_tokens"] == 100
        assert entry["response_tokens"] == 200
        assert entry["latency"] == 1500.0

    def test_debug_includes_full_text(
        self, json_logger: tuple[logging.Logger, Path]
    ) -> None:
        logger, log_file = json_logger
        logger.setLevel(logging.DEBUG)
        log_llm_call(
            logger, "llama3", 10, 20, 500.0,
            prompt_text="Tell me about patents",
            response_text="Patents are...",
        )
        entry = _read_last_entry(log_file)
        assert entry["prompt_text"] == "Tell me about patents"
        assert entry["response_text"] == "Patents are..."

    def test_info_excludes_full_text(self, tmp_path: Path) -> None:
        log_file = tmp_path / "info_only.log"
        logger = logging.getLogger(f"test_info_{id(log_file)}")
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(str(log_file), encoding="utf-8")
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)

        log_llm_call(
            logger, "llama3", 10, 20, 500.0,
            prompt_text="Tell me about patents",
            response_text="Patents are...",
        )
        entry = _read_last_entry(log_file)
        assert "prompt_text" not in entry
        assert "response_text" not in entry
