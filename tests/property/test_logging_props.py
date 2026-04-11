"""Property-based tests for structured logging.

Validates: Requirements 20.1, 20.2, 20.3, 20.4, 20.5, 20.6
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from patent_system.logging_config import (
    StructuredFormatter,
    log_agent_invocation,
    log_db_error,
    log_external_request,
    log_llm_call,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Safe printable text without control characters
_safe_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S"),
        whitelist_characters=" /-_:.",
    ),
    min_size=1,
    max_size=120,
)

_log_levels = st.sampled_from([logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR])

_pos_float = st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False)

_pos_int = st.integers(min_value=0, max_value=1_000_000)


def _make_logger(tmp_path: Path, level: int = logging.DEBUG) -> tuple[logging.Logger, Path]:
    """Create a fresh logger with StructuredFormatter writing to a temp file."""
    log_file = tmp_path / "test.log"
    logger = logging.getLogger(f"prop_logging_{id(log_file)}_{level}")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False
    handler = logging.FileHandler(str(log_file), encoding="utf-8")
    handler.setFormatter(StructuredFormatter())
    logger.addHandler(handler)
    return logger, log_file


def _read_last_entry(log_file: Path) -> dict:
    """Read and parse the last JSON log line from a file."""
    lines = log_file.read_text().strip().splitlines()
    return json.loads(lines[-1])


# ---------------------------------------------------------------------------
# Property 14: Structured log entry format
# Feature: patent-analysis-drafting, Property 14: Structured log entry format
# ---------------------------------------------------------------------------


class TestStructuredLogEntryFormat:
    """Property 14: Structured log entry format.

    For any log message at any level (DEBUG, INFO, WARNING, ERROR), the
    written log entry shall contain a valid ISO timestamp, the log level
    string, the module name, and the message content.

    **Validates: Requirements 20.1**
    """

    @given(message=_safe_text, level=_log_levels)
    @settings(max_examples=100)
    def test_entry_contains_required_fields(
        self, message: str, level: int, tmp_path_factory
    ) -> None:
        tmp_path = tmp_path_factory.mktemp("log14")
        logger, log_file = _make_logger(tmp_path, level=logging.DEBUG)

        logger.log(level, message)
        handler = logger.handlers[0]
        handler.flush()

        entry = _read_last_entry(log_file)

        # Valid ISO timestamp
        assert "timestamp" in entry
        datetime.fromisoformat(entry["timestamp"])

        # Log level string matches
        assert "level" in entry
        assert entry["level"] == logging.getLevelName(level)

        # Module name present
        assert "module" in entry
        assert isinstance(entry["module"], str)
        assert len(entry["module"]) > 0

        # Message content preserved
        assert "message" in entry
        assert entry["message"] == message


# ---------------------------------------------------------------------------
# Property 15: Event-specific log fields
# Feature: patent-analysis-drafting, Property 15: Event-specific log fields
# ---------------------------------------------------------------------------


class TestEventSpecificLogFields:
    """Property 15: Event-specific log fields.

    For any loggable event (agent invocation, external request, database
    error, or LLM API call), the log entry shall contain all fields
    required for that event type.

    **Validates: Requirements 20.2, 20.3, 20.4, 20.5**
    """

    @given(
        name=_safe_text,
        input_summary=_safe_text,
        output_summary=_safe_text,
        duration_ms=_pos_float,
    )
    @settings(max_examples=100)
    def test_agent_invocation_fields(
        self,
        name: str,
        input_summary: str,
        output_summary: str,
        duration_ms: float,
        tmp_path_factory,
    ) -> None:
        """Agent invocation entries contain name, input, output, duration."""
        tmp_path = tmp_path_factory.mktemp("log15a")
        logger, log_file = _make_logger(tmp_path)

        log_agent_invocation(logger, name, input_summary, output_summary, duration_ms)
        logger.handlers[0].flush()

        entry = _read_last_entry(log_file)
        assert entry["event_type"] == "agent_invocation"
        assert entry["name"] == name
        assert entry["input"] == input_summary
        assert entry["output"] == output_summary
        assert entry["duration"] == duration_ms

    @given(
        source=_safe_text,
        query=_safe_text,
        status=_safe_text,
        response_time_ms=_pos_float,
    )
    @settings(max_examples=100)
    def test_external_request_fields(
        self,
        source: str,
        query: str,
        status: str,
        response_time_ms: float,
        tmp_path_factory,
    ) -> None:
        """External request entries contain source, query, status, response_time."""
        tmp_path = tmp_path_factory.mktemp("log15b")
        logger, log_file = _make_logger(tmp_path)

        log_external_request(logger, source, query, status, response_time_ms)
        logger.handlers[0].flush()

        entry = _read_last_entry(log_file)
        assert entry["event_type"] == "external_request"
        assert entry["source"] == source
        assert entry["query"] == query
        assert entry["status"] == status
        assert entry["response_time"] == response_time_ms

    @given(
        operation=_safe_text,
        table=_safe_text,
        error=_safe_text,
    )
    @settings(max_examples=100)
    def test_db_error_fields(
        self,
        operation: str,
        table: str,
        error: str,
        tmp_path_factory,
    ) -> None:
        """DB error entries contain operation, table, error."""
        tmp_path = tmp_path_factory.mktemp("log15c")
        logger, log_file = _make_logger(tmp_path)

        log_db_error(logger, operation, table, error)
        logger.handlers[0].flush()

        entry = _read_last_entry(log_file)
        assert entry["event_type"] == "db_error"
        assert entry["operation"] == operation
        assert entry["table"] == table
        assert entry["error"] == error

    @given(
        model=_safe_text,
        prompt_tokens=_pos_int,
        response_tokens=_pos_int,
        latency_ms=_pos_float,
    )
    @settings(max_examples=100)
    def test_llm_call_fields(
        self,
        model: str,
        prompt_tokens: int,
        response_tokens: int,
        latency_ms: float,
        tmp_path_factory,
    ) -> None:
        """LLM call entries contain model, prompt_tokens, response_tokens, latency."""
        tmp_path = tmp_path_factory.mktemp("log15d")
        logger, log_file = _make_logger(tmp_path)

        log_llm_call(logger, model, prompt_tokens, response_tokens, latency_ms)
        logger.handlers[0].flush()

        entry = _read_last_entry(log_file)
        assert entry["event_type"] == "llm_call"
        assert entry["model"] == model
        assert entry["prompt_tokens"] == prompt_tokens
        assert entry["response_tokens"] == response_tokens
        assert entry["latency"] == latency_ms


# ---------------------------------------------------------------------------
# Property 16: DEBUG log level includes full LLM text
# Feature: patent-analysis-drafting, Property 16: DEBUG log level includes full LLM text
# ---------------------------------------------------------------------------


class TestDebugLlmText:
    """Property 16: DEBUG log level includes full LLM text.

    For any LLM API call logged at DEBUG level, the log entry shall
    include the full prompt text and full response text in addition to
    the standard LLM call fields.

    **Validates: Requirements 20.6**
    """

    @given(
        model=_safe_text,
        prompt_tokens=_pos_int,
        response_tokens=_pos_int,
        latency_ms=_pos_float,
        prompt_text=_safe_text,
        response_text=_safe_text,
    )
    @settings(max_examples=100)
    def test_debug_level_includes_prompt_and_response(
        self,
        model: str,
        prompt_tokens: int,
        response_tokens: int,
        latency_ms: float,
        prompt_text: str,
        response_text: str,
        tmp_path_factory,
    ) -> None:
        """At DEBUG level, prompt_text and response_text are present."""
        tmp_path = tmp_path_factory.mktemp("log16")
        logger, log_file = _make_logger(tmp_path, level=logging.DEBUG)

        log_llm_call(
            logger,
            model,
            prompt_tokens,
            response_tokens,
            latency_ms,
            prompt_text=prompt_text,
            response_text=response_text,
        )
        logger.handlers[0].flush()

        entry = _read_last_entry(log_file)

        # Standard fields still present
        assert entry["event_type"] == "llm_call"
        assert entry["model"] == model
        assert entry["prompt_tokens"] == prompt_tokens
        assert entry["response_tokens"] == response_tokens
        assert entry["latency"] == latency_ms

        # DEBUG-only fields
        assert entry["prompt_text"] == prompt_text
        assert entry["response_text"] == response_text
