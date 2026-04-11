"""Structured logging setup for the Patent Analysis & Drafting System.

Provides a JSON formatter producing log entries with ISO timestamp, log level,
module name, and message. Includes helper functions for event-specific log
fields: agent invocations, external requests, DB errors, and LLM calls.

Requirements: 20.1, 20.2, 20.3, 20.4, 20.5, 20.6
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StructuredFormatter(logging.Formatter):
    """JSON formatter producing structured log entries.

    Each log line is a valid JSON object with fields:
    - timestamp: ISO 8601 format
    - level: log level string (DEBUG, INFO, WARNING, ERROR)
    - module: module name from the log record
    - message: formatted log message
    - Plus any extra fields passed via the ``extra`` dict.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON string."""
        entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }

        # Merge any extra fields attached to the record.
        if hasattr(record, "extra_fields"):
            entry.update(record.extra_fields)

        return json.dumps(entry, default=str)


def setup_logging(settings: Any) -> None:
    """Configure the root logger with a structured JSON file handler.

    Args:
        settings: An ``AppSettings`` instance providing ``log_file_path``
            and ``log_level``.
    """
    log_path = Path(settings.log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(str(log_path), encoding="utf-8")
    handler.setFormatter(StructuredFormatter())

    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)


def _log_with_extras(
    logger: logging.Logger,
    level: int,
    message: str,
    extra_fields: dict[str, Any],
) -> None:
    """Emit a log record with additional structured fields.

    The *extra_fields* dict is attached to the record so that
    ``StructuredFormatter`` can merge them into the JSON output.
    """
    logger.log(level, message, extra={"extra_fields": extra_fields})


# ---------------------------------------------------------------------------
# Event-specific helper functions
# ---------------------------------------------------------------------------


def log_agent_invocation(
    logger: logging.Logger,
    name: str,
    input_summary: str,
    output_summary: str,
    duration_ms: float,
) -> None:
    """Log an agent invocation event.

    Requirement 20.2 — agent name, input, output, duration.
    """
    _log_with_extras(
        logger,
        logging.INFO,
        f"Agent invocation: {name}",
        {
            "event_type": "agent_invocation",
            "name": name,
            "input": input_summary,
            "output": output_summary,
            "duration": duration_ms,
        },
    )


def log_external_request(
    logger: logging.Logger,
    source: str,
    query: str,
    status: str,
    response_time_ms: float,
) -> None:
    """Log an external data-source request event.

    Requirement 20.3 — source, query, status, response_time.
    """
    _log_with_extras(
        logger,
        logging.INFO,
        f"External request: {source}",
        {
            "event_type": "external_request",
            "source": source,
            "query": query,
            "status": status,
            "response_time": response_time_ms,
        },
    )


def log_db_error(
    logger: logging.Logger,
    operation: str,
    table: str,
    error: str,
) -> None:
    """Log a database error event.

    Requirement 20.4 — operation, table, error.
    """
    _log_with_extras(
        logger,
        logging.ERROR,
        f"Database error: {operation} on {table}",
        {
            "event_type": "db_error",
            "operation": operation,
            "table": table,
            "error": error,
        },
    )


def log_llm_call(
    logger: logging.Logger,
    model: str,
    prompt_tokens: int,
    response_tokens: int,
    latency_ms: float,
    prompt_text: str | None = None,
    response_text: str | None = None,
) -> None:
    """Log an LLM API call event.

    Requirement 20.5 — model, prompt_tokens, response_tokens, latency.
    Requirement 20.6 — at DEBUG level, include full prompt and response text.
    """
    fields: dict[str, Any] = {
        "event_type": "llm_call",
        "model": model,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "latency": latency_ms,
    }

    # Include full text only when the logger is at DEBUG level.
    if logger.isEnabledFor(logging.DEBUG):
        if prompt_text is not None:
            fields["prompt_text"] = prompt_text
        if response_text is not None:
            fields["response_text"] = response_text

    _log_with_extras(
        logger,
        logging.INFO,
        f"LLM call: {model}",
        fields,
    )
