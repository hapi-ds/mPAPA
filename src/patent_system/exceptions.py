"""Custom exception hierarchy for the patent analysis system."""


class AgentError(Exception):
    """Base exception for agent errors."""

    pass


class LLMConnectionError(AgentError):
    """Raised when LM Studio is unreachable."""

    pass


class SourceUnavailableError(AgentError):
    """Raised when an external data source is unreachable.

    Attributes:
        source_name: Name of the unavailable data source.
        original_error: The underlying exception that caused the failure.
    """

    def __init__(self, source_name: str, original_error: Exception) -> None:
        self.source_name = source_name
        self.original_error = original_error
        super().__init__(f"Source {source_name} unavailable: {original_error}")
