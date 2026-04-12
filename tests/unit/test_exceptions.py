"""Unit tests for custom exception classes."""

from patent_system.exceptions import (
    AgentError,
    LLMConnectionError,
    SourceUnavailableError,
)


class TestAgentError:
    """Tests for the AgentError base exception."""

    def test_is_exception(self) -> None:
        assert issubclass(AgentError, Exception)

    def test_can_raise_and_catch(self) -> None:
        with __import__("pytest").raises(AgentError):
            raise AgentError("something went wrong")


class TestLLMConnectionError:
    """Tests for LLMConnectionError."""

    def test_inherits_agent_error(self) -> None:
        assert issubclass(LLMConnectionError, AgentError)

    def test_caught_as_agent_error(self) -> None:
        with __import__("pytest").raises(AgentError):
            raise LLMConnectionError("LM Studio unreachable")


class TestSourceUnavailableError:
    """Tests for SourceUnavailableError."""

    def test_inherits_agent_error(self) -> None:
        assert issubclass(SourceUnavailableError, AgentError)

    def test_stores_source_name(self) -> None:
        original = ConnectionError("timeout")
        err = SourceUnavailableError("EPO OPS", original)
        assert err.source_name == "EPO OPS"

    def test_stores_original_error(self) -> None:
        original = ConnectionError("timeout")
        err = SourceUnavailableError("ArXiv", original)
        assert err.original_error is original

    def test_message_format(self) -> None:
        original = TimeoutError("connection timed out")
        err = SourceUnavailableError("PubMed", original)
        assert "PubMed" in str(err)
        assert "connection timed out" in str(err)

    def test_caught_as_agent_error(self) -> None:
        with __import__("pytest").raises(AgentError):
            raise SourceUnavailableError("Google Patents", ConnectionError("fail"))


class TestReExportsFromAgents:
    """Tests that exceptions are re-exported from agents package."""

    def test_agent_error_reexported(self) -> None:
        from patent_system.agents import AgentError as ReExported

        assert ReExported is AgentError

    def test_llm_connection_error_reexported(self) -> None:
        from patent_system.agents import LLMConnectionError as ReExported

        assert ReExported is LLMConnectionError

    def test_source_unavailable_error_reexported(self) -> None:
        from patent_system.agents import SourceUnavailableError as ReExported

        assert ReExported is SourceUnavailableError
