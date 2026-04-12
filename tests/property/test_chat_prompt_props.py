"""Property-based tests for chat prompt construction.

Feature: placeholder-to-real-implementation, Property 7: Chat prompt contains context and question

Validates: Requirements 5.2
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from patent_system.gui.chat_panel import build_chat_prompt

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# A single RAG context document with a non-empty "text" field
_context_doc = st.fixed_dictionaries({"text": st.text(min_size=1)})

# A list of context documents (0 to 10 items)
_context_docs = st.lists(_context_doc, min_size=0, max_size=10)

# Non-empty user question
_question = st.text(min_size=1)


# ---------------------------------------------------------------------------
# Property 7: Chat prompt contains context and question
# Feature: placeholder-to-real-implementation, Property 7: Chat prompt contains context and question
# ---------------------------------------------------------------------------


class TestChatPromptContainsContextAndQuestion:
    """Property 7: Chat prompt contains context and question.

    For any list of RAG context documents (each with a "text" field) and
    any non-empty user question string, the prompt constructed by the chat
    panel shall contain the user question text and shall contain the text
    from each context document.

    **Validates: Requirements 5.2**
    """

    @given(context_docs=_context_docs, question=_question)
    @settings(max_examples=100)
    def test_prompt_contains_question(
        self,
        context_docs: list[dict],
        question: str,
    ) -> None:
        """The constructed prompt always contains the user question."""
        prompt = build_chat_prompt(context_docs, question)
        assert question in prompt

    @given(
        context_docs=st.lists(_context_doc, min_size=1, max_size=10),
        question=_question,
    )
    @settings(max_examples=100)
    def test_prompt_contains_all_context_texts(
        self,
        context_docs: list[dict],
        question: str,
    ) -> None:
        """The constructed prompt contains the text from every context document."""
        prompt = build_chat_prompt(context_docs, question)
        for doc in context_docs:
            assert doc["text"] in prompt

    @given(question=_question)
    @settings(max_examples=100)
    def test_empty_context_still_contains_question(
        self,
        question: str,
    ) -> None:
        """When no context documents are provided, the prompt still contains the question."""
        prompt = build_chat_prompt([], question)
        assert question in prompt
        assert "no prior art context" in prompt.lower()
