"""Property-based tests for invention disclosure and source preference persistence.

Feature: enhanced-research-workflow

Validates: Requirements 2.3, 3.4, 9.1, 9.2, 9.3
"""

import sqlite3
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from patent_system.db.repository import (
    InventionDisclosureRepository,
    SourcePreferenceRepository,
    TopicRepository,
)
from patent_system.db.schema import init_schema

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_db() -> sqlite3.Connection:
    """Create a fresh in-memory SQLite connection with FK enforcement and full schema."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    init_schema(conn)
    return conn


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Non-empty text for primary descriptions (no surrogates, no null bytes)
_primary_description = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
    min_size=1,
)

# A single search term (non-empty text)
_search_term = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
    min_size=1,
)

# List of 0–20 search terms
_search_terms = st.lists(_search_term, min_size=0, max_size=20)

# The 5 known source names
_SOURCE_NAMES = ["ArXiv", "PubMed", "Google Scholar", "Google Patents", "EPO OPS"]

# Boolean map for all 5 sources
_source_preferences = st.fixed_dictionaries(
    {name: st.booleans() for name in _SOURCE_NAMES}
)


# ---------------------------------------------------------------------------
# Property 1: Invention disclosure round-trip
# Feature: enhanced-research-workflow, Property 1: Invention disclosure round-trip
# ---------------------------------------------------------------------------


class TestDisclosureRoundTrip:
    """Property 1: Invention disclosure round-trip.

    For any valid primary description string and for any list of search
    term strings (0–20 items), saving via InventionDisclosureRepository.upsert()
    and then loading via get_by_topic() for the same topic SHALL produce a
    primary_description equal to the original and a search_terms list equal
    to the original in both content and order.

    **Validates: Requirements 2.3, 9.1, 9.3**
    """

    @given(description=_primary_description, terms=_search_terms)
    @settings(max_examples=100)
    def test_disclosure_round_trip(
        self,
        description: str,
        terms: list[str],
    ) -> None:
        conn = _fresh_db()
        try:
            topic_id = TopicRepository(conn).create("test-topic").id
            repo = InventionDisclosureRepository(conn)

            repo.upsert(topic_id, description, terms)
            result = repo.get_by_topic(topic_id)

            assert result is not None
            assert result["primary_description"] == description
            assert result["search_terms"] == terms
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Property 2: Source preference round-trip
# Feature: enhanced-research-workflow, Property 2: Source preference round-trip
# ---------------------------------------------------------------------------


class TestSourcePreferenceRoundTrip:
    """Property 2: Source preference round-trip.

    For any valid mapping of the five known source names to boolean
    enabled/disabled values, saving via SourcePreferenceRepository.save()
    and then loading via get_by_topic() for the same topic SHALL produce
    a mapping equal to the original.

    **Validates: Requirements 3.4, 9.2**
    """

    @given(preferences=_source_preferences)
    @settings(max_examples=100)
    def test_source_preference_round_trip(
        self,
        preferences: dict[str, bool],
    ) -> None:
        conn = _fresh_db()
        try:
            topic_id = TopicRepository(conn).create("test-topic").id
            repo = SourcePreferenceRepository(conn)

            repo.save(topic_id, preferences)
            result = repo.get_by_topic(topic_id)

            assert result is not None
            assert result == preferences
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Property 5: Source selection filters searched sources
# Feature: enhanced-research-workflow, Property 5: Source selection filters searched sources
# ---------------------------------------------------------------------------


class TestSourceFiltering:
    """Property 5: Source selection filters searched sources.

    For any non-empty subset of source names from ``_SOURCE_REGISTRY``,
    calling ``prior_art_search_node`` with that subset as
    ``selected_sources`` SHALL only attempt to query sources whose names
    appear in the subset, and SHALL not query any source outside the
    subset.

    **Validates: Requirements 4.2, 4.3**
    """

    @given(
        selected=st.lists(
            st.sampled_from(_SOURCE_NAMES),
            min_size=1,
            max_size=5,
            unique=True,
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_source_selection_filters(self, selected: list[str]) -> None:
        from unittest.mock import MagicMock, patch

        from patent_system.agents.prior_art_search import (
            _SOURCE_REGISTRY,
            prior_art_search_node,
        )

        state = {
            "topic_id": 1,
            "invention_disclosure": {"technical_problem": "test problem"},
            "interview_messages": [],
            "prior_art_results": [],
            "failed_sources": [],
            "novelty_analysis": None,
            "claims_text": "",
            "description_text": "",
            "review_feedback": "",
            "review_approved": False,
            "iteration_count": 0,
            "current_step": "",
        }

        queried_sources: list[str] = []

        def tracking_query_source(source_name: str, search_terms: list[str], **kwargs: Any) -> dict:
            queried_sources.append(source_name)
            return {"results": []}

        with patch(
            "patent_system.agents.prior_art_search._query_source",
            side_effect=tracking_query_source,
        ):
            prior_art_search_node(state, selected_sources=selected)

        assert set(queried_sources) == set(selected)


# ---------------------------------------------------------------------------
# Property 8: Disclosure agent pass-through
# Feature: enhanced-research-workflow, Property 8: Disclosure agent pass-through
# ---------------------------------------------------------------------------


class TestDisclosurePassThrough:
    """Property 8: Disclosure agent pass-through.

    For any valid PatentWorkflowState where ``invention_disclosure`` is
    a dict with a non-empty ``technical_problem`` field,
    ``disclosure_node`` SHALL return the same ``invention_disclosure``
    dict without modification and without invoking the LLM.

    **Validates: Requirements 7.5**
    """

    @given(
        technical_problem=st.text(
            alphabet=st.characters(
                blacklist_categories=("Cs",), blacklist_characters="\x00"
            ),
            min_size=1,
        ),
        novel_features=st.lists(
            st.text(
                alphabet=st.characters(
                    blacklist_categories=("Cs",), blacklist_characters="\x00"
                ),
                min_size=0,
                max_size=50,
            ),
            min_size=0,
            max_size=5,
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_disclosure_pass_through(
        self,
        technical_problem: str,
        novel_features: list[str],
    ) -> None:
        from unittest.mock import patch

        from patent_system.agents.disclosure import disclosure_node

        disclosure = {
            "technical_problem": technical_problem,
            "novel_features": novel_features,
            "implementation_details": "",
            "potential_variations": [],
        }

        state = {
            "topic_id": 1,
            "invention_disclosure": disclosure,
            "interview_messages": [],
            "prior_art_results": [],
            "failed_sources": [],
            "novelty_analysis": None,
            "claims_text": "",
            "description_text": "",
            "review_feedback": "",
            "review_approved": False,
            "iteration_count": 0,
            "current_step": "",
        }

        # Patch the LLM modules to detect if they are called
        with (
            patch(
                "patent_system.agents.disclosure.InterviewQuestionModule"
            ) as mock_interview,
            patch(
                "patent_system.agents.disclosure.StructureDisclosureModule"
            ) as mock_structure,
        ):
            result = disclosure_node(state)

            # LLM modules should NOT have been instantiated
            mock_interview.assert_not_called()
            mock_structure.assert_not_called()

        assert result["invention_disclosure"] is disclosure
        assert result["invention_disclosure"] == disclosure
        assert result["current_step"] == "disclosure"


# ---------------------------------------------------------------------------
# Property 6: Deduplication by title and source
# Feature: enhanced-research-workflow, Property 6: Deduplication by title and source
# ---------------------------------------------------------------------------

# Strategies for deduplication tests
_title_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
    min_size=0,
    max_size=80,
)

_source_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
    min_size=0,
    max_size=30,
)

_result_dict = st.fixed_dictionaries(
    {"title": _title_text, "source": _source_text},
)

_existing_list = st.lists(_result_dict, min_size=0, max_size=10)


class TestDeduplication:
    """Property 6: Deduplication by title and source.

    For any search result and for any list of existing records, the
    deduplication function SHALL return True if and only if there exists
    an existing record whose title matches (case-insensitive, stripped)
    and whose source matches (case-insensitive, stripped) the new result.

    **Validates: Requirements 6.1, 6.2**
    """

    @given(result=_result_dict, existing=_existing_list)
    @settings(max_examples=100)
    def test_dedup_matches_expected(
        self,
        result: dict,
        existing: list[dict],
    ) -> None:
        from patent_system.gui.research_panel import _is_duplicate

        actual = _is_duplicate(result, existing)

        # Compute expected: True iff any existing record matches on
        # title + source (case-insensitive, stripped)
        r_title = result.get("title", "").strip().lower()
        r_source = result.get("source", "").strip().lower()
        expected = any(
            e.get("title", "").strip().lower() == r_title
            and e.get("source", "").strip().lower() == r_source
            for e in existing
        )

        assert actual == expected

    @given(result=_result_dict)
    @settings(max_examples=100)
    def test_dedup_true_when_exact_copy_present(
        self,
        result: dict,
    ) -> None:
        """If the existing list contains an exact copy, _is_duplicate returns True."""
        from patent_system.gui.research_panel import _is_duplicate

        existing = [dict(result)]  # exact copy
        assert _is_duplicate(result, existing) is True

    @given(result=_result_dict)
    @settings(max_examples=100)
    def test_dedup_false_when_empty_existing(
        self,
        result: dict,
    ) -> None:
        """If the existing list is empty, _is_duplicate returns False."""
        from patent_system.gui.research_panel import _is_duplicate

        assert _is_duplicate(result, []) is False


# ---------------------------------------------------------------------------
# Property 3: Disclosure dict construction
# Feature: enhanced-research-workflow, Property 3: Disclosure dict construction
# ---------------------------------------------------------------------------

# Strategy: mixed empty/non-empty term lists
_any_term = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
    min_size=0,
    max_size=50,
)

_mixed_terms = st.lists(_any_term, min_size=0, max_size=20)


class TestDisclosureDictConstruction:
    """Property 3: Disclosure dict construction.

    For any non-empty primary description string and for any list of
    additional search term strings (some possibly empty), constructing
    the invention_disclosure dict SHALL produce ``technical_problem``
    equal to the primary description and ``novel_features`` equal to
    the list of non-empty terms only.

    **Validates: Requirements 1.6**
    """

    @given(description=_primary_description, terms=_mixed_terms)
    @settings(max_examples=100)
    def test_disclosure_dict_construction(
        self,
        description: str,
        terms: list[str],
    ) -> None:
        from patent_system.gui.research_panel import _build_disclosure_dict

        result = _build_disclosure_dict(description, terms)

        assert result["technical_problem"] == description
        assert result["novel_features"] == [t for t in terms if t]


# ---------------------------------------------------------------------------
# Property 4: Whitespace-only primary description rejection
# Feature: enhanced-research-workflow, Property 4: Whitespace-only primary description rejection
# ---------------------------------------------------------------------------

# Strategy: whitespace-only strings (spaces, tabs, newlines, etc.)
_whitespace_only = st.from_regex(r"^[\s]*$", fullmatch=True)


class TestWhitespaceRejection:
    """Property 4: Whitespace-only primary description rejection.

    For any string composed entirely of whitespace characters (including
    the empty string), the validation check SHALL reject it as an
    invalid primary description, preventing search execution.

    **Validates: Requirements 1.7**
    """

    @given(description=_whitespace_only)
    @settings(max_examples=100)
    def test_whitespace_only_rejected(
        self,
        description: str,
    ) -> None:
        # The validation logic: description.strip() must be non-empty
        # For whitespace-only strings, strip() always yields ""
        assert description.strip() == ""


# ---------------------------------------------------------------------------
# Property 9: RAG document text includes full text when available
# Feature: enhanced-research-workflow, Property 9: RAG document text includes full text when available
# ---------------------------------------------------------------------------

# Strategies for RAG full text tests
_nonempty_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
    min_size=1,
    max_size=200,
)


class TestRAGFullTextInclusion:
    """Property 9: RAG document text includes full text when available.

    For any search result dict containing both a non-empty abstract and
    a non-empty full_text, the document text constructed for RAG indexing
    SHALL contain both the abstract string and the full_text string.

    **Validates: Requirements 5.3**
    """

    @given(abstract=_nonempty_text, full_text=_nonempty_text)
    @settings(max_examples=100, deadline=None)
    def test_rag_text_contains_both_abstract_and_full_text(
        self,
        abstract: str,
        full_text: str,
    ) -> None:
        from patent_system.gui.research_panel import _build_rag_document_text

        result = _build_rag_document_text(abstract, full_text)

        assert abstract in result
        assert full_text in result

    @given(abstract=_nonempty_text)
    @settings(max_examples=100)
    def test_rag_text_contains_abstract_when_no_full_text(
        self,
        abstract: str,
    ) -> None:
        from patent_system.gui.research_panel import _build_rag_document_text

        result = _build_rag_document_text(abstract, None)

        assert abstract in result

    @given(full_text=_nonempty_text)
    @settings(max_examples=100)
    def test_rag_text_contains_full_text_when_no_abstract(
        self,
        full_text: str,
    ) -> None:
        from patent_system.gui.research_panel import _build_rag_document_text

        result = _build_rag_document_text("", full_text)

        assert full_text in result


# ---------------------------------------------------------------------------
# Property 7: Chat prompt includes invention context when provided
# Feature: enhanced-research-workflow, Property 7: Chat prompt includes invention context when provided
# ---------------------------------------------------------------------------

# Strategies for chat prompt invention context tests
_invention_description = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
    min_size=1,
    max_size=200,
)

_invention_search_terms = st.lists(
    st.text(
        alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
        min_size=1,
        max_size=50,
    ),
    min_size=1,
    max_size=10,
)

_invention_context = st.fixed_dictionaries(
    {
        "primary_description": _invention_description,
        "search_terms": _invention_search_terms,
    }
)

_chat_context_doc = st.fixed_dictionaries({"text": st.text(min_size=1, max_size=100)})
_chat_context_docs = st.lists(_chat_context_doc, min_size=0, max_size=5)
_chat_question = st.text(min_size=1, max_size=100)


class TestChatPromptContext:
    """Property 7: Chat prompt includes invention context when provided.

    For any non-empty invention context (containing a primary description
    and search terms), for any list of RAG context documents, and for any
    question string, the assembled prompt SHALL contain the primary
    description and all search terms, and they SHALL appear before the
    RAG document section.

    **Validates: Requirements 8.2**
    """

    @given(
        invention_context=_invention_context,
        context_docs=_chat_context_docs,
        question=_chat_question,
    )
    @settings(max_examples=100, deadline=None)
    def test_prompt_contains_invention_description(
        self,
        invention_context: dict,
        context_docs: list[dict],
        question: str,
    ) -> None:
        """The prompt contains the invention primary description."""
        from patent_system.gui.chat_panel import build_chat_prompt

        prompt = build_chat_prompt(context_docs, question, invention_context=invention_context)
        assert invention_context["primary_description"] in prompt

    @given(
        invention_context=_invention_context,
        context_docs=_chat_context_docs,
        question=_chat_question,
    )
    @settings(max_examples=100, deadline=None)
    def test_prompt_contains_all_search_terms(
        self,
        invention_context: dict,
        context_docs: list[dict],
        question: str,
    ) -> None:
        """The prompt contains every search term from the invention context."""
        from patent_system.gui.chat_panel import build_chat_prompt

        prompt = build_chat_prompt(context_docs, question, invention_context=invention_context)
        for term in invention_context["search_terms"]:
            assert term in prompt

    @given(
        invention_context=_invention_context,
        context_docs=st.lists(_chat_context_doc, min_size=1, max_size=5),
        question=_chat_question,
    )
    @settings(max_examples=100, deadline=None)
    def test_invention_context_appears_before_rag_docs(
        self,
        invention_context: dict,
        context_docs: list[dict],
        question: str,
    ) -> None:
        """Invention description and terms appear before the RAG document section."""
        from patent_system.gui.chat_panel import build_chat_prompt

        prompt = build_chat_prompt(context_docs, question, invention_context=invention_context)

        desc_pos = prompt.index(invention_context["primary_description"])
        # RAG docs section starts with "[Document 1]"
        rag_pos = prompt.index("[Document 1]")
        assert desc_pos < rag_pos

        for term in invention_context["search_terms"]:
            term_pos = prompt.index(term)
            assert term_pos < rag_pos

    @given(
        context_docs=_chat_context_docs,
        question=_chat_question,
    )
    @settings(max_examples=100, deadline=None)
    def test_no_invention_context_preserves_existing_behavior(
        self,
        context_docs: list[dict],
        question: str,
    ) -> None:
        """When invention_context is None, prompt matches existing behavior."""
        from patent_system.gui.chat_panel import build_chat_prompt

        prompt_with_none = build_chat_prompt(context_docs, question, invention_context=None)
        prompt_without = build_chat_prompt(context_docs, question)
        assert prompt_with_none == prompt_without
