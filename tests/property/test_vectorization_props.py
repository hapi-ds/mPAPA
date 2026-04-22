"""Property-based tests for the unified vectorization pipeline.

Validates: Requirements 6.2, 6.3, 6.4, 6.5
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from patent_system.rag.vectorization import prepare_vectorization_text

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Arbitrary text that may include whitespace, unicode, etc.
_any_text = st.text(min_size=0, max_size=500)

# Non-empty text (at least one non-whitespace character)
_nonempty_text = st.text(min_size=1, max_size=200).filter(lambda s: s.strip())

# Positive max_chars values
_max_chars = st.integers(min_value=1, max_value=10_000)

# Optional full_text (None or a string)
_optional_text = st.one_of(st.none(), _any_text)


# ---------------------------------------------------------------------------
# Property 1: Vectorization text length invariant (Req 6.2, 6.3, 6.5)
# ---------------------------------------------------------------------------


class TestVectorizationTextLengthInvariant:
    """Property 1: Vectorization text length invariant.

    For any combination of title, abstract, and full_text strings,
    len(result) <= max_chars.

    **Validates: Requirements 6.2, 6.3, 6.5**
    """

    @given(
        title=_any_text,
        abstract=_any_text,
        full_text=_optional_text,
        max_chars=_max_chars,
    )
    @settings(max_examples=200)
    def test_output_length_never_exceeds_max_chars(
        self,
        title: str,
        abstract: str,
        full_text: str | None,
        max_chars: int,
    ) -> None:
        """For any inputs, the result length is at most max_chars."""
        result = prepare_vectorization_text(title, abstract, full_text, max_chars)
        assert len(result) <= max_chars


# ---------------------------------------------------------------------------
# Property 2: Vectorization text construction order (Req 6.4)
# ---------------------------------------------------------------------------


class TestVectorizationTextConstructionOrder:
    """Property 2: Vectorization text construction order.

    For non-empty title, abstract, full_text with large max_chars,
    title appears before abstract appears before full_text.

    **Validates: Requirements 6.4**
    """

    @given(
        title=_nonempty_text,
        abstract=_nonempty_text,
        full_text=_nonempty_text,
    )
    @settings(max_examples=200)
    def test_title_before_abstract_before_full_text(
        self,
        title: str,
        abstract: str,
        full_text: str,
    ) -> None:
        """With large max_chars, title appears before abstract before full_text."""
        # Tag each part with a unique prefix to ensure non-overlapping substrings
        tagged_title = f"TITLE:{title}"
        tagged_abstract = f"ABSTRACT:{abstract}"
        tagged_full_text = f"FULLTEXT:{full_text}"

        max_chars = len(tagged_title) + len(tagged_abstract) + len(tagged_full_text) + 100
        result = prepare_vectorization_text(
            tagged_title, tagged_abstract, tagged_full_text, max_chars,
        )

        title_pos = result.index(tagged_title)
        abstract_pos = result.index(tagged_abstract)
        full_text_pos = result.index(tagged_full_text)
        assert title_pos < abstract_pos < full_text_pos


# ---------------------------------------------------------------------------
# Property 3: Vectorization text non-empty when any input is non-empty (Req 6.4)
# ---------------------------------------------------------------------------


class TestVectorizationTextNonEmpty:
    """Property 3: Vectorization text non-empty when any input is non-empty.

    When any input is non-empty (after stripping), result is non-empty.

    **Validates: Requirements 6.4**
    """

    @given(
        title=_any_text,
        abstract=_any_text,
        full_text=_optional_text,
    )
    @settings(max_examples=200)
    def test_non_empty_when_any_input_non_empty(
        self,
        title: str,
        abstract: str,
        full_text: str | None,
    ) -> None:
        """When at least one input has non-whitespace content, result is non-empty."""
        has_content = (
            bool(title.strip())
            or bool(abstract.strip())
            or (full_text is not None and bool(full_text.strip()))
        )
        if not has_content:
            return  # Skip: all inputs are empty/whitespace-only

        result = prepare_vectorization_text(title, abstract, full_text)
        assert len(result) > 0
