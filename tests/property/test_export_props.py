"""Property-based tests for DOCX export.

Validates: Requirements 10.1, 10.5
"""

import tempfile
from pathlib import Path

from docx import Document
from hypothesis import given, settings
from hypothesis import strategies as st

from patent_system.export.docx_exporter import DOCXExporter, validate_export

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Non-empty text for valid claims/description content.
# Restrict to printable characters that are XML-compatible (python-docx/lxml
# rejects NULL bytes and control characters).
_xml_safe_alphabet = st.characters(
    categories=("L", "M", "N", "P", "S", "Z"),
    exclude_characters="\x00",
)
_non_empty_text = st.text(
    alphabet=_xml_safe_alphabet, min_size=1, max_size=500
).filter(lambda s: s.strip() != "")

# Possibly-empty or None values for incomplete draft testing.
_empty_or_none = st.one_of(st.none(), st.just(""))


# ---------------------------------------------------------------------------
# Property 7: DOCX export contains required sections
# Feature: patent-analysis-drafting, Property 7: DOCX export contains required sections
# ---------------------------------------------------------------------------


class TestDOCXExportContainsRequiredSections:
    """Property 7: DOCX export contains required sections.

    For any non-empty claims text and non-empty description text, exporting
    to DOCX and reading back the document produces a file containing both
    the claims content and the description content.

    **Validates: Requirements 10.1**
    """

    @given(claims=_non_empty_text, description=_non_empty_text)
    @settings(max_examples=100)
    def test_exported_docx_contains_claims_and_description(
        self,
        claims: str,
        description: str,
    ) -> None:
        """Exported DOCX contains both claims and description text."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            exporter = DOCXExporter(template_dir=tmp, template_name=None)
            output_path = tmp / "export.docx"

            exporter.export(claims, description, output_path)

            doc = Document(str(output_path))
            full_text = "\n".join(p.text for p in doc.paragraphs)

            assert claims in full_text, (
                f"Claims text not found in exported document. "
                f"Expected: {claims!r}"
            )
            assert description in full_text, (
                f"Description text not found in exported document. "
                f"Expected: {description!r}"
            )


# ---------------------------------------------------------------------------
# Property 8: Export blocked for incomplete drafts
# Feature: patent-analysis-drafting, Property 8: Export blocked for incomplete drafts
# ---------------------------------------------------------------------------


class TestExportBlockedForIncompleteDrafts:
    """Property 8: Export blocked for incomplete drafts.

    For any state where claims is empty/None or description is empty/None,
    the validation function returns False (preventing export).

    **Validates: Requirements 10.5**
    """

    @given(description=_non_empty_text)
    @settings(max_examples=100)
    def test_none_claims_blocked(self, description: str) -> None:
        """validate_export returns False when claims is None."""
        assert validate_export(None, description) is False

    @given(claims=_non_empty_text)
    @settings(max_examples=100)
    def test_none_description_blocked(self, claims: str) -> None:
        """validate_export returns False when description is None."""
        assert validate_export(claims, None) is False

    @given(description=_non_empty_text)
    @settings(max_examples=100)
    def test_empty_claims_blocked(self, description: str) -> None:
        """validate_export returns False when claims is empty string."""
        assert validate_export("", description) is False

    @given(claims=_non_empty_text)
    @settings(max_examples=100)
    def test_empty_description_blocked(self, claims: str) -> None:
        """validate_export returns False when description is empty string."""
        assert validate_export(claims, "") is False

    @given(
        claims=_empty_or_none,
        description=_empty_or_none,
    )
    @settings(max_examples=100)
    def test_both_empty_or_none_blocked(
        self,
        claims: str | None,
        description: str | None,
    ) -> None:
        """validate_export returns False when both claims and description are empty/None."""
        assert validate_export(claims, description) is False
