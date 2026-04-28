"""Property-based tests for the FullTextDownloader service.

Validates: Requirements 11.3
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from patent_system.services.full_text_downloader import FullTextDownloader, _sanitize_filename

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Source names: printable strings that represent source identifiers
_source_names = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=50,
)

# Identifiers: printable strings that represent document IDs
_identifiers = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=80,
)

# Minimal PDF content (not a valid PDF, but enough to test path logic)
_pdf_bytes = st.binary(min_size=1, max_size=100)


# ---------------------------------------------------------------------------
# Property 5: PDF path follows source/identifier pattern (Req 11.3)
# ---------------------------------------------------------------------------


class TestPdfPathPattern:
    """Property 5: PDF path follows source/identifier pattern.

    For any source name and identifier, the saved PDF path follows
    ``{pdf_download_dir}/{sanitized_source}/{sanitized_id}.pdf``.

    **Validates: Requirements 11.3**
    """

    @given(
        source=_source_names,
        identifier=_identifiers,
        content=_pdf_bytes,
    )
    @settings(max_examples=200)
    def test_pdf_path_follows_source_identifier_pattern(
        self,
        source: str,
        identifier: str,
        content: bytes,
    ) -> None:
        """Saved PDF path has structure {source_dir}/{sanitized_id}.pdf."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from patent_system.config import AppSettings

            settings = AppSettings(
                pdf_download_dir=Path(tmpdir),
                full_text_download_enabled=True,
            )
            downloader = FullTextDownloader(settings)
            path = downloader.save_pdf(content, source, identifier)

            # Path must end with .pdf
            assert path.suffix == ".pdf"

            # Parent directory name must be the sanitized source
            expected_source_dir = _sanitize_filename(source)
            assert path.parent.name == expected_source_dir

            # Filename (without extension) must be the sanitized identifier
            expected_filename = _sanitize_filename(identifier)
            assert path.stem == expected_filename

            # File must actually exist and contain the content
            assert path.exists()
            assert path.read_bytes() == content
