"""Property-based tests for source parsers.

Validates: Requirements 18.1, 18.2, 18.3, 18.4, 18.5, 18.6
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from patent_system.db.models import PatentRecord, ScientificPaperRecord
from patent_system.parsers.arxiv_parser import ArXivParser
from patent_system.parsers.base import BaseSourceParser
from patent_system.parsers.depatisnet import DEPATISnetParser
from patent_system.parsers.google_patents import GooglePatentsParser
from patent_system.parsers.google_scholar import GoogleScholarParser
from patent_system.parsers.pubmed import PubMedParser

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_nonempty_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S"),
        whitelist_characters=" /-_:.",
    ),
    min_size=1,
    max_size=80,
)

_patent_source = st.sampled_from(
    ["DEPATISnet", "Google Patents", "ArXiv", "PubMed", "Google Scholar"]
)


def _patent_entry() -> st.SearchStrategy[dict]:
    """Strategy for a single valid patent result entry."""
    return st.fixed_dictionaries(
        {
            "patent_number": _nonempty_text,
            "title": _nonempty_text,
            "abstract": _nonempty_text,
        }
    )


def _paper_entry_with_doi() -> st.SearchStrategy[dict]:
    """Strategy for a single valid paper result entry (ArXiv / PubMed)."""
    return st.fixed_dictionaries(
        {
            "doi": _nonempty_text,
            "title": _nonempty_text,
            "abstract": _nonempty_text,
        }
    )


def _scholar_entry() -> st.SearchStrategy[dict]:
    """Strategy for a single valid Google Scholar result entry."""
    return st.fixed_dictionaries(
        {
            "title": _nonempty_text,
            "abstract": _nonempty_text,
            "citation_count": st.integers(min_value=0, max_value=100_000),
            "source_url": _nonempty_text,
        }
    )


def _patent_raw_response() -> st.SearchStrategy[dict]:
    """Strategy for a raw response containing 1+ patent entries."""
    return st.fixed_dictionaries(
        {"results": st.lists(_patent_entry(), min_size=1, max_size=5)}
    )


def _paper_raw_response() -> st.SearchStrategy[dict]:
    """Strategy for a raw response containing 1+ paper entries with DOI."""
    return st.fixed_dictionaries(
        {"results": st.lists(_paper_entry_with_doi(), min_size=1, max_size=5)}
    )


def _scholar_raw_response() -> st.SearchStrategy[dict]:
    """Strategy for a raw response containing 1+ Google Scholar entries."""
    return st.fixed_dictionaries(
        {"results": st.lists(_scholar_entry(), min_size=1, max_size=5)}
    )


# ---------------------------------------------------------------------------
# Property 10: Parser output contains required fields
# Feature: patent-analysis-drafting, Property 10: Parser output contains required fields
# ---------------------------------------------------------------------------


class TestParserOutputRequiredFields:
    """Property 10: Parser output contains required fields.

    For any source parser and any valid raw response, parsed records have
    all required fields present and non-empty.

    **Validates: Requirements 18.1, 18.2, 18.3, 18.4, 18.5**
    """

    @given(raw=_patent_raw_response())
    @settings(max_examples=100)
    def test_depatisnet_required_fields(self, raw: dict) -> None:
        """DEPATISnet parsed patents have patent_number, title, abstract, source non-empty."""
        parser = DEPATISnetParser()
        records = parser.parse_patent(raw)

        assert len(records) == len(raw["results"])
        for record in records:
            assert isinstance(record, PatentRecord)
            assert record.patent_number and len(record.patent_number) > 0
            assert record.title and len(record.title) > 0
            assert record.abstract and len(record.abstract) > 0
            assert record.source and len(record.source) > 0

    @given(raw=_patent_raw_response())
    @settings(max_examples=100)
    def test_google_patents_required_fields(self, raw: dict) -> None:
        """Google Patents parsed patents have patent_number, title, abstract, source non-empty."""
        parser = GooglePatentsParser()
        records = parser.parse_patent(raw)

        assert len(records) == len(raw["results"])
        for record in records:
            assert isinstance(record, PatentRecord)
            assert record.patent_number and len(record.patent_number) > 0
            assert record.title and len(record.title) > 0
            assert record.abstract and len(record.abstract) > 0
            assert record.source and len(record.source) > 0

    @given(raw=_paper_raw_response())
    @settings(max_examples=100)
    def test_arxiv_required_fields(self, raw: dict) -> None:
        """ArXiv parsed papers have doi, title, abstract, source non-empty."""
        parser = ArXivParser()
        records = parser.parse_paper(raw)

        assert len(records) == len(raw["results"])
        for record in records:
            assert isinstance(record, ScientificPaperRecord)
            assert record.doi and len(record.doi) > 0
            assert record.title and len(record.title) > 0
            assert record.abstract and len(record.abstract) > 0
            assert record.source and len(record.source) > 0

    @given(raw=_paper_raw_response())
    @settings(max_examples=100)
    def test_pubmed_required_fields(self, raw: dict) -> None:
        """PubMed parsed papers have doi, title, abstract, source non-empty."""
        parser = PubMedParser()
        records = parser.parse_paper(raw)

        assert len(records) == len(raw["results"])
        for record in records:
            assert isinstance(record, ScientificPaperRecord)
            assert record.doi and len(record.doi) > 0
            assert record.title and len(record.title) > 0
            assert record.abstract and len(record.abstract) > 0
            assert record.source and len(record.source) > 0

    @given(raw=_scholar_raw_response())
    @settings(max_examples=100)
    def test_google_scholar_required_fields(self, raw: dict) -> None:
        """Google Scholar parsed papers have title, abstract, source non-empty."""
        parser = GoogleScholarParser()
        records = parser.parse_paper(raw)

        assert len(records) == len(raw["results"])
        for record in records:
            assert isinstance(record, ScientificPaperRecord)
            assert record.title and len(record.title) > 0
            assert record.abstract and len(record.abstract) > 0
            assert record.source and len(record.source) > 0


# ---------------------------------------------------------------------------
# Property 11: Structured record serialization round-trip
# Feature: patent-analysis-drafting, Property 11: Structured record serialization round-trip
# ---------------------------------------------------------------------------


def _patent_record_strategy() -> st.SearchStrategy[PatentRecord]:
    """Strategy for generating valid PatentRecord instances."""
    return st.builds(
        PatentRecord,
        patent_number=_nonempty_text,
        title=_nonempty_text,
        abstract=_nonempty_text,
        source=_patent_source,
        # Keep optional fields as None to avoid embedding bytes complexity
        embedding=st.none(),
    )


def _paper_record_strategy() -> st.SearchStrategy[ScientificPaperRecord]:
    """Strategy for generating valid ScientificPaperRecord instances."""
    return st.builds(
        ScientificPaperRecord,
        doi=_nonempty_text,
        title=_nonempty_text,
        abstract=_nonempty_text,
        source=_patent_source,
        # Keep optional fields as None to avoid embedding bytes complexity
        embedding=st.none(),
    )


class TestSerializationRoundTrip:
    """Property 11: Structured record serialization round-trip.

    For any valid PatentRecord or ScientificPaperRecord, serialize then
    deserialize produces an equal record.

    **Validates: Requirements 18.6**
    """

    @given(record=_patent_record_strategy())
    @settings(max_examples=100)
    def test_patent_record_round_trip(self, record: PatentRecord) -> None:
        """Serialize then deserialize a PatentRecord produces an equal record."""
        parser: BaseSourceParser = DEPATISnetParser()

        serialized = parser.serialize(record)
        deserialized = parser.deserialize_patent(serialized)

        # Compare all fields except embedding (set to None in strategy)
        assert deserialized.patent_number == record.patent_number
        assert deserialized.title == record.title
        assert deserialized.abstract == record.abstract
        assert deserialized.source == record.source
        assert deserialized.full_text == record.full_text
        assert deserialized.claims == record.claims
        assert deserialized.pdf_path == record.pdf_path
        assert deserialized.discovered_date == record.discovered_date
        assert deserialized.id == record.id
        assert deserialized.session_id == record.session_id

    @given(record=_paper_record_strategy())
    @settings(max_examples=100)
    def test_paper_record_round_trip(self, record: ScientificPaperRecord) -> None:
        """Serialize then deserialize a ScientificPaperRecord produces an equal record."""
        parser: BaseSourceParser = ArXivParser()

        serialized = parser.serialize(record)
        deserialized = parser.deserialize_paper(serialized)

        # Compare all fields except embedding (set to None in strategy)
        assert deserialized.doi == record.doi
        assert deserialized.title == record.title
        assert deserialized.abstract == record.abstract
        assert deserialized.source == record.source
        assert deserialized.full_text == record.full_text
        assert deserialized.pdf_path == record.pdf_path
        assert deserialized.discovered_date == record.discovered_date
        assert deserialized.id == record.id
        assert deserialized.session_id == record.session_id
