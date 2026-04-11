"""Unit tests for CitationGraphIndex."""

from patent_system.rag.property_graph import CitationGraphIndex


class TestCitationGraphIndex:
    """Tests for the citation graph index."""

    def test_empty_graph_returns_empty_lists(self) -> None:
        graph = CitationGraphIndex()
        assert graph.get_citations("doc1") == []
        assert graph.get_cited_by("doc1") == []

    def test_add_citation_and_get_citations(self) -> None:
        graph = CitationGraphIndex()
        graph.add_citation("A", "B")
        citations = graph.get_citations("A")
        assert len(citations) == 1
        assert citations[0]["document_id"] == "B"

    def test_add_citation_and_get_cited_by(self) -> None:
        graph = CitationGraphIndex()
        graph.add_citation("A", "B")
        cited_by = graph.get_cited_by("B")
        assert len(cited_by) == 1
        assert cited_by[0]["document_id"] == "A"

    def test_citation_with_metadata(self) -> None:
        graph = CitationGraphIndex()
        graph.add_citation("A", "B", metadata={"section": "intro", "page": 3})
        citations = graph.get_citations("A")
        assert citations[0]["section"] == "intro"
        assert citations[0]["page"] == 3

    def test_multiple_citations_from_same_source(self) -> None:
        graph = CitationGraphIndex()
        graph.add_citation("A", "B")
        graph.add_citation("A", "C")
        citations = graph.get_citations("A")
        assert len(citations) == 2
        ids = {c["document_id"] for c in citations}
        assert ids == {"B", "C"}

    def test_multiple_documents_cite_same_target(self) -> None:
        graph = CitationGraphIndex()
        graph.add_citation("A", "C")
        graph.add_citation("B", "C")
        cited_by = graph.get_cited_by("C")
        assert len(cited_by) == 2
        ids = {c["document_id"] for c in cited_by}
        assert ids == {"A", "B"}

    def test_get_citation_path_direct(self) -> None:
        graph = CitationGraphIndex()
        graph.add_citation("A", "B")
        path = graph.get_citation_path("A", "B")
        assert path == ["A", "B"]

    def test_get_citation_path_transitive(self) -> None:
        graph = CitationGraphIndex()
        graph.add_citation("A", "B")
        graph.add_citation("B", "C")
        path = graph.get_citation_path("A", "C")
        assert path == ["A", "B", "C"]

    def test_get_citation_path_same_node(self) -> None:
        graph = CitationGraphIndex()
        path = graph.get_citation_path("A", "A")
        assert path == ["A"]

    def test_get_citation_path_no_path(self) -> None:
        graph = CitationGraphIndex()
        graph.add_citation("A", "B")
        path = graph.get_citation_path("B", "A")
        assert path is None

    def test_get_citation_path_no_path_disconnected(self) -> None:
        graph = CitationGraphIndex()
        graph.add_citation("A", "B")
        graph.add_citation("C", "D")
        assert graph.get_citation_path("A", "D") is None

    def test_get_citations_returns_copy(self) -> None:
        """Mutating the returned list should not affect the graph."""
        graph = CitationGraphIndex()
        graph.add_citation("A", "B")
        citations = graph.get_citations("A")
        citations.clear()
        assert len(graph.get_citations("A")) == 1
