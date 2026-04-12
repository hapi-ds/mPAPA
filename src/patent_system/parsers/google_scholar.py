"""Google Scholar result parser.

Parses raw API responses from Google Scholar into structured
ScientificPaperRecord instances. Google Scholar results use the title as
the primary identifier (no DOI), and include citation count and source URL.
Malformed records are logged and skipped.
"""

import logging

from patent_system.db.models import PatentRecord, ScientificPaperRecord
from patent_system.parsers.base import BaseSourceParser

logger = logging.getLogger(__name__)

SOURCE_NAME = "Google Scholar"


class GoogleScholarParser(BaseSourceParser):
    """Parser for Google Scholar search results."""

    def parse_patent(self, raw_response: dict) -> list[PatentRecord]:
        """Google Scholar is a paper source — returns an empty list."""
        return []

    def parse_paper(self, raw_response: dict) -> list[ScientificPaperRecord]:
        """Parse raw Google Scholar response into ScientificPaperRecord instances.

        Args:
            raw_response: Dict with a "results" key containing result dicts,
                each having title, abstract, citation_count, and source_url.
                doi is optional (defaults to empty string if absent).

        Returns:
            List of validated ScientificPaperRecord instances. Malformed
            records are skipped with a warning log.
        """
        records: list[ScientificPaperRecord] = []
        results = raw_response.get("results", [])

        for entry in results:
            try:
                record = ScientificPaperRecord(
                    doi=entry.get("doi", ""),
                    title=entry["title"],
                    abstract=entry["abstract"],
                    full_text=entry.get("full_text"),
                    source=SOURCE_NAME,
                )
                records.append(record)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "Skipping malformed Google Scholar record: %s | raw: %s",
                    exc,
                    entry,
                )

        return records
