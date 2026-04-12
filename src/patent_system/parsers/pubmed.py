"""PubMed result parser.

Parses raw API responses from PubMed into structured ScientificPaperRecord
instances. Malformed records are logged and skipped.
"""

import logging

from patent_system.db.models import PatentRecord, ScientificPaperRecord
from patent_system.parsers.base import BaseSourceParser

logger = logging.getLogger(__name__)

SOURCE_NAME = "PubMed"


class PubMedParser(BaseSourceParser):
    """Parser for PubMed search results."""

    def parse_patent(self, raw_response: dict) -> list[PatentRecord]:
        """PubMed is a paper source — returns an empty list."""
        return []

    def parse_paper(self, raw_response: dict) -> list[ScientificPaperRecord]:
        """Parse raw PubMed response into ScientificPaperRecord instances.

        Args:
            raw_response: Dict with a "results" key containing result dicts,
                each having doi, title, and abstract.

        Returns:
            List of validated ScientificPaperRecord instances. Malformed
            records are skipped with a warning log.
        """
        records: list[ScientificPaperRecord] = []
        results = raw_response.get("results", [])

        for entry in results:
            try:
                record = ScientificPaperRecord(
                    doi=entry["doi"],
                    title=entry["title"],
                    abstract=entry["abstract"],
                    full_text=entry.get("full_text"),
                    source=SOURCE_NAME,
                )
                records.append(record)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "Skipping malformed PubMed record: %s | raw: %s",
                    exc,
                    entry,
                )

        return records
