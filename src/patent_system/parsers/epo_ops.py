"""EPO Open Patent Services (OPS) result parser.

Parses XML responses from the EPO OPS published-data search API into
structured PatentRecord instances. Malformed records are logged and skipped.
"""

import logging

from patent_system.db.models import PatentRecord, ScientificPaperRecord
from patent_system.parsers.base import BaseSourceParser

logger = logging.getLogger(__name__)

SOURCE_NAME = "EPO OPS"


class EPOOPSParser(BaseSourceParser):
    """Parser for EPO OPS search results."""

    def parse_patent(self, raw_response: dict) -> list[PatentRecord]:
        """Parse raw EPO OPS response into PatentRecord instances.

        Args:
            raw_response: Dict with a "results" key containing result dicts,
                each having patent_number, title, and abstract.

        Returns:
            List of validated PatentRecord instances.
        """
        records: list[PatentRecord] = []
        results = raw_response.get("results", [])

        for entry in results:
            try:
                record = PatentRecord(
                    patent_number=entry["patent_number"],
                    title=entry["title"],
                    abstract=entry.get("abstract", ""),
                    full_text=entry.get("full_text"),
                    source=SOURCE_NAME,
                )
                records.append(record)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "Skipping malformed EPO OPS record: %s | raw: %s",
                    exc,
                    entry,
                )

        return records

    def parse_paper(self, raw_response: dict) -> list[ScientificPaperRecord]:
        """EPO OPS is a patent source — returns an empty list."""
        return []
