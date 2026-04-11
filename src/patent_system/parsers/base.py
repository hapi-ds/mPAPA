"""Abstract base class for external data source result parsers.

Defines the interface that all source-specific parsers (DEPATISnet,
Google Patents, Google Scholar, ArXiv, PubMed) must implement.
"""

from abc import ABC, abstractmethod

from patent_system.db.models import PatentRecord, ScientificPaperRecord


class BaseSourceParser(ABC):
    """Abstract interface for data source result parsers."""

    @abstractmethod
    def parse_patent(self, raw_response: dict) -> list[PatentRecord]:
        """Parse raw API response into structured patent records.

        Args:
            raw_response: Raw dictionary from the external data source API.

        Returns:
            List of validated PatentRecord instances.
        """
        ...

    @abstractmethod
    def parse_paper(self, raw_response: dict) -> list[ScientificPaperRecord]:
        """Parse raw API response into structured scientific paper records.

        Args:
            raw_response: Raw dictionary from the external data source API.

        Returns:
            List of validated ScientificPaperRecord instances.
        """
        ...

    def serialize(self, record: PatentRecord | ScientificPaperRecord) -> dict:
        """Serialize a structured record to a dictionary.

        Args:
            record: A PatentRecord or ScientificPaperRecord instance.

        Returns:
            Dictionary representation of the record.
        """
        return record.model_dump()

    def deserialize_patent(self, data: dict) -> PatentRecord:
        """Deserialize a dictionary back to a PatentRecord.

        Args:
            data: Dictionary containing patent record fields.

        Returns:
            Validated PatentRecord instance.
        """
        return PatentRecord.model_validate(data)

    def deserialize_paper(self, data: dict) -> ScientificPaperRecord:
        """Deserialize a dictionary back to a ScientificPaperRecord.

        Args:
            data: Dictionary containing scientific paper record fields.

        Returns:
            Validated ScientificPaperRecord instance.
        """
        return ScientificPaperRecord.model_validate(data)
