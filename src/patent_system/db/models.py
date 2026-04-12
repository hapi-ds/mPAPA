"""Pydantic data models for the patent system database records.

Defines domain models used across the application for data validation,
serialization, and transfer between layers. Models correspond to the
SQLite tables defined in schema.py, plus structured output types for
agent results.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


class Topic(BaseModel):
    """A user-created workspace grouping research, chat, and drafts."""

    id: int | None = None
    name: str
    created_at: datetime = Field(default_factory=_utc_now)


class ResearchSession(BaseModel):
    """A record of a prior art search query and its results."""

    id: int | None = None
    topic_id: int
    query: str
    search_date: datetime = Field(default_factory=_utc_now)
    status: str = "pending"


class PatentRecord(BaseModel):
    """A patent document retrieved from an external data source."""

    id: int | None = None
    session_id: int | None = None
    patent_number: str
    title: str
    abstract: str
    full_text: str | None = None
    claims: str | None = None
    pdf_path: str | None = None
    source: str
    discovered_date: datetime = Field(default_factory=_utc_now)
    embedding: bytes | None = None


class ScientificPaperRecord(BaseModel):
    """A scientific paper retrieved from an external data source."""

    id: int | None = None
    session_id: int | None = None
    doi: str
    title: str
    abstract: str
    full_text: str | None = None
    pdf_path: str | None = None
    source: str
    discovered_date: datetime = Field(default_factory=_utc_now)
    embedding: bytes | None = None


class ChatMessage(BaseModel):
    """A single message in the AI chat history."""

    id: int | None = None
    topic_id: int
    role: str  # "user" or "assistant"
    message: str
    timestamp: datetime = Field(default_factory=_utc_now)


class InventionDisclosure(BaseModel):
    """Structured output from the invention disclosure interview."""

    technical_problem: str
    novel_features: list[str]
    implementation_details: str
    potential_variations: list[str]


class InventionDisclosureRecord(BaseModel):
    """Persisted invention disclosure for a topic."""

    id: int | None = None
    topic_id: int
    primary_description: str
    search_terms: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utc_now)


class SourcePreference(BaseModel):
    """Per-topic source selection preference."""

    id: int | None = None
    topic_id: int
    source_name: str
    enabled: bool = True


class NoveltyAnalysisResult(BaseModel):
    """Structured output from the novelty analysis agent."""

    novel_aspects: list[str]
    potential_conflicts: list[dict]
    suggested_claim_scope: str
