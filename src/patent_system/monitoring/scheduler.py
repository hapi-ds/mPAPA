"""Background prior art monitoring scheduler.

Provides a ``MonitoringScheduler`` that periodically triggers prior art
searches for topics with active monitoring enabled.  Delegates to the
real ``prior_art_search_node`` and stores results via repository classes.

Requirements: 14.1, 14.2, 14.3, 7.1, 7.2, 7.3, 7.4, 7.5
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from typing import TYPE_CHECKING, Any

from patent_system.agents.prior_art_search import prior_art_search_node
from patent_system.db.models import PatentRecord
from patent_system.db.repository import (
    PatentRepository,
    ResearchSessionRepository,
    TopicRepository,
)

if TYPE_CHECKING:
    from patent_system.rag.engine import RAGEngine

logger = logging.getLogger(__name__)


class MonitoringScheduler:
    """Background scheduler that runs prior art searches at a fixed interval.

    Args:
        interval_hours: Hours between successive search cycles.  Defaults to
            24 as required by Requirement 14.1.
    """

    def __init__(
        self,
        interval_hours: int = 24,
        conn: sqlite3.Connection | None = None,
        rag_engine: RAGEngine | None = None,
    ) -> None:
        self._interval_hours = interval_hours
        self._conn = conn
        self._rag_engine = rag_engine
        self._monitored_topics: set[int] = set()
        self._timer: threading.Timer | None = None
        self._running = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Topic management
    # ------------------------------------------------------------------

    def enable_monitoring(self, topic_id: int) -> None:
        """Enable background monitoring for *topic_id*.

        Requirement 14.1 — topics with active monitoring receive periodic
        prior art searches.
        """
        with self._lock:
            self._monitored_topics.add(topic_id)
        logger.info("Monitoring enabled for topic %s", topic_id)

    def disable_monitoring(self, topic_id: int) -> None:
        """Disable background monitoring for *topic_id*.

        Requirement 14.3 — stopping monitoring for a topic removes it from
        the scheduled search set.
        """
        with self._lock:
            self._monitored_topics.discard(topic_id)
        logger.info("Monitoring disabled for topic %s", topic_id)

    def is_monitoring(self, topic_id: int) -> bool:
        """Return ``True`` if *topic_id* is actively monitored."""
        with self._lock:
            return topic_id in self._monitored_topics

    def get_monitored_topics(self) -> set[int]:
        """Return a snapshot of all currently monitored topic IDs."""
        with self._lock:
            return set(self._monitored_topics)

    # ------------------------------------------------------------------
    # Scheduler lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background scheduler.

        Schedules the first search cycle after ``interval_hours`` and
        repeats until :meth:`stop` is called.
        """
        with self._lock:
            if self._running:
                logger.warning("Scheduler is already running")
                return
            self._running = True
        logger.info(
            "Monitoring scheduler started (interval=%dh)", self._interval_hours
        )
        self._schedule_next()

    def stop(self) -> None:
        """Stop the background scheduler and cancel any pending timer."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        logger.info("Monitoring scheduler stopped")

    @property
    def running(self) -> bool:
        """Whether the scheduler is currently active."""
        with self._lock:
            return self._running

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _schedule_next(self) -> None:
        """Schedule the next search cycle after the configured interval."""
        with self._lock:
            if not self._running:
                return
            interval_seconds = self._interval_hours * 3600
            self._timer = threading.Timer(interval_seconds, self._tick)
            self._timer.daemon = True
            self._timer.start()

    def _tick(self) -> None:
        """Timer callback — run a search cycle then reschedule."""
        self._run_search_cycle()
        self._schedule_next()

    def _run_search_cycle(self) -> None:
        """Execute a prior art search for every monitored topic.

        For each topic: queries the DB for disclosure data, builds a
        minimal ``PatentWorkflowState``, invokes ``prior_art_search_node``,
        stores results via repositories, and indexes in RAG.

        On failure for any single topic the error is logged and the cycle
        continues with the remaining topics (Requirement 7.4).
        """
        with self._lock:
            topics = set(self._monitored_topics)

        if not topics:
            logger.info("Search cycle: no monitored topics — skipping")
            return

        for topic_id in topics:
            try:
                self._search_topic(topic_id)
            except Exception:
                logger.exception(
                    "Search cycle: failed for topic %s, continuing with next",
                    topic_id,
                )

    def _search_topic(self, topic_id: int) -> None:
        """Run a prior art search for a single topic and persist results."""
        logger.info(
            "Search cycle: running prior art search for topic %s", topic_id
        )

        # Build a minimal PatentWorkflowState with disclosure from DB
        disclosure = self._load_disclosure(topic_id)
        state: dict[str, Any] = {
            "topic_id": topic_id,
            "invention_disclosure": disclosure,
            "interview_messages": [],
            "prior_art_results": [],
            "failed_sources": [],
            "novelty_analysis": None,
            "claims_text": "",
            "description_text": "",
            "review_feedback": "",
            "review_approved": False,
            "iteration_count": 0,
            "current_step": "",
        }

        # Run the real prior art search
        result = prior_art_search_node(state, rag_engine=self._rag_engine)

        # Store results in DB if a connection is available
        if self._conn is not None:
            self._store_results(topic_id, result)

        logger.info(
            "Search cycle: topic %s completed — %d results, %d failed sources",
            topic_id,
            len(result.get("prior_art_results", [])),
            len(result.get("failed_sources", [])),
        )

    def _load_disclosure(self, topic_id: int) -> dict | None:
        """Load the stored invention disclosure for a topic from the DB.

        Returns None when no DB connection is available or no disclosure
        data exists for the topic.
        """
        if self._conn is None:
            return None

        # Check that the topic exists
        topic_repo = TopicRepository(self._conn)
        topic = topic_repo.get_by_id(topic_id)
        if topic is None:
            logger.warning("Topic %s not found in database", topic_id)
            return None

        # Use the topic name as a minimal disclosure context
        return {"technical_problem": topic.name, "novel_features": [], "implementation_details": ""}

    def _store_results(self, topic_id: int, result: dict[str, Any]) -> None:
        """Persist prior art search results to the database."""
        prior_art = result.get("prior_art_results", [])
        if not prior_art:
            return

        session_repo = ResearchSessionRepository(self._conn)  # type: ignore[arg-type]
        patent_repo = PatentRepository(self._conn)  # type: ignore[arg-type]

        # Create a research session for this monitoring cycle
        session_id = session_repo.create(topic_id, query="monitoring_cycle")

        for record_dict in prior_art:
            try:
                patent_record = PatentRecord(
                    session_id=session_id,
                    patent_number=record_dict.get("patent_number", record_dict.get("doi", "UNKNOWN")),
                    title=record_dict.get("title", "Untitled"),
                    abstract=record_dict.get("abstract", ""),
                    source=record_dict.get("source", "unknown"),
                )
                patent_repo.create(session_id, patent_record)
            except Exception:
                logger.warning(
                    "Failed to store record for topic %s: %s",
                    topic_id,
                    record_dict.get("title", "unknown"),
                    exc_info=True,
                )
