"""Background prior art monitoring scheduler.

Provides a ``MonitoringScheduler`` that periodically triggers prior art
searches for topics with active monitoring enabled.  The actual search
execution is a placeholder that logs the action — it will be wired to the
real ``Prior_Art_Search_Agent`` in a later integration step.

Requirements: 14.1, 14.2, 14.3
"""

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class MonitoringScheduler:
    """Background scheduler that runs prior art searches at a fixed interval.

    Args:
        interval_hours: Hours between successive search cycles.  Defaults to
            24 as required by Requirement 14.1.
    """

    def __init__(self, interval_hours: int = 24) -> None:
        self._interval_hours = interval_hours
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

        This is a placeholder implementation that logs the action.  The
        real implementation will delegate to the ``Prior_Art_Search_Agent``
        and store new results in the database (Requirement 14.2).
        """
        with self._lock:
            topics = set(self._monitored_topics)

        if not topics:
            logger.info("Search cycle: no monitored topics — skipping")
            return

        for topic_id in topics:
            logger.info(
                "Search cycle: running prior art search for topic %s",
                topic_id,
            )
            # TODO: delegate to Prior_Art_Search_Agent, store new results,
            # and notify the GUI of new findings (Requirement 14.2).
