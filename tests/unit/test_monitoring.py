"""Unit tests for the background prior art monitoring scheduler.

Requirements: 14.1, 14.2, 14.3
"""

import threading
import time

from patent_system.monitoring.scheduler import MonitoringScheduler


class TestMonitoringSchedulerTopicManagement:
    """Tests for enable/disable/query of monitored topics."""

    def test_no_topics_monitored_initially(self) -> None:
        scheduler = MonitoringScheduler()
        assert scheduler.get_monitored_topics() == set()

    def test_enable_monitoring(self) -> None:
        scheduler = MonitoringScheduler()
        scheduler.enable_monitoring(1)
        assert scheduler.is_monitoring(1)
        assert scheduler.get_monitored_topics() == {1}

    def test_disable_monitoring(self) -> None:
        scheduler = MonitoringScheduler()
        scheduler.enable_monitoring(1)
        scheduler.disable_monitoring(1)
        assert not scheduler.is_monitoring(1)
        assert scheduler.get_monitored_topics() == set()

    def test_disable_nonexistent_topic_is_noop(self) -> None:
        scheduler = MonitoringScheduler()
        scheduler.disable_monitoring(999)  # should not raise
        assert scheduler.get_monitored_topics() == set()

    def test_enable_multiple_topics(self) -> None:
        scheduler = MonitoringScheduler()
        scheduler.enable_monitoring(1)
        scheduler.enable_monitoring(2)
        scheduler.enable_monitoring(3)
        assert scheduler.get_monitored_topics() == {1, 2, 3}

    def test_enable_same_topic_twice_is_idempotent(self) -> None:
        scheduler = MonitoringScheduler()
        scheduler.enable_monitoring(1)
        scheduler.enable_monitoring(1)
        assert scheduler.get_monitored_topics() == {1}

    def test_get_monitored_topics_returns_snapshot(self) -> None:
        scheduler = MonitoringScheduler()
        scheduler.enable_monitoring(1)
        snapshot = scheduler.get_monitored_topics()
        scheduler.enable_monitoring(2)
        # The snapshot should not reflect the later addition.
        assert snapshot == {1}


class TestMonitoringSchedulerLifecycle:
    """Tests for start/stop behaviour."""

    def test_not_running_initially(self) -> None:
        scheduler = MonitoringScheduler()
        assert not scheduler.running

    def test_start_sets_running(self) -> None:
        scheduler = MonitoringScheduler(interval_hours=24)
        scheduler.start()
        try:
            assert scheduler.running
        finally:
            scheduler.stop()

    def test_stop_clears_running(self) -> None:
        scheduler = MonitoringScheduler(interval_hours=24)
        scheduler.start()
        scheduler.stop()
        assert not scheduler.running

    def test_stop_when_not_running_is_noop(self) -> None:
        scheduler = MonitoringScheduler()
        scheduler.stop()  # should not raise
        assert not scheduler.running

    def test_double_start_is_safe(self) -> None:
        scheduler = MonitoringScheduler(interval_hours=24)
        scheduler.start()
        scheduler.start()  # second call should be a no-op
        try:
            assert scheduler.running
        finally:
            scheduler.stop()

    def test_configurable_interval(self) -> None:
        scheduler = MonitoringScheduler(interval_hours=12)
        assert scheduler._interval_hours == 12


class TestMonitoringSchedulerSearchCycle:
    """Tests for the placeholder _run_search_cycle method."""

    def test_run_search_cycle_with_no_topics(self, caplog: object) -> None:
        scheduler = MonitoringScheduler()
        scheduler._run_search_cycle()  # should not raise

    def test_run_search_cycle_logs_topics(self) -> None:
        scheduler = MonitoringScheduler()
        scheduler.enable_monitoring(42)
        scheduler._run_search_cycle()  # should complete without error
