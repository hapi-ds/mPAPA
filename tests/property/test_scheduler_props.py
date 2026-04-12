"""Property-based tests for the monitoring scheduler.

Feature: placeholder-to-real-implementation, Property 8: Scheduler per-topic invocation

Validates: Requirements 7.1
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

from patent_system.monitoring.scheduler import MonitoringScheduler

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Sets of unique positive topic IDs (0 to 20 items)
_topic_ids = st.frozensets(st.integers(min_value=1, max_value=10_000), min_size=0, max_size=20)


# ---------------------------------------------------------------------------
# Property 8: Scheduler per-topic invocation
# Feature: placeholder-to-real-implementation, Property 8: Scheduler per-topic invocation
# ---------------------------------------------------------------------------


class TestSchedulerPerTopicInvocation:
    """Property 8: Scheduler per-topic invocation.

    For any set of monitored topic IDs, when ``_run_search_cycle`` executes,
    the Prior Art Search Agent shall be invoked exactly once per topic ID
    in the monitored set.

    **Validates: Requirements 7.1**
    """

    @given(topic_ids=_topic_ids)
    @settings(max_examples=100)
    def test_search_agent_called_once_per_topic(
        self,
        topic_ids: frozenset[int],
    ) -> None:
        """prior_art_search_node is invoked exactly once per monitored topic."""
        scheduler = MonitoringScheduler(interval_hours=24)

        # Enable monitoring for each generated topic ID
        for tid in topic_ids:
            scheduler.enable_monitoring(tid)

        mock_result: dict = {
            "prior_art_results": [],
            "failed_sources": [],
            "current_step": "prior_art_search",
        }

        with patch(
            "patent_system.monitoring.scheduler.prior_art_search_node",
            return_value=mock_result,
        ) as mock_search:
            scheduler._run_search_cycle()

            # Total call count must equal the number of monitored topics
            assert mock_search.call_count == len(topic_ids)

            # Each topic ID must appear exactly once across all calls
            called_topic_ids = {
                call.args[0]["topic_id"] for call in mock_search.call_args_list
            }
            assert called_topic_ids == set(topic_ids)


# ---------------------------------------------------------------------------
# Strategies for Property 9
# ---------------------------------------------------------------------------

# Pairs of (all_topic_ids, failing_subset) where failing_subset ⊆ all_topic_ids
def _topic_ids_with_failing_subset() -> st.SearchStrategy[tuple[frozenset[int], frozenset[int]]]:
    """Generate a set of topic IDs and a subset that should cause failures."""
    return _topic_ids.flatmap(
        lambda ids: st.frozensets(
            st.sampled_from(sorted(ids)) if ids else st.nothing(),
            max_size=len(ids),
        ).map(lambda failing: (ids, failing))
    )


# ---------------------------------------------------------------------------
# Property 9: Scheduler fault tolerance
# Feature: placeholder-to-real-implementation, Property 9: Scheduler fault tolerance
# ---------------------------------------------------------------------------


class TestSchedulerFaultTolerance:
    """Property 9: Scheduler fault tolerance.

    For any set of monitored topic IDs where a subset cause the search agent
    to raise an exception, the scheduler shall still invoke the search agent
    for all remaining topics and shall not terminate the cycle early.

    **Validates: Requirements 7.4**
    """

    @given(data=st.data())
    @settings(max_examples=100)
    def test_scheduler_continues_after_failures(
        self,
        data: st.DataObject,
    ) -> None:
        """Scheduler invokes search for ALL topics even when some raise."""
        all_topic_ids: frozenset[int] = data.draw(_topic_ids.filter(lambda s: len(s) > 0), label="all_topic_ids")
        failing_ids: frozenset[int] = data.draw(
            st.frozensets(
                st.sampled_from(sorted(all_topic_ids)),
                max_size=len(all_topic_ids),
            ),
            label="failing_ids",
        )

        scheduler = MonitoringScheduler(interval_hours=24)

        for tid in all_topic_ids:
            scheduler.enable_monitoring(tid)

        mock_result: dict = {
            "prior_art_results": [],
            "failed_sources": [],
            "current_step": "prior_art_search",
        }

        def _side_effect(state: dict, **kwargs: object) -> dict:
            tid = state["topic_id"]
            if tid in failing_ids:
                raise RuntimeError(f"Simulated failure for topic {tid}")
            return mock_result

        with patch(
            "patent_system.monitoring.scheduler.prior_art_search_node",
            side_effect=_side_effect,
        ) as mock_search:
            scheduler._run_search_cycle()

            # The search agent must be called for EVERY monitored topic,
            # regardless of which ones raised exceptions.
            called_topic_ids = {
                call.args[0]["topic_id"] for call in mock_search.call_args_list
            }
            assert called_topic_ids == set(all_topic_ids), (
                f"Expected calls for {set(all_topic_ids)}, got {called_topic_ids}"
            )

            # Total call count must equal the number of monitored topics
            assert mock_search.call_count == len(all_topic_ids)
