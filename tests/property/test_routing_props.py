"""Property-based tests for workflow routing correctness.

Validates: Requirements 3.3, 3.4
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from patent_system.agents.graph import should_revise_or_proceed
from patent_system.agents.state import PatentWorkflowState

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Iteration counts covering both < 3 and >= 3 ranges
_iteration_count = st.integers(min_value=0, max_value=10)


def _build_state(
    review_approved: bool,
    iteration_count: int,
) -> PatentWorkflowState:
    """Build a minimal PatentWorkflowState with the given routing-relevant fields."""
    return PatentWorkflowState(
        topic_id=1,
        invention_disclosure=None,
        interview_messages=[],
        prior_art_results=[],
        failed_sources=[],
        novelty_analysis=None,
        claims_text="",
        description_text="",
        review_feedback="",
        review_approved=review_approved,
        iteration_count=iteration_count,
        current_step="consistency_review",
    )


# ---------------------------------------------------------------------------
# Property 6: Workflow routing correctness
# Feature: placeholder-to-real-implementation, Property 6: Workflow routing correctness
# ---------------------------------------------------------------------------


class TestWorkflowRoutingCorrectness:
    """Property 6: Workflow routing correctness.

    For any PatentWorkflowState, the routing function returns
    "description_drafting" when approved, "claims_drafting" when not
    approved and iteration < 3, "human_review" when not approved and
    iteration >= 3.

    **Validates: Requirements 3.3, 3.4**
    """

    @given(iteration_count=_iteration_count)
    @settings(max_examples=100)
    def test_approved_routes_to_description_drafting(
        self,
        iteration_count: int,
    ) -> None:
        """When review_approved is True, routing returns 'description_drafting' regardless of iteration_count."""
        state = _build_state(review_approved=True, iteration_count=iteration_count)
        assert should_revise_or_proceed(state) == "description_drafting"

    @given(iteration_count=st.integers(min_value=0, max_value=2))
    @settings(max_examples=100)
    def test_not_approved_low_iteration_routes_to_claims_drafting(
        self,
        iteration_count: int,
    ) -> None:
        """When review_approved is False and iteration_count < 3, routing returns 'claims_drafting'."""
        state = _build_state(review_approved=False, iteration_count=iteration_count)
        assert should_revise_or_proceed(state) == "claims_drafting"

    @given(iteration_count=st.integers(min_value=3, max_value=10))
    @settings(max_examples=100)
    def test_not_approved_high_iteration_routes_to_human_review(
        self,
        iteration_count: int,
    ) -> None:
        """When review_approved is False and iteration_count >= 3, routing returns 'human_review'."""
        state = _build_state(review_approved=False, iteration_count=iteration_count)
        assert should_revise_or_proceed(state) == "human_review"
