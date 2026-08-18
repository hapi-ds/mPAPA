"""Persistence integration for the patent reviewer module.

Bridges ReviewEngine's ReviewReport output with the SQLite repository
layer — call save_report() after a review completes to persist the
session and all findings.
"""

import logging
import sqlite3

from patent_system.reviewer.engine import ReviewReport
from patent_system.reviewer.repository import (
    ReviewFindingRepository,
    ReviewSessionRepository,
)

logger = logging.getLogger(__name__)


def save_report(
    conn: sqlite3.Connection,
    report: ReviewReport,
    patent_text: str,
    topic_id: int | None = None,
) -> int:
    """Persist a completed ReviewReport to the database.

    Creates a review session marked 'completed' and bulk-inserts all
    findings. This is the single integration point between ReviewEngine
    output and the DB persistence layer.

    Args:
        conn: Active SQLite connection (schema must be initialized).
        report: The completed ReviewReport from ReviewEngine.review().
        patent_text: The full patent text that was reviewed.
        topic_id: Optional topic FK for linking to an existing topic.
            Pass None for standalone reviews.

    Returns:
        The created review session row ID.

    Raises:
        sqlite3.Error: On database failure (rolls back on finding insert
            failure but the session row may persist).
    """
    session_repo = ReviewSessionRepository(conn)
    finding_repo = ReviewFindingRepository(conn)

    # Create the session
    session_id = session_repo.create_session(
        patent_text=patent_text,
        card_ids=report.card_ids_used,
        jurisdiction=report.jurisdiction,
        topic_id=topic_id,
    )

    # Mark as running, then save findings, then mark completed
    session_repo.update_status(session_id, "running")

    # Convert ReviewFinding models to dicts for bulk insert
    finding_dicts: list[dict] = []
    for finding in report.findings:
        # Determine card_id: use first card_id from report if available
        # (each finding's rule_id prefix often indicates its card, but
        # the RuleCard system doesn't embed card_id in ReviewFinding —
        # use the first card for single-card reviews, or derive from rule_id)
        card_id = _infer_card_id(finding.rule_id, report.card_ids_used)
        finding_dicts.append({
            "card_id": card_id,
            "rule_id": finding.rule_id,
            "finding": finding.finding,
            "severity": finding.severity,
            "location": finding.location,
            "suggestion": finding.suggestion,
            "compliant": finding.compliant,
            "reference": finding.reference,
        })

    if finding_dicts:
        finding_repo.save_findings(session_id, finding_dicts)

    session_repo.update_status(session_id, "completed")

    logger.info(
        "Saved review report: session_id=%d, findings=%d, jurisdiction=%s",
        session_id,
        len(finding_dicts),
        report.jurisdiction,
    )

    return session_id


def _infer_card_id(rule_id: str, card_ids_used: list[str]) -> str:
    """Best-effort inference of which card a finding belongs to.

    For single-card reviews, returns the only card_id. For multi-card
    reviews, attempts to match the rule_id prefix against card_ids.
    Falls back to the first card_id if no match is found.

    Args:
        rule_id: The rule_id from a finding (e.g. "EPO-F-IV-2.1").
        card_ids_used: List of card IDs used in the review session.

    Returns:
        The best-matching card_id string.
    """
    if len(card_ids_used) == 1:
        return card_ids_used[0]

    # Try matching rule_id prefix to a card_id
    rule_lower = rule_id.lower()
    for card_id in card_ids_used:
        # Card IDs like "epo_claim_structure" → check if rule starts with "epo"
        prefix = card_id.split("_")[0]
        if rule_lower.startswith(prefix):
            return card_id

    # Fallback to first card
    return card_ids_used[0] if card_ids_used else "unknown"
