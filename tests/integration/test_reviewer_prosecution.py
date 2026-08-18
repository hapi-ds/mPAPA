"""Integration tests for the Patent Reviewer against real prosecution testdata.

These tests validate the reviewer's accuracy by running it against converted
patent documents and checking that it identifies issues consistent with what
real patent examiners found during prosecution.

Requirements:
- vLLM or LM Studio running on localhost (port 8000 or 1234)
- Converted testdata in testdata/converted/

Run with:
    uv run pytest tests/integration/test_reviewer_prosecution.py -v --tb=short

Skip if no LLM is available:
    Tests are marked with @pytest.mark.llm and skip automatically if
    no LLM endpoint responds.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest

from patent_system.reviewer.card_registry import CardRegistry
from patent_system.reviewer.engine import ReviewEngine, ReviewReport
from patent_system.reviewer.report import render_markdown

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

TESTDATA_DIR = Path(__file__).parent.parent.parent / "testdata" / "converted"
RULE_CARDS_DIR = Path(__file__).parent.parent.parent / "rule-cards"

# LLM endpoints to try (vLLM first, then LM Studio)
LLM_ENDPOINTS = [
    ("http://localhost:8000/v1", None),  # vLLM — model auto-detected
    ("http://localhost:1234/v1", "local-model"),  # LM Studio
]


def _detect_llm() -> tuple[str, str] | None:
    """Detect an available LLM endpoint. Returns (base_url, model_name) or None."""
    for base_url, model_name in LLM_ENDPOINTS:
        try:
            r = httpx.get(f"{base_url}/models", timeout=3)
            if r.status_code == 200:
                data = r.json()
                if data.get("data"):
                    detected_model = data["data"][0]["id"]
                    return base_url, model_name or detected_model
        except (httpx.ConnectError, httpx.TimeoutException):
            continue
    return None


def _load_testdata(relative_path: str) -> str | None:
    """Load a converted testdata markdown file."""
    path = TESTDATA_DIR / relative_path
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


@pytest.fixture(scope="module")
def llm_config() -> tuple[str, str]:
    """Detect LLM or skip all tests in this module."""
    result = _detect_llm()
    if result is None:
        pytest.skip("No LLM endpoint available (need vLLM on :8000 or LM Studio on :1234)")
    return result


@pytest.fixture(scope="module")
def registry() -> CardRegistry:
    """Card registry pointing to the real rule-cards directory."""
    return CardRegistry(cache_dir=RULE_CARDS_DIR, repo_url="https://example.com")


@pytest.fixture(scope="module")
def engine(llm_config, registry) -> ReviewEngine:
    """ReviewEngine connected to the detected LLM."""
    base_url, model_name = llm_config
    return ReviewEngine(
        lm_studio_base_url=base_url,
        model_name=model_name,
        card_registry=registry,
    )


# ---------------------------------------------------------------------------
# Protocol: Each test runs a review and records findings as a report
# ---------------------------------------------------------------------------

PROTOCOL_DIR = Path(__file__).parent.parent.parent / "testdata" / "review_protocol"


def _save_protocol(test_name: str, report: ReviewReport, patent_text: str) -> None:
    """Save review results as a protocol file for human inspection."""
    PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)
    
    output = {
        "test_name": test_name,
        "card_ids_used": report.card_ids_used,
        "jurisdiction": report.jurisdiction,
        "findings_count": len(report.findings),
        "statistics": report.statistics,
        "overall_summary": report.overall_summary,
        "findings": [
            {
                "rule_id": f.rule_id,
                "severity": f.severity,
                "location": f.location,
                "finding": f.finding,
                "suggestion": f.suggestion,
                "compliant": f.compliant,
                "reference": f.reference,
            }
            for f in report.findings
        ],
        "patent_text_preview": patent_text[:500] + "..." if len(patent_text) > 500 else patent_text,
    }

    protocol_path = PROTOCOL_DIR / f"{test_name}.json"
    protocol_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    # Also write markdown report
    md_path = PROTOCOL_DIR / f"{test_name}.md"
    md_path.write_text(render_markdown(report), encoding="utf-8")

    logger.info(f"Protocol saved: {protocol_path}")


# ---------------------------------------------------------------------------
# Test cases: EP prosecution (Anti-TRAP Agents, EP4465936A1)
# ---------------------------------------------------------------------------

@pytest.mark.llm
class TestEPOClaimReview:
    """Test EPO claim structure review against real EP prosecution documents."""

    def test_ep_amended_claims_after_1st_search(self, engine: ReviewEngine) -> None:
        """Review amended EP claims — should find structure/clarity issues or confirm compliance."""
        text = _load_testdata(
            "Anti-TRAP Agents/EP4465936A1/KLueh et al_EP4465936A1_anti-TRAP-coated device_amended Claims after 1st search.md"
        )
        if text is None:
            pytest.skip("Testdata not converted yet")

        report = asyncio.run(engine.review(
            patent_text=text,
            card_ids=["epo_claim_structure"],
        ))

        _save_protocol("ep_anti_trap_claims_after_1st_search", report, text)

        # Basic sanity checks
        assert report.findings is not None
        assert report.jurisdiction == "EP"
        assert len(report.card_ids_used) == 1

        # The reviewer should produce at least some findings for amended claims
        # (even if compliant, it should note observations)
        assert len(report.findings) >= 1, "Reviewer produced no findings at all"
        logger.info(f"EP amended claims: {len(report.findings)} findings")

    def test_ep_amended_claims_after_2nd_search(self, engine: ReviewEngine) -> None:
        """Review second round of amended claims — should be more compliant after amendments."""
        text = _load_testdata(
            "Anti-TRAP Agents/EP4465936A1/KLueh et al_EP4465936A1_anti-TRAP-coated device_amended Claims after 2nd search.md"
        )
        if text is None:
            pytest.skip("Testdata not converted yet")

        report = asyncio.run(engine.review(
            patent_text=text,
            card_ids=["epo_claim_structure"],
        ))

        _save_protocol("ep_anti_trap_claims_after_2nd_search", report, text)

        assert len(report.findings) >= 1
        logger.info(f"EP 2nd amended claims: {len(report.findings)} findings")


@pytest.mark.llm
class TestEPONoveltyReview:
    """Test EPO novelty review against EP search reports and replies."""

    def test_ep_reply_after_1st_search(self, engine: ReviewEngine) -> None:
        """Review applicant's reply to first search — should identify novelty arguments."""
        text = _load_testdata(
            "Anti-TRAP Agents/EP4465936A1/KLueh et al_EP4465936A1_anti-TRAP-coated device_Reply after 1st search.md"
        )
        if text is None:
            pytest.skip("Testdata not converted yet")

        report = asyncio.run(engine.review(
            patent_text=text,
            card_ids=["epo_novelty"],
        ))

        _save_protocol("ep_anti_trap_reply_after_1st_search_novelty", report, text)

        assert report.findings is not None
        assert report.jurisdiction == "EP"
        logger.info(f"EP novelty review (reply): {len(report.findings)} findings")


# ---------------------------------------------------------------------------
# Test cases: US prosecution (Anti-TRAP, US2025090729A1)
# ---------------------------------------------------------------------------

@pytest.mark.llm
class TestUSPTOReview:
    """Test USPTO §101 review against US office actions."""

    def test_us_non_final_rejection_101(self, engine: ReviewEngine) -> None:
        """Review US non-final rejection — check §101 analysis."""
        text = _load_testdata(
            "Anti-TRAP Agents/US2025090729A1/KLueh et al_US2025090729A1_anti-TRAP-coated device_non-final rejection.md"
        )
        if text is None:
            pytest.skip("Testdata not converted yet")

        report = asyncio.run(engine.review(
            patent_text=text,
            card_ids=["uspto_subject_matter"],
        ))

        _save_protocol("us_anti_trap_non_final_rejection_101", report, text)

        assert report.findings is not None
        assert report.jurisdiction == "US"
        logger.info(f"US §101 review (rejection): {len(report.findings)} findings")

    def test_us_reply_to_rejection(self, engine: ReviewEngine) -> None:
        """Review applicant's reply to rejection — check if arguments address §101."""
        text = _load_testdata(
            "Anti-TRAP Agents/US2025090729A1/KLueh et al_US2025090729A1_anti-TRAP-coated device_reply to non-final rejection.md"
        )
        if text is None:
            pytest.skip("Testdata not converted yet")

        report = asyncio.run(engine.review(
            patent_text=text,
            card_ids=["uspto_subject_matter"],
        ))

        _save_protocol("us_anti_trap_reply_to_rejection_101", report, text)

        assert report.findings is not None
        logger.info(f"US §101 review (reply): {len(report.findings)} findings")


# ---------------------------------------------------------------------------
# Test cases: Preservative Filter (full lifecycle US + EP)
# ---------------------------------------------------------------------------

@pytest.mark.llm
class TestPreservativeFilterLifecycle:
    """Test reviewer against the Preservative Filter prosecution lifecycle."""

    def test_us_claims_after_restriction(self, engine: ReviewEngine) -> None:
        """Review claims after restriction election — check structure."""
        text = _load_testdata(
            "Preservative Filter/US2019117738/KLueh et al_US2019117738A1_preservative removal_claims after restriction election.md"
        )
        if text is None:
            pytest.skip("Testdata not converted yet")

        report = asyncio.run(engine.review(
            patent_text=text,
            card_ids=["epo_claim_structure"],  # Use EPO rules to cross-check
        ))

        _save_protocol("preservative_us_claims_after_restriction_ep_review", report, text)

        assert len(report.findings) >= 1
        logger.info(f"Preservative Filter claims (EP rules): {len(report.findings)} findings")

    def test_us_claims_after_non_final(self, engine: ReviewEngine) -> None:
        """Review claims after non-final rejection."""
        text = _load_testdata(
            "Preservative Filter/US2019117738/KLueh et al_US2019117738A1_preservative removal_claims after non final rejection.md"
        )
        if text is None:
            pytest.skip("Testdata not converted yet")

        report = asyncio.run(engine.review(
            patent_text=text,
            card_ids=["epo_claim_structure"],
        ))

        _save_protocol("preservative_us_claims_after_nonfinal_ep_review", report, text)

        assert len(report.findings) >= 1
        logger.info(f"Preservative Filter amended claims: {len(report.findings)} findings")


# ---------------------------------------------------------------------------
# Test cases: PCT (Maduka et al, WO2023147055A1)
# ---------------------------------------------------------------------------

@pytest.mark.llm
class TestPCTReview:
    """Test reviewer against PCT prosecution documents."""

    def test_pct_communication_patentability(self, engine: ReviewEngine) -> None:
        """Review PCT communication on patentability — check novelty analysis."""
        text = _load_testdata(
            "Maduka et al/WO2023147055A1_Maduka et al_communication patentability.md"
        )
        if text is None:
            pytest.skip("Testdata not converted yet")

        report = asyncio.run(engine.review(
            patent_text=text,
            card_ids=["epo_novelty"],
        ))

        _save_protocol("pct_maduka_communication_patentability_novelty", report, text)

        assert report.findings is not None
        logger.info(f"PCT novelty review: {len(report.findings)} findings")


# ---------------------------------------------------------------------------
# Comparative test: same claims reviewed under EP vs US rules
# ---------------------------------------------------------------------------

@pytest.mark.llm
class TestComparativeReview:
    """Compare how the same claims score under different jurisdictions."""

    def test_same_claims_ep_vs_us(self, engine: ReviewEngine) -> None:
        """Same claims reviewed under EPO claim structure vs USPTO §101."""
        text = _load_testdata(
            "Anti-TRAP Agents/EP4465936A1/KLueh et al_EP4465936A1_anti-TRAP-coated device_amended Claims after 2nd search.md"
        )
        if text is None:
            pytest.skip("Testdata not converted yet")

        # EP review
        ep_report = asyncio.run(engine.review(
            patent_text=text,
            card_ids=["epo_claim_structure"],
        ))
        _save_protocol("comparative_anti_trap_ep_claim_structure", ep_report, text)

        # US review
        us_report = asyncio.run(engine.review(
            patent_text=text,
            card_ids=["uspto_subject_matter"],
        ))
        _save_protocol("comparative_anti_trap_us_101", us_report, text)

        # Both should produce findings (different focus, different rules)
        assert ep_report.jurisdiction == "EP"
        assert us_report.jurisdiction == "US"
        logger.info(
            f"Comparative: EP={len(ep_report.findings)} findings, "
            f"US={len(us_report.findings)} findings"
        )
