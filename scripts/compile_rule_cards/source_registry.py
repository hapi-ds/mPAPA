"""Internal source registry — maps (jurisdiction, chapter) → download metadata.

This file contains the actual URLs, source languages, and file paths for each
jurisdiction × chapter combination. It is maintained by the developer (not the user).

HOW TO UPDATE THIS FILE:
─────────────────────────────────────────────────────────────────────────────────
When a patent office publishes a new edition of their guidelines:

1. Check the office website for the new PDF/HTML URL:
   - EPO: https://www.epo.org/en/legal/guidelines-epc → look for "Part X" PDF links
   - USPTO: https://www.uspto.gov/web/offices/pac/mpep/ → chapter PDFs
   - JPO: https://www.jpo.go.jp/system/laws/rule/guideline/patent/tukujitu_kijun/
   - PCT: https://www.wipo.int/en/web/pct-system/texts/ispe/

2. Update the URL and edition in the relevant entry below.

3. Run the downloader to fetch the new version:
   uv run python scripts/compile_rule_cards/download_sources.py

4. The downloader's manifest tracks ETags/Last-Modified, so subsequent runs
   skip unchanged files automatically.

Last verified: 2026-08-19
─────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SourceEntry:
    """A single downloadable guideline source."""

    jurisdiction: str       # EP, US, PCT, JP, CN, KR, IN, DE
    chapter: str            # Matches key in [chapters] of sources.toml
    url: str                # Download URL (empty = manual placement required)
    language: str           # ISO 639-1 code of original language
    card_id: str            # Output rule card ID
    label: str              # Human-readable description
    edition: str            # Edition/version identifier
    part: str = ""          # Guideline part (e.g. "G" for EPO Part G)
    section: str = ""       # Specific sections within the chapter
    filename: str = ""      # Override output filename (auto-derived if empty)
    notes: str = ""         # Any special notes


# =============================================================================
# COMPLETE REGISTRY
# =============================================================================
# Key: (jurisdiction, chapter) — must match sources.toml keys exactly

REGISTRY: dict[tuple[str, str], SourceEntry] = {
    # =========================================================================
    # EPO — Guidelines for Examination (April 2025 edition)
    # HTML per chapter: https://www.epo.org/en/legal/guidelines-epc/2025/X_Y.html
    # Note: PDF URLs (link.epo.org/web/part_X_en.pdf) are unreliable (404 for most parts)
    # HTML is always available and up-to-date
    # =========================================================================
    ("EP", "novelty"): SourceEntry(
        jurisdiction="EP", chapter="novelty",
        url="https://www.epo.org/en/legal/guidelines-epc/2025/g_vi.html",
        language="en", card_id="epo_novelty",
        label="Novelty (Art. 54 EPC)",
        edition="April 2025", part="G", section="Chapter VI",
    ),
    ("EP", "inventive_step"): SourceEntry(
        jurisdiction="EP", chapter="inventive_step",
        url="https://www.epo.org/en/legal/guidelines-epc/2025/g_vii.html",
        language="en", card_id="epo_inventive_step",
        label="Inventive Step (Art. 56 EPC)",
        edition="April 2025", part="G", section="Chapter VII",
    ),
    ("EP", "claim_structure"): SourceEntry(
        jurisdiction="EP", chapter="claim_structure",
        url="https://www.epo.org/en/legal/guidelines-epc/2025/f_iv.html",
        language="en", card_id="epo_claim_structure",
        label="Claims — Art. 84 and Formal Requirements",
        edition="April 2025", part="F", section="Chapter IV",
    ),
    ("EP", "sufficiency"): SourceEntry(
        jurisdiction="EP", chapter="sufficiency",
        url="https://www.epo.org/en/legal/guidelines-epc/2025/f_iii.html",
        language="en", card_id="epo_sufficiency",
        label="Sufficiency of Disclosure (Art. 83)",
        edition="April 2025", part="F", section="Chapter III",
    ),
    ("EP", "unity"): SourceEntry(
        jurisdiction="EP", chapter="unity",
        url="https://www.epo.org/en/legal/guidelines-epc/2025/f_v.html",
        language="en", card_id="epo_unity",
        label="Unity of Invention (Art. 82)",
        edition="April 2025", part="F", section="Chapter V",
    ),
    ("EP", "subject_matter"): SourceEntry(
        jurisdiction="EP", chapter="subject_matter",
        url="https://www.epo.org/en/legal/guidelines-epc/2025/g_ii.html",
        language="en", card_id="epo_patentable_inventions",
        label="Patentable Inventions (Art. 52)",
        edition="April 2025", part="G", section="Chapter II",
    ),
    ("EP", "added_matter"): SourceEntry(
        jurisdiction="EP", chapter="added_matter",
        url="https://www.epo.org/en/legal/guidelines-epc/2025/h_iv.html",
        language="en", card_id="epo_added_matter",
        label="Amendments — Art. 123(2) and (3)",
        edition="April 2025", part="H", section="Chapter IV",
    ),
    ("EP", "priority"): SourceEntry(
        jurisdiction="EP", chapter="priority",
        url="https://www.epo.org/en/legal/guidelines-epc/2025/f_vi.html",
        language="en", card_id="epo_priority",
        label="Priority (Art. 87-89)",
        edition="April 2025", part="F", section="Chapter VI",
    ),
    ("EP", "industrial_application"): SourceEntry(
        jurisdiction="EP", chapter="industrial_application",
        url="https://www.epo.org/en/legal/guidelines-epc/2025/g_iii.html",
        language="en", card_id="epo_industrial_application",
        label="Industrial Application (Art. 57)",
        edition="April 2025", part="G", section="Chapter III",
    ),
    ("EP", "opposition"): SourceEntry(
        jurisdiction="EP", chapter="opposition",
        url="https://www.epo.org/en/legal/guidelines-epc/2025/d_v.html",
        language="en", card_id="epo_opposition",
        label="Substantive Examination of Opposition",
        edition="April 2025", part="D", section="Chapter V",
    ),
    ("EP", "description_format"): SourceEntry(
        jurisdiction="EP", chapter="description_format",
        url="https://www.epo.org/en/legal/guidelines-epc/2025/f_ii.html",
        language="en", card_id="epo_description_format",
        label="Content of a European Patent Application (Description, Drawings, Abstract)",
        edition="April 2025", part="F", section="Chapter II",
    ),
    ("EP", "state_of_art"): SourceEntry(
        jurisdiction="EP", chapter="state_of_art",
        url="https://www.epo.org/en/legal/guidelines-epc/2025/g_iv.html",
        language="en", card_id="epo_state_of_art",
        label="State of the Art (Art. 54(2) — what counts as prior art)",
        edition="April 2025", part="G", section="Chapter IV",
    ),
    ("EP", "non_prejudicial_disclosures"): SourceEntry(
        jurisdiction="EP", chapter="non_prejudicial_disclosures",
        url="https://www.epo.org/en/legal/guidelines-epc/2025/g_v.html",
        language="en", card_id="epo_non_prejudicial_disclosures",
        label="Non-Prejudicial Disclosures / Grace Period (Art. 55)",
        edition="April 2025", part="G", section="Chapter V",
    ),

    # =========================================================================
    # USPTO — MPEP 9th Edition, Rev. 01.2024 (November 2024)
    # Per-chapter PDF: https://www.uspto.gov/web/offices/pac/mpep/mpep-XXXX.pdf
    # =========================================================================
    ("US", "subject_matter"): SourceEntry(
        jurisdiction="US", chapter="subject_matter",
        url="https://www.uspto.gov/web/offices/pac/mpep/mpep-2100.pdf",
        language="en", card_id="uspto_subject_matter",
        label="Patentable Subject Matter (35 U.S.C. §101)",
        edition="MPEP 9th Ed. Rev. 01.2024", section="2104-2106",
    ),
    ("US", "novelty"): SourceEntry(
        jurisdiction="US", chapter="novelty",
        url="https://www.uspto.gov/web/offices/pac/mpep/mpep-2100.pdf",
        language="en", card_id="uspto_novelty",
        label="Novelty / Anticipation (35 U.S.C. §102)",
        edition="MPEP 9th Ed. Rev. 01.2024", section="2131-2138",
    ),
    ("US", "obviousness"): SourceEntry(
        jurisdiction="US", chapter="obviousness",
        url="https://www.uspto.gov/web/offices/pac/mpep/mpep-2100.pdf",
        language="en", card_id="uspto_obviousness",
        label="Obviousness (35 U.S.C. §103)",
        edition="MPEP 9th Ed. Rev. 01.2024", section="2141-2145",
    ),
    ("US", "written_description"): SourceEntry(
        jurisdiction="US", chapter="written_description",
        url="https://www.uspto.gov/web/offices/pac/mpep/mpep-2100.pdf",
        language="en", card_id="uspto_written_description",
        label="Written Description (35 U.S.C. §112(a))",
        edition="MPEP 9th Ed. Rev. 01.2024", section="2161-2164",
    ),
    ("US", "enablement"): SourceEntry(
        jurisdiction="US", chapter="enablement",
        url="https://www.uspto.gov/web/offices/pac/mpep/mpep-2100.pdf",
        language="en", card_id="uspto_enablement",
        label="Enablement (35 U.S.C. §112(a))",
        edition="MPEP 9th Ed. Rev. 01.2024", section="2164",
    ),
    ("US", "definiteness"): SourceEntry(
        jurisdiction="US", chapter="definiteness",
        url="https://www.uspto.gov/web/offices/pac/mpep/mpep-2100.pdf",
        language="en", card_id="uspto_definiteness",
        label="Definiteness (35 U.S.C. §112(b))",
        edition="MPEP 9th Ed. Rev. 01.2024", section="2171-2175",
    ),
    ("US", "claim_structure"): SourceEntry(
        jurisdiction="US", chapter="claim_structure",
        url="https://www.uspto.gov/web/offices/pac/mpep/mpep-2100.pdf",
        language="en", card_id="uspto_claim_structure",
        label="Claim Drafting (35 U.S.C. §112)",
        edition="MPEP 9th Ed. Rev. 01.2024", section="2171-2175",
    ),
    ("US", "double_patenting"): SourceEntry(
        jurisdiction="US", chapter="double_patenting",
        url="https://www.uspto.gov/web/offices/pac/mpep/mpep-0800.pdf",
        language="en", card_id="uspto_double_patenting",
        label="Double Patenting",
        edition="MPEP 9th Ed. Rev. 01.2024", section="804",
    ),
    ("US", "restriction"): SourceEntry(
        jurisdiction="US", chapter="restriction",
        url="https://www.uspto.gov/web/offices/pac/mpep/mpep-0800.pdf",
        language="en", card_id="uspto_restriction",
        label="Restriction / Election (35 U.S.C. §121)",
        edition="MPEP 9th Ed. Rev. 01.2024", section="803-808",
    ),

    # =========================================================================
    # PCT — WIPO ISPE Guidelines (effective January 2026)
    # HTML chapters: https://www.wipo.int/en/web/pct-system/texts/ispe/
    # =========================================================================
    ("PCT", "novelty"): SourceEntry(
        jurisdiction="PCT", chapter="novelty",
        url="https://www.wipo.int/en/web/pct-system/texts/ispe/15_43_51",
        language="en", card_id="pct_novelty",
        label="International Search — Prior Art & Novelty",
        edition="ISPE Guidelines 2026",
    ),
    ("PCT", "unity"): SourceEntry(
        jurisdiction="PCT", chapter="unity",
        url="https://www.wipo.int/en/web/pct-system/texts/ispe/1_01_04",
        language="en", card_id="pct_unity",
        label="Unity of Invention",
        edition="ISPE Guidelines 2026",
    ),
    ("PCT", "inventive_step"): SourceEntry(
        jurisdiction="PCT", chapter="inventive_step",
        url="https://www.wipo.int/en/web/pct-system/texts/ispe/3_16_22",
        language="en", card_id="pct_inventive_step",
        label="Written Opinion — Inventive Step",
        edition="ISPE Guidelines 2026",
    ),

    # =========================================================================
    # JPO — 特許・実用新案審査基準 (JAPANESE — translated during compilation)
    # Source: https://www.jpo.go.jp/system/laws/rule/guideline/patent/tukujitu_kijun/
    # =========================================================================
    ("JP", "novelty"): SourceEntry(
        jurisdiction="JP", chapter="novelty",
        url="https://www.jpo.go.jp/system/laws/rule/guideline/patent/tukujitu_kijun/document/index/03_0200.pdf",
        language="ja", card_id="jpo_novelty",
        label="新規性・進歩性 (Novelty & Inventive Step)",
        edition="2026",
    ),
    ("JP", "inventive_step"): SourceEntry(
        jurisdiction="JP", chapter="inventive_step",
        url="https://www.jpo.go.jp/system/laws/rule/guideline/patent/tukujitu_kijun/document/index/03_0200.pdf",
        language="ja", card_id="jpo_inventive_step",
        label="進歩性 (Inventive Step)",
        edition="2026",
        notes="Same source PDF as novelty — JPO combines them",
    ),
    ("JP", "claim_structure"): SourceEntry(
        jurisdiction="JP", chapter="claim_structure",
        url="https://www.jpo.go.jp/system/laws/rule/guideline/patent/tukujitu_kijun/document/index/03_0100.pdf",
        language="ja", card_id="jpo_claim_structure",
        label="発明の認定 (Claim Interpretation)",
        edition="2026",
    ),
    ("JP", "sufficiency"): SourceEntry(
        jurisdiction="JP", chapter="sufficiency",
        url="https://www.jpo.go.jp/system/laws/rule/guideline/patent/tukujitu_kijun/document/index/01_0000.pdf",
        language="ja", card_id="jpo_sufficiency",
        label="明細書の記載要件 (Description Requirements)",
        edition="2026",
    ),
    ("JP", "industrial_application"): SourceEntry(
        jurisdiction="JP", chapter="industrial_application",
        url="https://www.jpo.go.jp/system/laws/rule/guideline/patent/tukujitu_kijun/document/index/03_0300.pdf",
        language="ja", card_id="jpo_industrial_application",
        label="産業上利用可能性 (Industrial Applicability)",
        edition="2026",
    ),

    # =========================================================================
    # CNIPA — 专利审查指南 (CHINESE — URLs require manual acquisition)
    # =========================================================================
    ("CN", "novelty"): SourceEntry(
        jurisdiction="CN", chapter="novelty",
        url="",  # Manual: place PDF in sources/CN/
        language="zh", card_id="cnipa_novelty",
        label="新颖性 (Novelty)",
        edition="2024", notes="No stable public URL. Place manually.",
    ),
    ("CN", "inventive_step"): SourceEntry(
        jurisdiction="CN", chapter="inventive_step",
        url="",
        language="zh", card_id="cnipa_inventive_step",
        label="创造性 (Inventive Step)",
        edition="2024", notes="No stable public URL. Place manually.",
    ),
    ("CN", "sufficiency"): SourceEntry(
        jurisdiction="CN", chapter="sufficiency",
        url="",
        language="zh", card_id="cnipa_sufficiency",
        label="充分公开 (Sufficiency)",
        edition="2024", notes="No stable public URL. Place manually.",
    ),

    # =========================================================================
    # KIPO — 특허·실용신안 심사기준 (KOREAN — manual acquisition)
    # =========================================================================
    ("KR", "novelty"): SourceEntry(
        jurisdiction="KR", chapter="novelty",
        url="",
        language="ko", card_id="kipo_novelty",
        label="신규성 (Novelty)",
        edition="2024", notes="No stable public URL. Place manually.",
    ),
    ("KR", "inventive_step"): SourceEntry(
        jurisdiction="KR", chapter="inventive_step",
        url="",
        language="ko", card_id="kipo_inventive_step",
        label="진보성 (Inventive Step)",
        edition="2024", notes="No stable public URL. Place manually.",
    ),

    # =========================================================================
    # IPO India — Manual of Patent Office Practice and Procedure (English)
    # =========================================================================
    ("IN", "novelty"): SourceEntry(
        jurisdiction="IN", chapter="novelty",
        url="https://ipindia.gov.in/writereaddata/Portal/IPOGuideManual/1_32_1_manual-of-patent-office-practice-and-procedure.pdf",
        language="en", card_id="ipo_india_novelty",
        label="Novelty",
        edition="2019",
    ),
    ("IN", "inventive_step"): SourceEntry(
        jurisdiction="IN", chapter="inventive_step",
        url="https://ipindia.gov.in/writereaddata/Portal/IPOGuideManual/1_32_1_manual-of-patent-office-practice-and-procedure.pdf",
        language="en", card_id="ipo_india_inventive_step",
        label="Inventive Step",
        edition="2019",
    ),

    # =========================================================================
    # DPMA — Richtlinien für die Prüfung (GERMAN — translated during compilation)
    # =========================================================================
    ("DE", "novelty"): SourceEntry(
        jurisdiction="DE", chapter="novelty",
        url="https://www.dpma.de/docs/service/formulare/allgemein/p2796.pdf",
        language="de", card_id="dpma_novelty",
        label="Neuheit (Novelty)",
        edition="2024",
    ),
    ("DE", "inventive_step"): SourceEntry(
        jurisdiction="DE", chapter="inventive_step",
        url="https://www.dpma.de/docs/service/formulare/allgemein/p2796.pdf",
        language="de", card_id="dpma_inventive_step",
        label="Erfinderische Tätigkeit (Inventive Step)",
        edition="2024",
    ),
    ("DE", "sufficiency"): SourceEntry(
        jurisdiction="DE", chapter="sufficiency",
        url="https://www.dpma.de/docs/service/formulare/allgemein/p2796.pdf",
        language="de", card_id="dpma_sufficiency",
        label="Offenbarung (Sufficiency of Disclosure)",
        edition="2024",
    ),
}


def get_sources_for_config(
    enabled_jurisdictions: list[str],
    enabled_chapters: list[str],
) -> list[SourceEntry]:
    """Look up all sources matching enabled jurisdictions × enabled chapters.

    Args:
        enabled_jurisdictions: List of jurisdiction codes (EP, US, JP, ...).
        enabled_chapters: List of chapter keys (novelty, inventive_step, ...).

    Returns:
        List of SourceEntry objects to download/compile.
    """
    results = []
    seen_urls: set[str] = set()  # Deduplicate (e.g. EP novelty + inventive_step → same PDF)

    for jurisdiction in enabled_jurisdictions:
        for chapter in enabled_chapters:
            key = (jurisdiction.upper(), chapter)
            entry = REGISTRY.get(key)
            if entry is None:
                continue  # This combo doesn't exist (e.g. US + added_matter)
            if not entry.url:
                continue  # Manual placement required
            # Deduplicate by URL (some chapters share the same PDF)
            dedup_key = f"{entry.url}|{entry.card_id}"
            if dedup_key in seen_urls:
                continue
            seen_urls.add(dedup_key)
            results.append(entry)

    return results
