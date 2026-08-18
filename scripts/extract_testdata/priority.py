"""Priority list of testdata documents for extraction.

These are the most valuable documents for testing the Patent Reviewer module:
- Office actions (rejections, search reports) = ground truth for what a reviewer should find
- Applicant replies = show how issues were resolved
- Claim amendments = show evolution of claims

Ordered by priority: small docs first (faster), most informative for testing.
"""

# Priority 1: Office actions + search reports (examiner's findings = ground truth)
PRIORITY_1_PATTERNS = [
    # EPO search reports and communications
    "*search*",
    "*communication*patentability*",
    # USPTO rejections
    "*non final rejection*",
    "*non-final rejection*",
    "*restriction*election*",
    "*notice of allowance*",
    # TC (Telephone Conference) / Interview summaries
    "*TC*",
]

# Priority 2: Applicant replies (how issues were addressed)
PRIORITY_2_PATTERNS = [
    "*reply*",
    "*Reply*",
    "*replay*",  # typo in filenames
    "*response*",
]

# Priority 3: Claim amendments (show claim evolution)
PRIORITY_3_PATTERNS = [
    "*amended*claim*",
    "*claims after*",
    "*Claims after*",
]

# Priority 4: Original patents (reference, but large)
PRIORITY_4_PATTERNS = [
    "*patent pending*",
    "*accepted*",
]

# Skip: Very large original documents (>10MB) unless explicitly requested
SKIP_LARGE_THRESHOLD_MB = 10.0


def get_priority_files():
    """Return testdata PDFs ordered by extraction priority."""
    from pathlib import Path
    import fnmatch

    testdata = Path(__file__).parent.parent.parent / "testdata"
    all_pdfs = sorted(testdata.rglob("*.pdf"))

    priority_1 = []
    priority_2 = []
    priority_3 = []
    priority_4 = []
    remaining = []

    for pdf in all_pdfs:
        name = pdf.name.lower()
        size_mb = pdf.stat().st_size / (1024 * 1024)

        matched = False
        for pattern in PRIORITY_1_PATTERNS:
            if fnmatch.fnmatch(name, pattern.lower()):
                priority_1.append(pdf)
                matched = True
                break
        if matched:
            continue

        for pattern in PRIORITY_2_PATTERNS:
            if fnmatch.fnmatch(name, pattern.lower()):
                priority_2.append(pdf)
                matched = True
                break
        if matched:
            continue

        for pattern in PRIORITY_3_PATTERNS:
            if fnmatch.fnmatch(name, pattern.lower()):
                priority_3.append(pdf)
                matched = True
                break
        if matched:
            continue

        for pattern in PRIORITY_4_PATTERNS:
            if fnmatch.fnmatch(name, pattern.lower()):
                priority_4.append(pdf)
                matched = True
                break
        if matched:
            continue

        remaining.append(pdf)

    # Sort each group by file size (smallest first = fastest extraction)
    for group in [priority_1, priority_2, priority_3, priority_4, remaining]:
        group.sort(key=lambda p: p.stat().st_size)

    return {
        "priority_1_examiner": priority_1,
        "priority_2_replies": priority_2,
        "priority_3_amendments": priority_3,
        "priority_4_patents": priority_4,
        "remaining": remaining,
    }


if __name__ == "__main__":
    groups = get_priority_files()
    for label, files in groups.items():
        print(f"\n{'='*60}")
        print(f"{label.upper()} ({len(files)} files)")
        print(f"{'='*60}")
        for f in files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {size_mb:5.1f} MB  {f.name[:70]}")
