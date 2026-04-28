"""Unified vectorization text preparation for embedding generation and RAG indexing.

Provides a single function that constructs and truncates text consistently
across all code paths: RAG engine indexing, embedding generation in the
research panel, and local document upload flows.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
"""


def prepare_vectorization_text(
    title: str,
    abstract: str,
    full_text: str | None = None,
    max_chars: int = 4000,
) -> str:
    """Construct vectorization text: title + abstract + full_text, truncated to max_chars.

    Combines the document title, abstract, and optional full text in a
    deterministic order (title first, then abstract, then full text) and
    truncates the result to the configured character limit.

    Args:
        title: The document title.
        abstract: The document abstract or summary.
        full_text: The full text of the document, or None if unavailable.
        max_chars: Maximum number of characters in the output.

    Returns:
        Combined text string truncated to at most *max_chars* characters.
    """
    parts: list[str] = []
    if title:
        parts.append(title)
    if abstract:
        parts.append(abstract)
    if full_text:
        parts.append(full_text)
    combined = " ".join(parts)
    return combined[:max_chars]
