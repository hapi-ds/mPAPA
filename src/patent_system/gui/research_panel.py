"""Research panel UI for the Patent Analysis & Drafting System.

Provides the structured invention disclosure form (primary description +
dynamic additional search terms), source selection checkboxes, "Start
Research" button, sort controls, and a sortable results table displaying
prior art search results.

Requirements: 16.4, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 2.3, 2.4, 2.5,
              3.1, 3.2, 3.5, 3.6, 4.1, 5.1, 5.2, 5.3, 6.1, 6.2
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import datetime
from typing import Any

from nicegui import ui

from patent_system.agents.prior_art_search import (
    _SOURCE_REGISTRY,
    prior_art_search_node,
)
from patent_system.db.repository import (
    InventionDisclosureRepository,
    LocalDocumentRepository,
    PatentRepository,
    ResearchSessionRepository,
    ScientificPaperRepository,
    SourcePreferenceRepository,
)
from patent_system.rag.vectorization import prepare_vectorization_text
from patent_system.services.text_extraction import extract_text_from_file

logger = logging.getLogger(__name__)

# Source names that produce scientific papers (vs patents)
_PAPER_SOURCES = {
    name for name, info in _SOURCE_REGISTRY.items() if info["type"] == "paper"
}

_relevance_top_k: int | None = None


def _get_relevance_top_k() -> int:
    global _relevance_top_k
    if _relevance_top_k is None:
        try:
            from patent_system.config import AppSettings
            _relevance_top_k = AppSettings().search_relevance_top_k
        except Exception:
            _relevance_top_k = 200
    return _relevance_top_k

# Maximum number of additional search terms (Req 1.5)
_MAX_SEARCH_TERMS = 20

# Sort criteria options (Req 3.6)
SORT_OPTIONS: dict[str, str] = {
    "discovery_date": "Discovery Date",
    "relevance": "Relevance",
    "citation_count": "Citation Count",
}

# Source URL templates for linking to the original record
_SOURCE_URLS: dict[str, str] = {
    "ArXiv": "https://arxiv.org/abs/{id}",
    "PubMed": "https://pubmed.ncbi.nlm.nih.gov/{id}",
    "Google Scholar": "https://scholar.google.com/scholar?q={id}",
    "Google Patents": "https://patents.google.com/patent/{id}",
    "EPO OPS": "https://worldwide.espacenet.com/patent/search?q={id}",
}


def _is_duplicate(result: dict, existing: list[dict]) -> bool:
    """Check if a result already exists by matching title + source (case-insensitive, stripped).

    Args:
        result: A search result dict with 'title' and 'source' keys.
        existing: List of previously saved record dicts.

    Returns:
        True if a duplicate is found, False otherwise.
    """
    title = result.get("title", "").strip().lower()
    source = result.get("source", "").strip().lower()
    for existing_rec in existing:
        if (
            existing_rec.get("title", "").strip().lower() == title
            and existing_rec.get("source", "").strip().lower() == source
        ):
            return True
    return False


def _build_disclosure_dict(primary_description: str, search_terms: list[str]) -> dict:
    """Construct an invention_disclosure dict from user inputs.

    Args:
        primary_description: The primary invention description text.
        search_terms: List of additional search term strings (may contain empty strings).

    Returns:
        Dict with 'technical_problem' set to primary_description and
        'novel_features' set to the list of non-empty search terms.
    """
    return {
        "technical_problem": primary_description,
        "novel_features": [t for t in search_terms if t],
    }


def _sanitize_metadata(raw: dict[str, Any]) -> dict[str, str | int | float]:
    """Sanitize a metadata dict for LlamaIndex compatibility.

    LlamaIndex metadata values must be str, int, or float. This function
    converts datetime objects to ISO strings and drops None values.
    """
    clean: dict[str, str | int | float] = {}
    for k, v in raw.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float)):
            clean[k] = v
        elif isinstance(v, datetime):
            clean[k] = v.isoformat()
        elif isinstance(v, bool):
            clean[k] = int(v)
        else:
            clean[k] = str(v)
    return clean


def _build_rag_document_text(abstract: str, full_text: str | None) -> str:
    """Construct the document text for RAG indexing.

    Combines abstract and full_text (when available) into a single string
    for indexing in the RAG engine.

    Args:
        abstract: The abstract/summary text of the document.
        full_text: The full text of the document, or None if unavailable.

    Returns:
        Combined text string containing both abstract and full_text when
        available, separated by a newline.
    """
    parts = []
    if abstract:
        parts.append(abstract)
    if full_text:
        parts.append(full_text)
    return "\n".join(parts)


def _build_source_url(record: dict[str, Any]) -> str | None:
    """Build a URL to the original source for a search result record."""
    source = record.get("source", "")
    template = _SOURCE_URLS.get(source)
    if not template:
        return None
    record_id = record.get("patent_number") or record.get("doi") or ""
    if not record_id or record_id == "UNKNOWN":
        return None
    return template.format(id=record_id)


def _normalize_date_key(value: Any) -> str:
    """Coerce a discovered_date value to a comparable string.

    Handles datetime objects, ISO strings, and missing/None values so
    that mixed-type lists can be sorted without a TypeError.
    """
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _persist_relevance_scores(
    results: list[dict[str, Any]],
    conn: Any,
) -> None:
    """Persist relevance scores from in-memory results to the database.

    Iterates results and updates the relevance_score column for records
    that have both a database ID and a computed relevance score.
    """
    try:
        patent_repo = PatentRepository(conn)
        paper_repo = ScientificPaperRepository(conn)
        for rec in results:
            score = rec.get("relevance_score")
            rec_id = rec.get("id")
            if score is None or rec_id is None:
                continue
            record_type = rec.get("record_type", "patent")
            if record_type == "paper":
                paper_repo.update_relevance_score(rec_id, score)
            elif record_type == "patent":
                patent_repo.update_relevance_score(rec_id, score)
    except Exception:
        logging.getLogger(__name__).debug(
            "Failed to persist relevance scores", exc_info=True,
        )


def _sort_results(
    results: list[dict[str, Any]],
    criterion: str,
) -> list[dict[str, Any]]:
    """Sort search results by the given criterion."""
    if criterion == "discovery_date":
        return sorted(results, key=lambda r: _normalize_date_key(r.get("discovered_date")), reverse=True)
    if criterion == "relevance":
        return sorted(
            results,
            key=lambda r: (
                r.get("relevance_score") is not None,  # scored first (True > False)
                r.get("relevance_score") if r.get("relevance_score") is not None else 0,
            ),
            reverse=True,
        )
    if criterion == "citation_count":
        return sorted(results, key=lambda r: r.get("citation_count", 0), reverse=True)
    return list(results)


def _bg_recalculate(
    conn: sqlite3.Connection,
    topic_id: int,
    rag_engine: Any | None = None,
    recalc_state: dict[str, Any] | None = None,
) -> None:
    """Regenerate all embeddings and recompute relevance scores for a topic.

    This module-level function implements the full recalculation pipeline:
      Phase 1 — Embedding Regeneration
      Phase 2 — RAG Re-indexing
      Phase 3 — Relevance Score Recomputation

    Args:
        conn: SQLite connection for database operations.
        topic_id: The topic whose records should be recalculated.
        rag_engine: Optional RAG engine for re-indexing and querying.
        recalc_state: Optional shared dict for progress communication to UI.
    """
    import struct

    from patent_system.config import AppSettings

    if recalc_state is None:
        recalc_state = {}

    recalc_state["status"] = "starting"
    recalc_state["error"] = None
    recalc_state["current"] = 0
    recalc_state["total"] = 0
    recalc_state["failures"] = 0
    recalc_state["done"] = False

    settings = AppSettings()
    _vect_limit = settings.vectorization_text_limit

    # Determine the embedding service to use
    if rag_engine is not None:
        embedding_service = rag_engine._embedding_service
    else:
        from patent_system.rag.embeddings import EmbeddingService
        embedding_service = EmbeddingService(
            model_name=settings.embedding_model_name,
            api_base=settings.lm_studio_base_url,
            api_key=settings.lm_studio_api_key,
        )

    try:
        # --- Collect all records ---
        session_repo = ResearchSessionRepository(conn)
        patent_repo = PatentRepository(conn)
        paper_repo = ScientificPaperRepository(conn)
        doc_repo = LocalDocumentRepository(conn)

        sessions = session_repo.get_by_topic(topic_id)

        # Build list of (record_type, record_id, title, abstract, full_text, repo)
        all_records: list[tuple[str, int, str, str, str | None, Any]] = []

        for session in sessions:
            for rec in patent_repo.get_by_session(session["id"]):
                all_records.append((
                    "patent", rec.id, rec.title, rec.abstract or "",
                    rec.full_text, patent_repo,
                ))
            for rec in paper_repo.get_by_session(session["id"]):
                all_records.append((
                    "paper", rec.id, rec.title, rec.abstract or "",
                    rec.full_text, paper_repo,
                ))

        for doc in doc_repo.get_by_topic(topic_id):
            all_records.append((
                "local_document", doc["id"], doc["filename"], doc["content"],
                None, doc_repo,
            ))

        total = len(all_records)
        recalc_state["total"] = total
        recalc_state["status"] = "regenerating"

        # --- Phase 1: Embedding Regeneration ---
        # Store new embeddings for Phase 2 re-indexing
        new_embeddings: dict[tuple[str, int], bytes] = {}

        for idx, (rec_type, rec_id, title, abstract, full_text, repo) in enumerate(all_records):
            recalc_state["current"] = idx + 1

            text = prepare_vectorization_text(
                title=title,
                abstract=abstract,
                full_text=full_text,
                max_chars=_vect_limit,
            )
            if not text.strip():
                recalc_state["failures"] += 1
                continue

            try:
                emb_bytes = embedding_service.generate_embedding(text)
            except Exception as exc:
                # Connection error — stop immediately
                recalc_state["status"] = "error"
                recalc_state["error"] = f"Embedding service error: {exc}"
                recalc_state["done"] = True
                logger.error(
                    "Embedding service connection error during recalculation: %s", exc
                )
                return

            if emb_bytes is None:
                # Skip this record — do NOT overwrite existing embedding
                recalc_state["failures"] += 1
                continue

            # Persist the new embedding
            repo.update_embedding(rec_id, emb_bytes)
            new_embeddings[(rec_type, rec_id)] = emb_bytes

        # --- Phase 2: RAG Re-indexing ---
        recalc_state["status"] = "reindexing"

        if rag_engine is not None:
            # Clear existing index for this topic
            rag_engine._indexes.pop(topic_id, None)

            # Build documents with fresh embeddings for re-indexing
            rag_docs: list[dict[str, Any]] = []
            for rec_type, rec_id, title, abstract, full_text, _repo in all_records:
                emb_bytes = new_embeddings.get((rec_type, rec_id))
                if not emb_bytes:
                    continue

                text = prepare_vectorization_text(
                    title=title,
                    abstract=abstract,
                    full_text=full_text,
                    max_chars=_vect_limit,
                )
                metadata = _sanitize_metadata({
                    "record_type": rec_type,
                    "id": rec_id,
                    "title": title,
                    "source": "recalculated",
                })

                n_floats = len(emb_bytes) // 4
                embedding_list = list(struct.unpack(f"{n_floats}f", emb_bytes))

                rag_docs.append({
                    "text": text,
                    "metadata": metadata,
                    "embedding": embedding_list,
                })

            if rag_docs:
                rag_engine.index_with_embeddings(topic_id, rag_docs)

        # --- Phase 3: Relevance Score Recomputation ---
        recalc_state["status"] = "scoring"

        if rag_engine is not None:
            disclosure_repo = InventionDisclosureRepository(conn)
            disclosure = disclosure_repo.get_by_topic(topic_id)

            if disclosure and disclosure.get("primary_description"):
                disclosure_text = disclosure["primary_description"]
                # Build a results list for _persist_relevance_scores
                results_for_scoring: list[dict[str, Any]] = []
                for rec_type, rec_id, title, abstract, full_text, _repo in all_records:
                    results_for_scoring.append({
                        "id": rec_id,
                        "record_type": rec_type,
                        "title": title,
                        "abstract": abstract,
                        "full_text": full_text,
                    })

                rag_results = rag_engine.query(
                    topic_id, disclosure_text, top_k=len(results_for_scoring)
                )

                score_map: dict[str, float] = {}
                for rr in rag_results:
                    rr_text = rr.get("text", "")
                    rr_score = rr.get("score", 0.0) or 0.0
                    for rec in results_for_scoring:
                        rec_title = rec.get("title", "")
                        if not rec_title or rec_title in score_map:
                            continue
                        if rr_text.startswith(rec_title) or rec_title in rr_text:
                            score_map[rec_title] = rr_score

                for rec in results_for_scoring:
                    title = rec.get("title", "")
                    if title in score_map:
                        rec["relevance_score"] = round(score_map[title] * 100, 1)

                # Persist relevance scores to DB
                _persist_relevance_scores(results_for_scoring, conn)

                # Store results in recalc_state for UI update
                recalc_state["results"] = results_for_scoring

        recalc_state["status"] = "done"
        recalc_state["done"] = True

    except Exception as exc:
        recalc_state["status"] = "error"
        recalc_state["error"] = str(exc)
        recalc_state["done"] = True
        logger.exception("Recalculation failed for topic %d", topic_id)


def create_research_panel(
    container: Any,
    topic_id: int,
    conn: sqlite3.Connection | None = None,
    rag_engine: Any | None = None,
    disclosure_repo: InventionDisclosureRepository | None = None,
    source_pref_repo: SourcePreferenceRepository | None = None,
    max_results_per_source: int = 10,
    header_status_label: Any | None = None,
    header_spinner: Any | None = None,
) -> None:
    """Populate *container* with the Research panel UI components.

    Args:
        container: The NiceGUI container element to populate.
        topic_id: The active topic ID.
        conn: SQLite connection for creating per-request repositories.
        rag_engine: Optional RAG engine for document indexing.
        disclosure_repo: Repository for invention disclosure persistence.
        source_pref_repo: Repository for source preference persistence.
        max_results_per_source: Maximum results to fetch per source.
        header_status_label: Optional shared label in the header for
            showing activity status (search progress, import status).
        header_spinner: Optional shared spinner in the header shown
            while background work is running.
    """
    container.clear()

    def _set_header_status(text: str, busy: bool = False) -> None:
        """Update the shared header status line."""
        if header_status_label is not None:
            header_status_label.set_text(text)
        if header_spinner is not None:
            header_spinner.set_visibility(busy)

    panel_state: dict[str, Any] = {
        "results": [],
        "sort_criterion": "discovery_date",
        "term_inputs": [],  # list of ui.input references for additional terms
        "source_checkboxes": {},  # source_name -> ui.checkbox reference
    }

    # Load previously saved results from DB (Req 5.2)
    if conn is not None:
        try:
            session_repo = ResearchSessionRepository(conn)
            patent_repo = PatentRepository(conn)
            paper_repo = ScientificPaperRepository(conn)
            sessions = session_repo.get_by_topic(topic_id)
            saved_results: list[dict[str, Any]] = []
            for session in sessions:
                for rec in patent_repo.get_by_session(session["id"]):
                    saved_results.append({
                        "id": rec.id,
                        "record_type": "patent",
                        "title": rec.title,
                        "abstract": rec.abstract,
                        "full_text": rec.full_text,
                        "source": rec.source,
                        "patent_number": rec.patent_number,
                        "discovered_date": rec.discovered_date.isoformat() if rec.discovered_date else "",
                        "relevance_score": rec.relevance_score,
                    })
                for rec in paper_repo.get_by_session(session["id"]):
                    saved_results.append({
                        "id": rec.id,
                        "record_type": "paper",
                        "title": rec.title,
                        "abstract": rec.abstract,
                        "full_text": rec.full_text,
                        "source": rec.source,
                        "doi": rec.doi,
                        "discovered_date": rec.discovered_date.isoformat() if rec.discovered_date else "",
                        "relevance_score": rec.relevance_score,
                    })
            panel_state["results"] = saved_results
        except Exception:
            logger.exception("Failed to load saved results for topic %d", topic_id)

    # Load saved local documents
    if conn is not None:
        try:
            doc_repo = LocalDocumentRepository(conn)
            for doc in doc_repo.get_by_topic(topic_id):
                text = doc["content"]
                abstract = text[:500].strip()
                if len(text) > 500:
                    abstract += "…"
                panel_state["results"].append({
                    "id": doc["id"],
                    "record_type": "local_document",
                    "title": doc["filename"],
                    "abstract": abstract,
                    "full_text": text,
                    "source": "Local Document",
                })
        except Exception:
            logger.exception("Failed to load local documents for topic %d", topic_id)

    # Load saved disclosure data (Req 2.4)
    saved_disclosure: dict | None = None
    if disclosure_repo is not None:
        try:
            saved_disclosure = disclosure_repo.get_by_topic(topic_id)
        except Exception:
            logger.exception("Failed to load disclosure for topic %d", topic_id)

    # Index saved results in RAG and compute relevance in background
    # Uses stored embeddings from DB — no LM Studio calls needed
    if panel_state["results"] and rag_engine is not None:
        _results_ref = panel_state["results"]
        _desc_ref = saved_disclosure["primary_description"] if saved_disclosure else None

        # Build a map of (record_type, id) → embedding bytes from DB
        _emb_map: dict[tuple[str, int], bytes] = {}
        if conn is not None:
            try:
                _pat_repo = PatentRepository(conn)
                _paper_repo = ScientificPaperRepository(conn)
                _session_repo = ResearchSessionRepository(conn)
                sessions = _session_repo.get_by_topic(topic_id)
                for session in sessions:
                    for rec in _pat_repo.get_by_session(session["id"]):
                        if rec.embedding and rec.id is not None:
                            _emb_map[("patent", rec.id)] = rec.embedding
                    for rec in _paper_repo.get_by_session(session["id"]):
                        if rec.embedding and rec.id is not None:
                            _emb_map[("paper", rec.id)] = rec.embedding
                # Local documents
                _doc_repo = LocalDocumentRepository(conn)
                for doc in _doc_repo.get_by_topic(topic_id):
                    row = conn.execute(
                        "SELECT embedding FROM local_documents WHERE id = ?", (doc["id"],)
                    ).fetchone()
                    if row and row[0]:
                        _emb_map[("local_document", doc["id"])] = row[0]
            except Exception:
                logger.debug("Failed to load stored embeddings", exc_info=True)

        def _bg_index_and_score() -> None:
            import struct
            try:
                from patent_system.config import AppSettings as _AppSettings
                _vect_limit = _AppSettings().vectorization_text_limit

                rag_docs = []
                for rec in _results_ref:
                    title = rec.get("title", "")
                    abstract = rec.get("abstract", "")
                    full_text = rec.get("full_text")
                    text = prepare_vectorization_text(
                        title=title,
                        abstract=abstract,
                        full_text=full_text,
                        max_chars=_vect_limit,
                    )
                    if not text:
                        continue
                    metadata = _sanitize_metadata({
                        k: v for k, v in rec.items()
                        if k not in ("title", "abstract", "full_text", "embedding")
                    })
                    doc_entry: dict[str, Any] = {"text": text, "metadata": metadata}

                    # Look up stored embedding
                    rec_type = rec.get("record_type", "patent")
                    rec_id = rec.get("id")
                    if rec_id is not None:
                        emb_bytes = _emb_map.get((rec_type, rec_id))
                        if emb_bytes:
                            n_floats = len(emb_bytes) // 4
                            doc_entry["embedding"] = list(struct.unpack(f"{n_floats}f", emb_bytes))

                    rag_docs.append(doc_entry)

                if rag_docs:
                    rag_engine.index_with_embeddings(topic_id, rag_docs)

                if _desc_ref:
                    rag_results = rag_engine.query(topic_id, _desc_ref, top_k=len(_results_ref))
                    score_map: dict[str, float] = {}
                    for rr in rag_results:
                        rr_text = rr.get("text", "")
                        rr_score = rr.get("score", 0.0) or 0.0
                        for rec in _results_ref:
                            rec_title = rec.get("title", "")
                            if not rec_title or rec_title in score_map:
                                continue
                            # Match: title appears at the start of the RAG text
                            if rr_text.startswith(rec_title) or rec_title in rr_text:
                                score_map[rec_title] = rr_score
                    for rec in _results_ref:
                        title = rec.get("title", "")
                        if title in score_map:
                            rec["relevance_score"] = round(score_map[title] * 100, 1)
                    # Persist relevance scores to DB
                    if conn is not None:
                        _persist_relevance_scores(_results_ref, conn)
            except Exception:
                logger.debug("Failed to index/score saved results", exc_info=True)
            finally:
                _scoring_done[0] = True

        import threading
        _scoring_done = [False]
        threading.Thread(target=_bg_index_and_score, daemon=True).start()

    # Shared state for recalculation progress (used by _bg_recalculate closure and UI timer)
    recalc_state: dict[str, Any] = {
        "status": "idle",
        "error": None,
        "current": 0,
        "total": 0,
        "failures": 0,
        "done": False,
        "results": None,
    }

    def _bg_recalculate_closure() -> None:
        """Closure wrapper that calls the module-level _bg_recalculate with panel context."""
        _bg_recalculate(
            conn=conn,
            topic_id=topic_id,
            rag_engine=rag_engine,
            recalc_state=recalc_state,
        )
        # Update panel_state results with new relevance scores if available
        if recalc_state.get("results"):
            score_map: dict[str, float] = {}
            for rec in recalc_state["results"]:
                title = rec.get("title", "")
                score = rec.get("relevance_score")
                if title and score is not None:
                    score_map[title] = score
            for rec in panel_state["results"]:
                title = rec.get("title", "")
                if title in score_map:
                    rec["relevance_score"] = score_map[title]

    # Load saved source preferences (Req 3.5)
    saved_prefs: dict[str, bool] | None = None
    if source_pref_repo is not None:
        try:
            saved_prefs = source_pref_repo.get_by_topic(topic_id)
        except Exception:
            logger.exception("Failed to load source prefs for topic %d", topic_id)

    with container:
        with ui.row().classes("w-full items-center justify-between q-mb-sm"):
            ui.label("Prior Art Research").classes("text-h6")
            result_count = len(panel_state["results"])
            count_label = ui.label(
                f"{result_count} result(s)" if result_count else "No results yet"
            ).classes("text-caption text-grey")

        # --- Primary Invention Description (Req 1.1) ---
        primary_input = ui.textarea(
            label="Primary Invention Description",
            placeholder="Describe your invention…",
            value=saved_disclosure["primary_description"] if saved_disclosure else "",
        ).classes("w-full")

        # --- Additional Search Terms (Req 1.2, 1.3, 1.4, 1.5) ---
        ui.label("Additional Search Terms").classes("text-subtitle2 q-mt-md")
        terms_container = ui.column().classes("w-full gap-1")

        def _render_terms() -> None:
            """Re-render the dynamic term input list."""
            terms_container.clear()
            with terms_container:
                for idx, term_val in enumerate(panel_state["term_inputs"]):
                    _idx = idx  # capture for closure
                    with ui.row().classes("w-full items-center gap-1"):
                        inp = ui.input(
                            placeholder=f"Search term {_idx + 1}",
                            value=term_val,
                            on_change=lambda e, i=_idx: _on_term_change(i, e.value),
                        ).classes("flex-grow")
                        ui.button(
                            icon="remove",
                            on_click=lambda _, i=_idx: _remove_term(i),
                        ).props("flat dense color=negative")

        def _on_term_change(idx: int, value: str) -> None:
            """Update the term value in panel_state when user edits."""
            if 0 <= idx < len(panel_state["term_inputs"]):
                panel_state["term_inputs"][idx] = value

        def _add_term() -> None:
            """Append a new empty term input (Req 1.3, 1.5)."""
            if len(panel_state["term_inputs"]) >= _MAX_SEARCH_TERMS:
                ui.notify(
                    f"Maximum of {_MAX_SEARCH_TERMS} additional search terms reached.",
                    type="warning",
                )
                return
            panel_state["term_inputs"].append("")
            _render_terms()

        def _remove_term(idx: int) -> None:
            """Remove a term input by index (Req 1.4)."""
            if 0 <= idx < len(panel_state["term_inputs"]):
                panel_state["term_inputs"].pop(idx)
                _render_terms()

        # Populate saved terms (Req 2.4)
        if saved_disclosure and saved_disclosure.get("search_terms"):
            panel_state["term_inputs"] = list(saved_disclosure["search_terms"])

        _render_terms()

        ui.button("Add Term", icon="add", on_click=_add_term).props("flat dense")

        # --- Suggest Search Terms Button ---
        async def _on_suggest_terms() -> None:
            """Use LLM to suggest search terms from the primary description."""
            description = primary_input.value.strip() if primary_input.value else ""
            if not description:
                ui.notify("Please enter a primary invention description first.", type="warning")
                return

            _set_header_status("Suggesting search terms…", busy=True)
            try:
                from patent_system.dspy_modules.modules import SuggestSearchTermsModule

                module = SuggestSearchTermsModule()
                prediction = await asyncio.to_thread(
                    module, invention_description=description
                )
                raw = prediction.search_terms or ""
                suggested = [
                    t.strip().lstrip("•-–0123456789.) ")
                    for t in raw.splitlines()
                    if t.strip() and not t.strip().startswith("#")
                ]
                suggested = [t for t in suggested if t]

                if not suggested:
                    ui.notify("No terms suggested — try a more detailed description.", type="warning")
                    _set_header_status("No search terms suggested", busy=False)
                    return

                # Add suggested terms that aren't already present
                existing = {t.strip().lower() for t in panel_state["term_inputs"] if t.strip()}
                added = 0
                for term in suggested:
                    if term.lower() in existing:
                        continue
                    if len(panel_state["term_inputs"]) >= _MAX_SEARCH_TERMS:
                        break
                    panel_state["term_inputs"].append(term)
                    existing.add(term.lower())
                    added += 1

                _render_terms()
                msg = f"Added {added} suggested term(s)"
                if len(suggested) > added:
                    msg += f" ({len(suggested) - added} duplicates or limit reached)"
                ui.notify(msg, type="positive")
                _set_header_status(f"✓ {msg}", busy=False)
                logger.info("Suggested %d terms for topic %d, added %d", len(suggested), topic_id, added)

            except Exception as exc:
                logger.exception("Failed to suggest search terms for topic %d", topic_id)
                ui.notify(f"Term suggestion failed: {exc}", type="negative")
                _set_header_status("⚠ Term suggestion failed", busy=False)

        ui.button(
            "Suggest Search Terms", icon="auto_awesome", on_click=_on_suggest_terms
        ).props("flat dense color=primary")

        # --- Save Disclosure Button ---
        save_status = ui.label("").classes("text-caption text-grey")

        def _on_save_disclosure() -> None:
            """Persist the primary description and search terms to the DB."""
            description = primary_input.value.strip() if primary_input.value else ""
            if not description:
                ui.notify("Please enter a primary invention description.", type="warning")
                return

            terms = list(panel_state["term_inputs"])

            if disclosure_repo is not None:
                try:
                    disclosure_repo.upsert(topic_id, description, terms)
                    save_status.set_text("✓ Saved")
                    ui.notify("Disclosure saved.", type="positive")
                    logger.info("Saved disclosure for topic %d", topic_id)
                except Exception:
                    logger.exception("Failed to save disclosure for topic %d", topic_id)
                    ui.notify("Failed to save disclosure.", type="negative")
                    save_status.set_text("⚠ Save failed")
            else:
                ui.notify("No database connection — cannot save.", type="warning")

        with ui.row().classes("items-center gap-2 q-mt-sm"):
            ui.button("Save", icon="save", on_click=_on_save_disclosure).props(
                "color=secondary"
            )
            save_status

        # --- Source Selection Checkboxes (Req 3.1, 3.2, 3.5, 3.6) ---
        ui.label("Sources").classes("text-subtitle2 q-mt-md")

        source_names = list(_SOURCE_REGISTRY.keys())

        # Default all checked when no saved preference (Req 3.2)
        if saved_prefs is None:
            current_prefs = {name: True for name in source_names}
        else:
            current_prefs = {name: saved_prefs.get(name, True) for name in source_names}

        with ui.row().classes("w-full gap-4 flex-wrap"):
            for src_name in source_names:
                def _make_checkbox_handler(name: str):
                    def _on_change(e: Any) -> None:
                        new_val = e.value
                        # Prevent deselecting the last remaining source (Req 3.6)
                        enabled_count = sum(
                            1 for sn in source_names
                            if current_prefs.get(sn, True)
                        )
                        if not new_val and enabled_count <= 1:
                            ui.notify(
                                "At least one source must remain selected.",
                                type="warning",
                            )
                            # Re-check the checkbox
                            panel_state["source_checkboxes"][name].set_value(True)
                            return

                        current_prefs[name] = new_val

                        # Save preference (Req 3.4)
                        if source_pref_repo is not None:
                            try:
                                source_pref_repo.save(topic_id, current_prefs)
                            except Exception:
                                logger.exception(
                                    "Failed to save source prefs for topic %d", topic_id
                                )
                                ui.notify("Failed to save source preference.", type="negative")
                    return _on_change

                cb = ui.checkbox(
                    src_name,
                    value=current_prefs.get(src_name, True),
                    on_change=_make_checkbox_handler(src_name),
                )
                panel_state["source_checkboxes"][src_name] = cb

        # --- Local Document Upload ---
        ui.label("Upload Local Documents").classes("text-subtitle2 q-mt-md")

        import_status = ui.label("").classes("text-caption text-grey")

        async def _on_file_uploaded(e: Any) -> None:
            """Process uploaded file — read async, then persist/embed in thread."""
            if conn is None:
                return
            try:
                f = e.file
                filename = f.name
                content_bytes = await f.read()
                import_status.set_text(f"Importing {filename}…")
                _set_header_status(f"Importing {filename}…", busy=True)

                def _process() -> dict | None:
                    """Run in thread: extract, persist, embed, index."""
                    text = extract_text_from_file(filename, content_bytes)
                    if not text.strip():
                        return None

                    word_count = len(text.split())
                    abstract = text[:500].strip()
                    if len(text) > 500:
                        abstract += "…"

                    doc_repo = LocalDocumentRepository(conn)
                    row_id = doc_repo.create(topic_id, filename, text)

                    if rag_engine is not None:
                        try:
                            from patent_system.config import AppSettings as _AppSettings
                            _vect_limit = _AppSettings().vectorization_text_limit
                            _emb_text = prepare_vectorization_text(
                                title=filename,
                                abstract=text,
                                max_chars=_vect_limit,
                            )
                            emb = rag_engine._embedding_service.generate_embedding(_emb_text)
                            if emb:
                                doc_repo.update_embedding(row_id, emb)
                        except Exception:
                            logger.debug("Embedding failed for %s", filename, exc_info=True)
                        try:
                            _rag_text = prepare_vectorization_text(
                                title=filename,
                                abstract=text,
                                max_chars=_vect_limit,
                            )
                            rag_engine.index_documents(topic_id, [
                                {"text": _rag_text, "metadata": {"source": "Local Document", "filename": filename}},
                            ])
                        except Exception:
                            logger.debug("RAG index failed for %s", filename, exc_info=True)

                    return {
                        "id": row_id,
                        "record_type": "local_document",
                        "title": filename,
                        "abstract": abstract,
                        "full_text": text,
                        "source": "Local Document",
                        "word_count": word_count,
                    }

                result = await asyncio.to_thread(_process)

                if result is None:
                    import_status.set_text(
                        f"⚠ No text in {filename} — may be a scanned/image PDF"
                    )
                    _set_header_status(f"⚠ No text in {filename}", busy=False)
                    return

                panel_state["results"].append(result)
                import_status.set_text(f"✓ Imported {filename} ({result['word_count']:,} words)")
                _set_header_status(f"✓ Imported {filename} ({result['word_count']:,} words)", busy=False)
                logger.info("Imported '%s' for topic %d (%d words)", filename, topic_id, result["word_count"])
                _refresh_table()
            except Exception:
                logger.exception("Failed to import uploaded file")
                import_status.set_text("⚠ Import failed")
                _set_header_status("⚠ Import failed", busy=False)

        ui.upload(
            label="Upload PDF, DOCX, or TXT — files are imported automatically",
            on_upload=_on_file_uploaded,
            multiple=True,
            auto_upload=True,
        ).props('accept=".pdf,.docx,.txt"').classes("w-full").style("max-height: 100px;")

        # --- Status and Start Research ---
        search_status = ui.label("").classes("text-caption q-mt-sm")
        search_spinner = ui.spinner("dots", size="sm").classes("q-ml-sm")
        search_spinner.set_visibility(False)

        async def _on_start_research() -> None:
            """Run search fully in background with live status updates."""
            description = primary_input.value.strip() if primary_input.value else ""
            if not description:
                ui.notify("Please enter a primary invention description.", type="warning")
                return

            terms = list(panel_state["term_inputs"])

            # Validate search terms and warn about problematic formatting
            warnings: list[str] = []
            for idx, t in enumerate(terms):
                if not t or not t.strip():
                    continue
                if '"' in t and 'AND' in t:
                    warnings.append(
                        f"Term {idx + 1}: Contains quotes + AND — use simple phrases instead "
                        f"(e.g. 'glycolytic flux inhibition' not '\"Glycolytic flux inhibition\" AND ...')"
                    )
                elif '"' in t:
                    warnings.append(
                        f"Term {idx + 1}: Contains quotes — they'll be stripped for some sources. "
                        f"Use plain text instead."
                    )
                if '*' in t:
                    warnings.append(
                        f"Term {idx + 1}: Wildcards (*) not supported by all sources — will be removed."
                    )
                if "'" in t:
                    warnings.append(
                        f"Term {idx + 1}: Apostrophes may cause issues with patent databases."
                    )
                if len(t) > 100:
                    warnings.append(
                        f"Term {idx + 1}: Very long ({len(t)} chars) — may be truncated. "
                        f"Consider splitting into shorter terms."
                    )
            if warnings:
                ui.notify(
                    "Search term tips:\n" + "\n".join(warnings),
                    type="warning",
                    close_button=True,
                    timeout=10000,
                )

            if disclosure_repo is not None:
                try:
                    disclosure_repo.upsert(topic_id, description, terms)
                except Exception:
                    logger.exception("Failed to save disclosure for topic %d", topic_id)
                    ui.notify("Failed to save disclosure.", type="negative")
                    return

            selected_sources = [name for name, enabled in current_prefs.items() if enabled]
            invention_disclosure = _build_disclosure_dict(description, terms)

            # Shared state for background thread → UI communication
            search_state: dict[str, Any] = {
                "status": "Starting search…",
                "done": False,
                "new_results": [],
                "failed": [],
                "total_raw": 0,
            }

            search_spinner.set_visibility(True)
            search_status.set_text("Starting search…")
            _set_header_status("Research: Starting search…", busy=True)
            research_button.disable()
            logger.info("Start Research for topic %d", topic_id)

            def _bg_search() -> None:
                """Background: search, persist, embed, index."""
                state = {
                    "topic_id": topic_id,
                    "invention_disclosure": invention_disclosure,
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

                search_state["status"] = f"Searching {len(selected_sources)} sources…"

                from patent_system.config import AppSettings as _SearchSettings
                _search_settings = _SearchSettings()

                def _download_progress(current: int, total: int) -> None:
                    search_state["status"] = f"Downloading full text {current}/{total}…"

                result = prior_art_search_node(
                    state,
                    rag_engine=rag_engine,
                    selected_sources=selected_sources,
                    max_results_per_source=max_results_per_source,
                    settings=_search_settings,
                    progress_callback=_download_progress,
                )

                results = result.get("prior_art_results", [])
                failed = result.get("failed_sources", [])
                search_state["total_raw"] = len(results)
                search_state["failed"] = failed

                # Deduplicate
                new_results: list[dict[str, Any]] = []
                for rec in results:
                    if not _is_duplicate(rec, panel_state["results"]):
                        new_results.append(rec)

                search_state["status"] = f"Saving {len(new_results)} results…"

                # Persist
                if conn is not None:
                    try:
                        session_repo = ResearchSessionRepository(conn)
                        patent_repo = PatentRepository(conn)
                        paper_repo = ScientificPaperRepository(conn)
                        session_id = session_repo.create(topic_id, query=description)
                        for rec in new_results:
                            source_name = rec.get("source", "unknown")
                            is_paper = source_name in _PAPER_SOURCES
                            title = rec.get("title", "Untitled")
                            abstract = rec.get("abstract", "")
                            full_text = rec.get("full_text")

                            if is_paper:
                                from patent_system.db.models import ScientificPaperRecord as SPR
                                paper_record = SPR(
                                    session_id=session_id, doi=rec.get("doi", ""),
                                    title=title, abstract=abstract, full_text=full_text,
                                    pdf_path=rec.get("pdf_path"), source=source_name,
                                )
                                row_id = paper_repo.create(session_id, paper_record)
                                rec["id"] = row_id
                                rec["record_type"] = "paper"
                                repo_for_emb = paper_repo
                            else:
                                from patent_system.db.models import PatentRecord as PR
                                patent_record = PR(
                                    session_id=session_id,
                                    patent_number=rec.get("patent_number", "UNKNOWN"),
                                    title=title, abstract=abstract, full_text=full_text,
                                    claims=rec.get("claims"),
                                    pdf_path=rec.get("pdf_path"),
                                    source=source_name,
                                )
                                row_id = patent_repo.create(session_id, patent_record)
                                rec["id"] = row_id
                                rec["record_type"] = "patent"
                                repo_for_emb = patent_repo

                            if rag_engine is not None:
                                try:
                                    from patent_system.config import AppSettings as _AppSettings
                                    _vect_limit = _AppSettings().vectorization_text_limit
                                    emb_text = prepare_vectorization_text(
                                        title=title,
                                        abstract=abstract or "",
                                        full_text=full_text,
                                        max_chars=_vect_limit,
                                    )
                                    emb = rag_engine._embedding_service.generate_embedding(emb_text)
                                    if emb is not None:
                                        repo_for_emb.update_embedding(row_id, emb)
                                except Exception:
                                    logger.debug("Embedding failed for %d", row_id, exc_info=True)
                    except Exception:
                        logger.exception("Failed to persist results for topic %d", topic_id)

                search_state["status"] = "Indexing in RAG…"

                # RAG index
                if rag_engine is not None and new_results:
                    from patent_system.config import AppSettings as _AppSettings
                    _vect_limit_rag = _AppSettings().vectorization_text_limit

                    rag_docs = []
                    for rec in new_results:
                        title = rec.get("title", "")
                        abstract = rec.get("abstract", "")
                        full_text = rec.get("full_text")
                        text = prepare_vectorization_text(
                            title=title,
                            abstract=abstract,
                            full_text=full_text,
                            max_chars=_vect_limit_rag,
                        )
                        if text:
                            metadata = _sanitize_metadata({
                                k: v for k, v in rec.items()
                                if k not in ("title", "abstract", "full_text", "embedding")
                            })
                            rag_docs.append({"text": text, "metadata": metadata})
                    if rag_docs:
                        try:
                            rag_engine.index_documents(topic_id, rag_docs)
                        except Exception:
                            logger.exception("Failed to index in RAG for topic %d", topic_id)

                # Relevance scoring
                if rag_engine is not None and description:
                    search_state["status"] = "Computing relevance…"
                    try:
                        all_recs = new_results + panel_state["results"]
                        rag_results = rag_engine.query(topic_id, description, top_k=len(all_recs))
                        score_map: dict[str, float] = {}
                        for rr in rag_results:
                            rr_text = rr.get("text", "")
                            rr_score = rr.get("score", 0.0) or 0.0
                            for rec in all_recs:
                                rec_title = rec.get("title", "")
                                if not rec_title or rec_title in score_map:
                                    continue
                                if rr_text.startswith(rec_title) or rec_title in rr_text:
                                    score_map[rec_title] = rr_score
                        for rec in all_recs:
                            title = rec.get("title", "")
                            if title in score_map:
                                rec["relevance_score"] = round(score_map[title] * 100, 1)
                        # Persist relevance scores to DB
                        if conn is not None:
                            _persist_relevance_scores(all_recs, conn)
                    except Exception:
                        logger.debug("Failed to compute relevance", exc_info=True)

                search_state["new_results"] = new_results
                search_state["done"] = True

            # Start background thread
            import threading
            threading.Thread(target=_bg_search, daemon=True).start()

            # Poll for completion with a timer
            def _check_search() -> None:
                search_status.set_text(search_state["status"])
                _set_header_status(f"Research: {search_state['status']}", busy=True)
                if search_state["done"]:
                    _search_timer.deactivate()
                    search_spinner.set_visibility(False)
                    research_button.enable()

                    new_results = search_state["new_results"]
                    panel_state["results"].extend(new_results)
                    _refresh_table()

                    parts = [f"{len(new_results)} new result(s) found"]
                    dupes = search_state["total_raw"] - len(new_results)
                    if dupes > 0:
                        parts.append(f"{dupes} duplicate(s) skipped")
                    if search_state["failed"]:
                        parts.append(f"{len(search_state['failed'])} source(s) unavailable")
                    summary = " · ".join(parts)
                    search_status.set_text(summary)
                    _set_header_status(f"Research: {summary}", busy=False)

            _search_timer = ui.timer(2.0, _check_search)

        research_button = ui.button(
            "Start Research", on_click=_on_start_research
        ).props("color=primary").classes("q-mt-sm")

        # --- Recalculate Embeddings & Scores ---
        async def _on_recalculate() -> None:
            """Regenerate all embeddings and recompute relevance scores."""
            recalculate_button.disable()
            _set_header_status("Recalculating embeddings…", busy=True)

            # Reset recalc_state
            recalc_state["done"] = False
            recalc_state["error"] = None
            recalc_state["status"] = "starting"
            recalc_state["current"] = 0
            recalc_state["total"] = 0
            recalc_state["failures"] = 0
            recalc_state["results"] = None

            # Start background thread
            import threading
            threading.Thread(target=_bg_recalculate_closure, daemon=True).start()

            # Poll for completion
            def _check_recalc() -> None:
                if recalc_state["status"] == "regenerating":
                    current = recalc_state.get("current", 0)
                    total = recalc_state.get("total", 0)
                    _set_header_status(
                        f"Regenerating embeddings ({current}/{total})…", busy=True
                    )
                elif recalc_state["status"] == "reindexing":
                    _set_header_status("Re-indexing in RAG…", busy=True)
                elif recalc_state["status"] == "scoring":
                    _set_header_status("Computing relevance scores…", busy=True)

                if recalc_state["done"]:
                    _recalc_timer.deactivate()
                    recalculate_button.enable()

                    if recalc_state.get("error"):
                        ui.notify(recalc_state["error"], type="negative")
                        _set_header_status(
                            f"⚠ Recalculation failed: {recalc_state['error']}",
                            busy=False,
                        )
                    else:
                        # Update results with new scores
                        if recalc_state.get("results"):
                            panel_state["results"] = recalc_state["results"]
                        _refresh_table()

                        updated = recalc_state.get("total", 0) - recalc_state.get(
                            "failures", 0
                        )
                        failed = recalc_state.get("failures", 0)
                        _set_header_status(
                            f"✓ Recalculation complete ({updated} updated, {failed} failed)",
                            busy=False,
                        )
                        ui.notify("Recalculation complete!", type="positive")

            _recalc_timer = ui.timer(1.0, _check_recalc)

        recalculate_button = ui.button(
            "Recalculate Embeddings & Scores", icon="refresh", on_click=_on_recalculate
        ).props("color=accent").classes("q-mt-sm q-ml-sm")

        # --- Sort control ---
        def _on_sort_change(e: Any) -> None:
            panel_state["sort_criterion"] = e.value
            _refresh_table()

        ui.select(
            options=SORT_OPTIONS,
            value="discovery_date",
            label="Sort by",
            on_change=_on_sort_change,
        ).classes("w-48 q-mt-sm")

        # --- Results display ---
        results_container = ui.column().classes("w-full q-mt-md gap-2")

        def _delete_result(rec: dict[str, Any]) -> None:
            """Delete a result from the database and panel state."""
            db_id = rec.get("id")
            if db_id is not None and conn is not None:
                try:
                    record_type = rec.get("record_type")
                    if record_type == "paper":
                        ScientificPaperRepository(conn).delete(db_id)
                    elif record_type == "local_document":
                        LocalDocumentRepository(conn).delete(db_id)
                    else:
                        PatentRepository(conn).delete(db_id)
                    logger.info("Deleted record %d for topic %d", db_id, topic_id)
                except Exception:
                    logger.exception("Failed to delete record %d", db_id)
                    ui.notify("Failed to delete record from database.", type="negative")
                    return

            panel_state["results"] = [r for r in panel_state["results"] if r is not rec]
            _refresh_table()
            ui.notify("Result deleted.", type="info")

        def _render_results(results: list[dict[str, Any]]) -> None:
            """Render search results as expandable cards with source links."""
            results_container.clear()
            with results_container:
                if not results:
                    ui.label("No results yet.").classes("text-grey")
                    return
                for idx, rec in enumerate(results, 1):
                    title = rec.get("title", "Untitled")
                    abstract = rec.get("abstract", "")
                    source = rec.get("source", "unknown")
                    record_id = rec.get("patent_number") or rec.get("doi") or ""
                    source_url = _build_source_url(rec)

                    with ui.card().classes("w-full"):
                        with ui.row().classes("w-full items-center justify-between"):
                            with ui.row().classes("items-center gap-2"):
                                ui.badge(f"[{idx}]", color="dark").props("dense")
                                ui.label(title).classes("text-subtitle1 font-bold")
                            with ui.row().classes("items-center gap-1"):
                                relevance = rec.get("relevance_score")
                                if relevance is not None:
                                    color = "positive" if relevance >= 70 else "warning" if relevance >= 40 else "grey"
                                    ui.badge(f"{relevance}%", color=color).props("outline")
                                has_full_text = bool(rec.get("full_text"))
                                if has_full_text:
                                    ui.badge("Full Text", color="positive").props("outline")
                                else:
                                    ui.badge("Abstract Only", color="grey").props("outline")
                                ui.badge(source).props("color=primary outline")
                                ui.button(
                                    icon="delete",
                                    on_click=lambda _, r=rec: _delete_result(r),
                                ).props("flat dense color=negative size=sm")

                        if record_id and record_id != "UNKNOWN":
                            with ui.row().classes("items-center gap-2"):
                                ui.label(record_id).classes("text-caption text-grey")
                                if source_url:
                                    ui.link(
                                        "Open in " + source, source_url, new_tab=True
                                    ).classes("text-caption")

                        if abstract:
                            short = abstract[:200]
                            if len(abstract) > 200:
                                with ui.expansion(
                                    short + "…", icon="description"
                                ).classes("w-full text-body2"):
                                    ui.label(abstract).classes("text-body2").style(
                                        "white-space: pre-wrap; word-break: break-word;"
                                    )
                            else:
                                ui.label(abstract).classes(
                                    "text-body2 text-grey-8"
                                ).style(
                                    "white-space: pre-wrap; word-break: break-word;"
                                )

                        # Full-text viewer (Req 7.1, 7.4, 7.5)
                        full_text = rec.get("full_text") or ""
                        if full_text.strip():
                            with ui.expansion(
                                "View Full Text", icon="article"
                            ).classes("w-full"):
                                _ft_style = (
                                    "white-space: pre-wrap; "
                                    "word-break: break-word;"
                                )
                                if len(full_text) > 5000:
                                    with ui.scroll_area().style(
                                        "max-height: 400px;"
                                    ).classes("w-full"):
                                        ui.label(full_text).classes(
                                            "text-body2"
                                        ).style(_ft_style)
                                else:
                                    ui.label(full_text).classes(
                                        "text-body2"
                                    ).style(_ft_style)

        def _refresh_table() -> None:
            sorted_rows = _sort_results(
                panel_state["results"], panel_state["sort_criterion"]
            )
            _render_results(sorted_rows)
            n = len(panel_state["results"])
            count_label.set_text(f"{n} result(s)" if n else "No results yet")

        def set_results(results: list[dict[str, Any]]) -> None:
            panel_state["results"] = list(results)
            _refresh_table()

        container.set_results = set_results  # type: ignore[attr-defined]

        # Render any previously saved results
        if panel_state["results"]:
            _refresh_table()

        # Auto-refresh once background scoring finishes
        if rag_engine is not None and panel_state["results"]:
            def _check_scoring() -> None:
                if _scoring_done[0]:
                    _refresh_table()
                    _score_timer.deactivate()

            _score_timer = ui.timer(1.0, _check_scoring)
