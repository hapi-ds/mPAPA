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

logger = logging.getLogger(__name__)

# Source names that produce scientific papers (vs patents)
_PAPER_SOURCES = {
    name for name, info in _SOURCE_REGISTRY.items() if info["type"] == "paper"
}

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


def _sort_results(
    results: list[dict[str, Any]],
    criterion: str,
) -> list[dict[str, Any]]:
    """Sort search results by the given criterion."""
    if criterion == "discovery_date":
        return sorted(results, key=lambda r: _normalize_date_key(r.get("discovered_date")), reverse=True)
    if criterion == "relevance":
        return sorted(results, key=lambda r: r.get("relevance_score", 0), reverse=True)
    if criterion == "citation_count":
        return sorted(results, key=lambda r: r.get("citation_count", 0), reverse=True)
    return list(results)



def create_research_panel(
    container: Any,
    topic_id: int,
    conn: sqlite3.Connection | None = None,
    rag_engine: Any | None = None,
    disclosure_repo: InventionDisclosureRepository | None = None,
    source_pref_repo: SourcePreferenceRepository | None = None,
    max_results_per_source: int = 10,
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
    """
    container.clear()

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
                rag_docs = []
                for rec in _results_ref:
                    title = rec.get("title", "")
                    abstract = rec.get("abstract", "")
                    full_text = rec.get("full_text")
                    doc_text = _build_rag_document_text(abstract, full_text)
                    text = f"{title} {doc_text}".strip() if title else doc_text
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
                    rag_results = rag_engine.query(topic_id, _desc_ref, top_k=50)
                    score_map: dict[str, float] = {}
                    for rr in rag_results:
                        rr_text = rr.get("text", "")
                        rr_score = rr.get("score", 0.0) or 0.0
                        for rec in _results_ref:
                            rec_title = rec.get("title", "")
                            if rec_title and rec_title in rr_text and rec_title not in score_map:
                                score_map[rec_title] = rr_score
                    for rec in _results_ref:
                        title = rec.get("title", "")
                        if title in score_map:
                            rec["relevance_score"] = round(score_map[title] * 100, 1)
            except Exception:
                logger.debug("Failed to index/score saved results", exc_info=True)
            finally:
                _scoring_done[0] = True

        import threading
        _scoring_done = [False]
        threading.Thread(target=_bg_index_and_score, daemon=True).start()

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

        def _extract_text(filename: str, content_bytes: bytes) -> str:
            """Extract text from uploaded file bytes.

            For PDFs, tries multiple extraction methods:
            1. Standard text extraction
            2. Text from annotations/widgets
            3. Raw text blocks
            If all fail, returns empty string (likely a scanned image PDF).
            """
            lower = filename.lower()
            if lower.endswith(".txt"):
                return content_bytes.decode("utf-8", errors="replace")
            elif lower.endswith(".pdf"):
                import fitz  # PyMuPDF
                try:
                    doc = fitz.open(stream=content_bytes, filetype="pdf")
                    text_parts: list[str] = []
                    for page in doc:
                        # Method 1: standard text extraction
                        text = page.get_text("text")
                        if text and text.strip():
                            text_parts.append(text)
                            continue
                        # Method 2: OCR fallback for scanned pages
                        try:
                            tp = page.get_textpage_ocr(full=True)
                            ocr_text = page.get_text("text", textpage=tp)
                            if ocr_text and ocr_text.strip():
                                text_parts.append(ocr_text)
                                continue
                        except Exception:
                            pass
                    doc.close()
                    result = "\n".join(text_parts)
                    if not result.strip():
                        logger.warning(
                            "PDF '%s' — no text extracted even with OCR",
                            filename,
                        )
                    return result
                except Exception:
                    logger.exception("Failed to parse PDF '%s'", filename)
                    return ""
            elif lower.endswith(".docx"):
                import io
                from docx import Document as DocxDoc
                doc = DocxDoc(io.BytesIO(content_bytes))
                return "\n".join(p.text for p in doc.paragraphs if p.text)
            return ""

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

                def _process() -> dict | None:
                    """Run in thread: extract, persist, embed, index."""
                    text = _extract_text(filename, content_bytes)
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
                            emb = rag_engine._embedding_service.generate_embedding(text[:2000])
                            if emb:
                                doc_repo.update_embedding(row_id, emb)
                        except Exception:
                            logger.debug("Embedding failed for %s", filename, exc_info=True)
                        try:
                            rag_engine.index_documents(topic_id, [
                                {"text": f"{filename} {text[:4000]}", "metadata": {"source": "Local Document", "filename": filename}},
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
                    return

                panel_state["results"].append(result)
                import_status.set_text(f"✓ Imported {filename} ({result['word_count']:,} words)")
                logger.info("Imported '%s' for topic %d (%d words)", filename, topic_id, result["word_count"])
                _refresh_table()
            except Exception:
                logger.exception("Failed to import uploaded file")
                import_status.set_text("⚠ Import failed")

        ui.upload(
            label="Upload PDF, DOCX, or TXT — files are imported automatically",
            on_upload=_on_file_uploaded,
            multiple=True,
            auto_upload=True,
        ).props('accept=".pdf,.docx,.txt"').classes("w-full").style("max-height: 100px;")

        # --- Status and Start Research ---
        status_label = ui.label("").classes("text-caption q-mt-sm")

        async def _on_start_research() -> None:
            """Handle the Start Research button click.

            Validates input, saves disclosure, runs search with selected
            sources, deduplicates results, persists records, and indexes
            in RAG.

            Requirements: 1.6, 1.7, 2.3, 2.5, 4.1, 5.1, 5.2, 5.3, 6.1, 6.2
            """
            # Validate primary description (Req 1.7)
            description = primary_input.value.strip() if primary_input.value else ""
            if not description:
                ui.notify(
                    "Please enter a primary invention description.",
                    type="warning",
                )
                return

            # Gather additional search terms
            terms = list(panel_state["term_inputs"])

            # Save disclosure (Req 2.3)
            if disclosure_repo is not None:
                try:
                    disclosure_repo.upsert(topic_id, description, terms)
                except Exception:
                    logger.exception(
                        "Failed to save disclosure for topic %d", topic_id
                    )
                    ui.notify(
                        "Failed to save invention disclosure. Your data is retained in the form.",
                        type="negative",
                    )
                    return

            status_label.set_text("Searching…")
            logger.info("Start Research for topic %d", topic_id)

            # Build disclosure dict (Req 1.6)
            invention_disclosure = _build_disclosure_dict(description, terms)

            # Determine enabled sources (Req 4.1)
            selected_sources = [
                name for name, enabled in current_prefs.items() if enabled
            ]

            # Build workflow state
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

            # Run the blocking search in a background thread so the
            # NiceGUI event loop stays responsive and the WebSocket
            # connection is not dropped.
            result = await asyncio.to_thread(
                prior_art_search_node,
                state,
                rag_engine=rag_engine,
                selected_sources=selected_sources,
                max_results_per_source=max_results_per_source,
            )

            results = result.get("prior_art_results", [])
            failed = result.get("failed_sources", [])

            # Deduplicate against existing records (Req 6.1, 6.2)
            new_results: list[dict[str, Any]] = []
            for rec in results:
                if not _is_duplicate(rec, panel_state["results"]):
                    new_results.append(rec)

            status_label.set_text(f"Saving {len(new_results)} results…")

            # Run persistence, embeddings, and RAG indexing in a background
            # thread so the event loop stays responsive.
            def _persist_and_index() -> None:
                """Persist results to DB, generate embeddings, index in RAG."""
                # Persist search session and results to DB (Req 5.1)
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
                                    session_id=session_id,
                                    doi=rec.get("doi", ""),
                                    title=title,
                                    abstract=abstract,
                                    full_text=full_text,
                                    pdf_path=rec.get("pdf_path"),
                                    source=source_name,
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
                                    title=title,
                                    abstract=abstract,
                                    full_text=full_text,
                                    source=source_name,
                                )
                                row_id = patent_repo.create(session_id, patent_record)
                                rec["id"] = row_id
                                rec["record_type"] = "patent"
                                repo_for_emb = patent_repo

                            # Generate and store embedding
                            if rag_engine is not None:
                                try:
                                    emb_text = f"{title} {abstract or ''}"
                                    if full_text:
                                        emb_text += f" {full_text}"
                                    emb = rag_engine._embedding_service.generate_embedding(emb_text)
                                    if emb is not None:
                                        repo_for_emb.update_embedding(row_id, emb)
                                except Exception:
                                    logger.debug(
                                        "Failed to generate embedding for record %d",
                                        row_id,
                                        exc_info=True,
                                    )
                    except Exception:
                        logger.exception(
                            "Failed to persist search results for topic %d", topic_id
                        )

                # Index results in RAG with full text (Req 5.3)
                if rag_engine is not None and new_results:
                    rag_docs = []
                    for rec in new_results:
                        title = rec.get("title", "")
                        abstract = rec.get("abstract", "")
                        full_text = rec.get("full_text")
                        doc_text = _build_rag_document_text(abstract, full_text)
                        text = f"{title} {doc_text}".strip() if title else doc_text
                        if text:
                            metadata = _sanitize_metadata({
                                k: v
                                for k, v in rec.items()
                                if k not in ("title", "abstract", "full_text", "embedding")
                            })
                            rag_docs.append({"text": text, "metadata": metadata})
                    if rag_docs:
                        try:
                            rag_engine.index_documents(topic_id, rag_docs)
                            logger.info(
                                "Indexed %d docs in RAG for topic %d",
                                len(rag_docs),
                                topic_id,
                            )
                        except Exception:
                            logger.exception(
                                "Failed to index in RAG for topic %d", topic_id
                            )

                # Compute relevance scores
                if rag_engine is not None and description:
                    try:
                        rag_results = rag_engine.query(topic_id, description, top_k=50)
                        score_map: dict[str, float] = {}
                        for rr in rag_results:
                            rr_text = rr.get("text", "")
                            rr_score = rr.get("score", 0.0) or 0.0
                            for rec in new_results + panel_state["results"]:
                                rec_title = rec.get("title", "")
                                if rec_title and rec_title in rr_text and rec_title not in score_map:
                                    score_map[rec_title] = rr_score
                        for rec in new_results + panel_state["results"]:
                            title = rec.get("title", "")
                            if title in score_map:
                                rec["relevance_score"] = round(score_map[title] * 100, 1)
                    except Exception:
                        logger.debug("Failed to compute relevance scores", exc_info=True)

            await asyncio.to_thread(_persist_and_index)

            # Append new (non-duplicate) results to existing ones
            panel_state["results"].extend(new_results)
            _refresh_table()

            # Status feedback
            parts = []
            parts.append(f"{len(new_results)} new result(s) found")
            if len(results) - len(new_results) > 0:
                parts.append(f"{len(results) - len(new_results)} duplicate(s) skipped")
            if failed:
                parts.append(
                    f"{len(failed)} source(s) unavailable: {', '.join(failed)}"
                )
            status_label.set_text(" · ".join(parts))

            if not new_results:
                ui.notify(
                    "No new results found.",
                    type="info",
                    close_button=True,
                )

        ui.button(
            "Start Research", on_click=_on_start_research
        ).props("color=primary").classes("q-mt-sm")

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
