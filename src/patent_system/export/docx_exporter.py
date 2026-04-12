"""DOCX generation with optional template support.

Generates .docx patent documents containing claims and description sections.
Supports user-provided .docx templates for custom formatting; falls back to
a blank Document when no template is configured or the file is missing.
"""

import logging
from pathlib import Path

from docx import Document

logger = logging.getLogger(__name__)


def validate_export(claims: str | None, description: str | None) -> bool:
    """Check whether claims and description are non-empty and suitable for export.

    Args:
        claims: The patent claims text.
        description: The patent description text.

    Returns:
        True if both claims and description are non-empty strings, False otherwise.
    """
    if not claims or not description:
        return False
    return True


class DOCXExporter:
    """Generates .docx patent documents using optional templates from the templates directory."""

    def __init__(self, template_dir: Path, template_name: str | None = None) -> None:
        """Initialize the exporter.

        Args:
            template_dir: Path to the templates directory containing .docx template files.
            template_name: Filename of the template to use (e.g. "european_patent.docx").
                           If None or file not found, starts with a blank Document().
        """
        self._template_dir = template_dir
        self._template_name = template_name

    def _resolve_template(self) -> Path | None:
        """Resolve the template path from template_dir + template_name.

        Returns:
            The full Path to the template file, or None if template_name is None
            or the file does not exist.
        """
        if self._template_name is None:
            return None

        template_path = self._template_dir / self._template_name
        if not template_path.exists():
            logger.warning(
                "Configured template '%s' not found in '%s'; falling back to blank document.",
                self._template_name,
                self._template_dir,
            )
            return None

        return template_path

    def export(
        self,
        claims: str,
        description: str,
        output_path: Path,
        references: list[dict] | None = None,
    ) -> Path:
        """Generate a .docx file with claims, description, and references.

        Args:
            claims: The patent claims text.
            description: The patent description text.
            output_path: Destination path for the generated .docx file.
            references: Optional list of dicts with 'title', 'abstract',
                'source', and optionally 'patent_number' or 'doi' keys.

        Returns:
            The output_path where the document was saved.
        """
        template_path = self._resolve_template()

        if template_path is not None:
            doc = Document(str(template_path))
        else:
            doc = Document()

        doc.add_heading("Claims", level=1)
        doc.add_paragraph(claims)

        doc.add_heading("Description", level=1)
        doc.add_paragraph(description)

        if references:
            doc.add_heading("References", level=1)
            for i, ref in enumerate(references, 1):
                title = ref.get("title", "Untitled")
                source = ref.get("source", "")
                record_id = ref.get("patent_number") or ref.get("doi") or ""
                abstract = ref.get("abstract", "")

                heading = f"[{i}] {title}"
                if source:
                    heading += f" ({source})"
                if record_id and record_id != "UNKNOWN":
                    heading += f" — {record_id}"

                doc.add_heading(heading, level=2)
                if abstract:
                    doc.add_paragraph(abstract)

        # Ensure parent directories exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(output_path))

        return output_path

    def list_available_templates(self) -> list[str]:
        """List all .docx files in the templates directory.

        Returns:
            A list of .docx filenames found in the template directory.
            Returns an empty list if the directory does not exist.
        """
        if not self._template_dir.exists():
            return []

        return sorted(f.name for f in self._template_dir.iterdir() if f.suffix == ".docx")
