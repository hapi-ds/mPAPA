"""Application settings loaded from environment variables and .env file.

Uses Pydantic Settings with the PATENT_ prefix for all environment variables.
All settings have sensible defaults and are optional. A ValidationError is
raised at startup if any value fails type validation.
"""

import sys
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

from patent_system.agents.personality import PersonalityMode


def get_base_dir() -> Path:
    """Return the application base directory.

    Detects whether the app is running as a Nuitka-compiled executable or
    from source and returns the correct root directory for resolving all
    relative paths (database, logs, templates, etc.).

    When compiled:
        Returns the directory containing the ``.exe`` file, determined from
        ``sys.argv[0]``. Nuitka sets ``__compiled__`` on compiled modules;
        ``sys.frozen`` is checked as a fallback for other freezers.

    When running from source:
        Returns the project root, which is two levels up from this file's
        location (``src/patent_system/`` → project root).

    Returns:
        Path: The resolved base directory for the application.
    """
    # Primary: Nuitka sets __compiled__ on compiled modules
    # Fallback: PyInstaller / other freezers set sys.frozen
    if "__compiled__" in dir() or getattr(sys, "frozen", False):
        return Path(sys.argv[0]).resolve().parent

    return Path(__file__).resolve().parents[2]


def ensure_runtime_dirs(base: Path) -> None:
    """Create required runtime directories under the application base.

    Ensures that ``data/`` and ``logs/`` subdirectories exist beneath *base*.
    Uses ``parents=True`` and ``exist_ok=True`` so the call is idempotent and
    safe to invoke on every startup.

    Args:
        base: The application base directory (as returned by
            :func:`get_base_dir`).
    """
    (base / "data").mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(parents=True, exist_ok=True)


def _default_database_path() -> Path:
    """Return the default database path resolved relative to the base dir."""
    return get_base_dir() / "data" / "patent_system.db"


def _default_log_file_path() -> Path:
    """Return the default log file path resolved relative to the base dir."""
    return get_base_dir() / "logs" / "patent_system.log"


def _default_pdf_download_dir() -> Path:
    """Return the default PDF download directory resolved relative to the base dir."""
    return get_base_dir() / "data" / "pdfs"


def _default_domain_profiles_dir() -> Path:
    """Return the default domain profiles directory resolved relative to the base dir."""
    return get_base_dir() / "domain_profiles"


def _default_docx_template_dir() -> Path:
    """Return the default DOCX template directory resolved relative to the base dir.

    When compiled, templates are bundled alongside the executable under
    ``patent_system/export/templates``.  When running from source, they
    live under ``src/patent_system/export/templates``.
    """
    base = get_base_dir()
    if "__compiled__" in dir() or getattr(sys, "frozen", False):
        return base / "patent_system" / "export" / "templates"
    return base / "src" / "patent_system" / "export" / "templates"


class AppSettings(BaseSettings):
    """Application settings loaded from environment variables and .env file.

    Path-based settings (``database_path``, ``log_file_path``,
    ``docx_template_dir``) are resolved relative to :func:`get_base_dir`
    so they work correctly both when running from source and when running
    as a Nuitka-compiled executable.

    On initialization, :func:`ensure_runtime_dirs` is called to guarantee
    that the ``data/`` and ``logs/`` directories exist.
    """

    # LM Studio
    lm_studio_base_url: str = "http://localhost:1234/v1"
    lm_studio_api_key: str = "not-needed"

    # Model assignments per agent task
    model_disclosure: str = "default"
    model_search: str = "default"
    model_claims: str = "default"
    model_description: str = "default"
    model_review: str = "default"
    model_chat: str = "default"

    # Embedding
    embedding_model_name: str = "text-embedding-nomic-embed-text-v1.5"

    # Database
    database_path: Path = Field(default_factory=_default_database_path)

    # DOCX export
    docx_template_dir: Path = Field(default_factory=_default_docx_template_dir)
    docx_template_name: str | None = None

    # Monitoring
    monitoring_interval_hours: int = 24

    # Search
    search_max_results_per_source: int = 10
    search_request_delay_seconds: float = 3.0
    search_relevance_top_k: int = 200

    # Full-text download & vectorization
    full_text_download_enabled: bool = True
    vectorization_text_limit: int = 4000
    pdf_download_dir: Path = Field(default_factory=_default_pdf_download_dir)

    # EPO Open Patent Services (OPS)
    epo_ops_key: str = ""
    epo_ops_secret: str = ""

    # Web server
    nicegui_port: int = 8080
    nicegui_reload: bool = False

    # Logging
    log_file_path: Path = Field(default_factory=_default_log_file_path)
    log_level: str = "INFO"

    # Domain profiles
    domain_profiles_dir: Path = Field(default_factory=_default_domain_profiles_dir)
    default_domain_profile: str = "general-patent-drafting"

    # Personality modes
    default_personality_mode: PersonalityMode = PersonalityMode.CRITICAL
    agent_personality_overrides: str = ""  # JSON string, e.g. '{"novelty_analysis": "neutral"}'

    model_config = {"env_file": ".env", "env_prefix": "PATENT_"}

    def model_post_init(self, __context: object) -> None:
        """Create runtime directories after settings are initialized.

        Calls :func:`ensure_runtime_dirs` with the application base
        directory so that ``data/`` and ``logs/`` exist before any other
        code attempts to write to them.  Also ensures the PDF download
        directory exists.
        """
        ensure_runtime_dirs(get_base_dir())
        self.pdf_download_dir.mkdir(parents=True, exist_ok=True)
