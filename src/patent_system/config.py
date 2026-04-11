"""Application settings loaded from environment variables and .env file.

Uses Pydantic Settings with the PATENT_ prefix for all environment variables.
All settings have sensible defaults and are optional. A ValidationError is
raised at startup if any value fails type validation.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

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
    embedding_model_name: str = "BAAI/bge-large-en-v1.5"

    # Database
    database_path: Path = Path("data/patent_system.db")

    # DOCX export
    docx_template_dir: Path = Path("src/patent_system/export/templates")
    docx_template_name: str | None = None

    # Monitoring
    monitoring_interval_hours: int = 24

    # Logging
    log_file_path: Path = Path("logs/patent_system.log")
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_prefix": "PATENT_"}
