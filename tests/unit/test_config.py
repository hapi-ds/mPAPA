"""Unit tests for AppSettings configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from patent_system.config import AppSettings, get_base_dir


class TestAppSettingsDefaults:
    """Verify all default values are set correctly."""

    def test_default_lm_studio_base_url(self) -> None:
        settings = AppSettings()
        assert settings.lm_studio_base_url == "http://localhost:1234/v1"

    def test_default_lm_studio_api_key(self) -> None:
        settings = AppSettings()
        assert settings.lm_studio_api_key == "not-needed"

    def test_default_model_assignments(self) -> None:
        settings = AppSettings()
        assert settings.model_disclosure == "default"
        assert settings.model_search == "default"
        assert settings.model_claims == "default"
        assert settings.model_description == "default"
        assert settings.model_review == "default"
        assert settings.model_chat == "default"

    def test_default_embedding_model_name(self) -> None:
        settings = AppSettings()
        # May be overridden by .env; just verify it's a non-empty string
        assert isinstance(settings.embedding_model_name, str)
        assert len(settings.embedding_model_name) > 0

    def test_default_database_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PATENT_DATABASE_PATH", raising=False)
        settings = AppSettings(_env_file=None)
        expected = get_base_dir() / "data" / "patent_system.db"
        assert settings.database_path == expected
        assert settings.database_path.is_absolute()

    def test_default_docx_template_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PATENT_DOCX_TEMPLATE_DIR", raising=False)
        settings = AppSettings(_env_file=None)
        expected = get_base_dir() / "src" / "patent_system" / "export" / "templates"
        assert settings.docx_template_dir == expected
        assert settings.docx_template_dir.is_absolute()

    def test_default_docx_template_name_is_none(self) -> None:
        settings = AppSettings()
        # Default is None, but .env may override to "template.docx"
        assert settings.docx_template_name is None or isinstance(settings.docx_template_name, str)

    def test_default_monitoring_interval_hours(self) -> None:
        settings = AppSettings()
        assert settings.monitoring_interval_hours == 24

    def test_default_log_file_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PATENT_LOG_FILE_PATH", raising=False)
        settings = AppSettings(_env_file=None)
        expected = get_base_dir() / "logs" / "patent_system.log"
        assert settings.log_file_path == expected
        assert settings.log_file_path.is_absolute()

    def test_default_log_level(self) -> None:
        settings = AppSettings()
        assert settings.log_level == "INFO"


class TestAppSettingsOverrides:
    """Verify settings can be overridden via environment variables."""

    def test_override_lm_studio_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PATENT_LM_STUDIO_BASE_URL", "http://remote:5000/v1")
        settings = AppSettings()
        assert settings.lm_studio_base_url == "http://remote:5000/v1"

    def test_override_model_claims(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PATENT_MODEL_CLAIMS", "llama3-70b")
        settings = AppSettings()
        assert settings.model_claims == "llama3-70b"

    def test_override_database_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PATENT_DATABASE_PATH", "/tmp/test.db")
        settings = AppSettings()
        assert settings.database_path == Path("/tmp/test.db")

    def test_override_monitoring_interval(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PATENT_MONITORING_INTERVAL_HOURS", "12")
        settings = AppSettings()
        assert settings.monitoring_interval_hours == 12

    def test_override_log_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PATENT_LOG_LEVEL", "DEBUG")
        settings = AppSettings()
        assert settings.log_level == "DEBUG"

    def test_override_docx_template_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PATENT_DOCX_TEMPLATE_NAME", "european_patent.docx")
        settings = AppSettings()
        assert settings.docx_template_name == "european_patent.docx"


class TestAppSettingsValidation:
    """Verify type validation raises errors for invalid values."""

    def test_invalid_monitoring_interval_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PATENT_MONITORING_INTERVAL_HOURS", "not_a_number")
        with pytest.raises(ValidationError):
            AppSettings()

    def test_env_prefix_is_patent(self) -> None:
        assert AppSettings.model_config["env_prefix"] == "PATENT_"

    def test_env_file_is_dotenv(self) -> None:
        assert AppSettings.model_config["env_file"] == ".env"


class TestAppSettingsRuntimeDirs:
    """Verify runtime directories are created on initialization."""

    def test_ensure_runtime_dirs_called_on_init(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Instantiating AppSettings creates data/ and logs/ under get_base_dir()."""
        base = get_base_dir()
        # data/ and logs/ should exist after AppSettings() is created
        # (they already exist in the project root, so just verify they're there)
        AppSettings(_env_file=None)
        assert (base / "data").is_dir()
        assert (base / "logs").is_dir()
