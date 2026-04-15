"""Unit tests for get_base_dir() and ensure_runtime_dirs() in config.py.

Tests runtime path resolution for both source and compiled (frozen) modes,
and verifies that ensure_runtime_dirs() creates the required directories
idempotently.

Validates: Requirements 4.1, 4.2, 4.3, 4.5, 4.6
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from patent_system.config import ensure_runtime_dirs, get_base_dir


class TestGetBaseDirSourceMode:
    """Tests for get_base_dir() when running from source (not compiled)."""

    def test_source_mode_returns_project_root(self) -> None:
        """In source mode, get_base_dir() returns the project root directory."""
        result = get_base_dir()
        # config.py is at src/patent_system/config.py → parents[2] is project root
        assert result == Path(__file__).resolve().parents[2]

    def test_source_mode_returns_absolute_path(self) -> None:
        """The returned path is always absolute."""
        result = get_base_dir()
        assert result.is_absolute()

    def test_source_mode_contains_src_directory(self) -> None:
        """The project root should contain the src/ directory."""
        result = get_base_dir()
        assert (result / "src").is_dir()

    def test_source_mode_without_frozen_attr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When sys.frozen is absent, get_base_dir() returns the project root."""
        monkeypatch.delattr(sys, "frozen", raising=False)
        result = get_base_dir()
        expected = Path(__file__).resolve().parents[2]
        assert result == expected


class TestGetBaseDirCompiledMode:
    """Tests for get_base_dir() when running as a compiled executable."""

    def test_frozen_mode_returns_exe_parent(self, tmp_path: Path) -> None:
        """When sys.frozen is set, get_base_dir() returns the exe's parent dir."""
        fake_exe = tmp_path / "dist" / "main.exe"
        fake_exe.parent.mkdir(parents=True, exist_ok=True)
        fake_exe.touch()

        with (
            patch.object(sys, "frozen", True, create=True),
            patch.object(sys, "argv", [str(fake_exe)]),
        ):
            result = get_base_dir()

        assert result == fake_exe.resolve().parent

    def test_frozen_mode_returns_absolute_path(self, tmp_path: Path) -> None:
        """The returned path is always absolute in compiled mode."""
        fake_exe = tmp_path / "app.exe"
        fake_exe.touch()

        with (
            patch.object(sys, "frozen", True, create=True),
            patch.object(sys, "argv", [str(fake_exe)]),
        ):
            result = get_base_dir()

        assert result.is_absolute()

    def test_frozen_mode_with_nested_exe_path(self, tmp_path: Path) -> None:
        """Compiled mode works with deeply nested executable paths."""
        fake_exe = tmp_path / "release" / "v1" / "bin" / "main.exe"
        fake_exe.parent.mkdir(parents=True, exist_ok=True)
        fake_exe.touch()

        with (
            patch.object(sys, "frozen", True, create=True),
            patch.object(sys, "argv", [str(fake_exe)]),
        ):
            result = get_base_dir()

        assert result == fake_exe.resolve().parent
        assert result.name == "bin"


class TestEnsureRuntimeDirs:
    """Tests for ensure_runtime_dirs() directory creation and idempotency."""

    def test_creates_data_and_logs_dirs(self, tmp_path: Path) -> None:
        """ensure_runtime_dirs() creates data/ and logs/ under the base path."""
        ensure_runtime_dirs(tmp_path)

        assert (tmp_path / "data").is_dir()
        assert (tmp_path / "logs").is_dir()

    def test_idempotent_when_dirs_exist(self, tmp_path: Path) -> None:
        """Calling ensure_runtime_dirs() twice does not raise an error."""
        ensure_runtime_dirs(tmp_path)
        # Second call should be safe
        ensure_runtime_dirs(tmp_path)

        assert (tmp_path / "data").is_dir()
        assert (tmp_path / "logs").is_dir()

    def test_creates_nested_parent_directories(self, tmp_path: Path) -> None:
        """ensure_runtime_dirs() creates parent directories if needed."""
        deep_base = tmp_path / "level1" / "level2" / "level3"
        # deep_base does not exist yet
        assert not deep_base.exists()

        ensure_runtime_dirs(deep_base)

        assert (deep_base / "data").is_dir()
        assert (deep_base / "logs").is_dir()

    def test_preserves_existing_files_in_dirs(self, tmp_path: Path) -> None:
        """ensure_runtime_dirs() does not remove existing files in data/ or logs/."""
        (tmp_path / "data").mkdir()
        (tmp_path / "logs").mkdir()
        sentinel = tmp_path / "data" / "existing.db"
        sentinel.write_text("keep me")

        ensure_runtime_dirs(tmp_path)

        assert sentinel.read_text() == "keep me"

    def test_only_creates_data_and_logs(self, tmp_path: Path) -> None:
        """ensure_runtime_dirs() creates exactly data/ and logs/, nothing else."""
        ensure_runtime_dirs(tmp_path)

        created_dirs = {p.name for p in tmp_path.iterdir() if p.is_dir()}
        assert created_dirs == {"data", "logs"}


# ---------------------------------------------------------------------------
# Unit tests for build.py functions
# Validates: Requirements 1.1, 1.2, 1.3, 1.6, 2.1–2.7, 3.1, 3.2, 3.4, 3.5,
#            5.1, 5.3, 8.4
# ---------------------------------------------------------------------------

import argparse
import textwrap

# build.py lives at the project root — add it to sys.path so we can import it
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from build import (  # noqa: E402
    build_nuitka_command,
    load_nuitka_config,
    read_project_metadata,
    validate_prerequisites,
)


def _make_args(
    onefile: bool = False,
    output_dir: str | None = None,
    clean: bool = False,
) -> argparse.Namespace:
    """Create an argparse.Namespace mimicking CLI arguments."""
    return argparse.Namespace(onefile=onefile, output_dir=output_dir, clean=clean)


def _default_metadata() -> dict[str, str]:
    """Return a minimal valid metadata dict for testing."""
    return {
        "version": "1.0.0",
        "description": "Test application",
        "author_name": "Test Author",
    }


class TestLoadNuitkaConfig:
    """Tests for load_nuitka_config() reading [tool.nuitka] from pyproject.toml.

    Validates: Requirements 5.1, 5.2
    """

    def test_valid_nuitka_section(self, tmp_path: Path) -> None:
        """A valid [tool.nuitka] section is read correctly."""
        toml_path = tmp_path / "pyproject.toml"
        toml_path.write_text(
            textwrap.dedent("""\
                [tool.nuitka]
                output-dir = "build_out"
                include-package = ["nicegui", "dspy"]
                include-package-data = ["nicegui"]
                include-data-dir = ["src/templates=templates"]
                include-data-files = [".env.example=.env.example"]
                nofollow-import-to = ["tests", "pytest"]
            """),
            encoding="utf-8",
        )

        result = load_nuitka_config(toml_path)

        assert result["output-dir"] == "build_out"
        assert result["include-package"] == ["nicegui", "dspy"]
        assert result["include-package-data"] == ["nicegui"]
        assert result["include-data-dir"] == ["src/templates=templates"]
        assert result["include-data-files"] == [".env.example=.env.example"]
        assert result["nofollow-import-to"] == ["tests", "pytest"]

    def test_missing_nuitka_section_returns_empty_dict(self, tmp_path: Path) -> None:
        """When [tool.nuitka] is absent, an empty dict is returned."""
        toml_path = tmp_path / "pyproject.toml"
        toml_path.write_text(
            textwrap.dedent("""\
                [project]
                name = "test"
                version = "0.1.0"
            """),
            encoding="utf-8",
        )

        result = load_nuitka_config(toml_path)

        assert result == {}

    def test_reads_all_expected_keys(self, tmp_path: Path) -> None:
        """All expected configuration keys are present in the result."""
        toml_path = tmp_path / "pyproject.toml"
        toml_path.write_text(
            textwrap.dedent("""\
                [tool.nuitka]
                output-dir = "dist"
                include-package = ["patent_system"]
                include-package-data = ["nicegui"]
                include-data-dir = ["src/tpl=tpl"]
                include-data-files = [".env.example=.env.example"]
                nofollow-import-to = ["tests"]
            """),
            encoding="utf-8",
        )

        result = load_nuitka_config(toml_path)

        expected_keys = {
            "output-dir",
            "include-package",
            "include-package-data",
            "include-data-dir",
            "include-data-files",
            "nofollow-import-to",
        }
        assert expected_keys.issubset(result.keys())


class TestReadProjectMetadata:
    """Tests for read_project_metadata() extracting [project] metadata.

    Validates: Requirements 9.2, 9.3, 9.4
    """

    def test_valid_project_section(self, tmp_path: Path) -> None:
        """A valid [project] section returns the correct metadata dict."""
        toml_path = tmp_path / "pyproject.toml"
        toml_path.write_text(
            textwrap.dedent("""\
                [project]
                name = "my-app"
                version = "2.3.4"
                description = "A great app"
                authors = [{name = "Jane Doe", email = "jane@example.com"}]
            """),
            encoding="utf-8",
        )

        result = read_project_metadata(toml_path)

        assert result["version"] == "2.3.4"
        assert result["description"] == "A great app"
        assert result["author_name"] == "Jane Doe"

    def test_missing_project_section_raises_system_exit(self, tmp_path: Path) -> None:
        """Missing [project] section causes SystemExit."""
        toml_path = tmp_path / "pyproject.toml"
        toml_path.write_text(
            textwrap.dedent("""\
                [tool.nuitka]
                output-dir = "dist"
            """),
            encoding="utf-8",
        )

        with pytest.raises(SystemExit):
            read_project_metadata(toml_path)

    def test_missing_version_raises_system_exit(self, tmp_path: Path) -> None:
        """Missing 'version' field in [project] causes SystemExit."""
        toml_path = tmp_path / "pyproject.toml"
        toml_path.write_text(
            textwrap.dedent("""\
                [project]
                name = "my-app"
                description = "A great app"
                authors = [{name = "Jane Doe"}]
            """),
            encoding="utf-8",
        )

        with pytest.raises(SystemExit):
            read_project_metadata(toml_path)

    def test_missing_authors_raises_system_exit(self, tmp_path: Path) -> None:
        """Missing 'authors' field in [project] causes SystemExit."""
        toml_path = tmp_path / "pyproject.toml"
        toml_path.write_text(
            textwrap.dedent("""\
                [project]
                name = "my-app"
                version = "1.0.0"
                description = "A great app"
            """),
            encoding="utf-8",
        )

        with pytest.raises(SystemExit):
            read_project_metadata(toml_path)


class TestValidatePrerequisites:
    """Tests for validate_prerequisites() checking Nuitka, NiceGUI, and data files.

    Validates: Requirements 1.6, 3.5, 8.4
    """

    def test_nuitka_not_importable_raises_system_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When Nuitka is not importable, validate_prerequisites() exits."""
        import builtins

        original_import = builtins.__import__

        def _mock_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "nuitka":
                raise ImportError("No module named 'nuitka'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)

        with pytest.raises(SystemExit):
            validate_prerequisites()

    def test_nicegui_not_importable_raises_system_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When NiceGUI is not importable, validate_prerequisites() exits."""
        import builtins

        original_import = builtins.__import__

        def _mock_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "nicegui":
                raise ImportError("No module named 'nicegui'")
            return original_import(name, *args, **kwargs)

        # Nuitka must be importable for us to reach the NiceGUI check.
        # We mock nuitka to succeed and nicegui to fail.
        def _selective_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "nuitka":
                # Return a dummy module object
                import types

                return types.ModuleType("nuitka")
            if name == "nicegui":
                raise ImportError("No module named 'nicegui'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _selective_import)

        with pytest.raises(SystemExit):
            validate_prerequisites()

    def test_nicegui_static_dir_missing_raises_system_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When NiceGUI's static directory is missing, validate_prerequisites() exits."""
        import types

        import builtins

        original_import = builtins.__import__

        # Create a fake nicegui module with a __path__ pointing to a non-existent dir
        fake_nicegui = types.ModuleType("nicegui")
        fake_nicegui.__path__ = ["/nonexistent/path/nicegui"]  # type: ignore[attr-defined]

        def _selective_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "nuitka":
                return types.ModuleType("nuitka")
            if name == "nicegui":
                return fake_nicegui
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _selective_import)

        with pytest.raises(SystemExit):
            validate_prerequisites()

    def test_missing_data_files_logs_warnings_no_exit(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Missing data files produce warnings but do not cause an exit."""
        import types

        import builtins

        original_import = builtins.__import__

        # Create a fake nicegui with a valid static dir
        nicegui_dir = tmp_path / "nicegui"
        static_dir = nicegui_dir / "static"
        static_dir.mkdir(parents=True)

        fake_nicegui = types.ModuleType("nicegui")
        fake_nicegui.__path__ = [str(nicegui_dir)]  # type: ignore[attr-defined]

        def _selective_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "nuitka":
                return types.ModuleType("nuitka")
            if name == "nicegui":
                return fake_nicegui
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _selective_import)

        # Point to a non-existent templates dir and missing .env.example
        monkeypatch.chdir(tmp_path)

        import logging

        with caplog.at_level(logging.WARNING):
            # Should NOT raise SystemExit
            validate_prerequisites()

        # Verify warnings were logged for missing data files
        assert any("Templates directory not found" in msg or "Missing data file" in msg for msg in caplog.messages)


class TestBuildNuitkaCommand:
    """Tests for build_nuitka_command() constructing the Nuitka CLI invocation.

    Validates: Requirements 1.1, 1.2, 1.3, 2.1–2.7, 3.1, 3.2, 3.4, 5.1, 5.3
    """

    def test_command_starts_with_standalone(self) -> None:
        """Command always starts with [sys.executable, '-m', 'nuitka', '--standalone']."""
        config: dict = {"output-dir": "dist"}
        cmd = build_nuitka_command(config, _make_args(), _default_metadata())

        assert cmd[0] == sys.executable
        assert cmd[1] == "-m"
        assert cmd[2] == "nuitka"
        assert cmd[3] == "--standalone"

    def test_onefile_present_when_true(self) -> None:
        """--onefile is in the command when the flag is True."""
        config: dict = {"output-dir": "dist"}
        cmd = build_nuitka_command(config, _make_args(onefile=True), _default_metadata())

        assert "--onefile" in cmd

    def test_onefile_absent_when_false(self) -> None:
        """--onefile is NOT in the command when the flag is False."""
        config: dict = {"output-dir": "dist"}
        cmd = build_nuitka_command(config, _make_args(onefile=False), _default_metadata())

        assert "--onefile" not in cmd

    def test_all_hidden_imports_included(self) -> None:
        """All hidden imports from requirement 2 are present when config includes them."""
        required_packages = [
            "patent_system",
            "nicegui",
            "langgraph",
            "langgraph.checkpoint",
            "llama_index.core",
            "dspy",
            "pydantic",
            "pydantic_settings",
            "docx",
        ]
        config: dict = {
            "output-dir": "dist",
            "include-package": required_packages,
        }
        cmd = build_nuitka_command(config, _make_args(), _default_metadata())

        for pkg in required_packages:
            assert f"--include-package={pkg}" in cmd, f"Missing hidden import: {pkg}"

    def test_output_dir_uses_config_when_cli_not_set(self) -> None:
        """--output-dir uses the config value when CLI doesn't override."""
        config: dict = {"output-dir": "my_build"}
        cmd = build_nuitka_command(config, _make_args(output_dir=None), _default_metadata())

        assert "--output-dir=my_build" in cmd

    def test_output_dir_uses_cli_when_both_set(self) -> None:
        """--output-dir uses the CLI value when both config and CLI are set."""
        config: dict = {"output-dir": "config_dir"}
        cmd = build_nuitka_command(config, _make_args(output_dir="cli_dir"), _default_metadata())

        assert "--output-dir=cli_dir" in cmd
        assert "--output-dir=config_dir" not in cmd

    def test_metadata_flags_present(self) -> None:
        """Product metadata flags are present in the command."""
        metadata = {
            "version": "3.2.1",
            "description": "Patent drafting tool",
            "author_name": "Hans Koehler",
        }
        config: dict = {"output-dir": "dist"}
        cmd = build_nuitka_command(config, _make_args(), metadata)

        assert "--product-name=mPAPA - my Personal Artificial Patent Agent" in cmd
        assert "--file-version=3.2.1" in cmd
        assert "--product-version=3.2.1" in cmd
        assert "--company-name=Hans Koehler" in cmd
        assert "--file-description=Patent drafting tool" in cmd

    def test_entry_point_is_always_last(self) -> None:
        """The entry point src/patent_system/main.py is always the last element."""
        config: dict = {
            "output-dir": "dist",
            "include-package": ["nicegui", "dspy"],
            "nofollow-import-to": ["tests"],
        }
        cmd = build_nuitka_command(config, _make_args(onefile=True), _default_metadata())

        assert cmd[-1] == "src/patent_system/main.py"

    def test_data_dir_and_data_files_included(self) -> None:
        """--include-data-dir and --include-data-files entries from config are present."""
        config: dict = {
            "output-dir": "dist",
            "include-data-dir": [
                "src/patent_system/export/templates=patent_system/export/templates",
            ],
            "include-data-files": [
                ".env.example=.env.example",
            ],
        }
        cmd = build_nuitka_command(config, _make_args(), _default_metadata())

        assert "--include-data-dir=src/patent_system/export/templates=patent_system/export/templates" in cmd
        assert "--include-data-files=.env.example=.env.example" in cmd

    def test_nofollow_import_to_included(self) -> None:
        """--nofollow-import-to entries from config are present."""
        config: dict = {
            "output-dir": "dist",
            "nofollow-import-to": ["tests", "pytest", "hypothesis"],
        }
        cmd = build_nuitka_command(config, _make_args(), _default_metadata())

        assert "--nofollow-import-to=tests" in cmd
        assert "--nofollow-import-to=pytest" in cmd
        assert "--nofollow-import-to=hypothesis" in cmd
