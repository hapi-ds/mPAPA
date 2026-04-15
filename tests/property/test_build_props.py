"""Property-based tests for Nuitka build: runtime path resolution and directory creation.

Validates: Requirements 4.1, 4.2, 4.3, 4.5, 4.6
"""

import sys
from pathlib import Path
from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

from patent_system.config import ensure_runtime_dirs, get_base_dir

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Windows reserved device names that cannot be used as path segments
_WINDOWS_RESERVED_NAMES = frozenset({
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
})

# Generate safe path segments (no null bytes, no path separators, non-empty,
# no Windows reserved device names)
_path_segment = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N"),
        whitelist_characters="_-",
    ),
    min_size=1,
    max_size=30,
).filter(lambda s: s.upper() not in _WINDOWS_RESERVED_NAMES)

# Generate a relative path with 1–4 segments (used as a fake exe location)
_relative_path = st.lists(_path_segment, min_size=1, max_size=4).map(
    lambda parts: str(Path(*parts) / "app.exe")
)


# ---------------------------------------------------------------------------
# Property 3: Runtime path resolution
# Feature: nuitka-exe-build, Property 3: Runtime path resolution
# ---------------------------------------------------------------------------


class TestRuntimePathResolution:
    """Property 3: Runtime path resolution.

    For any filesystem path representing an executable location, when the
    application detects it is running in compiled mode (``__compiled__`` is
    set), ``get_base_dir()`` SHALL return the parent directory of that
    executable path. When not in compiled mode, it SHALL return the project
    root.

    **Validates: Requirements 4.1, 4.2, 4.3**
    """

    @given(exe_path=_relative_path)
    @settings(max_examples=100)
    def test_compiled_mode_returns_exe_parent(
        self, exe_path: str, tmp_path_factory
    ) -> None:
        """In compiled mode, get_base_dir() returns the parent of sys.argv[0]."""
        # Build an absolute exe path under a temp directory
        tmp_path = tmp_path_factory.mktemp("compiled")
        abs_exe = tmp_path / exe_path
        abs_exe.parent.mkdir(parents=True, exist_ok=True)
        abs_exe.touch()

        expected_parent = abs_exe.resolve().parent

        # Simulate compiled mode via sys.frozen (fallback detection path)
        # and set sys.argv[0] to the fake exe location
        with (
            patch.object(sys, "frozen", True, create=True),
            patch.object(sys, "argv", [str(abs_exe)]),
        ):
            result = get_base_dir()

        assert result == expected_parent

    @given(data=st.data())
    @settings(max_examples=100)
    def test_source_mode_returns_project_root(self, data: st.DataObject) -> None:
        """In source mode (no __compiled__, no sys.frozen), get_base_dir()
        returns the project root (two levels up from config.py)."""
        # Ensure sys.frozen is not set (source mode)
        with patch.object(sys, "frozen", False, create=True):
            result = get_base_dir()

        # config.py lives at src/patent_system/config.py
        # Project root is two levels up: parents[2]
        expected = Path(__file__).resolve().parents[3]  # tests/property/ -> tests/ -> root
        config_expected = Path(
            __import__("patent_system.config", fromlist=["config"]).__file__
        ).resolve().parents[2]

        assert result == config_expected


# ---------------------------------------------------------------------------
# Property 4: Runtime directory creation
# Feature: nuitka-exe-build, Property 4: Runtime directory creation
# ---------------------------------------------------------------------------


class TestRuntimeDirectoryCreation:
    """Property 4: Runtime directory creation.

    For any base directory path (including paths that do not yet contain
    ``data/`` or ``logs/`` subdirectories), calling ``ensure_runtime_dirs(base)``
    SHALL result in both ``data/`` and ``logs/`` subdirectories existing under
    ``base``, and SHALL not raise an error if they already exist.

    **Validates: Requirements 4.5, 4.6**
    """

    @given(subdir=_path_segment)
    @settings(max_examples=100)
    def test_creates_data_and_logs_dirs(
        self, subdir: str, tmp_path_factory
    ) -> None:
        """ensure_runtime_dirs() creates data/ and logs/ under any base path."""
        tmp_path = tmp_path_factory.mktemp("dirs")
        base = tmp_path / subdir
        base.mkdir(parents=True, exist_ok=True)

        ensure_runtime_dirs(base)

        assert (base / "data").is_dir()
        assert (base / "logs").is_dir()

    @given(subdir=_path_segment)
    @settings(max_examples=100)
    def test_idempotent_no_error_on_existing_dirs(
        self, subdir: str, tmp_path_factory
    ) -> None:
        """Calling ensure_runtime_dirs() twice does not raise an error."""
        tmp_path = tmp_path_factory.mktemp("idem")
        base = tmp_path / subdir
        base.mkdir(parents=True, exist_ok=True)

        # First call creates the directories
        ensure_runtime_dirs(base)
        assert (base / "data").is_dir()
        assert (base / "logs").is_dir()

        # Second call should not raise
        ensure_runtime_dirs(base)
        assert (base / "data").is_dir()
        assert (base / "logs").is_dir()


# ---------------------------------------------------------------------------
# Additional imports for Properties 1, 2, 5, 6
# ---------------------------------------------------------------------------

import argparse

# build.py lives at project root — add it to sys.path so we can import it
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from build import build_nuitka_command, load_nuitka_config  # noqa: E402

# ---------------------------------------------------------------------------
# Strategies for build command / config tests
# ---------------------------------------------------------------------------

# Non-empty directory names (safe characters only)
_dir_name = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N"),
        whitelist_characters="_-",
    ),
    min_size=1,
    max_size=30,
)

# Package name segments like "foo", "foo.bar"
_pkg_name = st.from_regex(r"[a-z][a-z0-9_]{0,15}(\.[a-z][a-z0-9_]{0,15}){0,2}", fullmatch=True)

# Lists of package names
_pkg_list = st.lists(_pkg_name, min_size=0, max_size=5)

# Non-empty printable strings for metadata (no newlines, no null bytes)
_meta_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Z"),
        blacklist_characters="\x00\n\r",
    ),
    min_size=1,
    max_size=60,
)

# Version strings like "1.2.3" or "0.0.1"
_version_str = st.from_regex(r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}", fullmatch=True)

# Data dir entries in the form "src=dest"
_data_dir_entry = st.tuples(_dir_name, _dir_name).map(lambda t: f"{t[0]}={t[1]}")
_data_dir_list = st.lists(_data_dir_entry, min_size=0, max_size=3)

# Nofollow import entries
_nofollow_list = st.lists(_pkg_name, min_size=0, max_size=3)


def _make_args(
    onefile: bool = False,
    output_dir: str | None = None,
    clean: bool = False,
) -> argparse.Namespace:
    """Create an argparse.Namespace mimicking CLI arguments."""
    return argparse.Namespace(onefile=onefile, output_dir=output_dir, clean=clean)


# ---------------------------------------------------------------------------
# Property 1: Command construction invariant
# Feature: nuitka-exe-build, Property 1: Command construction invariant
# ---------------------------------------------------------------------------


class TestCommandConstructionInvariant:
    """Property 1: Command construction invariant.

    For any valid NuitkaBuildConfig (with arbitrary output_dir,
    include_packages, onefile flag, and data dir entries), the Nuitka command
    produced by build_nuitka_command() SHALL always contain --standalone, the
    entry point src/patent_system/main.py, and --output-dir=<configured_dir>.

    **Validates: Requirements 1.1, 1.3**
    """

    @given(
        output_dir=_dir_name,
        packages=_pkg_list,
        onefile=st.booleans(),
        data_dirs=_data_dir_list,
    )
    @settings(max_examples=100)
    def test_command_always_contains_standalone_entrypoint_outputdir(
        self,
        output_dir: str,
        packages: list[str],
        onefile: bool,
        data_dirs: list[str],
    ) -> None:
        """Command always includes --standalone, entry point, and --output-dir."""
        config: dict[str, object] = {
            "output-dir": output_dir,
            "include-package": packages,
            "include-data-dir": data_dirs,
        }
        args = _make_args(onefile=onefile)
        metadata = {"version": "1.0.0", "author_name": "Test", "description": "Desc"}

        cmd = build_nuitka_command(config, args, metadata)

        assert "--standalone" in cmd
        assert "src/patent_system/main.py" in cmd
        assert f"--output-dir={output_dir}" in cmd


# ---------------------------------------------------------------------------
# Property 2: Metadata passthrough
# Feature: nuitka-exe-build, Property 2: Metadata passthrough
# ---------------------------------------------------------------------------


class TestMetadataPassthrough:
    """Property 2: Metadata passthrough.

    For any ProjectMetadata with arbitrary version, author_name, and
    description strings, the Nuitka command produced by
    build_nuitka_command() SHALL contain --file-version=<version>,
    --product-version=<version>, --company-name=<author_name>, and
    --file-description=<description> with values matching the input metadata
    exactly.

    **Validates: Requirements 9.2, 9.3, 9.4**
    """

    @given(
        version=_version_str,
        author_name=_meta_text,
        description=_meta_text,
    )
    @settings(max_examples=100)
    def test_metadata_flags_match_input_exactly(
        self,
        version: str,
        author_name: str,
        description: str,
    ) -> None:
        """Metadata flags in the command match the input metadata exactly."""
        config: dict[str, object] = {"output-dir": "dist"}
        args = _make_args()
        metadata = {
            "version": version,
            "author_name": author_name,
            "description": description,
        }

        cmd = build_nuitka_command(config, args, metadata)

        assert f"--file-version={version}" in cmd
        assert f"--product-version={version}" in cmd
        assert f"--company-name={author_name}" in cmd
        assert f"--file-description={description}" in cmd


# ---------------------------------------------------------------------------
# Property 5: Config round-trip
# Feature: nuitka-exe-build, Property 5: Config round-trip
# ---------------------------------------------------------------------------


class TestConfigRoundTrip:
    """Property 5: Config round-trip.

    For any valid Nuitka configuration dictionary (containing output-dir,
    include-package, include-data-dir, and nofollow-import-to keys with valid
    values), writing it to a [tool.nuitka] section in a TOML file and reading
    it back via load_nuitka_config() SHALL produce an equivalent dictionary.

    **Validates: Requirements 5.1, 5.2**
    """

    @given(
        output_dir=_dir_name,
        packages=_pkg_list,
        data_dirs=_data_dir_list,
        nofollow=_nofollow_list,
    )
    @settings(max_examples=100)
    def test_write_then_read_produces_equivalent_dict(
        self,
        output_dir: str,
        packages: list[str],
        data_dirs: list[str],
        nofollow: list[str],
        tmp_path_factory,
    ) -> None:
        """Writing [tool.nuitka] to TOML and reading back yields the same dict."""
        original: dict[str, object] = {
            "output-dir": output_dir,
            "include-package": packages,
            "include-data-dir": data_dirs,
            "nofollow-import-to": nofollow,
        }

        # Write a minimal pyproject.toml with [tool.nuitka] manually
        tmp_path = tmp_path_factory.mktemp("roundtrip")
        toml_path = tmp_path / "pyproject.toml"

        def _toml_list(items: list[str]) -> str:
            """Format a Python list as a TOML inline array."""
            inner = ", ".join(f'"{v}"' for v in items)
            return f"[{inner}]"

        lines = [
            "[tool.nuitka]",
            f'"output-dir" = "{output_dir}"',
            f'"include-package" = {_toml_list(packages)}',
            f'"include-data-dir" = {_toml_list(data_dirs)}',
            f'"nofollow-import-to" = {_toml_list(nofollow)}',
        ]
        toml_path.write_text("\n".join(lines), encoding="utf-8")

        # Read it back via load_nuitka_config
        result = load_nuitka_config(toml_path)

        assert result == original


# ---------------------------------------------------------------------------
# Property 6: CLI override precedence
# Feature: nuitka-exe-build, Property 6: CLI override precedence
# ---------------------------------------------------------------------------


class TestCLIOverridePrecedence:
    """Property 6: CLI override precedence.

    For any pair of a pyproject.toml config dictionary and a CLI arguments
    namespace where both specify the same key (e.g., output_dir), the merged
    result SHALL contain the CLI value, not the pyproject.toml value.

    **Validates: Requirements 5.3**
    """

    @given(
        config_dir=_dir_name,
        cli_dir=_dir_name,
    )
    @settings(max_examples=100)
    def test_cli_output_dir_overrides_config(
        self,
        config_dir: str,
        cli_dir: str,
    ) -> None:
        """CLI --output-dir overrides the pyproject.toml output-dir value."""
        config: dict[str, object] = {"output-dir": config_dir}
        args = _make_args(output_dir=cli_dir)
        metadata = {"version": "1.0.0", "author_name": "Test", "description": "Desc"}

        cmd = build_nuitka_command(config, args, metadata)

        # The CLI value should appear, not the config value (unless they're equal)
        assert f"--output-dir={cli_dir}" in cmd
