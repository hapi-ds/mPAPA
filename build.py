"""Nuitka build script for mPAPA.

Reads configuration from pyproject.toml, accepts CLI overrides, validates
prerequisites, constructs the Nuitka command line, and invokes the compiler.

Usage:
    uv run python build.py [--onefile] [--output-dir DIR] [--clean]
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_nuitka_config(
    pyproject_path: Path = Path("pyproject.toml"),
) -> dict[str, Any]:
    """Read the ``[tool.nuitka]`` section from *pyproject.toml*.

    Args:
        pyproject_path: Path to the pyproject.toml file.

    Returns:
        A dictionary with the Nuitka configuration values, or an empty
        dictionary if the section is missing.
    """
    with pyproject_path.open("rb") as fh:
        data = tomllib.load(fh)

    nuitka_config: dict[str, Any] = data.get("tool", {}).get("nuitka", {})

    if not nuitka_config:
        logger.info(
            "[tool.nuitka] section not found in %s — using defaults",
            pyproject_path,
        )

    return nuitka_config


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the build script.

    Returns:
        Parsed argument namespace with ``onefile``, ``output_dir``, and
        ``clean`` attributes.
    """
    parser = argparse.ArgumentParser(
        description="Build mPAPA as a standalone executable using Nuitka.",
    )
    parser.add_argument(
        "--onefile",
        action="store_true",
        default=False,
        help="Produce a single-file executable instead of a directory distribution.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the output directory (default: read from pyproject.toml or 'dist').",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Remove previous build artifacts before building.",
    )
    return parser.parse_args()


def read_project_metadata(
    pyproject_path: Path = Path("pyproject.toml"),
) -> dict[str, str]:
    """Extract project metadata from the ``[project]`` section of *pyproject.toml*.

    Reads ``version``, ``description``, and the first author's ``name`` from
    the ``[project]`` table.

    Args:
        pyproject_path: Path to the pyproject.toml file.

    Returns:
        A dictionary with keys ``version``, ``description``, and
        ``author_name``.

    Raises:
        SystemExit: If the ``[project]`` section or any required field is
            missing.
    """
    with pyproject_path.open("rb") as fh:
        data = tomllib.load(fh)

    project = data.get("project")
    if project is None:
        logger.error("[project] section missing from %s", pyproject_path)
        raise SystemExit(1)

    version = project.get("version")
    if not version:
        logger.error("'version' field missing from [project] in %s", pyproject_path)
        raise SystemExit(1)

    description = project.get("description")
    if not description:
        logger.error(
            "'description' field missing from [project] in %s", pyproject_path
        )
        raise SystemExit(1)

    authors = project.get("authors")
    if not authors or not isinstance(authors, list) or len(authors) == 0:
        logger.error(
            "'authors' list missing or empty in [project] in %s", pyproject_path
        )
        raise SystemExit(1)

    author_name = authors[0].get("name")
    if not author_name:
        logger.error(
            "First author entry missing 'name' in [project.authors] in %s",
            pyproject_path,
        )
        raise SystemExit(1)

    return {
        "version": version,
        "description": description,
        "author_name": author_name,
    }


# ---------------------------------------------------------------------------
# Prerequisite validation (task 4.2) and command construction (task 4.3)
# ---------------------------------------------------------------------------


def validate_prerequisites() -> None:
    """Check that Nuitka is installed and required assets exist.

    Performs three validation steps in order:

    1. **Nuitka import check** — verifies the ``nuitka`` package is importable.
       Exits with code 1 if missing.
    2. **NiceGUI static assets** — locates the ``static`` directory inside the
       installed ``nicegui`` package via ``nicegui.__path__``.  Exits with
       code 1 if the directory cannot be found.
    3. **Data file warnings** — checks that DOCX templates in
       ``src/patent_system/export/templates/`` and ``.env.example`` exist.
       Logs a WARNING for each missing file but does **not** abort the build.

    Raises:
        SystemExit: If Nuitka is not installed or NiceGUI static assets
            cannot be located.
    """
    # 1. Check Nuitka is importable
    try:
        import nuitka  # noqa: F401
    except ImportError:
        logger.error(
            "Nuitka is not installed. Add it via: `uv add --group build nuitka`"
        )
        sys.exit(1)

    # 2. Locate NiceGUI static assets
    try:
        import nicegui
    except ImportError:
        logger.error(
            "NiceGUI is not installed. Cannot locate static assets."
        )
        sys.exit(1)

    nicegui_static_found = False
    for package_dir in nicegui.__path__:
        static_dir = Path(package_dir) / "static"
        if static_dir.is_dir():
            nicegui_static_found = True
            logger.info("Found NiceGUI static assets at %s", static_dir)
            break

    if not nicegui_static_found:
        expected = Path(nicegui.__path__[0]) / "static"
        logger.error(
            "NiceGUI static assets not found. Expected at: %s", expected
        )
        sys.exit(1)

    # 3. Warn on missing data files (do NOT exit)
    templates_dir = Path("src/patent_system/export/templates")
    if templates_dir.is_dir():
        docx_files = list(templates_dir.glob("*.docx"))
        if not docx_files:
            logger.warning(
                "No DOCX templates found in %s", templates_dir
            )
    else:
        logger.warning("Templates directory not found: %s", templates_dir)

    env_example = Path(".env.example")
    if not env_example.is_file():
        logger.warning("Missing data file: %s", env_example)


def build_nuitka_command(
    config: dict[str, Any],
    args: argparse.Namespace,
    metadata: dict[str, str],
) -> list[str]:
    """Construct the full Nuitka command line as a list of strings.

    Merges configuration from ``pyproject.toml``, CLI argument overrides,
    and project metadata into a complete Nuitka invocation command.

    The command always includes ``--standalone`` mode and the application
    entry point ``src/patent_system/main.py``.

    Args:
        config: Nuitka configuration dictionary from ``[tool.nuitka]``.
        args: Parsed CLI arguments (``onefile``, ``output_dir``).
        metadata: Project metadata with ``version``, ``description``,
            and ``author_name`` keys.

    Returns:
        A list of strings suitable for passing to ``subprocess.run()``.
    """
    cmd: list[str] = [sys.executable, "-m", "nuitka", "--standalone"]

    # Output directory: CLI override takes precedence over config
    output_dir = args.output_dir if args.output_dir else config.get("output-dir", "dist")
    cmd.append(f"--output-dir={output_dir}")

    # Onefile mode (opt-in)
    if args.onefile:
        cmd.append("--onefile")

    # Executable metadata
    cmd.append("--product-name=mPAPA - my Personal Artificial Patent Agent")
    cmd.append(f"--file-version={metadata['version']}")
    cmd.append(f"--product-version={metadata['version']}")
    cmd.append(f"--company-name={metadata['author_name']}")
    cmd.append(f"--file-description={metadata['description']}")

    # Include packages (hidden imports)
    for pkg in config.get("include-package", []):
        cmd.append(f"--include-package={pkg}")

    # Include package data
    for pkg in config.get("include-package-data", []):
        cmd.append(f"--include-package-data={pkg}")

    # Include data directories
    for entry in config.get("include-data-dir", []):
        cmd.append(f"--include-data-dir={entry}")

    # Include data files
    for entry in config.get("include-data-files", []):
        cmd.append(f"--include-data-files={entry}")

    # Nofollow imports
    for entry in config.get("nofollow-import-to", []):
        cmd.append(f"--nofollow-import-to={entry}")

    # Entry point (always last)
    cmd.append("src/patent_system/main.py")

    return cmd


def main() -> None:
    """Entry point: validate prerequisites, configure, and build.

    Orchestrates the full build pipeline:

    1. Validate that Nuitka and required assets are available.
    2. Parse CLI arguments and load configuration.
    3. Read project metadata from ``pyproject.toml``.
    4. Optionally clean previous build artifacts.
    5. Construct and execute the Nuitka compilation command.
    6. Exit with Nuitka's return code on failure.
    """
    validate_prerequisites()

    args = parse_args()
    config = load_nuitka_config()
    metadata = read_project_metadata()

    # Determine output directory for clean step
    output_dir = args.output_dir if args.output_dir else config.get("output-dir", "dist")

    if args.clean:
        output_path = Path(output_dir)
        if output_path.exists():
            logger.info("Cleaning previous build artifacts in %s", output_path)
            shutil.rmtree(output_path)

    cmd = build_nuitka_command(config, args, metadata)
    logger.info("Executing: %s", " ".join(cmd))

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        logger.error("Nuitka build failed with exit code %d", result.returncode)
        sys.exit(result.returncode)

    logger.info("Build completed successfully. Artifacts in %s", output_dir)


if __name__ == "__main__":
    main()
