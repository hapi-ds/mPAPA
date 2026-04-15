"""Smoke test for the Nuitka-compiled executable.

Launches the build artifact as a subprocess, verifies the NiceGUI HTTP
server responds with HTTP 200, and tears down cleanly.  The test is
automatically skipped when no build artifact exists in ``dist/``.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DIST_DIR = Path("dist")
_STANDALONE_EXE = _DIST_DIR / "main.dist" / "main.exe"
_ONEFILE_EXE = _DIST_DIR / "main.exe"
_POLL_URL = "http://localhost:8080/"
_POLL_INTERVAL_S = 1
_POLL_TIMEOUT_S = 30
_TERMINATE_TIMEOUT_S = 5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def exe_path() -> Path:
    """Locate the built executable in ``dist/``.

    Checks for the standalone-mode artifact first
    (``dist/main.dist/main.exe``), then the onefile-mode artifact
    (``dist/main.exe``).  Skips the test when neither is found.

    Returns:
        Path: Resolved path to the executable.
    """
    if _STANDALONE_EXE.exists():
        return _STANDALONE_EXE.resolve()
    if _ONEFILE_EXE.exists():
        return _ONEFILE_EXE.resolve()
    pytest.skip("No build artifact found in dist/ — skipping smoke test")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_exe_starts_and_responds(exe_path: Path) -> None:
    """Launch the compiled exe and verify the HTTP server responds.

    The test starts the executable as a subprocess, polls
    ``http://localhost:8080/`` every second for up to 30 seconds, and
    asserts that an HTTP 200 response is received.  On timeout the test
    fails with captured stdout/stderr for diagnostics.

    The subprocess is always terminated in a ``finally`` block using
    ``process.terminate()`` with a fallback to ``process.kill()`` if the
    process does not exit within 5 seconds.

    Args:
        exe_path: Path to the built executable (provided by fixture).
    """
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            [str(exe_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        deadline = time.monotonic() + _POLL_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                response = urlopen(_POLL_URL, timeout=_POLL_INTERVAL_S)  # noqa: S310
                if response.status == 200:
                    return  # Success — server is up and responding
            except (URLError, OSError, ConnectionResetError):
                pass

            # Check if the process crashed before the next poll
            if process.poll() is not None:
                stdout = process.stdout.read().decode(errors="replace") if process.stdout else ""
                stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
                pytest.fail(
                    f"Executable exited prematurely with code {process.returncode}.\n"
                    f"--- stdout ---\n{stdout}\n"
                    f"--- stderr ---\n{stderr}"
                )

            time.sleep(_POLL_INTERVAL_S)

        # Timeout — collect diagnostics
        stdout = ""
        stderr = ""
        if process.stdout:
            stdout = process.stdout.read().decode(errors="replace")
        if process.stderr:
            stderr = process.stderr.read().decode(errors="replace")
        pytest.fail(
            f"Server did not respond with HTTP 200 within {_POLL_TIMEOUT_S}s.\n"
            f"--- stdout ---\n{stdout}\n"
            f"--- stderr ---\n{stderr}"
        )
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=_TERMINATE_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
