"""Automated patent guideline downloader with update checking.

Downloads patent examination guidelines from configured sources (EPO, USPTO, WIPO, JPO, etc.)
in their ORIGINAL language. Translation to English is handled by the rule card compiler, not here.

Features:
- Reads sources from sources.toml config (comment out to disable)
- Checks for updates via HTTP headers (Last-Modified, ETag, Content-Length)
- Only downloads if new/changed (uses local manifest for tracking)
- Supports PDF and HTML sources
- Stores downloads in scripts/compile_rule_cards/sources/<jurisdiction>/

Usage:
    # Download all enabled sources
    uv run python scripts/compile_rule_cards/download_sources.py

    # Download specific jurisdiction only
    uv run python scripts/compile_rule_cards/download_sources.py --jurisdiction EP

    # Force re-download (ignore cache)
    uv run python scripts/compile_rule_cards/download_sources.py --force

    # Dry run (check for updates without downloading)
    uv run python scripts/compile_rule_cards/download_sources.py --dry-run

    # List all configured sources
    uv run python scripts/compile_rule_cards/download_sources.py --list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # Python < 3.11 fallback

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
SOURCES_TOML = SCRIPT_DIR / "sources.toml"
SOURCES_DIR = SCRIPT_DIR / "sources"
MANIFEST_FILE = SOURCES_DIR / ".download_manifest.json"

# HTTP settings
TIMEOUT = 60  # seconds
USER_AGENT = "mPAPA-GuidelineDownloader/1.0 (patent examination guideline archiver)"
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5  # seconds between retries


# ---------------------------------------------------------------------------
# Manifest (tracks what we've downloaded and when)
# ---------------------------------------------------------------------------


def load_manifest() -> dict[str, Any]:
    """Load the download manifest (tracks versions, ETags, timestamps)."""
    if MANIFEST_FILE.exists():
        return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    return {"version": 1, "downloads": {}}


def save_manifest(manifest: dict[str, Any]) -> None:
    """Save the download manifest."""
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    manifest["last_updated"] = datetime.now(timezone.utc).isoformat()
    MANIFEST_FILE.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Source config parsing
# ---------------------------------------------------------------------------


def load_sources_config() -> dict[str, Any]:
    """Load and parse the sources.toml configuration."""
    if not SOURCES_TOML.exists():
        logger.error(f"Config file not found: {SOURCES_TOML}")
        sys.exit(1)
    return tomllib.loads(SOURCES_TOML.read_text(encoding="utf-8"))


def get_enabled_sources(config: dict, jurisdiction_filter: str | None = None) -> list[dict]:
    """Get all enabled sources, optionally filtered by jurisdiction."""
    sources = []
    skip_keys = {"metadata"}

    for key, section in config.items():
        if key in skip_keys:
            continue
        if not isinstance(section, dict):
            continue
        if not section.get("enabled", False):
            continue
        if jurisdiction_filter and key != jurisdiction_filter:
            continue

        jurisdiction = key
        office = section.get("office", jurisdiction)
        language = section.get("language", "en")
        translate_to = section.get("translate_to")

        for source in section.get("sources", []):
            if not source.get("enabled", True):
                continue

            # Determine URL (supports url, url_pdf, url_html)
            url = source.get("url") or source.get("url_pdf") or source.get("url_html", "")
            if not url:
                logger.warning(f"  Skipping {source['id']}: no URL configured (place file manually)")
                continue

            sources.append({
                "id": source["id"],
                "label": source.get("label", source["id"]),
                "url": url,
                "jurisdiction": jurisdiction,
                "office": office,
                "language": language,
                "translate_to": translate_to,
                "task": source.get("task", ""),
                "part": source.get("part", ""),
                "chapter": source.get("chapter", ""),
            })

    return sources


# ---------------------------------------------------------------------------
# Download logic
# ---------------------------------------------------------------------------


def check_for_update(url: str, manifest_entry: dict | None, client: httpx.Client) -> dict | None:
    """Check if a source has been updated (HEAD request).

    Returns metadata dict if update available, None if unchanged.
    """
    try:
        response = client.head(url, follow_redirects=True)
        response.raise_for_status()
    except (httpx.HTTPError, httpx.TimeoutException) as e:
        logger.warning(f"  HEAD request failed: {e}")
        return {"status": "check_failed", "error": str(e)}

    headers = response.headers
    remote_etag = headers.get("etag", "").strip('"')
    remote_last_modified = headers.get("last-modified", "")
    remote_content_length = headers.get("content-length", "")

    if manifest_entry:
        local_etag = manifest_entry.get("etag", "")
        local_last_modified = manifest_entry.get("last_modified", "")
        local_content_length = str(manifest_entry.get("content_length", ""))

        # Check ETag first (most reliable)
        if remote_etag and local_etag and remote_etag == local_etag:
            return None  # Unchanged

        # Check Last-Modified
        if remote_last_modified and local_last_modified and remote_last_modified == local_last_modified:
            return None  # Unchanged

        # Check Content-Length as fallback
        if remote_content_length and local_content_length and remote_content_length == local_content_length:
            if not remote_etag and not remote_last_modified:
                return None  # Probably unchanged (weak check)

    return {
        "status": "update_available",
        "etag": remote_etag,
        "last_modified": remote_last_modified,
        "content_length": int(remote_content_length) if remote_content_length else None,
    }


def download_file(url: str, dest: Path, client: httpx.Client) -> dict:
    """Download a file with retry logic. Returns metadata."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = client.get(url, follow_redirects=True)
            response.raise_for_status()
            content = response.content

            dest.write_bytes(content)

            # Compute hash for integrity tracking
            sha256 = hashlib.sha256(content).hexdigest()

            return {
                "success": True,
                "size_bytes": len(content),
                "sha256": sha256,
                "etag": response.headers.get("etag", "").strip('"'),
                "last_modified": response.headers.get("last-modified", ""),
                "content_length": len(content),
                "content_type": response.headers.get("content-type", ""),
            }

        except (httpx.HTTPError, httpx.TimeoutException) as e:
            if attempt < RETRY_ATTEMPTS - 1:
                logger.warning(f"  Attempt {attempt + 1} failed: {e}. Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                return {"success": False, "error": str(e)}

    return {"success": False, "error": "Max retries exceeded"}


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def list_sources(config: dict) -> None:
    """Print all configured sources."""
    print("\n📋 Configured Guideline Sources\n")
    for key, section in config.items():
        if key == "metadata" or not isinstance(section, dict):
            continue
        enabled = "✅" if section.get("enabled") else "⬜"
        lang = section.get("language", "?")
        translate = f" → {section['translate_to']}" if section.get("translate_to") else ""
        print(f"{enabled} [{key}] {section.get('full_name', key)} ({lang}{translate})")
        for source in section.get("sources", []):
            src_enabled = "  ✅" if source.get("enabled", True) else "  ⬜"
            url_status = "🔗" if (source.get("url") or source.get("url_pdf") or source.get("url_html")) else "❌ no URL"
            print(f"    {src_enabled} {source['id']}: {source.get('label', '')} [{url_status}]")
    print()


def run_download(
    jurisdiction_filter: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> None:
    """Main download routine."""
    config = load_sources_config()
    sources = get_enabled_sources(config, jurisdiction_filter)
    manifest = load_manifest()

    if not sources:
        print("No enabled sources found. Check sources.toml configuration.")
        return

    print(f"\n📥 Checking {len(sources)} source(s) for updates...\n")

    client = httpx.Client(
        timeout=TIMEOUT,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
    )

    updated = 0
    skipped = 0
    failed = 0

    try:
        for source in sources:
            source_id = source["id"]
            url = source["url"]
            jurisdiction = source["jurisdiction"]

            # Determine output path
            ext = ".pdf" if "pdf" in url.lower() else ".html"
            dest = SOURCES_DIR / jurisdiction / f"{source_id}{ext}"

            manifest_entry = manifest["downloads"].get(source_id)
            print(f"  [{jurisdiction}] {source['label']}")

            # Check for update (unless --force)
            if not force and manifest_entry and dest.exists():
                update_info = check_for_update(url, manifest_entry, client)
                if update_info is None:
                    print(f"         ⏭️  Up to date (skipped)")
                    skipped += 1
                    continue
                elif update_info.get("status") == "check_failed":
                    print(f"         ⚠️  Could not check: {update_info.get('error', 'unknown')}")
                    # Download anyway if we can't verify

            if dry_run:
                if not dest.exists():
                    print(f"         📦 Would download (new)")
                else:
                    print(f"         🔄 Would update")
                updated += 1
                continue

            # Download
            print(f"         ⬇️  Downloading from {url[:80]}...")
            result = download_file(url, dest, client)

            if result["success"]:
                size_mb = result["size_bytes"] / (1024 * 1024)
                print(f"         ✅ Downloaded ({size_mb:.1f} MB)")

                # Update manifest
                manifest["downloads"][source_id] = {
                    "url": url,
                    "jurisdiction": jurisdiction,
                    "language": source["language"],
                    "translate_to": source.get("translate_to"),
                    "file": str(dest.relative_to(SCRIPT_DIR)),
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                    "etag": result.get("etag", ""),
                    "last_modified": result.get("last_modified", ""),
                    "content_length": result.get("content_length", 0),
                    "sha256": result.get("sha256", ""),
                    "size_bytes": result["size_bytes"],
                }
                updated += 1
            else:
                print(f"         ❌ Failed: {result.get('error', 'unknown')}")
                failed += 1

    finally:
        client.close()

    # Save manifest
    if not dry_run:
        save_manifest(manifest)

    # Summary
    print(f"\n{'─' * 50}")
    action = "Would download" if dry_run else "Downloaded"
    print(f"  {action}: {updated}  |  Skipped (up to date): {skipped}  |  Failed: {failed}")
    print(f"  Sources stored in: {SOURCES_DIR}/")
    if not dry_run and updated > 0:
        print(f"  Manifest updated: {MANIFEST_FILE}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download patent examination guidelines for rule card compilation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s                          # Download all enabled sources
  %(prog)s --jurisdiction EP        # Download EPO guidelines only
  %(prog)s --dry-run                # Check for updates without downloading
  %(prog)s --force                  # Force re-download all
  %(prog)s --list                   # Show all configured sources
""",
    )
    parser.add_argument(
        "--jurisdiction", "-j",
        help="Download only this jurisdiction (e.g. EP, US, JP, CN, KR, PCT)",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download (ignore cache/manifest)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Check for updates without actually downloading",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all configured sources and exit",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.list:
        config = load_sources_config()
        list_sources(config)
        return

    run_download(
        jurisdiction_filter=args.jurisdiction,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
