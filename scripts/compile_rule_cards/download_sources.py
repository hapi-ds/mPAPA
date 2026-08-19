"""Automated patent guideline downloader.

Reads the compact sources.toml (jurisdictions on/off, chapters on/off),
looks up actual URLs from the internal source_registry.py, downloads new/updated files.

Usage:
    uv run python scripts/compile_rule_cards/download_sources.py          # download all enabled
    uv run python scripts/compile_rule_cards/download_sources.py --dry-run  # check only
    uv run python scripts/compile_rule_cards/download_sources.py --list     # show what would be fetched
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
    import tomli as tomllib  # type: ignore[no-redef]

from .source_registry import SourceEntry, get_sources_for_config

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
SOURCES_TOML = SCRIPT_DIR / "sources.toml"
SOURCES_DIR = SCRIPT_DIR / "sources"
MANIFEST_FILE = SOURCES_DIR / ".download_manifest.json"

TIMEOUT = 60
USER_AGENT = "mPAPA-GuidelineDownloader/1.0"
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5


def load_config() -> tuple[list[str], list[str]]:
    """Load sources.toml → (enabled_jurisdictions, enabled_chapters)."""
    if not SOURCES_TOML.exists():
        logger.error("Config not found: %s", SOURCES_TOML)
        sys.exit(1)
    config = tomllib.loads(SOURCES_TOML.read_text(encoding="utf-8"))
    jurisdictions = [k for k, v in config.get("jurisdictions", {}).items() if v]
    chapters = [k for k, v in config.get("chapters", {}).items() if v]
    return jurisdictions, chapters


def load_manifest() -> dict[str, Any]:
    """Load download manifest (tracks what was downloaded)."""
    if MANIFEST_FILE.exists():
        return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    return {"version": 1, "downloads": {}}


def save_manifest(manifest: dict[str, Any]) -> None:
    """Save download manifest."""
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    manifest["last_updated"] = datetime.now(timezone.utc).isoformat()
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def check_update(url: str, manifest_entry: dict | None, client: httpx.Client) -> bool:
    """HEAD request to check if remote file changed. Returns True if update needed."""
    try:
        r = client.head(url, follow_redirects=True)
        r.raise_for_status()
    except (httpx.HTTPError, httpx.TimeoutException):
        return True  # Can't check → download anyway

    if not manifest_entry:
        return True

    remote_etag = r.headers.get("etag", "").strip('"')
    remote_modified = r.headers.get("last-modified", "")
    remote_length = r.headers.get("content-length", "")

    if remote_etag and manifest_entry.get("etag") == remote_etag:
        return False
    if remote_modified and manifest_entry.get("last_modified") == remote_modified:
        return False
    if remote_length and str(manifest_entry.get("content_length", "")) == remote_length:
        return False

    return True


def download_file(url: str, dest: Path, client: httpx.Client) -> dict[str, Any] | None:
    """Download with retries. Returns metadata or None on failure."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(RETRY_ATTEMPTS):
        try:
            r = client.get(url, follow_redirects=True)
            r.raise_for_status()
            dest.write_bytes(r.content)
            return {
                "size_bytes": len(r.content),
                "sha256": hashlib.sha256(r.content).hexdigest(),
                "etag": r.headers.get("etag", "").strip('"'),
                "last_modified": r.headers.get("last-modified", ""),
                "content_length": len(r.content),
            }
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
            else:
                logger.error("  ❌ Failed after %d attempts: %s", RETRY_ATTEMPTS, e)
                return None
    return None


def run(dry_run: bool = False, list_only: bool = False) -> None:
    """Main download routine."""
    jurisdictions, chapters = load_config()
    sources = get_sources_for_config(jurisdictions, chapters)

    if not sources:
        print("No sources matched. Check sources.toml (jurisdictions × chapters).")
        return

    if list_only:
        print(f"\n📋 {len(sources)} source(s) to download:\n")
        for s in sources:
            lang = f" [{s.language}→en]" if s.language != "en" else ""
            print(f"  [{s.jurisdiction}] {s.label}{lang}")
            print(f"         → {s.url[:80]}")
        print()
        return

    manifest = load_manifest()
    client = httpx.Client(timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})
    updated, skipped, failed = 0, 0, 0

    print(f"\n📥 Checking {len(sources)} source(s)...\n")

    try:
        for entry in sources:
            dest = SOURCES_DIR / entry.jurisdiction / f"{entry.card_id}.pdf"
            lang = f" [{entry.language}→en]" if entry.language != "en" else ""
            print(f"  [{entry.jurisdiction}] {entry.label}{lang}")

            # Check if update needed
            mf_entry = manifest["downloads"].get(entry.card_id)
            if dest.exists() and mf_entry and not check_update(entry.url, mf_entry, client):
                print("         ⏭️  Up to date")
                skipped += 1
                continue

            if dry_run:
                print("         📦 Would download" if not dest.exists() else "         🔄 Would update")
                updated += 1
                continue

            # Download
            print(f"         ⬇️  Downloading...")
            result = download_file(entry.url, dest, client)
            if result:
                mb = result["size_bytes"] / (1024 * 1024)
                print(f"         ✅ {mb:.1f} MB")
                manifest["downloads"][entry.card_id] = {
                    "url": entry.url,
                    "jurisdiction": entry.jurisdiction,
                    "language": entry.language,
                    "card_id": entry.card_id,
                    "file": str(dest.relative_to(SCRIPT_DIR)),
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                    **result,
                }
                updated += 1
            else:
                failed += 1
    finally:
        client.close()

    if not dry_run:
        save_manifest(manifest)

    print(f"\n{'─' * 50}")
    verb = "Would download" if dry_run else "Downloaded"
    print(f"  {verb}: {updated}  |  Up to date: {skipped}  |  Failed: {failed}")
    print(f"  Sources: {SOURCES_DIR}/\n")


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser(description="Download patent examination guidelines")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Check without downloading")
    parser.add_argument("--list", "-l", action="store_true", help="List sources and exit")
    parser.add_argument("--force", "-f", action="store_true", help="Force re-download all")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.force:
        # Clear manifest to force re-download
        if MANIFEST_FILE.exists():
            MANIFEST_FILE.unlink()

    run(dry_run=args.dry_run, list_only=args.list)


if __name__ == "__main__":
    main()
