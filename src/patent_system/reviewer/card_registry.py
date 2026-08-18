"""Card registry: discovers, indexes, and downloads Rule Cards.

Manages the local card cache and fetches cards on demand from a
configurable GitHub raw-content URL. Works fully offline once cards
are downloaded.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx

from .rule_card import RuleCard

logger = logging.getLogger(__name__)


class CardRegistryError(Exception):
    """Base exception for card registry operations."""


class CardDownloadError(CardRegistryError):
    """Raised when a card download fails."""


class CardNotFoundError(CardRegistryError):
    """Raised when a requested card is not found locally or remotely."""


class CardRegistry:
    """Manages local card cache and GitHub downloads.

    Discovers cards from a local cache directory and fetches individual
    cards on demand from a remote repository. Validates all downloaded
    cards via RuleCard.from_json_file().

    Args:
        cache_dir: Local directory for cached card JSON files.
        repo_url: Base URL for raw GitHub content (e.g.
            "https://raw.githubusercontent.com/hapi-ds/mPAPA/main/rule-cards").
    """

    _INDEX_FILENAME = "index.json"
    _REQUEST_TIMEOUT = 30.0

    def __init__(self, cache_dir: Path, repo_url: str) -> None:
        self._cache_dir = cache_dir
        self._repo_url = repo_url.rstrip("/")
        self._index: list[dict] | None = None

        # Ensure cache directory exists
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cache_dir(self) -> Path:
        """Return the local cache directory path."""
        return self._cache_dir

    @property
    def repo_url(self) -> str:
        """Return the configured remote repository URL."""
        return self._repo_url

    # ------------------------------------------------------------------
    # Local operations (fully offline)
    # ------------------------------------------------------------------

    def list_local(self) -> list[RuleCard]:
        """Discover and load all valid cards from the local cache.

        Scans the cache directory recursively for .json files (excluding
        index.json), validates each via RuleCard.from_json_file(), and
        returns successfully loaded cards. Invalid files are logged and
        skipped.

        Returns:
            List of validated RuleCard instances from the local cache.
        """
        cards: list[RuleCard] = []
        for json_file in sorted(self._cache_dir.rglob("*.json")):
            if json_file.name == self._INDEX_FILENAME:
                continue
            try:
                card = RuleCard.from_json_file(json_file)
                cards.append(card)
            except Exception as exc:
                logger.warning(
                    "Skipping invalid card file %s: %s", json_file, exc
                )
        return cards

    def get_card(self, card_id: str) -> RuleCard | None:
        """Load a card from the local cache by its card_id.

        Searches cached files for one whose card_id matches. Does NOT
        attempt a remote download — use download_card() for that.

        Args:
            card_id: The unique identifier of the card.

        Returns:
            The RuleCard if found locally, otherwise None.
        """
        for json_file in self._cache_dir.rglob("*.json"):
            if json_file.name == self._INDEX_FILENAME:
                continue
            try:
                card = RuleCard.from_json_file(json_file)
                if card.card_id == card_id:
                    return card
            except Exception:
                continue
        return None

    def is_cached(self, card_id: str) -> bool:
        """Check whether a card is available in the local cache.

        Args:
            card_id: The unique identifier of the card.

        Returns:
            True if the card exists locally and is valid.
        """
        return self.get_card(card_id) is not None

    # ------------------------------------------------------------------
    # Remote operations (require network)
    # ------------------------------------------------------------------

    def refresh_index(self) -> None:
        """Fetch or refresh the remote index.json manifest.

        Downloads index.json from the configured repo_url and caches it
        locally. The index lists all available cards with metadata.

        Raises:
            CardDownloadError: If the index cannot be fetched.
        """
        url = f"{self._repo_url}/{self._INDEX_FILENAME}"
        logger.info("Fetching card index from %s", url)

        try:
            with httpx.Client(timeout=self._REQUEST_TIMEOUT) as client:
                response = client.get(url)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise CardDownloadError(
                f"Failed to fetch card index from {url}: {exc}"
            ) from exc

        try:
            index_data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise CardDownloadError(
                f"Invalid JSON in index from {url}: {exc}"
            ) from exc

        # Persist index locally for offline use
        index_path = self._cache_dir / self._INDEX_FILENAME
        index_path.write_text(
            json.dumps(index_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._index = index_data if isinstance(index_data, list) else index_data.get("cards", [])
        logger.info("Card index refreshed: %d cards available", len(self._index))

    def list_available(self) -> list[dict]:
        """Return the list of remotely available cards from the index.

        Loads from the in-memory cache if refresh_index() was already
        called this session, otherwise loads from the locally cached
        index.json file. Call refresh_index() first to get the latest
        remote state.

        Returns:
            List of card metadata dicts from index.json. Empty list if
            no index is available (offline and never fetched).
        """
        if self._index is not None:
            return self._index

        # Try loading from local cache
        index_path = self._cache_dir / self._INDEX_FILENAME
        if index_path.exists():
            try:
                data = json.loads(index_path.read_text(encoding="utf-8"))
                self._index = data if isinstance(data, list) else data.get("cards", [])
                return self._index
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning("Corrupted local index.json: %s", exc)

        return []

    def download_card(self, card_id: str) -> RuleCard:
        """Download a single card from the remote repository.

        Looks up the card's relative path in the index, downloads it,
        validates it via RuleCard.from_json_file(), and stores it in the
        local cache.

        Args:
            card_id: The unique identifier of the card to download.

        Returns:
            The validated RuleCard instance.

        Raises:
            CardNotFoundError: If the card_id is not in the index.
            CardDownloadError: If the download or validation fails.
        """
        # Ensure we have an index
        available = self.list_available()
        if not available:
            self.refresh_index()
            available = self.list_available()

        # Find the card entry in the index
        card_entry = self._find_in_index(card_id, available)
        if card_entry is None:
            raise CardNotFoundError(
                f"Card '{card_id}' not found in remote index. "
                f"Run refresh_index() to update."
            )

        # Determine the remote path from the index entry
        relative_path = card_entry.get("path") or card_entry.get("file")
        if not relative_path:
            # Fall back to convention: {jurisdiction}/{card_id}.json
            jurisdiction = card_entry.get("jurisdiction", "")
            relative_path = f"{jurisdiction}/{card_id}.json"

        url = f"{self._repo_url}/{relative_path}"
        logger.info("Downloading card '%s' from %s", card_id, url)

        try:
            with httpx.Client(timeout=self._REQUEST_TIMEOUT) as client:
                response = client.get(url)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise CardDownloadError(
                f"Failed to download card '{card_id}' from {url}: {exc}"
            ) from exc

        # Write to local cache preserving directory structure
        local_path = self._cache_dir / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(response.content)
        logger.info("Card '%s' saved to %s", card_id, local_path)

        # Validate by loading through RuleCard
        try:
            card = RuleCard.from_json_file(local_path)
        except Exception as exc:
            # Remove invalid file
            local_path.unlink(missing_ok=True)
            raise CardDownloadError(
                f"Downloaded card '{card_id}' failed validation: {exc}"
            ) from exc

        return card

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_in_index(
        self, card_id: str, index: list[dict]
    ) -> dict | None:
        """Find a card entry in the index by card_id.

        Args:
            card_id: The card identifier to search for.
            index: The list of card metadata dicts.

        Returns:
            The matching dict, or None if not found.
        """
        for entry in index:
            if entry.get("card_id") == card_id:
                return entry
        return None
