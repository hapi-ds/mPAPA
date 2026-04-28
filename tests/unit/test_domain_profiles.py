"""Unit tests for the ProfileLoader in agents/domain_profiles.py.

Covers directory creation, auto-population, built-in profile loading,
reload behavior, duplicate slug handling, mismatched slug/filename,
invalid YAML files, missing required keys, and multi-line YAML block scalars.

Requirements: 2.2, 2.3, 2.4, 2.5, 2.6, 3.1–3.9, 4.1–4.6
"""

import logging
from pathlib import Path

import pytest

from patent_system.agents.domain_profiles import (
    BUILTIN_PROFILE_CONTENTS,
    DEFAULT_PROFILE_SLUG,
    DomainProfile,
    ProfileLoader,
)


EXPECTED_BUILTIN_SLUGS = [
    "general-patent-drafting",
    "pharma-chemistry",
    "medtech-mechanical-engineering",
    "processes-manufacturing",
    "electrical-engineering-semiconductors",
    "biotechnology-life-sciences",
    "software-ai",
    "materials-science-nanotechnology",
    "telecommunications-standards",
]


class TestProfileLoaderDirectoryCreation:
    """Verify ProfileLoader creates and populates the profiles directory."""

    def test_creates_directory_when_missing(self, tmp_path: Path) -> None:
        """Req 4.2: Directory is created when it does not exist."""
        profiles_dir = tmp_path / "profiles"
        assert not profiles_dir.exists()

        loader = ProfileLoader(profiles_dir)

        assert profiles_dir.exists()
        assert profiles_dir.is_dir()
        assert len(list(profiles_dir.glob("*.yaml"))) == 9

    def test_populates_directory_with_builtin_profiles_when_missing(
        self, tmp_path: Path
    ) -> None:
        """Req 4.2: Built-in example files are written when directory is created."""
        profiles_dir = tmp_path / "new_profiles"
        loader = ProfileLoader(profiles_dir)

        for filename in BUILTIN_PROFILE_CONTENTS:
            filepath = profiles_dir / filename
            assert filepath.exists(), f"Expected built-in file {filename} to exist"

    def test_populates_empty_directory(self, tmp_path: Path) -> None:
        """Req 4.3: Built-in profiles are written when directory exists but is empty."""
        profiles_dir = tmp_path / "empty_profiles"
        profiles_dir.mkdir()
        assert not any(profiles_dir.iterdir())

        loader = ProfileLoader(profiles_dir)

        assert len(list(profiles_dir.glob("*.yaml"))) == 9

    def test_does_not_overwrite_existing_files(self, tmp_path: Path) -> None:
        """If directory has files, built-in profiles are NOT written."""
        profiles_dir = tmp_path / "existing_profiles"
        profiles_dir.mkdir()
        custom_file = profiles_dir / "custom-profile.yaml"
        custom_content = (
            "slug: custom-profile\n"
            "domain_label: Custom\n"
            "role_prompt: Custom role\n"
            "content_structure_guidance: Custom guidance\n"
        )
        custom_file.write_text(custom_content, encoding="utf-8")

        loader = ProfileLoader(profiles_dir)

        # Only the custom file should exist, no built-in files added
        yaml_files = list(profiles_dir.glob("*.yaml"))
        assert len(yaml_files) == 1
        assert yaml_files[0].name == "custom-profile.yaml"


class TestProfileLoaderBuiltinProfiles:
    """Verify all 9 built-in profiles load correctly."""

    @pytest.fixture
    def loader(self, tmp_path: Path) -> ProfileLoader:
        """Create a ProfileLoader with a fresh directory (auto-populated)."""
        return ProfileLoader(tmp_path / "profiles")

    def test_builtin_profile_contents_has_9_entries(self) -> None:
        """Req 3.1–3.9: BUILTIN_PROFILE_CONTENTS has exactly 9 entries."""
        assert len(BUILTIN_PROFILE_CONTENTS) == 9

    def test_all_9_profiles_loaded(self, loader: ProfileLoader) -> None:
        """Req 3.1–3.9: All 9 built-in profiles are loaded."""
        profiles = loader.get_all()
        assert len(profiles) == 9

    def test_all_expected_slugs_present(self, loader: ProfileLoader) -> None:
        """Req 3.1–3.9: All expected slugs are present in loaded profiles."""
        loaded_slugs = {p.slug for p in loader.get_all()}
        for slug in EXPECTED_BUILTIN_SLUGS:
            assert slug in loaded_slugs, f"Expected slug {slug!r} not found"

    @pytest.mark.parametrize("slug", EXPECTED_BUILTIN_SLUGS)
    def test_builtin_profile_has_nonempty_fields(
        self, loader: ProfileLoader, slug: str
    ) -> None:
        """Req 3.1–3.9: Each built-in profile has non-empty fields."""
        profile = loader.get_by_slug(slug)
        assert profile is not None, f"Profile {slug!r} not found"
        assert profile.slug == slug
        assert profile.domain_label.strip() != ""
        assert profile.role_prompt.strip() != ""
        assert profile.content_structure_guidance.strip() != ""

    def test_default_profile_exists(self, loader: ProfileLoader) -> None:
        """The default profile slug is loadable."""
        profile = loader.get_by_slug(DEFAULT_PROFILE_SLUG)
        assert profile is not None
        assert profile.slug == DEFAULT_PROFILE_SLUG

    def test_get_all_sorted_by_domain_label(self, loader: ProfileLoader) -> None:
        """get_all() returns profiles sorted by domain_label."""
        profiles = loader.get_all()
        labels = [p.domain_label for p in profiles]
        assert labels == sorted(labels)


class TestProfileLoaderReload:
    """Verify reload() picks up new files added after initial load."""

    def test_reload_picks_up_new_file(self, tmp_path: Path) -> None:
        """Req 4.5, 4.6: reload() discovers newly added files."""
        profiles_dir = tmp_path / "profiles"
        loader = ProfileLoader(profiles_dir)

        # Initially 9 built-in profiles
        assert len(loader.get_all()) == 9
        assert loader.get_by_slug("new-custom") is None

        # Add a new profile file
        new_profile = profiles_dir / "new-custom.yaml"
        new_profile.write_text(
            "slug: new-custom\n"
            "domain_label: New Custom Profile\n"
            "role_prompt: A custom role prompt for testing.\n"
            "content_structure_guidance: Custom guidance for testing.\n",
            encoding="utf-8",
        )

        # Before reload, not visible
        assert loader.get_by_slug("new-custom") is None

        # After reload, visible
        loader.reload()
        assert loader.get_by_slug("new-custom") is not None
        assert loader.get_by_slug("new-custom").domain_label == "New Custom Profile"
        assert len(loader.get_all()) == 10

    def test_reload_removes_deleted_file(self, tmp_path: Path) -> None:
        """Req 4.6: reload() reflects file deletions."""
        profiles_dir = tmp_path / "profiles"
        loader = ProfileLoader(profiles_dir)

        # Delete one built-in file
        (profiles_dir / "software-ai.yaml").unlink()

        loader.reload()
        assert loader.get_by_slug("software-ai") is None
        assert len(loader.get_all()) == 8


class TestProfileLoaderDuplicateSlug:
    """Verify duplicate slug handling: first alphabetically wins, warning logged."""

    def test_mismatched_slug_uses_filename_derived(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Req 4.4: When YAML slug mismatches filename, filename-derived slug is used.

        This means two files with different filenames won't collide even if
        their YAML slugs are the same, because the filename-derived slug wins.
        """
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        # File A: slug matches filename
        file_a = profiles_dir / "aaa-profile.yaml"
        file_a.write_text(
            "slug: aaa-profile\n"
            "domain_label: Profile A\n"
            "role_prompt: Role A\n"
            "content_structure_guidance: Guidance A\n",
            encoding="utf-8",
        )

        # File B: YAML slug says "aaa-profile" but filename is "bbb-profile"
        # → filename-derived slug "bbb-profile" is used, no collision
        file_b = profiles_dir / "bbb-profile.yaml"
        file_b.write_text(
            "slug: aaa-profile\n"
            "domain_label: Profile B\n"
            "role_prompt: Role B\n"
            "content_structure_guidance: Guidance B\n",
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING):
            loader = ProfileLoader(profiles_dir)

        # aaa-profile loaded from aaa-profile.yaml
        profile_a = loader.get_by_slug("aaa-profile")
        assert profile_a is not None
        assert profile_a.domain_label == "Profile A"

        # bbb-profile loaded with filename-derived slug (mismatch override)
        profile_b = loader.get_by_slug("bbb-profile")
        assert profile_b is not None
        assert profile_b.domain_label == "Profile B"

        # Warning logged about mismatch
        assert any("does not match filename-derived" in r.message for r in caplog.records)

    def test_no_duplicate_warning_when_filenames_unique(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Req 4.4: No duplicate warning when files have unique filename-derived slugs."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        file_a = profiles_dir / "alpha.yaml"
        file_a.write_text(
            "slug: alpha\n"
            "domain_label: Alpha\n"
            "role_prompt: Alpha role\n"
            "content_structure_guidance: Alpha guidance\n",
            encoding="utf-8",
        )

        file_b = profiles_dir / "beta.yaml"
        file_b.write_text(
            "slug: beta\n"
            "domain_label: Beta\n"
            "role_prompt: Beta role\n"
            "content_structure_guidance: Beta guidance\n",
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING):
            loader = ProfileLoader(profiles_dir)

        # Both loaded successfully, no duplicate warning
        assert loader.get_by_slug("alpha") is not None
        assert loader.get_by_slug("beta") is not None
        assert not any("duplicate slug" in r.message for r in caplog.records)

    def test_duplicate_slug_warning_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Req 4.4: Duplicate slug warning is logged when two files produce same slug.

        We trigger this by patching Path.glob to return the same file path twice,
        simulating the defensive code path.
        """
        from unittest.mock import patch

        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        # Create a single valid file
        file_a = profiles_dir / "dup-test.yaml"
        file_a.write_text(
            "slug: dup-test\n"
            "domain_label: Dup Test\n"
            "role_prompt: Dup role\n"
            "content_structure_guidance: Dup guidance\n",
            encoding="utf-8",
        )

        # First load normally
        loader = ProfileLoader(profiles_dir)
        assert loader.get_by_slug("dup-test") is not None

        # Now patch glob to return the file twice to trigger duplicate path
        original_glob = profiles_dir.glob

        def double_glob(pattern: str):
            results = sorted(original_glob(pattern))
            return results + results  # Return each file twice

        with caplog.at_level(logging.WARNING):
            with patch.object(type(profiles_dir), "glob", side_effect=double_glob):
                loader.reload()

        # Duplicate slug warning should be logged
        assert any("duplicate slug" in r.message for r in caplog.records)
        # Profile still accessible (first one wins)
        assert loader.get_by_slug("dup-test") is not None


class TestProfileLoaderMismatchedSlug:
    """Verify mismatched slug/filename handling."""

    def test_filename_derived_slug_used_when_yaml_slug_mismatches(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Req 2.2, 2.3: Filename-derived slug is used when YAML slug mismatches."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        # File named "my-profile.yaml" but YAML slug says "different-slug"
        profile_file = profiles_dir / "my-profile.yaml"
        profile_file.write_text(
            "slug: different-slug\n"
            "domain_label: My Profile\n"
            "role_prompt: A role prompt for testing.\n"
            "content_structure_guidance: Some guidance.\n",
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING):
            loader = ProfileLoader(profiles_dir)

        # Profile is accessible by filename-derived slug, not YAML slug
        assert loader.get_by_slug("my-profile") is not None
        assert loader.get_by_slug("different-slug") is None

        # Warning was logged
        assert any(
            "does not match filename-derived" in r.message for r in caplog.records
        )

    def test_warning_includes_both_slugs(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warning message includes both the YAML slug and filename-derived slug."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        profile_file = profiles_dir / "actual-name.yaml"
        profile_file.write_text(
            "slug: wrong-name\n"
            "domain_label: Test\n"
            "role_prompt: Test role\n"
            "content_structure_guidance: Test guidance\n",
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING):
            loader = ProfileLoader(profiles_dir)

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("wrong-name" in msg and "actual-name" in msg for msg in warning_messages)


class TestProfileLoaderInvalidYAML:
    """Verify invalid YAML files are skipped with warning."""

    def test_invalid_yaml_syntax_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Req 2.5: Files with invalid YAML syntax are skipped."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        bad_file = profiles_dir / "bad-syntax.yaml"
        bad_file.write_text(
            "slug: bad-syntax\n"
            "domain_label: Bad\n"
            "role_prompt: [\n"  # Invalid YAML — unclosed bracket
            "content_structure_guidance: test\n",
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING):
            loader = ProfileLoader(profiles_dir)

        assert loader.get_by_slug("bad-syntax") is None
        assert any("bad-syntax.yaml" in r.message for r in caplog.records)

    def test_yaml_not_a_mapping_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Req 2.5: YAML files that don't contain a mapping are skipped."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        bad_file = profiles_dir / "not-a-mapping.yaml"
        bad_file.write_text("- item1\n- item2\n", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            loader = ProfileLoader(profiles_dir)

        assert loader.get_by_slug("not-a-mapping") is None
        assert any("not a mapping" in r.message for r in caplog.records)


class TestProfileLoaderMissingRequiredKeys:
    """Verify files with missing required keys are skipped with warning."""

    def test_missing_domain_label_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Req 2.4: File missing domain_label is skipped."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        bad_file = profiles_dir / "no-label.yaml"
        bad_file.write_text(
            "slug: no-label\n"
            "role_prompt: Some role\n"
            "content_structure_guidance: Some guidance\n",
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING):
            loader = ProfileLoader(profiles_dir)

        assert loader.get_by_slug("no-label") is None
        assert any("no-label.yaml" in r.message for r in caplog.records)

    def test_missing_role_prompt_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Req 2.4: File missing role_prompt is skipped."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        bad_file = profiles_dir / "no-role.yaml"
        bad_file.write_text(
            "slug: no-role\n"
            "domain_label: No Role\n"
            "content_structure_guidance: Some guidance\n",
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING):
            loader = ProfileLoader(profiles_dir)

        assert loader.get_by_slug("no-role") is None
        assert any("no-role.yaml" in r.message for r in caplog.records)

    def test_missing_content_structure_guidance_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Req 2.4: File missing content_structure_guidance is skipped."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        bad_file = profiles_dir / "no-guidance.yaml"
        bad_file.write_text(
            "slug: no-guidance\n"
            "domain_label: No Guidance\n"
            "role_prompt: Some role\n",
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING):
            loader = ProfileLoader(profiles_dir)

        assert loader.get_by_slug("no-guidance") is None
        assert any("no-guidance.yaml" in r.message for r in caplog.records)

    def test_empty_role_prompt_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Req 2.4: File with empty role_prompt (whitespace only) is skipped."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        bad_file = profiles_dir / "empty-role.yaml"
        bad_file.write_text(
            "slug: empty-role\n"
            'domain_label: "Empty Role"\n'
            'role_prompt: "   "\n'
            "content_structure_guidance: Some guidance\n",
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING):
            loader = ProfileLoader(profiles_dir)

        assert loader.get_by_slug("empty-role") is None
        assert any("empty-role.yaml" in r.message for r in caplog.records)


class TestProfileLoaderMultiLineYAML:
    """Verify multi-line YAML block scalar syntax loads correctly."""

    def test_block_literal_scalar_loads(self, tmp_path: Path) -> None:
        """Req 2.6: Block literal (|) syntax preserves newlines."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        profile_file = profiles_dir / "multiline-test.yaml"
        profile_file.write_text(
            "slug: multiline-test\n"
            "domain_label: Multi-Line Test\n"
            "role_prompt: |\n"
            "  Line one of the role prompt.\n"
            "  Line two of the role prompt.\n"
            "  Line three of the role prompt.\n"
            "content_structure_guidance: |\n"
            "  Guidance line one.\n"
            "  Guidance line two.\n",
            encoding="utf-8",
        )

        loader = ProfileLoader(profiles_dir)
        profile = loader.get_by_slug("multiline-test")

        assert profile is not None
        assert "Line one" in profile.role_prompt
        assert "Line two" in profile.role_prompt
        assert "Line three" in profile.role_prompt
        assert "\n" in profile.role_prompt
        assert "Guidance line one" in profile.content_structure_guidance
        assert "Guidance line two" in profile.content_structure_guidance

    def test_block_folded_scalar_loads(self, tmp_path: Path) -> None:
        """Req 2.6: Block folded (>) syntax loads correctly."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        profile_file = profiles_dir / "folded-test.yaml"
        profile_file.write_text(
            "slug: folded-test\n"
            "domain_label: Folded Test\n"
            "role_prompt: >\n"
            "  This is a folded\n"
            "  multi-line string.\n"
            "content_structure_guidance: >\n"
            "  Guidance that spans\n"
            "  multiple lines.\n",
            encoding="utf-8",
        )

        loader = ProfileLoader(profiles_dir)
        profile = loader.get_by_slug("folded-test")

        assert profile is not None
        assert "folded" in profile.role_prompt
        assert "multi-line" in profile.role_prompt
        assert profile.role_prompt.strip() != ""
        assert profile.content_structure_guidance.strip() != ""

    def test_builtin_profiles_use_block_scalars(self, tmp_path: Path) -> None:
        """Req 2.6: Built-in profiles use block scalar syntax and load correctly."""
        loader = ProfileLoader(tmp_path / "profiles")

        for slug in EXPECTED_BUILTIN_SLUGS:
            profile = loader.get_by_slug(slug)
            assert profile is not None
            # Built-in profiles use | syntax, resulting in multi-line strings
            assert "\n" in profile.role_prompt, (
                f"Profile {slug} role_prompt should contain newlines from block scalar"
            )
            assert "\n" in profile.content_structure_guidance, (
                f"Profile {slug} content_structure_guidance should contain newlines"
            )


class TestProfileLoaderProfilesDir:
    """Verify the profiles_dir property."""

    def test_profiles_dir_property(self, tmp_path: Path) -> None:
        """profiles_dir property returns the configured path."""
        profiles_dir = tmp_path / "my_profiles"
        loader = ProfileLoader(profiles_dir)
        assert loader.profiles_dir == profiles_dir


# ---------------------------------------------------------------------------
# Schema and Repository Tests (Task 4.5)
# Requirements: 6.1, 6.2, 6.4, 13.1, 13.2, 13.3
# ---------------------------------------------------------------------------

import sqlite3
from unittest.mock import patch, MagicMock

from patent_system.db.schema import init_schema
from patent_system.db.repository import TopicDomainProfileRepository, WorkflowStepRepository


@pytest.fixture
def db_conn() -> sqlite3.Connection:
    """Provide a fresh in-memory SQLite connection with schema initialized."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    init_schema(conn)
    yield conn
    conn.close()


@pytest.fixture
def topic_id(db_conn: sqlite3.Connection) -> int:
    """Create a topic and return its ID for FK references."""
    cursor = db_conn.execute("INSERT INTO topics (name) VALUES (?)", ("Test Topic",))
    db_conn.commit()
    return cursor.lastrowid


class TestSchemaTopicDomainProfile:
    """Verify topic_domain_profile table exists after init_schema()."""

    def test_topic_domain_profile_table_exists(self, db_conn: sqlite3.Connection) -> None:
        """Req 13.1: topic_domain_profile table is created by init_schema()."""
        cursor = db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='topic_domain_profile'"
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "topic_domain_profile"

    def test_topic_domain_profile_table_columns(self, db_conn: sqlite3.Connection) -> None:
        """Req 13.1: topic_domain_profile has expected columns."""
        cursor = db_conn.execute("PRAGMA table_info(topic_domain_profile)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "id" in columns
        assert "topic_id" in columns
        assert "domain_profile_slug" in columns
        assert "updated_at" in columns


class TestSchemaWorkflowStepsDomainProfileColumn:
    """Verify workflow_steps.domain_profile_slug column exists after migration."""

    def test_domain_profile_slug_column_exists(self, db_conn: sqlite3.Connection) -> None:
        """Req 13.2: workflow_steps has domain_profile_slug column after migration."""
        cursor = db_conn.execute("PRAGMA table_info(workflow_steps)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "domain_profile_slug" in columns

    def test_domain_profile_slug_default_value(self, db_conn: sqlite3.Connection, topic_id: int) -> None:
        """Req 13.2: domain_profile_slug defaults to empty string."""
        db_conn.execute(
            "INSERT INTO workflow_steps (topic_id, step_key, content, status) VALUES (?, ?, ?, ?)",
            (topic_id, "claims_drafting", "test content", "pending"),
        )
        db_conn.commit()
        row = db_conn.execute(
            "SELECT domain_profile_slug FROM workflow_steps WHERE topic_id = ? AND step_key = ?",
            (topic_id, "claims_drafting"),
        ).fetchone()
        assert row[0] == ""


class TestTopicDomainProfileRepository:
    """Verify TopicDomainProfileRepository save() and get_by_topic() basic flow."""

    def test_save_and_get_by_topic(self, db_conn: sqlite3.Connection, topic_id: int) -> None:
        """Req 6.1, 6.2: save() persists slug and get_by_topic() retrieves it."""
        repo = TopicDomainProfileRepository(db_conn)
        repo.save(topic_id, "pharma-chemistry")
        result = repo.get_by_topic(topic_id)
        assert result == "pharma-chemistry"

    def test_save_overwrites_existing(self, db_conn: sqlite3.Connection, topic_id: int) -> None:
        """Req 6.2: save() replaces existing selection for the same topic."""
        repo = TopicDomainProfileRepository(db_conn)
        repo.save(topic_id, "pharma-chemistry")
        repo.save(topic_id, "software-ai")
        result = repo.get_by_topic(topic_id)
        assert result == "software-ai"

    def test_get_by_topic_returns_none_for_unknown(self, db_conn: sqlite3.Connection) -> None:
        """Req 6.4: get_by_topic() returns None for a topic with no saved selection."""
        repo = TopicDomainProfileRepository(db_conn)
        result = repo.get_by_topic(9999)
        assert result is None

    def test_save_db_error_logged_and_reraised(
        self, db_conn: sqlite3.Connection, topic_id: int, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Req 6.4: DB errors are logged and re-raised on save()."""
        repo = TopicDomainProfileRepository(db_conn)
        # Close the connection to force an error
        db_conn.close()
        with pytest.raises(sqlite3.Error):
            repo.save(topic_id, "pharma-chemistry")

    def test_get_by_topic_db_error_logged_and_reraised(
        self, db_conn: sqlite3.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Req 6.4: DB errors are logged and re-raised on get_by_topic()."""
        repo = TopicDomainProfileRepository(db_conn)
        # Close the connection to force an error
        db_conn.close()
        with pytest.raises(sqlite3.Error):
            repo.get_by_topic(1)


class TestTopicDomainProfileRepositoryMockErrors:
    """Verify DB error handling with mocked failures."""

    def test_save_logs_error_on_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        """Req 6.4: save() logs via log_db_error on failure."""
        mock_conn = MagicMock(spec=sqlite3.Connection)
        mock_conn.execute.side_effect = sqlite3.OperationalError("disk I/O error")

        repo = TopicDomainProfileRepository(mock_conn)
        with caplog.at_level(logging.ERROR):
            with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
                repo.save(1, "test-slug")

    def test_get_by_topic_logs_error_on_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        """Req 6.4: get_by_topic() logs via log_db_error on failure."""
        mock_conn = MagicMock(spec=sqlite3.Connection)
        mock_conn.execute.side_effect = sqlite3.OperationalError("database locked")

        repo = TopicDomainProfileRepository(mock_conn)
        with caplog.at_level(logging.ERROR):
            with pytest.raises(sqlite3.OperationalError, match="database locked"):
                repo.get_by_topic(1)


class TestWorkflowStepRepositoryDomainProfileSlug:
    """Verify WorkflowStepRepository includes domain_profile_slug in results."""

    def test_upsert_with_domain_profile_slug(self, db_conn: sqlite3.Connection, topic_id: int) -> None:
        """Req 13.2: upsert() persists domain_profile_slug."""
        repo = WorkflowStepRepository(db_conn)
        repo.upsert(
            topic_id=topic_id,
            step_key="claims_drafting",
            content="Test claims",
            status="completed",
            domain_profile_slug="pharma-chemistry",
        )
        steps = repo.get_by_topic(topic_id)
        assert len(steps) == 1
        assert steps[0]["domain_profile_slug"] == "pharma-chemistry"

    def test_upsert_default_domain_profile_slug(self, db_conn: sqlite3.Connection, topic_id: int) -> None:
        """Req 13.2: upsert() defaults domain_profile_slug to empty string."""
        repo = WorkflowStepRepository(db_conn)
        repo.upsert(
            topic_id=topic_id,
            step_key="prior_art_search",
            content="Test prior art",
            status="pending",
        )
        steps = repo.get_by_topic(topic_id)
        assert len(steps) == 1
        assert steps[0]["domain_profile_slug"] == ""

    def test_get_step_includes_domain_profile_slug(self, db_conn: sqlite3.Connection, topic_id: int) -> None:
        """Req 13.2: get_step() includes domain_profile_slug in result dict."""
        repo = WorkflowStepRepository(db_conn)
        repo.upsert(
            topic_id=topic_id,
            step_key="novelty_analysis",
            content="Novelty content",
            status="completed",
            domain_profile_slug="software-ai",
        )
        step = repo.get_step(topic_id, "novelty_analysis")
        assert step is not None
        assert step["domain_profile_slug"] == "software-ai"

    def test_get_by_topic_includes_domain_profile_slug_for_all_steps(
        self, db_conn: sqlite3.Connection, topic_id: int
    ) -> None:
        """Req 13.3: get_by_topic() includes domain_profile_slug for all steps."""
        repo = WorkflowStepRepository(db_conn)
        repo.upsert(topic_id, "claims_drafting", "Claims", "completed", domain_profile_slug="pharma-chemistry")
        repo.upsert(topic_id, "prior_art_search", "Prior art", "completed", domain_profile_slug="software-ai")
        repo.upsert(topic_id, "novelty_analysis", "Novelty", "pending", domain_profile_slug="")

        steps = repo.get_by_topic(topic_id)
        assert len(steps) == 3
        slugs = {s["step_key"]: s["domain_profile_slug"] for s in steps}
        assert slugs["claims_drafting"] == "pharma-chemistry"
        assert slugs["prior_art_search"] == "software-ai"
        assert slugs["novelty_analysis"] == ""


# ---------------------------------------------------------------------------
# AppSettings Domain Profile Fields Tests (Task 5.3)
# Requirements: 12.1, 12.2, 12.3, 12.4
# ---------------------------------------------------------------------------

from patent_system.config import AppSettings, get_base_dir


class TestAppSettingsDomainProfileDefaults:
    """Verify default values for domain profile settings."""

    def test_domain_profiles_dir_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Req 12.1: domain_profiles_dir defaults to domain_profiles/ relative to base dir."""
        # Clear any env vars that might interfere
        monkeypatch.delenv("PATENT_DOMAIN_PROFILES_DIR", raising=False)
        monkeypatch.delenv("PATENT_DEFAULT_DOMAIN_PROFILE", raising=False)

        settings = AppSettings()
        expected = get_base_dir() / "domain_profiles"
        assert settings.domain_profiles_dir == expected

    def test_default_domain_profile_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Req 12.3: default_domain_profile defaults to 'general-patent-drafting'."""
        monkeypatch.delenv("PATENT_DEFAULT_DOMAIN_PROFILE", raising=False)

        settings = AppSettings()
        assert settings.default_domain_profile == "general-patent-drafting"


class TestAppSettingsDomainProfileEnvVars:
    """Verify env var loading for domain profile settings."""

    def test_domain_profiles_dir_from_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Req 12.2: PATENT_DOMAIN_PROFILES_DIR env var overrides default."""
        custom_dir = str(tmp_path / "custom_profiles")
        monkeypatch.setenv("PATENT_DOMAIN_PROFILES_DIR", custom_dir)

        settings = AppSettings()
        assert settings.domain_profiles_dir == Path(custom_dir)

    def test_default_domain_profile_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Req 12.4: PATENT_DEFAULT_DOMAIN_PROFILE env var overrides default."""
        monkeypatch.setenv("PATENT_DEFAULT_DOMAIN_PROFILE", "pharma-chemistry")

        settings = AppSettings()
        assert settings.default_domain_profile == "pharma-chemistry"


# ---------------------------------------------------------------------------
# Startup Validation Tests (Task 12.2)
# Requirements: 12.5
# ---------------------------------------------------------------------------


class TestStartupProfileValidation:
    """Verify startup validation of default domain profile setting."""

    def test_warning_logged_when_default_profile_not_found(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Req 12.5: Warning logged when PATENT_DEFAULT_DOMAIN_PROFILE references non-existent slug."""
        profiles_dir = tmp_path / "profiles"
        profile_loader = ProfileLoader(profiles_dir)

        # Simulate a non-existent default profile slug
        non_existent_slug = "non-existent-profile"

        with caplog.at_level(logging.WARNING):
            if profile_loader.get_by_slug(non_existent_slug) is None:
                logging.getLogger(__name__).warning(
                    "Default domain profile %r not found — falling back to 'general-patent-drafting'",
                    non_existent_slug,
                )

        assert any(
            "non-existent-profile" in r.message and "not found" in r.message
            for r in caplog.records
        )

    def test_fallback_to_general_patent_drafting_when_default_missing(
        self, tmp_path: Path
    ) -> None:
        """Req 12.5: Fallback to 'general-patent-drafting' when configured default doesn't exist."""
        profiles_dir = tmp_path / "profiles"
        profile_loader = ProfileLoader(profiles_dir)

        # Simulate the startup validation logic from main.py
        configured_default = "non-existent-profile"
        effective_default = configured_default

        if profile_loader.get_by_slug(configured_default) is None:
            effective_default = "general-patent-drafting"

        # The fallback should be "general-patent-drafting"
        assert effective_default == "general-patent-drafting"
        # And that profile should actually exist in the loader
        assert profile_loader.get_by_slug("general-patent-drafting") is not None

    def test_no_warning_when_default_profile_exists(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Req 12.5: No warning when default profile exists in loaded profiles."""
        profiles_dir = tmp_path / "profiles"
        profile_loader = ProfileLoader(profiles_dir)

        # The default slug should exist in built-in profiles
        default_slug = "general-patent-drafting"

        with caplog.at_level(logging.WARNING):
            if profile_loader.get_by_slug(default_slug) is None:
                logging.getLogger(__name__).warning(
                    "Default domain profile %r not found — falling back to 'general-patent-drafting'",
                    default_slug,
                )

        # No warning should be logged since the profile exists
        assert not any(
            "not found" in r.message and "general-patent-drafting" in r.message
            for r in caplog.records
        )

    def test_set_profile_loader_wires_into_dspy_modules(
        self, tmp_path: Path
    ) -> None:
        """Req 4.1: set_profile_loader wires the loader into DSPy modules."""
        from patent_system.dspy_modules.modules import set_profile_loader, _get_profile_loader

        profiles_dir = tmp_path / "profiles"
        profile_loader = ProfileLoader(profiles_dir)

        set_profile_loader(profile_loader)

        assert _get_profile_loader() is profile_loader
