"""Domain profile definitions and prefix generation for agent nodes.

Provides the ``DomainProfile`` Pydantic model, domain prefix generation,
prefix round-trip parsing, and profile resolution with fallback logic.

Domain profiles are orthogonal to personality modes — personality controls
tone/rigor, domain profiles control expertise and vocabulary.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 5.1, 5.2, 5.3, 5.4, 8.4, 14.1, 14.2, 14.3
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Protocol

import yaml
from pydantic import BaseModel, field_validator, ValidationError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

DEFAULT_PROFILE_SLUG = "general-patent-drafting"

# Regex for valid slugs: starts with lowercase alphanumeric, followed by
# lowercase alphanumeric, hyphens, or underscores.
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

# Regex for extracting the slug from a ``[Domain: <slug>]`` tag.
_DOMAIN_TAG_RE = re.compile(r"\[Domain:\s*([a-z0-9][a-z0-9_-]*)\]")


# ---------------------------------------------------------------------------
# Protocol for loader (avoids circular imports with ProfileLoader)
# ---------------------------------------------------------------------------


class ProfileLoaderProtocol(Protocol):
    """Protocol for objects that can look up domain profiles by slug."""

    def get_by_slug(self, slug: str) -> DomainProfile | None: ...


# ---------------------------------------------------------------------------
# DomainProfile model
# ---------------------------------------------------------------------------


class DomainProfile(BaseModel):
    """A domain expertise profile for patent drafting.

    Loaded from YAML files in the profiles directory. Each profile
    encapsulates domain-specific LLM persona and content structuring
    instructions.

    Attributes:
        slug: Unique identifier (lowercase alphanumeric, hyphens, underscores).
        domain_label: Human-readable domain label for display.
        role_prompt: LLM persona text describing domain expertise.
        content_structure_guidance: Domain-specific content structuring instructions.
    """

    slug: str
    domain_label: str
    role_prompt: str
    content_structure_guidance: str

    @field_validator("slug")
    @classmethod
    def validate_slug(cls, v: str) -> str:
        """Slug must be non-empty, lowercase alphanumeric + hyphens + underscores.

        Must start with a lowercase letter or digit.
        """
        if not v or not _SLUG_RE.match(v):
            raise ValueError(
                f"Invalid slug {v!r}: must match ^[a-z0-9][a-z0-9_-]*$"
            )
        return v

    @field_validator("domain_label")
    @classmethod
    def validate_domain_label(cls, v: str) -> str:
        """Domain label must be non-empty after stripping whitespace."""
        if not v or not v.strip():
            raise ValueError("domain_label must be non-empty after stripping whitespace")
        return v

    @field_validator("role_prompt")
    @classmethod
    def validate_role_prompt(cls, v: str) -> str:
        """Role prompt must be non-empty after stripping whitespace."""
        if not v or not v.strip():
            raise ValueError("role_prompt must be non-empty after stripping whitespace")
        return v

    @field_validator("content_structure_guidance")
    @classmethod
    def validate_content_structure_guidance(cls, v: str) -> str:
        """Content structure guidance must be non-empty after stripping whitespace."""
        if not v or not v.strip():
            raise ValueError(
                "content_structure_guidance must be non-empty after stripping whitespace"
            )
        return v

    def to_yaml(self) -> str:
        """Serialize this profile to a YAML string.

        Returns:
            A YAML-formatted string containing all profile fields.
        """
        data = {
            "slug": self.slug,
            "domain_label": self.domain_label,
            "role_prompt": self.role_prompt,
            "content_structure_guidance": self.content_structure_guidance,
        }
        return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> DomainProfile:
        """Deserialize a DomainProfile from a YAML string.

        Args:
            yaml_str: A YAML-formatted string containing profile fields.

        Returns:
            A validated DomainProfile instance.

        Raises:
            ValueError: If the YAML is invalid or missing required fields.
            yaml.YAMLError: If the YAML syntax is invalid.
        """
        data = yaml.safe_load(yaml_str)
        if not isinstance(data, dict):
            raise ValueError("YAML content must be a mapping")
        return cls(**data)


# ---------------------------------------------------------------------------
# Domain prefix generation and parsing
# ---------------------------------------------------------------------------


def generate_domain_prefix(slug: str, loader: ProfileLoaderProtocol) -> str:
    """Return the domain prefix text for the given profile slug.

    Combines role_prompt and content_structure_guidance with a
    parseable [Domain: <slug>] tag. Falls back to default profile
    if slug is not found in the loader.

    Args:
        slug: The domain profile slug to generate a prefix for.
        loader: An object with a ``get_by_slug(slug)`` method.

    Returns:
        A non-empty string of at least 30 characters containing the
        role prompt, content structure guidance, and domain tag.
    """
    profile = loader.get_by_slug(slug)
    if profile is None:
        logger.warning(
            "Domain profile %r not found — falling back to default %r",
            slug,
            DEFAULT_PROFILE_SLUG,
        )
        profile = loader.get_by_slug(DEFAULT_PROFILE_SLUG)
        if profile is None:
            raise RuntimeError(
                f"Default domain profile {DEFAULT_PROFILE_SLUG!r} not found in loader"
            )

    prefix = (
        f"[Domain: {profile.slug}] "
        f"{profile.role_prompt}\n\n"
        f"{profile.content_structure_guidance}"
    )
    return prefix


def parse_slug_from_prefix(prefix: str) -> str:
    """Extract the domain profile slug from a [Domain: <slug>] tag.

    Args:
        prefix: A string containing a ``[Domain: <slug>]`` tag.

    Returns:
        The extracted slug string.

    Raises:
        ValueError: If no valid [Domain: ...] tag is found in *prefix*.
    """
    match = _DOMAIN_TAG_RE.search(prefix)
    if match is None:
        raise ValueError(f"No [Domain: ...] tag found in prefix: {prefix!r}")
    return match.group(1)


def resolve_domain_profile(
    slug: str | None,
    loader: ProfileLoaderProtocol,
) -> DomainProfile:
    """Resolve a slug to a DomainProfile with fallback chain.

    Resolution order:
    1. ``loader.get_by_slug(slug)`` if slug is non-empty
    2. ``loader.get_by_slug(DEFAULT_PROFILE_SLUG)``
    3. Raise RuntimeError (should never happen if built-in files exist)

    Args:
        slug: The profile slug to resolve, or None/empty for default.
        loader: An object with a ``get_by_slug(slug)`` method.

    Returns:
        The resolved DomainProfile.

    Raises:
        RuntimeError: If neither the requested slug nor the default
            profile can be found.
    """
    if slug:
        profile = loader.get_by_slug(slug)
        if profile is not None:
            return profile
        logger.warning(
            "Domain profile %r not found — falling back to default %r",
            slug,
            DEFAULT_PROFILE_SLUG,
        )

    # Fall back to default profile
    default_profile = loader.get_by_slug(DEFAULT_PROFILE_SLUG)
    if default_profile is not None:
        return default_profile

    raise RuntimeError(
        f"Default domain profile {DEFAULT_PROFILE_SLUG!r} not found. "
        "Ensure the profiles directory contains the built-in example files."
    )


# ---------------------------------------------------------------------------
# Built-in profile YAML content strings
# ---------------------------------------------------------------------------

BUILTIN_PROFILE_CONTENTS: dict[str, str] = {
    "general-patent-drafting.yaml": """\
slug: general-patent-drafting
domain_label: "General Patent Drafting"
role_prompt: |
  You are an experienced European patent attorney skilled in drafting patent
  applications across all technical domains. You produce precise, legally robust,
  and technically detailed patent specifications that comply with EPO and PCT
  requirements. You balance breadth of protection with clarity of disclosure.
content_structure_guidance: |
  Structure the application with clear technical problem-solution framing.
  Use hierarchical claim sets with independent claims of appropriate scope
  and dependent claims adding specific embodiments. Ensure the description
  supports every claim element with sufficient detail for a person skilled
  in the art to reproduce the invention without undue experimentation.
""",
    "pharma-chemistry.yaml": """\
slug: pharma-chemistry
domain_label: "Pharma & Chemistry (Focus on Formulation & Synergy)"
role_prompt: |
  You are an experienced European patent attorney with a PhD in Pharmacy and
  Chemistry. Your task is to draft precise, legally robust, and detailed patent
  applications in the pharmaceutical and chemical domain. You have deep expertise
  in formulation science, drug delivery systems, chemical synthesis pathways,
  and pharmacokinetics. You understand Markush structures, polymorphic forms,
  and regulatory considerations for pharmaceutical patents.
content_structure_guidance: |
  Focus on formulation details, synergy effects, and chemical compound specificity.
  Structure claims around composition, method of preparation, and use claims.
  Emphasize dosage forms, excipients, and pharmacokinetic properties. Use precise
  chemical nomenclature (IUPAC) and reference relevant pharmacopeial standards.
  Include experimental data supporting unexpected technical effects, particularly
  synergistic combinations and improved bioavailability.
""",
    "medtech-mechanical-engineering.yaml": """\
slug: medtech-mechanical-engineering
domain_label: "MedTech & Mechanical Engineering"
role_prompt: |
  You are an experienced European patent attorney specializing in mechanical
  engineering and medical technology. You draft patent applications for medical
  devices, surgical instruments, implants, and complex mechanical assemblies.
  You understand tolerance specifications, material selection criteria, and
  regulatory requirements (MDR/FDA) that influence claim scope and disclosure.
content_structure_guidance: |
  Structure claims as apparatus/device claims with clear structural relationships
  between components. Use precise spatial language (proximal, distal, longitudinal
  axis) and reference drawings extensively. Emphasize functional cooperation between
  mechanical elements. For medical devices, clearly distinguish structural features
  from method-of-use features and include manufacturing method claims where the
  process imparts novel structural characteristics.
""",
    "processes-manufacturing.yaml": """\
slug: processes-manufacturing
domain_label: "Processes & Manufacturing (CII)"
role_prompt: |
  You are an experienced European patent attorney specializing in process
  engineering and computer-implemented inventions (CII). You draft patent
  applications for manufacturing processes, industrial methods, and technical
  processes that may involve software control. You understand the EPO's
  approach to technical character and the distinction between technical and
  non-technical method steps.
content_structure_guidance: |
  Structure method claims with clearly ordered steps, each step specifying
  the technical action, parameters, and conditions. Articulate the technical
  effect achieved by each step and the overall process. For CII, emphasize
  the technical contribution beyond mere data processing. Include process
  parameters (temperature, pressure, flow rates) with ranges supported by
  examples. Provide alternative embodiments showing parameter variations.
""",
    "electrical-engineering-semiconductors.yaml": """\
slug: electrical-engineering-semiconductors
domain_label: "Electrical Engineering & Semiconductors"
role_prompt: |
  You are an experienced European patent attorney with expertise in electrical
  engineering and semiconductor technology. You draft patent applications for
  integrated circuits, signal processing systems, power electronics, and
  semiconductor fabrication processes. You understand circuit topologies,
  transistor-level design, and the interplay between device physics and
  system-level performance.
content_structure_guidance: |
  Structure claims around circuit configurations, signal processing methods,
  and semiconductor device structures. Use precise electrical terminology
  (impedance, transconductance, threshold voltage) and reference circuit
  diagrams. For semiconductor fabrication, describe layer-by-layer process
  sequences with material compositions and dimensional parameters. Emphasize
  the technical advantage in terms of performance metrics (speed, power
  consumption, area efficiency, signal-to-noise ratio).
""",
    "biotechnology-life-sciences.yaml": """\
slug: biotechnology-life-sciences
domain_label: "Biotechnology & Life Sciences"
role_prompt: |
  You are an experienced European patent attorney with a PhD in molecular
  biology or biochemistry. You draft patent applications for biotechnological
  inventions including recombinant proteins, antibodies, gene therapies, CRISPR
  applications, and diagnostic methods. You understand sequence listing
  requirements (WIPO ST.26), deposit requirements for biological materials,
  and the sufficiency of disclosure standards for biotech inventions.
content_structure_guidance: |
  Structure claims around nucleic acid sequences, polypeptides, vectors,
  host cells, and methods of production. Include sequence listings in WIPO
  ST.26 format with proper feature annotations. Describe biological processes
  with sufficient experimental detail including controls and statistical
  significance. Address industrial applicability explicitly for therapeutic
  and diagnostic claims. Include deposit information for novel biological
  materials where reproducibility cannot be ensured by written description alone.
""",
    "software-ai.yaml": """\
slug: software-ai
domain_label: "Software & Artificial Intelligence"
role_prompt: |
  You are an experienced European patent attorney specializing in software
  and artificial intelligence inventions. You draft patent applications that
  navigate the EPO's technical effect requirement for computer-implemented
  inventions. You understand machine learning architectures, data processing
  pipelines, and how to frame algorithmic innovations as technical solutions
  to technical problems.
content_structure_guidance: |
  Emphasize the technical effect over abstract algorithmic ideas. Frame claims
  as computer-implemented methods with clear technical purpose and measurable
  technical improvement (speed, accuracy, resource efficiency). Describe the
  system architecture with specific hardware-software interactions. For AI/ML
  inventions, specify training data characteristics, model architecture choices
  that produce technical advantages, and concrete technical applications.
  Avoid purely mathematical or business method framing. Include system claims
  mirroring method claims with processor and memory elements.
""",
    "materials-science-nanotechnology.yaml": """\
slug: materials-science-nanotechnology
domain_label: "Materials Science & Nanotechnology"
role_prompt: |
  You are an experienced European patent attorney with expertise in materials
  science and nanotechnology. You draft patent applications for novel materials,
  composites, coatings, nanostructures, and material processing methods. You
  understand crystallography, surface characterization techniques, and
  structure-property relationships at the nanoscale.
content_structure_guidance: |
  Structure claims around material compositions with precise stoichiometry or
  weight percentage ranges. Include characterization data (XRD, SEM, TEM,
  mechanical testing) that defines the material's novel properties. Describe
  synthesis or fabrication methods with reproducible parameters. Emphasize
  structure-property relationships and unexpected technical effects compared
  to prior art materials. For nanostructures, specify dimensional ranges and
  morphological features with measurement methods.
""",
    "telecommunications-standards.yaml": """\
slug: telecommunications-standards
domain_label: "Telecommunications & Standards"
role_prompt: |
  You are an experienced European patent attorney specializing in
  telecommunications and wireless communication systems. You draft patent
  applications for protocol implementations, network architectures, and
  signal processing methods relevant to standards bodies (3GPP, IEEE, ETSI).
  You understand standards-essential patent (SEP) considerations and FRAND
  licensing obligations.
content_structure_guidance: |
  Structure claims around protocol procedures, signaling sequences, and
  network entity interactions. Use precise telecommunications terminology
  aligned with relevant standards specifications. Describe message formats,
  channel configurations, and timing relationships. For SEP-relevant
  inventions, clearly map claim elements to specific standard sections.
  Include both method claims (from the perspective of individual network
  entities) and system claims showing entity cooperation. Reference
  standard document numbers where applicable.
""",
}


# ---------------------------------------------------------------------------
# ProfileLoader
# ---------------------------------------------------------------------------


class ProfileLoader:
    """Scans a directory for YAML profile files and maintains an in-memory registry.

    On construction or reload(), scans the profiles directory for .yaml files,
    validates each one, and builds a slug → DomainProfile mapping.

    Satisfies the ProfileLoaderProtocol interface.
    """

    def __init__(self, profiles_dir: Path) -> None:
        """Initialize the ProfileLoader.

        Args:
            profiles_dir: Path to the directory containing profile YAML files.
        """
        self._profiles_dir = Path(profiles_dir)
        self._profiles: dict[str, DomainProfile] = {}
        self._ensure_directory()
        self.reload()

    def _ensure_directory(self) -> None:
        """Create the profiles directory and populate with built-in examples if needed.

        If the directory does not exist, it is created and all built-in profile
        YAML files are written into it. If the directory exists but is empty,
        built-in profiles are also written.
        """
        newly_created = not self._profiles_dir.exists()
        self._profiles_dir.mkdir(parents=True, exist_ok=True)

        # Populate with built-in examples if directory is empty or newly created
        is_empty = not any(self._profiles_dir.iterdir())
        if newly_created or is_empty:
            logger.info(
                "Populating profiles directory %s with built-in examples",
                self._profiles_dir,
            )
            for filename, content in BUILTIN_PROFILE_CONTENTS.items():
                filepath = self._profiles_dir / filename
                filepath.write_text(content, encoding="utf-8")

    def reload(self) -> None:
        """Re-scan the directory and refresh the in-memory profile registry.

        Scans for .yaml files sorted alphabetically, validates each one,
        and builds a slug → DomainProfile mapping. Skips files with invalid
        YAML, missing required fields, or duplicate slugs. Logs warnings
        for each skipped file. Uses filename-derived slug if YAML slug
        mismatches the filename.
        """
        new_profiles: dict[str, DomainProfile] = {}

        yaml_files = sorted(self._profiles_dir.glob("*.yaml"))

        for filepath in yaml_files:
            try:
                content = filepath.read_text(encoding="utf-8")
                data = yaml.safe_load(content)

                if not isinstance(data, dict):
                    logger.warning(
                        "Skipping %s: YAML content is not a mapping", filepath.name
                    )
                    continue

                # Derive slug from filename (without .yaml extension)
                filename_slug = filepath.stem

                # Check if YAML slug mismatches filename
                yaml_slug = data.get("slug", "")
                if yaml_slug and yaml_slug != filename_slug:
                    logger.warning(
                        "Profile %s: YAML slug %r does not match filename-derived "
                        "slug %r — using filename-derived slug",
                        filepath.name,
                        yaml_slug,
                        filename_slug,
                    )
                    data["slug"] = filename_slug

                # Ensure slug is set from filename if missing
                if not data.get("slug"):
                    data["slug"] = filename_slug

                # Validate by constructing the model
                profile = DomainProfile(**data)

                # Check for duplicate slugs
                if profile.slug in new_profiles:
                    logger.warning(
                        "Skipping %s: duplicate slug %r (already loaded from "
                        "an earlier file)",
                        filepath.name,
                        profile.slug,
                    )
                    continue

                new_profiles[profile.slug] = profile

            except yaml.YAMLError as e:
                logger.warning("Skipping %s: invalid YAML syntax — %s", filepath.name, e)
            except ValidationError as e:
                logger.warning(
                    "Skipping %s: validation error — %s", filepath.name, e
                )
            except (OSError, ValueError) as e:
                logger.warning("Skipping %s: error reading file — %s", filepath.name, e)

        self._profiles = new_profiles
        logger.info(
            "Loaded %d domain profile(s) from %s", len(self._profiles), self._profiles_dir
        )

    def get_all(self) -> list[DomainProfile]:
        """Return all loaded profiles, sorted by domain_label.

        Returns:
            A list of DomainProfile instances sorted alphabetically by domain_label.
        """
        return sorted(self._profiles.values(), key=lambda p: p.domain_label)

    def get_by_slug(self, slug: str) -> DomainProfile | None:
        """Return a profile by slug, or None if not found.

        Args:
            slug: The profile slug to look up.

        Returns:
            The matching DomainProfile, or None if no profile with that slug exists.
        """
        return self._profiles.get(slug)

    @property
    def profiles_dir(self) -> Path:
        """Return the path to the profiles directory."""
        return self._profiles_dir
