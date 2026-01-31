"""Configuration loader for the causal agent pipeline."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Stage1Config:
    """Stage 1: Structure Proposal (Orchestrator)."""

    model: str
    sample_chunks: int
    chunk_size: int


@dataclass(frozen=True)
class Stage2Config:
    """Stage 2: Dimension Population (Workers)."""

    model: str
    chunk_size: int


@dataclass(frozen=True)
class LiteratureSearchConfig:
    """Literature search configuration for grounding priors."""

    enabled: bool = True
    model: str = "exa-research"
    timeout_ms: int = 120000


@dataclass(frozen=True)
class Stage4Config:
    """Stage 4: Prior Elicitation."""

    model: str
    literature_search: LiteratureSearchConfig = LiteratureSearchConfig()


@dataclass(frozen=True)
class PipelineConfig:
    """Full pipeline configuration."""

    stage1_structure_proposal: Stage1Config
    stage2_workers: Stage2Config
    stage4_prior_elicitation: Stage4Config


def _find_config_path() -> Path:
    """Find config.yaml by walking up from this file to the project root."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        config_path = parent / "config.yaml"
        if config_path.exists():
            return config_path
    raise FileNotFoundError("config.yaml not found in any parent directory")


@lru_cache(maxsize=1)
def load_config() -> PipelineConfig:
    """Load and parse the pipeline configuration.

    Returns cached config on subsequent calls.
    """
    config_path = _find_config_path()

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    stage4_raw = raw["stage4_prior_elicitation"]
    lit_search_raw = stage4_raw.get("literature_search", {})
    stage4_config = Stage4Config(
        model=stage4_raw["model"],
        literature_search=LiteratureSearchConfig(**lit_search_raw) if lit_search_raw else LiteratureSearchConfig(),
    )

    return PipelineConfig(
        stage1_structure_proposal=Stage1Config(**raw["stage1_structure_proposal"]),
        stage2_workers=Stage2Config(**raw["stage2_workers"]),
        stage4_prior_elicitation=stage4_config,
    )


def get_config() -> PipelineConfig:
    """Get the pipeline configuration."""
    return load_config()
