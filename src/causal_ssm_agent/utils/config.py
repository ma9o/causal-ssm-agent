"""Configuration loader for the causal agent pipeline."""

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


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
class ParaphrasingConfig:
    """AutoElicit-style paraphrased prompting configuration."""

    enabled: bool = False  # Off by default (cost)
    n_paraphrases: int = 10


@dataclass(frozen=True)
class Stage4Config:
    """Stage 4: Prior Elicitation (Orchestrator-Worker Architecture)."""

    model: str
    literature_search: LiteratureSearchConfig = LiteratureSearchConfig()
    paraphrasing: ParaphrasingConfig = ParaphrasingConfig()
    worker_model: str | None = None  # If None, uses stage2_workers.model


@dataclass(frozen=True)
class SVIConfig:
    """SVI-specific inference settings."""

    num_steps: int = 5000
    learning_rate: float = 0.01
    guide_type: str = "mvn"


@dataclass(frozen=True)
class NUTSConfig:
    """NUTS-specific inference settings."""

    target_accept_prob: float = 0.85
    max_tree_depth: int = 8


@dataclass(frozen=True)
class InferenceConfig:
    """Inference configuration (method + sampler settings)."""

    method: str = "svi"
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    seed: int = 0
    gpu: str | None = None
    svi: SVIConfig = SVIConfig()
    nuts: NUTSConfig = NUTSConfig()

    def to_sampler_config(self, method_override: str | None = None) -> dict:
        """Build a flat sampler config dict for SSMModelBuilder.

        Args:
            method_override: Override the configured method (e.g. "nuts")

        Returns:
            Flat dict with method + relevant sampler keys
        """
        method = method_override or self.method
        config: dict = {
            "method": method,
            "num_warmup": self.num_warmup,
            "num_samples": self.num_samples,
            "num_chains": self.num_chains,
            "seed": self.seed,
        }
        if method == "svi":
            config["num_steps"] = self.svi.num_steps
            config["learning_rate"] = self.svi.learning_rate
            config["guide_type"] = self.svi.guide_type
        elif method == "nuts":
            config["target_accept_prob"] = self.nuts.target_accept_prob
            config["max_tree_depth"] = self.nuts.max_tree_depth
        return config


@dataclass(frozen=True)
class LLMConfig:
    """LLM generation settings shared across all model calls."""

    max_tokens: int = 65536
    timeout: int = 900
    reasoning_effort: str = "high"
    reasoning_tokens: int = 32768


@dataclass(frozen=True)
class PipelineBehaviorConfig:
    """Pipeline-level behavioral settings."""

    max_prior_retries: int = 3


@dataclass(frozen=True)
class PipelineConfig:
    """Full pipeline configuration."""

    stage1_structure_proposal: Stage1Config
    stage2_workers: Stage2Config
    stage4_prior_elicitation: Stage4Config
    inference: InferenceConfig = InferenceConfig()
    llm: LLMConfig = LLMConfig()
    pipeline: PipelineBehaviorConfig = PipelineBehaviorConfig()


def get_secret(name: str) -> str | None:
    """Get a secret value, trying Prefect Secret block first, then env var.

    Args:
        name: Secret name (used as both block slug and env var name)

    Returns:
        Secret value, or None if not found in either location
    """
    # Try Prefect Secret block first
    try:
        from prefect.blocks.system import Secret

        block = Secret.load(name.lower().replace("_", "-"))
        return block.get()
    except Exception:
        pass

    # Fall back to environment variable
    return os.getenv(name)


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

    with config_path.open() as f:
        raw = yaml.safe_load(f)

    stage4_raw = raw["stage4_prior_elicitation"]
    lit_search_raw = stage4_raw.get("literature_search", {})
    paraphrasing_raw = stage4_raw.get("paraphrasing", {})
    stage4_config = Stage4Config(
        model=stage4_raw["model"],
        literature_search=LiteratureSearchConfig(**lit_search_raw)
        if lit_search_raw
        else LiteratureSearchConfig(),
        paraphrasing=ParaphrasingConfig(**paraphrasing_raw)
        if paraphrasing_raw
        else ParaphrasingConfig(),
        worker_model=stage4_raw.get("worker_model"),
    )

    # Parse inference section (optional)
    inference_raw = raw.get("inference", {})
    svi_raw = inference_raw.pop("svi", {})
    nuts_raw = inference_raw.pop("nuts", {})
    inference_config = InferenceConfig(
        **inference_raw,
        svi=SVIConfig(**svi_raw) if svi_raw else SVIConfig(),
        nuts=NUTSConfig(**nuts_raw) if nuts_raw else NUTSConfig(),
    )

    # Parse llm section (optional)
    llm_raw = raw.get("llm", {})
    llm_config = LLMConfig(**llm_raw) if llm_raw else LLMConfig()

    # Parse pipeline section (optional)
    pipeline_raw = raw.get("pipeline", {})
    pipeline_config = (
        PipelineBehaviorConfig(**pipeline_raw) if pipeline_raw else PipelineBehaviorConfig()
    )

    return PipelineConfig(
        stage1_structure_proposal=Stage1Config(**raw["stage1_structure_proposal"]),
        stage2_workers=Stage2Config(**raw["stage2_workers"]),
        stage4_prior_elicitation=stage4_config,
        inference=inference_config,
        llm=llm_config,
        pipeline=pipeline_config,
    )


def get_config() -> PipelineConfig:
    """Get the pipeline configuration."""
    return load_config()
