"""Configuration loader for the causal agent pipeline."""

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Centralized .env loading — all modules that need env vars import from config.py
# (or from modules that import config.py), so this runs once at import time.
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")


@dataclass(frozen=True)
class Stage0Config:
    """Stage 0: Agentic Data Ingestion."""

    model: str = "openrouter/anthropic/claude-sonnet-4"


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
    max_concurrent_workers: int = 4
    submission_batch_size: int = 50


@dataclass(frozen=True)
class LiteratureSearchConfig:
    """Literature search configuration for grounding priors."""

    enabled: bool = True


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
class SMCConfig:
    """Tempered SMC / Laplace-EM / Structured VI / DPF inference settings."""

    n_outer: int = 100
    n_csmc_particles: int = 20
    n_mh_steps: int = 10
    param_step_size: float = 0.1
    n_warmup: int | None = None
    n_leapfrog: int = 5
    adaptive_tempering: bool = True
    target_ess_ratio: float = 0.5
    waste_free: bool = False
    n_ieks_iters: int = 5


@dataclass(frozen=True)
class InferenceConfig:
    """Inference configuration (method + sampler settings)."""

    method: str = "auto"
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    seed: int = 0
    gpu: str | None = None
    svi: SVIConfig = SVIConfig()
    nuts: NUTSConfig = NUTSConfig()
    smc: SMCConfig = SMCConfig()

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
        elif method in ("laplace_em", "tempered_smc", "structured_vi", "dpf"):
            config["n_outer"] = self.smc.n_outer
            config["n_csmc_particles"] = self.smc.n_csmc_particles
            config["n_mh_steps"] = self.smc.n_mh_steps
            config["param_step_size"] = self.smc.param_step_size
            config["n_leapfrog"] = self.smc.n_leapfrog
            config["adaptive_tempering"] = self.smc.adaptive_tempering
            config["target_ess_ratio"] = self.smc.target_ess_ratio
            config["waste_free"] = self.smc.waste_free
            if self.smc.n_warmup is not None:
                config["n_warmup"] = self.smc.n_warmup
            if method == "laplace_em":
                config["n_ieks_iters"] = self.smc.n_ieks_iters
        return config


@dataclass(frozen=True)
class LLMConfig:
    """LLM generation settings shared across all model calls."""

    max_tokens: int = 65536
    timeout: int = 900
    reasoning_effort: str = "high"
    verbose_logging: bool = False
    log_reasoning: bool = False
    log_output_char_limit: int = 8000


@dataclass(frozen=True)
class PipelineBehaviorConfig:
    """Pipeline-level behavioral settings."""

    max_prior_retries: int = 3
    override_gates: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    """Full pipeline configuration."""

    stage0_ingestion: Stage0Config
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
        return block.get()  # ty: ignore[unresolved-attribute]
    except Exception:
        logger.debug("Prefect Secret block '%s' not found; trying env var", name)

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


def _env_bool(name: str) -> bool | None:
    """Parse a boolean environment override, returning None when unset or invalid."""
    raw = os.getenv(name)
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning("Ignoring invalid boolean override %s=%r", name, raw)
    return None


def _env_int(name: str) -> int | None:
    """Parse an integer environment override, returning None when unset or invalid."""
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring invalid integer override %s=%r", name, raw)
        return None


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
    smc_raw = inference_raw.pop("smc", {})
    inference_config = InferenceConfig(
        **inference_raw,
        svi=SVIConfig(**svi_raw) if svi_raw else SVIConfig(),
        nuts=NUTSConfig(**nuts_raw) if nuts_raw else NUTSConfig(),
        smc=SMCConfig(**smc_raw) if smc_raw else SMCConfig(),
    )

    # Parse llm section (optional)
    llm_raw = dict(raw.get("llm", {}))
    verbose_logging = _env_bool("CAUSAL_SSM_LLM_VERBOSE_LOGGING")
    if verbose_logging is not None:
        llm_raw["verbose_logging"] = verbose_logging
    log_reasoning = _env_bool("CAUSAL_SSM_LLM_LOG_REASONING")
    if log_reasoning is not None:
        llm_raw["log_reasoning"] = log_reasoning
    log_output_char_limit = _env_int("CAUSAL_SSM_LLM_LOG_OUTPUT_CHAR_LIMIT")
    if log_output_char_limit is not None:
        llm_raw["log_output_char_limit"] = log_output_char_limit
    llm_config = LLMConfig(**llm_raw) if llm_raw else LLMConfig()

    # Parse pipeline section (optional)
    pipeline_raw = raw.get("pipeline", {})
    pipeline_config = (
        PipelineBehaviorConfig(**pipeline_raw) if pipeline_raw else PipelineBehaviorConfig()
    )

    # Parse stage0 section
    stage0_raw = raw.get("stage0_ingestion", {"model": "openrouter/anthropic/claude-sonnet-4"})
    stage0_config = Stage0Config(**stage0_raw)

    return PipelineConfig(
        stage0_ingestion=stage0_config,
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
