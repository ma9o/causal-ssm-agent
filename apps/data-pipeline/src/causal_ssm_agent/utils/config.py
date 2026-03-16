"""Configuration loader for the causal agent pipeline."""

import dataclasses
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

from causal_ssm_agent.flows import get_prefect_logger

logger = get_prefect_logger(__name__)

# Centralized .env loading — all modules that need env vars import from config.py
# (or from modules that import config.py), so this runs once at import time.
load_dotenv(Path(__file__).parent.parent.parent.parent.parent.parent / ".env")


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
    """Stage 2: Tick-Based Extraction (Workers)."""

    model: str
    ticks_per_chunk: int = 1
    max_concurrent_workers: int = 4
    max_events_per_tick: int = 300
    max_rpm: int = 450
    worker_timeout: int = 120
    # Deprecated: kept for config.yaml backward compat, ignored at runtime
    submission_batch_size: int = 50
    chunk_size: int = 50


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
    svi: SVIConfig = SVIConfig()
    nuts: NUTSConfig = NUTSConfig()
    smc: SMCConfig = SMCConfig()

    def to_sampler_config(self, method_override: str | None = None) -> dict:
        """Build a flat sampler config dict for SSMModelBuilder.

        Uses ``dataclasses.asdict`` on the relevant sub-config so that new
        fields are automatically included without manual enumeration.

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
        smc = dataclasses.asdict(self.smc)
        smc = {k: v for k, v in smc.items() if v is not None}
        if method == "auto":
            config["svi_config"] = dataclasses.asdict(self.svi)
            config["nuts_config"] = dataclasses.asdict(self.nuts)
            config["smc_config"] = smc
        elif method == "svi":
            config.update(dataclasses.asdict(self.svi))
        elif method == "nuts":
            config.update(dataclasses.asdict(self.nuts))
        elif method in ("laplace_em", "tempered_smc", "structured_vi", "dpf"):
            if method != "laplace_em":
                smc.pop("n_ieks_iters", None)
            config.update(smc)
        return config


@dataclass(frozen=True)
class LLMConfig:
    """LLM generation settings shared across all model calls."""

    max_tokens: int = 65536
    timeout: int = 900
    reasoning_effort: str = "high"


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
