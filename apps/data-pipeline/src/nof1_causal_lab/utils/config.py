"""Configuration loader for the N-of-1 Causal Lab pipeline."""

from __future__ import annotations

import dataclasses
import math
import os
import shutil
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv

# Centralized .env loading — all modules that need env vars import from config.py
# (or from modules that import config.py), so this runs once at import time.
load_dotenv(Path(__file__).parent.parent.parent.parent.parent.parent / ".env")


# ---------------------------------------------------------------------------
# LLM backend defaults (global)
# ---------------------------------------------------------------------------

HARNESS_VALUES = ("none", "claude-code", "codex")
EMBEDDED_REASONING_EFFORT_VALUES = ("none", "minimal", "low", "medium", "high", "xhigh")
HARNESS_EFFORT_VALUES = ("low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True)
class EmbeddedLLMDefaults:
    """Defaults for ``harness: none`` (OpenRouter) stages."""

    max_tokens: int = 65536
    timeout: int = 900
    reasoning_effort: str = "xhigh"


@dataclass(frozen=True)
class ClaudeCodeDefaults:
    """Defaults for ``harness: claude-code`` stages."""

    bin: str = "claude"
    effort: str = "high"
    max_turns: int = 40
    max_budget_usd: float | None = None
    fallback_model: str | None = None


@dataclass(frozen=True)
class CodexDefaults:
    """Defaults for ``harness: codex`` stages."""

    bin: str = "codex"
    reasoning_effort: str = "high"


@dataclass(frozen=True)
class LLMDefaults:
    """Global LLM backend defaults (one section per backend)."""

    embedded: EmbeddedLLMDefaults = field(default_factory=EmbeddedLLMDefaults)
    claude_code: ClaudeCodeDefaults = field(default_factory=ClaudeCodeDefaults)
    codex: CodexDefaults = field(default_factory=CodexDefaults)


@dataclass(frozen=True)
class StageLLMConfig:
    """Per-stage LLM selection and optional overrides.

    ``harness`` discriminates the backend. ``model`` is always required.
    The remaining fields are optional and override the corresponding
    :class:`LLMDefaults` section when set. A given field is valid only
    for a subset of harness values; :func:`validate_config` rejects
    incompatible combinations.
    """

    harness: str
    model: str
    # embedded overrides
    max_tokens: int | None = None
    timeout: int | None = None
    # embedded + codex share reasoning_effort (different scales)
    reasoning_effort: str | None = None
    # claude-code overrides
    effort: str | None = None
    max_turns: int | None = None
    max_budget_usd: float | None = None
    fallback_model: str | None = None
    # shared
    bin: str | None = None


# ---------------------------------------------------------------------------
# Stage configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage0Config:
    """Stage 0: Agentic Data Ingestion."""

    llm: StageLLMConfig
    max_tool_turns: int = 40


@dataclass(frozen=True)
class Stage1Config:
    """Stage 1: Structure Proposal (Orchestrator)."""

    llm: StageLLMConfig
    sample_chunks: int = 10
    chunk_size: int = 100
    stage1a_max_tool_turns: int = 40
    stage1b_max_tool_turns: int = 40


@dataclass(frozen=True)
class Stage2Config:
    """Stage 2: Support-Window Extraction (Workers).

    ``max_rpm`` only applies when ``llm.harness == 'none'``. Stage 2 must
    stay on the embedded backend because fanning out thousands of workers
    through a harness CLI is economically and latency-wise untenable.
    """

    llm: StageLLMConfig
    windows_per_chunk: int = 1
    max_concurrent_workers: int = 4
    max_events_per_window: int = 300
    max_rpm: int = 450
    worker_timeout: int = 120
    chunk_size: int = 50
    max_tool_turns: int = 40
    max_free_windows: int = 100


@dataclass(frozen=True)
class LiteratureSearchConfig:
    """Literature search configuration for grounding priors."""

    enabled: bool = True


@dataclass(frozen=True)
class ParaphrasingConfig:
    """AutoElicit-style paraphrased prompting configuration."""

    enabled: bool = False
    n_paraphrases: int = 10
    gmm_model: str | None = None
    """Override model name for inner paraphrase calls; inherits the
    stage's ``llm.harness``. ``None`` means use the stage's main model."""


@dataclass(frozen=True)
class Stage4Config:
    """Stage 4: Model Specification & Prior Elicitation."""

    llm: StageLLMConfig
    max_tool_turns: int = 40
    literature_search: LiteratureSearchConfig = field(default_factory=LiteratureSearchConfig)
    paraphrasing: ParaphrasingConfig = field(default_factory=ParaphrasingConfig)
    state_machine_enabled: bool = True
    """When ``True`` (default), Stage 4 runs the frontier-reduced state machine,
    prompting the LLM one block at a time. When ``False``, Stage 4 runs a single
    "megaprompt" agent session that exposes the same submit tools all at once
    and lets the model submit decisions and priors in any order; the same
    validation checks (schema, compile, prior-predictive) apply."""
    megaprompt_max_outer_turns: int = 8
    """Maximum outer agent turns allowed when running in megaprompt mode. Each
    outer turn gives the model one opportunity to call tools; the loop stops
    earlier once the accepted state passes full validation."""


@dataclass(frozen=True)
class Stage6Config:
    """Stage 6: Narrative commentary over intervention results and fit diagnostics."""

    llm: StageLLMConfig


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MAPConfig:
    """Internal IEKS/Laplace settings used by MCMC initializers."""

    n_ieks_iters: int = 6


@dataclass(frozen=True)
class MarginalParticleGibbsConfig:
    """Marginalized Particle Gibbs inference settings."""

    n_particles: int = 64
    n_parameter_particles: int = 2
    latent_smoother: Literal["dsmc"] = "dsmc"
    latent_delta: float = 0.2
    parameter_proposal: Literal["random_walk", "pseudo_langevin"] = "pseudo_langevin"
    amala_delta_init: float = 1e-2
    amala_delta_min: float = 1e-5
    amala_delta_max: float = 1e1
    amala_target_accept: float = 0.75
    amala_adaptation_window: int = 100
    amala_adaptation_tolerance: float = 0.05
    amala_adaptation_rho: float = 0.5
    amala_adaptation_rho_min: float = 1e-3
    amala_adaptation_gamma: float = -0.5
    amala_kappa: float = 0.75
    amala_grad_clip: float = math.inf
    dsmc_leaf_proposal: Literal["amala_exact", "paid_mix"] = "amala_exact"
    # Coordinate-block proposals: number of latent coordinates proposed per sweep
    # (None = all). Blocks of 2-4 sidestep the joint-coherence weight degeneracy of
    # full-state proposals at higher latent dimension.
    latent_block_coords: int | None = None
    # paid_mix leaf mixture: z-anchored (amala_exact core) + fixed IEKS-pilot
    # component + wide tail (weight = 1 - z - pilot). Pilot variances are
    # pilot_var_scale x the IEKS paths' per-coordinate spread; the wide tail is
    # wide_mult x the same spread.
    paid_mix_z_weight: float = 0.85
    paid_mix_pilot_weight: float = 0.10
    paid_mix_pilot_var_scale: float = 0.25
    paid_mix_wide_mult: float = 4.0
    # Joint (latent coordinate, loading column) sign-flip MH move composed after the
    # smoother sweep — the escape route between factor-sign mirror basins that
    # alternating conditionals cannot cross. Requires unconstrained (identity
    # transform) free loadings.
    latent_sign_flip_moves: bool = False
    diagnostic_metrics_all: bool = False
    diagnostic_metrics: tuple[str, ...] = ()
    param_step_size: float = 0.02
    param_step_size_min: float = 1e-6
    param_step_size_max: float = 1e3
    param_target_accept: float = 0.35
    adaptation_rate: float = 0.05
    init_method: Literal["random", "pathfinder"] = "pathfinder"
    latent_init_method: Literal["predictive"] = "predictive"
    pathfinder_num_elbo_samples: int = 20
    pathfinder_maxiter: int = 20
    n_pathfinder_starts: int = 8
    pathfinder_parallel_workers: int | None = None
    pathfinder_init_scale: float | None = 0.1
    auto_preconditioner_method: Literal["map", "none", "pathfinder"] = "pathfinder"
    auto_preconditioner_maxiter: int = 200
    init_scale: float = 0.05
    retain_latent_paths: bool = True
    compute_latent_posterior_summary: bool = True


@dataclass(frozen=True)
class InferenceConfig:
    """Inference configuration (method + sampler settings)."""

    method: Literal["marginal_particle_gibbs"] = "marginal_particle_gibbs"
    num_warmup: int = 4000
    num_samples: int = 1000
    num_chains: int = 4
    seed: int = 0
    compute_loo_diagnostics: bool = True
    map: MAPConfig = field(default_factory=MAPConfig)
    marginal_particle_gibbs: MarginalParticleGibbsConfig = field(
        default_factory=MarginalParticleGibbsConfig
    )

    def to_sampler_config(self, method_override: str | None = None) -> dict:
        """Build a flat sampler config dict for SSM inference."""
        method = method_override or self.method
        if method != "marginal_particle_gibbs":
            raise ValueError(
                f"Unsupported inference method {method!r}; expected 'marginal_particle_gibbs'."
            )
        config: dict = {
            "method": method,
            "num_warmup": self.num_warmup,
            "num_samples": self.num_samples,
            "num_chains": self.num_chains,
            "seed": self.seed,
        }
        config.update(
            {
                "n_ieks_iters": self.map.n_ieks_iters,
                **dataclasses.asdict(self.marginal_particle_gibbs),
            }
        )
        return config


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineBehaviorConfig:
    """Pipeline-level behavioral settings."""


@dataclass(frozen=True)
class PipelineConfig:
    """Full pipeline configuration."""

    stage0_ingestion: Stage0Config
    stage1_structure_proposal: Stage1Config
    stage2_workers: Stage2Config
    stage4_prior_elicitation: Stage4Config
    stage6_commentary: Stage6Config
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    llm: LLMDefaults = field(default_factory=LLMDefaults)
    pipeline: PipelineBehaviorConfig = field(default_factory=PipelineBehaviorConfig)


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------


def get_secret(name: str) -> str | None:
    """Get a secret from environment variables."""
    return os.getenv(name)


async def get_secret_async(name: str) -> str | None:
    """Async variant of ``get_secret``."""
    return os.getenv(name)


# ---------------------------------------------------------------------------
# YAML parsing
# ---------------------------------------------------------------------------


def _find_config_path() -> Path:
    """Find config.yaml by walking up from this file to the project root."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        config_path = parent / "config.yaml"
        if config_path.exists():
            return config_path
    raise FileNotFoundError("config.yaml not found in any parent directory")


def _parse_stage_llm(raw: dict, stage_name: str) -> StageLLMConfig:
    """Parse a per-stage llm: block into a StageLLMConfig."""
    if not isinstance(raw, dict):
        raise ValueError(f"{stage_name}.llm must be a mapping")
    if "harness" not in raw:
        raise ValueError(f"{stage_name}.llm.harness is required")
    if "model" not in raw:
        raise ValueError(f"{stage_name}.llm.model is required")
    return StageLLMConfig(**raw)


def _parse_llm_defaults(raw: dict) -> LLMDefaults:
    """Parse the global llm: section into LLMDefaults."""
    embedded_raw = raw.get("embedded", {}) or {}
    claude_code_raw = raw.get("claude_code", {}) or {}
    codex_raw = raw.get("codex", {}) or {}
    return LLMDefaults(
        embedded=EmbeddedLLMDefaults(**embedded_raw) if embedded_raw else EmbeddedLLMDefaults(),
        claude_code=ClaudeCodeDefaults(**claude_code_raw)
        if claude_code_raw
        else ClaudeCodeDefaults(),
        codex=CodexDefaults(**codex_raw) if codex_raw else CodexDefaults(),
    )


def _parse_inference(raw: dict) -> InferenceConfig:
    """Parse the inference: section into InferenceConfig."""
    inference_raw = dict(raw)
    map_raw = inference_raw.pop("map", {}) or {}
    marginal_particle_gibbs_raw = inference_raw.pop("marginal_particle_gibbs", {}) or {}
    return InferenceConfig(
        **inference_raw,
        map=MAPConfig(**map_raw) if map_raw else MAPConfig(),
        marginal_particle_gibbs=(
            MarginalParticleGibbsConfig(**marginal_particle_gibbs_raw)
            if marginal_particle_gibbs_raw
            else MarginalParticleGibbsConfig()
        ),
    )


@lru_cache(maxsize=1)
def load_config() -> PipelineConfig:
    """Load, parse, and validate the pipeline configuration."""
    config_path = _find_config_path()
    with config_path.open() as f:
        raw = yaml.safe_load(f) or {}

    llm_defaults = _parse_llm_defaults(raw.get("llm", {}) or {})
    inference_config = _parse_inference(raw.get("inference", {}) or {})

    stage0_raw = raw.get("stage0_ingestion", {}) or {}
    stage0_config = Stage0Config(
        llm=_parse_stage_llm(stage0_raw["llm"], "stage0_ingestion"),
        max_tool_turns=stage0_raw.get("max_tool_turns", 40),
    )

    stage1_raw = raw.get("stage1_structure_proposal", {}) or {}
    stage1_llm = _parse_stage_llm(stage1_raw["llm"], "stage1_structure_proposal")
    stage1_config = Stage1Config(
        llm=stage1_llm,
        sample_chunks=stage1_raw.get("sample_chunks", 10),
        chunk_size=stage1_raw.get("chunk_size", 100),
        stage1a_max_tool_turns=stage1_raw.get("stage1a_max_tool_turns", 40),
        stage1b_max_tool_turns=stage1_raw.get("stage1b_max_tool_turns", 40),
    )

    stage2_raw = dict(raw.get("stage2_workers", {}) or {})
    stage2_llm = _parse_stage_llm(stage2_raw.pop("llm"), "stage2_workers")
    stage2_config = Stage2Config(llm=stage2_llm, **stage2_raw)

    stage4_raw = dict(raw.get("stage4_prior_elicitation", {}) or {})
    stage4_llm = _parse_stage_llm(stage4_raw.pop("llm"), "stage4_prior_elicitation")
    lit_search_raw = stage4_raw.pop("literature_search", {}) or {}
    paraphrasing_raw = stage4_raw.pop("paraphrasing", {}) or {}
    stage4_config = Stage4Config(
        llm=stage4_llm,
        max_tool_turns=stage4_raw.get("max_tool_turns", 40),
        literature_search=LiteratureSearchConfig(**lit_search_raw)
        if lit_search_raw
        else LiteratureSearchConfig(),
        paraphrasing=ParaphrasingConfig(**paraphrasing_raw)
        if paraphrasing_raw
        else ParaphrasingConfig(),
        state_machine_enabled=stage4_raw.get("state_machine_enabled", True),
        megaprompt_max_outer_turns=stage4_raw.get("megaprompt_max_outer_turns", 8),
    )

    stage6_raw = raw.get("stage6_commentary", {}) or {}
    stage6_config = Stage6Config(llm=_parse_stage_llm(stage6_raw["llm"], "stage6_commentary"))

    pipeline_raw = raw.get("pipeline", {}) or {}
    pipeline_config = (
        PipelineBehaviorConfig(**pipeline_raw) if pipeline_raw else PipelineBehaviorConfig()
    )

    config = PipelineConfig(
        stage0_ingestion=stage0_config,
        stage1_structure_proposal=stage1_config,
        stage2_workers=stage2_config,
        stage4_prior_elicitation=stage4_config,
        stage6_commentary=stage6_config,
        inference=inference_config,
        llm=llm_defaults,
        pipeline=pipeline_config,
    )

    schema_errors = validate_config(config)
    if schema_errors:
        raise ValueError(
            "config.yaml failed validation:\n" + "\n".join(f"  - {e}" for e in schema_errors)
        )
    return config


def get_config() -> PipelineConfig:
    """Get the pipeline configuration."""
    return load_config()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _iter_stage_llms(config: PipelineConfig) -> list[tuple[str, StageLLMConfig]]:
    return [
        ("stage0_ingestion", config.stage0_ingestion.llm),
        ("stage1_structure_proposal", config.stage1_structure_proposal.llm),
        ("stage2_workers", config.stage2_workers.llm),
        ("stage4_prior_elicitation", config.stage4_prior_elicitation.llm),
        ("stage6_commentary", config.stage6_commentary.llm),
    ]


def validate_config(config: PipelineConfig) -> list[str]:
    """Validate the config's schema and cross-field constraints.

    Returns a list of error strings (empty on success). Each error is
    prefixed with the config path (e.g. ``stage2_workers.llm.harness``).
    """
    errors: list[str] = []

    for name, llm in _iter_stage_llms(config):
        path = f"{name}.llm"
        if llm.harness not in HARNESS_VALUES:
            errors.append(f"{path}.harness: {llm.harness!r} not in {list(HARNESS_VALUES)}")
            continue

        # Stage 2 fan-out constraint
        if name == "stage2_workers" and llm.harness != "none":
            errors.append(
                f"{path}.harness: must be 'none' for Stage 2 workers "
                "(harness cold-start × thousands of workers is untenable); "
                f"got {llm.harness!r}"
            )

        # Harness-specific field compatibility
        if llm.harness == "none":
            if llm.effort is not None:
                errors.append(
                    f"{path}.effort: only valid for harness=claude-code; "
                    "use reasoning_effort for harness=none"
                )
            if llm.max_turns is not None:
                errors.append(f"{path}.max_turns: only valid for harness=claude-code")
            if llm.max_budget_usd is not None:
                errors.append(f"{path}.max_budget_usd: only valid for harness=claude-code")
            if llm.fallback_model is not None:
                errors.append(f"{path}.fallback_model: only valid for harness=claude-code")
            if llm.bin is not None:
                errors.append(f"{path}.bin: only valid for harness=claude-code or codex")
            if (
                llm.reasoning_effort is not None
                and llm.reasoning_effort not in EMBEDDED_REASONING_EFFORT_VALUES
            ):
                errors.append(
                    f"{path}.reasoning_effort: {llm.reasoning_effort!r} not in "
                    f"{list(EMBEDDED_REASONING_EFFORT_VALUES)}"
                )
            if not llm.model.startswith("openrouter/"):
                errors.append(
                    f"{path}.model: {llm.model!r} should start with 'openrouter/' for harness=none"
                )

        elif llm.harness == "claude-code":
            if llm.reasoning_effort is not None:
                errors.append(f"{path}.reasoning_effort: use 'effort' for harness=claude-code")
            if llm.max_tokens is not None:
                errors.append(f"{path}.max_tokens: not configurable for harness=claude-code")
            if llm.timeout is not None:
                errors.append(f"{path}.timeout: not configurable for harness=claude-code")
            if llm.effort is not None and llm.effort not in HARNESS_EFFORT_VALUES:
                errors.append(f"{path}.effort: {llm.effort!r} not in {list(HARNESS_EFFORT_VALUES)}")

        elif llm.harness == "codex":
            if llm.effort is not None:
                errors.append(f"{path}.effort: only valid for harness=claude-code")
            if llm.max_turns is not None:
                errors.append(f"{path}.max_turns: only valid for harness=claude-code")
            if llm.max_budget_usd is not None:
                errors.append(f"{path}.max_budget_usd: only valid for harness=claude-code")
            if llm.fallback_model is not None:
                errors.append(f"{path}.fallback_model: only valid for harness=claude-code")
            if llm.max_tokens is not None:
                errors.append(f"{path}.max_tokens: not configurable for harness=codex")
            # ``timeout`` is honoured as a per-turn ceiling when opening the
            # codex session (see :func:`open_codex_harness_session`), so the
            # field is meaningful for this harness and must not be rejected.
            if (
                llm.reasoning_effort is not None
                and llm.reasoning_effort not in HARNESS_EFFORT_VALUES
            ):
                errors.append(
                    f"{path}.reasoning_effort: {llm.reasoning_effort!r} not in "
                    f"{list(HARNESS_EFFORT_VALUES)}"
                )

    # Global LLM defaults: enum checks
    embedded = config.llm.embedded
    if embedded.reasoning_effort not in EMBEDDED_REASONING_EFFORT_VALUES:
        errors.append(
            f"llm.embedded.reasoning_effort: {embedded.reasoning_effort!r} not in "
            f"{list(EMBEDDED_REASONING_EFFORT_VALUES)}"
        )
    if config.llm.claude_code.effort not in HARNESS_EFFORT_VALUES:
        errors.append(
            f"llm.claude_code.effort: {config.llm.claude_code.effort!r} not in "
            f"{list(HARNESS_EFFORT_VALUES)}"
        )
    if config.llm.codex.reasoning_effort not in HARNESS_EFFORT_VALUES:
        errors.append(
            f"llm.codex.reasoning_effort: {config.llm.codex.reasoning_effort!r} not in "
            f"{list(HARNESS_EFFORT_VALUES)}"
        )

    return errors


def _check_embedded_prereqs() -> list[str]:
    errors: list[str] = []
    if not os.getenv("OPENROUTER_API_KEY"):
        errors.append("OPENROUTER_API_KEY is not set (required for harness=none)")
    return errors


def _check_claude_code_prereqs(config: PipelineConfig) -> list[str]:
    import subprocess

    errors: list[str] = []
    bin_name = config.llm.claude_code.bin
    bin_path = shutil.which(bin_name)
    if bin_path is None:
        errors.append(
            f"claude binary {bin_name!r} not found on PATH (required for harness=claude-code)"
        )
        return errors
    try:
        status = subprocess.run(
            [bin_path, "auth", "status", "--text"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        errors.append(f"`claude auth status` failed: {exc}")
    else:
        if status.returncode != 0:
            errors.append(
                f"claude is not logged in — run `claude auth login` (exit={status.returncode})"
            )
    return errors


def _check_codex_prereqs(config: PipelineConfig) -> list[str]:
    errors: list[str] = []
    bin_name = config.llm.codex.bin
    if shutil.which(bin_name) is None:
        errors.append(f"codex binary {bin_name!r} not found on PATH (required for harness=codex)")
    if not Path("~/.codex/auth.json").expanduser().exists():
        errors.append(
            "codex is not logged in — run `codex login` (expected ~/.codex/auth.json to exist)"
        )
    return errors


def validate_runtime_prereqs(config: PipelineConfig) -> list[str]:
    """Check that binaries and credentials required by the configured harnesses exist.

    ``harness: none`` still requires ``OPENROUTER_API_KEY``. ``claude-code``
    and ``codex`` authenticate via their respective CLIs' subscription
    logins (Claude Max/Pro and ChatGPT Plus/Pro/Team/Enterprise) — we
    check that the CLI is logged in via ``claude auth status`` and
    ``~/.codex/auth.json``. Safe to call repeatedly; the pipeline also
    runs per-harness subsets of this via :func:`ensure_harness_prereqs`
    the first time each backend opens a session.
    """
    errors: list[str] = []
    harnesses_used = {llm.harness for _name, llm in _iter_stage_llms(config)}
    if "none" in harnesses_used:
        errors.extend(_check_embedded_prereqs())
    if "claude-code" in harnesses_used:
        errors.extend(_check_claude_code_prereqs(config))
    if "codex" in harnesses_used:
        errors.extend(_check_codex_prereqs(config))
    return errors


_verified_harnesses: set[str] = set()


def ensure_harness_prereqs(harness: str) -> None:
    """Run the prereq check for ``harness``, once per process, or raise.

    Harness openers call this on first invocation so a pipeline with a
    logged-out CLI or a missing ``OPENROUTER_API_KEY`` fails within
    milliseconds of starting the relevant stage, instead of crashing
    deep inside the subprocess or the OpenAI SDK.
    """
    if harness in _verified_harnesses:
        return
    config = get_config()
    if harness == "none":
        errors = _check_embedded_prereqs()
    elif harness == "claude-code":
        errors = _check_claude_code_prereqs(config)
    elif harness == "codex":
        errors = _check_codex_prereqs(config)
    else:
        raise ValueError(f"Unknown harness: {harness!r}")
    if errors:
        raise RuntimeError(
            f"Harness {harness!r} prereqs not satisfied:\n" + "\n".join(f"  - {e}" for e in errors)
        )
    _verified_harnesses.add(harness)


def _reset_verified_harnesses_for_testing() -> None:
    """Clear the per-process cache; used by tests that patch env/subprocess."""
    _verified_harnesses.clear()
