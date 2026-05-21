"""Tests for config.py: dataclass methods and load_config parsing."""

import textwrap

import pytest

from nof1_causal_lab.utils.config import (
    AuxKalmanMCMCConfig,
    AuxKalmanMCMCLatentKernelConfig,
    AuxKalmanMCMCParameterKernelConfig,
    ClaudeCodeDefaults,
    CodexDefaults,
    EmbeddedLLMDefaults,
    InferenceConfig,
    LLMDefaults,
    MAPConfig,
    PipelineBehaviorConfig,
    PipelineConfig,
    PITParticleMGradConfig,
    Stage0Config,
    Stage1Config,
    Stage2Config,
    Stage4Config,
    Stage6Config,
    StageLLMConfig,
    get_secret,
    get_secret_async,
    load_config,
    validate_config,
)
from tests.helpers import run_async

# =============================================================================
# InferenceConfig.to_sampler_config
# =============================================================================


class TestToSamplerConfig:
    def test_pit_particle_mgrad_defaults(self):
        cfg = InferenceConfig()
        result = cfg.to_sampler_config()
        assert result["method"] == "pit_particle_mgrad"
        assert result["num_warmup"] == 4000
        assert result["num_samples"] == 1000
        assert result["num_chains"] == 4
        assert result["seed"] == 0
        assert result["n_ieks_iters"] == 6
        assert result["latent_delta"] == 0.2
        assert result["latent_delta_min"] is None
        assert result["latent_delta_max"] is None
        assert result["latent_target_accept"] == 0.5
        assert result["n_particles"] == 64
        assert result["adaptation_scheme"] == "simple"
        assert result["init_method"] == "pathfinder"
        assert result["latent_init_method"] == "ieks"
        assert result["latent_init_num_particles"] == 64
        assert result["latent_init_guidance"] == "bffg"
        assert result["pathfinder_num_elbo_samples"] == 20
        assert result["pathfinder_maxiter"] == 20
        assert result["n_pathfinder_starts"] == 4
        assert result["pathfinder_init_scale"] == 0.1
        assert result["parameter_kernel"] == "mala"
        assert result["param_step_size"] == 0.05
        assert result["param_target_accept"] == 0.57
        assert result["param_max_num_doublings"] == 10
        assert result["adaptation_rate"] == 0.05
        assert result["init_scale"] == 0.05
        assert result["retain_latent_paths"] is True
        assert result["compute_latent_posterior_summary"] is True
        assert "num_steps" not in result
        assert "learning_rate" not in result
        assert "guide_type" not in result

    def test_aux_kalman_mcmc_explicit(self):
        cfg = InferenceConfig(method="aux_kalman_mcmc")
        result = cfg.to_sampler_config()
        assert result["method"] == "aux_kalman_mcmc"
        assert result["num_warmup"] == 4000
        assert result["num_samples"] == 1000
        assert result["num_chains"] == 4
        assert result["latent_kernel"] == "kalman"
        assert result["parameter_kernel"] == "mala"
        assert result["param_max_num_doublings"] == 10
        assert result["retain_latent_paths"] is True
        assert result["compute_latent_posterior_summary"] is True

    def test_method_override(self):
        cfg = InferenceConfig(method="pit_particle_mgrad")
        result = cfg.to_sampler_config(method_override="aux_kalman_mcmc")
        assert result["method"] == "aux_kalman_mcmc"
        assert result["latent_kernel"] == "kalman"

    def test_custom_map_settings(self):
        cfg = InferenceConfig(
            method="pit_particle_mgrad",
            map=MAPConfig(n_ieks_iters=10),
        )
        result = cfg.to_sampler_config()
        assert result["method"] == "pit_particle_mgrad"
        assert result["n_ieks_iters"] == 10

    def test_aux_kalman_mcmc_settings(self):
        cfg = InferenceConfig(
            method="aux_kalman_mcmc",
            aux_kalman_mcmc=AuxKalmanMCMCConfig(
                adaptation_rate=0.07,
                init_scale=0.02,
                retain_latent_paths=True,
                compute_latent_posterior_summary=False,
                latent_kernel=AuxKalmanMCMCLatentKernelConfig(
                    kernel="kalman",
                    proposal_family="eq10_11",
                    delta=0.15,
                    target_accept=0.45,
                ),
                parameter_kernel=AuxKalmanMCMCParameterKernelConfig(
                    kernel="mala",
                    step_size=0.03,
                    target_accept=0.61,
                    max_num_doublings=8,
                ),
            ),
        )
        result = cfg.to_sampler_config()
        assert result["method"] == "aux_kalman_mcmc"
        assert result["latent_kernel"] == "kalman"
        assert result["latent_proposal_family"] == "eq10_11"
        assert result["latent_delta"] == 0.15
        assert result["latent_target_accept"] == 0.45
        assert result["parameter_kernel"] == "mala"
        assert result["param_step_size"] == 0.03
        assert result["param_target_accept"] == 0.61
        assert result["param_max_num_doublings"] == 8
        assert result["adaptation_rate"] == 0.07
        assert result["init_scale"] == 0.02
        assert result["retain_latent_paths"] is True
        assert result["compute_latent_posterior_summary"] is False

    def test_unknown_method_raises(self):
        cfg = InferenceConfig(method="hmc")
        with pytest.raises(ValueError, match="Unsupported inference method"):
            cfg.to_sampler_config()

    def test_custom_chains_and_seed(self):
        cfg = InferenceConfig(num_chains=8, seed=42)
        result = cfg.to_sampler_config()
        assert result["num_chains"] == 8
        assert result["seed"] == 42

    def test_pit_particle_mgrad_parameter_kernel_settings(self):
        cfg = InferenceConfig(
            aux_kalman_mcmc=AuxKalmanMCMCConfig(
                parameter_kernel=AuxKalmanMCMCParameterKernelConfig(
                    kernel="nuts",
                    step_size=0.04,
                    target_accept=0.8,
                    max_num_doublings=7,
                ),
            ),
        )
        result = cfg.to_sampler_config()
        assert result["method"] == "pit_particle_mgrad"
        assert result["parameter_kernel"] == "nuts"
        assert result["param_step_size"] == 0.04
        assert result["param_target_accept"] == 0.8
        assert result["param_max_num_doublings"] == 7

    def test_pit_particle_mgrad_settings(self):
        cfg = InferenceConfig(
            pit_particle_mgrad=PITParticleMGradConfig(
                latent_delta=0.12,
                latent_delta_min=1e-5,
                latent_delta_max=0.3,
                latent_target_accept=0.47,
                n_particles=32,
                adaptation_scheme="simple",
                init_method="random",
                latent_init_method="particle_smoother",
                latent_init_num_particles=48,
                latent_init_guidance="bootstrap",
                pathfinder_num_elbo_samples=12,
                pathfinder_maxiter=13,
                n_pathfinder_starts=3,
                pathfinder_init_scale=None,
            ),
        )
        result = cfg.to_sampler_config()
        assert result["method"] == "pit_particle_mgrad"
        assert result["latent_delta"] == 0.12
        assert result["latent_delta_min"] == 1e-5
        assert result["latent_delta_max"] == 0.3
        assert result["latent_target_accept"] == 0.47
        assert result["n_particles"] == 32
        assert result["adaptation_scheme"] == "simple"
        assert result["init_method"] == "random"
        assert result["latent_init_method"] == "particle_smoother"
        assert result["latent_init_num_particles"] == 48
        assert result["latent_init_guidance"] == "bootstrap"
        assert result["pathfinder_num_elbo_samples"] == 12
        assert result["pathfinder_maxiter"] == 13
        assert result["n_pathfinder_starts"] == 3
        assert result["pathfinder_init_scale"] is None


# =============================================================================
# load_config (with temp config file)
# =============================================================================


MINIMAL_CONFIG = textwrap.dedent("""\
    stage6_commentary:
      llm:
        harness: none
        model: openrouter/gpt-4
    stage0_ingestion:
      llm:
        harness: none
        model: openrouter/gpt-4
    stage1_structure_proposal:
      sample_chunks: 3
      chunk_size: 500
      llm:
        harness: none
        model: openrouter/gpt-4
    stage2_workers:
      chunk_size: 300
      llm:
        harness: none
        model: openrouter/gpt-4
    stage4_prior_elicitation:
      llm:
        harness: none
        model: openrouter/gpt-4
""")

FULL_CONFIG = textwrap.dedent("""\
    llm:
      embedded:
        max_tokens: 4096
        timeout: 120
        reasoning_effort: low
      claude_code:
        effort: medium
      codex:
        reasoning_effort: medium

    stage0_ingestion:
      max_tool_turns: 30
      llm:
        harness: none
        model: openrouter/claude-3

    stage6_commentary:
      llm:
        harness: none
        model: openrouter/claude-3

    stage1_structure_proposal:
      sample_chunks: 5
      chunk_size: 800
      stage1a_max_tool_turns: 25
      stage1b_max_tool_turns: 35
      llm:
        harness: none
        model: openrouter/claude-3

    stage2_workers:
      chunk_size: 400
      max_concurrent_workers: 6
      max_tool_turns: 45
      llm:
        harness: none
        model: openrouter/claude-3

    stage4_prior_elicitation:
      max_tool_turns: 100
      literature_search:
        enabled: false
      paraphrasing:
        enabled: true
        n_paraphrases: 5
      llm:
        harness: none
        model: openrouter/claude-3

    inference:
      method: pit_particle_mgrad
      num_warmup: 500
      num_samples: 2000
      num_chains: 2
      seed: 123
      compute_loo_diagnostics: false
      map:
        n_ieks_iters: 10
      pit_particle_mgrad:
        latent_delta: 0.11
        latent_delta_min: 0.00001
        latent_delta_max: 0.2
        latent_target_accept: 0.49
        n_particles: 48
        adaptation_scheme: simple
        init_method: random
        latent_init_method: particle_smoother
        latent_init_num_particles: 40
        latent_init_guidance: bootstrap
        pathfinder_num_elbo_samples: 11
        pathfinder_maxiter: 12
        n_pathfinder_starts: 5
        pathfinder_init_scale:
      aux_kalman_mcmc:
        parameter_kernel:
          kernel: nuts
          target_accept: 0.8
          max_num_doublings: 9
""")


class TestLoadConfig:
    def test_load_minimal(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(MINIMAL_CONFIG)

        load_config.cache_clear()

        import nof1_causal_lab.utils.config as config_mod

        monkeypatch.setattr(config_mod, "_find_config_path", lambda: config_file)

        cfg = load_config()
        assert cfg.stage1_structure_proposal.llm.model == "openrouter/gpt-4"
        assert cfg.stage1_structure_proposal.llm.harness == "none"
        assert cfg.stage1_structure_proposal.sample_chunks == 3
        assert cfg.stage1_structure_proposal.stage1a_max_tool_turns == 40
        assert cfg.stage1_structure_proposal.stage1b_max_tool_turns == 40
        assert cfg.stage2_workers.chunk_size == 300
        assert cfg.stage2_workers.max_concurrent_workers == 4
        assert cfg.stage2_workers.max_tool_turns == 40
        assert cfg.stage4_prior_elicitation.llm.model == "openrouter/gpt-4"
        assert cfg.stage4_prior_elicitation.max_tool_turns == 40
        assert cfg.stage6_commentary.llm.model == "openrouter/gpt-4"
        # Defaults for optional sections
        assert cfg.inference.method == "pit_particle_mgrad"
        assert cfg.llm.embedded.max_tokens == 65536

        load_config.cache_clear()

    def test_load_full(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(FULL_CONFIG)

        load_config.cache_clear()

        import nof1_causal_lab.utils.config as config_mod

        monkeypatch.setattr(config_mod, "_find_config_path", lambda: config_file)

        cfg = load_config()
        assert cfg.stage0_ingestion.max_tool_turns == 30
        assert cfg.stage1_structure_proposal.stage1a_max_tool_turns == 25
        assert cfg.stage1_structure_proposal.stage1b_max_tool_turns == 35
        assert cfg.stage2_workers.max_concurrent_workers == 6
        assert cfg.stage2_workers.max_tool_turns == 45
        assert cfg.stage4_prior_elicitation.max_tool_turns == 100
        assert cfg.stage4_prior_elicitation.literature_search.enabled is False
        assert cfg.stage4_prior_elicitation.paraphrasing.enabled is True
        assert cfg.stage4_prior_elicitation.paraphrasing.n_paraphrases == 5
        assert cfg.stage6_commentary.llm.model == "openrouter/claude-3"
        assert cfg.inference.method == "pit_particle_mgrad"
        assert cfg.inference.num_warmup == 500
        assert cfg.inference.num_samples == 2000
        assert cfg.inference.num_chains == 2
        assert cfg.inference.seed == 123
        assert cfg.inference.compute_loo_diagnostics is False
        assert cfg.inference.map.n_ieks_iters == 10
        assert cfg.inference.pit_particle_mgrad.latent_delta == 0.11
        assert cfg.inference.pit_particle_mgrad.latent_delta_min == 1e-5
        assert cfg.inference.pit_particle_mgrad.latent_delta_max == 0.2
        assert cfg.inference.pit_particle_mgrad.latent_target_accept == 0.49
        assert cfg.inference.pit_particle_mgrad.n_particles == 48
        assert cfg.inference.pit_particle_mgrad.adaptation_scheme == "simple"
        assert cfg.inference.pit_particle_mgrad.init_method == "random"
        assert cfg.inference.pit_particle_mgrad.latent_init_method == "particle_smoother"
        assert cfg.inference.pit_particle_mgrad.latent_init_num_particles == 40
        assert cfg.inference.pit_particle_mgrad.latent_init_guidance == "bootstrap"
        assert cfg.inference.pit_particle_mgrad.pathfinder_num_elbo_samples == 11
        assert cfg.inference.pit_particle_mgrad.pathfinder_maxiter == 12
        assert cfg.inference.pit_particle_mgrad.n_pathfinder_starts == 5
        assert cfg.inference.pit_particle_mgrad.pathfinder_init_scale is None
        assert cfg.inference.aux_kalman_mcmc.parameter_kernel.kernel == "nuts"
        assert cfg.inference.aux_kalman_mcmc.parameter_kernel.target_accept == 0.8
        assert cfg.inference.aux_kalman_mcmc.parameter_kernel.max_num_doublings == 9
        assert cfg.llm.embedded.max_tokens == 4096
        assert cfg.llm.embedded.reasoning_effort == "low"
        assert cfg.llm.claude_code.effort == "medium"
        assert cfg.llm.codex.reasoning_effort == "medium"

        load_config.cache_clear()

    def test_sampler_config_roundtrip(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(FULL_CONFIG)

        load_config.cache_clear()

        import nof1_causal_lab.utils.config as config_mod

        monkeypatch.setattr(config_mod, "_find_config_path", lambda: config_file)

        cfg = load_config()
        sampler = cfg.inference.to_sampler_config()
        assert sampler["method"] == "pit_particle_mgrad"
        assert sampler["num_warmup"] == 500
        assert sampler["n_ieks_iters"] == 10

        load_config.cache_clear()


# =============================================================================
# validate_config
# =============================================================================


def _make_pipeline_config(**stage_llm_overrides) -> PipelineConfig:
    """Build a valid PipelineConfig with optional per-stage llm overrides."""
    defaults = {
        "stage0_ingestion": StageLLMConfig(harness="none", model="openrouter/x"),
        "stage1_structure_proposal": StageLLMConfig(harness="none", model="openrouter/x"),
        "stage2_workers": StageLLMConfig(harness="none", model="openrouter/x"),
        "stage4_prior_elicitation": StageLLMConfig(harness="none", model="openrouter/x"),
        "stage6_commentary": StageLLMConfig(harness="none", model="openrouter/x"),
    }
    defaults.update(stage_llm_overrides)
    return PipelineConfig(
        stage0_ingestion=Stage0Config(llm=defaults["stage0_ingestion"]),
        stage1_structure_proposal=Stage1Config(llm=defaults["stage1_structure_proposal"]),
        stage2_workers=Stage2Config(llm=defaults["stage2_workers"]),
        stage4_prior_elicitation=Stage4Config(llm=defaults["stage4_prior_elicitation"]),
        stage6_commentary=Stage6Config(llm=defaults["stage6_commentary"]),
        inference=InferenceConfig(),
        llm=LLMDefaults(
            embedded=EmbeddedLLMDefaults(),
            claude_code=ClaudeCodeDefaults(),
            codex=CodexDefaults(),
        ),
        pipeline=PipelineBehaviorConfig(),
    )


class TestValidateConfig:
    def test_happy_path_all_embedded(self):
        config = _make_pipeline_config()
        assert validate_config(config) == []

    def test_stage2_harness_claude_code_rejected(self):
        config = _make_pipeline_config(
            stage2_workers=StageLLMConfig(harness="claude-code", model="sonnet"),
        )
        errors = validate_config(config)
        assert any("stage2_workers.llm.harness" in e for e in errors)
        assert any("must be 'none'" in e for e in errors)

    def test_stage2_harness_codex_rejected(self):
        config = _make_pipeline_config(
            stage2_workers=StageLLMConfig(harness="codex", model="gpt-5.4"),
        )
        errors = validate_config(config)
        assert any("stage2_workers.llm.harness" in e for e in errors)
        assert any("must be 'none'" in e for e in errors)

    def test_unknown_harness_value_rejected(self):
        config = _make_pipeline_config(
            stage0_ingestion=StageLLMConfig(harness="anthropic", model="openrouter/x"),
        )
        errors = validate_config(config)
        assert any("stage0_ingestion.llm.harness" in e for e in errors)

    def test_embedded_model_must_be_openrouter_prefix(self):
        config = _make_pipeline_config(
            stage0_ingestion=StageLLMConfig(harness="none", model="gpt-5.4"),
        )
        errors = validate_config(config)
        assert any("openrouter/" in e for e in errors)

    def test_effort_only_valid_for_claude_code(self):
        config = _make_pipeline_config(
            stage0_ingestion=StageLLMConfig(harness="none", model="openrouter/x", effort="high"),
        )
        errors = validate_config(config)
        assert any(".effort" in e for e in errors)

    def test_claude_code_rejects_reasoning_effort(self):
        config = _make_pipeline_config(
            stage0_ingestion=StageLLMConfig(
                harness="claude-code", model="sonnet", reasoning_effort="high"
            ),
        )
        errors = validate_config(config)
        assert any(".reasoning_effort" in e and "claude-code" in e for e in errors)

    def test_claude_code_effort_enum(self):
        config = _make_pipeline_config(
            stage0_ingestion=StageLLMConfig(harness="claude-code", model="sonnet", effort="ultra"),
        )
        errors = validate_config(config)
        assert any(".effort" in e and "'ultra'" in e for e in errors)

    def test_codex_rejects_claude_only_fields(self):
        config = _make_pipeline_config(
            stage0_ingestion=StageLLMConfig(harness="codex", model="gpt-5.4", max_budget_usd=5.0),
        )
        errors = validate_config(config)
        assert any(".max_budget_usd" in e for e in errors)

    def test_load_config_raises_on_stage2_harness_violation(self, tmp_path, monkeypatch):
        bad_config = textwrap.dedent("""\
            stage6_commentary:
              llm:
                harness: none
                model: openrouter/gpt-4
            stage0_ingestion:
              llm:
                harness: none
                model: openrouter/gpt-4
            stage1_structure_proposal:
              sample_chunks: 3
              chunk_size: 500
              llm:
                harness: none
                model: openrouter/gpt-4
            stage2_workers:
              chunk_size: 300
              llm:
                harness: claude-code
                model: sonnet
            stage4_prior_elicitation:
              llm:
                harness: none
                model: openrouter/gpt-4
        """)
        config_file = tmp_path / "config.yaml"
        config_file.write_text(bad_config)

        load_config.cache_clear()

        import nof1_causal_lab.utils.config as config_mod

        monkeypatch.setattr(config_mod, "_find_config_path", lambda: config_file)

        with pytest.raises(ValueError, match=r"stage2_workers\.llm\.harness"):
            load_config()

        load_config.cache_clear()


# =============================================================================
# get_secret
# =============================================================================


class TestGetSecret:
    def test_reads_env_var(self, monkeypatch):
        monkeypatch.setenv("TEST_SECRET_ABC", "from-env")
        assert get_secret("TEST_SECRET_ABC") == "from-env"

    def test_returns_none_when_missing(self, monkeypatch):
        monkeypatch.delenv("DEFINITELY_NOT_SET_XYZ_789", raising=False)
        assert get_secret("DEFINITELY_NOT_SET_XYZ_789") is None

    def test_async_reads_env_var(self, monkeypatch):
        monkeypatch.setenv("TEST_SECRET_ABC", "from-env")
        assert run_async(get_secret_async("TEST_SECRET_ABC")) == "from-env"


# =============================================================================
# ensure_harness_prereqs
# =============================================================================


class TestEnsureHarnessPrereqs:
    def _reset(self):
        from nof1_causal_lab.utils.config import _reset_verified_harnesses_for_testing

        _reset_verified_harnesses_for_testing()

    def test_missing_openrouter_key_raises_for_embedded(self, monkeypatch):
        from nof1_causal_lab.utils.config import ensure_harness_prereqs

        self._reset()
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            ensure_harness_prereqs("none")

    def test_passes_when_openrouter_key_set(self, monkeypatch):
        from nof1_causal_lab.utils.config import ensure_harness_prereqs

        self._reset()
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        ensure_harness_prereqs("none")  # no raise

    def test_caches_successful_check(self, monkeypatch):
        """Once verified, removing the env var doesn't re-trigger the check."""
        from nof1_causal_lab.utils.config import ensure_harness_prereqs

        self._reset()
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        ensure_harness_prereqs("none")

        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        # Still cached — does not raise.
        ensure_harness_prereqs("none")

    def test_reset_clears_cache(self, monkeypatch):
        from nof1_causal_lab.utils.config import ensure_harness_prereqs

        self._reset()
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        ensure_harness_prereqs("none")

        self._reset()
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(RuntimeError):
            ensure_harness_prereqs("none")

    def test_unknown_harness_raises_value_error(self):
        from nof1_causal_lab.utils.config import ensure_harness_prereqs

        self._reset()
        with pytest.raises(ValueError, match="Unknown harness"):
            ensure_harness_prereqs("bedrock")
