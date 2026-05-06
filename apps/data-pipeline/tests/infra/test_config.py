"""Tests for config.py: dataclass methods and load_config parsing."""

import textwrap

import pytest

from causal_ssm_agent.utils.config import (
    AuxGibbsConfig,
    AuxGibbsLatentKernelConfig,
    AuxGibbsParameterKernelConfig,
    ClaudeCodeDefaults,
    CodexDefaults,
    EmbeddedLLMDefaults,
    InferenceConfig,
    LLMDefaults,
    MAPConfig,
    PipelineBehaviorConfig,
    PipelineConfig,
    Stage0Config,
    Stage1Config,
    Stage2Config,
    Stage4Config,
    Stage6Config,
    StageLLMConfig,
    SVIConfig,
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
    def test_aux_gibbs_defaults(self):
        cfg = InferenceConfig()
        result = cfg.to_sampler_config()
        assert result["method"] == "aux_gibbs"
        assert result["num_warmup"] == 1000
        assert result["num_samples"] == 1000
        assert result["num_chains"] == 4
        assert result["seed"] == 0
        assert "num_steps" not in result
        assert "learning_rate" not in result
        assert "guide_type" not in result
        assert result["latent_kernel"] == "kalman"
        assert result["parameter_kernel"] == "mala"
        assert result["retain_latent_paths"] is False
        assert result["compute_latent_posterior_summary"] is True

    def test_particle_mgrad_defaults(self):
        cfg = InferenceConfig(method="particle_mgrad")
        result = cfg.to_sampler_config()
        assert result["method"] == "particle_mgrad"
        assert result["n_ieks_iters"] == 5
        assert "num_steps" not in result
        assert "learning_rate" not in result

    def test_method_override(self):
        cfg = InferenceConfig(method="svi")
        result = cfg.to_sampler_config(method_override="map")
        assert result["method"] == "map"
        assert result["n_ieks_iters"] == 5
        assert "num_steps" not in result

    def test_custom_svi_settings(self):
        cfg = InferenceConfig(
            method="svi",
            svi=SVIConfig(num_steps=10000, learning_rate=0.001, guide_type="diagonal"),
        )
        result = cfg.to_sampler_config()
        assert result["num_steps"] == 10000
        assert result["learning_rate"] == 0.001
        assert result["guide_type"] == "diagonal"

    def test_custom_map_settings(self):
        cfg = InferenceConfig(
            method="map",
            map=MAPConfig(n_ieks_iters=10),
        )
        result = cfg.to_sampler_config()
        assert result["n_ieks_iters"] == 10

    def test_aux_gibbs_settings(self):
        cfg = InferenceConfig(
            method="aux_gibbs",
            aux_gibbs=AuxGibbsConfig(
                adaptation_rate=0.07,
                init_scale=0.02,
                retain_latent_paths=True,
                compute_latent_posterior_summary=False,
                latent_kernel=AuxGibbsLatentKernelConfig(
                    kernel="kalman",
                    proposal_family="eq10_11",
                    delta=0.15,
                    target_accept=0.45,
                ),
                parameter_kernel=AuxGibbsParameterKernelConfig(
                    kernel="mala",
                    step_size=0.03,
                    target_accept=0.61,
                ),
            ),
        )
        result = cfg.to_sampler_config()
        assert result["method"] == "aux_gibbs"
        assert result["latent_kernel"] == "kalman"
        assert result["latent_proposal_family"] == "eq10_11"
        assert result["latent_delta"] == 0.15
        assert result["latent_target_accept"] == 0.45
        assert result["parameter_kernel"] == "mala"
        assert result["param_step_size"] == 0.03
        assert result["param_target_accept"] == 0.61
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
      method: map
      num_warmup: 500
      num_samples: 2000
      num_chains: 2
      seed: 123
      svi:
        num_steps: 3000
      map:
        n_ieks_iters: 10
""")


class TestLoadConfig:
    def test_load_minimal(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(MINIMAL_CONFIG)

        load_config.cache_clear()

        import causal_ssm_agent.utils.config as config_mod

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
        assert cfg.inference.method == "aux_gibbs"
        assert cfg.llm.embedded.max_tokens == 65536

        load_config.cache_clear()

    def test_load_full(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(FULL_CONFIG)

        load_config.cache_clear()

        import causal_ssm_agent.utils.config as config_mod

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
        assert cfg.inference.method == "map"
        assert cfg.inference.num_warmup == 500
        assert cfg.inference.num_samples == 2000
        assert cfg.inference.num_chains == 2
        assert cfg.inference.seed == 123
        assert cfg.inference.svi.num_steps == 3000
        assert cfg.inference.map.n_ieks_iters == 10
        assert cfg.llm.embedded.max_tokens == 4096
        assert cfg.llm.embedded.reasoning_effort == "low"
        assert cfg.llm.claude_code.effort == "medium"
        assert cfg.llm.codex.reasoning_effort == "medium"

        load_config.cache_clear()

    def test_sampler_config_roundtrip(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(FULL_CONFIG)

        load_config.cache_clear()

        import causal_ssm_agent.utils.config as config_mod

        monkeypatch.setattr(config_mod, "_find_config_path", lambda: config_file)

        cfg = load_config()
        sampler = cfg.inference.to_sampler_config()
        assert sampler["method"] == "map"
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

        import causal_ssm_agent.utils.config as config_mod

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
        from causal_ssm_agent.utils.config import _reset_verified_harnesses_for_testing

        _reset_verified_harnesses_for_testing()

    def test_missing_openrouter_key_raises_for_embedded(self, monkeypatch):
        from causal_ssm_agent.utils.config import ensure_harness_prereqs

        self._reset()
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            ensure_harness_prereqs("none")

    def test_passes_when_openrouter_key_set(self, monkeypatch):
        from causal_ssm_agent.utils.config import ensure_harness_prereqs

        self._reset()
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        ensure_harness_prereqs("none")  # no raise

    def test_caches_successful_check(self, monkeypatch):
        """Once verified, removing the env var doesn't re-trigger the check."""
        from causal_ssm_agent.utils.config import ensure_harness_prereqs

        self._reset()
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        ensure_harness_prereqs("none")

        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        # Still cached — does not raise.
        ensure_harness_prereqs("none")

    def test_reset_clears_cache(self, monkeypatch):
        from causal_ssm_agent.utils.config import ensure_harness_prereqs

        self._reset()
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        ensure_harness_prereqs("none")

        self._reset()
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(RuntimeError):
            ensure_harness_prereqs("none")

    def test_unknown_harness_raises_value_error(self):
        from causal_ssm_agent.utils.config import ensure_harness_prereqs

        self._reset()
        with pytest.raises(ValueError, match="Unknown harness"):
            ensure_harness_prereqs("bedrock")
