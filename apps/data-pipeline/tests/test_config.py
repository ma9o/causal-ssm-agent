"""Tests for config.py: dataclass methods and load_config parsing."""

import asyncio
import textwrap

from causal_ssm_agent.utils.config import (
    AuxGibbsConfig,
    AuxGibbsLatentKernelConfig,
    AuxGibbsParameterKernelConfig,
    InferenceConfig,
    NUTSConfig,
    SVIConfig,
    get_secret,
    get_secret_async,
    load_config,
)

# =============================================================================
# InferenceConfig.to_sampler_config
# =============================================================================


class TestToSamplerConfig:
    def test_auto_defaults(self):
        cfg = InferenceConfig()
        result = cfg.to_sampler_config()
        assert result["method"] == "auto"
        assert result["num_warmup"] == 1000
        assert result["num_samples"] == 1000
        assert result["num_chains"] == 4
        assert result["seed"] == 0
        assert "num_steps" not in result
        assert "learning_rate" not in result
        assert "guide_type" not in result
        assert result["svi_config"]["num_steps"] == 5000
        assert result["nuts_config"]["max_tree_depth"] == 8
        assert result["smc_config"]["n_outer"] == 100

    def test_nuts_defaults(self):
        cfg = InferenceConfig(method="nuts")
        result = cfg.to_sampler_config()
        assert result["method"] == "nuts"
        assert result["target_accept_prob"] == 0.85
        assert result["max_tree_depth"] == 8
        assert "num_steps" not in result
        assert "learning_rate" not in result

    def test_method_override(self):
        cfg = InferenceConfig(method="svi")
        result = cfg.to_sampler_config(method_override="nuts")
        assert result["method"] == "nuts"
        assert result["target_accept_prob"] == 0.85
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

    def test_custom_nuts_settings(self):
        cfg = InferenceConfig(
            method="nuts",
            nuts=NUTSConfig(target_accept_prob=0.95, max_tree_depth=12),
        )
        result = cfg.to_sampler_config()
        assert result["target_accept_prob"] == 0.95
        assert result["max_tree_depth"] == 12

    def test_aux_gibbs_settings(self):
        cfg = InferenceConfig(
            method="aux_gibbs",
            aux_gibbs=AuxGibbsConfig(
                adaptation_rate=0.07,
                init_scale=0.02,
                retain_latent_paths=True,
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

    def test_unknown_method_returns_base_keys_only(self):
        cfg = InferenceConfig(method="hmc")
        result = cfg.to_sampler_config()
        assert result["method"] == "hmc"
        assert "num_warmup" in result
        assert "num_steps" not in result
        assert "target_accept_prob" not in result

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
      model: gpt-4
    stage1_structure_proposal:
      model: gpt-4
      sample_chunks: 3
      chunk_size: 500
    stage2_workers:
      model: gpt-4
      chunk_size: 300
    stage4_prior_elicitation:
      model: gpt-4
""")

FULL_CONFIG = textwrap.dedent("""\
    stage6_commentary:
      model: claude-3
    stage1_structure_proposal:
      model: claude-3
      sample_chunks: 5
      chunk_size: 800
      stage1a_max_tool_turns: 25
      stage1b_max_tool_turns: 35
    stage2_workers:
      model: claude-3
      chunk_size: 400
      max_concurrent_workers: 6
      max_tool_turns: 45
    stage4_prior_elicitation:
      model: claude-3
      max_tool_turns: 100
      literature_search:
        enabled: false
      paraphrasing:
        enabled: true
        n_paraphrases: 5
    inference:
      method: nuts
      num_warmup: 500
      num_samples: 2000
      num_chains: 2
      seed: 123
      svi:
        num_steps: 3000
      nuts:
        target_accept_prob: 0.9
        max_tree_depth: 10
    llm:
      max_tokens: 4096
      timeout: 120
      reasoning_effort: low
""")


class TestLoadConfig:
    def test_load_minimal(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(MINIMAL_CONFIG)

        load_config.cache_clear()

        import causal_ssm_agent.utils.config as config_mod

        monkeypatch.setattr(config_mod, "_find_config_path", lambda: config_file)

        cfg = load_config()
        assert cfg.stage1_structure_proposal.model == "gpt-4"
        assert cfg.stage1_structure_proposal.sample_chunks == 3
        assert cfg.stage1_structure_proposal.stage1a_max_tool_turns == 40
        assert cfg.stage1_structure_proposal.stage1b_max_tool_turns == 40
        assert cfg.stage2_workers.chunk_size == 300
        assert cfg.stage2_workers.max_concurrent_workers == 4
        assert cfg.stage2_workers.max_tool_turns == 40
        assert cfg.stage4_prior_elicitation.model == "gpt-4"
        assert cfg.stage4_prior_elicitation.max_tool_turns == 40
        assert cfg.stage6_commentary.model == "gpt-4"
        # Defaults for optional sections
        assert cfg.inference.method == "auto"
        assert cfg.llm.max_tokens == 65536

        load_config.cache_clear()

    def test_load_full(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(FULL_CONFIG)

        load_config.cache_clear()

        import causal_ssm_agent.utils.config as config_mod

        monkeypatch.setattr(config_mod, "_find_config_path", lambda: config_file)

        cfg = load_config()
        assert cfg.stage1_structure_proposal.stage1a_max_tool_turns == 25
        assert cfg.stage1_structure_proposal.stage1b_max_tool_turns == 35
        assert cfg.stage2_workers.max_concurrent_workers == 6
        assert cfg.stage2_workers.max_tool_turns == 45
        assert cfg.stage4_prior_elicitation.max_tool_turns == 100
        assert cfg.stage4_prior_elicitation.literature_search.enabled is False
        assert cfg.stage4_prior_elicitation.paraphrasing.enabled is True
        assert cfg.stage4_prior_elicitation.paraphrasing.n_paraphrases == 5
        assert cfg.stage6_commentary.model == "claude-3"
        assert cfg.inference.method == "nuts"
        assert cfg.inference.num_warmup == 500
        assert cfg.inference.num_samples == 2000
        assert cfg.inference.num_chains == 2
        assert cfg.inference.seed == 123
        assert cfg.inference.svi.num_steps == 3000
        assert cfg.inference.nuts.target_accept_prob == 0.9
        assert cfg.inference.nuts.max_tree_depth == 10
        assert cfg.llm.max_tokens == 4096
        assert cfg.llm.reasoning_effort == "low"

        load_config.cache_clear()

    def test_sampler_config_roundtrip(self, tmp_path, monkeypatch):
        """Load config and use to_sampler_config."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(FULL_CONFIG)

        load_config.cache_clear()

        import causal_ssm_agent.utils.config as config_mod

        monkeypatch.setattr(config_mod, "_find_config_path", lambda: config_file)

        cfg = load_config()
        sampler = cfg.inference.to_sampler_config()
        assert sampler["method"] == "nuts"
        assert sampler["num_warmup"] == 500
        assert sampler["target_accept_prob"] == 0.9

        load_config.cache_clear()


# =============================================================================
# get_secret
# =============================================================================


class TestGetSecret:
    def test_reads_env_var(self, monkeypatch):
        """get_secret reads from environment variables."""
        monkeypatch.setenv("TEST_SECRET_ABC", "from-env")
        assert get_secret("TEST_SECRET_ABC") == "from-env"

    def test_returns_none_when_missing(self, monkeypatch):
        """get_secret returns None when the env var is not set."""
        monkeypatch.delenv("DEFINITELY_NOT_SET_XYZ_789", raising=False)
        assert get_secret("DEFINITELY_NOT_SET_XYZ_789") is None

    def test_async_reads_env_var(self, monkeypatch):
        """get_secret_async reads from environment variables."""
        monkeypatch.setenv("TEST_SECRET_ABC", "from-env")
        assert asyncio.run(get_secret_async("TEST_SECRET_ABC")) == "from-env"
