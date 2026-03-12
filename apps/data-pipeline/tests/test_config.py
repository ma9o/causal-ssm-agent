"""Tests for config.py: dataclass methods and load_config parsing."""

import textwrap
from unittest.mock import MagicMock, patch

from causal_ssm_agent.utils.config import (
    InferenceConfig,
    NUTSConfig,
    SVIConfig,
    get_secret,
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
    stage1_structure_proposal:
      model: claude-3
      sample_chunks: 5
      chunk_size: 800
    stage2_workers:
      model: claude-3
      chunk_size: 400
      max_concurrent_workers: 6
      submission_batch_size: 25
    stage4_prior_elicitation:
      model: claude-3
      worker_model: claude-3-haiku
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
      verbose_logging: true
      log_reasoning: true
      log_output_char_limit: 1234
    pipeline:
      max_prior_retries: 5
      override_gates: true
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
        assert cfg.stage2_workers.chunk_size == 300
        assert cfg.stage2_workers.max_concurrent_workers == 4
        assert cfg.stage2_workers.submission_batch_size == 50
        assert cfg.stage4_prior_elicitation.model == "gpt-4"
        # Defaults for optional sections
        assert cfg.inference.method == "auto"
        assert cfg.llm.max_tokens == 65536
        assert cfg.llm.verbose_logging is False
        assert cfg.llm.log_reasoning is False
        assert cfg.llm.log_output_char_limit == 8000
        assert cfg.pipeline.override_gates is False

        load_config.cache_clear()

    def test_load_full(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(FULL_CONFIG)

        load_config.cache_clear()

        import causal_ssm_agent.utils.config as config_mod

        monkeypatch.setattr(config_mod, "_find_config_path", lambda: config_file)

        cfg = load_config()
        assert cfg.stage2_workers.max_concurrent_workers == 6
        assert cfg.stage2_workers.submission_batch_size == 25
        assert cfg.stage4_prior_elicitation.worker_model == "claude-3-haiku"
        assert cfg.stage4_prior_elicitation.literature_search.enabled is False
        assert cfg.stage4_prior_elicitation.paraphrasing.enabled is True
        assert cfg.stage4_prior_elicitation.paraphrasing.n_paraphrases == 5
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
        assert cfg.llm.verbose_logging is True
        assert cfg.llm.log_reasoning is True
        assert cfg.llm.log_output_char_limit == 1234
        assert cfg.pipeline.max_prior_retries == 5
        assert cfg.pipeline.override_gates is True

        load_config.cache_clear()

    def test_llm_env_overrides(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(MINIMAL_CONFIG)

        load_config.cache_clear()

        import causal_ssm_agent.utils.config as config_mod

        monkeypatch.setattr(config_mod, "_find_config_path", lambda: config_file)
        monkeypatch.setenv("CAUSAL_SSM_LLM_VERBOSE_LOGGING", "1")
        monkeypatch.setenv("CAUSAL_SSM_LLM_LOG_REASONING", "true")
        monkeypatch.setenv("CAUSAL_SSM_LLM_LOG_OUTPUT_CHAR_LIMIT", "42")

        cfg = load_config()
        assert cfg.llm.verbose_logging is True
        assert cfg.llm.log_reasoning is True
        assert cfg.llm.log_output_char_limit == 42

        load_config.cache_clear()


# =============================================================================
# get_secret
# =============================================================================


class TestGetSecret:
    def test_falls_back_to_env_var(self, monkeypatch):
        """When Prefect block fails, get_secret falls back to os.getenv."""
        monkeypatch.setenv("TEST_SECRET_ABC", "from-env")

        def mock_import(name, *args, **kwargs):
            if name == "prefect.blocks.system":
                raise ImportError("No prefect")
            return __import__(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)
        result = get_secret("TEST_SECRET_ABC")
        assert result == "from-env"

    def test_returns_none_when_both_miss(self, monkeypatch):
        """When neither Prefect nor env var has the secret, returns None."""
        monkeypatch.delenv("DEFINITELY_NOT_SET_XYZ_789", raising=False)

        def mock_import(name, *args, **kwargs):
            if name == "prefect.blocks.system":
                raise ImportError("No prefect")
            return __import__(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)
        result = get_secret("DEFINITELY_NOT_SET_XYZ_789")
        assert result is None

    def test_prefect_block_name_uses_slug_format(self):
        """get_secret converts underscores to hyphens and lowercases for Prefect block name."""
        mock_secret = MagicMock()
        mock_secret.get.return_value = "val"

        mock_module = MagicMock()
        mock_module.Secret.load.return_value = mock_secret

        with patch.dict("sys.modules", {"prefect.blocks.system": mock_module}):
            import importlib

            import causal_ssm_agent.utils.config as config_mod

            importlib.reload(config_mod)

            result = config_mod.get_secret("MY_API_KEY")
            assert result == "val"
            mock_module.Secret.load.assert_called_once_with("my-api-key")
