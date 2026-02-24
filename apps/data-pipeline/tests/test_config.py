"""Tests for centralized pipeline configuration."""

from unittest.mock import MagicMock, patch

import pytest

from causal_ssm_agent.utils.config import (
    InferenceConfig,
    LLMConfig,
    NUTSConfig,
    PipelineBehaviorConfig,
    PipelineConfig,
    Stage1Config,
    Stage2Config,
    Stage4Config,
    SVIConfig,
    get_secret,
    load_config,
)


@pytest.fixture(autouse=True)
def _clear_config_cache():
    """Clear the lru_cache between tests."""
    load_config.cache_clear()
    yield
    load_config.cache_clear()


# -- Minimal YAML for tests (no inference/llm/pipeline sections) --

MINIMAL_YAML = {
    "stage1_structure_proposal": {"model": "test-model", "sample_chunks": 5, "chunk_size": 50},
    "stage2_workers": {"model": "test-worker", "chunk_size": 10},
    "stage4_prior_elicitation": {"model": "test-prior"},
}


FULL_YAML = {
    **MINIMAL_YAML,
    "inference": {
        "method": "nuts",
        "num_warmup": 500,
        "num_samples": 2000,
        "num_chains": 2,
        "seed": 42,
        "svi": {"num_steps": 3000, "learning_rate": 0.005, "guide_type": "diag"},
        "nuts": {"target_accept_prob": 0.9, "max_tree_depth": 10},
    },
    "llm": {
        "max_tokens": 4096,
        "timeout": 300,
        "reasoning_effort": "low",
    },
    "pipeline": {"max_prior_retries": 5},
}


def _mock_load(raw_yaml: dict):
    """Return a patched load_config that reads from a dict instead of disk."""

    def _loader():

        # Simulate what load_config does, but with our dict
        raw = raw_yaml
        stage4_raw = raw["stage4_prior_elicitation"]
        lit_search_raw = stage4_raw.get("literature_search", {})
        paraphrasing_raw = stage4_raw.get("paraphrasing", {})

        from causal_ssm_agent.utils.config import (
            LiteratureSearchConfig,
            ParaphrasingConfig,
        )

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

        inference_raw = dict(raw.get("inference", {}))
        svi_raw = inference_raw.pop("svi", {})
        nuts_raw = inference_raw.pop("nuts", {})
        inference_config = InferenceConfig(
            **inference_raw,
            svi=SVIConfig(**svi_raw) if svi_raw else SVIConfig(),
            nuts=NUTSConfig(**nuts_raw) if nuts_raw else NUTSConfig(),
        )

        llm_raw = raw.get("llm", {})
        llm_config = LLMConfig(**llm_raw) if llm_raw else LLMConfig()

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

    return _loader


# ── Config defaults ──────────────────────────────────────────────────────────


class TestLoadConfigDefaults:
    """Verify new sections have correct defaults when missing from YAML."""

    def test_load_config_defaults(self):
        cfg = _mock_load(MINIMAL_YAML)()

        # Inference defaults
        assert cfg.inference.method == "svi"
        assert cfg.inference.num_warmup == 1000
        assert cfg.inference.num_samples == 1000
        assert cfg.inference.num_chains == 4
        assert cfg.inference.seed == 0
        assert cfg.inference.svi.num_steps == 5000
        assert cfg.inference.svi.learning_rate == 0.01
        assert cfg.inference.nuts.target_accept_prob == 0.85

        # LLM defaults
        assert cfg.llm.max_tokens == 65536
        assert cfg.llm.timeout == 900
        assert cfg.llm.reasoning_effort == "high"

        # Pipeline defaults
        assert cfg.pipeline.max_prior_retries == 3


class TestLoadConfigWithInferenceSection:
    """Verify full parsing of nested inference config."""

    def test_load_config_with_inference_section(self):
        cfg = _mock_load(FULL_YAML)()

        assert cfg.inference.method == "nuts"
        assert cfg.inference.num_warmup == 500
        assert cfg.inference.num_samples == 2000
        assert cfg.inference.num_chains == 2
        assert cfg.inference.seed == 42

        assert cfg.inference.svi.num_steps == 3000
        assert cfg.inference.svi.learning_rate == 0.005
        assert cfg.inference.svi.guide_type == "diag"

        assert cfg.inference.nuts.target_accept_prob == 0.9
        assert cfg.inference.nuts.max_tree_depth == 10

        assert cfg.llm.max_tokens == 4096
        assert cfg.llm.timeout == 300
        assert cfg.llm.reasoning_effort == "low"

        assert cfg.pipeline.max_prior_retries == 5


# ── InferenceConfig.to_sampler_config ────────────────────────────────────────


class TestToSamplerConfig:
    def test_svi_default(self):
        cfg = InferenceConfig()  # method="svi" by default
        sc = cfg.to_sampler_config()

        assert sc["method"] == "svi"
        assert sc["num_warmup"] == 1000
        assert sc["num_samples"] == 1000
        assert sc["num_chains"] == 4
        assert sc["seed"] == 0
        assert sc["num_steps"] == 5000
        assert sc["learning_rate"] == 0.01
        assert sc["guide_type"] == "mvn"
        # NUTS keys should NOT be present
        assert "target_accept_prob" not in sc
        assert "max_tree_depth" not in sc

    def test_nuts_with_method_override(self):
        cfg = InferenceConfig()  # method="svi" by default
        sc = cfg.to_sampler_config(method_override="nuts")

        assert sc["method"] == "nuts"
        assert sc["target_accept_prob"] == 0.85
        assert sc["max_tree_depth"] == 8
        # SVI keys should NOT be present
        assert "num_steps" not in sc
        assert "learning_rate" not in sc

    def test_custom_values(self):
        cfg = InferenceConfig(
            method="nuts",
            num_warmup=200,
            seed=99,
            nuts=NUTSConfig(target_accept_prob=0.95, max_tree_depth=12),
        )
        sc = cfg.to_sampler_config()

        assert sc["method"] == "nuts"
        assert sc["num_warmup"] == 200
        assert sc["seed"] == 99
        assert sc["target_accept_prob"] == 0.95
        assert sc["max_tree_depth"] == 12


# ── get_secret ───────────────────────────────────────────────────────────────


class TestGetSecret:
    def test_env_fallback(self, monkeypatch):
        """When Prefect block fails, falls back to env var."""
        monkeypatch.setenv("MY_SECRET", "from-env")

        mock_secret_cls = MagicMock()
        mock_secret_cls.load.side_effect = Exception("not found")
        mock_module = MagicMock()
        mock_module.Secret = mock_secret_cls

        with patch.dict("sys.modules", {"prefect.blocks.system": mock_module}):
            result = get_secret("MY_SECRET")

        assert result == "from-env"

    def test_prefect_block(self):
        """Successful Prefect block load returns its value."""
        mock_block = MagicMock()
        mock_block.get.return_value = "from-prefect"

        mock_secret_cls = MagicMock()
        mock_secret_cls.load.return_value = mock_block

        # Mock the import of Secret inside get_secret
        mock_module = MagicMock()
        mock_module.Secret = mock_secret_cls

        with patch.dict("sys.modules", {"prefect.blocks.system": mock_module}):
            result = get_secret("MY_SECRET")

        assert result == "from-prefect"
        mock_secret_cls.load.assert_called_once_with("my-secret")

    def test_neither(self, monkeypatch):
        """When both fail, returns None."""
        monkeypatch.delenv("NONEXISTENT_SECRET", raising=False)

        result = get_secret("NONEXISTENT_SECRET")
        # Prefect block will fail (not configured in test), env var doesn't exist
        assert result is None


# ── Wiring tests ─────────────────────────────────────────────────────────────


class TestGetGenerateConfigReadsConfig:
    """Verify LLM config is wired through to get_generate_config."""

    def test_get_generate_config_reads_config(self):
        custom_llm = LLMConfig(max_tokens=1024, timeout=60, reasoning_effort="low")
        mock_cfg = MagicMock()
        mock_cfg.llm = custom_llm

        with patch("causal_ssm_agent.utils.config.get_config", return_value=mock_cfg):
            from causal_ssm_agent.utils.llm import get_generate_config

            gc = get_generate_config()

        assert gc.max_tokens == 1024
        assert gc.timeout == 60
        assert gc.reasoning_effort == "low"


class TestGetDefaultSamplerConfigReadsConfig:
    """Verify SSMModelBuilder reads from config."""

    def test_get_default_sampler_config_reads_config(self):
        custom_inference = InferenceConfig(
            method="nuts",
            num_warmup=200,
            num_samples=500,
            num_chains=1,
            seed=7,
            nuts=NUTSConfig(target_accept_prob=0.9, max_tree_depth=6),
        )
        mock_cfg = MagicMock()
        mock_cfg.inference = custom_inference

        with patch("causal_ssm_agent.utils.config.get_config", return_value=mock_cfg):
            from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

            sc = SSMModelBuilder.get_default_sampler_config()

        assert sc["method"] == "nuts"
        assert sc["num_warmup"] == 200
        assert sc["num_samples"] == 500
        assert sc["num_chains"] == 1
        assert sc["seed"] == 7
        assert sc["target_accept_prob"] == 0.9
        assert sc["max_tree_depth"] == 6
