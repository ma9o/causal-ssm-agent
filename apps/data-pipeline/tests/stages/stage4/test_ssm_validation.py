"""Stage 4 assembly, prior predictive, and SSM compilation tests."""

from nof1_causal_lab.models.ssm import SSMSpec
from nof1_causal_lab.models.ssm.testing import (
    default_diffusion_block,
    default_input_effect_block,
    default_lambda_block,
    default_manifest_chol_block,
    default_manifest_means_block,
    default_static_state_sd_block,
    default_t0_chol_block,
    default_t0_means_block,
    dense_matrix_dynamics_spec,
    full_dense_matrix_dynamics_spec,
)
from tests.stages.stage4._support import (
    Any,
    GenerateConfig,
    PredictiveObservationMeanOverflow,
    PriorValidationResult,
    SimpleNamespace,
    _make_polars_data,
    _stage4_generate_config,
    _with_positive_indicator_polarity,
    compile_ssm_inputs_from_model_spec,
    compile_ssm_priors,
    get_failed_parameters,
    np,
    patch,
    pl,
    pytest,
    validate_prior_predictive,
)


def _prior_params(prior_registry, site_name: str):
    return prior_registry.priors_by_site[site_name].params


def _prior_params_for_parameter(prior_registry, index_maps, parameter: str) -> tuple[dict, int]:
    binding = index_maps.by_parameter[parameter]
    return prior_registry.priors_by_site[binding.site_name].params, binding.flat_index


def _positive_prior_mean_for_parameter(prior_registry, index_maps, parameter: str) -> float:
    params, flat_index = _prior_params_for_parameter(prior_registry, index_maps, parameter)
    if "concentration" in params and "rate" in params:
        concentration = np.asarray(params["concentration"], dtype=float).reshape(-1)
        rate = np.asarray(params["rate"], dtype=float).reshape(-1)
        return float(concentration[flat_index] / rate[flat_index])
    if "value" in params:
        return float(np.asarray(params["value"], dtype=float).reshape(-1)[flat_index])
    if "sigma" in params:
        sigma = np.asarray(params["sigma"], dtype=float).reshape(-1)[flat_index]
        return float(sigma * np.sqrt(2.0 / np.pi))
    raise AssertionError(f"Unsupported positive prior payload for {parameter}: {params}")


def _positive_prior_sd_for_parameter(prior_registry, index_maps, parameter: str) -> float:
    params, flat_index = _prior_params_for_parameter(prior_registry, index_maps, parameter)
    if "concentration" in params and "rate" in params:
        concentration = np.asarray(params["concentration"], dtype=float).reshape(-1)
        rate = np.asarray(params["rate"], dtype=float).reshape(-1)
        return float(np.sqrt(concentration[flat_index]) / rate[flat_index])
    raise AssertionError(f"Unsupported positive prior payload for {parameter}: {params}")


def _real_prior_mean_for_parameter(prior_registry, index_maps, parameter: str) -> float:
    params, flat_index = _prior_params_for_parameter(prior_registry, index_maps, parameter)
    if "mu" in params:
        return float(np.asarray(params["mu"], dtype=float).reshape(-1)[flat_index])
    if "loc" in params:
        return float(np.asarray(params["loc"], dtype=float).reshape(-1)[flat_index])
    if "value" in params:
        return float(np.asarray(params["value"], dtype=float).reshape(-1)[flat_index])
    raise AssertionError(f"Unsupported real prior payload for {parameter}: {params}")


def _default_ssm_spec(
    *,
    n_latent: int,
    n_manifest: int,
    latent_names: list[str] | None = None,
    edge_support=None,
) -> SSMSpec:
    """Build a SSMSpec with all default blocks plus optional dynamics support + names."""
    if edge_support is not None:
        dynamics_spec = dense_matrix_dynamics_spec(
            n_latent=n_latent,
            decay_support=np.ones(n_latent, dtype=bool),
            edge_support=np.asarray(edge_support, dtype=bool),
            coupling_template=np.zeros((n_latent, n_latent)),
            intercept_support=np.zeros(n_latent, dtype=bool),
            cint_template=np.zeros(n_latent),
        )
    else:
        dynamics_spec = full_dense_matrix_dynamics_spec(n_latent)
    return SSMSpec(
        n_latent=n_latent,
        n_manifest=n_manifest,
        dynamics_spec=dynamics_spec,
        diffusion_block=default_diffusion_block(n_latent),
        lambda_block=default_lambda_block(n_manifest, n_latent),
        manifest_means_block=default_manifest_means_block(n_manifest),
        manifest_chol_block=default_manifest_chol_block(n_manifest),
        t0_means_block=default_t0_means_block(n_latent),
        t0_chol_block=default_t0_chol_block(n_latent),
        input_effect_block=default_input_effect_block(n_latent),
        static_state_sd_block=default_static_state_sd_block(),
        latent_names=latent_names,
    )


def _require_text(value: str | None) -> str:
    """Assert an optional diagnostic field is present before string matching."""
    assert value is not None
    return value


def test_stage4_generate_config_sets_stage4_timeout(monkeypatch):
    monkeypatch.setattr(
        "nof1_causal_lab.flows.stages.stage4.flow.get_generate_config",
        lambda: GenerateConfig(
            max_tokens=65536,
            timeout=321,
            reasoning_effort="high",
            max_tool_output=1234,
        ),
    )

    config = _stage4_generate_config()

    assert config.max_tokens is None
    assert config.timeout == 180
    assert config.reasoning_effort == "high"
    assert config.max_tool_output is None


# --- SSM model construction tests ---


class TestSSMModelConstruction:
    """Test SSM model building."""

    def test_build_ssm_model_creates_model(self, simple_model_spec, simple_priors, simple_data):
        """Runtime construction creates an SSMModel with correct dimensions."""
        from nof1_causal_lab.models.ssm.runtime import build_ssm_model

        model = build_ssm_model(
            pl.from_pandas(simple_data),
            model_spec=simple_model_spec,
            priors=simple_priors,
        )
        assert model.spec.n_manifest == 1  # mood_score only
        assert model.spec.n_latent >= 1
        # Lambda should map latent to manifest
        assert model.spec.lambda_block.template.shape == (
            model.spec.n_manifest,
            model.spec.n_latent,
        )


# --- Prior Predictive Validation Tests ---


class TestPriorPredictiveValidation:
    """Test prior predictive validation end-to-end."""

    def test_valid_priors_pass(self, simple_model_spec, simple_priors):
        """Simple spec + priors + polars data -> is_valid=True with all checks passing."""
        data_for_model = _make_polars_data()
        is_valid, results, _samples = validate_prior_predictive(
            simple_model_spec, simple_priors, data_for_model, n_samples=10
        )
        assert is_valid is True
        assert len(results) > 0
        assert all(r.is_valid for r in results), (
            f"Expected all checks to pass but got failures: "
            f"{[(r.parameter, r.issue) for r in results if not r.is_valid]}"
        )

    def test_model_build_failure(self):
        """Broken spec -> is_valid=False, error in results."""
        broken_spec = {
            "likelihoods": [
                {
                    "variable": "nonexistent_col",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "test",
                }
            ],
            "parameters": [
                {
                    "name": "rho_x",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "AR coeff",
                }
            ],
        }
        broken_priors = {
            "rho_x": {
                "parameter": "rho_x",
                "distribution": "Beta",
                "params": {"alpha": 2.0, "beta": 2.0},
                "sources": [],
                "reasoning": "test",
            }
        }
        # Patch runtime preparation to make this a focused model-build failure.
        with patch(
            "nof1_causal_lab.models.ssm.runtime.prepare_wide_model_runtime",
            side_effect=ValueError("deliberate test failure"),
        ):
            is_valid, results, _samples = validate_prior_predictive(
                broken_spec, broken_priors, None, n_samples=10
            )
            assert is_valid is False
            assert any("model_build" in r.parameter for r in results)
            assert any("deliberate test failure" in (r.issue or "") for r in results)

    def test_no_data_uses_support_compatible_dummy_build_data(self):
        """Support-restricted likelihoods should still validate without raw data."""
        model_spec = {
            "likelihoods": [
                {
                    "variable": "screen_gap",
                    "distribution": "gamma",
                    "link": "log",
                    "reasoning": "Positive continuous gap",
                }
            ],
            "parameters": [
                {
                    "name": "rho_screen_gap",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "AR coefficient",
                }
            ],
        }
        priors = {
            "rho_screen_gap": {
                "parameter": "rho_screen_gap",
                "distribution": "Beta",
                "params": {"alpha": 2.0, "beta": 2.0},
                "sources": [],
                "reasoning": "Weakly informative",
            }
        }

        with patch(
            "nof1_causal_lab.models.ssm.runtime.sample_prior_predictive",
            return_value={"vf_0_decay": np.ones((2, 1))},
        ):
            is_valid, results, _samples = validate_prior_predictive(
                model_spec, priors, None, n_samples=2
            )

        assert is_valid is True
        assert not any(r.parameter == "model_build" for r in results)

    def test_materialize_stage4_result_persists_validation_warnings(
        self,
        simple_model_spec,
        simple_priors,
    ):
        """Final Stage 4 artifacts should carry non-fatal validation warnings."""
        from nof1_causal_lab.flows.stages.stage4.assembly import (
            AssemblyValidation,
            materialize_stage4_result,
        )

        validation = AssemblyValidation(
            normalized_model_spec=simple_model_spec,
            compile_ok=True,
            diagnostics=[
                PriorValidationResult(
                    parameter="beta_stress_sleep",
                    is_valid=True,
                    code="interval_reference_missing",
                    origin="compile",
                    severity="warning",
                    issue="Weekly evidence is being interpreted on the daily model interval.",
                    suggested_adjustment=(
                        "Set `reference_interval_days` if that weekly interval is intended."
                    ),
                )
            ],
            compiled_ssm={"compiled_prior_semantics": {}, "parameter_bindings": []},
            pp_checked=True,
            pp_valid=True,
            pp_raw_samples={},
        )

        with (
            patch(
                "nof1_causal_lab.flows.stages.stage4.assembly.compile_model_artifact",
                return_value={
                    "model_built": True,
                    "model_type": "test",
                    "version": "0",
                    "compiled_ssm": {"compiled_prior_semantics": {}, "parameter_bindings": []},
                },
            ),
            patch(
                "nof1_causal_lab.models.ssm.compile.artifact.resolve_prior_proposals",
                return_value=[],
            ),
        ):
            result = materialize_stage4_result(
                model_spec=simple_model_spec,
                authored_priors=simple_priors,
                data_for_model=_make_polars_data(),
                indicator_audits=None,
                causal_spec=None,
                validation=validation,
            )

        assert result["validation_warnings"] == [
            "Weekly evidence is being interpreted on the daily model interval."
        ]

    def test_validate_assembly_reuses_compiled_artifact_for_prior_checks(
        self,
        simple_model_spec,
        simple_priors,
    ):
        """Stage 4 should compile once per validation attempt and pass that artifact through."""
        from nof1_causal_lab.flows.stages.stage4.assembly import validate_assembly

        compiled_artifact = {"schema_version": 1}
        seen_compiled: list[dict[str, Any] | None] = []

        def stub_validate_prior_predictive(*args, compiled_ssm=None, **kwargs):
            seen_compiled.append(compiled_ssm)
            return True, [], {}

        with (
            patch(
                "nof1_causal_lab.models.ssm.compile.artifact.compile_ssm_artifact",
                return_value=compiled_artifact,
            ) as compile_mock,
            patch(
                "nof1_causal_lab.models.prior_predictive.validate_prior_predictive",
                side_effect=stub_validate_prior_predictive,
            ),
        ):
            validation = validate_assembly(
                simple_model_spec,
                simple_priors,
                _make_polars_data(),
                None,
                None,
            )

        assert compile_mock.call_count == 1
        assert seen_compiled == [compiled_artifact]
        assert validation.compiled_ssm == compiled_artifact

    def test_validate_assembly_keeps_lagged_prior_mismatches_as_warnings(
        self,
        simple_model_spec,
        simple_priors,
    ):
        """Lagged DT/CT heuristics should surface as warnings, not compile errors."""
        from nof1_causal_lab.flows.stages.stage4.assembly import validate_assembly

        compiled_artifact = {
            "schema_version": 1,
            "spec": {},
            "compiled_prior_semantics": {},
        }

        with (
            patch(
                "nof1_causal_lab.models.ssm.compile.artifact.compile_ssm_artifact",
                return_value=compiled_artifact,
            ),
            patch(
                "nof1_causal_lab.flows.stages.stage4.assembly._collect_compile_diagnostics",
                return_value=[
                    PriorValidationResult(
                        parameter="beta_stress_sleep",
                        is_valid=True,
                        code="lagged_response_weak",
                        origin="compile",
                        severity="warning",
                        issue="Median one-lag response is much slower than the nominal lag.",
                        suggested_adjustment="Confirm that this slow response is intended.",
                    )
                ],
            ),
            patch(
                "nof1_causal_lab.models.prior_predictive.validate_prior_predictive",
                return_value=(True, [], {}),
            ) as pp_mock,
        ):
            validation = validate_assembly(
                simple_model_spec,
                simple_priors,
                _make_polars_data(),
                None,
                {"measurement": {"model_clock": "1d"}},
            )

        assert validation.compile_ok is True
        assert validation.pp_valid is True
        assert [
            warning.model_dump()
            for warning in validation.compile_diagnostics
            if warning.severity == "warning"
        ] == [
            PriorValidationResult(
                parameter="beta_stress_sleep",
                is_valid=True,
                code="lagged_response_weak",
                origin="compile",
                severity="warning",
                issue="Median one-lag response is much slower than the nominal lag.",
                suggested_adjustment="Confirm that this slow response is intended.",
            ).model_dump()
        ]
        pp_mock.assert_called_once()

    def test_validate_prior_predictive_skips_recompile_when_artifact_provided(
        self,
        simple_model_spec,
        simple_priors,
    ):
        """Explicit compiled_ssm should bypass compile_ssm_artifact entirely."""

        runtime = SimpleNamespace(
            model=object(),
            times=np.arange(4, dtype=float),
            observation_support=None,
            observations=np.zeros((4, 1), dtype=float),
            transition_inputs=None,
        )

        with (
            patch(
                "nof1_causal_lab.models.ssm.compile.artifact.compile_ssm_artifact",
                side_effect=AssertionError("compile should not be called"),
            ),
            patch(
                "nof1_causal_lab.models.ssm.runtime.prepare_model_runtime",
                return_value=runtime,
            ),
            patch(
                "nof1_causal_lab.models.ssm.runtime.sample_prior_predictive",
                return_value={
                    "vf_0_decay": np.ones((3, 1)),
                    "observations": np.random.default_rng(0).normal(
                        loc=5.0,
                        scale=1.5,
                        size=(3, 4, 1),
                    ),
                },
            ),
        ):
            is_valid, results, _samples = validate_prior_predictive(
                simple_model_spec,
                simple_priors,
                _make_polars_data(),
                n_samples=3,
                compiled_ssm={"schema_version": 1},
            )

        assert is_valid is True
        assert results

    def test_validate_prior_predictive_reports_log_link_mean_overflow(
        self,
        simple_model_spec,
        simple_priors,
    ):
        """Prior predictive should surface predictive-mean overflow as a typed diagnostic."""

        runtime = SimpleNamespace(
            model=object(),
            times=np.arange(4, dtype=float),
            observation_support=None,
            observations=np.zeros((4, 1), dtype=float),
            transition_inputs=None,
        )

        with (
            patch(
                "nof1_causal_lab.models.ssm.runtime.prepare_model_runtime",
                return_value=runtime,
            ),
            patch(
                "nof1_causal_lab.models.ssm.runtime.sample_prior_predictive",
                side_effect=PredictiveObservationMeanOverflow(
                    bad_manifest_names=("monthly_eveningness_activity_timing",),
                    manifest_indices=(0,),
                    failing_draw_indices=(0, 2),
                    first_bad_time_index=73,
                    max_linear_predictor=111.52,
                    overflow_threshold=88.72,
                ),
            ),
        ):
            is_valid, results, _samples = validate_prior_predictive(
                simple_model_spec,
                simple_priors,
                _make_polars_data(),
                n_samples=3,
                compiled_ssm={"schema_version": 1, "compile_diagnostics": []},
            )

        assert is_valid is False
        assert [result.code for result in results if not result.is_valid] == [
            "prior_predictive_observation_mean_overflow"
        ]
        result = next(result for result in results if not result.is_valid)
        assert result.failure_stage == "observation_mean"
        assert result.bad_manifest_names == ["monthly_eveningness_activity_timing"]
        assert result.failing_draw_indices == [0, 2]
        assert result.first_bad_time_index == 73

    def test_resolve_prior_proposals_reads_compiled_semantics_per_state(self):
        """Implicit initial-state priors should come from compiled semantics."""
        from nof1_causal_lab.models.ssm.compile.artifact import resolve_prior_proposals

        compiled_ssm = {
            "spec": {"latent_names": ["stress", "sleep"]},
            "compiled_prior_semantics": {
                "schema_version": 5,
                "site_registry": [
                    {
                        "name": "t0_means_free",
                        "shape": [2],
                        "support": "real",
                        "assembly_group": "t0",
                        "site_kind": "t0_means",
                        "transform_kind": "identity",
                        "deterministic_name": "t0_means",
                        "fixed_spec_field": "t0_means",
                        "priors_field": "t0_means",
                        "runtime_prior_key": "t0_means_free",
                        "is_runtime_prior_controlled": True,
                    },
                    {
                        "name": "t0_var_diag_free",
                        "shape": [2],
                        "support": "positive",
                        "assembly_group": "t0",
                        "site_kind": "t0_var_diag",
                        "transform_kind": "exp",
                        "deterministic_name": "t0_cov",
                        "fixed_spec_field": "t0_var",
                        "priors_field": "t0_var_diag",
                        "runtime_prior_key": "t0_var_diag_free",
                        "is_runtime_prior_controlled": True,
                    },
                ],
                "prior_state": {
                    "t0_means_free": {"family": 0, "loc": [0.0, 1.0], "scale": [2.0, 3.0]},
                    "t0_var_diag_free": {
                        "family": 0,
                        "scale": [4.0, 5.0],
                        "concentration": [1.0, 1.0],
                        "rate": [1.0, 1.0],
                    },
                },
            },
            "parameter_bindings": [],
        }

        assert resolve_prior_proposals(compiled_ssm, authored_priors={}) == [
            {
                "parameter": "t0_mean_stress",
                "distribution": "Normal",
                "params": {"mu": 0.0, "sigma": 2.0},
                "sources": [],
                "reasoning": "Default weakly informative prior for the initial state mean of stress.",
                "reference_interval_days": None,
                "density_points": None,
            },
            {
                "parameter": "t0_mean_sleep",
                "distribution": "Normal",
                "params": {"mu": 1.0, "sigma": 3.0},
                "sources": [],
                "reasoning": "Default weakly informative prior for the initial state mean of sleep.",
                "reference_interval_days": None,
                "density_points": None,
            },
            {
                "parameter": "t0_sd_stress",
                "distribution": "HalfNormal",
                "params": {"sigma": 4.0},
                "sources": [],
                "reasoning": (
                    "Default weakly informative prior for the initial state standard deviation "
                    "of stress."
                ),
                "reference_interval_days": None,
                "density_points": None,
            },
            {
                "parameter": "t0_sd_sleep",
                "distribution": "HalfNormal",
                "params": {"sigma": 5.0},
                "sources": [],
                "reasoning": (
                    "Default weakly informative prior for the initial state standard deviation "
                    "of sleep."
                ),
                "reference_interval_days": None,
                "density_points": None,
            },
        ]

    def test_resolve_prior_proposals_preserves_authored_metadata_for_lossy_bindings(
        self,
        simple_model_spec,
        simple_priors,
    ):
        """Resolved public priors should retain authored semantics when compilation is lossy."""
        from nof1_causal_lab.models.ssm.compile.artifact import (
            compile_ssm_artifact,
            resolve_prior_proposals,
        )

        compiled_ssm = compile_ssm_artifact(simple_model_spec, simple_priors)
        resolved = {
            prior["parameter"]: prior
            for prior in resolve_prior_proposals(compiled_ssm, authored_priors=simple_priors)
        }

        assert resolved["rho_mood"]["distribution"] == "Beta"
        assert resolved["rho_mood"]["params"] == {"alpha": 2.0, "beta": 2.0}
        assert resolved["rho_mood"]["reasoning"] == "Weakly informative for AR coefficient"
        assert resolved["sigma_mood"]["distribution"] == "HalfNormal"
        assert resolved["sigma_mood"]["params"] == {"sigma": 1.0}

    def test_resolve_prior_proposals_roundtrips_new_supported_prior_families(self):
        """Compiled semantics should surface LogNormal and bounded real priors."""
        from nof1_causal_lab.models.ssm.compile.artifact import resolve_prior_proposals

        compiled_ssm = {
            "compiled_prior_semantics": {
                "schema_version": 5,
                "site_registry": [
                    {
                        "name": "diffusion_diag_free",
                        "shape": [1],
                        "support": "positive",
                        "assembly_group": "diffusion",
                        "site_kind": "diffusion_diag",
                        "transform_kind": "exp",
                        "deterministic_name": "diffusion",
                        "fixed_spec_field": "diffusion",
                        "priors_field": "diffusion_diag",
                        "runtime_prior_key": "diffusion_diag_free",
                        "is_runtime_prior_controlled": True,
                    },
                    {
                        "name": "vf_0_weight",
                        "shape": [1],
                        "support": "real",
                        "assembly_group": "dynamics",
                        "site_kind": "dynamics_weight",
                        "transform_kind": "identity",
                        "deterministic_name": None,
                        "fixed_spec_field": None,
                        "priors_field": "linear_edge_weight",
                        "runtime_prior_key": "vf_0_weight",
                        "is_runtime_prior_controlled": True,
                    },
                ],
                "prior_state": {
                    "diffusion_diag_free": {
                        "family": [2],
                        "loc": [0.2],
                        "scale": [0.7],
                        "concentration": [1.0],
                        "rate": [1.0],
                    },
                    "vf_0_weight": {
                        "family": 2,
                        "loc": [0.0],
                        "scale": [0.3],
                        "low": [-1.0],
                        "high": [1.0],
                    },
                },
            },
            "parameter_bindings": [
                {"parameter": "sigma_mood", "site_name": "diffusion_diag_free", "flat_index": 0},
                {
                    "parameter": "cor_stress_sleep",
                    "site_name": "vf_0_weight",
                    "flat_index": 0,
                },
            ],
        }

        resolved = {
            prior["parameter"]: prior
            for prior in resolve_prior_proposals(compiled_ssm, authored_priors={})
        }
        assert resolved["sigma_mood"]["distribution"] == "LogNormal"
        assert resolved["sigma_mood"]["params"]["mu"] == pytest.approx(0.2)
        assert resolved["sigma_mood"]["params"]["sigma"] == pytest.approx(0.7)
        assert resolved["cor_stress_sleep"]["distribution"] == "Uniform"
        assert resolved["cor_stress_sleep"]["params"]["lower"] == pytest.approx(-1.0)
        assert resolved["cor_stress_sleep"]["params"]["upper"] == pytest.approx(1.0)

    def test_resolve_prior_proposals_uses_family_over_canonical_bounds(self):
        """Canonical low/high leaves must not force Normal sites to look truncated."""
        from nof1_causal_lab.models.ssm.compile.artifact import resolve_prior_proposals

        compiled_ssm = {
            "compiled_prior_semantics": {
                "schema_version": 5,
                "site_registry": [
                    {
                        "name": "vf_0_weight",
                        "shape": [1],
                        "support": "real",
                        "assembly_group": "dynamics",
                        "site_kind": "dynamics_weight",
                        "transform_kind": "identity",
                        "deterministic_name": None,
                        "fixed_spec_field": None,
                        "priors_field": "linear_edge_weight",
                        "runtime_prior_key": "vf_0_weight",
                        "is_runtime_prior_controlled": True,
                    }
                ],
                "prior_state": {
                    "vf_0_weight": {
                        "family": [0],
                        "loc": [0.15],
                        "scale": [0.4],
                        "low": [-1000000.0],
                        "high": [1000000.0],
                    }
                },
            },
            "parameter_bindings": [
                {"parameter": "beta_sleep_mood", "site_name": "vf_0_weight", "flat_index": 0}
            ],
        }

        resolved = resolve_prior_proposals(compiled_ssm, authored_priors={})

        assert resolved == [
            {
                "parameter": "beta_sleep_mood",
                "distribution": "Normal",
                "params": {"mu": 0.15, "sigma": 0.4},
                "sources": [],
                "reasoning": "Compiler-resolved prior for beta_sleep_mood.",
                "reference_interval_days": None,
                "density_points": None,
            }
        ]

    def test_resolve_prior_proposals_roundtrips_correlation_support_sites(self):
        """Compiled correlation-support sites should reconstruct bounded real priors."""
        from nof1_causal_lab.models.ssm.compile.artifact import resolve_prior_proposals

        compiled_ssm = {
            "compiled_prior_semantics": {
                "schema_version": 5,
                "site_registry": [
                    {
                        "name": "t0_var_lower_free",
                        "shape": [1],
                        "support": "correlation",
                        "assembly_group": "t0",
                        "site_kind": "t0_var_lower",
                        "transform_kind": "identity",
                        "deterministic_name": "t0_cov",
                        "fixed_spec_field": "t0_var",
                        "priors_field": "t0_var_offdiag",
                        "runtime_prior_key": "t0_var_lower_free",
                        "is_runtime_prior_controlled": True,
                    }
                ],
                "prior_state": {
                    "t0_var_lower_free": {
                        "family": [2],
                        "loc": [0.0],
                        "scale": [0.25],
                        "low": [-1.0],
                        "high": [1.0],
                    }
                },
            },
            "parameter_bindings": [
                {
                    "parameter": "cor0_sleep_stress",
                    "site_name": "t0_var_lower_free",
                    "flat_index": 0,
                }
            ],
        }

        resolved = resolve_prior_proposals(compiled_ssm, authored_priors={})

        assert resolved == [
            {
                "parameter": "cor0_sleep_stress",
                "distribution": "Uniform",
                "params": {"lower": -1.0, "upper": 1.0},
                "sources": [],
                "reasoning": "Compiler-resolved prior for cor0_sleep_stress.",
                "reference_interval_days": None,
                "density_points": None,
            }
        ]


class TestFailedParameters:
    """Test failed parameter localization."""

    def test_scale_mismatch_with_causal_spec_targets_construct(self):
        """Scale mismatch with causal_spec targets only the affected construct."""
        results = [
            PriorValidationResult(
                parameter="scale_mood_score",
                is_valid=False,
                issue="Scale mismatch for mood_score",
                suggested_adjustment=None,
            ),
        ]
        causal_spec = _with_positive_indicator_polarity(
            {
                "measurement": {
                    "model_clock": "1d",
                    "indicators": [
                        {"name": "mood_score", "construct_name": "mood"},
                        {"name": "stress_score", "construct_name": "stress"},
                    ],
                },
            }
        )
        all_params = ["rho_mood", "sigma_mood", "rho_stress", "sigma_stress", "beta_stress_mood"]
        failed = get_failed_parameters(results, all_params, causal_spec=causal_spec)
        # Only mood-related params should be re-elicited
        assert "rho_mood" in failed
        assert "sigma_mood" in failed
        assert "beta_stress_mood" in failed  # contains "mood"
        assert "rho_stress" not in failed
        assert "sigma_stress" not in failed


# --- SSM Prior Conversion Tests ---


class TestSSMPriorConversion:
    """Test that priors with non-Normal distributions convert correctly."""

    def test_beta_prior_converts_to_mu_sigma(self, simple_model_spec):
        """Beta(2,2) AR prior converts via AR-to-dynamics transform."""
        import math

        priors = {
            "rho_mood": {
                "parameter": "rho_mood",
                "distribution": "Beta",
                "params": {"alpha": 2.0, "beta": 2.0},
                "sources": [],
                "reasoning": "test",
            },
        }
        ssm_spec = _default_ssm_spec(n_latent=1, n_manifest=1, latent_names=["mood"])
        ssm_priors, index_maps, _diagnostics = compile_ssm_priors(
            priors,
            simple_model_spec,
            ssm_spec=ssm_spec,
        )

        # Beta(2,2): E[X] = 0.5 → decay mean = -ln(0.5)/1.0 ≈ 0.693.
        # The compiler stores the positive dynamics-decay prior as a Gamma moment match.
        expected_mu = -math.log(0.5) / 1.0
        mu_val = _positive_prior_mean_for_parameter(ssm_priors, index_maps, "rho_mood")
        assert abs(mu_val - expected_mu) < 0.01
        sigma_val = _positive_prior_sd_for_parameter(ssm_priors, index_maps, "rho_mood")
        assert sigma_val > 0.4  # delta method sigma

    def test_structured_prior_requires_structural_binding_for_residual_sd(self, simple_model_spec):
        """Structured priors should fail without a translated SSM binding."""
        priors = {
            "sigma_mood": {
                "parameter": "sigma_mood",
                "distribution": "HalfNormal",
                "params": {"sigma": 0.5},
                "sources": [],
                "reasoning": "test",
            },
        }
        with pytest.raises(ValueError, match="requires a translated SSMSpec"):
            compile_ssm_priors(priors, simple_model_spec, ssm_spec=None)

    def test_compile_ssm_inputs_validates_dict_once(self, simple_model_spec, simple_priors):
        """Compilation should validate a dict spec once, then pass the parsed object through."""
        from nof1_causal_lab.artifacts import ModelSpec

        with patch.object(ModelSpec, "model_validate", wraps=ModelSpec.model_validate) as validate:
            compile_ssm_inputs_from_model_spec(simple_model_spec, simple_priors)

        assert validate.call_count == 1

    def test_structured_prior_requires_structural_binding_for_loading(self, simple_model_spec):
        """Loading priors should fail without a translated SSM binding."""
        spec = dict(simple_model_spec)
        spec["parameters"] = [
            {
                "name": "lambda_mood",
                "role": "loading",
                "constraint": "positive",
                "description": "Factor loading",
            },
        ]
        priors = {
            "lambda_mood": {
                "parameter": "lambda_mood",
                "distribution": "HalfNormal",
                "params": {"sigma": 0.8},
                "sources": [],
                "reasoning": "test",
            },
        }
        with pytest.raises(ValueError, match="requires a translated SSMSpec"):
            compile_ssm_priors(priors, spec, ssm_spec=None)

    def test_unbound_prior_name_fails_without_model_spec(self):
        """Semantic prior compilation should fail fast when model_spec is missing."""
        priors = {
            "rho_x": {
                "distribution": "Normal",
                "params": {"mu": -0.3, "sigma": 0.5},
            },
        }
        with pytest.raises(ValueError, match="requires model_spec"):
            compile_ssm_priors(priors, {}, ssm_spec=None)

    def test_compile_priors_aggregates_independent_prior_errors(self):
        """Independent prior compile failures should be reported together."""

        model_spec = {
            "likelihoods": [
                {
                    "variable": "mood",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                }
            ],
            "parameters": [
                {
                    "name": "rho_mood",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                }
            ],
        }
        priors = {
            "rho_mood": {
                "distribution": "Uniform",
                "params": {"lower": -1.0, "upper": 2.0},
            },
            "bogus_param": {
                "distribution": "Normal",
                "params": {"mu": 0.0, "sigma": 1.0},
            },
        }
        ssm_spec = _default_ssm_spec(n_latent=1, n_manifest=1, latent_names=["mood"])

        with pytest.raises(ValueError, match="Prior compilation failed") as exc_info:
            compile_ssm_priors(priors, model_spec, ssm_spec=ssm_spec)

        message = str(exc_info.value)
        assert "Prior compilation failed" in message
        assert "lower bound is -1" in message
        assert "upper bound is 2" in message
        assert "bogus_param" in message

    def test_compile_ssm_artifact_aggregates_strict_binding_errors(self):
        """Strict causal-spec binding errors should be reported together."""
        from nof1_causal_lab.models.ssm.compile.artifact import compile_ssm_artifact

        causal_spec = _with_positive_indicator_polarity(
            {
                "latent": {
                    "constructs": [
                        {
                            "name": "mood",
                            "role": "endogenous",
                            "temporal_status": "time_varying",
                            "is_outcome": True,
                        },
                        {
                            "name": "stress",
                            "role": "exogenous",
                            "temporal_status": "time_varying",
                        },
                    ],
                    "edges": [{"cause": "stress", "effect": "mood"}],
                },
                "estimation": {
                    "state_order": ["mood", "stress"],
                    "edges": [{"cause": "stress", "effect": "mood"}],
                    "induced_dependencies": [],
                },
                "measurement": {
                    "model_clock": "1d",
                    "indicators": [
                        {
                            "name": "mood_score",
                            "construct_name": "mood",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                            "how_to_measure": "Use mood_score directly",
                        },
                        {
                            "name": "stress_score",
                            "construct_name": "stress",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                            "how_to_measure": "Use stress_score directly",
                        },
                    ],
                },
            }
        )
        model_spec = {
            "likelihoods": [
                {
                    "variable": "mood_score",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "Continuous score",
                },
                {
                    "variable": "stress_score",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "Continuous score",
                },
            ],
            "parameters": [
                {
                    "name": "rho_affect",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "Invalid AR name",
                },
                {
                    "name": "rho_stress",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "Valid AR name",
                },
                {
                    "name": "beta_mood_stress",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "Wrong causal direction",
                },
            ],
        }
        priors = {
            "rho_affect": {
                "parameter": "rho_affect",
                "distribution": "Beta",
                "params": {"alpha": 2.0, "beta": 2.0},
                "sources": [],
                "reasoning": "test",
            },
            "rho_stress": {
                "parameter": "rho_stress",
                "distribution": "Beta",
                "params": {"alpha": 2.0, "beta": 2.0},
                "sources": [],
                "reasoning": "test",
            },
            "beta_mood_stress": {
                "parameter": "beta_mood_stress",
                "distribution": "Normal",
                "params": {"mu": 0.0, "sigma": 0.5},
                "sources": [],
                "reasoning": "test",
            },
        }

        with pytest.raises(ValueError, match="Prior index binding failed") as exc_info:
            compile_ssm_artifact(model_spec, priors, causal_spec=causal_spec)

        message = str(exc_info.value)
        assert "Prior index binding failed" in message
        assert "rho_affect" in message
        assert "beta_mood_stress" in message

    def test_multiple_ar_params_produce_per_element_decay_rate(self):
        """Multiple AR params map to separate dynamics-decay entries."""
        import math

        model_spec = {
            "likelihoods": [
                {
                    "variable": "mood_score",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "variable": "stress_score",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
            ],
            "parameters": [
                {
                    "name": "rho_mood",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
                {
                    "name": "rho_stress",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
            ],
        }
        priors = {
            "rho_mood": {"distribution": "Beta", "params": {"alpha": 5.0, "beta": 2.0}},
            "rho_stress": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 5.0}},
        }
        ssm_spec = _default_ssm_spec(n_latent=2, n_manifest=2, latent_names=["mood", "stress"])
        ssm_priors, index_maps, _diagnostics = compile_ssm_priors(
            priors,
            model_spec,
            ssm_spec=ssm_spec,
        )

        # Beta(5,2) → E=5/7≈0.714, Beta(2,5) → E=2/7≈0.286
        mu_ar_mood = 5.0 / 7.0
        mu_ar_stress = 2.0 / 7.0
        expected_mood = -math.log(mu_ar_mood) / 1.0
        expected_stress = -math.log(mu_ar_stress) / 1.0
        assert (
            abs(
                _positive_prior_mean_for_parameter(ssm_priors, index_maps, "rho_mood")
                - expected_mood
            )
            < 0.01
        )
        assert (
            abs(
                _positive_prior_mean_for_parameter(ssm_priors, index_maps, "rho_stress")
                - expected_stress
            )
            < 0.01
        )

    def test_ar_transform_respects_granularity(self):
        """Hourly construct → dt=1/24, producing larger dynamics magnitude."""
        import math

        model_spec = {
            "likelihoods": [
                {"variable": "hr", "distribution": "gaussian", "link": "identity", "reasoning": ""},
            ],
            "parameters": [
                {
                    "name": "rho_heart_rate",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
            ],
        }
        priors = {
            "rho_heart_rate": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
        }
        causal_spec = _with_positive_indicator_polarity(
            {
                "latent": {
                    "constructs": [
                        {
                            "name": "heart_rate",
                            "temporal_status": "time_varying",
                        },
                    ],
                    "edges": [],
                },
                "measurement": {"model_clock": "1h", "indicators": []},
            }
        )
        ssm_spec = _default_ssm_spec(n_latent=1, n_manifest=1, latent_names=["heart_rate"])
        ssm_priors, index_maps, _diagnostics = compile_ssm_priors(
            priors,
            model_spec,
            ssm_spec=ssm_spec,
            causal_spec=causal_spec,
        )

        # Beta(2,2) → E=0.5; hourly dt = 1/24
        # dynamics-decay mean = -ln(0.5) / (1/24) = 0.693 * 24 ≈ 16.64
        dt_hourly = 1.0 / 24.0
        expected_mu = -math.log(0.5) / dt_hourly
        mu_val = _positive_prior_mean_for_parameter(ssm_priors, index_maps, "rho_heart_rate")
        assert abs(mu_val - expected_mu) < 0.1

    def test_beta_prior_dt_to_ct_transform(self):
        """FIXED_EFFECT beta priors are converted via element-wise beta/dt scaling."""

        model_spec = {
            "likelihoods": [
                {
                    "variable": "mood_score",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "variable": "stress_score",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
            ],
            "parameters": [
                {
                    "name": "rho_mood",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
                {
                    "name": "rho_stress",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
                {
                    "name": "beta_stress_mood",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "",
                },
            ],
        }
        priors = {
            "rho_mood": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "rho_stress": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "beta_stress_mood": {"distribution": "Normal", "params": {"mu": 0.3, "sigma": 0.15}},
        }
        # off-diagonal support enables [mood, stress].
        edge_support = np.array([[False, True], [False, False]])
        ssm_spec = _default_ssm_spec(
            n_latent=2,
            n_manifest=2,
            latent_names=["mood", "stress"],
            edge_support=edge_support,
        )
        ssm_priors, index_maps, _diagnostics = compile_ssm_priors(
            priors,
            model_spec,
            ssm_spec=ssm_spec,
            edge_lag_days={(0, 1): 1.0},
        )

        # Resolved 1d lag metadata: beta_CT = beta_DT / dt = 0.3 / 1 = 0.3
        mu_val = _real_prior_mean_for_parameter(ssm_priors, index_maps, "beta_stress_mood")
        assert abs(mu_val - 0.3) < 0.01

    def test_dt_ct_warning_uses_full_matrix_logm(self):
        """Cross-lag diagnostics should use logm(Phi)/dt, not beta/dt."""

        model_spec = {
            "likelihoods": [
                {
                    "variable": "mood_score",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "variable": "stress_score",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
            ],
            "parameters": [
                {
                    "name": "rho_mood",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
                {
                    "name": "rho_stress",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
                {
                    "name": "beta_stress_mood",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "",
                },
            ],
        }
        priors = {
            "rho_mood": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "rho_stress": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "beta_stress_mood": {"distribution": "Normal", "params": {"mu": 0.3, "sigma": 0.15}},
        }
        ssm_spec = _default_ssm_spec(
            n_latent=2,
            n_manifest=2,
            latent_names=["mood", "stress"],
            edge_support=np.array([[False, True], [False, False]]),
        )

        _ssm_priors, _idx, diagnostics = compile_ssm_priors(
            priors,
            model_spec,
            ssm_spec=ssm_spec,
            edge_lag_days={(0, 1): 1.0},
        )

        warning = next(
            diagnostic
            for diagnostic in diagnostics
            if diagnostic.code == "dt_ct_approximation_warning"
        )
        assert "matrix-log mismatch; exact CT coupling" in _require_text(warning.issue)
        assert "0.600 1/day" in _require_text(warning.issue)
        assert "beta/dt value 0.300 1/day" in _require_text(warning.issue)
        assert warning.pathology_certificate is not None
        assert warning.pathology_certificate.primary_score == pytest.approx(0.5, abs=0.001)

    def test_lagged_beta_diagnostics_explain_default_authored_interval(self):
        """Lagged-edge diagnostics should mention the default authored interval semantics."""

        model_spec = {
            "likelihoods": [
                {
                    "variable": "sleep",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "variable": "stress",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
            ],
            "parameters": [
                {
                    "name": "beta_stress_sleep",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "",
                },
            ],
        }
        priors = {
            "beta_stress_sleep": {
                "distribution": "Normal",
                "params": {"mu": 0.1, "sigma": 0.05},
                "sources": [
                    {
                        "title": "Weekly study",
                        "snippet": "Observed at weekly intervals.",
                        "study_interval_days": 7.0,
                    }
                ],
            },
        }
        ssm_spec = _default_ssm_spec(
            n_latent=2,
            n_manifest=2,
            latent_names=["stress", "sleep"],
            edge_support=np.array([[False, False], [True, False]]),
        )

        _priors, _idx, diagnostics = compile_ssm_priors(
            priors,
            model_spec,
            ssm_spec=ssm_spec,
            edge_lag_days={(1, 0): 1.0},
        )

        warning = next(
            diagnostic
            for diagnostic in diagnostics
            if diagnostic.code == "interval_reference_missing"
        )
        assert "`reference_interval_days` is omitted" in _require_text(warning.issue)
        assert "default model interval (1.0d)" in _require_text(warning.issue)
        assert "`reference_interval_days`" in _require_text(warning.suggested_adjustment)

    def test_lagged_beta_diagnostics_preserve_reference_interval_language(self):
        """Lagged-edge diagnostics should talk about the authored reference interval."""

        model_spec = {
            "likelihoods": [
                {
                    "variable": "sleep",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "variable": "stress",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
            ],
            "parameters": [
                {
                    "name": "beta_stress_sleep",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "",
                },
            ],
        }
        priors = {
            "beta_stress_sleep": {
                "distribution": "Normal",
                "params": {"mu": 0.3, "sigma": 0.15},
                "reference_interval_days": 7.0,
                "sources": [
                    {
                        "title": "Daily study",
                        "snippet": "Observed at daily intervals.",
                        "study_interval_days": 1.0,
                    }
                ],
            },
        }
        ssm_spec = _default_ssm_spec(
            n_latent=2,
            n_manifest=2,
            latent_names=["stress", "sleep"],
            edge_support=np.array([[False, False], [True, False]]),
        )

        _priors, _idx, diagnostics = compile_ssm_priors(
            priors,
            model_spec,
            ssm_spec=ssm_spec,
            edge_lag_days={(1, 0): 1.0},
        )

        warning = next(
            diagnostic
            for diagnostic in diagnostics
            if diagnostic.code == "interval_reference_mismatch"
        )
        assert "`reference_interval_days`" in _require_text(warning.issue)
        assert "7.0d" in _require_text(warning.issue)

    def test_beta_prior_dt_to_ct_respects_granularity(self):
        """FIXED_EFFECT beta transform uses effect construct's granularity."""

        model_spec = {
            "likelihoods": [
                {"variable": "hr", "distribution": "gaussian", "link": "identity", "reasoning": ""},
                {
                    "variable": "act",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
            ],
            "parameters": [
                {
                    "name": "rho_heart_rate",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
                {
                    "name": "rho_activity",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
                {
                    "name": "beta_activity_heart_rate",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "",
                },
            ],
        }
        priors = {
            "rho_heart_rate": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "rho_activity": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "beta_activity_heart_rate": {
                "distribution": "Normal",
                "params": {"mu": 0.3, "sigma": 0.15},
            },
        }
        causal_spec = _with_positive_indicator_polarity(
            {
                "latent": {
                    "constructs": [
                        {
                            "name": "heart_rate",
                            "role": "endogenous",
                            "temporal_status": "time_varying",
                        },
                        {
                            "name": "activity",
                            "role": "endogenous",
                            "temporal_status": "time_varying",
                        },
                    ],
                    "edges": [{"cause": "activity", "effect": "heart_rate"}],
                },
                "measurement": {"model_clock": "1h", "indicators": []},
            }
        )
        edge_support = np.array([[False, True], [False, False]])
        ssm_spec = _default_ssm_spec(
            n_latent=2,
            n_manifest=2,
            latent_names=["heart_rate", "activity"],
            edge_support=edge_support,
        )
        ssm_priors, index_maps, _diagnostics = compile_ssm_priors(
            priors,
            model_spec,
            ssm_spec=ssm_spec,
            causal_spec=causal_spec,
        )

        # Hourly dt = 1/24 → beta_CT = 0.3 / (1/24) = 7.2
        dt_hourly = 1.0 / 24.0
        expected_mu = 0.3 / dt_hourly  # 7.2
        mu_val = _real_prior_mean_for_parameter(
            ssm_priors,
            index_maps,
            "beta_activity_heart_rate",
        )
        assert abs(mu_val - expected_mu) < 0.5

    def test_compile_ssm_inputs_attaches_direct_writer_to_dt_ct_warning(self):
        model_spec = {
            "likelihoods": [
                {
                    "variable": "hr",
                    "distribution": "gaussian",
                    "link": "identity",
                    "centered": True,
                    "reasoning": "",
                },
                {
                    "variable": "act",
                    "distribution": "gaussian",
                    "link": "identity",
                    "centered": True,
                    "reasoning": "",
                },
            ],
            "parameters": [
                {
                    "name": "rho_heart_rate",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
                {
                    "name": "rho_activity",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
                {
                    "name": "beta_activity_heart_rate",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "",
                },
                {
                    "name": "sigma_heart_rate",
                    "role": "residual_sd",
                    "constraint": "positive",
                    "description": "",
                },
                {
                    "name": "sigma_activity",
                    "role": "residual_sd",
                    "constraint": "positive",
                    "description": "",
                },
            ],
            "initialization_policy": "stationary",
            "observation_intercept_policy": "free",
            "equilibrium_forcing": False,
        }
        priors = {
            "rho_heart_rate": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "rho_activity": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "beta_activity_heart_rate": {
                "distribution": "Normal",
                "params": {"mu": 0.3, "sigma": 0.15},
            },
            "sigma_heart_rate": {"distribution": "HalfNormal", "params": {"sigma": 1.0}},
            "sigma_activity": {"distribution": "HalfNormal", "params": {"sigma": 1.0}},
        }
        causal_spec = _with_positive_indicator_polarity(
            {
                "latent": {
                    "constructs": [
                        {
                            "name": "heart_rate",
                            "role": "endogenous",
                            "temporal_status": "time_varying",
                        },
                        {
                            "name": "activity",
                            "role": "endogenous",
                            "temporal_status": "time_varying",
                        },
                    ],
                    "edges": [{"cause": "activity", "effect": "heart_rate"}],
                },
                "estimation": {
                    "state_order": ["heart_rate", "activity"],
                    "edges": [{"cause": "activity", "effect": "heart_rate"}],
                    "induced_dependencies": [],
                },
                "measurement": {
                    "model_clock": "1h",
                    "indicators": [
                        {
                            "name": "hr",
                            "construct_name": "heart_rate",
                            "measurement_dtype": "continuous",
                        },
                        {
                            "name": "act",
                            "construct_name": "activity",
                            "measurement_dtype": "continuous",
                        },
                    ],
                },
            }
        )

        _ssm_spec, _ssm_priors, _bindings, diagnostics, _edge_lag_days = (
            compile_ssm_inputs_from_model_spec(
                model_spec,
                priors,
                causal_spec=causal_spec,
            )
        )

        dt_ct_warning = next(
            diagnostic
            for diagnostic in diagnostics
            if diagnostic.code == "dt_ct_approximation_warning"
        )
        assert dt_ct_warning.parameter == "linear_edge_weight"
        assert dt_ct_warning.related_parameters == ["beta_activity_heart_rate"]


# --- Trial Compile Tests ---


class TestTrialCompile:
    """Test trial_compile_model_spec catches structural errors early."""

    def test_valid_spec_returns_none(self, simple_model_spec):
        """A well-formed spec compiles successfully with default priors."""
        from nof1_causal_lab.models.ssm.compile.artifact import trial_compile_model_spec

        result = trial_compile_model_spec(simple_model_spec)
        assert result is None

    def test_compile_failure_returns_error(self):
        """When compilation raises, trial_compile returns the error string."""
        from nof1_causal_lab.models.ssm.compile.artifact import trial_compile_model_spec

        spec = {
            "likelihoods": [
                {
                    "variable": "x",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "test",
                }
            ],
            "parameters": [
                {
                    "name": "rho_x",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "test",
                }
            ],
        }
        with patch(
            "nof1_causal_lab.models.ssm.compile.artifact._compile_validated_ssm_artifact",
            side_effect=ValueError("dimension mismatch in dynamics matrix"),
        ):
            result = trial_compile_model_spec(spec)
        assert result is not None
        assert "dimension mismatch" in result

    def test_role_constraint_mismatch_returns_error(self):
        """Compiler should reject parameter-role constraint mismatches."""
        from nof1_causal_lab.models.ssm.compile.artifact import trial_compile_model_spec

        spec = {
            "likelihoods": [
                {
                    "variable": "x",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "test",
                }
            ],
            "parameters": [
                {
                    "name": "rho_x",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "test",
                },
                {
                    "name": "sigma_x",
                    "role": "residual_sd",
                    "constraint": "none",
                    "description": "test",
                },
            ],
        }

        result = trial_compile_model_spec(spec)

        assert result is not None
        assert "constraint 'none' unexpected for role 'residual_sd'" in result

    def test_missing_ar_parameters_returns_error(self):
        """Compiler should reject ModelSpecs with no latent dimensionality signal."""
        from nof1_causal_lab.models.ssm.compile.artifact import trial_compile_model_spec

        spec = {
            "likelihoods": [
                {
                    "variable": "x",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "test",
                }
            ],
            "parameters": [
                {
                    "name": "sigma_x",
                    "role": "residual_sd",
                    "constraint": "positive",
                    "description": "test",
                }
            ],
        }

        result = trial_compile_model_spec(spec)

        assert result is not None
        assert "No AR_COEFFICIENT parameters found" in result

    def test_rank_deficient_structure_returns_error(self):
        """Compiler should reject model specs with fewer manifests than latents."""
        from nof1_causal_lab.models.ssm.compile.artifact import trial_compile_model_spec

        spec = {
            "likelihoods": [
                {
                    "variable": "outcome_score",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "test",
                }
            ],
            "parameters": [
                {
                    "name": "rho_outcome",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "test",
                }
            ],
        }
        causal_spec = _with_positive_indicator_polarity(
            {
                "latent": {
                    "constructs": [
                        {
                            "name": "Treatment",
                            "role": "exogenous",
                            "description": "Treatment",
                            "temporal_status": "time_varying",
                        },
                        {
                            "name": "Outcome",
                            "role": "endogenous",
                            "description": "Outcome",
                            "temporal_status": "time_varying",
                            "is_outcome": True,
                        },
                    ],
                    "edges": [],
                },
                "estimation": {
                    "state_order": ["Treatment", "Outcome"],
                    "edges": [],
                    "induced_dependencies": [],
                },
                "measurement": {
                    "model_clock": "1d",
                    "indicators": [
                        {
                            "name": "outcome_score",
                            "construct_name": "Outcome",
                            "how_to_measure": "Use the outcome column directly",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                        }
                    ],
                },
            }
        )

        result = trial_compile_model_spec(spec, causal_spec)

        assert result is not None
        assert "Loading matrix is rank-deficient" in result

    def test_trial_compile_aggregates_initial_state_translation_errors(self):
        """Translation should report multiple initial-state correlation errors together."""
        from nof1_causal_lab.models.ssm.compile.artifact import trial_compile_model_spec

        causal_spec = _with_positive_indicator_polarity(
            {
                "latent": {
                    "constructs": [
                        {
                            "name": "X",
                            "role": "exogenous",
                            "description": "X",
                            "temporal_status": "time_varying",
                        },
                        {
                            "name": "Y",
                            "role": "endogenous",
                            "description": "Y",
                            "temporal_status": "time_varying",
                            "is_outcome": True,
                        },
                    ],
                    "edges": [{"cause": "X", "effect": "Y"}],
                },
                "estimation": {
                    "state_order": ["X", "Y"],
                    "edges": [{"cause": "X", "effect": "Y"}],
                    "induced_dependencies": [],
                },
                "measurement": {
                    "model_clock": "1d",
                    "indicators": [
                        {
                            "name": "x_score",
                            "construct_name": "X",
                            "how_to_measure": "Use x_score directly",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                        },
                        {
                            "name": "y_score",
                            "construct_name": "Y",
                            "how_to_measure": "Use y_score directly",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                        },
                    ],
                },
            }
        )
        spec = {
            "likelihoods": [
                {
                    "variable": "x_score",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "test",
                },
                {
                    "variable": "y_score",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "test",
                },
            ],
            "parameters": [
                {
                    "name": "rho_X",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "test",
                },
                {
                    "name": "rho_Y",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "test",
                },
                {
                    "name": "cor0_X_X",
                    "role": "initial_state_correlation",
                    "constraint": "correlation",
                    "description": "invalid self correlation",
                },
                {
                    "name": "cor0_unknown_pair",
                    "role": "initial_state_correlation",
                    "constraint": "correlation",
                    "description": "invalid parse",
                },
            ],
        }

        result = trial_compile_model_spec(spec, causal_spec)

        assert result is not None
        assert "no longer accepts INITIAL_STATE_CORRELATION parameters" in result
        assert "STATIC_STATE_SD baseline factors" in result
