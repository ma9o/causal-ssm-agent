"""Recovery tests for all inference methods.

Smoke tests verify pipeline correctness (small settings, fast).
Recovery tests verify parameter recovery within 90% CIs (slow).

All tests share the lgss_data fixture from conftest.py.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import polars as pl
import pytest

from causal_ssm_agent.models.ssm import InferenceResult, SSMModel, fit
from tests.helpers import assert_recovery_ci

DOCTOLIB_FIXTURE_DIR = Path(__file__).resolve().parents[3] / "data" / "DOCTOLIB" / "run"


def _load_doctolib_fixture(name: str) -> dict:
    """Load the shared Doctolib mock fixture used by the web app."""
    return json.loads((DOCTOLIB_FIXTURE_DIR / name).read_text())


def _assert_core_smoke_result(
    result: InferenceResult,
    *,
    method: str,
    n_draws: int,
    extra_diag_keys: tuple[str, ...] = (),
) -> dict[str, jnp.ndarray]:
    assert isinstance(result, InferenceResult)
    assert result.method == method
    samples = result.get_samples()

    for site in ["drift_diag_free", "diffusion_diag_free", "manifest_var_diag_free"]:
        assert site in samples, f"Missing sample site: {site}"

    assert samples["drift_diag_free"].shape == (n_draws, 1)
    assert samples["diffusion_diag_free"].shape == (n_draws, 1)
    assert samples["manifest_var_diag_free"].shape == (n_draws, 1)

    for key in extra_diag_keys:
        assert key in result.diagnostics

    return samples


def _assert_lgss_recovery(samples: dict[str, jnp.ndarray], lgss_data) -> None:
    assert_recovery_ci(
        samples["drift_diag_free"][:, 0],
        lgss_data["true_drift_diag"],
        "Drift",
        transform=lambda s: -jnp.abs(s),
    )
    assert_recovery_ci(
        samples["diffusion_diag_free"][:, 0],
        lgss_data["true_diff_diag"],
        "Diffusion",
    )
    assert_recovery_ci(
        samples["manifest_var_diag_free"][:, 0],
        lgss_data["true_obs_sd"],
        "Obs SD",
    )


# =============================================================================
# Smoke Tests
# =============================================================================


@pytest.mark.slow
@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    (
        "method",
        "model_factory",
        "fit_kwargs",
        "n_draws",
        "extra_diag_keys",
        "expected_accept_rate_len",
    ),
    [
        pytest.param(
            "nuts_da",
            lambda spec: SSMModel(spec),
            {
                "num_warmup": 50,
                "num_samples": 50,
                "num_chains": 1,
                "seed": 0,
            },
            50,
            (),
            None,
            id="nuts_da",
        ),
        pytest.param(
            "laplace_em",
            lambda spec: SSMModel(spec, n_particles=50),
            {
                "n_outer": 6,
                "n_csmc_particles": 8,
                "n_mh_steps": 3,
                "param_step_size": 0.1,
                "n_warmup": 3,
                "n_ieks_iters": 3,
                "adaptive_tempering": False,
                "seed": 0,
            },
            8,
            ("accept_rates", "n_ieks_iters"),
            6,
            id="laplace_em",
        ),
        pytest.param(
            "structured_vi",
            lambda spec: SSMModel(spec, n_particles=50),
            {
                "n_outer": 6,
                "n_csmc_particles": 8,
                "n_mh_steps": 3,
                "param_step_size": 0.1,
                "n_warmup": 3,
                "adaptive_tempering": False,
                "seed": 0,
            },
            8,
            ("accept_rates",),
            6,
            id="structured_vi",
        ),
        pytest.param(
            "dpf",
            lambda spec: SSMModel(spec, n_particles=50),
            {
                "n_outer": 6,
                "n_csmc_particles": 8,
                "n_mh_steps": 3,
                "param_step_size": 0.1,
                "n_warmup": 3,
                "adaptive_tempering": False,
                "n_train_seqs": 5,
                "n_train_steps": 20,
                "n_particles_train": 8,
                "n_pf_particles": 20,
                "seed": 0,
            },
            8,
            ("accept_rates", "proposal_net"),
            6,
            id="dpf",
        ),
    ],
)
def test_core_inference_smoke_matrix(
    lgss_data,
    method,
    model_factory,
    fit_kwargs,
    n_draws,
    extra_diag_keys,
    expected_accept_rate_len,
):
    """Backend smoke tests share one fixture and one output-shape contract."""
    model = model_factory(lgss_data["spec"])

    result = fit(
        model,
        observations=lgss_data["observations"],
        times=lgss_data["times"],
        method=method,
        **fit_kwargs,
    )

    samples = _assert_core_smoke_result(
        result,
        method=method,
        n_draws=n_draws,
        extra_diag_keys=extra_diag_keys,
    )

    if method == "nuts_da":
        assert "innovations" not in samples
    if expected_accept_rate_len is not None:
        assert len(result.diagnostics["accept_rates"]) == expected_accept_rate_len


# =============================================================================
# NUTS Data Augmentation
# =============================================================================


class TestNutsDARecovery:
    """NUTS-DA recovery tests on 1D LGSS."""

    @pytest.mark.slow
    @pytest.mark.timeout(300)
    def test_nuts_da_recovery(self, lgss_data):
        """NUTS-DA recovers 1D LGSS params (D=3) within 90% CIs."""
        model = SSMModel(lgss_data["spec"])

        result = fit(
            model,
            observations=lgss_data["observations"],
            times=lgss_data["times"],
            method="nuts_da",
            num_warmup=500,
            num_samples=500,
            num_chains=1,
            seed=0,
        )

        samples = result.get_samples()

        _assert_lgss_recovery(samples, lgss_data)


# =============================================================================
# Hess-MC2
# =============================================================================


class TestHessMC2Recovery:
    """Hess-MC2 Hessian proposal recovery on 1D LGSS (D=3)."""

    @pytest.mark.slow
    @pytest.mark.timeout(300)
    def test_lgss_hessian_recovery(self, lgss_data):
        """Hess-MC2 Hessian proposal recovers 1D LGSS params (D=3).

        Paper reference: Section IV-A. With D=3 and proper settings,
        the SO proposal should recover parameters within 90% CIs.

        Uses tempered warmup (warmup_iters=10) to avoid initial particle
        collapse with diffuse HalfNormal priors. N=256 for reliable
        posterior approximation.
        """
        model = SSMModel(lgss_data["spec"], n_particles=200)

        result = fit(
            model,
            observations=lgss_data["observations"],
            times=lgss_data["times"],
            method="hessmc2",
            n_smc_particles=256,
            n_iterations=20,
            proposal="hessian",
            step_size=0.5,
            warmup_iters=10,
            warmup_step_size=0.5,
            adapt_step_size=False,
            seed=0,
        )

        samples = result.get_samples()

        _assert_lgss_recovery(samples, lgss_data)


# =============================================================================
# Tempered SMC
# =============================================================================


class TestTemperedSMCRecovery:
    """Tempered SMC recovery tests on 1D LGSS."""

    @pytest.mark.slow
    @pytest.mark.timeout(300)
    def test_tempered_smc_recovery(self, lgss_data):
        """Tempered SMC recovers 1D LGSS params (D=3) within 90% CIs."""
        model = SSMModel(lgss_data["spec"], n_particles=50)

        result = fit(
            model,
            observations=lgss_data["observations"],
            times=lgss_data["times"],
            method="tempered_smc",
            n_outer=100,
            n_csmc_particles=20,
            n_mh_steps=10,
            param_step_size=0.1,
            n_warmup=50,
            adaptive_tempering=False,
            waste_free=False,
            n_leapfrog=1,
            seed=0,
        )

        samples = result.get_samples()

        _assert_lgss_recovery(samples, lgss_data)


# =============================================================================
# Laplace-EM
# =============================================================================


class TestLaplaceEM:
    """Laplace-EM recovery tests on 1D LGSS."""

    @pytest.mark.slow
    @pytest.mark.timeout(300)
    def test_laplace_em_recovery(self, lgss_data):
        """Laplace-EM recovers 1D LGSS params (D=3) within 90% CIs.

        Uses Kalman likelihood backend (exact for linear Gaussian) for fast
        evaluation. The Laplace-EM outer loop (tempered SMC over parameters)
        is the same as tempered_smc -- the method's value is for non-Gaussian
        emissions where Laplace approximation replaces the PF.
        """
        model = SSMModel(lgss_data["spec"], likelihood="kalman")

        result = fit(
            model,
            observations=lgss_data["observations"],
            times=lgss_data["times"],
            method="laplace_em",
            n_outer=100,
            n_csmc_particles=20,
            n_mh_steps=10,
            param_step_size=0.1,
            n_warmup=50,
            n_ieks_iters=5,
            adaptive_tempering=False,
            seed=0,
        )

        samples = result.get_samples()

        _assert_lgss_recovery(samples, lgss_data)


# =============================================================================
# Structured VI
# =============================================================================


class TestStructuredVI:
    """Structured VI recovery tests on 1D LGSS."""

    @pytest.mark.slow
    @pytest.mark.timeout(300)
    def test_structured_vi_recovery(self, lgss_data):
        """Structured VI recovers 1D LGSS params (D=3) within 90% CIs."""
        model = SSMModel(lgss_data["spec"], likelihood="kalman")

        result = fit(
            model,
            observations=lgss_data["observations"],
            times=lgss_data["times"],
            method="structured_vi",
            n_outer=100,
            n_csmc_particles=20,
            n_mh_steps=10,
            param_step_size=0.1,
            n_warmup=50,
            adaptive_tempering=False,
            seed=0,
        )

        samples = result.get_samples()

        _assert_lgss_recovery(samples, lgss_data)


# =============================================================================
# DPF (Differentiable Particle Filter)
# =============================================================================


class TestDPF:
    """DPF recovery tests on 1D LGSS."""

    @pytest.mark.slow
    @pytest.mark.timeout(600)
    def test_dpf_recovery(self, lgss_data):
        """DPF recovers 1D LGSS params (D=3) within 90% CIs.

        Uses Kalman likelihood for fast evaluation in the outer loop.
        The DPF's value is the trained proposal for non-Gaussian models.
        """
        model = SSMModel(lgss_data["spec"], likelihood="kalman")

        result = fit(
            model,
            observations=lgss_data["observations"],
            times=lgss_data["times"],
            method="dpf",
            n_outer=100,
            n_csmc_particles=20,
            n_mh_steps=10,
            param_step_size=0.1,
            n_warmup=50,
            adaptive_tempering=False,
            n_train_seqs=5,
            n_train_steps=20,
            n_particles_train=8,
            n_pf_particles=20,
            seed=0,
        )

        samples = result.get_samples()

        _assert_lgss_recovery(samples, lgss_data)


# =============================================================================
# Laplace-EM Doctolib Fixture
# =============================================================================


def _build_executable_doctolib_fixture_v2() -> tuple[dict, dict, dict, pl.DataFrame]:
    """Normalize the shared Doctolib web fixture into an executable pipeline artifact.

    The shared web fixture predates the stricter compiler contract: it uses
    shorthand parameter names and a latent graph that is not directly executable
    under the current builder. For inference smoke tests we normalize the
    parameter names into the compiler naming convention and use the latent graph
    implied by the stage-4 fixed effects.
    """
    stage4 = _load_doctolib_fixture("stage-4.json")
    stage1b = _load_doctolib_fixture("stage-1b.json")["causal_spec"]
    data_for_model = pl.read_parquet(DOCTOLIB_FIXTURE_DIR / "stage2-raw-data.parquet")

    name_map = {
        "beta_lipid_cv": "beta_lipid_burden_cardiovascular_risk",
        "beta_pressure_cv": "beta_arterial_pressure_cardiovascular_risk",
        "beta_glycemic_cv": "beta_glycemic_control_cardiovascular_risk",
        "beta_lipid_inflammation": "beta_lipid_burden_vascular_inflammation",
        "beta_inflammation_cv": "beta_vascular_inflammation_cardiovascular_risk",
        "beta_adherence_lipid": "beta_medication_adherence_lipid_burden",
        "beta_adherence_pressure": "beta_medication_adherence_arterial_pressure",
        "rho_lipid": "rho_lipid_burden",
        "rho_pressure": "rho_arterial_pressure",
        "sigma_lipid": "sigma_lipid_burden",
        "sigma_pressure": "sigma_arterial_pressure",
        "rho_inflammation": "rho_vascular_inflammation",
    }

    model_spec = json.loads(json.dumps(stage4["model_spec"]))
    for parameter in model_spec["parameters"]:
        parameter["name"] = name_map.get(parameter["name"], parameter["name"])
        if parameter["role"] == "ar_coefficient":
            parameter["constraint"] = "unit_interval"

    priors = json.loads(json.dumps(stage4["authored_priors"]))
    for old_name, new_name in name_map.items():
        if old_name in priors:
            priors[new_name] = priors.pop(old_name)

    beta_variables = {
        likelihood["variable"]
        for likelihood in model_spec["likelihoods"]
        if likelihood["distribution"] == "beta"
    }
    executable_manifest_names = {likelihood["variable"] for likelihood in model_spec["likelihoods"]}
    if beta_variables:
        eps = 1e-3
        data_for_model = data_for_model.with_columns(
            pl.when(pl.col("indicator").is_in(sorted(beta_variables)))
            .then(
                pl.col("value")
                .cast(pl.Float64, strict=False)
                .clip(lower_bound=eps, upper_bound=1.0 - eps)
                .cast(pl.Utf8)
            )
            .otherwise(pl.col("value"))
            .alias("value")
        )

    stage4_construct_names = {
        "medication_adherence",
        "lipid_burden",
        "vascular_inflammation",
        "glycemic_control",
        "arterial_pressure",
        "cardiovascular_risk",
    }
    measurement = {
        "model_clock": "1d",
        "indicators": [
            {
                **indicator,
                "construct_polarity": (
                    "negative" if indicator["name"] == "hdl_cholesterol" else "positive"
                ),
            }
            for indicator in stage1b["measurement"]["indicators"]
            if indicator["construct_name"] in stage4_construct_names
        ],
    }
    latent_constructs = [
        {
            "name": "medication_adherence",
            "description": "Prescription refill and appointment follow-through.",
            "role": "exogenous",
            "temporal_status": "time_varying",
        },
        {
            "name": "lipid_burden",
            "description": "Atherogenic lipid profile.",
            "role": "endogenous",
            "temporal_status": "time_varying",
        },
        {
            "name": "vascular_inflammation",
            "description": "Inflammatory state relevant to cardiovascular risk.",
            "role": "endogenous",
            "temporal_status": "time_varying",
        },
        {
            "name": "glycemic_control",
            "description": "Blood-glucose regulation quality.",
            "role": "endogenous",
            "temporal_status": "time_varying",
        },
        {
            "name": "arterial_pressure",
            "description": "Blood-pressure burden.",
            "role": "endogenous",
            "temporal_status": "time_varying",
        },
        {
            "name": "cardiovascular_risk",
            "description": "Overall cardiovascular risk trajectory.",
            "role": "endogenous",
            "is_outcome": True,
            "temporal_status": "time_varying",
        },
    ]
    latent_edges = [
        {
            "cause": "medication_adherence",
            "effect": "lipid_burden",
            "description": "Medication adherence improves lipid control.",
        },
        {
            "cause": "medication_adherence",
            "effect": "arterial_pressure",
            "description": "Medication adherence improves blood-pressure control.",
        },
        {
            "cause": "lipid_burden",
            "effect": "vascular_inflammation",
            "description": "Higher lipid burden raises vascular inflammation.",
        },
        {
            "cause": "lipid_burden",
            "effect": "cardiovascular_risk",
            "description": "Higher lipid burden raises cardiovascular risk.",
        },
        {
            "cause": "vascular_inflammation",
            "effect": "cardiovascular_risk",
            "description": "Higher vascular inflammation raises cardiovascular risk.",
        },
        {
            "cause": "glycemic_control",
            "effect": "cardiovascular_risk",
            "description": "Poorer glycemic control raises cardiovascular risk.",
        },
        {
            "cause": "arterial_pressure",
            "effect": "cardiovascular_risk",
            "description": "Higher arterial pressure raises cardiovascular risk.",
        },
    ]
    retained_state_order = [
        construct["name"]
        for construct in latent_constructs
        if any(
            indicator["construct_name"] == construct["name"]
            and indicator["name"] in executable_manifest_names
            for indicator in measurement["indicators"]
        )
    ]
    excluded_effect_suffixes = tuple(
        f"_{construct_name}"
        for construct_name in stage4_construct_names
        if construct_name not in retained_state_order
    )
    if excluded_effect_suffixes:
        model_spec["parameters"] = [
            parameter
            for parameter in model_spec["parameters"]
            if not (
                parameter["role"] == "fixed_effect"
                and any(parameter["name"].endswith(suffix) for suffix in excluded_effect_suffixes)
            )
        ]
        priors = {
            name: payload
            for name, payload in priors.items()
            if not (
                name.startswith("beta_")
                and any(name.endswith(suffix) for suffix in excluded_effect_suffixes)
            )
        }
    retained_parameter_names = {parameter["name"] for parameter in model_spec["parameters"]}
    priors = {name: payload for name, payload in priors.items() if name in retained_parameter_names}
    causal_spec = {
        "latent": {
            "constructs": latent_constructs,
            "edges": latent_edges,
        },
        "estimation": {
            "state_order": retained_state_order,
            "edges": [
                edge
                for edge in latent_edges
                if edge["cause"] in retained_state_order and edge["effect"] in retained_state_order
            ],
            "induced_dependencies": [],
        },
        "measurement": measurement,
    }
    return causal_spec, model_spec, priors, data_for_model


class TestLaplaceEMDoctolib:
    """Fixture-backed Laplace-EM smoke tests on the Doctolib mock data."""

    @pytest.mark.slow
    @pytest.mark.timeout(180)
    def test_laplace_em_doctolib_fixture_smoke(self):
        """Laplace-EM fits the executable Doctolib fixture end-to-end."""
        from causal_ssm_agent.artifacts import DistributionFamily
        from causal_ssm_agent.models.ssm.inference import select_default_method
        from causal_ssm_agent.models.ssm_builder import build_ssm_builder
        from causal_ssm_agent.models.ssm_compiler import compile_ssm_artifact
        from causal_ssm_agent.utils.data import pivot_to_wide

        causal_spec, model_spec, priors, data_for_model = _build_executable_doctolib_fixture_v2()

        assert data_for_model.schema["anchor_time"] == pl.String

        compiled = compile_ssm_artifact(
            model_spec,
            priors,
            causal_spec=causal_spec,
        )
        builder = build_ssm_builder(
            wide_data=pivot_to_wide(data_for_model),
            compiled_ssm=compiled,
            sampler_config={
                "method": "laplace_em",
                "n_outer": 6,
                "n_csmc_particles": 8,
                "n_mh_steps": 3,
                "param_step_size": 0.05,
                "n_warmup": 3,
                "n_ieks_iters": 3,
                "adaptive_tempering": False,
                "seed": 0,
            },
        )

        assert builder._spec is not None
        assert builder._spec.manifest_dists is not None
        assert DistributionFamily.BETA in builder._spec.manifest_dists
        assert select_default_method(builder._spec) == "laplace_em"

        wide = pivot_to_wide(data_for_model)
        assert wide.schema["time"] == pl.Float64

        result = builder.fit(wide)

        assert isinstance(result, InferenceResult)
        assert result.method == "laplace_em"

        samples = result.get_samples()
        assert "drift_diag_free" in samples
        assert "diffusion_diag_free" in samples
        assert "manifest_var_diag_free" in samples
        assert samples["drift_diag_free"].shape == (8, builder._spec.n_latent)
        assert samples["diffusion_diag_free"].shape == (8, builder._spec.n_latent)
        assert samples["manifest_var_diag_free"].shape == (8, builder._spec.n_manifest)
        assert bool(jnp.isfinite(samples["drift_diag_free"]).all())
        assert bool(jnp.isfinite(samples["diffusion_diag_free"]).all())
        assert bool(jnp.isfinite(samples["manifest_var_diag_free"]).all())

        assert "accept_rates" in result.diagnostics
        assert "n_ieks_iters" in result.diagnostics
        assert result.diagnostics["n_ieks_iters"] == 3
        assert len(result.diagnostics["accept_rates"]) == 6
