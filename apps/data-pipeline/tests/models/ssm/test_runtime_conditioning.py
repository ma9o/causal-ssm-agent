from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from blackjax.adaptation.step_size import dual_averaging_adaptation

from nof1_causal_lab.artifacts.model_spec import DistributionFamily, LinkFunction
from nof1_causal_lab.distributions import PriorDistributionFamily
from nof1_causal_lab.models.ssm.compile.artifact import compile_runtime_conditioning_metadata
from nof1_causal_lab.models.ssm.inference import fit
from nof1_causal_lab.models.ssm.inference.targets.polya_gamma import (
    build_polya_gamma_observation_plan,
    expected_pg1,
    initialize_polya_gamma_auxiliary_state,
    mask_polya_gamma_observations,
    negative_binomial_finite_sum_base_log_terms,
    polya_gamma_quadratic_log_prob,
    refresh_polya_gamma_auxiliary_state,
    sample_pg1_devroye,
)
from nof1_causal_lab.models.ssm.inference.targets.rao_blackwell import (
    RBPFPartitionSpec,
    build_gaussian_rbpf_observation_plan,
    build_rbpf_marginal_context,
    build_rbpf_partition,
    derive_rbpf_partition,
    full_path_rbpf_partition,
    rbpf_marginal_log_likelihood,
    validate_rbpf_mode,
    validate_rbpf_partition,
)
from nof1_causal_lab.models.ssm.inference.trajectory_mcmc import (
    build_auxiliary_kalman_bundle,
    build_hybrid_gibbs_nuts_parameter_kernel,
)
from nof1_causal_lab.models.ssm.inference.trajectory_mcmc.gibbs import (
    AuxKalmanMCMCState,
    _apply_param_dual_averaging_update_batched,
    _apply_windowed_latent_update_batched,
    _normalize_dual_averaging_state_shape,
)
from nof1_causal_lab.models.ssm.model import SSMModel
from nof1_causal_lab.models.ssm.priors import PriorSpec
from nof1_causal_lab.models.ssm.structure import (
    DiffusionBlockSpec,
    ManifestCholBlockSpec,
    SparseMatrixBlockSpec,
    SparseVectorBlockSpec,
    T0CholBlockSpec,
)
from nof1_causal_lab.models.ssm.structure.sites import SiteKind, SupportClass
from tests.models.ssm.test_inference_strategies import _make_aux_kalman_mcmc_smoke_spec
from tests.ssm_test_utils import (
    block_ssm_spec,
    dense_matrix_dynamics_spec,
    diagonal_diffusion_block,
    make_observation_support_runtime,
    prior_registry,
)

_TOY_TRUE_DECAY = np.asarray([0.45, 0.60], dtype=np.float32)
_TOY_TRUE_DIFFUSION_SD = np.asarray([0.18, 0.16], dtype=np.float32)
_TOY_TRUE_T0_SD = np.asarray([0.45, 0.45], dtype=np.float32)
_TOY_TRUE_OBS_SD = 0.18
_TOY_LOGIT_LOADINGS = np.asarray([0.4, 1.1], dtype=np.float32)


def _binary_pg_spec():
    return _make_aux_kalman_mcmc_smoke_spec(
        manifest_chol_block=ManifestCholBlockSpec(
            n_manifest=1,
            diag_support=np.asarray([False]),
            template=jnp.zeros((1, 1), dtype=jnp.float32),
        ),
        manifest_dists=[DistributionFamily.BERNOULLI],
        manifest_links=[LinkFunction.LOGIT],
    )


def _binary_pg_free_intercept_spec():
    return _make_aux_kalman_mcmc_smoke_spec(
        manifest_means_block=SparseVectorBlockSpec(
            n=1,
            free_support=np.asarray([True]),
            template=jnp.asarray([0.0], dtype=jnp.float32),
            free_site_name="manifest_means_free",
            det_site_name="manifest_means",
            support=SupportClass.REAL,
            site_kind=SiteKind.MANIFEST_MEANS,
            assembly_group="manifest",
            fixed_spec_field="manifest_means",
            priors_field="manifest_means",
        ),
        manifest_chol_block=ManifestCholBlockSpec(
            n_manifest=1,
            diag_support=np.asarray([False]),
            template=jnp.zeros((1, 1), dtype=jnp.float32),
        ),
        manifest_dists=[DistributionFamily.BERNOULLI],
        manifest_links=[LinkFunction.LOGIT],
    )


def _gaussian_free_intercept_spec():
    return _make_aux_kalman_mcmc_smoke_spec(
        manifest_means_block=SparseVectorBlockSpec(
            n=1,
            free_support=np.asarray([True]),
            template=jnp.asarray([0.0], dtype=jnp.float32),
            free_site_name="manifest_means_free",
            det_site_name="manifest_means",
            support=SupportClass.REAL,
            site_kind=SiteKind.MANIFEST_MEANS,
            assembly_group="manifest",
            fixed_spec_field="manifest_means",
            priors_field="manifest_means",
        ),
    )


def _binary_observations_and_times():
    observations = jnp.asarray([[0.0], [1.0], [1.0], [0.0], [1.0], [1.0]], dtype=jnp.float32)
    times = jnp.arange(observations.shape[0], dtype=jnp.float32)
    return observations, times


def _negative_binomial_pg_observations_and_times():
    observations = jnp.asarray(
        [[2.0, 0.1], [4.0, jnp.nan], [0.0, -0.2], [3.0, 0.4]],
        dtype=jnp.float32,
    )
    times = jnp.arange(observations.shape[0], dtype=jnp.float32)
    return observations, times


def _run_binary_pg_fit(method: str, *, enabled: bool):
    observations, times = _binary_observations_and_times()
    kwargs = {
        "method": method,
        "observations": observations,
        "times": times,
        "num_warmup": 4,
        "num_samples": 6,
        "num_chains": 1,
        "seed": 3,
        "init_method": "random",
        "latent_init_method": "predictive",
        "init_scale": 0.01,
        "latent_delta": 0.15,
        "param_step_size": 0.03,
        "parameter_kernel": "hybrid_gibbs_nuts",
        "adaptation_scheme": "simple",
        "compute_latent_posterior_summary": True,
        "enable_polya_gamma": enabled,
        "rbpf_mode": "none",
        "polya_gamma_num_terms": 16,
    }
    if method == "pit_particle_mgrad":
        kwargs["n_particles"] = 4
    return fit(SSMModel(_binary_pg_spec()), **kwargs)


def _pg_rbpf_spec():
    n_latent = 2
    n_manifest = 2
    return block_ssm_spec(
        n_latent=n_latent,
        n_manifest=n_manifest,
        dynamics_spec=dense_matrix_dynamics_spec(
            n_latent=n_latent,
            decay_support=np.asarray([False, False]),
            edge_support=np.zeros((n_latent, n_latent), dtype=bool),
            coupling_template=jnp.asarray([[-0.4, 0.0], [0.0, -0.35]], dtype=jnp.float32),
            intercept_support=np.zeros(n_latent, dtype=bool),
            cint_template=jnp.zeros(n_latent, dtype=jnp.float32),
        ),
        diffusion_block=diagonal_diffusion_block(n_latent),
        lambda_block=SparseMatrixBlockSpec(
            n_rows=n_manifest,
            n_cols=n_latent,
            free_support=np.zeros((n_manifest, n_latent), dtype=bool),
            template=jnp.eye(n_manifest, n_latent, dtype=jnp.float32),
            free_site_name="lambda_free",
            det_site_name="lambda",
            support=SupportClass.REAL,
            site_kind=SiteKind.LOADING,
            assembly_group="lambda",
            fixed_spec_field="lambda_mat",
            priors_field="lambda_free",
        ),
        manifest_means_block=SparseVectorBlockSpec(
            n=n_manifest,
            free_support=np.zeros(n_manifest, dtype=bool),
            template=jnp.zeros(n_manifest, dtype=jnp.float32),
            free_site_name="manifest_means_free",
            det_site_name="manifest_means",
            support=SupportClass.REAL,
            site_kind=SiteKind.MANIFEST_MEANS,
            assembly_group="manifest",
            fixed_spec_field="manifest_means",
            priors_field="manifest_means",
        ),
        manifest_chol_block=ManifestCholBlockSpec(
            n_manifest=n_manifest,
            diag_support=np.asarray([False, False]),
            template=jnp.diag(jnp.asarray([1.0, 0.35], dtype=jnp.float32)),
        ),
        t0_means_block=SparseVectorBlockSpec(
            n=n_latent,
            free_support=np.zeros(n_latent, dtype=bool),
            template=jnp.zeros(n_latent, dtype=jnp.float32),
            free_site_name="t0_means_free",
            det_site_name="t0_means",
            support=SupportClass.REAL,
            site_kind=SiteKind.T0_MEANS,
            assembly_group="t0",
            fixed_spec_field="t0_means",
            priors_field="t0_means",
        ),
        t0_chol_block=T0CholBlockSpec(
            n_latent=n_latent,
            diag_support=np.asarray([True, True]),
            correlation_support=np.zeros((n_latent, n_latent), dtype=bool),
            template=jnp.eye(n_latent, dtype=jnp.float32),
        ),
        manifest_dists=[DistributionFamily.BERNOULLI, DistributionFamily.GAUSSIAN],
        manifest_links=[LinkFunction.LOGIT, LinkFunction.IDENTITY],
        manifest_names=["clicked", "score"],
        latent_names=["engagement", "burden"],
    )


def _conditional_pg_rbpf_spec():
    spec = _pg_rbpf_spec()
    return block_ssm_spec(
        n_latent=2,
        n_manifest=2,
        dynamics_spec=spec.dynamics_spec,
        diffusion_block=spec.diffusion_block,
        lambda_block=SparseMatrixBlockSpec(
            n_rows=2,
            n_cols=2,
            free_support=np.zeros((2, 2), dtype=bool),
            template=jnp.asarray([[0.4, 1.0], [1.0, 0.0]], dtype=jnp.float32),
            free_site_name="lambda_free",
            det_site_name="lambda",
            support=SupportClass.REAL,
            site_kind=SiteKind.LOADING,
            assembly_group="lambda",
            fixed_spec_field="lambda_mat",
            priors_field="lambda_free",
        ),
        manifest_means_block=spec.manifest_means_block,
        manifest_chol_block=spec.manifest_chol_block,
        t0_means_block=spec.t0_means_block,
        t0_chol_block=spec.t0_chol_block,
        input_effect_block=spec.input_effect_block,
        static_state_sd_block=spec.static_state_sd_block,
        manifest_dists=spec.manifest_dists,
        manifest_links=spec.manifest_links,
        manifest_names=spec.manifest_names,
        latent_names=spec.latent_names,
    )


def _conditional_negative_binomial_pg_rbpf_spec():
    spec = _conditional_pg_rbpf_spec()
    return block_ssm_spec(
        n_latent=2,
        n_manifest=2,
        dynamics_spec=spec.dynamics_spec,
        diffusion_block=spec.diffusion_block,
        lambda_block=spec.lambda_block,
        manifest_means_block=spec.manifest_means_block,
        manifest_chol_block=spec.manifest_chol_block,
        t0_means_block=spec.t0_means_block,
        t0_chol_block=spec.t0_chol_block,
        input_effect_block=spec.input_effect_block,
        static_state_sd_block=spec.static_state_sd_block,
        manifest_dists=[DistributionFamily.NEGATIVE_BINOMIAL, DistributionFamily.GAUSSIAN],
        manifest_links=[LinkFunction.LOG, LinkFunction.IDENTITY],
        manifest_names=["events", "score"],
        latent_names=spec.latent_names,
    )


def _conditional_gaussian_rbpf_spec():
    spec = _pg_rbpf_spec()
    return block_ssm_spec(
        n_latent=2,
        n_manifest=2,
        dynamics_spec=spec.dynamics_spec,
        diffusion_block=spec.diffusion_block,
        lambda_block=SparseMatrixBlockSpec(
            n_rows=2,
            n_cols=2,
            free_support=np.zeros((2, 2), dtype=bool),
            template=jnp.asarray([[1.0, 0.0], [0.4, 1.0]], dtype=jnp.float32),
            free_site_name="lambda_free",
            det_site_name="lambda",
            support=SupportClass.REAL,
            site_kind=SiteKind.LOADING,
            assembly_group="lambda",
            fixed_spec_field="lambda_mat",
            priors_field="lambda_free",
        ),
        manifest_means_block=spec.manifest_means_block,
        manifest_chol_block=spec.manifest_chol_block,
        t0_means_block=spec.t0_means_block,
        t0_chol_block=spec.t0_chol_block,
        input_effect_block=spec.input_effect_block,
        static_state_sd_block=spec.static_state_sd_block,
        manifest_dists=spec.manifest_dists,
        manifest_links=spec.manifest_links,
        manifest_names=spec.manifest_names,
        latent_names=spec.latent_names,
    )


def _residual_dynamics_closure_rbpf_spec():
    n_latent = 3
    n_manifest = 3
    return block_ssm_spec(
        n_latent=n_latent,
        n_manifest=n_manifest,
        dynamics_spec=dense_matrix_dynamics_spec(
            n_latent=n_latent,
            decay_support=np.zeros(n_latent, dtype=bool),
            edge_support=np.zeros((n_latent, n_latent), dtype=bool),
            coupling_template=jnp.asarray(
                [[-0.4, 0.25, 0.0], [0.0, -0.35, 0.0], [0.0, 0.20, -0.3]],
                dtype=jnp.float32,
            ),
            intercept_support=np.zeros(n_latent, dtype=bool),
            cint_template=jnp.zeros(n_latent, dtype=jnp.float32),
        ),
        diffusion_block=diagonal_diffusion_block(n_latent),
        lambda_block=SparseMatrixBlockSpec(
            n_rows=n_manifest,
            n_cols=n_latent,
            free_support=np.zeros((n_manifest, n_latent), dtype=bool),
            template=jnp.asarray(
                [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                dtype=jnp.float32,
            ),
            free_site_name="lambda_free",
            det_site_name="lambda",
            support=SupportClass.REAL,
            site_kind=SiteKind.LOADING,
            assembly_group="lambda",
            fixed_spec_field="lambda_mat",
            priors_field="lambda_free",
        ),
        manifest_means_block=SparseVectorBlockSpec(
            n=n_manifest,
            free_support=np.zeros(n_manifest, dtype=bool),
            template=jnp.zeros(n_manifest, dtype=jnp.float32),
            free_site_name="manifest_means_free",
            det_site_name="manifest_means",
            support=SupportClass.REAL,
            site_kind=SiteKind.MANIFEST_MEANS,
            assembly_group="manifest",
            fixed_spec_field="manifest_means",
            priors_field="manifest_means",
        ),
        manifest_chol_block=ManifestCholBlockSpec(
            n_manifest=n_manifest,
            diag_support=np.asarray([False, False, False]),
            template=jnp.diag(jnp.asarray([0.0, 0.35, 0.35], dtype=jnp.float32)),
        ),
        t0_means_block=SparseVectorBlockSpec(
            n=n_latent,
            free_support=np.zeros(n_latent, dtype=bool),
            template=jnp.zeros(n_latent, dtype=jnp.float32),
            free_site_name="t0_means_free",
            det_site_name="t0_means",
            support=SupportClass.REAL,
            site_kind=SiteKind.T0_MEANS,
            assembly_group="t0",
            fixed_spec_field="t0_means",
            priors_field="t0_means",
        ),
        t0_chol_block=T0CholBlockSpec(
            n_latent=n_latent,
            diag_support=np.asarray([True, True, True]),
            correlation_support=np.zeros((n_latent, n_latent), dtype=bool),
            template=jnp.eye(n_latent, dtype=jnp.float32),
        ),
        manifest_dists=[
            DistributionFamily.GAMMA,
            DistributionFamily.GAUSSIAN,
            DistributionFamily.GAUSSIAN,
        ],
        manifest_links=[LinkFunction.LOG, LinkFunction.IDENTITY, LinkFunction.IDENTITY],
        manifest_names=["latency", "score", "driver"],
        latent_names=["sleep", "affect", "activity"],
    )


def _pg_rbpf_observations_and_times():
    observations = jnp.asarray(
        [
            [0.0, -0.35],
            [1.0, 0.10],
            [1.0, 0.32],
            [0.0, -0.15],
            [1.0, 0.24],
            [1.0, 0.05],
        ],
        dtype=jnp.float32,
    )
    times = jnp.arange(observations.shape[0], dtype=jnp.float32)
    return observations, times


def _interval_score_support_runtime():
    return make_observation_support_runtime(
        anchor_times=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        manifest_names=["clicked", "score"],
        support_kinds=[None, "interval"],
        summary_operators=["last", "mean"],
        observation_windows=[None, "1d"],
        support_start_times=np.asarray(
            [[np.nan, np.nan], [np.nan, 0.0], [np.nan, 1.0]],
            dtype=np.float64,
        ),
        support_end_times=np.asarray(
            [[np.nan, np.nan], [np.nan, 1.0], [np.nan, 2.0]],
            dtype=np.float64,
        ),
        interval_prev_coeffs=np.zeros((3, 2), dtype=np.float64),
        interval_curr_coeffs=np.zeros((3, 2), dtype=np.float64),
        interval_weights=np.zeros((3, 2), dtype=np.float64),
        emission_slot_indices=np.asarray([[-1, -1], [-1, 0], [-1, 0]], dtype=np.int64),
    )


def _toy_interval_score_support_runtime(T: int):
    support_start_times = np.full((T, 2), np.nan, dtype=np.float64)
    support_end_times = np.full((T, 2), np.nan, dtype=np.float64)
    interval_prev_coeffs = np.zeros((T, 2), dtype=np.float64)
    interval_curr_coeffs = np.zeros((T, 2), dtype=np.float64)
    interval_weights = np.zeros((T, 2), dtype=np.float64)
    emission_slot_indices = np.full((T, 2), -1, dtype=np.int64)

    rows = np.arange(1, T)
    support_start_times[rows, 1] = rows - 1
    support_end_times[rows, 1] = rows
    interval_prev_coeffs[rows, 1] = 0.5
    interval_curr_coeffs[rows, 1] = 0.5
    interval_weights[rows, 1] = 1.0
    emission_slot_indices[rows, 1] = 0

    return make_observation_support_runtime(
        anchor_times=np.arange(T, dtype=np.float64),
        manifest_names=["clicked", "score"],
        support_kinds=[None, "interval"],
        summary_operators=["last", "mean"],
        observation_windows=[None, "1d"],
        support_start_times=support_start_times,
        support_end_times=support_end_times,
        interval_prev_coeffs=interval_prev_coeffs,
        interval_curr_coeffs=interval_curr_coeffs,
        interval_weights=interval_weights,
        emission_slot_indices=emission_slot_indices,
    )


def _toy_pg_interval_rbpf_spec():
    return block_ssm_spec(
        n_latent=2,
        n_manifest=2,
        dynamics_spec=dense_matrix_dynamics_spec(
            n_latent=2,
            decay_support=np.asarray([False, False]),
            edge_support=np.zeros((2, 2), dtype=bool),
            coupling_template=-jnp.diag(jnp.asarray(_TOY_TRUE_DECAY, dtype=jnp.float32)),
            intercept_support=np.zeros(2, dtype=bool),
            cint_template=jnp.zeros(2, dtype=jnp.float32),
        ),
        diffusion_block=DiffusionBlockSpec(
            n_latent=2,
            diffusion_chol_support=np.zeros((2, 2), dtype=bool),
            diffusion_chol_template=jnp.diag(
                jnp.asarray(_TOY_TRUE_DIFFUSION_SD, dtype=jnp.float32)
            ),
        ),
        lambda_block=SparseMatrixBlockSpec(
            n_rows=2,
            n_cols=2,
            free_support=np.zeros((2, 2), dtype=bool),
            template=jnp.asarray(
                [
                    [_TOY_LOGIT_LOADINGS[0], _TOY_LOGIT_LOADINGS[1]],
                    [0.0, 1.0],
                ],
                dtype=jnp.float32,
            ),
            free_site_name="lambda_free",
            det_site_name="lambda",
            support=SupportClass.REAL,
            site_kind=SiteKind.LOADING,
            assembly_group="lambda",
            fixed_spec_field="lambda_mat",
            priors_field="lambda_free",
        ),
        manifest_means_block=SparseVectorBlockSpec(
            n=2,
            free_support=np.zeros(2, dtype=bool),
            template=jnp.zeros(2, dtype=jnp.float32),
            free_site_name="manifest_means_free",
            det_site_name="manifest_means",
            support=SupportClass.REAL,
            site_kind=SiteKind.MANIFEST_MEANS,
            assembly_group="manifest",
            fixed_spec_field="manifest_means",
            priors_field="manifest_means",
        ),
        manifest_chol_block=ManifestCholBlockSpec(
            n_manifest=2,
            diag_support=np.asarray([False, False]),
            template=jnp.diag(jnp.asarray([1.0, _TOY_TRUE_OBS_SD], dtype=jnp.float32)),
        ),
        t0_means_block=SparseVectorBlockSpec(
            n=2,
            free_support=np.zeros(2, dtype=bool),
            template=jnp.zeros(2, dtype=jnp.float32),
            free_site_name="t0_means_free",
            det_site_name="t0_means",
            support=SupportClass.REAL,
            site_kind=SiteKind.T0_MEANS,
            assembly_group="t0",
            fixed_spec_field="t0_means",
            priors_field="t0_means",
        ),
        t0_chol_block=T0CholBlockSpec(
            n_latent=2,
            diag_support=np.asarray([False, False]),
            correlation_support=np.zeros((2, 2), dtype=bool),
            template=jnp.diag(jnp.asarray(_TOY_TRUE_T0_SD, dtype=jnp.float32)),
        ),
        manifest_dists=[DistributionFamily.BERNOULLI, DistributionFamily.GAUSSIAN],
        manifest_links=[LinkFunction.LOGIT, LinkFunction.IDENTITY],
        manifest_names=["clicked", "score"],
        latent_names=["carried_drive", "marginal_burden"],
    )


def _tight_positive_lognormal_prior(value: float) -> PriorSpec:
    return PriorSpec(
        PriorDistributionFamily.LOG_NORMAL,
        {"mu": float(np.log(value)), "sigma": 0.12},
    )


def _toy_pg_interval_rbpf_model():
    model = SSMModel(
        _toy_pg_interval_rbpf_spec(),
        priors=prior_registry(
            vf_0_decay=_tight_positive_lognormal_prior(float(_TOY_TRUE_DECAY[0])),
            vf_1_decay=_tight_positive_lognormal_prior(float(_TOY_TRUE_DECAY[1])),
        ),
    )
    model.set_observation_support(_toy_interval_score_support_runtime(T=12))
    return model


def _toy_pg_interval_rbpf_observations_and_times():
    T = 12
    rng = np.random.default_rng(19)
    transition_scale = np.exp(-_TOY_TRUE_DECAY)
    innovation_sd = _TOY_TRUE_DIFFUSION_SD * np.sqrt(
        (1.0 - np.exp(-2.0 * _TOY_TRUE_DECAY)) / (2.0 * _TOY_TRUE_DECAY)
    )
    latent = np.zeros((T, 2), dtype=np.float32)
    latent[0] = rng.normal(loc=0.0, scale=_TOY_TRUE_T0_SD).astype(np.float32)
    for t in range(1, T):
        latent[t] = transition_scale * latent[t - 1] + rng.normal(
            loc=0.0, scale=innovation_sd
        ).astype(np.float32)

    logits = latent @ _TOY_LOGIT_LOADINGS
    clicked = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    score = np.full(T, np.nan, dtype=np.float32)
    score[1:] = 0.5 * (latent[:-1, 1] + latent[1:, 1]) + rng.normal(
        loc=0.0, scale=_TOY_TRUE_OBS_SD, size=T - 1
    ).astype(np.float32)
    observations = np.column_stack([clicked, score]).astype(np.float32)
    times = np.arange(T, dtype=np.float32)
    return jnp.asarray(observations), jnp.asarray(times), jnp.asarray(latent)


def _run_toy_pg_interval_rbpf_pit_fit(*, enabled: bool):
    observations, times, _latent = _toy_pg_interval_rbpf_observations_and_times()
    kwargs = {
        "method": "pit_particle_mgrad",
        "observations": observations,
        "times": times,
        "num_warmup": 5,
        "num_samples": 7,
        "num_chains": 1,
        "seed": 23,
        "init_method": "random",
        "auto_preconditioner_method": "none",
        "latent_init_method": "predictive",
        "init_scale": 0.0,
        "latent_delta": 0.10,
        "param_step_size": 0.015,
        "parameter_kernel": "hybrid_gibbs_nuts",
        "adaptation_scheme": "simple",
        "compute_latent_posterior_summary": True,
        "enable_polya_gamma": enabled,
        "rbpf_mode": "conditional" if enabled else "none",
        "polya_gamma_num_terms": 16,
        "n_particles": 5,
        "reparam": None,
    }
    if enabled:
        kwargs["rbpf_marginalized_latent_indices"] = (1,)
        kwargs["latent_kernel_algorithm"] = "pit_aux_csmc"
    return fit(_toy_pg_interval_rbpf_model(), **kwargs)


def _toy_decay_posterior_mean(result) -> np.ndarray:
    samples = result.get_samples()
    return np.asarray(
        [
            jnp.mean(samples["vf_0_decay"]),
            jnp.mean(samples["vf_1_decay"]),
        ],
        dtype=np.float32,
    )


def _toy_pg_interval_carried_work_counts(*, enabled: bool) -> tuple[int, int]:
    observations, times, _latent = _toy_pg_interval_rbpf_observations_and_times()
    bundle = build_auxiliary_kalman_bundle(
        _toy_pg_interval_rbpf_model(),
        observations,
        times,
        trace_key=jax.random.PRNGKey(0),
        reparam=None,
        enable_polya_gamma=enabled,
        rbpf_mode="conditional" if enabled else "none",
        rbpf_marginalized_latent_indices=(1,) if enabled else None,
        polya_gamma_num_terms=16,
    )
    context = bundle["latent_context_fn"](bundle["flat_example"])
    latent = bundle["initial_latent_from_context_fn"](context)
    return latent.shape[-1], latent.size


def _run_pg_rbpf_fit(method: str, *, enabled: bool):
    observations, times = _pg_rbpf_observations_and_times()
    kwargs = {
        "method": method,
        "observations": observations,
        "times": times,
        "num_warmup": 4,
        "num_samples": 6,
        "num_chains": 1,
        "seed": 5,
        "init_method": "random",
        "latent_init_method": "predictive",
        "init_scale": 0.0,
        "latent_delta": 0.12,
        "param_step_size": 0.025,
        "parameter_kernel": "hybrid_gibbs_nuts",
        "adaptation_scheme": "simple",
        "compute_latent_posterior_summary": True,
        "enable_polya_gamma": enabled,
        "rbpf_mode": "independent" if enabled else "none",
        "polya_gamma_num_terms": 16,
    }
    if enabled:
        kwargs["rbpf_marginalized_latent_indices"] = (1,)
    if method == "pit_particle_mgrad":
        kwargs["n_particles"] = 4
    return fit(SSMModel(_pg_rbpf_spec()), **kwargs)


def _run_conditional_pg_rbpf_fit(method: str, *, enabled: bool):
    observations, times = _pg_rbpf_observations_and_times()
    kwargs = {
        "method": method,
        "observations": observations,
        "times": times,
        "num_warmup": 4,
        "num_samples": 6,
        "num_chains": 1,
        "seed": 7,
        "init_method": "random",
        "latent_init_method": "predictive",
        "init_scale": 0.0,
        "latent_delta": 0.12,
        "param_step_size": 0.025,
        "parameter_kernel": "hybrid_gibbs_nuts",
        "adaptation_scheme": "simple",
        "compute_latent_posterior_summary": True,
        "enable_polya_gamma": enabled,
        "rbpf_mode": "conditional" if enabled else "none",
        "polya_gamma_num_terms": 16,
    }
    if enabled:
        kwargs["rbpf_marginalized_latent_indices"] = (1,)
    if method == "pit_particle_mgrad":
        kwargs["n_particles"] = 4
        if enabled:
            kwargs["latent_kernel_algorithm"] = "pit_aux_csmc"
    return fit(SSMModel(_conditional_pg_rbpf_spec()), **kwargs)


def test_polya_gamma_plan_consumes_only_observed_bernoulli_logit_cells():
    plan = build_polya_gamma_observation_plan(
        [DistributionFamily.BERNOULLI, DistributionFamily.GAUSSIAN],
        [LinkFunction.LOGIT, LinkFunction.IDENTITY],
        num_terms=8,
    )
    context = SimpleNamespace(
        H=jnp.asarray([[2.0], [1.0]]),
        d_meas=jnp.asarray([0.1, -0.3]),
        H_rows=None,
        d_rows=None,
    )
    latent = jnp.asarray([[0.2], [-0.4]])
    observations = jnp.asarray([[1.0, 2.0], [0.0, jnp.nan]])

    residual_observations = mask_polya_gamma_observations(plan, observations)
    assert bool(jnp.isnan(residual_observations[0, 0]))
    assert bool(jnp.isnan(residual_observations[1, 0]))
    assert float(residual_observations[0, 1]) == 2.0

    state = initialize_polya_gamma_auxiliary_state(plan, context, latent, observations)
    eta = jnp.asarray([0.5, -0.7])
    expected = jnp.sum(state.kappa[:, 0] * eta - 0.5 * state.omega[:, 0] * eta * eta)
    actual = polya_gamma_quadratic_log_prob(plan, state, context, latent)
    assert actual == pytest.approx(float(expected))

    refreshed = refresh_polya_gamma_auxiliary_state(
        jax.random.PRNGKey(0),
        plan,
        context,
        latent,
        observations,
    )
    assert refreshed.omega.shape == observations.shape
    assert bool(jnp.all(refreshed.omega[:, 0] > 0.0))
    assert bool(jnp.all(refreshed.omega[:, 1] == 0.0))


def test_polya_gamma_plan_consumes_negative_binomial_log_cells():
    plan = build_polya_gamma_observation_plan(
        [
            DistributionFamily.BERNOULLI,
            DistributionFamily.NEGATIVE_BINOMIAL,
            DistributionFamily.BETA,
        ],
        [LinkFunction.LOGIT, LinkFunction.LOG, LinkFunction.LOGIT],
        num_terms=8,
    )
    context = SimpleNamespace(
        H=jnp.asarray([[1.0], [2.0], [3.0]], dtype=jnp.float32),
        d_meas=jnp.asarray([0.0, 0.3, -0.2], dtype=jnp.float32),
        H_rows=None,
        d_rows=None,
        extra_params={"obs_r": jnp.asarray(3.0, dtype=jnp.float32)},
    )
    latent = jnp.asarray([[0.2], [-0.4]], dtype=jnp.float32)
    observations = jnp.asarray([[1.0, 2.0, 0.4], [0.0, 4.0, 0.6]], dtype=jnp.float32)

    assert plan.channel_mask.tolist() == [True, True, False]
    assert plan.bernoulli_channel_mask.tolist() == [True, False, False]
    assert plan.negative_binomial_channel_mask.tolist() == [False, True, False]

    residual_observations = mask_polya_gamma_observations(plan, observations)
    assert bool(jnp.isnan(residual_observations[0, 0]))
    assert bool(jnp.isnan(residual_observations[0, 1]))
    assert float(residual_observations[0, 2]) == pytest.approx(0.4)

    state = initialize_polya_gamma_auxiliary_state(plan, context, latent, observations)
    eta = latent @ context.H.T + context.d_meas
    psi_nb = eta[:, 1] - jnp.log(3.0)
    expected_shape_nb = observations[:, 1] + 3.0
    expected_kappa_nb = 0.5 * (observations[:, 1] - 3.0)
    assert bool(jnp.allclose(state.shape[:, 1], expected_shape_nb))
    assert bool(jnp.allclose(state.kappa[:, 1], expected_kappa_nb))
    assert bool(jnp.allclose(state.linear_offset[:, 1], -jnp.log(3.0)))
    assert bool(jnp.all(state.omega[:, 1] > 0.0))
    assert state.gamma_base_terms.shape == (*observations.shape, plan.num_terms)

    nb_base_terms = negative_binomial_finite_sum_base_log_terms(
        observations[:, 1],
        jnp.asarray(3.0, dtype=jnp.float32),
        state.gamma_base_terms[:, 1],
        state.active_mask[:, 1] > 0,
    )
    expected = jnp.sum(
        state.kappa[:, 0] * eta[:, 0]
        - 0.5 * state.omega[:, 0] * eta[:, 0] * eta[:, 0]
        + state.kappa[:, 1] * psi_nb
        - 0.5 * state.omega[:, 1] * psi_nb * psi_nb
        + nb_base_terms
    )
    actual = polya_gamma_quadratic_log_prob(plan, state, context, latent)
    assert actual == pytest.approx(float(expected))

    nb_only_plan = build_polya_gamma_observation_plan(
        [
            DistributionFamily.GAUSSIAN,
            DistributionFamily.NEGATIVE_BINOMIAL,
            DistributionFamily.BETA,
        ],
        [LinkFunction.IDENTITY, LinkFunction.LOG, LinkFunction.LOGIT],
        num_terms=8,
    )
    nb_only_expected = jnp.sum(
        state.kappa[:, 1] * psi_nb - 0.5 * state.omega[:, 1] * psi_nb * psi_nb + nb_base_terms
    )
    nb_only_actual = polya_gamma_quadratic_log_prob(nb_only_plan, state, context, latent)
    assert nb_only_actual == pytest.approx(float(nb_only_expected))

    refreshed = refresh_polya_gamma_auxiliary_state(
        jax.random.PRNGKey(4),
        plan,
        context,
        latent,
        observations,
    )
    assert bool(jnp.all(refreshed.omega[:, :2] > 0.0))
    assert bool(jnp.all(refreshed.omega[:, 2] == 0.0))

    proposal_context = SimpleNamespace(
        H=context.H,
        d_meas=context.d_meas,
        H_rows=None,
        d_rows=None,
        extra_params={"obs_r": jnp.asarray(7.0, dtype=jnp.float32)},
    )
    proposal_psi_nb = eta[:, 1] - jnp.log(7.0)
    proposal_kappa_nb = 0.5 * (observations[:, 1] - 7.0)
    proposal_base_terms = negative_binomial_finite_sum_base_log_terms(
        observations[:, 1],
        jnp.asarray(7.0, dtype=jnp.float32),
        state.gamma_base_terms[:, 1],
        state.active_mask[:, 1] > 0,
    )
    proposal_expected = jnp.sum(
        proposal_kappa_nb * proposal_psi_nb
        - 0.5 * state.omega[:, 1] * proposal_psi_nb * proposal_psi_nb
        + proposal_base_terms
    )
    proposal_actual = polya_gamma_quadratic_log_prob(
        nb_only_plan,
        state,
        proposal_context,
        latent,
    )
    assert proposal_actual == pytest.approx(float(proposal_expected))
    assert float(proposal_actual) != pytest.approx(float(nb_only_actual))


def test_devroye_sampler_rejects_negative_binomial_pg_channels():
    with pytest.raises(ValueError, match="negative-binomial"):
        build_polya_gamma_observation_plan(
            [DistributionFamily.NEGATIVE_BINOMIAL],
            [LinkFunction.LOG],
            num_terms=8,
            sampler="devroye",
        )


def test_devroye_polya_gamma_sampler_matches_pg1_mean():
    eta = jnp.asarray([0.0, 1.0, 2.0], dtype=jnp.float32)
    tiled_eta = jnp.broadcast_to(eta, (2048, eta.shape[0]))
    draws = sample_pg1_devroye(jax.random.PRNGKey(11), tiled_eta)

    sample_means = jnp.mean(draws, axis=0)
    expected_means = expected_pg1(eta)
    assert bool(jnp.all(jnp.abs(sample_means - expected_means) < 0.025))


def test_polya_gamma_sampler_option_threads_to_runtime_bundle():
    observations, times = _binary_observations_and_times()
    bundle = build_auxiliary_kalman_bundle(
        SSMModel(_binary_pg_spec()),
        observations,
        times,
        trace_key=jax.random.PRNGKey(0),
        reparam=None,
        enable_polya_gamma=True,
        rbpf_mode="none",
        polya_gamma_num_terms=16,
        polya_gamma_sampler="devroye",
    )

    assert bundle["polya_gamma_plan"].sampler == "devroye"
    assert bundle["polya_gamma_sampler"] == "devroye"
    context = bundle["latent_context_fn"](bundle["flat_example"])
    latent = bundle["initial_latent_from_context_fn"](context)
    refreshed = bundle["refresh_observation_auxiliary_from_context_fn"](
        context,
        latent,
        bundle["initial_observation_auxiliary_from_context_fn"](context, latent),
        jax.random.PRNGKey(1),
    )
    assert bool(jnp.all(refreshed.omega[refreshed.active_mask.astype(bool)] > 0.0))


def test_hybrid_gibbs_nuts_plans_pg_measurement_intercept_exact_block():
    observations, times = _binary_observations_and_times()
    bundle = build_auxiliary_kalman_bundle(
        SSMModel(_binary_pg_free_intercept_spec()),
        observations,
        times,
        trace_key=jax.random.PRNGKey(0),
        reparam=None,
        enable_polya_gamma=True,
        polya_gamma_num_terms=16,
    )

    kernel = build_hybrid_gibbs_nuts_parameter_kernel(
        bundle,
        step_size=0.03,
        target_accept=0.8,
        max_num_doublings=2,
    )

    assert kernel["gibbs_block_count"] == 1
    assert kernel["residual_dim"] == bundle["dim"] - 1
    assert kernel["gibbs_indices"].shape == (1,)


def test_hybrid_gibbs_nuts_plans_gaussian_measurement_intercept_exact_block():
    observations = jnp.asarray([[0.05], [0.12], [-0.03]], dtype=jnp.float32)
    times = jnp.arange(observations.shape[0], dtype=jnp.float32)
    bundle = build_auxiliary_kalman_bundle(
        SSMModel(_gaussian_free_intercept_spec()),
        observations,
        times,
        trace_key=jax.random.PRNGKey(0),
        reparam=None,
        enable_polya_gamma=True,
    )

    kernel = build_hybrid_gibbs_nuts_parameter_kernel(
        bundle,
        step_size=0.03,
        target_accept=0.8,
        max_num_doublings=2,
    )

    assert kernel["gibbs_block_count"] == 1
    assert kernel["residual_dim"] == bundle["dim"] - 1
    assert kernel["gibbs_indices"].shape == (1,)


def test_aux_kalman_mcmc_runs_pg_measurement_intercept_exact_block():
    observations, times = _binary_observations_and_times()
    result = fit(
        SSMModel(_binary_pg_free_intercept_spec()),
        observations=observations,
        times=times,
        method="aux_kalman_mcmc",
        num_warmup=1,
        num_samples=1,
        num_chains=1,
        seed=17,
        init_method="random",
        latent_init_method="predictive",
        init_scale=0.01,
        latent_delta=0.15,
        param_step_size=0.03,
        param_max_num_doublings=2,
        adaptation_scheme="simple",
        compute_latent_posterior_summary=False,
        enable_polya_gamma=True,
        polya_gamma_num_terms=16,
    )

    diag = result.diagnostics["aux_kalman_mcmc"]
    samples = result.get_samples()
    assert diag["parameter_gibbs_block_count"] == 1
    residual_expected = (
        samples["diffusion_diag_free"].shape[-1]
        + samples["t0_var_diag_free"].shape[-1]
        + samples["vf_0_decay"].shape[-1]
    )
    assert diag["parameter_residual_dim"] == residual_expected
    assert bool(jnp.isfinite(samples["manifest_means_free"]).all())


def test_exact_integer_polya_gamma_sampler_threads_to_negative_binomial_runtime_bundle():
    observations, times = _negative_binomial_pg_observations_and_times()
    model = SSMModel(
        _conditional_negative_binomial_pg_rbpf_spec(),
        priors=prior_registry(
            obs_r=PriorSpec(PriorDistributionFamily.DELTA, {"value": 3.0}),
        ),
    )
    bundle = build_auxiliary_kalman_bundle(
        model,
        observations,
        times,
        trace_key=jax.random.PRNGKey(0),
        reparam=None,
        enable_polya_gamma=True,
        rbpf_mode="none",
        polya_gamma_num_terms=16,
        polya_gamma_sampler="devroye_integer",
    )

    plan = bundle["polya_gamma_plan"]
    assert plan.sampler == "devroye_integer"
    assert plan.max_integer_shape == 7
    assert bundle["polya_gamma_max_integer_shape"] == 7
    context = bundle["latent_context_fn"](bundle["flat_example"])
    latent = bundle["initial_latent_from_context_fn"](context)
    refreshed = bundle["refresh_observation_auxiliary_from_context_fn"](
        context,
        latent,
        bundle["initial_observation_auxiliary_from_context_fn"](context, latent),
        jax.random.PRNGKey(1),
    )
    assert refreshed.gamma_base_terms.shape == (*observations.shape, 0)
    assert bool(jnp.all(refreshed.omega[refreshed.active_mask.astype(bool)] > 0.0))

    larger_observations = observations.at[0, 0].set(8.0)
    larger_bundle = build_auxiliary_kalman_bundle(
        model,
        larger_observations,
        times,
        trace_key=jax.random.PRNGKey(0),
        reparam=None,
        enable_polya_gamma=True,
        rbpf_mode="none",
        polya_gamma_num_terms=16,
        polya_gamma_sampler="devroye_integer",
    )
    assert larger_bundle["polya_gamma_plan"].max_integer_shape == 11


def test_exact_integer_polya_gamma_sampler_rejects_sampled_negative_binomial_dispersion():
    observations, times = _negative_binomial_pg_observations_and_times()
    with pytest.raises(ValueError, match="obs_r to have a Delta prior"):
        build_auxiliary_kalman_bundle(
            SSMModel(_conditional_negative_binomial_pg_rbpf_spec()),
            observations,
            times,
            trace_key=jax.random.PRNGKey(0),
            reparam=None,
            enable_polya_gamma=True,
            rbpf_mode="none",
            polya_gamma_num_terms=16,
            polya_gamma_sampler="devroye_integer",
        )


def test_exact_integer_polya_gamma_sampler_rejects_noninteger_negative_binomial_counts():
    observations, times = _negative_binomial_pg_observations_and_times()
    observations = observations.at[0, 0].set(2.5)
    model = SSMModel(
        _conditional_negative_binomial_pg_rbpf_spec(),
        priors=prior_registry(
            obs_r=PriorSpec(PriorDistributionFamily.DELTA, {"value": 3.0}),
        ),
    )
    with pytest.raises(ValueError, match="integer-valued observations"):
        build_auxiliary_kalman_bundle(
            model,
            observations,
            times,
            trace_key=jax.random.PRNGKey(0),
            reparam=None,
            enable_polya_gamma=True,
            rbpf_mode="none",
            polya_gamma_num_terms=16,
            polya_gamma_sampler="devroye_integer",
        )


@pytest.mark.parametrize("method", ["aux_kalman_mcmc", "pit_particle_mgrad"])
def test_pg_conditioning_preserves_small_binary_sample_structure(method):
    conditioned = _run_binary_pg_fit(method, enabled=True)
    unconditioned = _run_binary_pg_fit(method, enabled=False)

    conditioned_diag = conditioned.diagnostics[method]
    unconditioned_diag = unconditioned.diagnostics[method]
    assert conditioned_diag["polya_gamma_enabled"] is True
    assert conditioned_diag["polya_gamma_sampler"] == "truncated_sum"
    assert conditioned_diag["polya_gamma_channels"] == 1
    assert conditioned_diag["rbpf_enabled"] is False
    assert conditioned_diag["rbpf_carried_latent_indices"] == [0]
    assert conditioned_diag["rbpf_marginalized_latent_indices"] == []
    assert unconditioned_diag["polya_gamma_enabled"] is False
    assert unconditioned_diag["polya_gamma_sampler"] == "truncated_sum"
    assert unconditioned_diag["polya_gamma_channels"] == 0
    assert unconditioned_diag["rbpf_enabled"] is False

    conditioned_samples = conditioned.get_samples()
    unconditioned_samples = unconditioned.get_samples()
    for site_name in ("diffusion_diag_free", "t0_var_diag_free", "vf_0_decay"):
        assert bool(jnp.isfinite(conditioned_samples[site_name]).all())
        assert bool(jnp.isfinite(unconditioned_samples[site_name]).all())

    assert conditioned_samples["diffusion_diag_free"].shape == (6, 1)
    assert unconditioned_samples["diffusion_diag_free"].shape == (6, 1)
    assert conditioned_samples["t0_var_diag_free"].shape == (6, 1)
    assert unconditioned_samples["t0_var_diag_free"].shape == (6, 1)
    assert bool(jnp.all(conditioned_samples["diffusion_diag_free"] > 0.0))
    assert bool(jnp.all(unconditioned_samples["diffusion_diag_free"] > 0.0))
    assert bool(jnp.all(conditioned_samples["t0_var_diag_free"] > 0.0))
    assert bool(jnp.all(unconditioned_samples["t0_var_diag_free"] > 0.0))

    conditioned_latent_summary = conditioned.get_latent_posterior_summary()
    unconditioned_latent_summary = unconditioned.get_latent_posterior_summary()
    assert conditioned_latent_summary is not None
    assert unconditioned_latent_summary is not None
    assert conditioned_latent_summary["mean"].shape == (6, 1)
    assert unconditioned_latent_summary["mean"].shape == (6, 1)


def _conditioning_gradient_work_counts(*, enabled: bool) -> tuple[int, int]:
    observations, times = _binary_observations_and_times()
    model = SSMModel(_binary_pg_spec())
    bundle = build_auxiliary_kalman_bundle(
        model,
        observations,
        times,
        trace_key=jax.random.PRNGKey(0),
        reparam=None,
        enable_polya_gamma=enabled,
        rbpf_mode="none",
        polya_gamma_num_terms=16,
    )
    context = bundle["latent_context_fn"](bundle["flat_example"])
    latent = bundle["initial_latent_from_context_fn"](context)
    observation_auxiliary = bundle["initial_observation_auxiliary_from_context_fn"](
        context,
        latent,
    )

    full_grad_jaxpr = jax.make_jaxpr(
        lambda latent_arg: bundle["observation_grad_conditioned_from_context_fn"](
            context,
            latent_arg,
            observation_auxiliary,
        )
    )(latent)

    def increment_grad(latent_t):
        return jax.grad(
            lambda state_t: bundle["observation_increment_log_prob_conditioned_from_context_fn"](
                context,
                state_t,
                observation_auxiliary,
                jnp.asarray(0, dtype=jnp.int32),
            )
        )(latent_t)

    increment_grad_jaxpr = jax.make_jaxpr(increment_grad)(latent[0])
    return len(full_grad_jaxpr.jaxpr.eqns), len(increment_grad_jaxpr.jaxpr.eqns)


def test_pg_conditioning_has_lower_gradient_work_than_unconditioned_runtime():
    conditioned_full, conditioned_increment = _conditioning_gradient_work_counts(enabled=True)
    unconditioned_full, unconditioned_increment = _conditioning_gradient_work_counts(enabled=False)

    assert conditioned_full < unconditioned_full
    assert conditioned_increment < unconditioned_increment


@pytest.mark.parametrize("method", ["aux_kalman_mcmc", "pit_particle_mgrad"])
def test_pg_plus_independent_rbpf_preserves_small_sample_structure(method):
    conditioned = _run_pg_rbpf_fit(method, enabled=True)
    unconditioned = _run_pg_rbpf_fit(method, enabled=False)

    conditioned_diag = conditioned.diagnostics[method]
    unconditioned_diag = unconditioned.diagnostics[method]
    assert conditioned_diag["polya_gamma_enabled"] is True
    assert conditioned_diag["polya_gamma_sampler"] == "truncated_sum"
    assert conditioned_diag["polya_gamma_channels"] == 1
    assert conditioned_diag["rbpf_enabled"] is True
    assert conditioned_diag["rbpf_mode"] == "independent"
    assert conditioned_diag["rbpf_structure"] == "independent"
    assert conditioned_diag["rbpf_carried_latent_indices"] == [0]
    assert conditioned_diag["rbpf_marginalized_latent_indices"] == [1]
    assert conditioned_diag["rbpf_observation_channels"] == 1
    assert conditioned_diag["rbpf_gaussian_observation_channels"] == 1
    assert conditioned_diag["rbpf_polya_gamma_observation_channels"] == 0
    if method == "pit_particle_mgrad":
        assert conditioned_diag["latent_kernel_algorithm"] == "sequential_particle_mgrad"
        assert conditioned_diag["latent_kernel_family"] == "particle_mgrad"
        assert conditioned_diag["parallel_time"] is False
    assert unconditioned_diag["polya_gamma_enabled"] is False
    assert unconditioned_diag["polya_gamma_sampler"] == "truncated_sum"
    assert unconditioned_diag["rbpf_enabled"] is False
    assert unconditioned_diag["rbpf_mode"] == "none"

    conditioned_samples = conditioned.get_samples()
    unconditioned_samples = unconditioned.get_samples()
    for site_name in ("diffusion_diag_free", "t0_var_diag_free"):
        assert conditioned_samples[site_name].shape[-1] == 2
        assert unconditioned_samples[site_name].shape[-1] == 2
        assert bool(jnp.isfinite(conditioned_samples[site_name]).all())
        assert bool(jnp.isfinite(unconditioned_samples[site_name]).all())

    conditioned_diffusion = jnp.mean(conditioned_samples["diffusion_diag_free"], axis=0)
    unconditioned_diffusion = jnp.mean(unconditioned_samples["diffusion_diag_free"], axis=0)
    conditioned_t0_sd = jnp.mean(conditioned_samples["t0_var_diag_free"], axis=0)
    unconditioned_t0_sd = jnp.mean(unconditioned_samples["t0_var_diag_free"], axis=0)

    assert bool(jnp.all(conditioned_diffusion > 0.0))
    assert bool(jnp.all(unconditioned_diffusion > 0.0))
    assert bool(jnp.all(conditioned_t0_sd > 0.0))
    assert bool(jnp.all(unconditioned_t0_sd > 0.0))
    conditioned_latent_summary = conditioned.get_latent_posterior_summary()
    unconditioned_latent_summary = unconditioned.get_latent_posterior_summary()
    assert conditioned_latent_summary is not None
    assert unconditioned_latent_summary is not None
    assert conditioned_latent_summary["mean"].shape == (6, 2)
    assert unconditioned_latent_summary["mean"].shape == (6, 2)


@pytest.mark.parametrize("method", ["aux_kalman_mcmc", "pit_particle_mgrad"])
def test_pg_plus_conditional_rbpf_preserves_small_sample_structure(method):
    conditioned = _run_conditional_pg_rbpf_fit(method, enabled=True)
    unconditioned = _run_conditional_pg_rbpf_fit(method, enabled=False)

    conditioned_diag = conditioned.diagnostics[method]
    unconditioned_diag = unconditioned.diagnostics[method]
    assert conditioned_diag["polya_gamma_enabled"] is True
    assert conditioned_diag["polya_gamma_sampler"] == "truncated_sum"
    assert conditioned_diag["rbpf_enabled"] is True
    assert conditioned_diag["rbpf_mode"] == "conditional"
    assert conditioned_diag["rbpf_structure"] == "conditional"
    assert conditioned_diag["rbpf_carried_latent_indices"] == [0]
    assert conditioned_diag["rbpf_marginalized_latent_indices"] == [1]
    assert conditioned_diag["rbpf_observation_channels"] == 1
    assert conditioned_diag["rbpf_gaussian_observation_channels"] == 0
    assert conditioned_diag["rbpf_polya_gamma_observation_channels"] == 1
    if method == "pit_particle_mgrad":
        assert conditioned_diag["latent_kernel_algorithm"] == "conditional_aux_rbpf_csmc"
        assert conditioned_diag["latent_kernel_family"] == "auxiliary_csmc"
        assert conditioned_diag["parallel_time"] is False
    assert unconditioned_diag["rbpf_enabled"] is False
    assert unconditioned_diag["polya_gamma_sampler"] == "truncated_sum"
    assert unconditioned_diag["rbpf_mode"] == "none"

    conditioned_samples = conditioned.get_samples()
    unconditioned_samples = unconditioned.get_samples()
    for site_name in ("diffusion_diag_free", "t0_var_diag_free"):
        assert bool(jnp.isfinite(conditioned_samples[site_name]).all())
        assert bool(jnp.isfinite(unconditioned_samples[site_name]).all())

    assert bool(jnp.all(conditioned_samples["diffusion_diag_free"] > 0.0))
    assert bool(jnp.all(unconditioned_samples["diffusion_diag_free"] > 0.0))
    assert bool(jnp.all(conditioned_samples["t0_var_diag_free"] > 0.0))
    assert bool(jnp.all(unconditioned_samples["t0_var_diag_free"] > 0.0))
    if method == "pit_particle_mgrad":
        assert unconditioned_diag["latent_kernel_algorithm"] == "sequential_particle_mgrad"
        assert unconditioned_diag["latent_kernel_family"] == "particle_mgrad"

    conditioned_latent_summary = conditioned.get_latent_posterior_summary()
    unconditioned_latent_summary = unconditioned.get_latent_posterior_summary()
    assert conditioned_latent_summary is not None
    assert unconditioned_latent_summary is not None
    assert conditioned_latent_summary["mean"].shape == (6, 2)
    assert unconditioned_latent_summary["mean"].shape == (6, 2)


def test_toy_pit_mgrad_pg_conditional_rbpf_interval_summary_recovers_known_decay():
    conditioned = _run_toy_pg_interval_rbpf_pit_fit(enabled=True)
    unconditioned = _run_toy_pg_interval_rbpf_pit_fit(enabled=False)

    conditioned_diag = conditioned.diagnostics["pit_particle_mgrad"]
    unconditioned_diag = unconditioned.diagnostics["pit_particle_mgrad"]
    assert conditioned_diag["polya_gamma_enabled"] is True
    assert conditioned_diag["rbpf_enabled"] is True
    assert conditioned_diag["rbpf_mode"] == "conditional"
    assert conditioned_diag["rbpf_structure"] == "conditional"
    assert conditioned_diag["rbpf_carried_latent_indices"] == [0]
    assert conditioned_diag["rbpf_marginalized_latent_indices"] == [1]
    assert conditioned_diag["rbpf_gaussian_observation_channels"] == 1
    assert conditioned_diag["rbpf_polya_gamma_observation_channels"] == 1
    assert conditioned_diag["latent_kernel_algorithm"] == "conditional_aux_rbpf_csmc"
    assert conditioned_diag["latent_kernel_family"] == "auxiliary_csmc"
    assert conditioned_diag["parallel_time"] is False
    assert unconditioned_diag["polya_gamma_enabled"] is False
    assert unconditioned_diag["rbpf_enabled"] is False

    conditioned_decay = _toy_decay_posterior_mean(conditioned)
    unconditioned_decay = _toy_decay_posterior_mean(unconditioned)
    assert bool(np.all(np.abs(conditioned_decay - _TOY_TRUE_DECAY) < 0.20))
    assert bool(np.all(np.abs(unconditioned_decay - _TOY_TRUE_DECAY) < 0.20))
    assert unconditioned_diag["latent_kernel_algorithm"] == "sequential_particle_mgrad"
    assert unconditioned_diag["latent_kernel_family"] == "particle_mgrad"

    conditioned_summary = conditioned.get_latent_posterior_summary()
    unconditioned_summary = unconditioned.get_latent_posterior_summary()
    assert conditioned_summary is not None
    assert unconditioned_summary is not None
    assert conditioned_summary["mean"].shape == (12, 2)
    assert unconditioned_summary["mean"].shape == (12, 2)
    assert bool(jnp.all(jnp.isfinite(conditioned_summary["mean"])))
    assert bool(jnp.all(jnp.isfinite(unconditioned_summary["mean"])))

    conditioned_dim, conditioned_work = _toy_pg_interval_carried_work_counts(enabled=True)
    unconditioned_dim, unconditioned_work = _toy_pg_interval_carried_work_counts(enabled=False)
    assert conditioned_dim == 1
    assert unconditioned_dim == 3
    assert conditioned_work < unconditioned_work


class _ScaleAdaptationStatic:
    latent_target_accept = 0.99
    latent_min_scale = 1e-6
    latent_max_scale = 1e3
    param_min_scale = 1e-6
    param_max_scale = 1e3


def test_pit_mgrad_adaptation_keeps_positive_proposal_scales():
    latent_delta = jnp.asarray([[0.05, 0.05]], dtype=jnp.float32)
    param_step_size = jnp.asarray([0.02], dtype=jnp.float32)
    da_latent_init, _da_latent_update, _ = dual_averaging_adaptation(target=0.99)
    da_param_init, da_param_update, _ = dual_averaging_adaptation(target=0.99)
    latent_da = jax.tree_util.tree_map(
        lambda value: jnp.expand_dims(jnp.asarray(value), axis=0),
        _normalize_dual_averaging_state_shape(
            da_latent_init(latent_delta[0]),
            latent_delta[0],
        ),
    )
    states = AuxKalmanMCMCState(
        position=jnp.zeros((1, 1), dtype=jnp.float32),
        latent_context=None,
        latent_trajectory=jnp.zeros((1, 2, 1), dtype=jnp.float32),
        observation_auxiliary=None,
        trajectory_log_prob=jnp.zeros((1,), dtype=jnp.float32),
        complete_log_posterior=jnp.zeros((1,), dtype=jnp.float32),
        latent_delta=latent_delta,
        param_step_size=param_step_size,
        latent_da=latent_da,
        param_da=jax.vmap(da_param_init)(param_step_size),
    )
    latent_accept_window = jnp.zeros((1, 100, 2), dtype=jnp.float32)
    static = _ScaleAdaptationStatic()

    states, latent_accept_window, _window_accept_rate = _apply_windowed_latent_update_batched(
        states,
        jnp.asarray([[0.0, 1.0]], dtype=jnp.float32),
        latent_accept_window,
        jnp.asarray(0, dtype=jnp.int32),
        static=static,
    )
    states = _apply_param_dual_averaging_update_batched(
        states,
        jnp.asarray([0.0], dtype=jnp.float32),
        static=static,
        da_param_update=da_param_update,
        is_final_warmup=False,
    )
    states = _apply_param_dual_averaging_update_batched(
        states,
        jnp.asarray([1.0], dtype=jnp.float32),
        static=static,
        da_param_update=da_param_update,
        is_final_warmup=True,
    )

    assert bool(jnp.all(jnp.isfinite(states.latent_delta)))
    assert bool(jnp.all(states.latent_delta >= _ScaleAdaptationStatic.latent_min_scale))
    assert bool(jnp.all(jnp.isfinite(states.param_step_size)))
    assert bool(jnp.all(states.param_step_size >= _ScaleAdaptationStatic.param_min_scale))


def test_particle_mgrad_rejects_conditional_rbpf_without_auxiliary_kernel():
    observations, times = _pg_rbpf_observations_and_times()

    with pytest.raises(ValueError, match="particle_mgrad does not support conditional RBPF"):
        fit(
            SSMModel(_conditional_pg_rbpf_spec()),
            method="pit_particle_mgrad",
            observations=observations,
            times=times,
            num_warmup=1,
            num_samples=1,
            num_chains=1,
            seed=17,
            init_method="random",
            latent_init_method="predictive",
            init_scale=0.0,
            latent_delta=0.12,
            param_step_size=0.025,
            parameter_kernel="hybrid_gibbs_nuts",
            adaptation_scheme="simple",
            compute_latent_posterior_summary=False,
            enable_polya_gamma=True,
            rbpf_mode="conditional",
            rbpf_marginalized_latent_indices=(1,),
            polya_gamma_num_terms=16,
            n_particles=4,
        )


def test_independent_rbpf_allows_missing_consumed_observation_cells():
    observations, times = _pg_rbpf_observations_and_times()
    observations = observations.at[1, 1].set(jnp.nan).at[4, 1].set(jnp.nan)
    bundle = build_auxiliary_kalman_bundle(
        SSMModel(_pg_rbpf_spec()),
        observations,
        times,
        trace_key=jax.random.PRNGKey(0),
        reparam=None,
        enable_polya_gamma=True,
        rbpf_mode="independent",
        rbpf_marginalized_latent_indices=(1,),
        polya_gamma_num_terms=16,
    )
    context = bundle["latent_context_fn"](bundle["flat_example"])
    latent = bundle["initial_latent_from_context_fn"](context)
    observation_auxiliary = bundle["initial_observation_auxiliary_from_context_fn"](
        context,
        latent,
    )
    lp = bundle["trajectory_log_prob_conditioned_from_context_fn"](
        context,
        latent,
        observation_auxiliary,
    )

    assert bool(jnp.isfinite(lp))
    assert latent.shape == (6, 1)
    public_latent = bundle["public_latent_trajectory_runtime_fn"](
        context,
        latent,
        observation_auxiliary,
        observations,
        jax.random.PRNGKey(1),
    )
    assert public_latent.shape == (6, 2)
    assert bool(jnp.all(jnp.isfinite(public_latent)))


def test_conditional_pg_rbpf_allows_missing_consumed_pg_cells():
    observations, times = _pg_rbpf_observations_and_times()
    observations = observations.at[2, 0].set(jnp.nan)
    bundle = build_auxiliary_kalman_bundle(
        SSMModel(_conditional_pg_rbpf_spec()),
        observations,
        times,
        trace_key=jax.random.PRNGKey(0),
        reparam=None,
        enable_polya_gamma=True,
        rbpf_mode="conditional",
        rbpf_marginalized_latent_indices=(1,),
        polya_gamma_num_terms=16,
    )
    context = bundle["latent_context_fn"](bundle["flat_example"])
    latent = bundle["initial_latent_from_context_fn"](context)
    observation_auxiliary = bundle["initial_observation_auxiliary_from_context_fn"](
        context,
        latent,
    )
    lp = bundle["trajectory_log_prob_conditioned_from_context_fn"](
        context,
        latent,
        observation_auxiliary,
    )

    assert bool(jnp.isfinite(lp))
    public_latent = bundle["public_latent_trajectory_runtime_fn"](
        context,
        latent,
        observation_auxiliary,
        observations,
        jax.random.PRNGKey(2),
    )
    assert public_latent.shape == (6, 2)
    assert bool(jnp.all(jnp.isfinite(public_latent)))


def test_rbpf_with_linear_summary_augmentation_reconstructs_original_latents():
    observations = jnp.asarray(
        [
            [0.0, jnp.nan],
            [1.0, 0.10],
            [1.0, jnp.nan],
        ],
        dtype=jnp.float32,
    )
    times = jnp.arange(observations.shape[0], dtype=jnp.float32)
    model = SSMModel(_pg_rbpf_spec())
    model.set_observation_support(_interval_score_support_runtime())

    bundle = build_auxiliary_kalman_bundle(
        model,
        observations,
        times,
        trace_key=jax.random.PRNGKey(0),
        reparam=None,
        enable_polya_gamma=True,
        rbpf_mode="independent",
        rbpf_marginalized_latent_indices=(1,),
        polya_gamma_num_terms=16,
    )
    context = bundle["latent_context_fn"](bundle["flat_example"])
    latent = bundle["initial_latent_from_context_fn"](context)
    observation_auxiliary = bundle["initial_observation_auxiliary_from_context_fn"](
        context,
        latent,
    )
    lp = bundle["trajectory_log_prob_conditioned_from_context_fn"](
        context,
        latent,
        observation_auxiliary,
    )
    public_latent = bundle["public_latent_trajectory_runtime_fn"](
        context,
        latent,
        observation_auxiliary,
        observations,
        jax.random.PRNGKey(3),
    )

    assert bool(jnp.isfinite(lp))
    assert latent.shape == (3, 1)
    assert context.rbpf_marginal_context.Ad_mm.shape[-1] == 2
    assert public_latent.shape == (3, 2)
    assert bool(jnp.all(jnp.isfinite(public_latent)))


def _pg_rbpf_carried_work_counts(*, enabled: bool) -> tuple[int, int, int]:
    observations, times = _pg_rbpf_observations_and_times()
    bundle = build_auxiliary_kalman_bundle(
        SSMModel(_pg_rbpf_spec()),
        observations,
        times,
        trace_key=jax.random.PRNGKey(0),
        reparam=None,
        enable_polya_gamma=enabled,
        rbpf_mode="independent" if enabled else "none",
        rbpf_marginalized_latent_indices=(1,) if enabled else None,
        polya_gamma_num_terms=16,
    )
    context = bundle["latent_context_fn"](bundle["flat_example"])
    latent = bundle["initial_latent_from_context_fn"](context)
    observation_auxiliary = bundle["initial_observation_auxiliary_from_context_fn"](
        context,
        latent,
    )

    del observation_auxiliary
    return (
        latent.shape[-1],
        latent.size,
        context.Ad.size + context.Qd.size,
    )


def test_pg_plus_independent_rbpf_has_lower_carried_state_work():
    conditioned_dim, conditioned_latent_work, conditioned_transition_work = (
        _pg_rbpf_carried_work_counts(enabled=True)
    )
    unconditioned_dim, unconditioned_latent_work, unconditioned_transition_work = (
        _pg_rbpf_carried_work_counts(enabled=False)
    )

    assert conditioned_dim == 1
    assert unconditioned_dim == 2
    assert conditioned_latent_work < unconditioned_latent_work
    assert conditioned_transition_work < unconditioned_transition_work


def test_runtime_conditioning_metadata_marks_pg_channels_and_full_path_rbpf():
    spec = SimpleNamespace(
        n_latent=2,
        n_manifest=3,
        manifest_names=["clicked", "events", "score"],
        manifest_dists=[
            DistributionFamily.BERNOULLI,
            DistributionFamily.NEGATIVE_BINOMIAL,
            DistributionFamily.GAUSSIAN,
        ],
        manifest_links=[LinkFunction.LOGIT, LinkFunction.LOG, LinkFunction.IDENTITY],
    )

    metadata = compile_runtime_conditioning_metadata(spec)

    assert metadata["polya_gamma"]["channels"] == [
        {
            "index": 0,
            "name": "clicked",
            "distribution": "bernoulli",
            "link": "logit",
        },
        {
            "index": 1,
            "name": "events",
            "distribution": "negative_binomial",
            "link": "log",
        },
    ]
    assert metadata["polya_gamma"]["default_sampler"] == "truncated_sum"
    assert metadata["polya_gamma"]["supported_samplers"] == [
        "truncated_sum",
        "devroye",
        "devroye_integer",
    ]
    assert metadata["rbpf"] == {
        "active": False,
        "mode": "none",
        "structure": "none",
        "supported_modes": ["none", "independent", "conditional"],
        "supported_structures": ["independent", "conditional"],
        "carried_latent_indices": [0, 1],
        "marginalized_latent_indices": [],
        "consumed_observation_family": "gaussian_identity_or_pg_conditioned_affine_logit",
    }


def test_rbpf_partition_accepts_independent_marginalized_latents():
    validate_rbpf_partition(full_path_rbpf_partition(2), n_latent=2)
    partition = build_rbpf_partition(2, (1,))
    validate_rbpf_partition(partition, n_latent=2)

    plan = build_gaussian_rbpf_observation_plan(
        _pg_rbpf_spec(),
        partition,
        [LinkFunction.LOGIT, LinkFunction.IDENTITY],
        jnp.asarray([True, False]),
    )
    assert plan.enabled is True
    assert plan.structure == "independent"
    assert plan.channel_mask.tolist() == [False, True]
    assert plan.gaussian_channel_mask.tolist() == [False, True]
    assert plan.polya_gamma_channel_mask.tolist() == [False, False]


def test_rbpf_partition_accepts_conditional_mixed_gaussian_rows():
    partition = RBPFPartitionSpec(carried_latent_indices=(0,), marginalized_latent_indices=(1,))
    plan = build_gaussian_rbpf_observation_plan(
        _conditional_gaussian_rbpf_spec(),
        partition,
        [LinkFunction.LOGIT, LinkFunction.IDENTITY],
        jnp.asarray([True, False]),
    )
    assert plan.enabled is True
    assert plan.structure == "conditional"
    assert plan.channel_mask.tolist() == [False, True]
    assert plan.gaussian_channel_mask.tolist() == [False, True]
    assert plan.polya_gamma_channel_mask.tolist() == [False, False]


def test_rbpf_partition_accepts_pg_conditioned_marginalized_logit_rows():
    partition = RBPFPartitionSpec(carried_latent_indices=(0,), marginalized_latent_indices=(1,))
    plan = build_gaussian_rbpf_observation_plan(
        _conditional_pg_rbpf_spec(),
        partition,
        [LinkFunction.LOGIT, LinkFunction.IDENTITY],
        jnp.asarray([True, False]),
    )
    assert plan.enabled is True
    assert plan.structure == "conditional"
    assert plan.channel_mask.tolist() == [True, False]
    assert plan.gaussian_channel_mask.tolist() == [False, False]
    assert plan.polya_gamma_channel_mask.tolist() == [True, False]


def test_rbpf_partition_accepts_pg_conditioned_negative_binomial_rows():
    partition = RBPFPartitionSpec(carried_latent_indices=(0,), marginalized_latent_indices=(1,))
    plan = build_gaussian_rbpf_observation_plan(
        _conditional_negative_binomial_pg_rbpf_spec(),
        partition,
        [LinkFunction.LOG, LinkFunction.IDENTITY],
        jnp.asarray([True, False]),
    )
    assert plan.enabled is True
    assert plan.structure == "conditional"
    assert plan.channel_mask.tolist() == [True, False]
    assert plan.gaussian_channel_mask.tolist() == [False, False]
    assert plan.polya_gamma_channel_mask.tolist() == [True, False]
    assert plan.negative_binomial_polya_gamma_channel_mask.tolist() == [True, False]


def test_derive_conditional_rbpf_partition_forces_residual_rows_and_dynamics_closure():
    spec = _residual_dynamics_closure_rbpf_spec()
    partition, diagnostics = derive_rbpf_partition(
        spec=spec,
        rbpf_mode="conditional",
        marginalized_latent_indices=None,
        manifest_links=spec.manifest_links,
        polya_gamma_channel_mask=jnp.asarray([False, False, False]),
    )
    plan = build_gaussian_rbpf_observation_plan(
        spec,
        partition,
        spec.manifest_links,
        jnp.asarray([False, False, False]),
    )
    validate_rbpf_mode("conditional", partition, plan)

    assert partition.carried_latent_indices == (0, 1)
    assert partition.marginalized_latent_indices == (2,)
    assert plan.structure == "conditional"
    assert plan.channel_mask.tolist() == [False, True, False]
    assert diagnostics.residual_observation_channels == (0,)
    forced_reasons = {item["latent_index"]: item["reason"] for item in diagnostics.forced_carried}
    assert forced_reasons[0] == "residual_observation"
    assert forced_reasons[1] == "dynamics_to_carried"


def test_derive_independent_rbpf_partition_forces_conditional_dependencies_carried():
    spec = _residual_dynamics_closure_rbpf_spec()
    partition, diagnostics = derive_rbpf_partition(
        spec=spec,
        rbpf_mode="independent",
        marginalized_latent_indices=None,
        manifest_links=spec.manifest_links,
        polya_gamma_channel_mask=jnp.asarray([False, False, False]),
    )
    plan = build_gaussian_rbpf_observation_plan(
        spec,
        partition,
        spec.manifest_links,
        jnp.asarray([False, False, False]),
    )
    validate_rbpf_mode("independent", partition, plan)

    assert partition.carried_latent_indices == (0, 1, 2)
    assert partition.marginalized_latent_indices == ()
    assert plan.structure == "none"
    forced_reasons = {item["latent_index"]: item["reason"] for item in diagnostics.forced_carried}
    assert forced_reasons[2] == "independent_mode_dynamics_dependency"


def test_runtime_bundle_derives_rbpf_partition_when_indices_are_omitted():
    observations, times = _pg_rbpf_observations_and_times()
    bundle = build_auxiliary_kalman_bundle(
        SSMModel(_pg_rbpf_spec()),
        observations,
        times,
        trace_key=jax.random.PRNGKey(0),
        reparam=None,
        enable_polya_gamma=True,
        rbpf_mode="independent",
        polya_gamma_num_terms=16,
    )

    assert bundle["rbpf_enabled"] is True
    assert bundle["rbpf_partition"].carried_latent_indices == (0,)
    assert bundle["rbpf_partition"].marginalized_latent_indices == (1,)
    assert bundle["rbpf_partition_diagnostics"]["candidate_marginalized_latent_indices"] == [0, 1]
    forced = bundle["rbpf_partition_diagnostics"]["forced_carried"]
    assert forced == [
        {
            "latent_index": 0,
            "latent_name": "engagement",
            "reason": "runtime_requires_carried_latent",
        }
    ]


def test_rbpf_negative_binomial_pg_uses_current_obs_r_in_marginal_likelihood():
    partition = RBPFPartitionSpec(carried_latent_indices=(0,), marginalized_latent_indices=(1,))
    spec = _conditional_negative_binomial_pg_rbpf_spec()
    rbpf_plan = build_gaussian_rbpf_observation_plan(
        spec,
        partition,
        [LinkFunction.LOG, LinkFunction.IDENTITY],
        jnp.asarray([True, False]),
    )
    pg_plan = build_polya_gamma_observation_plan(
        [DistributionFamily.NEGATIVE_BINOMIAL, DistributionFamily.GAUSSIAN],
        [LinkFunction.LOG, LinkFunction.IDENTITY],
        num_terms=8,
    )
    H = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)
    d_meas = jnp.zeros((2,), dtype=jnp.float32)
    observations = jnp.asarray([[2.0, jnp.nan], [4.0, jnp.nan]], dtype=jnp.float32)
    latent = jnp.asarray([[0.0, 0.2], [0.0, -0.1]], dtype=jnp.float32)
    context_r3 = SimpleNamespace(
        H=H,
        d_meas=d_meas,
        H_rows=None,
        d_rows=None,
        extra_params={"obs_r": jnp.asarray(3.0, dtype=jnp.float32)},
    )
    observation_auxiliary = refresh_polya_gamma_auxiliary_state(
        jax.random.PRNGKey(42),
        pg_plan,
        context_r3,
        latent,
        observations,
    )
    T = observations.shape[0]
    Ad = jnp.broadcast_to(jnp.eye(2, dtype=jnp.float32), (T, 2, 2))
    Qd = jnp.broadcast_to(0.15 * jnp.eye(2, dtype=jnp.float32), (T, 2, 2))
    cd = jnp.zeros((T, 2), dtype=jnp.float32)
    init_mean = jnp.zeros((2,), dtype=jnp.float32)
    init_cov = jnp.eye(2, dtype=jnp.float32)
    R = jnp.zeros((2, 2), dtype=jnp.float32)
    carried = latent[:, :1]
    rbpf_context_r3 = build_rbpf_marginal_context(
        Ad=Ad,
        Qd=Qd,
        cd=cd,
        init_mean=init_mean,
        init_cov=init_cov,
        H=H,
        d_meas=d_meas,
        R=R,
        partition=partition,
        observation_plan=rbpf_plan,
        extra_params=context_r3.extra_params,
    )
    rbpf_context_r7 = build_rbpf_marginal_context(
        Ad=Ad,
        Qd=Qd,
        cd=cd,
        init_mean=init_mean,
        init_cov=init_cov,
        H=H,
        d_meas=d_meas,
        R=R,
        partition=partition,
        observation_plan=rbpf_plan,
        extra_params={"obs_r": jnp.asarray(7.0, dtype=jnp.float32)},
    )

    ll_r3 = rbpf_marginal_log_likelihood(
        rbpf_context_r3,
        observations,
        observation_auxiliary,
        carried,
    )
    ll_r7 = rbpf_marginal_log_likelihood(
        rbpf_context_r7,
        observations,
        observation_auxiliary,
        carried,
    )
    assert bool(jnp.isfinite(ll_r3))
    assert bool(jnp.isfinite(ll_r7))
    assert float(ll_r7) != pytest.approx(float(ll_r3))


def test_conditional_rbpf_mode_accepts_validated_independent_structure():
    partition = RBPFPartitionSpec(carried_latent_indices=(0,), marginalized_latent_indices=(1,))
    plan = build_gaussian_rbpf_observation_plan(
        _pg_rbpf_spec(),
        partition,
        [LinkFunction.LOGIT, LinkFunction.IDENTITY],
        jnp.asarray([True, False]),
    )

    validate_rbpf_mode("conditional", partition, plan)


def test_rbpf_mode_must_match_validated_conditional_structure():
    partition = RBPFPartitionSpec(carried_latent_indices=(0,), marginalized_latent_indices=(1,))
    plan = build_gaussian_rbpf_observation_plan(
        _conditional_gaussian_rbpf_spec(),
        partition,
        [LinkFunction.LOGIT, LinkFunction.IDENTITY],
        jnp.asarray([True, False]),
    )

    with pytest.raises(ValueError, match="does not match"):
        validate_rbpf_mode("independent", partition, plan)


def test_rbpf_partition_rejects_nongaussian_marginalized_rows():
    spec = _pg_rbpf_spec()
    invalid_spec = block_ssm_spec(
        n_latent=2,
        n_manifest=2,
        dynamics_spec=spec.dynamics_spec,
        diffusion_block=spec.diffusion_block,
        lambda_block=SparseMatrixBlockSpec(
            n_rows=2,
            n_cols=2,
            free_support=np.zeros((2, 2), dtype=bool),
            template=jnp.asarray([[1.0, 1.0], [0.0, 1.0]], dtype=jnp.float32),
            free_site_name="lambda_free",
            det_site_name="lambda",
            support=SupportClass.REAL,
            site_kind=SiteKind.LOADING,
            assembly_group="lambda",
            fixed_spec_field="lambda_mat",
            priors_field="lambda_free",
        ),
        manifest_means_block=spec.manifest_means_block,
        manifest_chol_block=spec.manifest_chol_block,
        t0_means_block=spec.t0_means_block,
        t0_chol_block=spec.t0_chol_block,
        input_effect_block=spec.input_effect_block,
        static_state_sd_block=spec.static_state_sd_block,
        manifest_dists=[DistributionFamily.BERNOULLI, DistributionFamily.GAUSSIAN],
        manifest_links=spec.manifest_links,
    )
    with pytest.raises(NotImplementedError, match="Gaussian identity"):
        build_gaussian_rbpf_observation_plan(
            invalid_spec,
            RBPFPartitionSpec(carried_latent_indices=(0,), marginalized_latent_indices=(1,)),
            [LinkFunction.LOGIT, LinkFunction.IDENTITY],
            jnp.asarray([False, False]),
        )
