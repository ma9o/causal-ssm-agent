"""Synthetic nonlinear SSM fixture for PG/RBPF recovery notebooks."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.artifacts.model_spec import DistributionFamily, LinkFunction
from nof1_causal_lab.distributions import PriorDistributionFamily
from nof1_causal_lab.models.ssm.dynamics.spec import (
    DynamicsSpec,
    HillEdgeSpec,
    MultiplicativeEdgeSpec,
    StateDecaySpec,
)
from nof1_causal_lab.models.ssm.model import SSMModel
from nof1_causal_lab.models.ssm.priors import (
    PriorRegistry,
    PriorSpec,
    default_prior_for_descriptor,
)
from nof1_causal_lab.models.ssm.structure import (
    DiffusionBlockSpec,
    ManifestCholBlockSpec,
    SparseMatrixBlockSpec,
    SparseVectorBlockSpec,
    T0CholBlockSpec,
)
from nof1_causal_lab.models.ssm.structure.sites import SiteKind, SupportClass
from nof1_causal_lab.models.ssm.testing import block_ssm_spec, make_observation_support_runtime

LATENT_NAMES = [
    "affective_state",
    "sleep_quality",
    "physical_activity",
]

INPUT_NAMES = [
    "serotonergic_exposure",
    "seasonal_load",
    "prescription_event",
    "adherence",
    "cyp2c19_metabolizer_status",
]

MANIFEST_NAMES = [
    "state_of_mind_valence",
    "late_night_message_count",
    "total_sleep_hours",
    "sleep_efficiency_pct",
    "sleep_onset_latency_min",
    "nightly_hrv_ms",
    "daily_step_count",
    "exercise_minutes",
    "journal_affect_severity",
    "journal_rumination_intensity",
    "gp_clinical_severity",
]

MANIFEST_DISTS = [
    DistributionFamily.GAUSSIAN,
    DistributionFamily.NEGATIVE_BINOMIAL,
    DistributionFamily.GAUSSIAN,
    DistributionFamily.GAUSSIAN,
    DistributionFamily.GAMMA,
    DistributionFamily.GAUSSIAN,
    DistributionFamily.NEGATIVE_BINOMIAL,
    DistributionFamily.GAUSSIAN,
    DistributionFamily.GAUSSIAN,
    DistributionFamily.GAUSSIAN,
    DistributionFamily.GAUSSIAN,
]

MANIFEST_LINKS = [
    LinkFunction.IDENTITY,
    LinkFunction.LOG,
    LinkFunction.IDENTITY,
    LinkFunction.IDENTITY,
    LinkFunction.LOG,
    LinkFunction.IDENTITY,
    LinkFunction.LOG,
    LinkFunction.IDENTITY,
    LinkFunction.IDENTITY,
    LinkFunction.IDENTITY,
    LinkFunction.IDENTITY,
]

TRUE_DECAY = np.asarray([0.28, 0.35, 0.32], dtype=np.float32)
TRUE_HILL_BY_SITE = {
    "vf_3_Emax": 0.06,  # diminishing returns: sleep_quality -> affective_state
    "vf_3_EC50": 0.70,
    "vf_3_n": 2.0,
    "vf_4_Emax": 0.045,  # diminishing returns: physical_activity -> affective_state
    "vf_4_EC50": 0.75,
    "vf_4_n": 2.0,
    "vf_5_Emax": 0.035,  # positive mood improves sleep with saturation
    "vf_5_EC50": 0.80,
    "vf_5_n": 2.0,
    "vf_6_Emax": 0.030,  # behavioral activation saturates
    "vf_6_EC50": 0.80,
    "vf_6_n": 2.0,
}
PINNED_HILL_SHAPE_BY_SITE = {
    name: value
    for name, value in TRUE_HILL_BY_SITE.items()
    if name.endswith("_EC50") or name.endswith("_n")
}
TRUE_MULTIPLICATIVE_BY_SITE = {
    "vf_7_weight": 0.025,  # sleep and activity reinforce mood together
    "vf_8_weight": 0.020,  # mood and sleep jointly support activity
}
TRUE_OBS_R = 12.0
TRUE_OBS_SHAPE = 8.0

TRUE_DRIFT = np.asarray(
    [
        [-0.28, 0.00, 0.00],
        [0.00, -0.35, 0.00],
        [0.00, 0.00, -0.32],
    ],
    dtype=np.float32,
)

TRUE_DIFFUSION_SD = np.asarray([0.18, 0.16, 0.20], dtype=np.float32)
TRUE_T0_MEAN = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
TRUE_T0_SD = np.asarray([0.55, 0.50, 0.55], dtype=np.float32)
FIXED_CYP2C19_VALUE = np.float32(0.5)

TRUE_INPUT_EFFECT = np.asarray(
    [
        [0.08, 0.03, 0.00, 0.00, 0.00],
        [0.02, 0.00, 0.00, 0.00, 0.00],
        [0.04, 0.00, 0.00, 0.00, 0.00],
    ],
    dtype=np.float32,
)
TRUE_INPUT_EFFECT_POSITIONS = (
    (0, 0),  # serotonergic_exposure -> affective_state
    (0, 1),  # seasonal_load -> affective_state
    (1, 0),  # serotonergic_exposure -> sleep_quality
    (2, 0),  # serotonergic_exposure -> physical_activity
)

TRUE_LOADINGS = np.asarray(
    [
        [0.75, 0.00, 0.00],
        [-0.35, 0.00, 0.00],
        [0.00, 0.35, 0.00],
        [0.00, 2.50, 0.00],
        [0.00, -0.25, 0.00],
        [0.00, 3.00, 0.00],
        [0.00, 0.00, 0.25],
        [0.00, 0.00, 5.00],
        [-0.70, 0.00, 0.00],
        [-1.00, 0.00, 0.00],
        [-1.20, 0.00, 0.00],
    ],
    dtype=np.float32,
)

TRUE_MANIFEST_MEANS = np.asarray(
    [
        0.20,
        np.log(2.0),
        7.10,
        87.0,
        np.log(18.0),
        45.0,
        np.log(80.0),
        28.0,
        2.0,
        4.0,
        5.0,
    ],
    dtype=np.float32,
)

TRUE_MANIFEST_SD = np.asarray(
    [0.25, 1.0, 0.20, 1.00, 1.0, 1.50, 1.0, 2.00, 0.30, 0.40, 0.35],
    dtype=np.float32,
)

EXACT_MEASUREMENT_MANIFEST_INDICES = tuple(
    idx
    for idx, (dist, link) in enumerate(zip(MANIFEST_DISTS, MANIFEST_LINKS, strict=True))
    if (dist == DistributionFamily.GAUSSIAN and link == LinkFunction.IDENTITY)
    or (dist == DistributionFamily.NEGATIVE_BINOMIAL and link == LinkFunction.LOG)
)
EXACT_MEASUREMENT_SUPPORT = np.asarray(
    [idx in EXACT_MEASUREMENT_MANIFEST_INDICES for idx in range(len(MANIFEST_NAMES))],
    dtype=bool,
)
# Gaussian/identity manifest means are PINNED at truth: their raw-scale
# locations (e.g. sleep_efficiency_pct ~ 87) sit tens of prior sd outside the
# canonical Normal(0, 2) manifest-mean prior, so leaving them free makes the
# truth unreachable and distorts every coupled posterior coordinate. Production
# removes the raw location via deterministic centering (prepare_model_runtime);
# this fixture bypasses that path, so the location is pinned instead — it is
# not a recovery axis here. Count intercepts (NegBin/log) stay free: their
# log-scale truths sit within the canonical prior.
MEASUREMENT_MEANS_FREE_SUPPORT = np.asarray(
    [
        MANIFEST_DISTS[idx] == DistributionFamily.NEGATIVE_BINOMIAL
        and MANIFEST_LINKS[idx] == LinkFunction.LOG
        for idx in range(len(MANIFEST_NAMES))
    ],
    dtype=bool,
)
ANCHOR_LOADING_POSITIONS = (
    (0, 0),  # state_of_mind_valence anchors affective_state
    (2, 1),  # total_sleep_hours anchors sleep_quality
    (6, 2),  # daily_step_count anchors physical_activity
)
MEASUREMENT_LOADINGS_FREE_SUPPORT = (
    (~np.isclose(TRUE_LOADINGS, 0.0))
    & EXACT_MEASUREMENT_SUPPORT[:, None]
    & np.asarray(
        [
            [(row, col) not in ANCHOR_LOADING_POSITIONS for col in range(TRUE_LOADINGS.shape[1])]
            for row in range(TRUE_LOADINGS.shape[0])
        ],
        dtype=bool,
    )
)
MEASUREMENT_MEANS_FREE_POSITIONS = tuple(
    idx for idx in range(len(MANIFEST_NAMES)) if bool(MEASUREMENT_MEANS_FREE_SUPPORT[idx])
)
MEASUREMENT_LOADINGS_FREE_POSITIONS = tuple(
    (row, col)
    for row in range(TRUE_LOADINGS.shape[0])
    for col in range(TRUE_LOADINGS.shape[1])
    if bool(MEASUREMENT_LOADINGS_FREE_SUPPORT[row, col])
)


@dataclass(frozen=True)
class SyntheticNonlinearData:
    observations: jnp.ndarray
    times: jnp.ndarray
    latent: jnp.ndarray
    transition_inputs: jnp.ndarray
    observation_support: object


def _synthetic_nonlinear_dynamics_spec() -> DynamicsSpec:
    return DynamicsSpec(
        n_latent=3,
        components=(
            StateDecaySpec(target=0),
            StateDecaySpec(target=1),
            StateDecaySpec(target=2),
            HillEdgeSpec(
                source=1,
                target=0,
                fixed_ec50=TRUE_HILL_BY_SITE["vf_3_EC50"],
                fixed_n=TRUE_HILL_BY_SITE["vf_3_n"],
            ),
            HillEdgeSpec(
                source=2,
                target=0,
                fixed_ec50=TRUE_HILL_BY_SITE["vf_4_EC50"],
                fixed_n=TRUE_HILL_BY_SITE["vf_4_n"],
            ),
            HillEdgeSpec(
                source=0,
                target=1,
                fixed_ec50=TRUE_HILL_BY_SITE["vf_5_EC50"],
                fixed_n=TRUE_HILL_BY_SITE["vf_5_n"],
            ),
            HillEdgeSpec(
                source=0,
                target=2,
                fixed_ec50=TRUE_HILL_BY_SITE["vf_6_EC50"],
                fixed_n=TRUE_HILL_BY_SITE["vf_6_n"],
            ),
            MultiplicativeEdgeSpec(source_a=1, source_b=2, target=0),
            MultiplicativeEdgeSpec(source_a=0, source_b=1, target=2),
        ),
    )


def build_synthetic_nonlinear_spec():
    input_support = np.zeros((3, len(INPUT_NAMES)), dtype=bool)
    for row, col in TRUE_INPUT_EFFECT_POSITIONS:
        input_support[row, col] = True
    return block_ssm_spec(
        n_latent=3,
        n_manifest=11,
        dynamics_spec=_synthetic_nonlinear_dynamics_spec(),
        diffusion_block=DiffusionBlockSpec(
            n_latent=3,
            # Process-noise SDs FREE (estimated) — numerically stable. The dynamics
            # regime is set by a regime-scaled PRIOR in build_synthetic_nonlinear_
            # priors, not by pinning the value (which overflowed predictive init at
            # T=1000). The data still uses the scaled true noise (simulate(...)).
            diffusion_chol_support=np.eye(3, dtype=bool),
            diffusion_chol_template=jnp.diag(jnp.asarray(TRUE_DIFFUSION_SD)),
        ),
        lambda_block=SparseMatrixBlockSpec(
            n_rows=11,
            n_cols=3,
            free_support=MEASUREMENT_LOADINGS_FREE_SUPPORT,
            template=jnp.asarray(TRUE_LOADINGS),
            free_site_name="lambda_free",
            det_site_name="lambda",
            support=SupportClass.REAL,
            site_kind=SiteKind.LOADING,
            assembly_group="lambda",
            fixed_spec_field="lambda_mat",
            priors_field="lambda_free",
        ),
        manifest_means_block=SparseVectorBlockSpec(
            n=11,
            free_support=MEASUREMENT_MEANS_FREE_SUPPORT,
            template=jnp.asarray(TRUE_MANIFEST_MEANS),
            free_site_name="manifest_means_free",
            det_site_name="manifest_means",
            support=SupportClass.REAL,
            site_kind=SiteKind.MANIFEST_MEANS,
            assembly_group="manifest",
            fixed_spec_field="manifest_means",
            priors_field="manifest_means",
        ),
        manifest_chol_block=ManifestCholBlockSpec(
            n_manifest=11,
            # Measurement-noise SDs stay FREE (estimated). Only the process noise
            # (diffusion) is fixed for the clean dynamics-axis test; fixing the
            # measurement SDs too made the tight observation likelihood numerically
            # brittle (all-particle -inf -> nan) under predictive init.
            diag_support=np.ones(11, dtype=bool),
            template=jnp.diag(jnp.asarray(TRUE_MANIFEST_SD)),
        ),
        t0_means_block=SparseVectorBlockSpec(
            n=3,
            free_support=np.zeros(3, dtype=bool),
            template=jnp.asarray(TRUE_T0_MEAN),
            free_site_name="t0_means_free",
            det_site_name="t0_means",
            support=SupportClass.REAL,
            site_kind=SiteKind.T0_MEANS,
            assembly_group="t0",
            fixed_spec_field="t0_means",
            priors_field="t0_means",
        ),
        t0_chol_block=T0CholBlockSpec(
            n_latent=3,
            diag_support=np.zeros(3, dtype=bool),
            correlation_support=np.zeros((3, 3), dtype=bool),
            template=jnp.diag(jnp.asarray(TRUE_T0_SD)),
        ),
        input_effect_block=SparseMatrixBlockSpec(
            n_rows=3,
            n_cols=len(INPUT_NAMES),
            free_support=input_support,
            template=jnp.zeros((3, len(INPUT_NAMES)), dtype=jnp.float32),
            free_site_name="input_effect_free",
            det_site_name="input_effect",
            support=SupportClass.REAL,
            site_kind=SiteKind.INPUT_EFFECT,
            assembly_group="input_effect",
            fixed_spec_field="input_effect",
            priors_field="input_effect",
        ),
        manifest_dists=MANIFEST_DISTS,
        manifest_links=MANIFEST_LINKS,
        manifest_names=MANIFEST_NAMES,
        latent_names=LATENT_NAMES,
        input_names=INPUT_NAMES,
    )


def build_synthetic_nonlinear_priors(spec, diffusion_scale: float = 1.0) -> PriorRegistry:
    """Honest off-truth priors with a REGIME-SCALED process-noise prior.

    Diffusion stays free/estimated (numerically stable, unlike pinning the tight
    true value which overflowed predictive init at T=1000). Its prior width scales
    with the regime, ``HalfNormal(0.4 * diffusion_scale)``, so the analyst's prior
    is appropriately scaled to each dynamics regime — removing the cross-regime
    mis-scaling confound that made a single fixed-width prior non-monotonic across
    scales. Structural sites use the canonical defaults; observation-dispersion
    params (NegBin ``r``, Gamma ``shape``) use off-truth ``LogNormal`` priors with
    negligible mass near zero so they cannot collapse to degenerate overdispersion.
    Free loadings use a widened ``Normal(0, 2.5)``: the true free loadings reach
    5.0, which sits 9 prior sd outside the canonical ``Normal(0.5, 0.5)`` — an
    unreachable truth that poisoned recovery the same way the raw-scale manifest
    means did before they were pinned.
    """
    priors: dict[str, PriorSpec] = {
        site.name: default_prior_for_descriptor(site) for site in spec.iter_sample_sites()
    }
    priors["lambda_free"] = PriorSpec(
        PriorDistributionFamily.NORMAL,
        {"mu": 0.0, "sigma": 2.5},
    )
    priors["diffusion_diag_free"] = PriorSpec(
        PriorDistributionFamily.HALF_NORMAL,
        {"sigma": 0.4 * float(diffusion_scale)},
    )
    priors["obs_r"] = PriorSpec(
        PriorDistributionFamily.LOG_NORMAL,
        {"mu": float(np.log(6.0)), "sigma": 0.6},
    )
    priors["obs_shape"] = PriorSpec(
        PriorDistributionFamily.LOG_NORMAL,
        {"mu": float(np.log(5.0)), "sigma": 0.6},
    )
    return PriorRegistry(priors)


def build_synthetic_nonlinear_model(
    data: SyntheticNonlinearData | None = None,
    *,
    include_interval_support: bool = False,
    diffusion_scale: float = 1.0,
) -> SSMModel:
    spec = build_synthetic_nonlinear_spec()
    model = SSMModel(
        spec,
        priors=build_synthetic_nonlinear_priors(spec, diffusion_scale=diffusion_scale),
    )
    if data is not None:
        model.set_transition_inputs(data.transition_inputs)
        if include_interval_support:
            model.set_observation_support(data.observation_support)
    return model


def _build_transition_inputs(T: int) -> np.ndarray:
    t = np.arange(T, dtype=np.float32)
    serotonergic = 1.0 / (1.0 + np.exp(-(t - 8.0) / 2.0))
    seasonal = np.sin(2.0 * np.pi * t / max(T - 1, 1))
    prescription_event = np.zeros(T, dtype=np.float32)
    adherence = np.ones(T, dtype=np.float32)
    cyp2c19 = np.full(T, FIXED_CYP2C19_VALUE, dtype=np.float32)
    for event_idx, event_size in ((8, 1.0), (17, 0.8), (25, 0.6)):
        if event_idx < T:
            prescription_event[event_idx] = event_size
    for idx in range(1, T):
        prescription_event[idx] = max(prescription_event[idx], 0.55 * prescription_event[idx - 1])
        if idx % 19 == 0:
            adherence[idx : min(idx + 2, T)] = 0.0
    return np.column_stack([serotonergic, seasonal, prescription_event, adherence, cyp2c19]).astype(
        np.float32
    )


def _build_gp_interval_support(T: int, gp_rows: np.ndarray, window: int = 3):
    n_manifest = len(MANIFEST_NAMES)
    support_start_times = np.full((T, n_manifest), np.nan, dtype=np.float64)
    support_end_times = np.full((T, n_manifest), np.nan, dtype=np.float64)
    interval_prev_coeffs = np.zeros((T, n_manifest), dtype=np.float64)
    interval_curr_coeffs = np.zeros((T, n_manifest), dtype=np.float64)
    interval_weights = np.zeros((T, n_manifest), dtype=np.float64)
    emission_slot_indices = np.full((T, n_manifest), -1, dtype=np.int64)
    gp_idx = MANIFEST_NAMES.index("gp_clinical_severity")

    for end in gp_rows:
        start = max(0, int(end) - window)
        support_start_times[end, gp_idx] = float(start)
        support_end_times[end, gp_idx] = float(end)
        emission_slot_indices[end, gp_idx] = 0
        for step in range(start + 1, int(end) + 1):
            interval_prev_coeffs[step, gp_idx] += 0.5
            interval_curr_coeffs[step, gp_idx] += 0.5
            interval_weights[step, gp_idx] += 1.0

    support_kinds = [None] * n_manifest
    summary_operators = ["last"] * n_manifest
    observation_windows = [None] * n_manifest
    support_kinds[gp_idx] = "interval"
    summary_operators[gp_idx] = "mean"
    observation_windows[gp_idx] = f"{window}d"

    return make_observation_support_runtime(
        anchor_times=np.arange(T, dtype=np.float64),
        manifest_names=MANIFEST_NAMES,
        support_kinds=support_kinds,
        summary_operators=summary_operators,
        observation_windows=observation_windows,
        support_start_times=support_start_times,
        support_end_times=support_end_times,
        interval_prev_coeffs=interval_prev_coeffs,
        interval_curr_coeffs=interval_curr_coeffs,
        interval_weights=interval_weights,
        emission_slot_indices=emission_slot_indices,
    )


def _sample_negative_binomial(rng: np.random.Generator, mean: np.ndarray, r: float) -> np.ndarray:
    gamma_rate = rng.gamma(shape=r, scale=mean / r)
    return rng.poisson(gamma_rate).astype(np.float32)


def _hill_effect(source_value: np.ndarray | float, prefix: str) -> np.ndarray | float:
    x = np.maximum(source_value, 0.0)
    emax = TRUE_HILL_BY_SITE[f"{prefix}_Emax"]
    ec50 = TRUE_HILL_BY_SITE[f"{prefix}_EC50"]
    n_hill = TRUE_HILL_BY_SITE[f"{prefix}_n"]
    x_n = x**n_hill
    return emax * x_n / (ec50**n_hill + x_n + 1e-12)


def _synthetic_nonlinear_drift(state: np.ndarray, transition_input: np.ndarray) -> np.ndarray:
    drift = TRUE_DRIFT @ state + TRUE_INPUT_EFFECT @ transition_input
    drift = drift.copy()
    drift[0] += _hill_effect(state[1], "vf_3")
    drift[0] += _hill_effect(state[2], "vf_4")
    drift[1] += _hill_effect(state[0], "vf_5")
    drift[2] += _hill_effect(state[0], "vf_6")
    drift[0] += TRUE_MULTIPLICATIVE_BY_SITE["vf_7_weight"] * state[1] * state[2]
    drift[2] += TRUE_MULTIPLICATIVE_BY_SITE["vf_8_weight"] * state[0] * state[1]
    return drift


def _sample_nonlinear_transition(
    rng: np.random.Generator,
    state: np.ndarray,
    transition_input: np.ndarray,
    *,
    diffusion_sd: np.ndarray,
    dt: float = 1.0,
    substeps: int = 8,
) -> np.ndarray:
    next_state = np.asarray(state, dtype=np.float32).copy()
    step = float(dt) / int(substeps)
    innovation_sd = np.asarray(diffusion_sd, dtype=np.float32) * np.sqrt(step)
    for _ in range(int(substeps)):
        innovation = rng.normal(0.0, innovation_sd).astype(np.float32)
        next_state = next_state + step * _synthetic_nonlinear_drift(next_state, transition_input)
        next_state = next_state + innovation
    return next_state.astype(np.float32)


def simulate_synthetic_nonlinear_data(
    T: int = 32, seed: int = 71, diffusion_scale: float = 1.0
) -> SyntheticNonlinearData:
    rng = np.random.default_rng(seed)
    transition_inputs = _build_transition_inputs(T)
    # Process-noise SD lever: scale=1 is the informative regime; >1 gives diffuse
    # dynamics where the prior-as-proposal advantage of cSMC should erode.
    diffusion_sd = np.asarray(TRUE_DIFFUSION_SD, dtype=np.float32) * float(diffusion_scale)
    latent = np.zeros((T, 3), dtype=np.float32)
    latent[0] = rng.normal(TRUE_T0_MEAN, TRUE_T0_SD).astype(np.float32)
    for time_idx in range(1, T):
        latent[time_idx] = _sample_nonlinear_transition(
            rng,
            latent[time_idx - 1],
            transition_inputs[time_idx],
            diffusion_sd=diffusion_sd,
        )

    linear_predictor = latent @ TRUE_LOADINGS.T + TRUE_MANIFEST_MEANS
    observations = np.full((T, len(MANIFEST_NAMES)), np.nan, dtype=np.float32)
    gaussian_indices = [0, 2, 3, 5, 7, 8, 9, 10]
    for idx in gaussian_indices:
        observations[:, idx] = rng.normal(
            linear_predictor[:, idx],
            TRUE_MANIFEST_SD[idx],
        ).astype(np.float32)

    observations[:, 1] = _sample_negative_binomial(
        rng,
        np.exp(linear_predictor[:, 1]),
        TRUE_OBS_R,
    )
    observations[:, 6] = _sample_negative_binomial(
        rng,
        np.exp(linear_predictor[:, 6]),
        TRUE_OBS_R,
    )
    gamma_mean = np.exp(linear_predictor[:, 4])
    observations[:, 4] = rng.gamma(
        shape=TRUE_OBS_SHAPE,
        scale=gamma_mean / TRUE_OBS_SHAPE,
    ).astype(np.float32)

    dense_rows = np.arange(T)
    mood_rows = dense_rows[dense_rows % 4 == 0]
    journal_rows = np.asarray([3, 8, 14, 21, 27], dtype=np.int64)
    journal_rows = journal_rows[journal_rows < T]
    gp_rows = np.asarray([6, 18, 30], dtype=np.int64)
    gp_rows = gp_rows[gp_rows < T]

    sparse_keep = {
        0: mood_rows,
        8: journal_rows,
        9: journal_rows,
        10: gp_rows,
    }
    for manifest_idx, rows in sparse_keep.items():
        mask = np.ones(T, dtype=bool)
        mask[rows] = False
        observations[mask, manifest_idx] = np.nan

    support = _build_gp_interval_support(T, gp_rows)
    gp_idx = MANIFEST_NAMES.index("gp_clinical_severity")
    for end in gp_rows:
        start = max(0, int(end) - 3)
        coeff_prev = np.zeros(T, dtype=np.float32)
        coeff_curr = np.zeros(T, dtype=np.float32)
        weights = np.zeros(T, dtype=np.float32)
        coeff_prev[start + 1 : int(end) + 1] = 0.5
        coeff_curr[start + 1 : int(end) + 1] = 0.5
        weights[start + 1 : int(end) + 1] = 1.0
        numerator = np.sum(
            coeff_prev
            * np.concatenate([[linear_predictor[0, gp_idx]], linear_predictor[:-1, gp_idx]])
            + coeff_curr * linear_predictor[:, gp_idx]
        )
        denominator = np.maximum(np.sum(weights), 1e-8)
        observations[end, gp_idx] = rng.normal(
            numerator / denominator,
            TRUE_MANIFEST_SD[gp_idx],
        )

    return SyntheticNonlinearData(
        observations=jnp.asarray(observations),
        times=jnp.arange(T, dtype=jnp.float32),
        latent=jnp.asarray(latent),
        transition_inputs=jnp.asarray(transition_inputs),
        observation_support=support,
    )


SCALAR_RECOVERY_TARGETS = {
    "vf_0_decay": TRUE_DECAY[0],
    "vf_1_decay": TRUE_DECAY[1],
    "vf_2_decay": TRUE_DECAY[2],
    **{
        name: value
        for name, value in TRUE_HILL_BY_SITE.items()
        if name not in PINNED_HILL_SHAPE_BY_SITE
    },
    **TRUE_MULTIPLICATIVE_BY_SITE,
    "obs_r": TRUE_OBS_R,
    "obs_shape": TRUE_OBS_SHAPE,
}

INPUT_EFFECT_RECOVERY_TARGETS = {
    f"input_{INPUT_NAMES[col]}_{LATENT_NAMES[row]}": {
        "site": "input_effect_free",
        "index": idx,
        "true": TRUE_INPUT_EFFECT[row, col],
    }
    for idx, (row, col) in enumerate(TRUE_INPUT_EFFECT_POSITIONS)
}

MEASUREMENT_MEAN_RECOVERY_TARGETS = {
    f"manifest_mean_{MANIFEST_NAMES[manifest_idx]}": {
        "site": "manifest_means_free",
        "index": idx,
        "true": TRUE_MANIFEST_MEANS[manifest_idx],
    }
    for idx, manifest_idx in enumerate(MEASUREMENT_MEANS_FREE_POSITIONS)
}

MEASUREMENT_LOADING_RECOVERY_TARGETS = {
    f"loading_{MANIFEST_NAMES[row]}_{LATENT_NAMES[col]}": {
        "site": "lambda_free",
        "index": idx,
        "true": TRUE_LOADINGS[row, col],
    }
    for idx, (row, col) in enumerate(MEASUREMENT_LOADINGS_FREE_POSITIONS)
}

RECOVERY_TARGETS = {
    **SCALAR_RECOVERY_TARGETS,
    **INPUT_EFFECT_RECOVERY_TARGETS,
    **MEASUREMENT_MEAN_RECOVERY_TARGETS,
    **MEASUREMENT_LOADING_RECOVERY_TARGETS,
}
