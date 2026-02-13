"""NumPyro State-Space Model.

Bayesian State-Space Model definition using NumPyro.
This module defines the probabilistic model only — inference is in inference.py.

Supports:
- Single-subject time series
- Any noise family (Gaussian, Poisson, Student-t, Gamma)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

if TYPE_CHECKING:
    import numpy as np

from causal_ssm_agent.models.likelihoods.base import CTParams, InitialStateParams, MeasurementParams
from causal_ssm_agent.models.ssm.constants import MIN_DT
from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily


@dataclass
class SSMSpec:
    """Specification for a state-space model.

    Matrix naming convention:
    - DRIFT: n_latent x n_latent continuous-time auto/cross effects
    - DIFFUSION: n_latent x n_latent process noise (Cholesky)
    - CINT: n_latent x 1 continuous intercept
    - LAMBDA: n_manifest x n_latent factor loadings
    - MANIFESTMEANS: n_manifest x 1 manifest intercepts
    - MANIFESTVAR: n_manifest x n_manifest measurement error (Cholesky)
    - T0MEANS: n_latent x 1 initial state means
    - T0VAR: n_latent x n_latent initial state variance (Cholesky)
    """

    n_latent: int
    n_manifest: int

    # Fixed or "free" specification for each matrix
    # If a matrix, use those fixed values; if "free", estimate
    drift: jnp.ndarray | Literal["free"] = "free"
    diffusion: jnp.ndarray | Literal["free", "diag"] = "free"
    cint: jnp.ndarray | Literal["free"] | None = None
    lambda_mat: jnp.ndarray | Literal["free"] = "free"
    manifest_means: jnp.ndarray | Literal["free"] | None = None
    manifest_var: jnp.ndarray | Literal["free", "diag"] = "diag"
    t0_means: jnp.ndarray | Literal["free"] = "free"
    t0_var: jnp.ndarray | Literal["free", "diag"] = "free"

    # Distribution families for observation and process noise
    diffusion_dist: DistributionFamily = DistributionFamily.GAUSSIAN
    manifest_dist: DistributionFamily = DistributionFamily.GAUSSIAN

    # Per-variable diffusion noise (overrides scalar diffusion_dist if set)
    diffusion_dists: list[DistributionFamily] | None = None

    # Per-channel observation noise (overrides scalar manifest_dist if set)
    manifest_dists: list[DistributionFamily] | None = None

    # Toggle first-pass (unconditional, model-level) Rao-Blackwellization
    first_pass_rb: bool = True

    # Toggle second-pass (conditional, sampler-level) Rao-Blackwellization
    second_pass_rb: bool = True

    # Parameter names for interpretability
    latent_names: list[str] | None = None
    manifest_names: list[str] | None = None

    # DAG-constrained masks (None = fully free, backward compat)
    # drift_mask: (n_latent, n_latent) bool — True where drift entry is free
    drift_mask: np.ndarray | None = None
    # lambda_mask: (n_manifest, n_latent) bool — True where loading is free to sample
    lambda_mask: np.ndarray | None = None

    # Time-invariant latent mask: (n_latent,) bool — True for quasi-constant latents.
    # These get near-zero drift diagonal and near-zero diffusion, so η_i(t) ≈ η_i(0).
    time_invariant_mask: np.ndarray | None = None


@dataclass
class SSMPriors:
    """Prior specifications for state-space model parameters.

    Each prior is specified as a dict with distribution parameters.
    """

    # Drift diagonal (auto-effects, typically negative for stability)
    drift_diag: dict = field(default_factory=lambda: {"mu": -0.5, "sigma": 1.0})
    # Drift off-diagonal (cross-effects)
    drift_offdiag: dict = field(default_factory=lambda: {"mu": 0.0, "sigma": 0.5})

    # Diffusion (log scale for positivity)
    diffusion_diag: dict = field(default_factory=lambda: {"sigma": 1.0})
    diffusion_offdiag: dict = field(default_factory=lambda: {"mu": 0.0, "sigma": 0.5})

    # Continuous intercept
    cint: dict = field(default_factory=lambda: {"mu": 0.0, "sigma": 1.0})

    # Factor loadings
    lambda_free: dict = field(default_factory=lambda: {"mu": 0.5, "sigma": 0.5})

    # Manifest means
    manifest_means: dict = field(default_factory=lambda: {"mu": 0.0, "sigma": 2.0})

    # Manifest variance (measurement error)
    manifest_var_diag: dict = field(default_factory=lambda: {"sigma": 1.0})

    # Initial state
    t0_means: dict = field(default_factory=lambda: {"mu": 0.0, "sigma": 2.0})
    t0_var_diag: dict = field(default_factory=lambda: {"sigma": 2.0})


def _make_prior_dist(prior: dict) -> dist.Distribution:
    """Build the appropriate numpyro distribution from a prior dict.

    If `lower`/`upper` bounds are present, uses TruncatedNormal to respect
    hard parameter bounds. Otherwise uses Normal (or HalfNormal if only sigma).

    Supports array-valued mu/sigma for per-element priors.
    """
    if "lower" in prior and "upper" in prior:
        return dist.TruncatedNormal(
            loc=jnp.asarray(prior["mu"]),
            scale=jnp.asarray(prior["sigma"]),
            low=jnp.asarray(prior["lower"]),
            high=jnp.asarray(prior["upper"]),
        )
    if "mu" in prior:
        return dist.Normal(jnp.asarray(prior["mu"]), jnp.asarray(prior["sigma"]))
    return dist.HalfNormal(jnp.asarray(prior["sigma"]))


def _make_prior_batch(prior: dict, n: int) -> dist.Distribution:
    """Build a batched prior distribution with shape (n,).

    If prior already has array-valued params with length n, use directly.
    If scalar, expand to batch shape [n].
    """
    d = _make_prior_dist(prior)
    if d.batch_shape == (n,):
        return d
    if d.batch_shape == ():
        return d.expand([n])
    raise ValueError(f"Prior batch shape {d.batch_shape} does not match expected ({n},)")


class SSMModel:
    """NumPyro state-space model definition.

    Defines the probabilistic model for Bayesian state-space models.
    Inference is handled externally by ssm.inference.fit().

    Features:
    - Continuous-time dynamics via stochastic differential equations
    - Multiple likelihood backends (Kalman, particle filter)
    """

    def __init__(
        self,
        spec: SSMSpec,
        priors: SSMPriors | None = None,
        n_particles: int = 200,
        pf_seed: int = 0,
        likelihood: Literal["particle", "kalman"] = "particle",
    ):
        """Initialize state-space model.

        Args:
            spec: Model specification
            priors: Prior distributions (uses defaults if None)
            n_particles: Number of particles for bootstrap PF
            pf_seed: Seed for fixed PF random key (deterministic for NUTS)
            likelihood: Likelihood backend - "particle" (universal, any noise family)
                or "kalman" (exact, linear Gaussian only)
        """
        self.spec = spec
        self.priors = priors or SSMPriors()
        self.n_particles = n_particles
        self.pf_key = jax.random.PRNGKey(pf_seed)
        self.likelihood = likelihood

    def _sample_drift(self, spec: SSMSpec) -> jnp.ndarray:
        """Sample drift matrix with stability constraints.

        When spec.drift_mask is set, only off-diagonal entries where the mask
        is True are sampled; the rest stay zero. This enforces DAG-constrained
        sparsity. When drift_mask is None, all off-diagonal entries are free
        (backward-compatible).

        Args:
            spec: Model specification

        Returns:
            drift: (n_latent, n_latent)
        """
        n = spec.n_latent

        if isinstance(spec.drift, jnp.ndarray):
            return spec.drift

        # Build off-diagonal positions list from mask (static, unrolled by XLA)
        offdiag_positions: list[tuple[int, int]] = []
        if spec.drift_mask is not None:
            for i in range(n):
                for j in range(n):
                    if i != j and spec.drift_mask[i, j]:
                        offdiag_positions.append((i, j))
        else:
            # No mask: all off-diagonal entries are free
            for i in range(n):
                for j in range(n):
                    if i != j:
                        offdiag_positions.append((i, j))

        n_offdiag = len(offdiag_positions)

        # Diagonal (auto-effects)
        drift_diag_pop = numpyro.sample(
            "drift_diag_pop",
            _make_prior_batch(self.priors.drift_diag, n),
        )

        # Off-diagonal (cross-effects)
        if n_offdiag > 0:
            drift_offdiag_pop = numpyro.sample(
                "drift_offdiag_pop",
                _make_prior_batch(self.priors.drift_offdiag, n_offdiag),
            )
        else:
            drift_offdiag_pop = jnp.array([])

        drift_diag = -jnp.abs(drift_diag_pop)

        # Time-invariant latents: near-zero drift diagonal (quasi-constant)
        if spec.time_invariant_mask is not None:
            ti_mask = jnp.array(spec.time_invariant_mask)
            drift_diag = jnp.where(ti_mask, -1e-6, drift_diag)

        drift = jnp.diag(drift_diag)
        for idx, (i, j) in enumerate(offdiag_positions):
            drift = drift.at[i, j].set(drift_offdiag_pop[idx])

        # Stability guard: penalise drift matrices whose max real eigenvalue
        # approaches zero (i.e. the system is near-unstable).  Only needed
        # for multi-latent models with off-diagonal coupling.
        if n > 1 and n_offdiag > 0:
            eigvals_real = jnp.real(jnp.linalg.eigvals(drift))
            max_eig = jnp.max(eigvals_real)
            margin = 1e-2
            penalty = jnp.where(
                max_eig > -margin,
                -1e4 * jnp.maximum(max_eig + margin, 0.0),
                0.0,
            )
            numpyro.factor("drift_stability", penalty)

        numpyro.deterministic("drift", drift)
        return drift

    def _sample_diffusion(self, spec: SSMSpec) -> jnp.ndarray:
        """Sample diffusion matrix (lower Cholesky)."""
        n = spec.n_latent

        if isinstance(spec.diffusion, jnp.ndarray):
            return spec.diffusion

        # Diagonal
        diff_diag_pop = numpyro.sample(
            "diffusion_diag_pop",
            dist.HalfNormal(self.priors.diffusion_diag["sigma"]).expand([n]),
        )

        if spec.diffusion == "diag":
            diffusion = jnp.diag(diff_diag_pop)
        else:
            # Full lower triangular
            n_lower = n * (n - 1) // 2
            if n_lower > 0:
                diff_lower = numpyro.sample(
                    "diffusion_lower",
                    dist.Normal(
                        self.priors.diffusion_offdiag["mu"],
                        self.priors.diffusion_offdiag["sigma"],
                    ).expand([n_lower]),
                )
            else:
                diff_lower = jnp.array([])

            diffusion = jnp.diag(diff_diag_pop)
            lower_idx = 0
            for i in range(n):
                for j in range(i):
                    diffusion = diffusion.at[i, j].set(diff_lower[lower_idx])
                    lower_idx += 1

        # Time-invariant latents: near-zero diffusion (quasi-constant)
        if spec.time_invariant_mask is not None:
            ti_mask = jnp.array(spec.time_invariant_mask)
            diag_vals = jnp.diag(diffusion)
            new_diag = jnp.where(ti_mask, 1e-6, diag_vals)
            diffusion = diffusion - jnp.diag(diag_vals) + jnp.diag(new_diag)

        numpyro.deterministic("diffusion", diffusion)
        return diffusion

    def _sample_cint(self, spec: SSMSpec) -> jnp.ndarray | None:
        """Sample continuous intercept."""
        if spec.cint is None:
            return None

        n = spec.n_latent

        if isinstance(spec.cint, jnp.ndarray):
            return spec.cint

        cint = numpyro.sample(
            "cint_pop",
            _make_prior_dist(self.priors.cint).expand([n]),
        )

        numpyro.deterministic("cint", cint)
        return cint

    def _sample_lambda(self, spec: SSMSpec) -> jnp.ndarray:
        """Sample factor loading matrix (shared across subjects).

        Three modes:
        1. lambda_mat is array AND lambda_mask is not None: template+mask mode.
           Start from the fixed template (with 1.0 for reference indicators),
           sample free loadings at positions where lambda_mask is True.
        2. lambda_mat is array, lambda_mask is None: fully fixed (return as-is).
        3. lambda_mat is "free": legacy identity + extra rows mode.
        """
        if isinstance(spec.lambda_mat, jnp.ndarray) and spec.lambda_mask is not None:
            # Template+mask mode: sample free positions from mask
            lambda_mat = jnp.array(spec.lambda_mat)

            # Build free positions list from mask (static for XLA)
            free_positions: list[tuple[int, int]] = []
            for i in range(spec.n_manifest):
                for j in range(spec.n_latent):
                    if spec.lambda_mask[i, j]:
                        free_positions.append((i, j))

            n_free = len(free_positions)
            if n_free > 0:
                free_loadings = numpyro.sample(
                    "lambda_free",
                    _make_prior_batch(self.priors.lambda_free, n_free),
                )
                for idx, (i, j) in enumerate(free_positions):
                    lambda_mat = lambda_mat.at[i, j].set(free_loadings[idx])

            numpyro.deterministic("lambda", lambda_mat)
            return lambda_mat

        if isinstance(spec.lambda_mat, jnp.ndarray):
            return spec.lambda_mat

        n_m, n_l = spec.n_manifest, spec.n_latent

        # Start with identity mapping for first n_latent manifests
        lambda_mat = jnp.eye(n_m, n_l)

        # Sample additional loadings if needed
        if n_m > n_l:
            n_free = (n_m - n_l) * n_l
            free_loadings = numpyro.sample(
                "lambda_free",
                _make_prior_dist(self.priors.lambda_free).expand([n_free]),
            )

            idx = 0
            for i in range(n_l, n_m):
                for j in range(n_l):
                    lambda_mat = lambda_mat.at[i, j].set(free_loadings[idx])
                    idx += 1

        numpyro.deterministic("lambda", lambda_mat)
        return lambda_mat

    def _sample_manifest_params(self, spec: SSMSpec) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample manifest means and variance (shared across subjects)."""
        n_m = spec.n_manifest

        # Means
        if spec.manifest_means is None:
            manifest_means = jnp.zeros(n_m)
        elif isinstance(spec.manifest_means, jnp.ndarray):
            manifest_means = spec.manifest_means
        else:
            manifest_means = numpyro.sample(
                "manifest_means",
                _make_prior_dist(self.priors.manifest_means).expand([n_m]),
            )

        # Variance (Cholesky)
        if isinstance(spec.manifest_var, jnp.ndarray):
            manifest_chol = spec.manifest_var
        else:
            var_diag = numpyro.sample(
                "manifest_var_diag",
                dist.HalfNormal(self.priors.manifest_var_diag["sigma"]).expand([n_m]),
            )
            manifest_chol = jnp.diag(var_diag)

        numpyro.deterministic("manifest_cov", manifest_chol @ manifest_chol.T)
        return manifest_means, manifest_chol

    def _sample_t0_params(self, spec: SSMSpec) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample initial state parameters."""
        n_l = spec.n_latent

        # Means
        if isinstance(spec.t0_means, jnp.ndarray):
            t0_means = spec.t0_means
        else:
            t0_means = numpyro.sample(
                "t0_means_pop",
                _make_prior_dist(self.priors.t0_means).expand([n_l]),
            )

        # Variance (Cholesky)
        if isinstance(spec.t0_var, jnp.ndarray):
            t0_chol = spec.t0_var
        else:
            var_diag = numpyro.sample(
                "t0_var_diag",
                dist.HalfNormal(self.priors.t0_var_diag["sigma"]).expand([n_l]),
            )
            t0_chol = jnp.diag(var_diag)

        numpyro.deterministic("t0_means", t0_means)
        numpyro.deterministic("t0_cov", t0_chol @ t0_chol.T)
        return t0_means, t0_chol

    def make_likelihood_backend(self):
        """Construct the default likelihood backend from model configuration.

        Uses self.likelihood to select between Kalman and Particle backends.
        When first_pass_rb is enabled, analyzes the model structure to identify
        decoupled linear-Gaussian sub-blocks that can be handled by exact Kalman
        filtering, composing with a particle filter for the remainder.

        Callers that need a different backend (Laplace, Structured VI, DPF)
        construct it themselves instead of calling this.
        """
        spec = self.spec
        if self.likelihood == "kalman":
            from causal_ssm_agent.models.likelihoods.kalman import KalmanLikelihood

            return KalmanLikelihood(
                n_latent=spec.n_latent,
                n_manifest=spec.n_manifest,
            )

        # Resolve per-variable diffusion dist — passed as-is to ParticleLikelihood
        # which handles its own list→scalar normalization internally.
        from causal_ssm_agent.models.likelihoods.graph_analysis import (
            get_per_channel_manifest,
            get_per_variable_diffusion,
        )

        per_var = get_per_variable_diffusion(spec)
        per_obs = get_per_channel_manifest(spec)

        # First-pass RB analysis: identify decoupled Gaussian sub-blocks
        if spec.first_pass_rb:
            from causal_ssm_agent.models.likelihoods.graph_analysis import (
                analyze_first_pass_rb,
            )

            partition = analyze_first_pass_rb(spec)

            if partition.has_kalman_block and not partition.has_particle_block:
                # Everything is marginalizable — use pure Kalman
                from causal_ssm_agent.models.likelihoods.kalman import KalmanLikelihood

                return KalmanLikelihood(
                    n_latent=spec.n_latent,
                    n_manifest=spec.n_manifest,
                )

            # Only create ComposedLikelihood when the Kalman block has
            # exclusive observations — otherwise it can't contribute to LL.
            if (
                partition.has_kalman_block
                and partition.has_particle_block
                and len(partition.obs_kalman_idx) > 0
            ):
                from causal_ssm_agent.models.likelihoods.composed import ComposedLikelihood
                from causal_ssm_agent.models.likelihoods.kalman import KalmanLikelihood
                from causal_ssm_agent.models.likelihoods.particle import ParticleLikelihood

                n_k = len(partition.kalman_idx)
                n_obs_k = len(partition.obs_kalman_idx)
                n_p = len(partition.particle_idx)
                n_obs_p = len(partition.obs_particle_idx)

                # Per-variable diffusion for the particle sub-block
                particle_diffs = [per_var[int(i)] for i in partition.particle_idx]

                # Determine manifest_dist for PF channels from per-channel info
                pf_obs_dists = [per_obs[int(k)] for k in partition.obs_particle_idx]
                pf_manifest_dist = spec.manifest_dist.value
                for d in pf_obs_dists:
                    if d != "gaussian":
                        pf_manifest_dist = d
                        break

                return ComposedLikelihood(
                    partition=partition,
                    kalman_backend=KalmanLikelihood(
                        n_latent=n_k,
                        n_manifest=n_obs_k,
                    ),
                    particle_backend=ParticleLikelihood(
                        n_latent=n_p,
                        n_manifest=n_obs_p,
                        n_particles=self.n_particles,
                        rng_key=self.pf_key,
                        manifest_dist=pf_manifest_dist,
                        diffusion_dist=particle_diffs,
                        block_rb=spec.second_pass_rb,
                    ),
                )

        # Fallthrough: full particle filter (ParticleLikelihood normalizes
        # the per_var list internally — no need to collapse here)
        from causal_ssm_agent.models.likelihoods.particle import ParticleLikelihood

        return ParticleLikelihood(
            n_latent=spec.n_latent,
            n_manifest=spec.n_manifest,
            n_particles=self.n_particles,
            rng_key=self.pf_key,
            manifest_dist=spec.manifest_dist.value,
            diffusion_dist=per_var,
            block_rb=spec.second_pass_rb,
        )

    def model(
        self,
        observations: jnp.ndarray,
        times: jnp.ndarray,
        likelihood_backend=None,
    ) -> None:
        """NumPyro model function.

        Args:
            observations: (N, n_manifest) observed data
            times: (N,) observation times
            likelihood_backend: Likelihood backend instance (e.g. ParticleLikelihood,
                KalmanLikelihood, LaplaceLikelihood). Required — use
                model.make_likelihood_backend() for the default.
        """
        if likelihood_backend is None:
            raise ValueError(
                "likelihood_backend is required. "
                "Use model.make_likelihood_backend() for the default."
            )

        spec = self.spec

        # Sample parameters
        drift = self._sample_drift(spec)
        diffusion_chol = self._sample_diffusion(spec)
        cint = self._sample_cint(spec)
        lambda_mat = self._sample_lambda(spec)
        manifest_means, manifest_chol = self._sample_manifest_params(spec)
        t0_means, t0_chol = self._sample_t0_params(spec)

        # Convert to covariances
        diffusion_cov = diffusion_chol @ diffusion_chol.T
        manifest_cov = manifest_chol @ manifest_chol.T
        t0_cov = t0_chol @ t0_chol.T

        # Sample noise family hyperparameters
        extra_params = {}
        if spec.manifest_dist == DistributionFamily.STUDENT_T:
            extra_params["obs_df"] = numpyro.sample("obs_df", dist.Gamma(5.0, 1.0))
        if spec.manifest_dist == DistributionFamily.GAMMA:
            extra_params["obs_shape"] = numpyro.sample("obs_shape", dist.Gamma(2.0, 1.0))
        if spec.manifest_dist == DistributionFamily.NEGATIVE_BINOMIAL:
            extra_params["obs_r"] = numpyro.sample("obs_r", dist.Gamma(2.0, 0.5))
        if spec.manifest_dist == DistributionFamily.BETA:
            extra_params["obs_concentration"] = numpyro.sample(
                "obs_concentration", dist.Gamma(5.0, 0.5)
            )
        if spec.diffusion_dist == DistributionFamily.STUDENT_T:
            extra_params["proc_df"] = numpyro.sample("proc_df", dist.Gamma(5.0, 1.0))

        ct_params = CTParams(drift=drift, diffusion_cov=diffusion_cov, cint=cint)
        meas_params = MeasurementParams(
            lambda_mat=lambda_mat,
            manifest_means=manifest_means,
            manifest_cov=manifest_cov,
        )

        time_intervals = jnp.diff(times, prepend=times[0])
        time_intervals = time_intervals.at[0].set(MIN_DT)

        init = InitialStateParams(mean=t0_means, cov=t0_cov)
        ll = likelihood_backend.compute_log_likelihood(
            ct_params,
            meas_params,
            init,
            observations,
            time_intervals,
            extra_params=extra_params if extra_params else None,
        )

        numpyro.factor("log_likelihood", ll)
