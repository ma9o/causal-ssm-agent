"""NumPyro CT-SEM Model.

Hierarchical Bayesian Continuous-Time Structural Equation Model
using NumPyro for inference.

Supports:
- Single-subject time series
- Multi-subject panel data with shared parameters
- Hierarchical models with individual variation in parameters
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from jax import lax, vmap
from numpyro.infer import MCMC, NUTS

from dsem_agent.models.ctsem.kalman import kalman_log_likelihood


@dataclass
class CTSEMSpec:
    """Specification for a CT-SEM model.

    Follows the ctsem convention for matrix naming:
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

    # Hierarchical structure
    hierarchical: bool = False
    n_subjects: int = 1

    # Which parameters vary across individuals (only used if hierarchical=True)
    # Options: "drift_diag", "drift_offdiag", "diffusion", "cint", "t0_means"
    indvarying: list[str] = field(default_factory=lambda: ["t0_means"])

    # Parameter names for interpretability
    latent_names: list[str] | None = None
    manifest_names: list[str] | None = None


@dataclass
class CTSEMPriors:
    """Prior specifications for CT-SEM parameters.

    Each prior is specified as a dict with distribution parameters.
    """

    # Drift diagonal (auto-effects, typically negative for stability)
    drift_diag: dict = field(
        default_factory=lambda: {"mu": -0.5, "sigma": 1.0}
    )
    # Drift off-diagonal (cross-effects)
    drift_offdiag: dict = field(
        default_factory=lambda: {"mu": 0.0, "sigma": 0.5}
    )

    # Diffusion (log scale for positivity)
    diffusion_diag: dict = field(
        default_factory=lambda: {"sigma": 1.0}
    )
    diffusion_offdiag: dict = field(
        default_factory=lambda: {"mu": 0.0, "sigma": 0.5}
    )

    # Continuous intercept
    cint: dict = field(
        default_factory=lambda: {"mu": 0.0, "sigma": 1.0}
    )

    # Factor loadings
    lambda_free: dict = field(
        default_factory=lambda: {"mu": 0.5, "sigma": 0.5}
    )

    # Manifest means
    manifest_means: dict = field(
        default_factory=lambda: {"mu": 0.0, "sigma": 2.0}
    )

    # Manifest variance (measurement error)
    manifest_var_diag: dict = field(
        default_factory=lambda: {"sigma": 1.0}
    )

    # Initial state
    t0_means: dict = field(
        default_factory=lambda: {"mu": 0.0, "sigma": 2.0}
    )
    t0_var_diag: dict = field(
        default_factory=lambda: {"sigma": 2.0}
    )

    # Hierarchical (population-level SD for random effects)
    pop_sd: dict = field(
        default_factory=lambda: {"sigma": 1.0}
    )


class CTSEMModel:
    """NumPyro CT-SEM model.

    Implements hierarchical Bayesian CT-SEM with:
    - Continuous-time dynamics via stochastic differential equations
    - Kalman filter likelihood computation
    - Optional hierarchical structure for multiple subjects with individual variation
    """

    def __init__(
        self,
        spec: CTSEMSpec,
        priors: CTSEMPriors | None = None,
    ):
        """Initialize CT-SEM model.

        Args:
            spec: Model specification
            priors: Prior distributions (uses defaults if None)
        """
        self.spec = spec
        self.priors = priors or CTSEMPriors()

    def _sample_drift(
        self, spec: CTSEMSpec, n_subjects: int = 1, hierarchical: bool = False
    ) -> jnp.ndarray:
        """Sample drift matrix with stability constraints.

        Args:
            spec: Model specification
            n_subjects: Number of subjects (for hierarchical)
            hierarchical: Whether to use hierarchical sampling

        Returns:
            drift: (n_latent, n_latent) or (n_subjects, n_latent, n_latent)
        """
        n = spec.n_latent

        if isinstance(spec.drift, jnp.ndarray):
            if hierarchical and n_subjects > 1:
                return jnp.broadcast_to(spec.drift, (n_subjects, n, n))
            return spec.drift

        # Population-level diagonal (auto-effects)
        drift_diag_pop = numpyro.sample(
            "drift_diag_pop",
            dist.Normal(
                self.priors.drift_diag["mu"], self.priors.drift_diag["sigma"]
            ).expand([n]),
        )

        # Population-level off-diagonal (cross-effects)
        n_offdiag = n * n - n
        if n_offdiag > 0:
            drift_offdiag_pop = numpyro.sample(
                "drift_offdiag_pop",
                dist.Normal(
                    self.priors.drift_offdiag["mu"], self.priors.drift_offdiag["sigma"]
                ).expand([n_offdiag]),
            )
        else:
            drift_offdiag_pop = jnp.array([])

        if hierarchical and n_subjects > 1:
            # Individual variation in drift
            if "drift_diag" in spec.indvarying:
                drift_diag_sd = numpyro.sample(
                    "drift_diag_sd",
                    dist.HalfNormal(self.priors.pop_sd["sigma"]).expand([n]),
                )
                drift_diag_raw = numpyro.sample(
                    "drift_diag_raw",
                    dist.Normal(0, 1).expand([n_subjects, n]),
                )
                drift_diag = drift_diag_pop + drift_diag_sd * drift_diag_raw
            else:
                drift_diag = jnp.broadcast_to(drift_diag_pop, (n_subjects, n))

            if "drift_offdiag" in spec.indvarying and n_offdiag > 0:
                drift_offdiag_sd = numpyro.sample(
                    "drift_offdiag_sd",
                    dist.HalfNormal(self.priors.pop_sd["sigma"]).expand([n_offdiag]),
                )
                drift_offdiag_raw = numpyro.sample(
                    "drift_offdiag_raw",
                    dist.Normal(0, 1).expand([n_subjects, n_offdiag]),
                )
                drift_offdiag = drift_offdiag_pop + drift_offdiag_sd * drift_offdiag_raw
            else:
                drift_offdiag = jnp.broadcast_to(drift_offdiag_pop, (n_subjects, n_offdiag))

            # Assemble drift matrices for each subject
            def assemble_drift(diag, offdiag):
                # Constrain diagonal to be negative for stability
                diag_neg = -jnp.abs(diag)
                drift = jnp.diag(diag_neg)
                offdiag_idx = 0
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            drift = drift.at[i, j].set(offdiag[offdiag_idx])
                            offdiag_idx += 1
                return drift

            drift = vmap(assemble_drift)(drift_diag, drift_offdiag)
        else:
            # Single drift matrix
            drift_diag = -jnp.abs(drift_diag_pop)
            drift = jnp.diag(drift_diag)
            offdiag_idx = 0
            for i in range(n):
                for j in range(n):
                    if i != j:
                        drift = drift.at[i, j].set(drift_offdiag_pop[offdiag_idx])
                        offdiag_idx += 1

        numpyro.deterministic("drift", drift)
        return drift

    def _sample_diffusion(
        self, spec: CTSEMSpec, n_subjects: int = 1, hierarchical: bool = False
    ) -> jnp.ndarray:
        """Sample diffusion matrix (lower Cholesky)."""
        n = spec.n_latent

        if isinstance(spec.diffusion, jnp.ndarray):
            if hierarchical and n_subjects > 1:
                return jnp.broadcast_to(spec.diffusion, (n_subjects, n, n))
            return spec.diffusion

        # Population-level diagonal
        diff_diag_pop = numpyro.sample(
            "diffusion_diag_pop",
            dist.HalfNormal(self.priors.diffusion_diag["sigma"]).expand([n]),
        )

        if spec.diffusion == "diag":
            if hierarchical and n_subjects > 1 and "diffusion" in spec.indvarying:
                diff_diag_sd = numpyro.sample(
                    "diffusion_diag_sd",
                    dist.HalfNormal(self.priors.pop_sd["sigma"]).expand([n]),
                )
                diff_diag_raw = numpyro.sample(
                    "diffusion_diag_raw",
                    dist.Normal(0, 1).expand([n_subjects, n]),
                )
                diff_diag = jnp.abs(diff_diag_pop + diff_diag_sd * diff_diag_raw)
                diffusion = vmap(jnp.diag)(diff_diag)
            elif hierarchical and n_subjects > 1:
                diffusion = jnp.broadcast_to(jnp.diag(diff_diag_pop), (n_subjects, n, n))
            else:
                diffusion = jnp.diag(diff_diag_pop)
        else:
            # Full lower triangular (non-hierarchical for simplicity)
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

            if hierarchical and n_subjects > 1:
                diffusion = jnp.broadcast_to(diffusion, (n_subjects, n, n))

        numpyro.deterministic("diffusion", diffusion)
        return diffusion

    def _sample_cint(
        self, spec: CTSEMSpec, n_subjects: int = 1, hierarchical: bool = False
    ) -> jnp.ndarray | None:
        """Sample continuous intercept."""
        if spec.cint is None:
            return None

        n = spec.n_latent

        if isinstance(spec.cint, jnp.ndarray):
            if hierarchical and n_subjects > 1:
                return jnp.broadcast_to(spec.cint, (n_subjects, n))
            return spec.cint

        cint_pop = numpyro.sample(
            "cint_pop",
            dist.Normal(self.priors.cint["mu"], self.priors.cint["sigma"]).expand([n]),
        )

        if hierarchical and n_subjects > 1 and "cint" in spec.indvarying:
            cint_sd = numpyro.sample(
                "cint_sd",
                dist.HalfNormal(self.priors.pop_sd["sigma"]).expand([n]),
            )
            cint_raw = numpyro.sample(
                "cint_raw",
                dist.Normal(0, 1).expand([n_subjects, n]),
            )
            cint = cint_pop + cint_sd * cint_raw
        elif hierarchical and n_subjects > 1:
            cint = jnp.broadcast_to(cint_pop, (n_subjects, n))
        else:
            cint = cint_pop

        numpyro.deterministic("cint", cint)
        return cint

    def _sample_lambda(self, spec: CTSEMSpec) -> jnp.ndarray:
        """Sample factor loading matrix (shared across subjects)."""
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
                dist.Normal(
                    self.priors.lambda_free["mu"], self.priors.lambda_free["sigma"]
                ).expand([n_free]),
            )

            idx = 0
            for i in range(n_l, n_m):
                for j in range(n_l):
                    lambda_mat = lambda_mat.at[i, j].set(free_loadings[idx])
                    idx += 1

        numpyro.deterministic("lambda", lambda_mat)
        return lambda_mat

    def _sample_manifest_params(self, spec: CTSEMSpec) -> tuple[jnp.ndarray, jnp.ndarray]:
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
                dist.Normal(
                    self.priors.manifest_means["mu"], self.priors.manifest_means["sigma"]
                ).expand([n_m]),
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

    def _sample_t0_params(
        self, spec: CTSEMSpec, n_subjects: int = 1, hierarchical: bool = False
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample initial state parameters."""
        n_l = spec.n_latent

        # Means (can vary by subject)
        if isinstance(spec.t0_means, jnp.ndarray):
            t0_means = spec.t0_means
            if hierarchical and n_subjects > 1:
                t0_means = jnp.broadcast_to(t0_means, (n_subjects, n_l))
        else:
            t0_means_pop = numpyro.sample(
                "t0_means_pop",
                dist.Normal(
                    self.priors.t0_means["mu"], self.priors.t0_means["sigma"]
                ).expand([n_l]),
            )

            if hierarchical and n_subjects > 1 and "t0_means" in spec.indvarying:
                t0_means_sd = numpyro.sample(
                    "t0_means_sd",
                    dist.HalfNormal(self.priors.pop_sd["sigma"]).expand([n_l]),
                )
                t0_means_raw = numpyro.sample(
                    "t0_means_raw",
                    dist.Normal(0, 1).expand([n_subjects, n_l]),
                )
                t0_means = t0_means_pop + t0_means_sd * t0_means_raw
            elif hierarchical and n_subjects > 1:
                t0_means = jnp.broadcast_to(t0_means_pop, (n_subjects, n_l))
            else:
                t0_means = t0_means_pop

        # Variance (Cholesky) - shared across subjects
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

    def model(
        self,
        observations: jnp.ndarray,
        times: jnp.ndarray,
        subject_ids: jnp.ndarray | None = None,
    ) -> None:
        """NumPyro model function.

        Args:
            observations: (N, n_manifest) observed data
            times: (N,) observation times
            subject_ids: (N,) subject indices (0-indexed, for hierarchical models)
        """
        spec = self.spec
        hierarchical = spec.hierarchical and subject_ids is not None

        if hierarchical:
            n_subjects = int(jnp.max(subject_ids)) + 1
        else:
            n_subjects = 1

        # Sample parameters
        drift = self._sample_drift(spec, n_subjects, hierarchical)
        diffusion_chol = self._sample_diffusion(spec, n_subjects, hierarchical)
        cint = self._sample_cint(spec, n_subjects, hierarchical)
        lambda_mat = self._sample_lambda(spec)
        manifest_means, manifest_chol = self._sample_manifest_params(spec)
        t0_means, t0_chol = self._sample_t0_params(spec, n_subjects, hierarchical)

        # Convert to covariances
        if hierarchical and n_subjects > 1:
            diffusion_cov = vmap(lambda x: x @ x.T)(diffusion_chol)
        else:
            diffusion_cov = diffusion_chol @ diffusion_chol.T
        manifest_cov = manifest_chol @ manifest_chol.T
        t0_cov = t0_chol @ t0_chol.T

        # Compute log-likelihood
        if not hierarchical or n_subjects == 1:
            # Single subject
            time_intervals = jnp.diff(times, prepend=times[0])
            time_intervals = time_intervals.at[0].set(1e-6)

            ll = kalman_log_likelihood(
                observations=observations,
                time_intervals=time_intervals,
                drift=drift,
                diffusion_cov=diffusion_cov,
                cint=cint,
                lambda_mat=lambda_mat,
                manifest_means=manifest_means,
                manifest_cov=manifest_cov,
                t0_mean=t0_means,
                t0_cov=t0_cov,
            )
        else:
            # Multiple subjects with hierarchical structure
            ll = self._hierarchical_likelihood(
                observations,
                times,
                subject_ids,
                n_subjects,
                drift,
                diffusion_cov,
                cint,
                lambda_mat,
                manifest_means,
                manifest_cov,
                t0_means,
                t0_cov,
            )

        numpyro.factor("log_likelihood", ll)

    def _hierarchical_likelihood(
        self,
        observations: jnp.ndarray,
        times: jnp.ndarray,
        subject_ids: jnp.ndarray,
        n_subjects: int,
        drift: jnp.ndarray,
        diffusion_cov: jnp.ndarray,
        cint: jnp.ndarray | None,
        lambda_mat: jnp.ndarray,
        manifest_means: jnp.ndarray,
        manifest_cov: jnp.ndarray,
        t0_means: jnp.ndarray,
        t0_cov: jnp.ndarray,
    ) -> float:
        """Compute log-likelihood for hierarchical model with subject-varying params."""

        def subject_ll(subj_idx):
            # Get subject-specific parameters
            subj_drift = drift[subj_idx] if drift.ndim == 3 else drift
            subj_diff_cov = diffusion_cov[subj_idx] if diffusion_cov.ndim == 3 else diffusion_cov
            subj_cint = cint[subj_idx] if cint is not None and cint.ndim == 2 else cint
            subj_t0_mean = t0_means[subj_idx] if t0_means.ndim == 2 else t0_means

            # Get subject's observations
            mask = subject_ids == subj_idx
            # Use where to get a fixed-size slice (JAX compatible)
            # This requires padding, handled by the caller
            subj_obs = jnp.where(
                mask[:, None],
                observations,
                jnp.nan,
            )
            subj_times = jnp.where(mask, times, jnp.inf)

            # Sort by time and filter
            sort_idx = jnp.argsort(subj_times)
            subj_obs_sorted = subj_obs[sort_idx]
            subj_times_sorted = subj_times[sort_idx]

            # Count valid observations
            n_valid = jnp.sum(mask)

            # Compute time intervals
            time_intervals = jnp.diff(subj_times_sorted, prepend=subj_times_sorted[0])
            time_intervals = time_intervals.at[0].set(1e-6)

            # Compute likelihood
            ll = kalman_log_likelihood(
                observations=subj_obs_sorted,
                time_intervals=time_intervals,
                drift=subj_drift,
                diffusion_cov=subj_diff_cov,
                cint=subj_cint,
                lambda_mat=lambda_mat,
                manifest_means=manifest_means,
                manifest_cov=manifest_cov,
                t0_mean=subj_t0_mean,
                t0_cov=t0_cov,
            )

            # Only count if subject has observations
            return jnp.where(n_valid > 0, ll, 0.0)

        # Sum over all subjects
        subject_indices = jnp.arange(n_subjects)
        return jnp.sum(vmap(subject_ll)(subject_indices))

    def fit(
        self,
        observations: jnp.ndarray,
        times: jnp.ndarray,
        subject_ids: jnp.ndarray | None = None,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        num_chains: int = 4,
        seed: int = 0,
        **mcmc_kwargs: Any,
    ) -> MCMC:
        """Fit the model using MCMC.

        Args:
            observations: (N, n_manifest) observed data
            times: (N,) observation times
            subject_ids: (N,) subject indices, 0-indexed (for hierarchical)
            num_warmup: Number of warmup samples
            num_samples: Number of posterior samples
            num_chains: Number of MCMC chains
            seed: Random seed
            **mcmc_kwargs: Additional MCMC arguments

        Returns:
            MCMC object with posterior samples
        """
        kernel = NUTS(self.model)
        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            **mcmc_kwargs,
        )

        rng_key = random.PRNGKey(seed)
        mcmc.run(rng_key, observations, times, subject_ids)

        return mcmc

    def prior_predictive(
        self,
        times: jnp.ndarray,
        num_samples: int = 100,
        seed: int = 0,
    ) -> dict[str, jnp.ndarray]:
        """Sample from the prior predictive distribution.

        Args:
            times: (T,) time points
            num_samples: Number of prior samples
            seed: Random seed

        Returns:
            Dict of prior predictive samples
        """
        from numpyro.infer import Predictive

        rng_key = random.PRNGKey(seed)
        predictive = Predictive(self.model, num_samples=num_samples)

        # Create dummy observations
        dummy_obs = jnp.zeros((len(times), self.spec.n_manifest))

        return predictive(rng_key, dummy_obs, times)


def build_ctsem_model(
    n_latent: int,
    n_manifest: int,
    lambda_mat: jnp.ndarray | None = None,
    hierarchical: bool = False,
    n_subjects: int = 1,
    indvarying: list[str] | None = None,
) -> CTSEMModel:
    """Convenience function to build a CT-SEM model.

    Args:
        n_latent: Number of latent processes
        n_manifest: Number of manifest indicators
        lambda_mat: Fixed factor loadings (optional)
        hierarchical: Whether to use hierarchical structure
        n_subjects: Number of subjects (for hierarchical)
        indvarying: Which parameters vary across individuals

    Returns:
        CTSEMModel instance
    """
    spec = CTSEMSpec(
        n_latent=n_latent,
        n_manifest=n_manifest,
        lambda_mat=lambda_mat if lambda_mat is not None else "free",
        hierarchical=hierarchical,
        n_subjects=n_subjects,
        indvarying=indvarying or ["t0_means"],
    )
    return CTSEMModel(spec)
