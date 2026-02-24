"""Four-latent recovery problem: Stress -> Fatigue -> Focus -> Performance.

Ground truth: 4 latent processes with lower-triangular drift, diagonal diffusion,
6 Gaussian indicators (4 identity + 2 cross-loading), and free continuous intercepts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla
import numpy as np

from causal_ssm_agent.models.ssm import SSMPriors, SSMSpec, discretize_system


@dataclass
class RecoveryProblem:
    """Standardized ground truth for parameter recovery benchmarks."""

    true_drift: jnp.ndarray
    true_diff_diag: jnp.ndarray
    true_cint: jnp.ndarray
    true_lambda: jnp.ndarray
    true_manifest_means: jnp.ndarray
    true_mvar_diag: jnp.ndarray
    latent_names: list[str]
    n_latent: int
    n_manifest: int
    dt: float
    spec: SSMSpec
    priors: SSMPriors
    # Non-Gaussian observation noise
    manifest_dist: str = "gaussian"
    extra_true_params: dict = field(default_factory=dict)

    def simulate(self, T: int, seed: int = 42):
        """Discretize + forward simulate from ground truth.

        Returns (observations, times, latent) arrays.
        """
        diff_cov = jnp.diag(self.true_diff_diag) @ jnp.diag(self.true_diff_diag).T
        Ad, Qd, cd = discretize_system(self.true_drift, diff_cov, self.true_cint, self.dt)
        Qd_chol = jla.cholesky(Qd + jnp.eye(self.n_latent) * 1e-8, lower=True)

        key = random.PRNGKey(seed)
        key, init_key = random.split(key)
        x0 = jla.cholesky(jnp.eye(self.n_latent) * 0.5, lower=True) @ random.normal(
            init_key, (self.n_latent,)
        )

        states = [x0]
        for _ in range(T - 1):
            key, nk = random.split(key)
            states.append(
                Ad @ states[-1] + cd.flatten() + Qd_chol @ random.normal(nk, (self.n_latent,))
            )
        latent = jnp.stack(states)

        mu = jax.vmap(lambda x: self.true_lambda @ x + self.true_manifest_means)(latent)

        key, ok = random.split(key)
        if self.manifest_dist == "student_t":
            obs_df = self.extra_true_params["obs_df"]
            scale = jnp.sqrt(self.true_mvar_diag)
            # Student-t: normal / sqrt(chi2/df)
            z = random.normal(ok, (T, self.n_manifest))
            key, chi2_key = random.split(key)
            chi2 = random.gamma(chi2_key, obs_df / 2.0, (T, self.n_manifest)) * 2.0
            t_noise = z / jnp.sqrt(chi2 / obs_df)
            obs = mu + t_noise * scale
        elif self.manifest_dist == "poisson":
            obs = random.poisson(ok, jax.nn.softplus(mu)).astype(jnp.float32)
        else:
            R_chol = jla.cholesky(jnp.diag(self.true_mvar_diag), lower=True)
            obs = mu + random.normal(ok, (T, self.n_manifest)) @ R_chol.T

        times = jnp.arange(T, dtype=float) * self.dt

        return obs, times, latent

    def print_ground_truth(self):
        """Print ground truth summary."""
        from benchmarks.metrics import header, print_matrix

        header("GROUND TRUTH")
        eigs = np.linalg.eigvals(np.array(self.true_drift))
        print("Drift matrix A (continuous-time):")
        print_matrix(self.true_drift, self.latent_names, self.latent_names)
        print()
        print("Eigenvalues:", "  ".join(f"{e.real:+.3f}" for e in eigs))
        assert all(e.real < 0 for e in eigs), "Drift is not stable!"
        print("All eigenvalues negative — stable.")
        print()
        print(
            f"Lambda ({self.n_manifest}x{self.n_latent}), "
            f"diffusion diag={[round(float(x), 2) for x in self.true_diff_diag]}"
        )
        print(f"CINT={[round(float(x), 2) for x in self.true_cint]}")
        print()


def _make_four_latent() -> RecoveryProblem:
    n_latent, n_manifest, dt = 4, 6, 0.5
    names = ["Stress", "Fatigue", "Focus", "Perf"]

    true_drift = jnp.array(
        [
            [-0.8, 0.0, 0.0, 0.0],
            [0.3, -0.5, 0.0, 0.0],
            [-0.2, -0.3, -0.6, 0.0],
            [0.0, 0.0, 0.4, -0.7],
        ]
    )
    true_diff_diag = jnp.array([0.4, 0.3, 0.3, 0.25])
    true_cint = jnp.array([0.5, 0.0, 0.3, 0.0])

    true_lambda = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.6, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.4, 0.7],
        ]
    )
    true_manifest_means = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0])
    true_mvar_diag = jnp.array([0.1, 0.1, 0.1, 0.1, 0.15, 0.15])

    spec = SSMSpec(
        n_latent=n_latent,
        n_manifest=n_manifest,
        drift="free",
        diffusion="diag",
        cint="free",
        lambda_mat="free",
        manifest_means="free",
        manifest_var="diag",
        t0_means="free",
        t0_var="diag",
        latent_names=names,
    )
    priors = SSMPriors(
        drift_diag={"mu": -0.5, "sigma": 0.5},
        drift_offdiag={"mu": 0.0, "sigma": 0.5},
        diffusion_diag={"sigma": 0.5},
        cint={"mu": 0.0, "sigma": 1.0},
        lambda_free={"mu": 0.5, "sigma": 0.5},
        manifest_means={"mu": 0.0, "sigma": 2.0},
        manifest_var_diag={"sigma": 0.5},
    )

    return RecoveryProblem(
        true_drift=true_drift,
        true_diff_diag=true_diff_diag,
        true_cint=true_cint,
        true_lambda=true_lambda,
        true_manifest_means=true_manifest_means,
        true_mvar_diag=true_mvar_diag,
        latent_names=names,
        n_latent=n_latent,
        n_manifest=n_manifest,
        dt=dt,
        spec=spec,
        priors=priors,
    )


FOUR_LATENT = _make_four_latent()
