"""Quick diagnostic: test all three proposals on D=3 LGSS."""

import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla

from causal_ssm_agent.models.ssm import SSMModel, SSMSpec, discretize_system, fit

# -- Ground truth (1D Gaussian SSM) --
n_latent, n_manifest = 1, 1
T, dt = 100, 1.0

true_drift = jnp.array([[-0.3]])
true_diff_cov = jnp.array([[0.3**2]])
true_obs_var = jnp.array([[0.5**2]])

Ad, Qd, _ = discretize_system(true_drift, true_diff_cov, None, dt)
Qd_chol = jla.cholesky(Qd + jnp.eye(n_latent) * 1e-8, lower=True)
R_chol = jla.cholesky(true_obs_var, lower=True)

key = random.PRNGKey(42)
states = [jnp.zeros(n_latent)]
for _ in range(T - 1):
    key, nk = random.split(key)
    states.append(Ad @ states[-1] + Qd_chol @ random.normal(nk, (n_latent,)))
latent = jnp.stack(states)

key, obs_key = random.split(key)
observations = latent + random.normal(obs_key, (T, n_manifest)) @ R_chol.T
times = jnp.arange(T, dtype=float) * dt

spec = SSMSpec(
    n_latent=n_latent,
    n_manifest=n_manifest,
    lambda_mat=jnp.eye(n_manifest, n_latent),
    manifest_means=jnp.zeros(n_manifest),
    diffusion="diag",
    t0_means=jnp.zeros(n_latent),
    t0_var=jnp.eye(n_latent),
)

print("True drift=-0.3, diff=0.3, obs_sd=0.5")
print(f"T={T}, dt={dt}")
print()

for proposal in ["rw", "mala", "hessian"]:
    for step_size in [0.01, 0.1, 0.5, 1.0]:
        model = SSMModel(spec, n_particles=200, pf_seed=42)
        result = fit(
            model,
            observations=observations,
            times=times,
            method="hessmc2",
            n_smc_particles=32,
            n_iterations=15,
            proposal=proposal,
            step_size=step_size,
            adapt_step_size=False,
            seed=0,
        )
        samples = result.get_samples()
        drift = -jnp.abs(samples["drift_diag_pop"][:, 0])
        diff = samples["diffusion_diag_pop"][:, 0]
        obs = samples["manifest_var_diag"][:, 0]
        ess = result.diagnostics["ess_history"]
        avg_ess = sum(ess) / len(ess)

        print(
            f"{proposal:>7s} eps={step_size:.2f}  "
            f"drift={float(jnp.mean(drift)):+.3f}+-{float(jnp.std(drift)):.3f}  "
            f"diff={float(jnp.mean(diff)):.3f}+-{float(jnp.std(diff)):.3f}  "
            f"obs={float(jnp.mean(obs)):.3f}+-{float(jnp.std(obs)):.3f}  "
            f"avgESS={avg_ess:.1f}/{32}"
        )
    print()
