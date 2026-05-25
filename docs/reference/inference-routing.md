# Inference Routing for State-Space Models

The implemented inference surface is deliberately small: `aux_kalman_mcmc` and `pit_particle_mgrad`. For the CT-SDE formulation and likelihood backends, see [estimation.md](estimation.md).

## The Marginalization Challenge

Given a state-space model with latent states **x**_1:T and observations **y**_1:T, parameter inference requires the marginal likelihood:

```text
p(y_1:T | theta) = integral p(y_1:T, x_1:T | theta) dx_1:T
```

For SSMs with T timesteps and n latent dimensions, this integral is over an `(n x T)`-dimensional space. The public methods use blocked complete-data MCMC updates over parameters and latent trajectories.

## Method Taxonomy

| Method | Coupling | State-side objective | Parameter update | Primary use |
|---|---|---|---|---|
| `aux_kalman_mcmc` | Blocked complete-data MCMC | Auxiliary Kalman latent trajectory proposal | MALA parameter kernel | Default blocked MCMC fit |
| `pit_particle_mgrad` | Blocked complete-data MCMC | Sequential Particle-mGRAD latent trajectory proposal | Hybrid Gibbs/NUTS parameter kernel | Prior-informed particle latent trajectory updates |

## Structural Routing

The default routing resolves to `aux_kalman_mcmc`. Users can override the method with `pit_particle_mgrad` when they need particle latent trajectory updates instead of the auxiliary Kalman proposal.

Routing still records the structural backend for runtime/frontend diagnostics:

- **`laplace`**: non-Gaussian observations or support-aware summaries use the IEKS/Laplace approximate marginal likelihood.

## User Overrides

| Need | Override to | Why |
|---|---|---|
| Blocked MCMC with Gaussian latent diffusion and Kalman-style auxiliary proposals | `aux_kalman_mcmc` | Alternates latent trajectory and parameter updates without relying on a marginal likelihood sampler. |
| Blocked MCMC with prior-informed particle latent proposals | `pit_particle_mgrad` | Uses the retained Particle-mGRAD latent kernel with sequential conditional particle smoothing. |

## Runtime Conditioning

Runtime conditioning transforms the compiled state-space model before a latent kernel consumes it. Polya-Gamma augmentation is applied first for PG-able non-Gaussian observation rows, which can make those rows Gaussian for the downstream latent update. Rao-Blackwellized particle filtering (RBPF) then partitions the latent state into carried dimensions sampled by the latent kernel and marginalized dimensions integrated by Kalman updates.

RBPF has two explicit modes because the computational contract is different:

| Mode | Exactness target | Particle-time structure | Use when |
|---|---|---|---|
| `independent` | Exact for marginalized linear-Gaussian blocks that do not depend on the carried particle history. | Preserves the PIT dSMC tree and logarithmic parallel depth in `T`. | The collapsed block is independent enough that each time slice can be scored without carrying a path-dependent Kalman filter state. |
| `conditional` | Exact for conditionally linear-Gaussian marginalized blocks whose dynamics or observations depend on the carried state. | Uses a sequential conditional RBPF particle kernel; the Kalman filter state is part of each particle prefix, so the old PIT increment interface is not valid. | The variance reduction from true conditional Rao-Blackwellization is worth giving up the `O(log T)` PIT structure. |

The separation is intentional. Treating conditional RBPF as if it were independent would silently score the wrong target because the marginalized filtering distribution changes with the carried trajectory history. Conditional mode can still produce an independent partition when that is the maximal legal exact structure; the actual `rbpf_structure` diagnostic determines whether the PIT tree is preserved.

When `rbpf_mode` is not `none`, the runtime derives the exact legal partition after Polya-Gamma planning. If `rbpf_marginalized_latent_indices` is omitted, every latent dimension is considered a candidate for marginalization. Residual non-Gaussian/non-PG observations force their loaded latents to remain carried; the carried set is then closed over dynamics dependencies, nonlinear marginalized dynamics, and process/initial covariance blocks. The resulting diagnostics record which latents were forced carried and why. If no nontrivial exact RBPF block remains, the final partition is the full carried path with `rbpf_requested=true` and `rbpf_enabled=false`.

## Method Reference

### Auxiliary Kalman MCMC

`aux_kalman_mcmc` alternates an auxiliary Kalman latent trajectory update with a MALA parameter update.

**When to use:** Default complete-data MCMC when the latent diffusion path is Gaussian and the auxiliary Kalman proposal is appropriate.

**Limitations:** Requires Gaussian latent diffusion. Mixing depends on the latent step scale, parameter step size, and posterior coupling between states and parameters.

### Particle-mGRAD

`pit_particle_mgrad` alternates a sequential Particle-mGRAD latent trajectory kernel with a hybrid Gibbs/NUTS parameter update. The latent block draws Gaussian auxiliary pseudo-observations from the current trajectory, proposes each non-reference particle from the Gaussian prior dynamics conditioned on its selected ancestor and pseudo-observation, and uses marginal Particle-mGRAD weights that integrate out the pseudo-observation.

The default `latent_kernel_algorithm="particle_mgrad"` is not parallel in time because the prior-informed proposal depends on the selected ancestor. The alternate `latent_kernel_algorithm="pit_aux_csmc"` keeps the separable PIT auxiliary cSMC construction, but that is a different kernel family.

**When to use:** Complete-data MCMC when the retained particle latent trajectory kernel is needed.

**Limitations:** Requires tuning the latent step scale and particle count. The default Particle-mGRAD kernel is sequential in the number of time points. `latent_kernel_algorithm="particle_mgrad"` rejects `rbpf_mode="conditional"` because each particle prefix owns a different marginalized Kalman filter state. Use `latent_kernel_algorithm="pit_aux_csmc"` when conditional RBPF is required.
