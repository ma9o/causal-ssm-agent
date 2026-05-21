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
| `pit_particle_mgrad` | Blocked complete-data MCMC | PIT dSMC Particle-mGRAD latent trajectory proposal | MALA parameter kernel | Particle latent trajectory updates |

## Structural Routing

The default routing resolves to `aux_kalman_mcmc`. Users can override the method with `pit_particle_mgrad` when they need particle latent trajectory updates instead of the auxiliary Kalman proposal.

Routing still records the structural backend for runtime/frontend diagnostics:

- **`laplace`**: non-Gaussian observations or support-aware summaries use the IEKS/Laplace approximate marginal likelihood.

## User Overrides

| Need | Override to | Why |
|---|---|---|
| Blocked MCMC with Gaussian latent diffusion and Kalman-style auxiliary proposals | `aux_kalman_mcmc` | Alternates latent trajectory and parameter updates without relying on a marginal likelihood sampler. |
| Blocked MCMC with particle latent proposals | `pit_particle_mgrad` | Uses the retained PIT Particle-mGRAD latent kernel with divide-and-conquer conditional particle smoothing. |

## Method Reference

### Auxiliary Kalman MCMC

`aux_kalman_mcmc` alternates an auxiliary Kalman latent trajectory update with a MALA parameter update.

**When to use:** Default complete-data MCMC when the latent diffusion path is Gaussian and the auxiliary Kalman proposal is appropriate.

**Limitations:** Requires Gaussian latent diffusion. Mixing depends on the latent step scale, parameter step size, and posterior coupling between states and parameters.

### PIT Particle-mGRAD

`pit_particle_mgrad` alternates a PIT dSMC Particle-mGRAD latent trajectory kernel with a MALA parameter update. The latent block draws PIT Particle-mGRAD auxiliary pseudo-observations, proposes independent particles at each time point, and stitches partial trajectories with a divide-and-conquer conditional particle smoother instead of the auxiliary Kalman latent kernel.

**When to use:** Complete-data MCMC when the retained particle latent trajectory kernel is needed.

**Limitations:** Requires tuning the latent step scale and particle count. The dSMC tree has logarithmic parallel depth in the number of time points, but each stitch performs particle-pair weighting and the wall-clock gain depends on JAX compilation and available parallel hardware.
