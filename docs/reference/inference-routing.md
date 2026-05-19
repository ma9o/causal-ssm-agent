# Inference Routing for State-Space Models

The implemented inference surface is deliberately small: `map`, `svi`, `aux_gibbs`, and `particle_mgrad`. For the CT-SDE formulation and likelihood backends, see [estimation.md](estimation.md).

## The Marginalization Challenge

Given a state-space model with latent states **x**_1:T and observations **y**_1:T, parameter inference requires the marginal likelihood:

```text
p(y_1:T | theta) = integral p(y_1:T, x_1:T | theta) dx_1:T
```

For SSMs with T timesteps and n latent dimensions, this integral is over an `(n x T)`-dimensional space. The implemented methods either approximate this marginal likelihood for parameter-only inference or use blocked complete-data MCMC updates over parameters and latent trajectories.

## Method Taxonomy

| Method | Coupling | State-side objective | Parameter update | Primary use |
|---|---|---|---|---|
| `map` | Marginalized | Kalman or IEKS/Laplace approximate marginal likelihood | L-BFGS-B mode plus local Gaussian posterior | Deterministic local fit |
| `svi` | Marginalized | NumPyro ELBO over the model target | Auto-guide optimization | Fast approximate posterior exploration |
| `aux_gibbs` | Complete-data Gibbs | Auxiliary Kalman latent trajectory proposal | MALA parameter kernel | Default blocked MCMC fit |
| `particle_mgrad` | Complete-data Gibbs | PIT dSMC Particle-mGRAD latent trajectory proposal | MALA parameter kernel | Particle latent updates for non-Gaussian/support-aware likelihood paths |

## Structural Routing

The default routing now resolves to `aux_gibbs`. Users can override the method with `map`, `svi`, or `particle_mgrad` when they need a different approximation or MCMC behavior.

Routing still computes the likelihood path because the runtime and frontend need to know how latent and observed variables are evaluated:

- **`kalman`**: all variables are Kalman-eligible, so the entire model uses the closed-form Kalman filter.
- **`composed`**: a Kalman sub-block is evaluated first, then a particle block handles the remainder.
- **`particle`**: no executable first-pass Kalman split exists, or interval-summary support forces the support-aware particle path.

## User Overrides

| Need | Override to | Why |
|---|---|---|
| Fast approximate posterior while iterating on a model | `svi` | ELBO optimization is usually cheaper than MCMC. |
| Blocked MCMC with Gaussian latent diffusion and Kalman-style auxiliary proposals | `aux_gibbs` | Alternates latent trajectory and parameter updates without relying on a marginal likelihood sampler. |
| Blocked MCMC with particle latent proposals | `particle_mgrad` | Uses the retained Particle-mGRAD latent kernel with divide-and-conquer conditional particle smoothing. |

## First-Pass Rao-Blackwellization

Before fitting, graph analysis partitions the model's latent and observed variables into a Kalman sub-block and a particle sub-block. This first pass operates on the fixed `SSMSpec`, not on per-iteration parameter values.

The analysis (`graph_analysis.analyze_first_pass_rb`) examines drift sparsity, observation dependencies, and noise families to identify decoupled linear-Gaussian sub-blocks that can be marginalized exactly via the Kalman filter before the particle backend runs.

The resulting `RBPartition` assigns each latent variable and each observation channel to either `kalman` or `particle`. This determines the `likelihood_path` re-derived by [Stage 5b](../pipeline/05b-inference-diagnostics.md) at fit time.

First-pass RB is disabled when:

- The spec opts out with `first_pass_rb = False`.
- Interval-summary observations are present.
- No executable partition exists because all variables couple to non-Gaussian components.

## Method Reference

### MAP

`map` optimizes the approximate marginal posterior over parameters, then samples a local Gaussian approximation in unconstrained parameter space. The likelihood side uses the Kalman backend for linear-Gaussian paths and the IEKS/Laplace backend otherwise.

**When to use:** Deterministic geometry diagnostics and local posterior approximations.

**Limitations:** Local and unimodal by construction. Posterior skewness and separated modes are not represented.

### SVI

`svi` uses NumPyro stochastic variational inference with an auto-guide. It returns approximate posterior samples from the learned guide.

**When to use:** Fast model iteration or preflight posterior summaries.

**Limitations:** The guide family can underestimate posterior variance and does not represent multimodality well.

### Auxiliary Gibbs

`aux_gibbs` alternates an auxiliary Kalman latent trajectory update with a MALA parameter update.

**When to use:** Default complete-data MCMC when the latent diffusion path is Gaussian and the auxiliary Kalman proposal is appropriate.

**Limitations:** Requires Gaussian latent diffusion. Mixing depends on the latent step scale, parameter step size, and posterior coupling between states and parameters.

### Particle-mGRAD

`particle_mgrad` alternates a PIT dSMC Particle-mGRAD latent trajectory kernel with a MALA parameter update. The latent block draws Particle-mGRAD auxiliary pseudo-observations, proposes independent particles at each time point, and stitches partial trajectories with a divide-and-conquer conditional particle smoother instead of the auxiliary Kalman latent kernel.

**When to use:** Complete-data MCMC when the retained particle latent kernel is needed, especially for particle/support-aware likelihood paths.

**Limitations:** Requires tuning the latent step scale and particle count. The dSMC tree has logarithmic parallel depth in the number of time points, but each stitch performs particle-pair weighting and the wall-clock gain depends on JAX compilation and available parallel hardware.
